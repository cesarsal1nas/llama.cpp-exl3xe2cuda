#include "fusion.hpp"
#include "binbcast.hpp"
#include "dmmv.hpp"

#include <algorithm>

// Xe2 GDN_GATE / ADD_RMS fuses slow EXL3 MTP verify (~25 vs ~30). IQ4 graphs have no EXL3 types.
static bool ggml_sycl_cgraph_has_exl3(const ggml_cgraph * cgraph) {
    static thread_local const ggml_cgraph * last = nullptr;
    static thread_local bool                last_has = false;
    if (cgraph == last) {
        return last_has;
    }
    last     = cgraph;
    last_has = false;
    if (!cgraph) {
        return false;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (!t) {
            continue;
        }
        if (ggml_is_exl3(t->type)) {
            last_has = true;
            return true;
        }
        if (t->src[0] && ggml_is_exl3(t->src[0]->type)) {
            last_has = true;
            return true;
        }
        if (t->src[1] && ggml_is_exl3(t->src[1]->type)) {
            last_has = true;
            return true;
        }
    }
    return false;
}

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_mul_mat_glu_mmvq_fused().
static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    if (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        return false;
    }

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * act = up->src[1];

    // one set of block offsets and one quantized activation must serve both weights
    if (wu->type != wg->type || !ggml_are_same_shape(wu, wg) || !ggml_are_same_stride(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // Q4_K fused reorder GEMV. IQ4_XS decode / MTP (ne11 <= 4) uses ESIMD up+gate+SWIGLU.
    const bool iq4_decode = wu->type == GGML_TYPE_IQ4_XS && act->ne[1] <= GGML_SYCL_IQ4_ESIMD_MAX_NCOLS &&
                            ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU;
    if ((wu->type != GGML_TYPE_Q4_K && !iq4_decode) || wu->ne[0] % QK_K != 0) {
        return false;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    if (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        return false;
    }
    if (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        return false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    if (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        return false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    if (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }

    return true;
}

static bool ggml_sycl_keep_xe2_dgpu() {
    const int dev = ggml_sycl_get_device();
    return ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[dev].hw_info.arch);
}

static bool ggml_sycl_keep_env_on(const char * name) {
    const int env = ggml_sycl_get_env(name, -1);
    if (env == 0) {
        return false;
    }
    if (env > 0) {
        return true;
    }
    return ggml_sycl_keep_xe2_dgpu();
}

bool ggml_sycl_is_gdn_conv_concat(const ggml_tensor * concat) {
    static const int fuse = ggml_sycl_get_env("GGML_SYCL_FUSE_CONV_CONCAT", 1);
    if (!fuse || !concat || concat->op != GGML_OP_CONCAT || concat->type != GGML_TYPE_F32) {
        return false;
    }
    const int32_t dim = ((const int32_t *) concat->op_params)[0];
    if (dim != 0) {
        return false;
    }
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * x  = concat->src[1];
    if (!st || !x || st->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) {
        return false;
    }
    if (st->ne[0] != 3 || x->ne[0] < 1 || !ggml_is_contiguous(st) || !ggml_is_transposed(x)) {
        return false;
    }
    if (x->ne[1] != st->ne[1] || x->ne[2] != st->ne[2] || x->ne[3] != st->ne[3]) {
        return false;
    }
    if (concat->ne[0] != st->ne[0] + x->ne[0] || concat->ne[1] != st->ne[1] ||
        concat->ne[2] != st->ne[2]) {
        return false;
    }
    if (st->ne[1] <= 0 || (st->ne[1] % 4) != 0) {
        return false;
    }
    if (x->nb[1] != sizeof(float) || x->nb[0] != (size_t) x->ne[1] * sizeof(float)) {
        return false;
    }
    if (st->nb[0] != sizeof(float) || concat->nb[0] != sizeof(float)) {
        return false;
    }
    if (!ggml_is_contiguous(concat)) {
        return false;
    }
    return true;
}

bool ggml_sycl_ssm_conv_can_split(const ggml_tensor * ssm) {
    if (!ssm || ssm->op != GGML_OP_SSM_CONV || !ssm->src[0] || !ssm->src[1]) {
        return false;
    }
    if (!ggml_sycl_is_gdn_conv_concat(ssm->src[0])) {
        return false;
    }
    const ggml_tensor * concat = ssm->src[0];
    const ggml_tensor * w      = ssm->src[1];
    const ggml_tensor * st     = concat->src[0];
    const ggml_tensor * x      = concat->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != st->ne[1]) {
        return false;
    }
    if (st->ne[0] + 1 != w->ne[0]) {
        return false;
    }
    if (ssm->ne[0] != st->ne[1] || ssm->ne[1] != x->ne[0] || ssm->ne[2] != st->ne[2]) {
        return false;
    }
    if (w->nb[0] != sizeof(float) || w->nb[1] != (size_t) w->ne[0] * sizeof(float)) {
        return false;
    }
    return true;
}

bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1 &&
        unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS) {
        if (!ggml_sycl_keep_env_on("GGML_SYCL_FUSE_GDN_GATE")) {
            return false;
        }
        if (cgraph->nodes[node_idx]->ne[1] <= 32 && ggml_sycl_cgraph_has_exl3(cgraph)) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }
        const ggml_tensor * add = cgraph->nodes[node_idx];
        const ggml_tensor * sp  = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul = cgraph->nodes[node_idx + 2];
        if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || sp->src[0] != add) {
            return false;
        }
        if (mul->src[0] != sp && mul->src[1] != sp) {
            return false;
        }
        if (add->type != GGML_TYPE_F32 || sp->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }
        if (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(add->src[0], add) || !ggml_are_same_shape(mul, add)) {
            return false;
        }
        if (!ggml_is_contiguous(add) || !ggml_is_contiguous_1(add->src[0]) || !ggml_is_contiguous(mul)) {
            return false;
        }
        if (add->src[0]->ne[0] != add->ne[0]) {
            return false;
        }
        auto ok_bias = [&](const ggml_tensor * t) {
            if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != add->ne[0]) {
                return false;
            }
            return ggml_are_same_shape(t, add) || ggml_nelements(t) == add->ne[0];
        };
        const ggml_tensor * w = (mul->src[0] == sp) ? mul->src[1] : mul->src[0];
        if (!ok_bias(add->src[1]) || !ok_bias(w)) {
            return false;
        }
        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM &&
        ops.begin()[2] == GGML_OP_MUL) {
        if (!ggml_sycl_keep_env_on("GGML_SYCL_FUSE_ADD_RMS")) {
            return false;
        }
        if (cgraph->nodes[node_idx]->ne[1] <= 32 && ggml_sycl_cgraph_has_exl3(cgraph)) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx, node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * add = cgraph->nodes[node_idx];
        const ggml_tensor * rms = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul = cgraph->nodes[node_idx + 2];

        if (rms->src[0] != add) {
            return false;
        }
        if (mul->src[0] != rms && mul->src[1] != rms) {
            return false;
        }
        if (add->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }
        if (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(add->src[0], add) || !ggml_are_same_shape(add->src[1], add)) {
            return false;
        }
        if (!ggml_is_contiguous(add) || !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
            !ggml_is_contiguous(rms) || !ggml_is_contiguous(mul)) {
            return false;
        }
        static const int prefill_ok = ggml_sycl_get_env("GGML_SYCL_FUSE_ADD_RMS_PREFILL", 0);
        const int64_t tlimit = (add->ne[1] > 32 || prefill_ok) ? (int64_t) 16384 : (int64_t) 8;
        if (add->ne[1] > tlimit || add->ne[2] != 1 || add->ne[3] != 1) {
            return false;
        }
        if (add->ne[0] < 1024 || (add->ne[0] % 4) != 0) {
            return false;
        }
        if (add->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms) ? mul->src[1] : mul->src[0];
        if (mul_w->type != GGML_TYPE_F32 || mul_w->ne[0] != rms->ne[0] ||
            mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            return false;
        }
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
        return true;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        const bool ok = (gate == cgraph->nodes[node_idx] && up == cgraph->nodes[node_idx + 1]) ||
                        (gate == cgraph->nodes[node_idx + 1] && up == cgraph->nodes[node_idx]);
        if (!ok) {
            return false;
        }

        return ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        if (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            return false;
        }

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[node_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        if (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            return false;
        }

        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add != nullptr) {
            if (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                return false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            const ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            if (!ggml_are_same_shape(add_w, add)) {
                return false;
            }

            if (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                return false;
            }
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        const ggml_tensor * add0 = cgraph->nodes[node_idx];
        const ggml_tensor * add1 = cgraph->nodes[node_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        if (add1->src[0] != add0) {
            return false;
        }

        const ggml_tensor * c = add1->src[1];
        if (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        // the ops ggml_sycl_op_unary_mul_fused() has a kernel for
        if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
            unary_op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        if (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        if (!ggml_sycl_keep_env_on("GGML_SYCL_FUSE_SSM_SILU")) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 1 })) {
            return false;
        }
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 1];
        if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != ssm_conv) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        return true;
    }

    return false;
}
