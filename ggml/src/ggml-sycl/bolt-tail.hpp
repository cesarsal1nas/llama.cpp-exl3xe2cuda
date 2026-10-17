#pragma once

static int ggml_sycl_try_bolt_tail(ggml_backend_sycl_context & ctx, ggml_cgraph * g, int i) {
    if (!ggml_sycl_get_env("GGML_SYCL_BOLT_TAIL", 0) || i + 7 >= g->n_nodes) return 0;
    const int idx[] = {i, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7};
    const ggml_op ops[] = {GGML_OP_CONT, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_GLU,
                           GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL};
    const int outputs[] = {i + 7};
    if (g->nodes[i + 1]->op != GGML_OP_VIEW ||
        !ggml_can_fuse_subgraph_ext(g, idx, 7, ops, outputs, 1)) return 0;
    if (!ggml_check_edges(g, i, {{3, 0, 1}, {3, 1, 2}, {4, 0, 0}, {4, 1, 3},
                               {5, 0, 4}, {6, 0, 5}, {7, 0, 6}})) return 0;
    const auto * z = g->nodes[i]->src[0];
    const auto * s = g->nodes[i + 1];
    const auto * x = g->nodes[i + 2]->src[0];
    const auto * d = g->nodes[i + 2]->src[1];
    const auto * glu = g->nodes[i + 4];
    const auto * rms = g->nodes[i + 6];
    const auto * w = g->nodes[i + 7]->src[1];
    auto * out = g->nodes[i + 7];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->op_params[1] != 0) return 0;
    for (auto * t : {z, s, x, d, w}) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) return 0;
    }
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
        out->ne[0] != 1024 || out->ne[1] != 8 || out->ne[3] != 1 ||
        w->ne[0] != 1024 || w->ne[1] != 8 || w->ne[2] != 1 || w->ne[3] != 1 ||
        !ggml_is_contiguous(w) || d->ne[0] != 1 || d->ne[1] != 128 ||
        d->ne[2] != 1 || d->ne[3] != 1) return 0;
    for (auto * t : {z, s, x}) {
        if (t->ne[0] != 64 || t->ne[1] != 128 || t->ne[2] != out->ne[2] ||
            t->ne[3] != 1 || t->nb[1] != 64 * sizeof(float)) return 0;
    }
    const float * zp = (const float *)z->data;
    const float * sp = (const float *)s->data;
    const float * xp = (const float *)x->data;
    const float * dp = (const float *)d->data;
    const float * wp = (const float *)w->data;
    float * yp = (float *)out->data;
    const size_t zs = z->nb[2] / 4, ss = s->nb[2] / 4, xs = x->nb[2] / 4;
    const size_t ds = d->nb[1] / 4;
    float eps;
    memcpy(&eps, rms->op_params, sizeof(eps));
    const size_t groups = size_t(out->ne[1]) * out->ne[2];
    ctx.stream()->parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t group = it.get_group_linear_id();
        const size_t token = group / 8, gidx = group % 8;
        const size_t lane = it.get_local_linear_id();
        float vals[4];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t col = gidx * 1024 + lane + k * 256;
            const float zval = zp[token * zs + col];
            const float scaled = xp[token * xs + col] * dp[(col / 64) * ds];
            const float sval = sp[token * ss + col] + scaled;
            vals[k] = (zval / (1.0f + sycl::exp(-zval))) * sval;
            sum += vals[k] * vals[k];
        }
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = sycl::rsqrt(sum * (1.0f / 1024.0f) + eps);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t col = gidx * 1024 + lane + k * 256;
            yp[token * 8192 + col] = (vals[k] * inv) * wp[col];
        }
    });
    static int logged = 0;
    if (logged++ < 8) fprintf(stderr, "bolt tail fused grouped RMS, tokens=%ld\n", long(out->ne[2]));
    return 8;
}
