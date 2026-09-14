/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#pragma once

#include <sycl/nd_item.hpp>

#include "ggml-sycl/dpct/helper.hpp"

template <int ElementsPerWI>
__dpct_inline__ static void quantize_q8_1_impl(const float * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_values, float & d,
                                               float & sum, const sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<float, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);

    float amax = 0.0f;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_values[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <int ElementsPerWI> struct no_quantize_q8_1 {
    void operator()(const float *, void *, int, int, const sycl::nd_item<1> &) const {}
};

template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// Same SoA layout as quantize_and_reorder_q8_1_soa, but inside every 32-block the
// values are stored in the IQ4_XS "pair" order consumed by the ESIMD IQ4_XS decode
// kernel (mul_mat_vec_iq4_xs_q8_1_esimd): position 2j holds element j and 2j+1
// holds element 16+j, matching the (low nibble, high nibble) pair of qs byte j.
// Only selected together with that kernel (ggml_backend_sycl_context::iq4_y_pairs).
template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa_iq4pair {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);

        int8_t * quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + QK8_1 * col);
#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            const int e = wi_id * ElementsPerWI + i;
            const int p = e < 16 ? 2 * e : 2 * (e - 16) + 1;
            quant_ptr[p] = quantized_values[i];
        }

        auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// Same SoA layout as quantize_and_reorder_q8_1_soa, but the 8 sub-blocks (32
// values + {d,sum} each) of every 256-block are stored in slot order
// (0,2,1,3,4,6,5,7), i.e. sub-block s goes to slot s with its two low bits
// swapped. Consumed by the ESIMD Q4_K / Q5_K dp4a kernels
// (mul_mat_vec_q4_K_q8_1_esimd / mul_mat_vec_q5_K_q8_1_esimd), whose 16-dword nibble groups cover weights
// {128j..+31, 128j+64..+95} and {128j+32..+63, 128j+96..+127}. kx % 256 == 0.
// The second half of each {d, sum} holds the int16 sum of the 32 quantized
// values (not the float sum of x): the kernel's min term is dmin*m*d*sum(q8),
// the same estimator as vec_dot_q5_K_q8_1_impl_vmmq, so the activation
// quantization error meets the centered weight (sum(x) instead cost +0.3% PPL).
template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa_q5k_slots {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        int qsum = 0;
#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            qsum += quantized_values[i];
        }
        qsum = sycl::reduce_over_group(it.get_sub_group(), qsum, sycl::plus<int>());

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      slot               = (col & ~3) | ((col & 1) << 1) | ((col >> 1) & 1);
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * slot + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (uint16_t *) ((char *) reordered_q8_tensor + row_offset + kx + slot * sizeof(sycl::half2));
        if (wi_id == 0) {
            const sycl::half dh = sycl::half(d);
            ds_ptr[0] = sycl::bit_cast<uint16_t>(dh);
            ds_ptr[1] = (uint16_t) (int16_t) qsum;
        }
    }
};

// quantize_and_reorder_q8_1_soa with the int16 sum(q8) of the 32 values in the
// second half of every {d, sum} (natural value order). Consumed only by the ESIMD
// IQ4_XS dpas kernel (mul_mat_vec_iq4_xs_q8_1_dpas), whose -127 * sum(y) term
// needs the integer sum. kx % 256 == 0.
template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa_iq4nat {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        int qsum = 0;
#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            qsum += quantized_values[i];
        }
        qsum = sycl::reduce_over_group(it.get_sub_group(), qsum, sycl::plus<int>());

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (uint16_t *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            const sycl::half dh = sycl::half(d);
            ds_ptr[0] = sycl::bit_cast<uint16_t>(dh);
            ds_ptr[1] = (uint16_t) (int16_t) qsum;
        }
    }
};

template <int ElementsPerWI> struct quantize_q8_1 {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * q8_tensor, const int kx, const int kx_padded,
                                    const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        const int pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % num_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_values;
        if (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// One launch: IQ4 pair-order + Q5_K slot-order SoA. Reads stored f32 (not in-RMS emit).
inline void quantize_row_q8_1_dual_sycl(const float * x, void * pair, void * slots, const int kx, const int ky,
                                        const int kx_padded, dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             constexpr int ElementsPerWI = QK8_1 / WARP_SIZE;
                             auto          subgroup_id   = it.get_group(0);
                             auto          wi_id         = it.get_local_id(0);

                             sycl::vec<int8_t, ElementsPerWI> quantized_values;
                             float                            d   = 0.0f;
                             float                            sum = 0.0f;
                             quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

                             const int num_blocks_per_row = kx / QK8_1;
                             auto      row                = subgroup_id / num_blocks_per_row;
                             auto      col                = subgroup_id % num_blocks_per_row;
                             auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);

                             int8_t * pair_ptr = (int8_t *) ((char *) pair + row_offset + QK8_1 * col);
#pragma unroll(ElementsPerWI)
                             for (int i = 0; i < ElementsPerWI; i++) {
                                 const int e     = wi_id * ElementsPerWI + i;
                                 const int p     = e < 16 ? 2 * e : 2 * (e - 16) + 1;
                                 pair_ptr[p] = quantized_values[i];
                             }
                             auto pair_ds = (sycl::half2 *) ((char *) pair + row_offset + kx + col * sizeof(sycl::half2));
                             if (wi_id == 0) {
                                 *pair_ds = sycl::half2(sycl::half(d), sycl::half(sum));
                             }

                             int qsum = 0;
#pragma unroll(ElementsPerWI)
                             for (int i = 0; i < ElementsPerWI; i++) {
                                 qsum += quantized_values[i];
                             }
                             qsum = sycl::reduce_over_group(it.get_sub_group(), qsum, sycl::plus<int>());

                             auto slot       = (col & ~3) | ((col & 1) << 1) | ((col >> 1) & 1);
                             auto col_offset = QK8_1 * slot + wi_id * ElementsPerWI;
                             auto slot_ptr =
                                 (int8_t *) ((char *) slots + row_offset + col_offset);
                             *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(slot_ptr) = quantized_values;
                             auto slot_ds = (uint16_t *) ((char *) slots + row_offset + kx + slot * sizeof(sycl::half2));
                             if (wi_id == 0) {
                                 const sycl::half dh = sycl::half(d);
                                 slot_ds[0]          = sycl::bit_cast<uint16_t>(dh);
                                 slot_ds[1]          = (uint16_t) (int16_t) qsum;
                             }
                         });
}

template <template <int> typename quantize_f>
void quantize_row_q8_1_sycl(const float * x, void * vy, const int kx, const int ky, const int kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
