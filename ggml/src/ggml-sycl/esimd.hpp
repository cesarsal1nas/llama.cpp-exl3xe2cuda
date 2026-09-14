#ifndef GGML_SYCL_ESIMD_HPP
#define GGML_SYCL_ESIMD_HPP

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

#include "common.hpp"

namespace ggml_sycl_esimd {

constexpr int GGML_SYCL_DMMV_ESIMD_WG_SIZE = 4;
// output rows per work-group (even); see dequantize_mul_mat_vec_reorder_esimd
constexpr int GGML_SYCL_DMMV_ESIMD_ROWS_PER_WG = 2;

//
// Shared ESIMD building blocks for the reordered K-quant dequantize-matvec
// kernels.
//
// The reordered K-quant ESIMD matvec kernels share one skeleton: per super-block,
// load a 256-float activation slice, load one weight block, dequantize it into 8
// chunks of 32 and MAC each chunk against the matching activation slice, then
// reduce and run a lane-0 epilogue.
//
// Each K-quant kernel emits exactly 8 chunks of 32 mapping to activation slices
// 0..7, so the per-block work is captured by esimd_reorder_q_traits<T>::mac_pair,
// which dequantizes two weight blocks and MACs both against a shared activation
// vector with the two FMA chains interleaved (co-scheduled to hide FMA latency).
// The "pair" is the (row0,row1) row pair owned by one work-group, so the
// layout+dequant is written once per quant type here.
//

template <ggml_type T> struct esimd_reorder_q_traits;

// build a 32-lane vector whose low 16 lanes are `lo` and high 16 are `hi`
// (a super-chunk splits into two 16-wide halves with distinct scale/min codes).
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> splat_lo_hi(float lo, float hi) {
    using namespace sycl::ext::intel::esimd;
    simd<float, 32> v;
    v.select<16, 1>(0)  = lo;
    v.select<16, 1>(16) = hi;
    return v;
}

// unpack one block of Q4_K/Q5_K scale/min codes (get_scale_min_k4 layout) into 8
// float scales (dall * sc) and 8 float mins (-dmin * m); the min carries the
// negation so the dequant epilogue adds.
static ESIMD_INLINE void unpack_scale_min_k4(
        sycl::ext::intel::esimd::simd<uint8_t, 12> scales, float dall, float dmin,
        sycl::ext::intel::esimd::simd<float, 8> & scale_f,
        sycl::ext::intel::esimd::simd<float, 8> & min_f) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, 8> sc = 0;
    simd<uint8_t, 8> m  = 0;
    simd<uint8_t, 4> scale_lo = scales.select<4, 1>(0);
    simd<uint8_t, 4> min_lo   = scales.select<4, 1>(4);
    simd<uint8_t, 4> hi_bits  = scales.select<4, 1>(8);
    sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                         ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                         ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<float>(sc) * dall;
    min_f   = convert<float>(m) * (-dmin);
}

// ---------------------------------------------------------------------------
// Q2_K, SOA reorder layout produced by reorder_qw_q2_k:
//   [qs: nb*(QK_K/4)] [scales: nb*(QK_K/16)] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// 2 bits per weight. The 8 output chunks of 32 (matching dequantize_row_q2_K)
// map to super-chunk s (0..7): byte base 32*(s/4) into the 64-byte qs array,
// bit shift 2*(s%4); the low 16 lanes use scales[2s], the high 16 use
// scales[2s+1], with dl = d*(sc & 0xF), ml = dmin*(sc >> 4), deq = dl*q - ml.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q2_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 4);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { qs, scales, dm };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 16> scales_a = block_load<uint8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<uint8_t, 16> scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            scales_b = block_load<uint8_t, 16>(pb.scales + bib * (QK_K / 16));
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        // per-chunk scale (d * (sc & 0xF)) and min (-dmin * (sc >> 4)), all 16 codes;
        // min carries the negation so the dequant epilogue adds (matches Q4_K/Q5_K)
        simd<float, 16> scale_f_a = convert<float>(scales_a & simd<uint8_t, 16>(0x0F)) * dall_a;
        simd<float, 16> min_f_a   = convert<float>(scales_a >> simd<uint8_t, 16>(4))  * (-dmin_a);
        simd<float, 16> scale_f_b = convert<float>(scales_b & simd<uint8_t, 16>(0x0F)) * dall_b;
        simd<float, 16> min_f_b   = convert<float>(scales_b >> simd<uint8_t, 16>(4))  * (-dmin_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<float, 32> y_s = y_vec.select<32, 1>(s * 32);

            simd<uint8_t, 32> qa = (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);
            simd<uint8_t, 32> qb = (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float min_a_lo   = min_f_a[2 * s + 0];
            const float min_a_hi   = min_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];
            const float min_b_lo   = min_f_b[2 * s + 0];
            const float min_b_hi   = min_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> min_vec_a   = splat_lo_hi(min_a_lo, min_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);
            simd<float, 32> min_vec_b   = splat_lo_hi(min_b_lo, min_b_hi);

            simd<float, 32> deq_a = convert<float>(qa) * scale_vec_a + min_vec_a;
            simd<float, 32> deq_b = convert<float>(qb) * scale_vec_b + min_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q3_K, SOA reorder layout produced by reorder_qw_q3_k:
//   [qs: nb*(QK_K/4)] [hmask: nb*(QK_K/8)] [scales: nb*12] [d: nb*sizeof(half)]
// with nb = nrows*num_blocks_per_row. Single super-block scale d, no dmin.
//
// 3 bits per weight: 2 low bits in qs, 1 high bit in hmask. The 8 output chunks
// of 32 (matching dequantize_row_q3_K) map to super-chunk s (0..7): byte base
// 32*(s/4) into the 64-byte qs array, bit shift 2*(s%4); the low 16 lanes use
// scale code 2s, the high 16 use 2s+1. hmask is a 32-byte array (like Q5_K's
// qh) where chunk s uses bit s of the same 32 bytes, but INVERTED: the value is
// (q & 3) - (hmask_bit_set ? 0 : 4), i.e. (q & 3) + 4*bit - 4.
//
// The 16 6-bit scale codes are packed into 12 bytes (get_scale_min layout for
// Q3_K): low nibbles from bytes 0..7, high 2 bits from bytes 8..11 shifted by
// 0/2/4/6; the dequant scale is d * (code - 32).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q3_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    hmask;
        const uint8_t *    scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * hmask  = qs + nb * (QK_K / 4);
        const uint8_t * scales = hmask + nb * (QK_K / 8);
        const sycl::half * d   = (const sycl::half *) (scales + nb * 12);
        return { qs, hmask, scales, d };
    }

    // unpack the 12 packed bytes into 16 6-bit scale codes (dequantize_row_q3_K
    // aux layout), returned as float scale = d * (code - 32).
    // done with wide (8/16-lane) ops rather than four 4-lane groups.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 16> unpack_scales(
            sycl::ext::intel::esimd::simd<uint8_t, 12> in, float d) {
        using namespace sycl::ext::intel::esimd;

        // low 6-bit part: codes 0..7 = low nibble of bytes 0..7,
        //                 codes 8..15 = high nibble of bytes 0..7
        simd<uint8_t, 8>  lo8 = in.select<8, 1>(0);
        simd<uint8_t, 16> code;
        code.select<8, 1>(0) = lo8 & simd<uint8_t, 8>(0x0F);
        code.select<8, 1>(8) = lo8 >> simd<uint8_t, 8>(4);

        // high 2-bit part: bytes 8..11 replicated 4x, group g (0..3) shifted 2*g
        simd<uint8_t, 16> hib;
        hib.select<4, 1>(0)  = in.select<4, 1>(8);
        hib.select<4, 1>(4)  = in.select<4, 1>(8);
        hib.select<4, 1>(8)  = in.select<4, 1>(8);
        hib.select<4, 1>(12) = in.select<4, 1>(8);
        simd<uint8_t, 16> hshift;
        hshift.select<4, 1>(0)  = 0;
        hshift.select<4, 1>(4)  = 2;
        hshift.select<4, 1>(8)  = 4;
        hshift.select<4, 1>(12) = 6;
        hib = (hib >> hshift) & simd<uint8_t, 16>(0x03);

        code = code | (hib << simd<uint8_t, 16>(4));
        return (convert<float>(code) - 32.0f) * d;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 32> hmask_a  = block_load<uint8_t, 32>(pa.hmask + bia * (QK_K / 8));
        simd<uint8_t, 32> hmask_b  = 0;
        simd<uint8_t, 12> scales_a = block_load<uint8_t, 12>(pa.scales + bia * 12);
        simd<uint8_t, 12> scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            hmask_b  = block_load<uint8_t, 32>(pb.hmask + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * 12);
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> scale_f_a = unpack_scales(scales_a, d_a);
        simd<float, 16> scale_f_b = unpack_scales(scales_b, d_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<float, 32> y_s = y_vec.select<32, 1>(s * 32);

            // 2 low bits from qs, high bit from hmask (bit s of the same 32 bytes);
            // value = (q & 3) + 4*bit - 4  (inverted hmask: subtract 4 when bit clear).
            // merge in the integer domain: q3 = (q & 3) | (bit << 2) in {0..7},
            // then a single convert + subtract yields q3 - 4 (one convert, not two)
            simd<uint16_t, 32> q3_a = convert<uint16_t>(
                    (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_a |= convert<uint16_t>(
                    ((hmask_a >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));
            simd<uint16_t, 32> q3_b = convert<uint16_t>(
                    (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_b |= convert<uint16_t>(
                    ((hmask_b >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));

            simd<float, 32> qf_a = convert<float>(q3_a) - 4.0f;
            simd<float, 32> qf_b = convert<float>(q3_b) - 4.0f;

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

            simd<float, 32> deq_a = qf_a * scale_vec_a;
            simd<float, 32> deq_b = qf_b * scale_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q4_K, SOA reorder layout produced by reorder_qw_q4_k:
//   [qs: nb*(QK_K/2)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q4_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 2);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, scales, dm };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;
            simd<float, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<float, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi = qs_hi_b.select<32, 1>(q_offset);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q5_K, SOA reorder layout produced by reorder_qw_q5_k:
//   [qs: nb*(QK_K/2)] [qh: nb*(QK_K/8)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// Identical to Q4_K except each 4-bit quant gains a 5th (high) bit from qh:
// output chunk c (0..7) adds 16 when bit c of qh[l] is set, where qh[l] indexes
// the same 32 bytes for every chunk (matches dequantize_row_q5_K).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q5_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * qh     = qs + nb * (QK_K / 2);
        const uint8_t * scales = qh + nb * (QK_K / 8);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, qh, scales, dm };
    }

    // extract bit `bit` (0..7) of each lane and move it to bit position 4,
    // e.g. for the 4-bit base quant's 5th (high) bit. `bit` is always a
    // compile-time-known unrolled loop constant at call sites, so this folds
    // to a single mask (bit==4), mask+left-shift (bit<4), or mask+right-shift
    // (bit>4) instead of the shift+mask+shift a naive `(qh>>bit & 1) << 4` emits.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint16_t, 32> extract_bit_to_pos4(
            sycl::ext::intel::esimd::simd<uint8_t, 32> qh, int bit) {
        using namespace sycl::ext::intel::esimd;
        simd<uint16_t, 32> masked = convert<uint16_t>(qh & simd<uint8_t, 32>((uint8_t) (1u << bit)));
        if (bit < 4) {
            return masked << simd<uint16_t, 32>((uint16_t) (4 - bit));
        } else if (bit > 4) {
            return masked >> simd<uint16_t, 32>((uint16_t) (bit - 4));
        }
        return masked;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 32>  qh_a     = block_load<uint8_t, 32>(pa.qh + bia * (QK_K / 8));
        simd<uint8_t, 32>  qh_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 32>(pb.qh + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;
            simd<float, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<float, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo_u8 = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi_u8 = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo_u8 = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi_u8 = qs_hi_b.select<32, 1>(q_offset);
            simd<uint16_t, 32> qa_lo = convert<uint16_t>(qa_lo_u8);
            simd<uint16_t, 32> qa_hi = convert<uint16_t>(qa_hi_u8);
            simd<uint16_t, 32> qb_lo = convert<uint16_t>(qb_lo_u8);
            simd<uint16_t, 32> qb_hi = convert<uint16_t>(qb_hi_u8);

            // add the 5th bit: chunk sb uses qh bit sb, chunk sb+1 uses qh bit sb+1;
            // qh always indexes the same 32 bytes regardless of chunk
            qa_lo += extract_bit_to_pos4(qh_a, sb);
            qa_hi += extract_bit_to_pos4(qh_a, sb + 1);
            qb_lo += extract_bit_to_pos4(qh_b, sb);
            qb_hi += extract_bit_to_pos4(qh_b, sb + 1);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K, SOA reorder layout:
//   [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales(int8): nb*(QK_K/16)] [d: nb*half]
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q6_K> {
    struct ptrs {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t *    ql     = (const uint8_t *) vx;
        const uint8_t *    qh     = ql + nb * (QK_K / 2);
        const int8_t *     scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d      = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { ql, qh, scales, d };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a     = block_load<uint8_t, 128>(pa.ql + bia * (QK_K / 2));
        simd<uint8_t, 128> ql_b     = 0;
        simd<uint8_t, 64>  qh_a     = block_load<uint8_t, 64>(pa.qh + bia * (QK_K / 4));
        simd<uint8_t, 64>  qh_b     = 0;
        simd<int8_t, 16>   scales_a = block_load<int8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<int8_t, 16>   scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            ql_b     = block_load<uint8_t, 128>(pb.ql + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 64>(pb.qh + bib * (QK_K / 4));
            scales_b = block_load<int8_t, 16>(pb.scales + bib * (QK_K / 16));
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> sc_a = convert<float>(scales_a);
        simd<float, 16> sc_b = convert<float>(scales_b);

#pragma unroll
        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                simd<float, 32> y_g = y_vec.select<32, 1>(32 * (4 * im + g));

                const float scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                const float scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                const float scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                const float scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
                simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

                simd<uint8_t, 32> qa;
                simd<uint8_t, 32> qb;
                switch (g) {
                    case 0:
                        qa = (ql_lo_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        qb = (ql_lo_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        break;
                    case 1:
                        qa = (ql_hi_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        qb = (ql_hi_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        break;
                    case 2:
                        qa = (ql_lo_a >> simd<uint8_t, 32>(4)) | (qh_bits_a & simd<uint8_t, 32>(0x30));
                        qb = (ql_lo_b >> simd<uint8_t, 32>(4)) | (qh_bits_b & simd<uint8_t, 32>(0x30));
                        break;
                    default:
                        qa = (ql_hi_a >> simd<uint8_t, 32>(4)) | ((qh_bits_a & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        qb = (ql_hi_b >> simd<uint8_t, 32>(4)) | ((qh_bits_b & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        break;
                }

                simd<float, 32> deq_a = (convert<float>(qa) - 32.0f) * scale_vec_a;
                simd<float, 32> deq_b = (convert<float>(qb) - 32.0f) * scale_vec_b;

                acc_a += y_g * deq_a;
                acc_b += y_g * deq_b;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// IQ4_XS, SOA reorder layout produced by reorder_qw_iq4_xs:
//   [qs: nb*(QK_K/2)] [scales_l: nb*4] [packed: nb*4 = {scales_h u16, d half}]
//
// 4 bits per weight through the non-linear kvalues_iq4nl table. Sub-block s
// (0..7) is the 16 bytes qs[16s..16s+16): low nibbles are weights 32s+0..15,
// high nibbles 32s+16..31; ls = scales_l nibble s | (scales_h 2-bit field s)<<4,
// deq = d*(ls-32)*kvalues_iq4nl[q].
//
// The table lookup is the whole problem on Xe2: in MMVQ a 16-entry byte LUT is
// 32 LSC gathers per 32 weights (~240 GB/s), ESIMD iselect compiles to
// GRF-indirect movs (night 78, 0.35x). Here kvalues_iq4nl is evaluated as a
// degree-5 minimax polynomial in the nibble: max |p(x)-k[x]| = 0.371 over
// x=0..15, so rnde(p(x)) == kvalues_iq4nl[x] exactly (verified in float32),
// i.e. 7 SIMD32 ALU ops per 32 weights, no memory, no indirect addressing.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_IQ4_XS> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales_l;
        const uint16_t *   sh;   // packed[2*ib+0] = scales_h
        const sycl::half * d;    // packed[2*ib+1] = d
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs       = (const uint8_t *) vx;
        const uint8_t * scales_l = qs + nb * (QK_K / 2);
        const uint8_t * packed   = scales_l + nb * 4;
        return { qs, scales_l, (const uint16_t *) packed, (const sycl::half *) packed };
    }

    // kvalues_iq4nl[q] for 32 nibbles, exact, ALU only (see header comment).
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> kvalues_poly(
            sycl::ext::intel::esimd::simd<uint8_t, 32> q) {
        using namespace sycl::ext::intel::esimd;
        simd<float, 32> x = convert<float>(q);
        simd<float, 32> p = x * 0.000144791394f + (-0.00434531958f);
        p = p * x + 0.118699693f;
        p = p * x + (-1.92853458f);
        p = p * x + 25.5559771f;
        p = p * x + (-127.370971f);
        return rnde<float>(p);
    }

    // 8 sub-block scales d*(ls-32): ls bits 0..3 are nibble s of the 32-bit
    // scales_l word (bit 4s), bits 4..5 are field s of scales_h (bit 2s).
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 8> unpack_scales(
            uint32_t sl, uint32_t sh, float d) {
        using namespace sycl::ext::intel::esimd;
        const simd<uint32_t, 8> idx(0, 1);
        simd<uint32_t, 8> ls = (simd<uint32_t, 8>(sl) >> (idx * 4u)) & 0xFu;
        ls |= ((simd<uint32_t, 8>(sh) >> (idx * 2u)) & 3u) << 4u;
        return (convert<float>(ls) - 32.0f) * d;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b = 0;
        const uint32_t sl_a = *(const uint32_t *) (pa.scales_l + bia * 4);
        const uint32_t sh_a = pa.sh[bia * 2 + 0];
        const float    d_a  = (float) pa.d[bia * 2 + 1];
        uint32_t sl_b = 0, sh_b = 0;
        float    d_b  = 0.0f;
        if (has_b) {
            qs_b = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            sl_b = *(const uint32_t *) (pb.scales_l + bib * 4);
            sh_b = pb.sh[bib * 2 + 0];
            d_b  = (float) pb.d[bib * 2 + 1];
        }

        const simd<float, 8> sc_a = unpack_scales(sl_a, sh_a, d_a);
        const simd<float, 8> sc_b = unpack_scales(sl_b, sh_b, d_b);

        simd<uint8_t, 128> lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            simd<float, 32> y = y_vec.select<32, 1>(s * 32);

            simd<uint8_t, 32> qa, qb;
            qa.select<16, 1>(0)  = lo_a.select<16, 1>(s * 16);
            qa.select<16, 1>(16) = hi_a.select<16, 1>(s * 16);
            qb.select<16, 1>(0)  = lo_b.select<16, 1>(s * 16);
            qb.select<16, 1>(16) = hi_b.select<16, 1>(s * 16);

            simd<float, 32> ka = kvalues_poly(qa);
            simd<float, 32> kb = kvalues_poly(qb);

            acc_a += (y * sc_a[s]) * ka;
            acc_b += (y * sc_b[s]) * kb;
        }
    }
};


// ---------------------------------------------------------------------------
// IQ4_XS x q8_1 matvec (ncols_dst == 1), ESIMD, MMVQ numerics.
//
// vx: IQ4_XS SOA reorder layout (see esimd_reorder_q_traits<GGML_TYPE_IQ4_XS>).
// vy: activation quantized by quantize_and_reorder_q8_1_soa_iq4pair for one row:
//     [int8 q[ncols]] [half2 {d,sum}[ncols/32]], where inside each 32-block
//     position 2j holds element j and 2j+1 holds element 16+j: the (lo, hi)
//     nibble pair of qs byte j, so the gathered u16 k-pairs and y line up as
//     dwords with no in-kernel permutation.
// Same math as reorder_vec_dot_q_sycl<GGML_TYPE_IQ4_XS>: per 32-weight
// sub-block, d*(ls-32) * dy * sum_i kvalues_iq4nl[q_i] * y8_i.
//
// Why ESIMD: on Xe2 both the MMVQ IQ4 kernel (~360 instructions / 512 weights)
// and the float ESIMD DMMV (~400 cycles / 512 weights) are instruction-issue
// bound at ~0.7 cycles per weight (~240 GB/s), while Q8_0 MMVQ (~0.2) reaches
// ~590 GB/s. Here the kvalues LUT is a 16-byte SLM table read with 32-lane
// byte gathers (two per 32 weight bytes), the results interleave straight
// into dword-packed (lo, hi) pairs, and one SIMD16 dp4a does 64 MACs.
// ---------------------------------------------------------------------------
// NM = 1: dst[row] = vx0[row] . y.
// NM = 2: fused FFN gate/up + SWIGLU: dst[row] = silu(vx1[row] . y) * (vx0[row] . y)
//         (vx0 = up, vx1 = gate). One work-group does RPW rows of both matrices,
//         sharing the y loads, the pair table and the launch.
// NC = number of y columns (dst columns) done per pass, 1..4: the weight loads
//      and the kvalues lookups are shared by all NC columns (MTP verification
//      runs the decode matmuls with n = 1 + drafts). Column c of y starts at
//      (char *) vy + c * stride_y_bytes, dst column c at dst + c * stride_dst.
template <int RPW, int NM, int NC = 1>
ESIMD_INLINE void mul_mat_vec_iq4_xs_q8_1_esimd(
        const void * vx0, const void * vx1, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const sycl::nd_item<1> & it, const int row0) {
    using namespace sycl::ext::intel::esimd;
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    constexpr int NR = RPW * NM;  // row accumulators per thread (x NC columns)
    static_assert(RPW <= WG, "epilogue: one thread per row");
    static_assert(NM == 1 || NM == 2, "one matrix, or up+gate");
    static_assert(NC >= 1 && NC <= 4, "1..4 y columns");
    // SLM: [0,16) kvalues_iq4nl as bytes, [64, 64 + WG*NR*NC*4) per-thread partial sums
    constexpr uint32_t SLM_SUMS = 64;
    slm_init<SLM_SUMS + WG * NR * NC * 4>();

    const int      nblocks  = ncols / QK_K;
    const size_t   nb       = (size_t) nrows * nblocks;
    const uint8_t * qs_base[NM];
    const uint32_t * sl32[NM];
    const uint32_t * pk32[NM];
#pragma unroll
    for (int m = 0; m < NM; ++m) {
        qs_base[m] = (const uint8_t *) (m == 0 ? vx0 : vx1);
        sl32[m]    = (const uint32_t *) (qs_base[m] + nb * (QK_K / 2));
        pk32[m]    = (const uint32_t *) (qs_base[m] + nb * (QK_K / 2) + nb * 4);  // {scales_h u16, d half}
    }
    const int tid  = it.get_local_id(0);

    const simd<uint32_t, 8> sidx(0, 1);

    simd<float, 16> acc[NR * NC];  // [a * NC + c]
#pragma unroll
    for (int r = 0; r < NR * NC; ++r) {
        acc[r] = 0.0f;
    }

    // Memory-level parallelism is the limiter on Xe2 (Little's law: ~1280
    // resident threads x bytes in flight per thread). All RPW row loads are
    // issued unconditionally (row index clamped, result masked at the end) and
    // the next block's loads are issued before the current block's math, so
    // ~2*NR*128 B of weights per thread are outstanding at any time.
    //
    // Thread t handles blocks t, t+WG, ...; every stream is walked with an
    // incremented pointer (the per-block 64-bit multiply/add chains were ~40
    // instructions per iteration, measurable on an issue-co-limited loop).
    simd<uint32_t, RPW> soff;  // byte offset of {scales_l | pk}[rowc * nblocks + ib]
    const uint8_t * qp[NR];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int rowc = row0 + r < nrows ? row0 + r : nrows - 1;
        soff[r] = ((uint32_t) rowc * nblocks + tid) * 4u;
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            qp[m * RPW + r] = qs_base[m] + ((size_t) rowc * nblocks + tid) * (QK_K / 2);
        }
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) c * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;  // half2 {d,sum} per 32
    }

    struct blk {
        simd<uint8_t, 128>  qs[NR];
        simd<uint32_t, RPW> sl[NM], pk[NM];
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            b.y8[c]  = block_load<int8_t, 256>(yp[c]);
            b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            b.sl[m] = gather<uint32_t, RPW>(sl32[m], soff);
            b.pk[m] = gather<uint32_t, RPW>(pk32[m], soff);
        }
        soff += (uint32_t) (WG * 4);
#pragma unroll
        for (int a = 0; a < NR; ++a) {
            b.qs[a] = block_load<uint8_t, 128>(qp[a]);
            qp[a] += WG * (QK_K / 2);
        }
    };

    blk A, B;
    // first block's loads go out before the table init so they overlap it
    if (tid < nblocks) {
        load_block(A);
    }

    // kvalues table: 16 bytes in SLM, indexed by nibble, read with two 32-lane
    // byte gathers per 32 weight bytes. Random 32-lane gathers over a 512 B
    // byte-indexed pair table cost ~4 clk each (64 banks x 4 B: two dwords per
    // bank -> conflicts); over 16 B every lane hits one of 4 dwords and the
    // pair costs ~1.8 clk. The interleave of the two byte results is the same
    // (lo, hi) pair order the y quantizer produces.
    // kvalues_iq4nl[x] = rnde(p5(x)) exactly for x in 0..15 (minimax degree-5
    // polynomial, max |p-k| = 0.371, verified in float32).
    if (tid == 0) {
        simd<uint16_t, 16> j(0, 1);
        simd<float, 16>    x = convert<float>(j);
        simd<float, 16>    p = x * 0.000144791394f + (-0.00434531958f);
        p = p * x + 0.118699693f;
        p = p * x + (-1.92853458f);
        p = p * x + 25.5559771f;
        p = p * x + (-127.370971f);
        simd<int16_t, 16> k = convert<int16_t>(rnde<float>(p));
        slm_block_store<uint8_t, 16>(0, convert<uint8_t>(convert<uint16_t>(k) & (uint16_t) 0xFF));
    }
    barrier();

    auto math = [&](blk & b) {
        simd<float, 8> dy[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            dy[c] = convert<float>(dyh);
        }

#pragma unroll
        for (int m = 0; m < NM; ++m) {
            simd<uint16_t, RPW>   dbits = convert<uint16_t>(b.pk[m] >> 16u);
            simd<sycl::half, RPW> dh    = dbits.template bit_cast_view<sycl::half>();
            simd<float, RPW>      d_m   = convert<float>(dh);
            simd<uint32_t, RPW>   sh_m  = b.pk[m] & 0xFFFFu;

#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const int      a  = m * RPW + r;
                const uint32_t sl = b.sl[m][r];
                const uint32_t sh = sh_m[r];
                const float    d  = d_m[r];

                simd<uint32_t, 8> ls = (simd<uint32_t, 8>(sl) >> (sidx * 4u)) & 0xFu;
                ls |= ((simd<uint32_t, 8>(sh) >> (sidx * 2u)) & 3u) << 4u;
                simd<float, 8> lsd = (convert<float>(ls) - 32.0f) * d;
                simd<float, 8> sc[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    sc[c] = lsd * dy[c];
                }

#pragma unroll
                for (int p = 0; p < 4; ++p) {  // sub-blocks 2p, 2p+1 = weights 64p..64p+63
                    simd<uint8_t, 32>  q   = b.qs[a].template select<32, 1>(32 * p);
                    simd<uint32_t, 32> q32 = q;
                    simd<uint32_t, 32> lo  = q32 & 15u;
                    simd<uint32_t, 32> hi  = q32 >> 4u;
                    simd<uint8_t, 64>  k8;
                    k8.template select<32, 2>(0) = slm_gather<uint8_t, 32>(lo);
                    k8.template select<32, 2>(1) = slm_gather<uint8_t, 32>(hi);
                    simd<int, 16> k32 = k8.template bit_cast_view<int>();

#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<int8_t, 64> yb  = b.y8[c].template select<64, 1>(64 * p);
                        simd<int, 16>    y32 = yb.template bit_cast_view<int>();

                        simd<int, 16>   dp  = dp4a<int, int, int, int, 16>(simd<int, 16>(0), k32, y32);
                        simd<float, 16> dpf = convert<float>(dp);

                        const float sc_lo = sc[c][2 * p];
                        const float sc_hi = sc[c][2 * p + 1];
                        acc[a * NC + c].template select<8, 1>(0) += dpf.template select<8, 1>(0) * sc_lo;
                        acc[a * NC + c].template select<8, 1>(8) += dpf.template select<8, 1>(8) * sc_hi;
                    }
                }
            }
        }
    };

    // Two-buffer ping-pong, unrolled x2: the loads for block ib+WG are issued
    // before the math for block ib, and no buffer copies (cur = nxt was ~22
    // movs per iteration).
    int ib = tid;
    for (; ib + WG < nblocks; ib += 2 * WG) {  // uniform branches
        load_block(B);
        math(A);
        if (ib + 2 * WG < nblocks) {
            load_block(A);
        }
        math(B);
    }
    if (ib < nblocks) {
        math(A);
    }

#pragma unroll
    for (int a = 0; a < NR * NC; ++a) {
        slm_scalar_store<float>(SLM_SUMS + (tid * NR * NC + a) * 4, reduce<float>(acc[a], std::plus<>{}));
    }
    barrier();

    // thread r finishes row r (all NC columns): sum the WG partials of
    // accumulator r (and RPW+r for the gate)
    if (tid < RPW && row0 + tid < nrows) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float u = 0.0f, g = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_scalar_load<float>(SLM_SUMS + (q * NR * NC + tid * NC + c) * 4);
                if constexpr (NM == 2) {
                    g += slm_scalar_load<float>(SLM_SUMS + (q * NR * NC + (RPW + tid) * NC + c) * 4);
                }
            }
            if constexpr (NM == 2) {
                // SWIGLU: silu(gate) * up, gate = matrix 1, up = matrix 0
                simd<float, 1> gv(g);
                simd<float, 1> sg = gv / (1.0f + exp(-gv));
                u *= sg[0];
            }
            dst[(size_t) c * stride_dst + row0 + tid] = u;
        }
    }
}

// ---------------------------------------------------------------------------
// IQ4_XS x q8_1 GEMV on the XMX unit (dpas), NC = 4 y columns (MTP verification
// batches). One work-group = 16 output rows (one dpas N tile); its WG threads
// split the 256-blocks of those rows and reduce through SLM.
//
// Per block and thread: the 16 rows x 128 B of qs are read with four LSC 2D
// transposed block loads (8 dwords x 16 rows each), which lands raw dword i of
// row n at T[k][16*i + n] -- already the VNNI B layout (dword i of column n =
// four consecutive k). The nibbles are dequantized without a table:
//   K[v] + 127 = 16 v + ((v^4 + 52 v^3 - 1608 v^2 + 8997 v + 721) >> 10),
// exact for v = 0..15 with Horner in int16. The four nibble streams of a raw
// dword (bytes 0/2 lo, 1/3 lo, 0/2 hi, 1/3 hi) are evaluated on 64 int16 lanes
// and recombined into the u8 VNNI dwords (K + 127 in 0..240) with a shift + or,
// then C[NC x 16] = dpas(B u8, A s8 = the y columns' 32 q8 of the sub-block).
// The -127 * sum(y) term uses the int16 sum(q8) that
// quantize_and_reorder_q8_1_soa_iq4nat stores in ds.y; y is in natural order.
// The polynomial coefficients are read from a 16 B device buffer (coef): as
// immediates IGC folds them into separate mul + add instead of the int16 mad,
// and as kernel arguments the schedule came out 20-30% slower.
//
// Why NC = 4 only, measured on B65 (5120x17408, n = 4): dp4a ESIMD 155 us,
// this 132 us (down 17408x5120: 140 -> 120). At n = 1..3 the dp4a kernel is
// as fast or faster (n=1 102 vs 119: the 2D-load skeleton alone is ~85 us and
// the polynomial does not overlap it), so the dispatcher keeps it there.
// Needs the 256-GRF mode (the launcher sets grf_size<256>; at 128 GRF it spills).
// ---------------------------------------------------------------------------
template <int NC>
ESIMD_INLINE void mul_mat_vec_iq4_xs_q8_1_dpas(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const int16_t * coef, const int n_row_groups, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace ex = sycl::ext::intel::experimental::esimd;
    constexpr int WG  = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    constexpr int RPW = 16;
    static_assert(RPW % WG == 0, "epilogue: RPW / WG rows per thread");
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<SLM_SUMS + WG * RPW * NC * 4>();

    const int        nblocks = ncols / QK_K;
    const size_t     nb      = (size_t) nrows * nblocks;
    const uint8_t *  qs_base = (const uint8_t *) vx;
    const uint32_t * sl32    = (const uint32_t *) (qs_base + nb * (QK_K / 2));
    const uint32_t * pk32    = (const uint32_t *) (qs_base + nb * (QK_K / 2) + nb * 4);
    const int        tid     = it.get_local_id(0);
    const int        wg      = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0    = n_row_groups > 0 ? (wg / n_row_groups) * NC : 0;
    const int        row0    = row_gid * RPW;
    const unsigned   pitch   = (unsigned) nblocks * (QK_K / 2);

    simd<float, 16> acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;

    simd<uint32_t, RPW> soff;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int rowc = row0 + r < nrows ? row0 + r : nrows - 1;
        soff[r] = ((uint32_t) rowc * nblocks + tid) * 4u;
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) (col0 + c) * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;
    }

    struct blk {
        simd<uint32_t, 128> T[4];  // T[k][16*i + n]: dword 8k+i of row n
        simd<uint32_t, RPW> sl, pk;
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    int xib = tid;
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            b.y8[c]  = block_load<int8_t, 256>(yp[c]);
            b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
        b.sl = gather<uint32_t, RPW>(sl32, soff);
        b.pk = gather<uint32_t, RPW>(pk32, soff);
        soff += (uint32_t) (WG * 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t *) qs_base, pitch - 1, nrows - 1,
                                                                     pitch - 1, xib * 32 + 8 * k, row0);
        }
        xib += WG;
    };

    blk A, B;
    if (tid < nblocks) load_block(A);

    // {52, -1608, 8997, 721}: read from memory, see the header note (as kernel arguments
    // the same code came out at 158-175 us instead of 132)
    const simd<int16_t, 8> kc = block_load<int16_t, 8>(coef);
    const int16_t c0 = kc[0], c1 = kc[1], c2 = kc[2], c3 = kc[3];
    auto kpoly = [&](simd<uint32_t, 32> s) -> simd<uint32_t, 32> {
        simd<int16_t, 64> v = s.template bit_cast_view<int16_t>();
        simd<int16_t, 64> h = v + c0;
        h = h * v + c1;
        h = h * v + c2;
        h = h * v + c3;
        h = h >> 10;
        h = h + (v << 4);
        return h.template bit_cast_view<uint32_t>();
    };
    // 32 raw dwords -> 32 VNNI dwords of K+127 for the lo nibbles and 32 for the hi nibbles
    auto dequant = [&](simd<uint32_t, 32> src, simd<uint32_t, 32> & lo, simd<uint32_t, 32> & hi) {
        simd<uint32_t, 32> e_lo = src & 0x000F000Fu;
        simd<uint32_t, 32> o_lo = (src >> 8u) & 0x000F000Fu;
        simd<uint32_t, 32> e_hi = (src >> 4u) & 0x000F000Fu;
        simd<uint32_t, 32> o_hi = (src >> 12u) & 0x000F000Fu;
        lo = kpoly(e_lo) | (kpoly(o_lo) << 8u);
        hi = kpoly(e_hi) | (kpoly(o_hi) << 8u);
    };

    auto math = [&](blk & b) {
        simd<uint16_t, 16>   dbits = convert<uint16_t>(b.pk >> 16u);
        simd<sycl::half, 16> dh    = dbits.template bit_cast_view<sycl::half>();
        simd<float, 16>      d16   = convert<float>(dh);
        simd<uint32_t, 16>   sh16  = b.pk & 0xFFFFu;
        simd<float, 8>       dy[NC];
        simd<float, 8>       ysum[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            dy[c] = convert<float>(dyh);
            simd<int16_t, 8> ys = b.yds[c].template select<8, 2>(1).template bit_cast_view<int16_t>();
            ysum[c] = convert<float>(ys) * 127.0f;
        }

#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            simd<uint32_t, 128> Bt;
#pragma unroll
            for (int ip = 0; ip < 4; ip += 2) {
                simd<uint32_t, 32> src = b.T[sb / 2].template select<32, 1>(16 * ((sb % 2) * 4 + ip));
                simd<uint32_t, 32> lo, hi;
                dequant(src, lo, hi);
                Bt.template select<32, 1>(16 * ip)      = lo;
                Bt.template select<32, 1>(64 + 16 * ip) = hi;
            }
            simd<int8_t, NC * 32> Am;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
            }
            simd<uint8_t, 512>  B8 = Bt.template bit_cast_view<uint8_t>();
            simd<int, NC * 16>  C  = xmx::dpas<8, NC, int, int, uint8_t, int8_t>(simd<int, NC * 16>(0), B8, Am);

            simd<uint32_t, 16> ls = (b.sl >> (uint32_t) (4 * sb)) & 0xFu;
            ls |= ((sh16 >> (uint32_t) (2 * sb)) & 3u) << 4u;
            simd<float, 16> lsd = (convert<float>(ls) - 32.0f) * d16;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<int, 16>   ci  = C.template select<16, 1>(16 * c);
                const float     ys  = ysum[c][sb];
                const float     dyc = dy[c][sb];
                simd<float, 16> cf  = convert<float>(ci) - ys;
                acc[c] += cf * (lsd * dyc);
            }
        }
    };

    // ping-pong: the next block's loads are issued before this block's math; the
    // empty asm keeps IGC from sinking them into the math (measured 142 -> 132 us)
    int ib = tid;
    for (; ib + WG < nblocks; ib += 2 * WG) {
        load_block(B);
        __asm__ volatile("");
        math(A);
        if (ib + 2 * WG < nblocks) load_block(A);
        __asm__ volatile("");
        math(B);
    }
    if (ib < nblocks) math(A);

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NC + c) * 64, acc[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        simd<float, RT> u = 0.0f;
#pragma unroll
        for (int q = 0; q < WG; ++q) {
            u += slm_block_load<float, RT>(SLM_SUMS + (q * NC + c) * 64 + tid * RT * 4);
        }
        simd<uint32_t, RT> ridx(row0 + tid * RT, 1);
        simd_mask<RT>      m = ridx < (uint32_t) nrows;
        scatter<float, RT>(dst + (size_t) (col0 + c) * stride_dst, ridx * 4u, u, m);
    }
}


#include "iq4-dpas-ninner.hpp"

// ---------------------------------------------------------------------------
// Q4_K x q8_1 matvec, 1..4 y columns, ESIMD, dp4a on int8.
// Same as Q5_K minus the qh 5th bit. vx is Q4_K SOA (esimd_reorder_q_traits).
// vy: quantize_and_reorder_q8_1_soa_q5k_slots (slot order 0,2,1,3,4,6,5,7).
// Per sub-block: d*sc * dy * sum(q4*y8) - dmin*m * dy * sum(y8).
// ---------------------------------------------------------------------------
template <int RPW, int NC>
ESIMD_INLINE void mul_mat_vec_q4_K_q8_1_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const sycl::nd_item<1> & it, int row0) {
    using namespace sycl::ext::intel::esimd;
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    static_assert(RPW <= WG, "epilogue: one thread per row");
    static_assert(NC >= 1 && NC <= 4, "1..4 y columns");
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<WG * RPW * NC * 4>();

    const int    nblocks = ncols / QK_K;
    const size_t nb      = (size_t) nrows * nblocks;
    const auto   px      = esimd_reorder_q_traits<GGML_TYPE_Q4_K>::make_ptrs(vx, nb);
    const uint32_t * dm32 = (const uint32_t *) px.dm;

    const int tid  = it.get_local_id(0);

    simd<float, 16> acc[RPW * NC];
    simd<float, 8>  accm[RPW * NC];
#pragma unroll
    for (int r = 0; r < RPW * NC; ++r) {
        acc[r]  = 0.0f;
        accm[r] = 0.0f;
    }

    const uint8_t * qp[RPW];
    const uint8_t * sp[RPW];
    simd<uint32_t, RPW> doff;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int    rowc = row0 + r < nrows ? row0 + r : nrows - 1;
        const size_t b0   = (size_t) rowc * nblocks + tid;
        qp[r]   = px.qs + b0 * (QK_K / 2);
        sp[r]   = px.scales + b0 * K_SCALE_SIZE;
        doff[r] = (uint32_t) (b0 * 4u);
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) c * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;
    }

    struct blk {
        simd<uint8_t, 128>  qs[RPW];
        simd<uint32_t, 4>   sc[RPW];
        simd<uint32_t, RPW> dm;
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    const simd<uint32_t, 4> sc_off(0, 4);
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            b.y8[c]  = block_load<int8_t, 256>(yp[c]);
            b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
        b.dm = gather<uint32_t, RPW>(dm32, doff);
        doff += (uint32_t) (WG * 4);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            b.qs[r] = block_load<uint8_t, 128>(qp[r]);
            b.sc[r] = gather<uint32_t, 4>((const uint32_t *) sp[r], sc_off);
            qp[r] += WG * (QK_K / 2);
            sp[r] += WG * K_SCALE_SIZE;
        }
    };

    auto math = [&](blk & b) {
        simd<float, 8> dy[NC], dys[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<uint16_t, 8>   ysb = b.yds[c].template select<8, 2>(1);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            simd<int16_t, 8>    ysi = ysb.template bit_cast_view<int16_t>();
            dy[c]  = convert<float>(dyh);
            dys[c] = convert<float>(ysi) * dy[c];
        }
        simd<uint16_t, RPW>   dbits  = convert<uint16_t>(b.dm & 0xFFFFu);
        simd<uint16_t, RPW>   mbits  = convert<uint16_t>(b.dm >> 16u);
        simd<sycl::half, RPW> dh     = dbits.template bit_cast_view<sycl::half>();
        simd<sycl::half, RPW> mh     = mbits.template bit_cast_view<sycl::half>();
        simd<float, RPW>      d_r    = convert<float>(dh);
        simd<float, RPW>      dmin_r = convert<float>(mh);

#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            simd<uint8_t, 16> scb = b.sc[r].template bit_cast_view<uint8_t>();
            simd<float, 8>    scale_f, min_f;
            unpack_scale_min_k4(scb.template select<12, 1>(0), d_r[r], dmin_r[r], scale_f, min_f);
            simd<float, 8> scale_s = scale_f, min_s = min_f;
            scale_s.template select<2, 4>(1) = scale_f.template select<2, 4>(2);
            scale_s.template select<2, 4>(2) = scale_f.template select<2, 4>(1);
            min_s.template select<2, 4>(1)   = min_f.template select<2, 4>(2);
            min_s.template select<2, 4>(2)   = min_f.template select<2, 4>(1);

            simd<float, 8> sdy[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sdy[c] = scale_s * dy[c];
            }

            simd<uint32_t, 32> q32 = b.qs[r].template bit_cast_view<uint32_t>();

#pragma unroll
            for (int j = 0; j < 2; ++j) {
                simd<uint32_t, 16> q16  = q32.template select<16, 1>(16 * j);
                simd<uint32_t, 16> lo16 = q16 & 0x0F0F0F0Fu;
                simd<uint32_t, 16> hi16 = (q16 >> 4u) & 0x0F0F0F0Fu;
                simd<int, 16>      klo  = lo16.template bit_cast_view<int>();
                simd<int, 16>      khi  = hi16.template bit_cast_view<int>();

#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int        a    = r * NC + c;
                    simd<int8_t, 64> yblo = b.y8[c].template select<64, 1>(128 * j);
                    simd<int8_t, 64> ybhi = b.y8[c].template select<64, 1>(128 * j + 64);
                    simd<int, 16>    ylo  = yblo.template bit_cast_view<int>();
                    simd<int, 16>    yhi  = ybhi.template bit_cast_view<int>();
                    simd<float, 16>  plo  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), klo, ylo));
                    simd<float, 16>  phi  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), khi, yhi));
                    const float s0 = sdy[c][4 * j + 0], s1 = sdy[c][4 * j + 1];
                    const float s2 = sdy[c][4 * j + 2], s3 = sdy[c][4 * j + 3];
                    acc[a].template select<8, 1>(0) += plo.template select<8, 1>(0) * s0;
                    acc[a].template select<8, 1>(8) += plo.template select<8, 1>(8) * s1;
                    acc[a].template select<8, 1>(0) += phi.template select<8, 1>(0) * s2;
                    acc[a].template select<8, 1>(8) += phi.template select<8, 1>(8) * s3;
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                accm[r * NC + c] += min_s * dys[c];
            }
        }
    };

    blk A, B;
    int ib = tid;
    if (ib < nblocks) {
        load_block(A);
    }
    for (; ib + WG < nblocks; ib += 2 * WG) {
        load_block(B);
        math(A);
        if (ib + 2 * WG < nblocks) {
            load_block(A);
        }
        math(B);
    }
    if (ib < nblocks) {
        math(A);
    }

#pragma unroll
    for (int a = 0; a < RPW * NC; ++a) {
        const float s = reduce<float>(acc[a], std::plus<>{}) + reduce<float>(accm[a], std::plus<>{});
        slm_scalar_store<float>(SLM_SUMS + (tid * RPW * NC + a) * 4, s);
    }
    barrier();

    if (tid < RPW && row0 + tid < nrows) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_scalar_load<float>(SLM_SUMS + (q * RPW * NC + tid * NC + c) * 4);
            }
            dst[(size_t) c * stride_dst + row0 + tid] = u;
        }
    }
}

// Fused Q4_K up+gate+SWIGLU. vx0=up, vx1=gate. Shared y, silu(gate)*up.
// Not fused MMVQ GLU (declined when Q4K_ESIMD). NM=2. RPW kept small (IQ4 fused RPW=2).
// MDBLK: SoA metadata via block_load<u8,12> scales + block_load<u32,1> dm (mac_pair load),
// not gather, not packed 256 B qs∥sc∥dm.
// LSC: qs via experimental lsc_block_load L1 streaming / L2 cached (not prefetch, not pack).
// YLSC: shared q8_1 y via lsc_block_load L1 cached (activation reused across NM=2; not qs LSC).
// YU32: y via block_load<uint32_t,64> + bitcast (same 256 B; not LSC hint).
// LD2D: qs via lsc_load_2d<u8,64,RPW> lo/hi (2D LSC, pitch nblocks*128). Not 1D lsc_block_load, not PF1D.
template <int RPW, int NC, bool PF = false, bool MDBLK = false, bool LSC = false, bool YLSC = false, bool YU32 = false,
          bool LD2D = false, int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE>
ESIMD_INLINE void mul_mat_vec_q4_K_q8_1_glu_esimd(
        const void * vx0, const void * vx1, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace ex = sycl::ext::intel::experimental::esimd;
    constexpr int NM = 2;
    constexpr int NR = RPW * NM;
    static_assert(RPW <= WG, "epilogue: one thread per row");
    static_assert(NC >= 1 && NC <= 4, "1..4 y columns");
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<WG * NR * NC * 4>();

    const int    nblocks = ncols / QK_K;
    const size_t nb      = (size_t) nrows * nblocks;
    const auto   p0      = esimd_reorder_q_traits<GGML_TYPE_Q4_K>::make_ptrs(vx0, nb);
    const auto   p1      = esimd_reorder_q_traits<GGML_TYPE_Q4_K>::make_ptrs(vx1, nb);
    const uint32_t * dm32[NM] = { (const uint32_t *) p0.dm, (const uint32_t *) p1.dm };

    const int tid  = it.get_local_id(0);
    const int row0 = it.get_group(0) * RPW;

    simd<float, 16> acc[NR * NC];
    simd<float, 8>  accm[NR * NC];
#pragma unroll
    for (int r = 0; r < NR * NC; ++r) {
        acc[r]  = 0.0f;
        accm[r] = 0.0f;
    }

    const uint8_t * qp[NM][RPW];
    const uint8_t * sp[NM][RPW];
    const uint32_t * dp[NM][RPW];
    simd<uint32_t, RPW> doff[NM];
#pragma unroll
    for (int m = 0; m < NM; ++m) {
        const auto & px = m == 0 ? p0 : p1;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int    rowc = row0 + r < nrows ? row0 + r : nrows - 1;
            const size_t b0   = (size_t) rowc * nblocks + tid;
            qp[m][r]   = px.qs + b0 * (QK_K / 2);
            sp[m][r]   = px.scales + b0 * K_SCALE_SIZE;
            dp[m][r]   = dm32[m] + b0;
            doff[m][r] = (uint32_t) (b0 * 4u);
        }
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) c * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;
    }

    struct blk {
        simd<uint8_t, 128>  qs[NM][RPW];
        simd<uint32_t, 4>   sc[NM][RPW];
        simd<uint32_t, RPW> dm[NM];
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    [[maybe_unused]] const simd<uint32_t, 4> sc_off(0, 4);
    const uint8_t * qs_base[NM] = { p0.qs, p1.qs };
    const unsigned  qs_pitch    = (unsigned) nblocks * (QK_K / 2);
    int             xib         = tid;
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if constexpr (YLSC) {
                b.y8[c] = sycl::ext::intel::experimental::esimd::lsc_block_load<
                        int8_t, 256, sycl::ext::intel::esimd::detail::lsc_data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(yp[c]);
                b.yds[c] = sycl::ext::intel::experimental::esimd::lsc_block_load<
                        uint16_t, 16, sycl::ext::intel::esimd::detail::lsc_data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(ydp[c]);
            } else if constexpr (YU32) {
                simd<uint32_t, 64> y32 = block_load<uint32_t, 64>((const uint32_t *) yp[c]);
                b.y8[c]  = y32.template bit_cast_view<int8_t>();
                b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            } else {
                b.y8[c]  = block_load<int8_t, 256>(yp[c]);
                b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            }
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
        if constexpr (LD2D) {
            if (row0 + RPW <= nrows) {
#pragma unroll
                for (int m = 0; m < NM; ++m) {
                    b.dm[m] = gather<uint32_t, RPW>(dm32[m], doff[m]);
                    doff[m] += (uint32_t) (WG * 4);
                    simd<uint8_t, 64 * RPW> lo = ex::lsc_load_2d<uint8_t, 64, RPW, 1, false, false>(
                            qs_base[m], (int) qs_pitch - 1, nrows - 1, (int) qs_pitch - 1, xib * (QK_K / 2), row0);
                    simd<uint8_t, 64 * RPW> hi = ex::lsc_load_2d<uint8_t, 64, RPW, 1, false, false>(
                            qs_base[m], (int) qs_pitch - 1, nrows - 1, (int) qs_pitch - 1, xib * (QK_K / 2) + 64,
                            row0);
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        b.qs[m][r].template select<64, 1>(0)  = lo.template select<64, 1>(64 * r);
                        b.qs[m][r].template select<64, 1>(64) = hi.template select<64, 1>(64 * r);
                        b.sc[m][r] = gather<uint32_t, 4>((const uint32_t *) sp[m][r], sc_off);
                        sp[m][r] += WG * K_SCALE_SIZE;
                    }
                }
                xib += WG;
            } else {
#pragma unroll
                for (int m = 0; m < NM; ++m) {
                    b.dm[m] = gather<uint32_t, RPW>(dm32[m], doff[m]);
                    doff[m] += (uint32_t) (WG * 4);
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        b.qs[m][r] = block_load<uint8_t, 128>(qp[m][r]);
                        b.sc[m][r] = gather<uint32_t, 4>((const uint32_t *) sp[m][r], sc_off);
                        qp[m][r] += WG * (QK_K / 2);
                        sp[m][r] += WG * K_SCALE_SIZE;
                    }
                }
            }
        } else {
#pragma unroll
            for (int m = 0; m < NM; ++m) {
                if constexpr (MDBLK) {
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        b.qs[m][r] = block_load<uint8_t, 128>(qp[m][r]);
                        simd<uint8_t, 16> scu = 0;
                        scu.template select<12, 1>(0) = block_load<uint8_t, 12>(sp[m][r]);
                        b.sc[m][r] = scu.template bit_cast_view<uint32_t>();
                        simd<uint32_t, 1> dmv = block_load<uint32_t, 1>(dp[m][r]);
                        b.dm[m].template select<1, 1>(r) = dmv;
                        qp[m][r] += WG * (QK_K / 2);
                        sp[m][r] += WG * K_SCALE_SIZE;
                        dp[m][r] += WG;
                    }
                } else {
                    b.dm[m] = gather<uint32_t, RPW>(dm32[m], doff[m]);
                    doff[m] += (uint32_t) (WG * 4);
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        if constexpr (LSC) {
                            b.qs[m][r] = sycl::ext::intel::experimental::esimd::lsc_block_load<
                                    uint8_t, 128, sycl::ext::intel::esimd::detail::lsc_data_size::default_size,
                                    cache_hint::streaming, cache_hint::cached>(qp[m][r]);
                        } else {
                            b.qs[m][r] = block_load<uint8_t, 128>(qp[m][r]);
                        }
                        b.sc[m][r] = gather<uint32_t, 4>((const uint32_t *) sp[m][r], sc_off);
                        qp[m][r] += WG * (QK_K / 2);
                        sp[m][r] += WG * K_SCALE_SIZE;
                    }
                }
            }
        }
    };

    auto math = [&](blk & b) {
        simd<float, 8> dy[NC], dys[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<uint16_t, 8>   ysb = b.yds[c].template select<8, 2>(1);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            simd<int16_t, 8>    ysi = ysb.template bit_cast_view<int16_t>();
            dy[c]  = convert<float>(dyh);
            dys[c] = convert<float>(ysi) * dy[c];
        }
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            simd<uint16_t, RPW>   dbits  = convert<uint16_t>(b.dm[m] & 0xFFFFu);
            simd<uint16_t, RPW>   mbits  = convert<uint16_t>(b.dm[m] >> 16u);
            simd<sycl::half, RPW> dh     = dbits.template bit_cast_view<sycl::half>();
            simd<sycl::half, RPW> mh     = mbits.template bit_cast_view<sycl::half>();
            simd<float, RPW>      d_r    = convert<float>(dh);
            simd<float, RPW>      dmin_r = convert<float>(mh);
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                simd<uint8_t, 16> scb = b.sc[m][r].template bit_cast_view<uint8_t>();
                simd<float, 8>    scale_f, min_f;
                unpack_scale_min_k4(scb.template select<12, 1>(0), d_r[r], dmin_r[r], scale_f, min_f);
                simd<float, 8> scale_s = scale_f, min_s = min_f;
                scale_s.template select<2, 4>(1) = scale_f.template select<2, 4>(2);
                scale_s.template select<2, 4>(2) = scale_f.template select<2, 4>(1);
                min_s.template select<2, 4>(1)   = min_f.template select<2, 4>(2);
                min_s.template select<2, 4>(2)   = min_f.template select<2, 4>(1);
                simd<float, 8> sdy[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    sdy[c] = scale_s * dy[c];
                }
                simd<uint32_t, 32> q32 = b.qs[m][r].template bit_cast_view<uint32_t>();
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    simd<uint32_t, 16> q16  = q32.template select<16, 1>(16 * j);
                    simd<uint32_t, 16> lo16 = q16 & 0x0F0F0F0Fu;
                    simd<uint32_t, 16> hi16 = (q16 >> 4u) & 0x0F0F0F0Fu;
                    simd<int, 16>      klo  = lo16.template bit_cast_view<int>();
                    simd<int, 16>      khi  = hi16.template bit_cast_view<int>();
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const int        a    = (m * RPW + r) * NC + c;
                        simd<int8_t, 64> yblo = b.y8[c].template select<64, 1>(128 * j);
                        simd<int8_t, 64> ybhi = b.y8[c].template select<64, 1>(128 * j + 64);
                        simd<int, 16>    ylo  = yblo.template bit_cast_view<int>();
                        simd<int, 16>    yhi  = ybhi.template bit_cast_view<int>();
                        simd<float, 16>  plo  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), klo, ylo));
                        simd<float, 16>  phi  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), khi, yhi));
                        const float s0 = sdy[c][4 * j + 0], s1 = sdy[c][4 * j + 1];
                        const float s2 = sdy[c][4 * j + 2], s3 = sdy[c][4 * j + 3];
                        acc[a].template select<8, 1>(0) += plo.template select<8, 1>(0) * s0;
                        acc[a].template select<8, 1>(8) += plo.template select<8, 1>(8) * s1;
                        acc[a].template select<8, 1>(0) += phi.template select<8, 1>(0) * s2;
                        acc[a].template select<8, 1>(8) += phi.template select<8, 1>(8) * s3;
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    accm[(m * RPW + r) * NC + c] += min_s * dys[c];
                }
            }
        }
    };

    blk A, B;
    int ib = tid;
    if (ib < nblocks) {
        load_block(A);
    }
    for (; ib + WG < nblocks; ib += 2 * WG) {
        load_block(B);
        const bool more = ib + 2 * WG < nblocks;
        if constexpr (PF) {
            if (more) {
#pragma unroll
                for (int m = 0; m < NM; ++m) {
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        sycl::ext::intel::experimental::esimd::lsc_prefetch<
                                uint8_t, 128, sycl::ext::intel::esimd::detail::lsc_data_size::default_size,
                                cache_hint::cached, cache_hint::cached>(qp[m][r]);
                    }
                }
            }
        }
        math(A);
        if (more) {
            load_block(A);
        }
        math(B);
    }
    if (ib < nblocks) {
        math(A);
    }

#pragma unroll
    for (int a = 0; a < NR * NC; ++a) {
        const float s = reduce<float>(acc[a], std::plus<>{}) + reduce<float>(accm[a], std::plus<>{});
        slm_scalar_store<float>(SLM_SUMS + (tid * NR * NC + a) * 4, s);
    }
    barrier();

    if (tid < RPW && row0 + tid < nrows) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float u = 0.0f, g = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_scalar_load<float>(SLM_SUMS + (q * NR * NC + tid * NC + c) * 4);
                g += slm_scalar_load<float>(SLM_SUMS + (q * NR * NC + (RPW + tid) * NC + c) * 4);
            }
            simd<float, 1> gv(g);
            simd<float, 1> sg = gv / (1.0f + exp(-gv));
            dst[(size_t) c * stride_dst + row0 + tid] = u * sg[0];
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_K x q8_1 matvec, 1..4 y columns, ESIMD, dp4a on int8.
//
// vx: Q5_K SOA reorder layout (esimd_reorder_q_traits<GGML_TYPE_Q5_K>).
// vy: activation quantized by quantize_and_reorder_q8_1_soa_q5k_slots: the
//     SoA q8_1 layout ([int8 q[ncols]] [half2 {d,sum}[ncols/32]] per column)
//     with the 8 sub-blocks of every 256-block stored in slot order
//     (0,2,1,3,4,6,5,7). With that, for chunk pair j (weights 128j..128j+127,
//     qs bytes 64j..64j+63 as 16 dwords) the low nibbles are the 64 weights
//     {128j..+31, 128j+64..+95} = y bytes [128j, 128j+64) and the high nibbles
//     {128j+32..+63, 128j+96..+127} = y bytes [128j+64, 128j+128): two SIMD16
//     dp4a per 128 weights with no in-kernel permutation. The 5th bit of
//     weight w is bit w/32 of qh[w%32].
// Per sub-block: d*sc * dy * sum(q5*y8) - dmin*m * dy * sum(y8), the
// vec_dot_q5_K_q8_1_impl_vmmq estimator (sum(y8) is stored as int16 in the
// ds.y slot by that quantizer).
//
// Why: the float DMMV ESIMD kernel (n=1) is instruction-bound at ~103 us for
// the 36 MB GDN qkv matrix (floor 61 us) and n=2..4 (MTP verification) went
// through MMVQ at ~230 us. Here ~11 SIMD16 ops unpack 128 weights of one row
// and each y column costs 2 dp4a + 2 cvt + 4 mad per 128 weights.
// ---------------------------------------------------------------------------
template <int RPW, int NC>
ESIMD_INLINE void mul_mat_vec_q5_K_q8_1_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    static_assert(RPW <= WG, "epilogue: one thread per row");
    static_assert(NC >= 1 && NC <= 4, "1..4 y columns");
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<WG * RPW * NC * 4>();

    const int    nblocks = ncols / QK_K;
    const size_t nb      = (size_t) nrows * nblocks;
    const auto   px      = esimd_reorder_q_traits<GGML_TYPE_Q5_K>::make_ptrs(vx, nb);
    const uint32_t * dm32 = (const uint32_t *) px.dm;

    const int tid  = it.get_local_id(0);
    const int row0 = it.get_group(0) * RPW;

    simd<float, 16> acc[RPW * NC];   // [r * NC + c], lanes: dwords of a chunk pair
    simd<float, 8>  accm[RPW * NC];  // min term per slot
#pragma unroll
    for (int r = 0; r < RPW * NC; ++r) {
        acc[r]  = 0.0f;
        accm[r] = 0.0f;
    }

    // per-row streams walked with incremented pointers/offsets (see the IQ4_XS kernel)
    const uint8_t * qp[RPW];
    const uint8_t * hp[RPW];
    const uint8_t * sp[RPW];
    simd<uint32_t, RPW> doff;  // byte offset of dm[rowc * nblocks + ib]
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int    rowc = row0 + r < nrows ? row0 + r : nrows - 1;
        const size_t b0   = (size_t) rowc * nblocks + tid;
        qp[r]   = px.qs + b0 * (QK_K / 2);
        hp[r]   = px.qh + b0 * (QK_K / 8);
        sp[r]   = px.scales + b0 * K_SCALE_SIZE;
        doff[r] = (uint32_t) (b0 * 4u);
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) c * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;
    }

    struct blk {
        simd<uint8_t, 128>  qs[RPW];
        simd<uint8_t, 32>   qh[RPW];
        simd<uint32_t, 4>   sc[RPW];  // 12 scale bytes (+4 over-read into the dm array)
        simd<uint32_t, RPW> dm;
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    const simd<uint32_t, 4> sc_off(0, 4);
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            b.y8[c]  = block_load<int8_t, 256>(yp[c]);
            b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
        b.dm = gather<uint32_t, RPW>(dm32, doff);
        doff += (uint32_t) (WG * 4);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            b.qs[r] = block_load<uint8_t, 128>(qp[r]);
            b.qh[r] = block_load<uint8_t, 32>(hp[r]);
            b.sc[r] = gather<uint32_t, 4>((const uint32_t *) sp[r], sc_off);
            qp[r] += WG * (QK_K / 2);
            hp[r] += WG * (QK_K / 8);
            sp[r] += WG * K_SCALE_SIZE;
        }
    };

    // 5th bit of weight w is bit w/32 of qh[w%32]: for chunk pair j the low
    // nibbles are weights 128j.. (lanes 0-7, bit 4j) and 128j+64.. (lanes 8-15,
    // bit 4j+2); the high nibbles are +32 of each (bits 4j+1, 4j+3)
    simd<uint32_t, 16> shv0;
    shv0.template select<8, 1>(0) = 0u;
    shv0.template select<8, 1>(8) = 2u;

    auto math = [&](blk & b) {
        simd<float, 8> dy[NC], dys[NC];  // dy, dy * sum(q8) per slot
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<uint16_t, 8>   ysb = b.yds[c].template select<8, 2>(1);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            simd<int16_t, 8>    ysi = ysb.template bit_cast_view<int16_t>();
            dy[c]  = convert<float>(dyh);
            dys[c] = convert<float>(ysi) * dy[c];
        }
        simd<uint16_t, RPW>   dbits  = convert<uint16_t>(b.dm & 0xFFFFu);
        simd<uint16_t, RPW>   mbits  = convert<uint16_t>(b.dm >> 16u);
        simd<sycl::half, RPW> dh     = dbits.template bit_cast_view<sycl::half>();
        simd<sycl::half, RPW> mh     = mbits.template bit_cast_view<sycl::half>();
        simd<float, RPW>      d_r    = convert<float>(dh);
        simd<float, RPW>      dmin_r = convert<float>(mh);

#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            simd<uint8_t, 16> scb = b.sc[r].template bit_cast_view<uint8_t>();
            simd<float, 8>    scale_f, min_f;
            unpack_scale_min_k4(scb.template select<12, 1>(0), d_r[r], dmin_r[r], scale_f, min_f);
            // sub-block order -> slot order (0,2,1,3,4,6,5,7): swap lanes 1<->2, 5<->6
            simd<float, 8> scale_s = scale_f, min_s = min_f;
            scale_s.template select<2, 4>(1) = scale_f.template select<2, 4>(2);
            scale_s.template select<2, 4>(2) = scale_f.template select<2, 4>(1);
            min_s.template select<2, 4>(1)   = min_f.template select<2, 4>(2);
            min_s.template select<2, 4>(2)   = min_f.template select<2, 4>(1);

            simd<float, 8> sdy[NC];  // d*sc*dy per slot and column
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sdy[c] = scale_s * dy[c];
            }

            simd<uint32_t, 32> q32 = b.qs[r].template bit_cast_view<uint32_t>();
            simd<uint32_t, 8>  h8  = b.qh[r].template bit_cast_view<uint32_t>();
            simd<uint32_t, 16> h16;
            h16.template select<8, 1>(0) = h8;
            h16.template select<8, 1>(8) = h8;

#pragma unroll
            for (int j = 0; j < 2; ++j) {  // chunk pair: weights 128j..128j+127
                simd<uint32_t, 16> q16  = q32.template select<16, 1>(16 * j);
                simd<uint32_t, 16> shv  = shv0 + (uint32_t) (4 * j);
                simd<uint32_t, 16> hlo  = ((h16 >> shv) & 0x01010101u) << 4u;
                simd<uint32_t, 16> hhi  = ((h16 >> (shv + 1u)) & 0x01010101u) << 4u;
                simd<uint32_t, 16> lo16 = (q16 & 0x0F0F0F0Fu) | hlo;
                simd<uint32_t, 16> hi16 = ((q16 >> 4u) & 0x0F0F0F0Fu) | hhi;
                simd<int, 16>      klo  = lo16.template bit_cast_view<int>();
                simd<int, 16>      khi  = hi16.template bit_cast_view<int>();

#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int        a    = r * NC + c;
                    simd<int8_t, 64> yblo = b.y8[c].template select<64, 1>(128 * j);
                    simd<int8_t, 64> ybhi = b.y8[c].template select<64, 1>(128 * j + 64);
                    simd<int, 16>    ylo  = yblo.template bit_cast_view<int>();
                    simd<int, 16>    yhi  = ybhi.template bit_cast_view<int>();
                    simd<float, 16>  plo  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), klo, ylo));
                    simd<float, 16>  phi  = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), khi, yhi));
                    // slots 4j (lanes 0-7), 4j+1 (8-15) for lo; 4j+2, 4j+3 for hi
                    const float s0 = sdy[c][4 * j + 0], s1 = sdy[c][4 * j + 1];
                    const float s2 = sdy[c][4 * j + 2], s3 = sdy[c][4 * j + 3];
                    acc[a].template select<8, 1>(0) += plo.template select<8, 1>(0) * s0;
                    acc[a].template select<8, 1>(8) += plo.template select<8, 1>(8) * s1;
                    acc[a].template select<8, 1>(0) += phi.template select<8, 1>(0) * s2;
                    acc[a].template select<8, 1>(8) += phi.template select<8, 1>(8) * s3;
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                accm[r * NC + c] += min_s * dys[c];  // min_s = -dmin*m per slot
            }
        }
    };

    blk A, B;
    int ib = tid;
    if (ib < nblocks) {
        load_block(A);
    }
    for (; ib + WG < nblocks; ib += 2 * WG) {
        load_block(B);
        math(A);
        if (ib + 2 * WG < nblocks) {
            load_block(A);
        }
        math(B);
    }
    if (ib < nblocks) {
        math(A);
    }

#pragma unroll
    for (int a = 0; a < RPW * NC; ++a) {
        const float s = reduce<float>(acc[a], std::plus<>{}) + reduce<float>(accm[a], std::plus<>{});
        slm_scalar_store<float>(SLM_SUMS + (tid * RPW * NC + a) * 4, s);
    }
    barrier();

    if (tid < RPW && row0 + tid < nrows) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_scalar_load<float>(SLM_SUMS + (q * RPW * NC + tid * NC + c) * 4);
            }
            dst[(size_t) c * stride_dst + row0 + tid] = u;
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K x q8_1 matvec, 1..4 y columns, ESIMD, dp4a on int8.
//
// vx: Q6_K SOA reorder layout (esimd_reorder_q_traits<GGML_TYPE_Q6_K>):
//     [ql][qh][scales int8][d half]. Weight order matches dequantize_row_q6_K.
// vy: activation quantized by quantize_and_reorder_q8_1_soa (natural 32-wide
//     sub-block order 0..7). No min term, so the {d,sum} sum half is unused.
// Per 16-weight scale: d * sc * dy * sum((q6-32)*y8).
// ---------------------------------------------------------------------------
template <int RPW, int NC>
ESIMD_INLINE void mul_mat_vec_q6_K_q8_1_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int stride_y_bytes, const int stride_dst,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    static_assert(RPW <= WG, "epilogue: one thread per row");
    static_assert(NC >= 1 && NC <= 4, "1..4 y columns");
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<WG * RPW * NC * 4>();

    const int    nblocks = ncols / QK_K;
    const size_t nb      = (size_t) nrows * nblocks;
    const auto   px      = esimd_reorder_q_traits<GGML_TYPE_Q6_K>::make_ptrs(vx, nb);
    const uint16_t * d16 = (const uint16_t *) px.d;

    const int tid  = it.get_local_id(0);
    const int row0 = it.get_group(0) * RPW;

    simd<float, 8> acc[RPW * NC];
#pragma unroll
    for (int r = 0; r < RPW * NC; ++r) {
        acc[r] = 0.0f;
    }

    const uint8_t * ql[RPW];
    const uint8_t * qh[RPW];
    const int8_t  * sp[RPW];
    simd<uint32_t, RPW> doff;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int    rowc = row0 + r < nrows ? row0 + r : nrows - 1;
        const size_t b0   = (size_t) rowc * nblocks + tid;
        ql[r]   = px.ql + b0 * (QK_K / 2);
        qh[r]   = px.qh + b0 * (QK_K / 4);
        sp[r]   = px.scales + b0 * (QK_K / 16);
        doff[r] = (uint32_t) (b0 * 2u);
    }
    const int8_t *   yp[NC];
    const uint16_t * ydp[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const char * ycol = (const char *) vy + (size_t) c * stride_y_bytes;
        yp[c]  = (const int8_t *) ycol + (size_t) tid * QK_K;
        ydp[c] = (const uint16_t *) (ycol + ncols) + (size_t) tid * 16;
    }

    struct blk {
        simd<uint8_t, 128>  ql[RPW];
        simd<uint8_t, 64>   qh[RPW];
        simd<int8_t, 16>    sc[RPW];
        simd<uint16_t, RPW> d;
        simd<int8_t, 256>   y8[NC];
        simd<uint16_t, 16>  yds[NC];
    };
    auto load_block = [&](blk & b) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            b.y8[c]  = block_load<int8_t, 256>(yp[c]);
            b.yds[c] = block_load<uint16_t, 16>(ydp[c]);
            yp[c]  += WG * QK_K;
            ydp[c] += WG * 16;
        }
        b.d = gather<uint16_t, RPW>(d16, doff);
        doff += (uint32_t) (WG * 2);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            b.ql[r] = block_load<uint8_t, 128>(ql[r]);
            b.qh[r] = block_load<uint8_t, 64>(qh[r]);
            b.sc[r] = block_load<int8_t, 16>(sp[r]);
            ql[r] += WG * (QK_K / 2);
            qh[r] += WG * (QK_K / 4);
            sp[r] += WG * (QK_K / 16);
        }
    };

    auto math = [&](blk & b) {
        simd<float, 8> dy[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            simd<uint16_t, 8>   dyb = b.yds[c].template select<8, 2>(0);
            simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
            dy[c] = convert<float>(dyh);
        }
        simd<sycl::half, RPW> dh  = b.d.template bit_cast_view<sycl::half>();
        simd<float, RPW>      d_r = convert<float>(dh);

#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            simd<float, 16> scf = convert<float>(b.sc[r]);
#pragma unroll
            for (int im = 0; im < 2; ++im) {
                simd<uint8_t, 32> ql_lo = b.ql[r].template select<32, 1>(64 * im);
                simd<uint8_t, 32> ql_hi = b.ql[r].template select<32, 1>(64 * im + 32);
                simd<uint8_t, 32> qhb   = b.qh[r].template select<32, 1>(32 * im);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    simd<uint8_t, 32> qa;
                    if (g == 0) {
                        qa = (ql_lo & simd<uint8_t, 32>(0x0F)) | ((qhb & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                    } else if (g == 1) {
                        qa = (ql_hi & simd<uint8_t, 32>(0x0F)) | ((qhb & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                    } else if (g == 2) {
                        qa = (ql_lo >> simd<uint8_t, 32>(4)) | (qhb & simd<uint8_t, 32>(0x30));
                    } else {
                        qa = (ql_hi >> simd<uint8_t, 32>(4)) | ((qhb & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                    }
                    simd<int16_t, 32> q16 = convert<int16_t>(qa) - int16_t(32);
                    simd<int8_t, 32>  q8  = convert<int8_t>(q16);
                    simd<int, 8>      k   = q8.template bit_cast_view<int>();
                    const int slot = 4 * im + g;
                    const float sc0 = scf[8 * im + 2 * g + 0];
                    const float sc1 = scf[8 * im + 2 * g + 1];
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<int8_t, 32> yb = b.y8[c].template select<32, 1>(32 * slot);
                        simd<int, 8>     y  = yb.template bit_cast_view<int>();
                        simd<float, 8>   p  = convert<float>(dp4a<int, int, int, int, 8>(simd<int, 8>(0), k, y));
                        const float      s0 = d_r[r] * sc0 * dy[c][slot];
                        const float      s1 = d_r[r] * sc1 * dy[c][slot];
                        acc[r * NC + c].template select<4, 1>(0) += p.template select<4, 1>(0) * s0;
                        acc[r * NC + c].template select<4, 1>(4) += p.template select<4, 1>(4) * s1;
                    }
                }
            }
        }
    };

    blk A, B;
    int ib = tid;
    if (ib < nblocks) {
        load_block(A);
    }
    for (; ib + WG < nblocks; ib += 2 * WG) {
        load_block(B);
        math(A);
        if (ib + 2 * WG < nblocks) {
            load_block(A);
        }
        math(B);
    }
    if (ib < nblocks) {
        math(A);
    }

#pragma unroll
    for (int a = 0; a < RPW * NC; ++a) {
        slm_scalar_store<float>(SLM_SUMS + (tid * RPW * NC + a) * 4, reduce<float>(acc[a], std::plus<>{}));
    }
    barrier();

    if (tid < RPW && row0 + tid < nrows) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_scalar_load<float>(SLM_SUMS + (q * RPW * NC + tid * NC + c) * 4);
            }
            dst[(size_t) c * stride_dst + row0 + tid] = u;
        }
    }
}

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_ESIMD_HPP
