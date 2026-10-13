#include "xe2mx-a8.hpp"

bool xe2mx_quant_a_s4_nibble(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                             int64_t M, int64_t K, int64_t g, float * za) {
    if (!stream || !xh || !A4 || !sa || M < 1 || K < 2 || g < 2) {
        return false;
    }
    if ((K % g) != 0 || (g % 2) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    size_t wg = 128;
    if (g <= 64) {
        wg = (size_t) g;
    } else if (g >= 256 && (g % 256) == 0) {
        wg = 256;
    }
    if ((g % (int64_t) wg) != 0) {
        wg = 128;
    }
    if ((g % (int64_t) wg) != 0) {
        return false;
    }
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const int meanon = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44MEAN", 0);
    const int64_t kh = K / 2;
    const int wgi = (int) wg;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
            const int k0 = ig * (int) g;
            const int np = (int) g / 2;
            float v0[4];
            float v1[4];
            int npp = 0;
            float sm = 0.0f;
            for (int p = lid; p < np && npp < 4; p += wgi) {
                const int k = k0 + p * 2;
                const sycl::vec<sycl::half, 2> h =
                    *reinterpret_cast<const sycl::vec<sycl::half, 2> *>(row + k);
                const float a = (float) h[0];
                const float b = (float) h[1];
                v0[npp] = a;
                v1[npp] = b;
                npp++;
                sm += a + b;
            }
            float mean = 0.0f;
            if (meanon) {
                sm = sycl::reduce_over_group(it.get_group(), sm, sycl::plus<float>());
                mean = sm / (float) g;
            }
            float amax = 0.0f;
            for (int t = 0; t < npp; t++) {
                v0[t] -= mean;
                v1[t] -= mean;
                amax = sycl::fmax(amax, sycl::fabs(v0[t]));
                amax = sycl::fmax(amax, sycl::fabs(v1[t]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float s = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
            if (lid == 0) {
                sa[(size_t) m * (size_t) ng + (size_t) ig] = s;
            }
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            float qs = 0.0f;
            for (int t = 0; t < npp; t++) {
                int q0 = (int) sycl_rint_i(v0[t] * id);
                int q1 = (int) sycl_rint_i(v1[t] * id);
                if (q0 < -8) {
                    q0 = -8;
                }
                if (q0 > 7) {
                    q0 = 7;
                }
                if (q1 < -8) {
                    q1 = -8;
                }
                if (q1 > 7) {
                    q1 = 7;
                }
                qs += (float) (q0 + q1);
                const uint8_t u0 = (uint8_t) (int8_t) q0;
                const uint8_t u1 = (uint8_t) (int8_t) q1;
                dst[lid + t * wgi] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
            }
            if (za) {
                qs = sycl::reduce_over_group(it.get_group(), qs, sycl::plus<float>());
                if (lid == 0) {
                    za[(size_t) m * (size_t) ng + (size_t) ig] = s * qs;
                }
            }
        });
    return true;
}

template <typename Src, bool FromF32Rte>
static inline float xe2mx_apack_ld(const Src * row, int i) {
    float x;
    if constexpr (FromF32Rte) {
        const sycl::vec<float, 1> vf((float) row[i]);
        const sycl::half h = vf.template convert<sycl::half, sycl::rounding_mode::rte>()[0];
        x = (float) h;
    } else {
        x = (float) row[i];
    }
    return sycl::fmin(sycl::fmax(x, -65504.0f), 65504.0f);
}

template <typename Src, bool FromF32Rte, bool StoreS8 = false>
static bool xe2mx_quant_a_s4_nibble_had_impl(queue_ptr stream, const Src * xh, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g, int64_t lda, int64_t koff) {
    if (!stream || !xh || !A4 || !sa || M < 1 || K < 2 || g < 2) {
        return false;
    }
    if ((K % g) != 0 || (g % 2) != 0) {
        return false;
    }
    if (lda <= 0) {
        lda = K;
    }
    if (koff < 0 || (koff % g) != 0 || (lda % g) != 0 || koff + K > lda) {
        return false;
    }
    int hs = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HS", 128);
    if (hs != 32 && hs != 64 && hs != 128) {
        hs = 128;
    }
    if ((g % (int64_t) hs) != 0 || g < (int64_t) hs) {
        return false;
    }
    if (g > 1024 && (g != K || g > 17408 || hs != 128)) {
        return false;
    }
    const int64_t ng = K / g;
    const int nblk = (int) (g / (int64_t) hs);
    const int wgi = hs;
    const float inv = (hs == 32) ? 0.1767766952966369f : ((hs == 64) ? 0.125f : 0.08838834764831845f);
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const int64_t kh = K / 2;
    int hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (hilp != 2 && hilp != 4) {
        hilp = 0;
    }
    if (!FromF32Rte && sizeof(Src) > 2 && hilp == 4) {
        hilp = 2;
    }
    if constexpr (StoreS8) {
        if (!(hs == 128 && hilp == 4)) {
            return false;
        }
    }
    if (g > 1024) {
        if (StoreS8 || hilp != 4 || nblk < 16 || (nblk % 8) != 0) {
            return false;
        }
        // Two passes instead of one work-group per row holding the whole row in SLM:
        // pass 1 writes each 128-block's max, pass 2 takes the row max and quantizes.
        // Pass 2 recomputes the same Hadamard floats, so the packed bytes match.
        if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN2P", 1)) {
            static float * bm = nullptr;
            static size_t bm_cap = 0;
            const size_t bm_n = (size_t) M * (size_t) nblk;
            if (bm_cap < bm_n) {
                if (bm) {
                    sycl::free(bm, *stream);
                }
                bm = sycl::malloc_device<float>(bm_n, *stream);
                bm_cap = bm ? bm_n : 0;
            }
            if (bm) {
                float * bmp = bm;
                const size_t ntask = bm_n;
                const size_t nwg = (ntask + 7) / 8;
                const int nb = nblk;
                auto had = [=](const Src * row, int base, const sycl::sub_group & sg, int lane, float * a) {
                    if constexpr (FromF32Rte) {
                        const sycl::vec<float, 8> vf =
                            *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                        const sycl::vec<sycl::half, 8> vh =
                            vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    } else {
                        const sycl::vec<sycl::half, 8> vh =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    }
#pragma unroll
                    for (int h = 1; h < 8; h *= 2) {
#pragma unroll
                        for (int j = 0; j < 8; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = a[j + k];
                                const float ay = a[j + k + h];
                                a[j + k] = ax + ay;
                                a[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = 8, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                            a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a[i] = a[i] * inv;
                    }
                };
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(nwg * 128u), sycl::range<1>(128)),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        const int lid = (int) it.get_local_id(0);
                        const int lane = lid & 15;
                        const size_t task = (size_t) it.get_group(0) * 8u + (size_t) (lid >> 4);
                        if (task >= ntask) {
                            return;
                        }
                        const int m = (int) (task / (size_t) nb);
                        const int blk = (int) (task - (size_t) m * (size_t) nb);
                        const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                        const sycl::sub_group sg = it.get_sub_group();
                        float a[8];
                        had(row, blk * 128 + lane * 8, sg, lane, a);
                        float amax = 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            amax = sycl::fmax(amax, sycl::fabs(a[i]));
                        }
                        const float bmax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
                        if (lane == 0) {
                            bmp[task] = bmax;
                        }
                    });
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(nwg * 128u), sycl::range<1>(128)),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        const int lid = (int) it.get_local_id(0);
                        const int lane = lid & 15;
                        const size_t task = (size_t) it.get_group(0) * 8u + (size_t) (lid >> 4);
                        if (task >= ntask) {
                            return;
                        }
                        const int m = (int) (task / (size_t) nb);
                        const int blk = (int) (task - (size_t) m * (size_t) nb);
                        const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                        uint8_t * dst = A4 + (size_t) m * (size_t) kh;
                        const sycl::sub_group sg = it.get_sub_group();
                        float rmax = 0.0f;
                        for (int j = lane; j < nb; j += 16) {
                            rmax = sycl::fmax(rmax, bmp[(size_t) m * (size_t) nb + (size_t) j]);
                        }
                        const float gmax = sycl::reduce_over_group(sg, rmax, sycl::maximum<float>());
                        const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                        if (blk == 0 && lane == 0) {
                            sa[(size_t) m] = sc;
                        }
                        const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                        float a[8];
                        had(row, blk * 128 + lane * 8, sg, lane, a);
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            int q0 = (int) sycl_rint_i(a[t * 2] * id);
                            int q1 = (int) sycl_rint_i(a[t * 2 + 1] * id);
                            q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
                            q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
                            const uint8_t u0 = (uint8_t) (int8_t) q0;
                            const uint8_t u1 = (uint8_t) (int8_t) q1;
                            dst[blk * 64 + lane * 4 + t] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                        }
                    });
                return true;
            }
        }
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * 128u), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int m = (int) it.get_group(0);
                const int lid = (int) it.get_local_id(0);
                const int lane = lid & 15;
                const int sgid = lid >> 4;
                const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                uint8_t * dst = A4 + (size_t) m * (size_t) kh;
                const sycl::sub_group sg = it.get_sub_group();
                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[17408]>(it.get_group());
                float * s = *slm;
                constexpr int EL = 8;
                for (int wave = 0; wave < nblk; wave += 8) {
                    const int blk = wave + sgid;
                    float a[EL];
                    const int base = blk * 128 + lane * EL;
                    if constexpr (FromF32Rte) {
                        const sycl::vec<float, 8> vf =
                            *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                        const sycl::vec<sycl::half, 8> vh =
                            vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    } else {
                        const sycl::vec<sycl::half, 8> vh =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    }
#pragma unroll
                    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = a[j + k];
                                const float ay = a[j + k + h];
                                a[j + k] = ax + ay;
                                a[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                            a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        s[blk * 128 + lane * EL + i] = a[i] * inv;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                float amax = 0.0f;
                for (int k = lid; k < (int) K; k += 128) {
                    amax = sycl::fmax(amax, sycl::fabs(s[k]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                const int np = (int) K / 2;
                for (int p = lid; p < np; p += 128) {
                    int q0 = (int) sycl_rint_i(s[p * 2] * id);
                    int q1 = (int) sycl_rint_i(s[p * 2 + 1] * id);
                    if (q0 < -8) {
                        q0 = -8;
                    }
                    if (q0 > 7) {
                        q0 = 7;
                    }
                    if (q1 < -8) {
                        q1 = -8;
                    }
                    if (q1 > 7) {
                        q1 = 7;
                    }
                    const uint8_t u0 = (uint8_t) (int8_t) q0;
                    const uint8_t u1 = (uint8_t) (int8_t) q1;
                    dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                }
            });
        return true;
    }
    if (hs == 128 && hilp == 4) {
        const int hwg = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWG", 128);
        const int dn64 = (K >= 16384) && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN64", 0);
        if (nblk == 8 && (hwg == 64 || dn64) && !StoreS8) {
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * 64u), sycl::range<1>(64)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const int gid = (int) it.get_group(0);
                    const int m = gid / (int) ng;
                    const int ig = gid - m * (int) ng;
                    const int lid = (int) it.get_local_id(0);
                    const int lane = lid & 15;
                    const int sgid = lid >> 4;
                    const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                    uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                    const int k0 = ig * (int) g;
                    const sycl::sub_group sg = it.get_sub_group();
                    constexpr int EL = 8;
                    float a[2][EL];
                    float amax = 0.0f;
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        const int blk = sgid + b * 4;
                        const int base = k0 + blk * 128 + lane * EL;
                        if constexpr (FromF32Rte) {
                            const sycl::vec<float, 8> vf =
                                *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                            const sycl::vec<sycl::half, 8> vh =
                                vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                            for (int i = 0; i < EL; ++i) {
                                a[b][i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                            }
                        } else {
                            const sycl::vec<sycl::half, 8> vh =
                                *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                            for (int i = 0; i < EL; ++i) {
                                a[b][i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                            }
                        }
#pragma unroll
                        for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                            for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                                for (int k = 0; k < h; ++k) {
                                    const float ax = a[b][j + k];
                                    const float ay = a[b][j + k + h];
                                    a[b][j + k] = ax + ay;
                                    a[b][j + k + h] = ax - ay;
                                }
                            }
                        }
#pragma unroll
                        for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                            for (int j = 0; j < EL; ++j) {
                                const float a2 = sycl::permute_group_by_xor(sg, a[b][j], (uint32_t) xm);
                                a[b][j] = (lane & xm) == 0 ? a[b][j] + a2 : a2 - a[b][j];
                            }
                        }
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[b][i] = a[b][i] * inv;
                            amax = sycl::fmax(amax, sycl::fabs(a[b][i]));
                        }
                    }
                    const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                    const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                    if (lid == 0) {
                        sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                    }
                    const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        const int blk = sgid + b * 4;
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            int q0 = (int) sycl_rint_i(a[b][t * 2] * id);
                            int q1 = (int) sycl_rint_i(a[b][t * 2 + 1] * id);
                            if (q0 < -8) {
                                q0 = -8;
                            }
                            if (q0 > 7) {
                                q0 = 7;
                            }
                            if (q1 < -8) {
                                q1 = -8;
                            }
                            if (q1 > 7) {
                                q1 = 7;
                            }
                            const uint8_t u0 = (uint8_t) (int8_t) q0;
                            const uint8_t u1 = (uint8_t) (int8_t) q1;
                            dst[blk * 64 + lane * 4 + t] =
                                (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                        }
                    }
                });
            return true;
        }
        const int hilp2 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP2", 0);
        if (hilp2 && !StoreS8 && nblk == 8) {
            const int64_t ng2 = (ng + 1) / 2;
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng2 * 128u), sycl::range<1>(128)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const int gid = (int) it.get_group(0);
                    const int m = gid / (int) ng2;
                    const int pair = gid - m * (int) ng2;
                    const int lid = (int) it.get_local_id(0);
                    const int lane = lid & 15;
                    const int sgid = lid >> 4;
                    const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                    uint8_t * dst0 = A4 + (size_t) m * (size_t) kh;
                    const sycl::sub_group sg = it.get_sub_group();
                    constexpr int EL = 8;
                    for (int t = 0; t < 2; ++t) {
                        const int ig = pair * 2 + t;
                        if (ig >= (int) ng) {
                            break;
                        }
                        uint8_t * dst = dst0 + (size_t) (ig * (int) g / 2);
                        const int k0 = ig * (int) g;
                        float a[EL];
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = 0.0f;
                        }
                        float amax = 0.0f;
                        if (sgid < nblk) {
                            const int base = k0 + sgid * 128 + lane * EL;
                            if constexpr (FromF32Rte) {
                                const sycl::vec<float, 8> vf =
                                    *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                                const sycl::vec<sycl::half, 8> vh =
                                    vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                                for (int i = 0; i < EL; ++i) {
                                    a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                                }
                            } else {
                                const sycl::vec<sycl::half, 8> vh =
                                    *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                                for (int i = 0; i < EL; ++i) {
                                    a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                                }
                            }
#pragma unroll
                            for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                                for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                                    for (int k = 0; k < h; ++k) {
                                        const float ax = a[j + k];
                                        const float ay = a[j + k + h];
                                        a[j + k] = ax + ay;
                                        a[j + k + h] = ax - ay;
                                    }
                                }
                            }
#pragma unroll
                            for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                                for (int j = 0; j < EL; ++j) {
                                    const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                                    a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                                }
                            }
#pragma unroll
                            for (int i = 0; i < EL; ++i) {
                                a[i] = a[i] * inv;
                                amax = sycl::fmax(amax, sycl::fabs(a[i]));
                            }
                        }
                        const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                        const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                        if (lid == 0) {
                            sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                        }
                        const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                        if (sgid < nblk) {
#pragma unroll
                            for (int tt = 0; tt < 4; ++tt) {
                                int q0 = (int) sycl_rint_i(a[tt * 2] * id);
                                int q1 = (int) sycl_rint_i(a[tt * 2 + 1] * id);
                                if (q0 < -8) {
                                    q0 = -8;
                                }
                                if (q0 > 7) {
                                    q0 = 7;
                                }
                                if (q1 < -8) {
                                    q1 = -8;
                                }
                                if (q1 > 7) {
                                    q1 = 7;
                                }
                                const uint8_t u0 = (uint8_t) (int8_t) q0;
                                const uint8_t u1 = (uint8_t) (int8_t) q1;
                                dst[sgid * 64 + lane * 4 + tt] =
                                    (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                            }
                        }
                    }
                });
            return true;
        }
        // g == 128: one Hadamard block per group, so give each sub-group its own group
        // instead of launching 8 sub-groups where only the first has work.
        if (nblk == 1 && !StoreS8 && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44G1SG", 1)) {
            const size_t ntask = (size_t) M * (size_t) ng;
            const size_t nwg = (ntask + 7) / 8;
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nwg * 128u), sycl::range<1>(128)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const int lid = (int) it.get_local_id(0);
                    const int lane = lid & 15;
                    const size_t task = (size_t) it.get_group(0) * 8u + (size_t) (lid >> 4);
                    if (task >= ntask) {
                        return;
                    }
                    const int m = (int) (task / (size_t) ng);
                    const int ig = (int) (task - (size_t) m * (size_t) ng);
                    const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                    uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                    const int base = ig * (int) g + lane * 8;
                    const sycl::sub_group sg = it.get_sub_group();
                    constexpr int EL = 8;
                    float a[EL];
                    if constexpr (FromF32Rte) {
                        const sycl::vec<float, 8> vf =
                            *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                        const sycl::vec<sycl::half, 8> vh =
                            vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    } else {
                        const sycl::vec<sycl::half, 8> vh =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    }
#pragma unroll
                    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = a[j + k];
                                const float ay = a[j + k + h];
                                a[j + k] = ax + ay;
                                a[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                            a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                        }
                    }
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        a[i] = a[i] * inv;
                        amax = sycl::fmax(amax, sycl::fabs(a[i]));
                    }
                    const float gmax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
                    const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                    if (lane == 0) {
                        sa[task] = sc;
                    }
                    const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int q0 = (int) sycl_rint_i(a[t * 2] * id);
                        int q1 = (int) sycl_rint_i(a[t * 2 + 1] * id);
                        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
                        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
                        const uint8_t u0 = (uint8_t) (int8_t) q0;
                        const uint8_t u1 = (uint8_t) (int8_t) q1;
                        dst[lane * 4 + t] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                    }
                });
            return true;
        }
        const int hilpm = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILPM", 0);
        const int mgrp = (hilpm && !StoreS8 && nblk == 8 && (M % 2) == 0) ? 2 : 1;
        const int64_t Md = M / mgrp;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) Md * (size_t) ng * 128u), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int gid = (int) it.get_group(0);
                const int mb = gid / (int) ng;
                const int ig = gid - mb * (int) ng;
                const int lid = (int) it.get_local_id(0);
                const int lane = lid & 15;
                const int sgid = lid >> 4;
                const int k0 = ig * (int) g;
                const sycl::sub_group sg = it.get_sub_group();
                constexpr int EL = 8;
                for (int mm = 0; mm < mgrp; ++mm) {
                    const int m = mb * mgrp + mm;
                    const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                    uint8_t * dst;
                    if constexpr (StoreS8) {
                        dst = A4 + (size_t) m * (size_t) K + (size_t) (ig * (int) g);
                    } else {
                        dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                    }
                float a[EL];
#pragma unroll
                for (int i = 0; i < EL; ++i) {
                    a[i] = 0.0f;
                }
                float amax = 0.0f;
                if (sgid < nblk) {
                    const int base = k0 + sgid * 128 + lane * EL;
                    if constexpr (FromF32Rte) {
                        const sycl::vec<float, 8> vf =
                            *reinterpret_cast<const sycl::vec<float, 8> *>(row + base);
                        const sycl::vec<sycl::half, 8> vh =
                            vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    } else {
                        const sycl::vec<sycl::half, 8> vh =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
#pragma unroll
                        for (int i = 0; i < EL; ++i) {
                            a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                        }
                    }
#pragma unroll
                    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = a[j + k];
                                const float ay = a[j + k + h];
                                a[j + k] = ax + ay;
                                a[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                            a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        a[i] = a[i] * inv;
                        amax = sycl::fmax(amax, sycl::fabs(a[i]));
                    }
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                if (sgid < nblk) {
                    if constexpr (StoreS8) {
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            int q = (int) sycl_rint_i(a[t] * id);
                            if (q < -8) {
                                q = -8;
                            }
                            if (q > 7) {
                                q = 7;
                            }
                            dst[sgid * 128 + lane * 8 + t] = (uint8_t) (int8_t) q;
                        }
                    } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int q0 = (int) sycl_rint_i(a[t * 2] * id);
                        int q1 = (int) sycl_rint_i(a[t * 2 + 1] * id);
                        if (q0 < -8) {
                            q0 = -8;
                        }
                        if (q0 > 7) {
                            q0 = 7;
                        }
                        if (q1 < -8) {
                            q1 = -8;
                        }
                        if (q1 > 7) {
                            q1 = 7;
                        }
                        const uint8_t u0 = (uint8_t) (int8_t) q0;
                        const uint8_t u1 = (uint8_t) (int8_t) q1;
                        dst[sgid * 64 + lane * 4 + t] =
                            (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                    }
                    }
                }
                }
            });
        return true;
    }
    if (hs == 128 && hilp > 0) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * 128u), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = (int) it.get_group(0);
                const int m = gid / (int) ng;
                const int ig = gid - m * (int) ng;
                const int lid = (int) it.get_local_id(0);
                const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                const int k0 = ig * (int) g;
                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
                float * s = *slm;
                const sycl::sub_group sg = it.get_sub_group();
                const int sl = (int) sg.get_local_linear_id();
                float amax = 0.0f;
                for (int bs = 0; bs < nblk; bs += hilp) {
                    float v0 = 0.0f;
                    float v1 = 0.0f;
                    float v2 = 0.0f;
                    float v3 = 0.0f;
                    const int nlive = nblk - bs;
                    if constexpr (FromF32Rte) {
                        float f0 = row[k0 + bs * 128 + lid];
                        float f1 = 0.0f;
                        float f2 = 0.0f;
                        float f3 = 0.0f;
                        if (nlive > 1) {
                            f1 = row[k0 + (bs + 1) * 128 + lid];
                        }
                        if (hilp > 2 && nlive > 2) {
                            f2 = row[k0 + (bs + 2) * 128 + lid];
                        }
                        if (hilp > 2 && nlive > 3) {
                            f3 = row[k0 + (bs + 3) * 128 + lid];
                        }
                        const sycl::vec<float, 4> vf(f0, f1, f2, f3);
                        const sycl::vec<sycl::half, 4> vh =
                            vf.template convert<sycl::half, sycl::rounding_mode::rte>();
                        v0 = sycl::fmin(sycl::fmax((float) vh[0], -65504.0f), 65504.0f);
                        v1 = sycl::fmin(sycl::fmax((float) vh[1], -65504.0f), 65504.0f);
                        v2 = sycl::fmin(sycl::fmax((float) vh[2], -65504.0f), 65504.0f);
                        v3 = sycl::fmin(sycl::fmax((float) vh[3], -65504.0f), 65504.0f);
                    } else {
                        v0 = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + bs * 128 + lid);
                        if (nlive > 1) {
                            v1 = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + (bs + 1) * 128 + lid);
                        }
                        if (hilp > 2 && nlive > 2) {
                            v2 = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + (bs + 2) * 128 + lid);
                        }
                        if (hilp > 2 && nlive > 3) {
                            v3 = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + (bs + 3) * 128 + lid);
                        }
                    }
                    for (int h = 1; h < 32; h *= 2) {
                        const float y0 = sycl::permute_group_by_xor(sg, v0, (uint32_t) h);
                        const float y1 = sycl::permute_group_by_xor(sg, v1, (uint32_t) h);
                        v0 = ((sl & h) == 0) ? (v0 + y0) : (y0 - v0);
                        v1 = ((sl & h) == 0) ? (v1 + y1) : (y1 - v1);
                        if (hilp > 2) {
                            const float y2 = sycl::permute_group_by_xor(sg, v2, (uint32_t) h);
                            const float y3 = sycl::permute_group_by_xor(sg, v3, (uint32_t) h);
                            v2 = ((sl & h) == 0) ? (v2 + y2) : (y2 - v2);
                            v3 = ((sl & h) == 0) ? (v3 + y3) : (y3 - v3);
                        }
                    }
                    s[bs * 128 + lid] = v0;
                    if (nlive > 1) {
                        s[(bs + 1) * 128 + lid] = v1;
                    }
                    if (hilp > 2 && nlive > 2) {
                        s[(bs + 2) * 128 + lid] = v2;
                    }
                    if (hilp > 2 && nlive > 3) {
                        s[(bs + 3) * 128 + lid] = v3;
                    }
                    for (int h = 32; h < 128; h *= 2) {
                        it.barrier(sycl::access::fence_space::local_space);
                        const int pair = lid ^ h;
                        const float m0 = s[bs * 128 + lid];
                        const float u0 = s[bs * 128 + pair];
                        float m1 = 0.0f;
                        float u1 = 0.0f;
                        float m2 = 0.0f;
                        float u2 = 0.0f;
                        float m3 = 0.0f;
                        float u3 = 0.0f;
                        if (nlive > 1) {
                            m1 = s[(bs + 1) * 128 + lid];
                            u1 = s[(bs + 1) * 128 + pair];
                        }
                        if (hilp > 2 && nlive > 2) {
                            m2 = s[(bs + 2) * 128 + lid];
                            u2 = s[(bs + 2) * 128 + pair];
                        }
                        if (hilp > 2 && nlive > 3) {
                            m3 = s[(bs + 3) * 128 + lid];
                            u3 = s[(bs + 3) * 128 + pair];
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                        s[bs * 128 + lid] = ((lid & h) == 0) ? (m0 + u0) : (u0 - m0);
                        if (nlive > 1) {
                            s[(bs + 1) * 128 + lid] = ((lid & h) == 0) ? (m1 + u1) : (u1 - m1);
                        }
                        if (hilp > 2 && nlive > 2) {
                            s[(bs + 2) * 128 + lid] = ((lid & h) == 0) ? (m2 + u2) : (u2 - m2);
                        }
                        if (hilp > 2 && nlive > 3) {
                            s[(bs + 3) * 128 + lid] = ((lid & h) == 0) ? (m3 + u3) : (u3 - m3);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    {
                        const float t0 = s[bs * 128 + lid] * inv;
                        s[bs * 128 + lid] = t0;
                        amax = sycl::fmax(amax, sycl::fabs(t0));
                    }
                    if (nlive > 1) {
                        const float t1 = s[(bs + 1) * 128 + lid] * inv;
                        s[(bs + 1) * 128 + lid] = t1;
                        amax = sycl::fmax(amax, sycl::fabs(t1));
                    }
                    if (hilp > 2 && nlive > 2) {
                        const float t2 = s[(bs + 2) * 128 + lid] * inv;
                        s[(bs + 2) * 128 + lid] = t2;
                        amax = sycl::fmax(amax, sycl::fabs(t2));
                    }
                    if (hilp > 2 && nlive > 3) {
                        const float t3 = s[(bs + 3) * 128 + lid] * inv;
                        s[(bs + 3) * 128 + lid] = t3;
                        amax = sycl::fmax(amax, sycl::fabs(t3));
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                const int np = (int) g / 2;
                for (int p = lid; p < np; p += 128) {
                    int q0 = (int) sycl_rint_i(s[p * 2] * id);
                    int q1 = (int) sycl_rint_i(s[p * 2 + 1] * id);
                    if (q0 < -8) {
                        q0 = -8;
                    }
                    if (q0 > 7) {
                        q0 = 7;
                    }
                    if (q1 < -8) {
                        q1 = -8;
                    }
                    if (q1 > 7) {
                        q1 = 7;
                    }
                    const uint8_t u0 = (uint8_t) (int8_t) q0;
                    const uint8_t u1 = (uint8_t) (int8_t) q1;
                    dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                }
            });
        return true;
    }
    int wide = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWIDE", 0);
    if (wide != 2 && wide != 4 && wide != 8) {
        wide = 0;
    }
    if (hs == 128 && wide > 0) {
        const int npar = (nblk < wide) ? nblk : wide;
        const int wgn = npar * 128;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * (size_t) wgn), sycl::range<1>((size_t) wgn)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = (int) it.get_group(0);
                const int m = gid / (int) ng;
                const int ig = gid - m * (int) ng;
                const int lid = (int) it.get_local_id(0);
                const int t = lid % 128;
                const int pb = lid / 128;
                const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                const int k0 = ig * (int) g;
                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
                float * s = *slm;
                const sycl::sub_group sg = it.get_sub_group();
                const int sl = (int) sg.get_local_linear_id();
                for (int bs = 0; bs < nblk; bs += npar) {
                    const int b = bs + pb;
                    const int live = b < nblk ? 1 : 0;
                    float v = 0.0f;
                    if (live) {
                        if constexpr (FromF32Rte) {
                            v = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + b * 128 + t);
                        } else {
                            v = (float) row[k0 + b * 128 + t];
                        }
                    }
                    for (int h = 1; h < 32; h *= 2) {
                        const float you = sycl::permute_group_by_xor(sg, v, (uint32_t) h);
                        v = ((sl & h) == 0) ? (v + you) : (you - v);
                    }
                    if (live) {
                        s[b * 128 + t] = v;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int h = 32; h < 128; h *= 2) {
                        it.barrier(sycl::access::fence_space::local_space);
                        const int pair = t ^ h;
                        float me = 0.0f;
                        float you = 0.0f;
                        if (live) {
                            me = s[b * 128 + t];
                            you = s[b * 128 + pair];
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                        if (live && (t & h) == 0) {
                            s[b * 128 + t] = me + you;
                            s[b * 128 + pair] = me - you;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (live) {
                        s[b * 128 + t] = s[b * 128 + t] * inv;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                float amax = 0.0f;
                for (int i = lid; i < (int) g; i += wgn) {
                    amax = sycl::fmax(amax, sycl::fabs(s[i]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                const int np = (int) g / 2;
                for (int p = lid; p < np; p += wgn) {
                    int q0 = (int) sycl_rint_i(s[p * 2] * id);
                    int q1 = (int) sycl_rint_i(s[p * 2 + 1] * id);
                    if (q0 < -8) {
                        q0 = -8;
                    }
                    if (q0 > 7) {
                        q0 = 7;
                    }
                    if (q1 < -8) {
                        q1 = -8;
                    }
                    if (q1 > 7) {
                        q1 = 7;
                    }
                    const uint8_t u0 = (uint8_t) (int8_t) q0;
                    const uint8_t u1 = (uint8_t) (int8_t) q1;
                    dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                }
            });
        return true;
    }
    if (hs == 128) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * 128u), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = (int) it.get_group(0);
                const int m = gid / (int) ng;
                const int ig = gid - m * (int) ng;
                const int lid = (int) it.get_local_id(0);
                const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
                uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
                const int k0 = ig * (int) g;
                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
                float * s = *slm;
                const sycl::sub_group sg = it.get_sub_group();
                const int sl = (int) sg.get_local_linear_id();
                for (int b = 0; b < nblk; b++) {
                    float v = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + b * 128 + lid);
                    for (int h = 1; h < 32; h *= 2) {
                        const float you = sycl::permute_group_by_xor(sg, v, (uint32_t) h);
                        v = ((sl & h) == 0) ? (v + you) : (you - v);
                    }
                    s[b * 128 + lid] = v;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int h = 32; h < 128; h *= 2) {
                        it.barrier(sycl::access::fence_space::local_space);
                        const int pair = lid ^ h;
                        const float me = s[b * 128 + lid];
                        const float you = s[b * 128 + pair];
                        it.barrier(sycl::access::fence_space::local_space);
                        if ((lid & h) == 0) {
                            s[b * 128 + lid] = me + you;
                            s[b * 128 + pair] = me - you;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    s[b * 128 + lid] = s[b * 128 + lid] * inv;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float amax = 0.0f;
                for (int b = 0; b < nblk; b++) {
                    amax = sycl::fmax(amax, sycl::fabs(s[b * 128 + lid]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                const int np = (int) g / 2;
                for (int p = lid; p < np; p += 128) {
                    int q0 = (int) sycl_rint_i(s[p * 2] * id);
                    int q1 = (int) sycl_rint_i(s[p * 2 + 1] * id);
                    if (q0 < -8) {
                        q0 = -8;
                    }
                    if (q0 > 7) {
                        q0 = 7;
                    }
                    if (q1 < -8) {
                        q1 = -8;
                    }
                    if (q1 > 7) {
                        q1 = 7;
                    }
                    const uint8_t u0 = (uint8_t) (int8_t) q0;
                    const uint8_t u1 = (uint8_t) (int8_t) q1;
                    dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                }
            });
        return true;
    }
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * (size_t) hs), sycl::range<1>((size_t) hs)),
        [=](sycl::nd_item<1> it) {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const Src * row = xh + (size_t) m * (size_t) lda + (size_t) koff;
            uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
            const int k0 = ig * (int) g;
            auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
            float * s = *slm;
            for (int b = 0; b < nblk; b++) {
                s[b * wgi + lid] = xe2mx_apack_ld<Src, FromF32Rte>(row, k0 + b * wgi + lid);
                it.barrier(sycl::access::fence_space::local_space);
                for (int h = 1; h < wgi; h *= 2) {
                    it.barrier(sycl::access::fence_space::local_space);
                    const int pair = lid ^ h;
                    const float me = s[b * wgi + lid];
                    const float you = s[b * wgi + pair];
                    it.barrier(sycl::access::fence_space::local_space);
                    if ((lid & h) == 0) {
                        s[b * wgi + lid] = me + you;
                        s[b * wgi + pair] = me - you;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                s[b * wgi + lid] = s[b * wgi + lid] * inv;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float amax = 0.0f;
            for (int b = 0; b < nblk; b++) {
                amax = sycl::fmax(amax, sycl::fabs(s[b * wgi + lid]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
            if (lid == 0) {
                sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
            }
            const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
            const int np = (int) g / 2;
            for (int p = lid; p < np; p += wgi) {
                int q0 = (int) sycl_rint_i(s[p * 2] * id);
                int q1 = (int) sycl_rint_i(s[p * 2 + 1] * id);
                if (q0 < -8) {
                    q0 = -8;
                }
                if (q0 > 7) {
                    q0 = 7;
                }
                if (q1 < -8) {
                    q1 = -8;
                }
                if (q1 > 7) {
                    q1 = 7;
                }
                const uint8_t u0 = (uint8_t) (int8_t) q0;
                const uint8_t u1 = (uint8_t) (int8_t) q1;
                dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
            }
        });
    return true;
}

bool xe2mx_quant_a_s4_nibble_had(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<sycl::half, false>(stream, xh, A4, sa, M, K, g, 0, 0);
}

bool xe2mx_quant_a_s4_nibble_had_off(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                                     int64_t M, int64_t lda, int64_t koff, int64_t klen, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<sycl::half, false>(stream, xh, A4, sa, M, klen, g, lda, koff);
}

bool xe2mx_quant_a_s4_nibble_had_f32(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                     int64_t M, int64_t K, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<float, false>(stream, x, A4, sa, M, K, g, 0, 0);
}

bool xe2mx_quant_a_s4_nibble_had_f32_rte(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                         int64_t M, int64_t K, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<float, true>(stream, x, A4, sa, M, K, g, 0, 0);
}

bool xe2mx_quant_a_s4_nibble_had_f32_rte_off(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                             int64_t M, int64_t lda, int64_t koff, int64_t klen, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<float, true>(stream, x, A4, sa, M, klen, g, lda, koff);
}

bool xe2mx_quant_a_s8_had(queue_ptr stream, const sycl::half * xh, uint8_t * A8, float * sa,
                          int64_t M, int64_t K, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<sycl::half, false, true>(stream, xh, A8, sa, M, K, g, 0, 0);
}

bool xe2mx_quant_a_s8_had_f32_rte(queue_ptr stream, const float * x, uint8_t * A8, float * sa,
                                  int64_t M, int64_t K, int64_t g) {
    return xe2mx_quant_a_s4_nibble_had_impl<float, true, true>(stream, x, A8, sa, M, K, g, 0, 0);
}

bool xe2mx_quant_a_s4_nibble_f32(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g, float * za) {
    if (!stream || !x || !A4 || !sa || M < 1 || K < 2 || g < 2) {
        return false;
    }
    if ((K % g) != 0 || (g % 2) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    size_t wg = 128;
    if (g <= 64) {
        wg = (size_t) g;
    } else if (g >= 256 && (g % 256) == 0) {
        wg = 256;
    }
    if ((g % (int64_t) wg) != 0) {
        wg = 128;
    }
    if ((g % (int64_t) wg) != 0) {
        return false;
    }
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const int64_t kh = K / 2;
    const int wgi = (int) wg;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const float * row = x + (size_t) m * (size_t) K + (size_t) (ig * (int) g);
            uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
            float amax = 0.0f;
            for (int k = lid; k < (int) g; k += wgi) {
                amax = sycl::fmax(amax, sycl::fabs(row[k]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float s = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
            if (lid == 0) {
                sa[(size_t) m * (size_t) ng + (size_t) ig] = s;
            }
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            const int np = (int) g / 2;
            float qs = 0.0f;
            for (int p = lid; p < np; p += wgi) {
                int q0 = (int) sycl_rint_i(row[p * 2] * id);
                int q1 = (int) sycl_rint_i(row[p * 2 + 1] * id);
                if (q0 < -8) {
                    q0 = -8;
                }
                if (q0 > 7) {
                    q0 = 7;
                }
                if (q1 < -8) {
                    q1 = -8;
                }
                if (q1 > 7) {
                    q1 = 7;
                }
                qs += (float) (q0 + q1);
                const uint8_t u0 = (uint8_t) (int8_t) q0;
                const uint8_t u1 = (uint8_t) (int8_t) q1;
                dst[p] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
            }
            if (za) {
                qs = sycl::reduce_over_group(it.get_group(), qs, sycl::plus<float>());
                if (lid == 0) {
                    za[(size_t) m * (size_t) ng + (size_t) ig] = s * qs;
                }
            }
        });
    return true;
}

bool xe2mx_rms_mul_hilp_pack(queue_ptr stream, const float * x, const float * mul, float * y,
                             uint8_t * A4, float * sa, int64_t M, int64_t K, int64_t g, float eps) {
    if (!stream || !x || !mul || !y || !A4 || !sa || M < 32 || K < 128 || g < 128) {
        return false;
    }
    if ((K % g) != 0 || (g % 128) != 0 || g > 1024) {
        return false;
    }
    const int64_t ng = K / g;
    const int nblk = (int) (g / 128);
    if (nblk < 1 || nblk > 8) {
        return false;
    }
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const float inv = 0.08838834764831845f;
    const int64_t kh = K / 2;
    const float kinv = 1.0f / (float) K;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * 256u), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int lane = lid & 15;
            const int sgid = lid >> 4;
            const float * row = x + (size_t) m * (size_t) K;
            float * yrow = y + (size_t) m * (size_t) K;
            uint8_t * adst = A4 + (size_t) m * (size_t) kh;
            const sycl::sub_group sg = it.get_sub_group();
            float ss = 0.0f;
            for (int col = lid; col < (int) K; col += 256) {
                const float v = row[col];
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float mean = ss * kinv;
            const float scale = sycl::rsqrt(mean + eps);
            for (int col = lid; col < (int) K; col += 256) {
                yrow[col] = scale * row[col] * mul[col];
            }
            it.barrier(sycl::access::fence_space::global_and_local);
            constexpr int EL = 8;
            for (int ig = 0; ig < (int) ng; ig++) {
                const int k0 = ig * (int) g;
                float a[EL];
#pragma unroll
                for (int i = 0; i < EL; ++i) {
                    a[i] = 0.0f;
                }
                float amax = 0.0f;
                if (sgid < nblk) {
                    const int base = k0 + sgid * 128 + lane * EL;
                    const sycl::vec<float, 8> vf =
                        *reinterpret_cast<const sycl::vec<float, 8> *>(yrow + base);
                    const sycl::vec<sycl::half, 8> vh =
                        vf.template convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                    }
#pragma unroll
                    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = a[j + k];
                                const float ay = a[j + k + h];
                                a[j + k] = ax + ay;
                                a[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                            a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        a[i] = a[i] * inv;
                        amax = sycl::fmax(amax, sycl::fabs(a[i]));
                    }
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                if (sgid < nblk) {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int q0 = (int) sycl_rint_i(a[t * 2] * id);
                        int q1 = (int) sycl_rint_i(a[t * 2 + 1] * id);
                        if (q0 < -8) {
                            q0 = -8;
                        }
                        if (q0 > 7) {
                            q0 = 7;
                        }
                        if (q1 < -8) {
                            q1 = -8;
                        }
                        if (q1 > 7) {
                            q1 = 7;
                        }
                        const uint8_t u0 = (uint8_t) (int8_t) q0;
                        const uint8_t u1 = (uint8_t) (int8_t) q1;
                        adst[(size_t) (ig * (int) g / 2) + sgid * 64 + lane * 4 + t] =
                            (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                    }
                }
            }
        });
    return true;
}

bool xe2mx_add_rms_mul_hilp_pack(queue_ptr stream, const float * a, const float * b, float * add_dst,
                                 const float * mul, float * y, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g, float eps, int drop_add) {
    if (!stream || !a || !b || !mul || !y || !A4 || !sa || M < 32 || K < 128 || g < 128) {
        return false;
    }
    if ((K % g) != 0 || (g % 128) != 0 || g > 1024) {
        return false;
    }
    const int64_t ng = K / g;
    const int nblk = (int) (g / 128);
    if (nblk < 1 || nblk > 8) {
        return false;
    }
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const float inv = 0.08838834764831845f;
    const int64_t kh = K / 2;
    const float kinv = 1.0f / (float) K;
    const int drop = drop_add ? 1 : 0;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * 256u), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int lane = lid & 15;
            const int sgid = lid >> 4;
            const float * ar = a + (size_t) m * (size_t) K;
            const float * br = b + (size_t) m * (size_t) K;
            float * sr = drop ? nullptr : add_dst + (size_t) m * (size_t) K;
            float * yrow = y + (size_t) m * (size_t) K;
            uint8_t * adst = A4 + (size_t) m * (size_t) kh;
            const sycl::sub_group sg = it.get_sub_group();
            float ss = 0.0f;
            for (int col = lid; col < (int) K; col += 256) {
                const float sv = ar[col] + br[col];
                ss += sv * sv;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float scale = sycl::rsqrt(ss * kinv + eps);
            constexpr int EL = 8;
            for (int ig = 0; ig < (int) ng; ig++) {
                const int k0 = ig * (int) g;
                float av[EL];
#pragma unroll
                for (int i = 0; i < EL; ++i) {
                    av[i] = 0.0f;
                }
                float amax = 0.0f;
                if (sgid < nblk) {
                    const int base = k0 + sgid * 128 + lane * EL;
                    float yv[EL];
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        const float sv = ar[base + i] + br[base + i];
                        if (sr) {
                            sr[base + i] = sv;
                        }
                        yv[i] = scale * sv * mul[base + i];
                        yrow[base + i] = yv[i];
                    }
                    const sycl::vec<float, 8> vf(yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]);
                    const sycl::vec<sycl::half, 8> vh = vf.convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        av[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
                    }
#pragma unroll
                    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                            for (int k = 0; k < h; ++k) {
                                const float ax = av[j + k];
                                const float ay = av[j + k + h];
                                av[j + k] = ax + ay;
                                av[j + k + h] = ax - ay;
                            }
                        }
                    }
#pragma unroll
                    for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                        for (int j = 0; j < EL; ++j) {
                            const float a2 = sycl::permute_group_by_xor(sg, av[j], (uint32_t) xm);
                            av[j] = (lane & xm) == 0 ? av[j] + a2 : a2 - av[j];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        av[i] = av[i] * inv;
                        amax = sycl::fmax(amax, sycl::fabs(av[i]));
                    }
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                if (sgid < nblk) {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int q0 = (int) sycl_rint_i(av[t * 2] * id);
                        int q1 = (int) sycl_rint_i(av[t * 2 + 1] * id);
                        if (q0 < -8) {
                            q0 = -8;
                        }
                        if (q0 > 7) {
                            q0 = 7;
                        }
                        if (q1 < -8) {
                            q1 = -8;
                        }
                        if (q1 > 7) {
                            q1 = 7;
                        }
                        const uint8_t u0 = (uint8_t) (int8_t) q0;
                        const uint8_t u1 = (uint8_t) (int8_t) q1;
                        adst[(size_t) (ig * (int) g / 2) + sgid * 64 + lane * 4 + t] =
                            (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
                    }
                }
            }
        });
    return true;
}

bool xe2mx_silu_up_hilp_pack(queue_ptr stream, const float * gate, const float * up, float * y,
                             uint8_t * A4, float * sa, int64_t M, int64_t K, int64_t g) {
    if (!stream || !gate || !up || !y || !A4 || !sa || M < 32 || g != 1024 || (K % g) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const float inv = 0.08838834764831845f;
    const int64_t kh = K / 2;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * 128u), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const int lane = lid & 15;
            const int sgid = lid >> 4;
            const int k0 = ig * (int) g;
            const float * grow = gate + (size_t) m * (size_t) K;
            const float * urow = up + (size_t) m * (size_t) K;
            float * yrow = y + (size_t) m * (size_t) K;
            uint8_t * dst = A4 + (size_t) m * (size_t) kh + (size_t) (ig * (int) g / 2);
            const sycl::sub_group sg = it.get_sub_group();
            constexpr int EL = 8;
            const int base = k0 + sgid * 128 + lane * EL;
            float yv[EL];
#pragma unroll
            for (int i = 0; i < EL; ++i) {
                const float gv = grow[base + i];
                yv[i] = (gv / (1.0f + sycl::exp(-gv))) * urow[base + i];
                yrow[base + i] = yv[i];
            }
            const sycl::vec<float, 8> vf(yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7]);
            const sycl::vec<sycl::half, 8> vh = vf.convert<sycl::half, sycl::rounding_mode::rte>();
            float a[EL];
#pragma unroll
            for (int i = 0; i < EL; ++i) {
                a[i] = sycl::fmin(sycl::fmax((float) vh[i], -65504.0f), 65504.0f);
            }
#pragma unroll
            for (int h = 1; h < EL; h *= 2) {
#pragma unroll
                for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
                    for (int k = 0; k < h; ++k) {
                        const float ax = a[j + k];
                        const float ay = a[j + k + h];
                        a[j + k] = ax + ay;
                        a[j + k + h] = ax - ay;
                    }
                }
            }
#pragma unroll
            for (int h = EL, xm = 1; h < 128; h *= 2, xm *= 2) {
#pragma unroll
                for (int j = 0; j < EL; ++j) {
                    const float a2 = sycl::permute_group_by_xor(sg, a[j], (uint32_t) xm);
                    a[j] = (lane & xm) == 0 ? a[j] + a2 : a2 - a[j];
                }
            }
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < EL; ++i) {
                a[i] = a[i] * inv;
                amax = sycl::fmax(amax, sycl::fabs(a[i]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
            if (lid == 0) {
                sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
            }
            const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                int q0 = (int) sycl_rint_i(a[t * 2] * id);
                int q1 = (int) sycl_rint_i(a[t * 2 + 1] * id);
                if (q0 < -8) {
                    q0 = -8;
                }
                if (q0 > 7) {
                    q0 = 7;
                }
                if (q1 < -8) {
                    q1 = -8;
                }
                if (q1 > 7) {
                    q1 = 7;
                }
                const uint8_t u0 = (uint8_t) (int8_t) q0;
                const uint8_t u1 = (uint8_t) (int8_t) q1;
                dst[sgid * 64 + lane * 4 + t] = (uint8_t) ((u0 & 15u) | (uint8_t) ((u1 & 15u) << 4));
            }
        });
    return true;
}

void xe2mx_s44_quant_w_gptq(queue_ptr stream, float * src, int8_t * w, float * sc, float * bsum,
                            int64_t K, int64_t N, int64_t g) {
    const int64_t ng = K / g;
    const size_t wg = 128;
    const int wcen = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44WCEN", 0);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float * col = src + (size_t) n * (size_t) K;
            int8_t * dst = w + (size_t) n * (size_t) K;
            for (int ig = 0; ig < (int) ng; ig++) {
                const int k0 = ig * (int) g;
                const int k1 = k0 + (int) g;
                float sm = 0.0f;
                if (wcen || bsum) {
                    for (int k = k0 + lid; k < k1; k += (int) wg) {
                        sm += col[k];
                    }
                    sm = sycl::reduce_over_group(it.get_group(), sm, sycl::plus<float>());
                }
                const float mean = wcen ? (sm / (float) g) : 0.0f;
                float amax = 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    amax = sycl::fmax(amax, sycl::fabs(col[k] - mean));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float s = gmax > 0.0f ? gmax / 7.0f : 0.0f;
                if (lid == 0) {
                    sc[(size_t) ig * (size_t) N + (size_t) n] = s;
                    if (bsum) {
                        bsum[(size_t) ig * (size_t) N + (size_t) n] = wcen ? mean : sm;
                    }
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    int q = (int) sycl_rint_i((col[k] - mean) * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    dst[k] = (int8_t) q;
                    if (ig + 1 < (int) ng) {
                        col[k + (int) g] += (col[k] - mean) - (float) q * s;
                    }
                }
                if (ig + 1 < (int) ng) {
                    it.barrier(sycl::access::fence_space::global_and_local);
                }
            }
        });
}
