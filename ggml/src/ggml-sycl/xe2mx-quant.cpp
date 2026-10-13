#include "common.hpp"
#include "ggml-sycl.h"

#include <cstdio>
#include <cstring>
#include <vector>

static void xe2mx_host_f32(const void * src, int src_type, int64_t n, std::vector<float> & out) {
    out.resize((size_t) n);
    if (src_type == GGML_TYPE_F32) {
        memcpy(out.data(), src, (size_t) n * sizeof(float));
        return;
    }
    if (src_type == GGML_TYPE_F16) {
        const sycl::half * h = (const sycl::half *) src;
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (float) h[i];
        }
        return;
    }
    const uint16_t * h = (const uint16_t *) src;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t u = ((uint32_t) h[i]) << 16;
        memcpy(&out[i], &u, sizeof(float));
    }
}

static void xe2mx_quant_submit(sycl::queue & q, const float * src_h, void * dst_h,
                               float * d_src, float * d_d, int * d_q, const float * d_qw,
                               int64_t nrows, int64_t n_per_row) {
    const int64_t nb = n_per_row / QK_XE2MX4;
    const size_t src_bytes = (size_t) nrows * (size_t) n_per_row * sizeof(float);
    const size_t d_bytes = (size_t) nrows * (size_t) nb * sizeof(float);
    const size_t q_bytes = (size_t) nrows * (size_t) n_per_row * sizeof(int);

    q.memcpy(d_src, src_h, src_bytes);
    q.parallel_for(sycl::range<1>((size_t) nrows), [=](sycl::id<1> gid) {
        const int64_t row = (int64_t) gid[0];
        const int64_t row_base = row * n_per_row;
        float sigma2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            const float v = d_src[row_base + j];
            sigma2 += v * v;
        }
        sigma2 /= (float) n_per_row;
        for (int64_t ib = 0; ib < nb; ++ib) {
            float x[QK_XE2MX4];
            float w[QK_XE2MX4];
            float vmax = 0.0f;
            float amax = 0.0f;
            for (int j = 0; j < QK_XE2MX4; ++j) {
                const float v = d_src[row_base + ib * QK_XE2MX4 + j];
                x[j] = v;
                const float av = v < 0.0f ? -v : v;
                if (av > amax) {
                    amax = av;
                    vmax = v;
                }
                if (d_qw) {
                    w[j] = d_qw[ib * QK_XE2MX4 + j] * sycl::sqrt(sigma2 + v * v);
                } else {
                    w[j] = v * v;
                }
            }
            int best_q[QK_XE2MX4];
            float d = 0.0f;
            for (int j = 0; j < QK_XE2MX4; ++j) {
                best_q[j] = 8;
            }
            if (amax > 1e-15f) {
                float iscale = -8.0f / vmax;
                float sumlx = 0.0f;
                float suml2 = 0.0f;
                for (int j = 0; j < QK_XE2MX4; ++j) {
                    int l = (int) sycl_rint_i(iscale * x[j]);
                    if (l < -8) l = -8;
                    if (l >  7) l = 7;
                    best_q[j] = l + 8;
                    sumlx += w[j] * x[j] * (float) l;
                    suml2 += w[j] * (float) l * (float) l;
                }
                float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
                float best = scale * sumlx;
                for (int is = -9; is <= 9; ++is) {
                    if (is == 0) {
                        continue;
                    }
                    iscale = -(8.0f + 0.1f * (float) is) / vmax;
                    sumlx = 0.0f;
                    suml2 = 0.0f;
                    int cand[QK_XE2MX4];
                    for (int j = 0; j < QK_XE2MX4; ++j) {
                        int l = (int) sycl_rint_i(iscale * x[j]);
                        if (l < -8) l = -8;
                        if (l >  7) l = 7;
                        cand[j] = l + 8;
                        sumlx += w[j] * x[j] * (float) l;
                        suml2 += w[j] * (float) l * (float) l;
                    }
                    if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
                        for (int j = 0; j < QK_XE2MX4; ++j) {
                            best_q[j] = cand[j];
                        }
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                    }
                }
                d = scale;
            }
            d_d[row * nb + ib] = d;
            for (int j = 0; j < QK_XE2MX4; ++j) {
                d_q[row_base + ib * QK_XE2MX4 + j] = best_q[j];
            }
        }
    });
    std::vector<float> hd((size_t) nrows * (size_t) nb);
    std::vector<int> hq((size_t) nrows * (size_t) n_per_row);
    q.memcpy(hd.data(), d_d, d_bytes);
    q.memcpy(hq.data(), d_q, q_bytes);
    q.wait();
    static int dlog = 0;
    if (!dlog && nrows > 0) {
        dlog = 1;
        GGML_SYCL_XE2_LOG( "xe2mx first d=%g amax_proxy=%g\n", hd[0], hd[0] * 8.0f);
    }
    block_xe2mx4 * dst = (block_xe2mx4 *) dst_h;
    for (int64_t i = 0; i < nrows * nb; ++i) {
        dst[i].d = sycl::half(hd[i]);
        const int * qq = hq.data() + i * QK_XE2MX4;
        for (int j = 0; j < QK_XE2MX4 / 2; ++j) {
            const int lo = qq[2 * j];
            const int hi = qq[2 * j + 1];
            dst[i].qs[j] = (uint8_t) ((lo & 15) | ((hi & 15) << 4));
        }
    }
}

size_t ggml_sycl_quantize_xe2mx4(const void * src, enum ggml_type src_type,
                                   void * dst, int64_t nrows, int64_t n_per_row,
                                   const float * imatrix) {
    if (!src || !dst || nrows <= 0 || n_per_row <= 0) {
        return 0;
    }
    if (n_per_row % QK_XE2MX4 != 0) {
        return 0;
    }
    if (src_type != GGML_TYPE_F32 && src_type != GGML_TYPE_F16 && src_type != GGML_TYPE_BF16) {
        return 0;
    }

    const int nd = ggml_sycl_info().device_count;
    if (nd <= 0) {
        return 0;
    }

    static int logged = 0;
    if (!logged) {
        logged = 1;
        GGML_SYCL_XE2_LOG( "xe2mx quantize: %d SYCL device(s), row split, scale search%s\n",
                nd, imatrix ? " + imatrix" : "");
        ggml_backend_sycl_print_sycl_devices();
    }

    const size_t dst_row = ggml_row_size(GGML_TYPE_XE2MX4, n_per_row);
    const int64_t nb = n_per_row / QK_XE2MX4;
    const size_t qw_bytes = (size_t) n_per_row * sizeof(float);
    const int64_t n = nrows * n_per_row;

    std::vector<float> src_f32;
    xe2mx_host_f32(src, src_type, n, src_f32);

    struct job {
        sycl::queue q;
        float * d_src;
        float * d_d;
        int * d_q;
        float * d_qw;
        int64_t row0;
        int64_t nr;
    };
    std::vector<job> jobs;
    jobs.reserve((size_t) nd);

    for (int d = 0; d < nd; ++d) {
        const int64_t row0 = (nrows * (int64_t) d) / nd;
        const int64_t row1 = (nrows * (int64_t) (d + 1)) / nd;
        const int64_t nr = row1 - row0;
        if (nr <= 0) {
            continue;
        }
        ggml_sycl_set_device(d);
        sycl::device dev = dpct::dev_mgr::instance().get_device(d);
        sycl::queue qdev(dev, sycl::property::queue::in_order());
        float * d_src = sycl::malloc_device<float>((size_t) nr * (size_t) n_per_row, qdev);
        float * d_d = sycl::malloc_device<float>((size_t) nr * (size_t) nb, qdev);
        int * d_q = sycl::malloc_device<int>((size_t) nr * (size_t) n_per_row, qdev);
        float * d_qw = nullptr;
        if (imatrix) {
            d_qw = sycl::malloc_device<float>((size_t) n_per_row, qdev);
            qdev.memcpy(d_qw, imatrix, qw_bytes);
        }
        jobs.push_back(job{ std::move(qdev), d_src, d_d, d_q, d_qw, row0, nr });
    }

    for (auto & j : jobs) {
        const float * src_slice = src_f32.data() + j.row0 * n_per_row;
        char * dst_slice = (char *) dst + (size_t) j.row0 * dst_row;
        xe2mx_quant_submit(j.q, src_slice, dst_slice, j.d_src, j.d_d, j.d_q, j.d_qw, j.nr, n_per_row);
    }
    for (auto & j : jobs) {
        sycl::free(j.d_src, j.q);
        sycl::free(j.d_d, j.q);
        sycl::free(j.d_q, j.q);
        if (j.d_qw) {
            sycl::free(j.d_qw, j.q);
        }
    }

    return (size_t) nrows * dst_row;
}
