//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

class DnnlGemmWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
#ifdef GGML_SYCL_HAS_BF16
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
#endif
        else static_assert(0);
    }

    struct GemmCacheKey {
        sycl::queue * q;
        int m, n, k;
        int at, bt, ct;
        dnnl_dim_t stra0, stra1, stra2;
        dnnl_dim_t strb0, strb1, strb2;
        dnnl_dim_t batches_a, batches_b;

        bool operator==(const GemmCacheKey & o) const {
            return q == o.q && m == o.m && n == o.n && k == o.k &&
                   at == o.at && bt == o.bt && ct == o.ct &&
                   stra0 == o.stra0 && stra1 == o.stra1 && stra2 == o.stra2 &&
                   strb0 == o.strb0 && strb1 == o.strb1 && strb2 == o.strb2 &&
                   batches_a == o.batches_a && batches_b == o.batches_b;
        }
    };

    struct GemmCacheHash {
        size_t operator()(const GemmCacheKey & k) const {
            size_t h = std::hash<void *>{}(k.q);
            auto mix = [&](size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix((size_t) k.m);
            mix((size_t) k.n);
            mix((size_t) k.k);
            mix((size_t) k.at | ((size_t) k.bt << 8) | ((size_t) k.ct << 16));
            mix((size_t) k.stra0);
            mix((size_t) k.stra1);
            mix((size_t) k.stra2);
            mix((size_t) k.strb0);
            mix((size_t) k.strb1);
            mix((size_t) k.strb2);
            mix((size_t) k.batches_a);
            mix((size_t) k.batches_b);
            return h;
        }
    };

    struct GemmCacheEntry {
        dnnl::memory::desc a_in_md;
        dnnl::memory::desc b_in_md;
        dnnl::memory::desc c_md;
        dnnl::memory::desc scratchpad_md;
        dnnl::matmul prim;
    };

    static void gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        const void * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        void * c, dt ct, const queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        static std::mutex cache_mu;
        static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheHash> cache;

        GemmCacheKey key{
            q, m, n, k,
            (int) at, (int) bt, (int) ct,
            stra0, stra1, stra2,
            strb0, strb1, strb2,
            batches_a, batches_b
        };

        GemmCacheEntry entry;
        {
            std::lock_guard<std::mutex> lk(cache_mu);
            auto it = cache.find(key);
            if (it != cache.end()) {
                entry = it->second;
            } else {
                dnnl::memory::dims a_dims = {batches_a, m, k};
                dnnl::memory::dims a_strides = {stra2, stra1, stra0};
                const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

                dnnl::memory::dims b_dims = {batches_b, k, n};
                dnnl::memory::dims b_strides = {strb2, strb0, strb1};
                const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

                dnnl::memory::dims c_dims = {std::max(batches_a, batches_b), m, n};
                dnnl::memory::dims c_strides = {m * n, 1, m};
                const auto c_in_md = dnnl::memory::desc(c_dims, ct, c_strides);

                dnnl::primitive_attr primitive_attr;
                primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef GGML_SYCL_F16
                primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif
                auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_in_md, primitive_attr);
                entry = GemmCacheEntry{
                    a_in_md,
                    b_in_md,
                    matmul_pd.dst_desc(),
                    matmul_pd.scratchpad_desc(),
                    dnnl::matmul(matmul_pd)
                };
                if (cache.size() > 4096) {
                    cache.clear();
                }
                cache.emplace(key, entry);
            }
        }

        auto a_mem = dnnl::memory(entry.a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(entry.b_in_md, eng, const_cast<void*>(b));
        auto c_mem = dnnl::memory(entry.c_md, eng, c);
        auto scratchpad_mem = ctx.get_scratchpad_mem(entry.scratchpad_md, eng, q);

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        entry.prim.execute(stream, matmul_args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
