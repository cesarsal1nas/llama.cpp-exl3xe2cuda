#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 types.cuh
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

#include <cuda_fp16.h>
#include <cstdint>
#include "ptx.cuh"

union half2_uint32 {
    uint32_t as_uint32;
    half2    as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16 {
    uint16_t as_uint16;
    half     as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

typedef struct __align__(8) half4 {
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
} half4;

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id) {
    for (int i = 1; i < 32; i <<= 1) {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t * vi = reinterpret_cast<uint32_t *>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}
