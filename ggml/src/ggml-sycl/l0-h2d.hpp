#pragma once

// Level Zero copy-engine H2D for Xe2 dGPU. Default SYCL memcpy.wait hangs on spare B65.
// GGML_SYCL_L0_H2D: unset = on for bmg_g21/g31 only, 0 = off, 1 = force on.

#include "common.hpp"

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API) && !defined(_WIN32)

#include <atomic>
#include <vector>
#include <level_zero/ze_api.h>

static bool ggml_sycl_l0_h2d_enabled() {
    const int env = ggml_sycl_get_env("GGML_SYCL_L0_H2D", -1);
    if (env == 0) {
        return false;
    }
    if (env > 0) {
        return true;
    }
    const int dev = ggml_sycl_get_device();
    if (dev < 0 || dev >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    return ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[dev].hw_info.arch);
}

static uint32_t ggml_sycl_l0_copy_ordinal(ze_device_handle_t ze_dev) {
    uint32_t n = 0;
    if (zeDeviceGetCommandQueueGroupProperties(ze_dev, &n, nullptr) != ZE_RESULT_SUCCESS || n == 0) {
        return 0;
    }
    std::vector<ze_command_queue_group_properties_t> props(n);
    for (uint32_t i = 0; i < n; ++i) {
        props[i] = {};
        props[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    if (zeDeviceGetCommandQueueGroupProperties(ze_dev, &n, props.data()) != ZE_RESULT_SUCCESS) {
        return 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const ze_command_queue_group_property_flags_t f = props[i].flags;
        if ((f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
            return i;
        }
    }
    return 0;
}

struct ggml_sycl_l0_h2d_engine {
    ze_command_list_handle_t cl = nullptr;
    ze_context_handle_t ctx = nullptr;
    ze_device_handle_t dev = nullptr;
    void * host = nullptr;
    void * staging = nullptr;
    size_t host_cap = 0;
};

static ggml_sycl_l0_h2d_engine g_l0_h2d[GGML_SYCL_MAX_DEVICES];

static bool ggml_sycl_l0_ensure(int device, sycl::queue & q) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    if (q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        return false;
    }
    ggml_sycl_l0_h2d_engine & eng = g_l0_h2d[device];
    if (eng.cl != nullptr) {
        return true;
    }
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    uint32_t copy_ord = ggml_sycl_l0_copy_ordinal(ze_dev);
    uint32_t ordinals[2] = { copy_ord, 0 };
    ze_result_t zr = ZE_RESULT_ERROR_UNKNOWN;
    for (uint32_t oi = 0; oi < 2; ++oi) {
        const uint32_t ordinal = ordinals[oi];
        if (oi == 1 && ordinal == ordinals[0]) {
            continue;
        }
        ze_command_queue_desc_t cq_desc = {
            ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 0,
            ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        zr = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &eng.cl);
        if (zr == ZE_RESULT_SUCCESS) {
            eng.ctx = ze_ctx;
            if (eng.host == nullptr) {
                constexpr size_t chunk = 4ull * 1024ull * 1024ull;
                ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
                const ze_result_t hr = zeMemAllocHost(ze_ctx, &hd, chunk, 4096, &eng.host);
                if (hr != ZE_RESULT_SUCCESS || !eng.host) {
                    zeCommandListDestroy(eng.cl);
                    eng.cl = nullptr;
                    eng.ctx = nullptr;
                    eng.host = nullptr;
                    continue;
                }
                eng.host_cap = chunk;
            }
            eng.dev = ze_dev;
            if (eng.staging == nullptr) {
                ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
                const ze_result_t sr = zeMemAllocDevice(ze_ctx, &dd, eng.host_cap, 4096, ze_dev, &eng.staging);
                if (sr != ZE_RESULT_SUCCESS || !eng.staging) {
                    eng.staging = nullptr;
                }
            }
            return true;
        }
        eng.cl = nullptr;
    }
    return false;
}

static bool ggml_sycl_l0_h2d_copy(int device, sycl::queue & q, void * dst, const void * src, size_t size) {
    if (size == 0) {
        return true;
    }
    if (!ggml_sycl_l0_ensure(device, q)) {
        return false;
    }
    ggml_sycl_l0_h2d_engine & eng = g_l0_h2d[device];
    if (eng.host == nullptr || eng.cl == nullptr) {
        return false;
    }

    ze_memory_allocation_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t alloc_dev = nullptr;
    const ze_result_t pr = zeMemGetAllocProperties(eng.ctx, dst, &props, &alloc_dev);
    if (pr == ZE_RESULT_SUCCESS &&
        (props.type == ZE_MEMORY_TYPE_HOST || props.type == ZE_MEMORY_TYPE_SHARED)) {
        memcpy(dst, src, size);
        return true;
    }

    const char * s = (const char *) src;
    char * d = (char *) dst;
    size_t left = size;
    static std::atomic<uint64_t> copied{0};
    while (left > 0) {
        const size_t n = left < eng.host_cap ? left : eng.host_cap;
        memcpy(eng.host, s, n);
        ze_result_t r;
        if (eng.staging && props.type != ZE_MEMORY_TYPE_DEVICE) {
            r = zeCommandListAppendMemoryCopy(eng.cl, eng.staging, eng.host, n, nullptr, 0, nullptr);
            if (r != ZE_RESULT_SUCCESS) {
                return false;
            }
            r = zeCommandListAppendMemoryCopy(eng.cl, d, eng.staging, n, nullptr, 0, nullptr);
        } else {
            r = zeCommandListAppendMemoryCopy(eng.cl, d, eng.host, n, nullptr, 0, nullptr);
        }
        if (r != ZE_RESULT_SUCCESS) {
            return false;
        }
        s += n;
        d += n;
        left -= n;
        const uint64_t tot = copied.fetch_add(n) + n;
        if ((tot / (2ull << 30)) != ((tot - n) / (2ull << 30))) {
            zeCommandListDestroy(eng.cl);
            eng.cl = nullptr;
            if (!ggml_sycl_l0_ensure(device, q)) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_sycl_l0_fill(int device, sycl::queue & q, void * dst, uint8_t value, size_t size) {
    if (size == 0) {
        return true;
    }
    if (!ggml_sycl_l0_ensure(device, q)) {
        return false;
    }
    ggml_sycl_l0_h2d_engine & eng = g_l0_h2d[device];
    if (eng.host == nullptr || eng.cl == nullptr) {
        return false;
    }
    char * d = (char *) dst;
    size_t left = size;
    while (left > 0) {
        const size_t n = left < eng.host_cap ? left : eng.host_cap;
        memset(eng.host, value, n);
        const ze_result_t r = zeCommandListAppendMemoryCopy(eng.cl, d, eng.host, n, nullptr, 0, nullptr);
        if (r != ZE_RESULT_SUCCESS) {
            return false;
        }
        d += n;
        left -= n;
    }
    return true;
}

#else

static bool ggml_sycl_l0_h2d_enabled() {
    return false;
}

#endif
