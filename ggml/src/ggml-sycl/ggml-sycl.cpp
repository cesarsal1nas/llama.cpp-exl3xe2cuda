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

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <functional>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <optional>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <typeinfo>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#if defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif
#include <sycl/half_type.hpp>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/l0-h2d.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/exl3.hpp"
#include "ggml-sycl/xe2mx.hpp"
#include "ggml-sycl/xe2mx-a8.hpp"
#include "ggml-sycl/fwht.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/mem.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/dsv4-hc.hpp"
#include "ggml-sycl/lightning-indexer.hpp"
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/opt-step.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_mkl_fa = 1;
int g_ggml_sycl_memtrace = 0;
int g_ggml_sycl_memtrace_step = 64;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_enable_esimd = 1;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;
int g_ggml_sycl_enable_host_pinned_mem = 1;
int g_ggml_sycl_host_pinned_mem_2g = 0;
int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;


static void ggml_sycl_apply_xe2_profile(const ggml_sycl_device_info & info);

static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    try {
        info.device_count = dpct::dev_mgr::instance().device_count();
    } catch (sycl::exception const &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_count = 0;
    }
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#if !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#else
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            if (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = true;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    // Large buffers can be allocated before ggml_check_sycl() initializes other
    // g_ggml_sycl_enable_* globals, so initialize this one as early as we can.
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
    g_ggml_sycl_use_level_zero_api = 0;
#endif

    ggml_sycl_apply_xe2_profile(info);

    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else if (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        return "Host Forward";
    } else {
        return "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
void initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ze_result_t zes_init = zesInit(0);
    if (zes_init != ZE_RESULT_SUCCESS) {
        std::cerr << "Warning: zesInit failed [ggml_check_sycl] with code " << static_cast<int>(zes_init)
                            << ". Sysman free-memory query may be unavailable.\n";
    }
#endif
}


#include <dlfcn.h>

// The XE2MX4 s4 path (LO packs, S44 GEMMs) needs the two oneDNN patches in
// docs/xe2mx4.md. Stock oneDNN picks a generic kernel for that matmul and the
// device can be lost on the first long prefill. The patched library is the one
// that reads S44_STRAT_EXTRA, so look for that string in the loaded libdnnl.
static bool ggml_sycl_dnnl_has_s4_patch() {
#if GGML_SYCL_DNNL
    static const bool ok = [] {
        Dl_info info{};
        if (!dladdr((const void *) &dnnl_version, &info) || !info.dli_fname) {
            return false;
        }
        std::ifstream f(info.dli_fname, std::ios::binary);
        if (!f) {
            return false;
        }
        static const std::string needle = "S44_STRAT_EXTRA";
        std::string carry;
        std::vector<char> buf(1 << 20);
        while (f) {
            f.read(buf.data(), (std::streamsize) buf.size());
            const std::streamsize n = f.gcount();
            if (n <= 0) {
                break;
            }
            std::string chunk = carry + std::string(buf.data(), (size_t) n);
            if (chunk.find(needle) != std::string::npos) {
                return true;
            }
            carry = chunk.substr(chunk.size() > needle.size() ? chunk.size() - needle.size() : 0);
        }
        return false;
    }();
    return ok;
#else
    return false;
#endif
}

static void ggml_sycl_setenv_default(const char * k, const char * v) {
    setenv(k, v, 0);
}

// Battlemage dGPU defaults. These are the settings the XE2MX4 numbers were measured
// with. Every value can still be overridden from the environment, and
// GGML_SYCL_XE2_PROFILE=0 turns the whole block off.
static void ggml_sycl_apply_xe2_profile(const ggml_sycl_device_info & info) {
    if (!ggml_sycl_get_env("GGML_SYCL_XE2_PROFILE", 1)) {
        return;
    }
    bool xe2 = false;
    for (int i = 0; i < info.device_count; ++i) {
        xe2 = xe2 || ggml_sycl_is_xe2_dgpu(info.devices[i].hw_info.arch);
    }
    if (!xe2) {
        return;
    }
    static const char * generic[][2] = {
        { "GGML_SYCL_FUSE_CONT_UNARY", "1" },     { "GGML_SYCL_FUSE_ZSILU", "1" },
        { "GGML_SYCL_FUSE_CONT_UNARY_MUL", "1" }, { "GGML_SYCL_FUSE_GDN_GATE", "1" },
        { "GGML_SYCL_FA_ONEDNN_PKV", "0" },       { "GGML_SYCL_FA_XMX_MIN_KV", "4096" },
        { "GGML_SYCL_FA_DEC_WALK", "24" },        { "GGML_SYCL_XE2MX_CONV_NOCAT", "1" },
        { "GGML_SYCL_XE2MX_GLU_TILE", "92" },     { "GGML_SYCL_XE2MX_XMX_TILE", "82" },
        { "GGML_MTP_DRAFT_NUBATCH", "4" },        { "LLAMA_MTP_KEEP_KV", "1" },
    };
    for (auto & kv : generic) {
        ggml_sycl_setenv_default(kv[0], kv[1]);
    }
    GGML_SYCL_XE2_LOG("xe2 profile: generic defaults applied\n");
    if (!ggml_sycl_dnnl_has_s4_patch()) {
        // Stock oneDNN: keep the stock s4 matmul. The S44 kernels below can lose the device.
        GGML_LOG_WARN("%s: oneDNN without the XE2MX4 s4 patches, XE2MX4 prefill uses the slower stock s4 path\n", __func__);
        return;
    }
    static const char * s4[][2] = {
        { "S44_STRAT_EXTRA", "k256" },            { "GGML_SYCL_XE2MX_LO", "1" },
        { "GGML_SYCL_XE2MX_LO_HOST", "0" },       { "GGML_SYCL_XE2MX_LO_INLINE", "0" },     { "GGML_SYCL_XE2MX_LO_NOPACK_FFN", "1" },
        { "GGML_SYCL_XE2MX_LO_GEMV", "1" },       { "GGML_SYCL_XE2MX_S4ACC", "3" },
        { "GGML_SYCL_XE2MX_Q32V", "1" },          { "GGML_SYCL_XE2MX_NF_KEEP_MB", "0" },
        { "GGML_SYCL_XE2MX_S44", "1" },           { "GGML_SYCL_XE2MX_S44N", "4096" },
        { "GGML_SYCL_XE2MX_S44FROM", "0" },       { "GGML_SYCL_XE2MX_S44FFNFROM", "0" },
        { "GGML_SYCL_XE2MX_S44AROW", "0" },
        { "GGML_SYCL_XE2MX_S44A", "8" },          { "GGML_SYCL_XE2MX_S44CLIP", "80" },
        { "GGML_SYCL_XE2MX_S44ADIV", "8" },       { "GGML_SYCL_XE2MX_S44FPACK", "1" },
        { "GGML_SYCL_XE2MX_S44HILP", "4" },       { "GGML_SYCL_XE2MX_S44H", "1" },
        { "GGML_SYCL_XE2MX_S44PN", "1" },         { "GGML_SYCL_XE2MX_S44RMSHILP", "1" },
        { "GGML_SYCL_XE2MX_S44G", "1024" },       { "GGML_SYCL_XE2MX_S44G_UP", "128" },
        { "GGML_SYCL_XE2MX_S44G_MID", "128" },
    };
    for (auto & kv : s4) {
        ggml_sycl_setenv_default(kv[0], kv[1]);
    }
    GGML_SYCL_XE2_LOG("xe2 profile: s4 defaults applied\n");
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_step = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);

        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);

        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        g_ggml_sycl_host_pinned_mem_2g =
            ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);

#if defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
        GGML_LOG_INFO("  GGML_SYCL_EXL3: %d\n", ggml_sycl_get_env("GGML_SYCL_EXL3", -1));
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
        GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS: {
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
    }

    if (tensor->type == GGML_TYPE_XE2MX4 && tensor->extra == nullptr) {
        ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
        tensor->extra                 = extra;
        ctx->tensor_extras.push_back(extra);
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Graph inputs during decode (tokens, positions, KQ mask, state ids, ~19 uploads per MTP step) went through the
// load-time path: full device wait, malloc staging, blocking memcpy. Once compute has started, small uploads are
// copied into a pinned ring and enqueued on the same in-order queue the graph runs on, without a host wait.
// A ring slot is only reused after a full queue wait at wrap. GGML_SYCL_ASYNC_INPUT=1 enables .
static bool g_sycl_compute_started = false;

static bool ggml_sycl_async_input(int device, sycl::queue & q, void * dst, const void * src, size_t size) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_ASYNC_INPUT", 0);
    if (!on || !g_sycl_compute_started || size == 0 || size > ((size_t) 8 << 20) || device < 0 ||
        device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    struct input_ring {
        char *        base = nullptr;
        size_t        cap  = 0;
        size_t        off  = 0;
        sycl::queue * q    = nullptr;
        long          n    = 0;
        long          wraps = 0;
    };
    static input_ring rings[GGML_SYCL_MAX_DEVICES];
    input_ring & r = rings[device];
    if (!r.base) {
        r.cap = (size_t) 64 << 20;
        r.base = (char *) sycl::malloc_host(r.cap, q);
        if (!r.base) {
            r.cap = 0;
            return false;
        }
        r.q = &q;
        GGML_SYCL_XE2_LOG( "sycl async input ring: device %d, %zu MiB pinned\n", device, r.cap >> 20);
        fflush(stderr);
    }
    if (r.q != &q) {
        return false;
    }
    const size_t need = (size + 255) & ~(size_t) 255;
    if (r.off + need > r.cap) {
        q.wait();
        r.off = 0;
        r.wraps++;
    }
    char * p = r.base + r.off;
    r.off += need;
    std::memcpy(p, src, size);
    q.memcpy(dst, p, size);
    r.n++;
    return true;
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (offset == 0 && tensor->name && strstr(tensor->name, "ffn_gate.weight")) {
        static int nlog = 0;
        if (nlog < 4) {
            nlog++;
            GGML_SYCL_XE2_LOG( "xe2mx tp set dev=%d %s ne=%ld x %ld bytes=%zu\n",
                    ctx->device, tensor->name, (long) tensor->ne[0], (long) tensor->ne[1], size);
            fflush(stderr);
        }
    }
    if (offset == 0 && ggml_sycl_xe2mx_s8dw_upload(tensor, data, size, *stream)) {
        return;
    }
    if (offset == 0 && ggml_sycl_xe2mx_lo_upload(tensor, data, size, *stream)) {
        return;
    }
    if (buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_sycl_async_input(ctx->device, *stream, (char *) tensor->data + offset, data, size)) {
        return;
    }
#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API) && !defined(_WIN32)
    if (ggml_sycl_l0_h2d_enabled() &&
        ggml_sycl_l0_h2d_copy(ctx->device, *stream, (char *) tensor->data + offset, data, size)) {
        return;
    }
    if (ggml_sycl_l0_h2d_enabled()) {
        return;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // LO_INLINE: t->data is s4 pack, not block_xe2mx4 - refuse silent wrong export.
    if (tensor && tensor->type == GGML_TYPE_XE2MX4 && !ggml_sycl_xe2mx_lo_bytes_are_soa(tensor)) {
        fprintf(stderr, "xe2mx: refuse get_tensor of LO_INLINE weight %s (no SOA materialize)\n",
                tensor->name ? tensor->name : "?");
        fflush(stderr);
        GGML_ABORT("xe2mx LO_INLINE get_tensor unsupported");
    }
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    if(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } else {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    char *host_buf = (char *)malloc(size);
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    // LO_INLINE src/dst: raw memcpy would treat s4 pack as XE2MX4 SOA - refuse.
    if ((src && src->type == GGML_TYPE_XE2MX4 && !ggml_sycl_xe2mx_lo_bytes_are_soa(src))
        || (dst && dst->type == GGML_TYPE_XE2MX4 && !ggml_sycl_xe2mx_lo_bytes_are_soa(dst))) {
        fprintf(stderr, "xe2mx: refuse cpy_tensor LO_INLINE %s -> %s\n",
                src && src->name ? src->name : "?",
                dst && dst->name ? dst->name : "?");
        fflush(stderr);
        return false;
    }
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;
        static const bool pair_thread = getenv("LLAMA_XE2_PREFILL_THREAD") && atoi(getenv("LLAMA_XE2_PREFILL_THREAD")) == 1;
        if (pair_thread && src_ctx->device != dst_ctx->device) {
            // The scheduler already waited on the producing split's event (source) and on this slot's
            // previous use of dst (event). Copy on side in-order queues so later work another slot queued
            // on either compute queue is neither waited on nor overtaken.
            static std::mutex mu;
            std::lock_guard<std::mutex> lk(mu);
            static sycl::queue * side[GGML_SYCL_MAX_DEVICES] = {};
            static char * pin[GGML_SYCL_MAX_DEVICES] = {};
            static size_t pin_cap[GGML_SYCL_MAX_DEVICES] = {};
            const size_t size = ggml_nbytes(src);
            const int ds[2] = { src_ctx->device, dst_ctx->device };
            queue_ptr qs[2] = { src_ctx->stream, dst_ctx->stream };
            for (int k = 0; k < 2; ++k) {
                const int d = ds[k];
                if (!side[d]) {
                    side[d] = new sycl::queue(qs[k]->get_context(), qs[k]->get_device(), sycl::property::queue::in_order());
                }
                if (pin_cap[d] < size) {
                    if (pin[d]) sycl::free(pin[d], *side[d]);
                    pin[d] = (char *) sycl::malloc_host(size, *side[d]);
                    GGML_ASSERT(pin[d]);
                    pin_cap[d] = size;
                }
            }
            SYCL_CHECK(CHECK_TRY_ERROR(side[ds[0]]->memcpy(pin[ds[0]], src->data, size).wait()));
            memcpy(pin[ds[1]], pin[ds[0]], size);
            SYCL_CHECK(CHECK_TRY_ERROR(side[ds[1]]->memcpy(dst->data, pin[ds[1]], size).wait()));
            return true;
        }

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    ggml_sycl_xe2mx_lo_hostpack_release(*stream);
#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API) && !defined(_WIN32)
    {
        auto & defq = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();
        if (ggml_sycl_l0_h2d_enabled() &&
            ggml_sycl_l0_fill(ctx->device, defq, ctx->dev_ptr, value, buffer->size)) {
            return;
        }
        if (ggml_sycl_l0_h2d_enabled()) {
            return;
        }
    }
#endif
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API) && !defined(_WIN32)
    if (ggml_sycl_l0_h2d_enabled()) {
        auto & defq = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();
        if (ggml_sycl_l0_fill(ctx->device, defq, target_ptr, value, size)) {
            return;
        }
        return;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    size_t max_alloc_size = dpct::get_current_device().get_max_mem_alloc_size();
    if (g_ggml_sycl_host_pinned_mem_2g) {
        return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } else {
        return max_alloc_size;
    }
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    const size_t s8dw = ggml_sycl_xe2mx_s8dw_nbytes(tensor);
    if (s8dw) {
        return s8dw;
    }
    const size_t lo = ggml_sycl_xe2mx_lo_nbytes(tensor);
    if (lo) {
        return lo;
    }
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_XE2MX4:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        case GGML_TYPE_EXL3_1:
        case GGML_TYPE_EXL3_2:
        case GGML_TYPE_EXL3_3:
        case GGML_TYPE_EXL3_4:
        case GGML_TYPE_EXL3_5:
        case GGML_TYPE_EXL3_6:
        case GGML_TYPE_EXL3_7:
        case GGML_TYPE_EXL3_8:
            return 128;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    if (tensor->type != GGML_TYPE_XE2MX4) {
        const int home = ggml_sycl_info().device_count - 1;
        if (id == home) {
            *row_low = 0;
            *row_high = nrows;
        } else {
            *row_low = 0;
            *row_high = 0;
        }
        return;
    }
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    const int ndev = ggml_sycl_info().device_count;
    ctx->streams.resize(ndev);
    for (int i = 0; i < ndev; ++i) {
        ctx->streams[i] = &(dpct::dev_mgr::instance().get_device(i).default_queue());
    }

    for (int i = 0; i < ndev; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;
        if (tensor->type != GGML_TYPE_XE2MX4 && i == ggml_sycl_info().device_count - 1) {
            tensor->data = buf;
        }

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
        if (tensor->type == GGML_TYPE_XE2MX4 && tensor->ne[2] == 1 && tensor->ne[3] == 1
                && ggml_sycl_xe2mx_lo_on(tensor) && (nrows_split % 16) == 0) {
            ggml_tensor view = *tensor;
            view.ne[1] = nrows_split;
            view.data = extra->data_device[i];
            view.buffer = nullptr;
            view.view_src = nullptr;
            if (ggml_nbytes(&view) == original_size &&
                    ggml_sycl_xe2mx_lo_upload(&view, buf_host, original_size, *stream, true)) {
                void * slice = extra->data_device[i];
                void * packed = ggml_sycl_xe2mx_lo_packed(slice);
                if (packed && packed != slice) {
                    ggml_sycl_xe2mx_lo_rekey(slice, packed);
                    ggml_sycl_free_device(slice, *stream);
                    extra->data_device[i] = packed;
                }
            } else if (ggml_nbytes(&view) != original_size) {
                static int nmiss = 0;
                if (nmiss < 4) {
                    nmiss++;
                    GGML_SYCL_XE2_LOG( "xe2mx row upload skip %s nbytes=%zu slice=%zu\n",
                            tensor->name, ggml_nbytes(&view), original_size);
                    fflush(stderr);
                }
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    main_device = ggml_sycl_info().device_count - 1;

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), main_device),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

//host pinned memory
static void * ggml_backend_sycl_host_malloc(size_t size) {
    void * ptr = nullptr;
    try {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } catch (...) {
        ptr = nullptr;
    }
    if (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    return ptr;
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context == nullptr) {
        return;
    }
    if (g_ggml_sycl_enable_host_pinned_mem) {
        auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(buffer->context, q)));
    } else {
        free_aligned_mem_host((void *) buffer->context);
    }
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = g_ggml_sycl_enable_host_pinned_mem ? ggml_backend_sycl_host_malloc(size) :
                                                      aligned_malloc_host(TENSOR_ALIGNMENT, size);
    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    if (g_ggml_sycl_enable_host_pinned_mem) {
        ggml_backend_sycl_device_context * dev_ctx = (ggml_backend_sycl_device_context *) buft->device->context;
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(dev_ctx->device).get_max_mem_alloc_size();
        if (g_ggml_sycl_host_pinned_mem_2g) {
            return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } else {
            return max_alloc_size;
        }
    } else {
        return SIZE_MAX;
    }
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ ggml_backend_sycl_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_SUPPORT_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            if (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_size);

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    inline static int counter{ 0 };

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
        counter = 0;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            ggml_sycl_buffer b               = buffer_pool[0];
            void *           ptr             = b.ptr;
            *actual_size                     = b.size;
            counter                          = 1;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];

        if (b.ptr == nullptr) {
            void * ptr;

            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size += size;
            *actual_size = size;
            counter      = counter + 1;
            return ptr;
        } else {
            ++counter;
            b.size = size;
            return b.ptr;
        }
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise do nothing, pointers will be freed once the pool is deallocated.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}


std::unique_ptr<ggml_sycl_fattn_kv_buffers> ggml_backend_sycl_context::new_fattn_kv_buffers(queue_ptr qptr, int device) {
    return std::unique_ptr<ggml_sycl_fattn_kv_buffers>(new ggml_sycl_fattn_kv_buffers(qptr, device));
}

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

// Scan and block merge, shared by every launch shape below so a partitioned row uses the
// same insertion order as an unpartitioned one.
//
// src_map  != nullptr: report src_map[col] instead of col, so a merge pass can carry the
//                      original column index through.
// out_vals != nullptr: also emit the k winning values, for a later merge pass.
// swap01:              emit in the output order the single-pass path uses.
static void top_k_scan_merge_f32(
    const float *   src_vals,
    const int32_t * src_map,
    const int       begin,
    const int       end,
    const int       k,
    const int       block_size,
    float *         shared_vals,
    int *           shared_idx,
    float *         out_vals,
    int32_t *       out_idx,
    const bool      swap01,
    const sycl::nd_item<1> & item_ct1
) {
    const int tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    float * lv = shared_vals + tid;
    int * li = shared_idx + tid;

    for (int i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    float kth = -FLT_MAX;

    for (int col = begin + tid; col < end; col += block_size) {
        float val = src_vals[col];

        if (val > kth) {
            int pos = k - 1;
            while (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            for (int i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tid != 0) {
        return;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    float * fv = shared_vals + (size_t) k * block_size;
    int * fi = shared_idx + (size_t) k * block_size;

    for (int i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    float fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    for (int t = 0; t < block_size; t++) {
        for (int i = 0; i < k; i++) {
            float val = shared_vals[i * block_size + t];

            if (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                break;
            }

            int idx = shared_idx[i * block_size + t];

            int pos = k - 1;
            while (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            for (int j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    if (out_vals) {
        for (int i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    for (int i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    if (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

static void top_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    // A row is scanned by exactly one work-group, so a vocabulary-sized row leaves the
    // rest of the device idle. What the scan is short of is memory requests in flight,
    // not bandwidth or per-request latency, so lanes in flight is the lever: split the
    // row across independent work-groups, have each emit its partition's top-k, and
    // merge those nsplit*k candidates in a second launch.
    //
    // split_block trades parallelism against SLM residency. Its cost is
    // (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128
    // lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit
    // if the supported k ever grows.
    constexpr int split_block = 128;
    constexpr int max_splits  = 128;
    constexpr int min_cols    = 8192;

    int nsplit = 1;
    if (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        const int64_t want = ncols / split_block;
        nsplit = (int) (want > max_splits ? max_splits : want);
    }

    if (nsplit > 1) {
        const int nchunk = (int) ((ncols + nsplit - 1) / nsplit);
        const size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<float>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        float *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        const sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int grp = item_ct1.get_group(0);
                    const int row = grp / nsplit;
                    const int part = grp % nsplit;

                    const int begin = part * nchunk;
                    int end = begin + nchunk;
                    if (end > (int) ncols) {
                        end = (int) ncols;
                    }

                    top_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, false, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int row = item_ct1.get_group(0);
                    const size_t off = (size_t) row * nsplit * k;

                    top_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
                });
        });

        return;
    }

    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);

                if (row >= nrows) return;

                top_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (int) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
            });
    });
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                if (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = true;
#endif

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#if GGML_SYCL_DNNL
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0 && k <= 32);
    GGML_ASSERT(k <= ncols);

    top_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int nrows0 = ggml_nrows(dst->src[0]);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                            src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
#if GGML_SYCL_DNNL
    const size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#if GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (g_ggml_sycl_enable_dnn) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            if (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_reorder_esimd(enum ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_XE2MX4:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        void * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        return ptr;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0* x = (const block_q8_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q2_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q2_K) == 0);

    const int nblocks = size / sizeof(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q2_K * x  = (const block_q2_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q3_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q3_K) == 0);

    const int nblocks = size / sizeof(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q3_K * x  = (const block_q3_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        for (int j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q5_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q5_K) == 0);

    const int nblocks = size / sizeof(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q5_K * x  = (const block_q5_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}


static bool reorder_qw_iq4_xs(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_iq4_xs) == 0);
    GGML_ASSERT(offset % sizeof(block_iq4_xs) == 0);

    const int nblocks = size / sizeof(block_iq4_xs);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr       = data_device;
    auto * scales_l_ptr = qs_ptr + (QK_K / 2) * nblocks;
    auto * packed_ptr   = scales_l_ptr + 4 * nblocks;

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_iq4_xs * x  = (const block_iq4_xs *) tmp_buf;
        const int            ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < 4; ++j) {
            scales_l_ptr[ib * 4 + j] = x[ib].scales_l[j];
        }
        uint16_t *  sh = (uint16_t *) (packed_ptr + ib * 4);
        ggml_half * d  = (ggml_half *) (packed_ptr + ib * 4 + 2);
        *sh            = x[ib].scales_h;
        *d             = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_iq4_xs_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_iq4_xs) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_iq4_xs));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for IQ4 MoE reorder, skipping\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int            gb   = gb_;
        const int            e    = gb / blocks_per_expert;
        const int            ib   = gb % blocks_per_expert;
        const block_iq4_xs * x    = (const block_iq4_xs *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *            base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr       = base;
        auto * scales_l_ptr = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * packed_ptr   = scales_l_ptr + 4 * blocks_per_expert;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < 4; ++j) {
            scales_l_ptr[ib * 4 + j] = x[ib].scales_l[j];
        }
        uint16_t *  sh = (uint16_t *) (packed_ptr + ib * 4);
        ggml_half * d  = (ggml_half *) (packed_ptr + ib * 4 + 2);
        *sh            = x[ib].scales_h;
        *d             = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_IQ4_XS:
                return reorder_qw_iq4_xs_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        case GGML_TYPE_IQ4_XS:
            return reorder_qw_iq4_xs(data_device, size, 0, stream);
        default:
            return false;
    }
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        return;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K &&
        src0->type != GGML_TYPE_IQ4_XS) {
        return;
    }
    if (src0->type == GGML_TYPE_IQ4_XS) {
        const int env = ggml_sycl_get_env("GGML_SYCL_MOE_IQ", -1);
        if (env == 0) {
            return;
        }
        if (env < 0) {
            const int dev = ggml_sycl_get_device();
            if (!ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[dev].hw_info.arch)) {
                return;
            }
        }
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_is_quantized(src0->type) && !ggml_is_exl3(src0->type) && src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

static void ggml_sycl_mul_mat_xe2mx_split(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx =
        (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
    const int main_dev = ctx.device;
    const int ndev = ggml_sycl_info().device_count;
    const int64_t Nfull = src0->ne[1];
    const int64_t nout = Nfull > 0 ? ggml_nelements(dst) / Nfull : 0;
    GGML_ASSERT(nout > 0);
    const queue_ptr saved_ovl = ctx.ovl_q;

    for (int i = 0; i < ndev; ++i) {
        int64_t row_low = 0;
        int64_t row_high = 0;
        get_row_split(&row_low, &row_high, src0, buft_ctx->tensor_split, i);
        const int64_t nr = row_high - row_low;
        if (nr <= 0) {
            continue;
        }
        GGML_ASSERT(extra->data_device[i]);

        static int npre = 0;
        if (npre < 8) {
            npre++;
            GGML_SYCL_XE2_LOG( "xe2mx row pre dev=%d main=%d nr=%ld nout=%ld src1=%zu %s\n",
                    i, main_dev, (long) nr, (long) nout, ggml_nbytes(src1), src0->name ? src0->name : "");
            fflush(stderr);
        }

        ggml_sycl_set_device(i);
        ctx.device = i;
        ctx.ovl_q = nullptr;
        queue_ptr sq = ctx.stream();

        ggml_sycl_pool_alloc<char> a_hold;
        void * src1_local = src1->data;
        if (i != main_dev) {
            const size_t nb1 = ggml_nbytes(src1);
            src1_local = a_hold.alloc(ctx.pool(i), nb1);
            if (!src1_local) {
                fprintf(stderr, "xe2mx row src1 alloc failed dev=%d bytes=%zu\n", i, nb1);
                fflush(stderr);
                std::exit(1);
            }
            queue_ptr mq_src = ctx.stream(main_dev, 0);
            dev2dev_memcpy(i, *sq, main_dev, *mq_src, src1_local, src1->data, nb1);
        }

        ggml_sycl_pool_alloc<float> y_hold(ctx.pool(i));
        float * y_local = y_hold.alloc((size_t) nr * (size_t) nout);
        if (!y_local) {
            fprintf(stderr, "xe2mx row y alloc failed dev=%d floats=%zu\n", i, (size_t) nr * (size_t) nout);
            fflush(stderr);
            std::exit(1);
        }

        ggml_tensor w = *src0;
        w.ne[1] = nr;
        w.data = extra->data_device[i];
        w.buffer = nullptr;
        w.view_src = nullptr;

        ggml_tensor a = *src1;
        a.data = src1_local;
        a.buffer = nullptr;
        a.view_src = nullptr;

        ggml_tensor y = *dst;
        y.ne[0] = nr;
        y.nb[0] = sizeof(float);
        y.nb[1] = y.nb[0] * (size_t) nr;
        y.nb[2] = y.nb[1] * (size_t) y.ne[1];
        y.nb[3] = y.nb[2] * (size_t) y.ne[2];
        y.data = y_local;
        y.buffer = nullptr;
        y.view_src = nullptr;

        static int nlog = 0;
        if (nlog < 8) {
            nlog++;
            GGML_SYCL_XE2_LOG( "xe2mx row dev=%d rows=%ld-%ld N=%ld M=%ld\n",
                    i, (long) row_low, (long) row_high, (long) Nfull, (long) ggml_nrows(src1));
            fflush(stderr);
        }

        ggml_sycl_mul_mat_xe2mx(ctx, &w, &a, &y);
        sq->wait();

        ggml_sycl_set_device(main_dev);
        queue_ptr mq = ctx.stream(main_dev, 0);
        ggml_sycl_pool_alloc<float> main_hold;
        float * src_sc = y_local;
        if (i != main_dev) {
            const size_t nby = (size_t) nr * (size_t) nout;
            src_sc = main_hold.alloc(ctx.pool(main_dev), nby);
            if (!src_sc) {
                fprintf(stderr, "xe2mx row y back alloc failed floats=%zu\n", nby);
                fflush(stderr);
                std::exit(1);
            }
            dev2dev_memcpy(main_dev, *mq, i, *sq, src_sc, y_local, nby * sizeof(float));
        }
        float * out = (float *) dst->data;
        const int64_t row0 = row_low;
        const int64_t nrc = nr;
        const int64_t nf = Nfull;
        const int64_t nc = nout;
        mq->parallel_for(sycl::range<1>((size_t) nrc * (size_t) nc), [=](sycl::id<1> id) {
            const size_t t = (size_t) id[0];
            const size_t r = t % (size_t) nrc;
            const size_t c = t / (size_t) nrc;
            out[(size_t) row0 + r + c * (size_t) nf] = src_sc[r + c * (size_t) nrc];
        }).wait();
    }

    ctx.device = main_dev;
    ctx.ovl_q = saved_ovl;
    ggml_sycl_set_device(main_dev);
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

    if (src0->type == GGML_TYPE_XE2MX4) {
        if (ggml_backend_buffer_is_sycl_split(src0->buffer)) {
            ggml_sycl_mul_mat_xe2mx_split(ctx, src0, src1, dst);
        } else {
            ggml_sycl_mul_mat_xe2mx(ctx, src0, src1, dst);
        }
        return;
    }

    if (src0->type == GGML_TYPE_Q8_0 &&
        (ggml_sycl_get_env("GGML_SYCL_Q8_XMX", 0) || ggml_sycl_get_env("GGML_SYCL_Q8_DNNL", 0))) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        if (ggml_sycl_q8_xmx(ctx, src0, src1, dst)) {
            return;
        }
    }

    if (ggml_is_exl3(src0->type)) {
        GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
        ggml_sycl_mul_mat_exl3(ctx, src0, src1, dst);
        return;
    }

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    const bool q4k_esimd_q8 = src0->type == GGML_TYPE_Q4_K && src1->ne[1] <= GGML_SYCL_IQ4_ESIMD_MAX_NCOLS &&
                              src1->ne[2] == 1 && src1->ne[3] == 1 && !split && ggml_sycl_q4k_esimd_available() &&
                              !g_ggml_sycl_prioritize_dmmv && should_reorder_tensor(ctx, dst) && use_mul_mat_vec_q;
    if (q4k_esimd_q8) {
        use_dequantize_mul_mat_vec = false;
    }
    const bool q5k_esimd_q8 = src0->type == GGML_TYPE_Q5_K && src1->ne[1] <= GGML_SYCL_IQ4_ESIMD_MAX_NCOLS &&
                              src1->ne[2] == 1 && src1->ne[3] == 1 && !split && ggml_sycl_q5k_esimd_available() &&
                              !g_ggml_sycl_prioritize_dmmv && should_reorder_tensor(ctx, dst) && use_mul_mat_vec_q;
    if (q5k_esimd_q8) {
        use_dequantize_mul_mat_vec = false;
    }

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            if (src0->type == GGML_TYPE_IQ4_XS && src1->ne[1] <= GGML_SYCL_IQ4_ESIMD_MAX_NCOLS &&
                src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_sycl_iq4_esimd_available()) {
                ctx.iq4_y_pairs = true;
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa_iq4pair>(ctx, src0, src1, dst,
                                                                           ggml_sycl_op_mul_mat_vec_q);
                ctx.iq4_y_pairs = false;
            } else if (q5k_esimd_q8) {
                ctx.q5k_y_slots = true;
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa_q5k_slots>(ctx, src0, src1, dst,
                                                                             ggml_sycl_op_mul_mat_vec_q);
                ctx.q5k_y_slots = false;
            } else if (q4k_esimd_q8) {
                ctx.q4k_y_slots = true;
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa_q5k_slots>(ctx, src0, src1, dst,
                                                                             ggml_sycl_op_mul_mat_vec_q);
                ctx.q4k_y_slots = false;
            } else {
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
            }
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
static bool ggml_sycl_mul_mat_glu_mmvq_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return false;
    }

    ggml_tensor *       glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    if (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        return false;
    }

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char *                     src1_ddq = src1_q8_alloc.get();

    if (wu->type == GGML_TYPE_Q4_K && ggml_sycl_q4k_esimd_available()) {
        const int glu_esimd = ggml_sycl_get_env("GGML_SYCL_Q4K_ESIMD_GLU", -1);
        if (glu_esimd == 0 || ne11 > GGML_SYCL_IQ4_ESIMD_MAX_NCOLS || wg->type != GGML_TYPE_Q4_K ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
            return false;
        }
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa_q5k_slots>(
            (const float *) act->data, src1_ddq, (int) ne00, (int) ne11, src1_padded_cols, stream);
        return ggml_sycl_mul_mat_vec_q4_K_q8_1_glu_esimd(
            ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq, (float *) glu->data, (int) ne00, (int) wu->ne[1],
            (int) ne11, src1_padded_cols * (int) sizeof(block_q8_1) / QK8_1, (int) glu->ne[0], stream);
    }

    if (wu->type == GGML_TYPE_IQ4_XS) {
        if (ne11 > GGML_SYCL_IQ4_ESIMD_MAX_NCOLS || wg->type != GGML_TYPE_IQ4_XS ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || !ggml_sycl_iq4_esimd_available()) {
            return false;
        }
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa_iq4pair>((const float *) act->data, src1_ddq, (int) ne00,
                                                                      (int) ne11, src1_padded_cols, stream);
        return ggml_sycl_mul_mat_vec_iq4_xs_q8_1_glu_esimd(
            ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq, (float *) glu->data, (int) ne00, (int) wu->ne[1],
            (int) ne11, src1_padded_cols * (int) sizeof(block_q8_1) / QK8_1, (int) glu->ne[0], stream);
    }

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (int) sizeof(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(int) glu->ne[0], stream);
}

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static bool ggml_sycl_xe2mx_pack_moe_soa(ggml_backend_sycl_context & ctx, const ggml_tensor * w) {
    if (!w || w->type != GGML_TYPE_XE2MX4 || !w->data) {
        return false;
    }
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) w->extra;
    if (!extra) {
        return false;
    }
    if (extra->optimized_feature.reorder) {
        return true;
    }
    const int64_t K = w->ne[0], N = w->ne[1], E = w->ne[2];
    if ((K % 64) != 0 || (N % 16) != 0 || E <= 1) {
        return false;
    }
    const int64_t nb = K / 32;
    const size_t pitch = (size_t) nb * 16;
    const size_t we = w->nb[2], wr = w->nb[1];
    if (we != (size_t) N * pitch + (size_t) nb * (size_t) N * sizeof(uint16_t)) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<uint8_t> tmp(ctx.pool(), we);
    uint8_t * tmp_e = tmp.get();
    uint8_t * wdata = (uint8_t *) w->data;
    for (int64_t e = 0; e < E; ++e) {
        uint8_t * we_ptr = wdata + (size_t) e * we;
        stream->memcpy(tmp_e, we_ptr, we);
        stream->parallel_for(sycl::range<1>((size_t) N * (size_t) nb * 4), [=](sycl::id<1> it) {
            const int64_t idx = (int64_t) it[0];
            const int64_t j4  = idx & 3;
            const int64_t b   = idx >> 2;
            const int64_t kb  = b % nb;
            const int64_t row = b / nb;
            const uint8_t * blk = tmp_e + (size_t) row * wr + (size_t) kb * 18;
            if (j4 == 0) {
                uint16_t * d16 = (uint16_t *)(we_ptr + (size_t) N * pitch);
                uint16_t dval;
                memcpy(&dval, blk, sizeof(uint16_t));
                d16[(size_t) kb * (size_t) N + (size_t) row] = dval;
            }
            uint32_t qv;
            memcpy(&qv, blk + 2 + j4 * 4, sizeof(uint32_t));
            uint32_t * qs32 = (uint32_t *)(we_ptr + (size_t) row * pitch + (size_t) kb * 16);
            qs32[j4] = qv;
        });
    }
    stream->wait();
    extra->optimized_feature.reorder = true;
    return true;
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;

    if (src0->type == GGML_TYPE_IQ4_XS) {
        const int env = ggml_sycl_get_env("GGML_SYCL_MOE_IQ", -1);
        if (env == 0) {
            return false;
        }
        if (env < 0 && !ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch)) {
            return false;
        }
    }

    const int64_t n_ids_per_group = ids->ne[0];
    const int64_t n_tokens        = ids->ne[1];
    if (n_tokens < 1 || n_tokens != ne12) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    const int ntok_max = []() {
        const int v = ggml_sycl_get_env("GGML_SYCL_MOE_IQ_NTOK", 8);
        return v > 0 ? v : 8;
    }();
    if (n_tokens > ntok_max) return false;

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];
    const int       n_src1_rows      = (int) (ne11 * n_tokens);

    if (src0->type == GGML_TYPE_XE2MX4) {
        ggml_sycl_xe2mx_pack_moe_soa(ctx, src0);
    } else {
        opt_for_reorder_id(&ctx, src0);
    }
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    const size_t bytes_per_qrow  = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;
    const size_t src1_tok_stride = (size_t) ne11 * bytes_per_qrow;
    const size_t q8_bytes        = (size_t) n_src1_rows * bytes_per_qrow;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool());
    char * src1_ddq = nullptr;
    if (ctx.moe_q8 && ctx.moe_q8_src == src1->data && ctx.moe_q8_nrows == n_src1_rows &&
        ctx.moe_q8_padded == src1_padded_cols && ctx.moe_q8_soa == (int) use_reorder) {
        src1_ddq = ctx.moe_q8;
    } else {
        if (ctx.moe_q8_cap < q8_bytes) {
            if (ctx.moe_q8) {
                ggml_sycl_free_device(ctx.moe_q8, *stream);
            }
            const size_t alloc_bytes = std::max(q8_bytes, (size_t) 64 * bytes_per_qrow);
            ctx.moe_q8     = (char *) ggml_sycl_malloc_device(alloc_bytes, *stream);
            ctx.moe_q8_cap = ctx.moe_q8 ? alloc_bytes : 0;
        }
        src1_ddq = ctx.moe_q8;
        if (!src1_ddq) {
            src1_ddq = src1_q8_alloc.alloc(q8_bytes);
        }
        if (use_reorder) {
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                (const float *) src1->data, src1_ddq, (int) ne10, n_src1_rows,
                src1_padded_cols, stream);
        } else {
            quantize_row_q8_1_sycl<quantize_q8_1>(
                (const float *) src1->data, src1_ddq, (int) ne10, n_src1_rows,
                src1_padded_cols, stream);
        }
        if (src1_ddq == ctx.moe_q8) {
            ctx.moe_q8_src    = src1->data;
            ctx.moe_q8_nrows  = n_src1_rows;
            ctx.moe_q8_padded = src1_padded_cols;
            ctx.moe_q8_soa    = (int) use_reorder;
        }
    }

    bool ok;
    if (use_reorder) {
        ok = ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data, (float *) dst->data,
            (int) ne10, nrows, n_experts_used, src0->nb[2], dst->nb[1], src1_row_stride, stream,
            (int) n_tokens, ids->nb[1], dst->nb[2], src1_tok_stride);
    } else {
        ok = ggml_sycl_mul_mat_vec_q_id(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data, (float *) dst->data,
            (int) ne10, nrows, n_experts_used, src0->nb[2], dst->nb[1], src1_row_stride, stream,
            (int) n_tokens, ids->nb[1], dst->nb[2], src1_tok_stride);
    }
    return ok;
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    for (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

// Expert stacks use either serialized block layout or in-place per-expert SOA layout (reorder == true).
// One subgroup computes one output row; ids remain on the Intel device.
static void ggml_sycl_mul_mat_id_xe2mx_native(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    GGML_ASSERT(w->type == GGML_TYPE_XE2MX4 && x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && w->ne[3] == 1 && x->ne[3] == 1);
    GGML_ASSERT(w->ne[0] == x->ne[0] && w->ne[0] % QK_XE2MX4 == 0);
    GGML_ASSERT(x->ne[1] == 1 || x->ne[1] == ids->ne[0]);
    GGML_ASSERT(x->ne[2] == ids->ne[1] && dst->ne[1] == ids->ne[0] && dst->ne[2] == ids->ne[1]);
    const ggml_tensor_extra_gpu * extra = (const ggml_tensor_extra_gpu *) w->extra;
    const bool is_soa = extra && extra->optimized_feature.reorder;
    const char * wp = (const char *) w->data;
    const char * xp = (const char *) x->data;
    const char * ip = (const char *) ids->data;
    char * yp = (char *) dst->data;
    const int64_t kblocks = w->ne[0] / QK_XE2MX4;
    const int64_t nr = w->ne[1], selected = ids->ne[0], nt = ids->ne[1], xc = x->ne[1];
    const size_t pitch = (size_t) kblocks * 16;
    const size_t wr = w->nb[1], we = w->nb[2];
    const size_t x0 = x->nb[0], x1 = x->nb[1], x2 = x->nb[2];
    const size_t i0 = ids->nb[0], i1 = ids->nb[1];
    const size_t y0 = dst->nb[0], y1 = dst->nb[1], y2 = dst->nb[2];
    const sycl::range<3> groups((size_t) nt, (size_t) selected, (size_t) nr);
    const sycl::range<3> local(1, 1, WARP_SIZE);
    ctx.stream()->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(groups * local, local),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t tok = it.get_group(0), slot = it.get_group(1), row = it.get_group(2);
                const int32_t e = *(const int32_t *) (ip + tok * i1 + slot * i0);
                const char * xv = xp + tok * x2 + (slot % xc) * x1;
                float sum = 0.0f;
                const int lane = it.get_local_id(2);
                if (is_soa) {
                    const uint8_t * ep = (const uint8_t *)(wp + (size_t) e * we);
                    const uint8_t * qrow = ep + (size_t) row * pitch;
                    const sycl::half * dpl = (const sycl::half *)(ep + (size_t) nr * pitch);
                    for (int64_t b = lane; b < kblocks; b += WARP_SIZE) {
                        const float d = (float) dpl[(size_t) b * (size_t) nr + (size_t) row];
                        const uint8_t * qs = qrow + (size_t) b * 16;
                        float acc = 0.0f;
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t q = qs[j];
                            const float a = *(const float *) (xv + (b * 32 + 2 * j) * x0);
                            const float z = *(const float *) (xv + (b * 32 + 2 * j + 1) * x0);
                            acc += ((int) (q & 15) - 8) * a + ((int) (q >> 4) - 8) * z;
                        }
                        sum += d * acc;
                    }
                } else {
                    const block_xe2mx4 * wb = (const block_xe2mx4 *) (wp + (size_t) e * we + row * wr);
                    for (int64_t b = lane; b < kblocks; b += WARP_SIZE) {
                        const float d = (float) wb[b].d;
                        float acc = 0.0f;
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t q = wb[b].qs[j];
                            const float a = *(const float *) (xv + (b * 32 + 2 * j) * x0);
                            const float z = *(const float *) (xv + (b * 32 + 2 * j + 1) * x0);
                            acc += ((int) (q & 15) - 8) * a + ((int) (q >> 4) - 8) * z;
                        }
                        sum += d * acc;
                    }
                }
                sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                if (lane == 0) {
                    *(float *) (yp + tok * y2 + slot * y1 + row * y0) = sum;
                }
            });
    });
}

struct Xe2mxRoutingCache {
    sycl::queue * q = nullptr;
    const void * ids_data = nullptr;
    int64_t E = 0;
    int64_t routed = 0;
    int64_t group = 0;
    int64_t total_padded_rows = 0;
    int64_t max_group_rows = 0;
    int64_t max_batch = 1;
    std::vector<int64_t> counts;
    std::vector<int64_t> offsets;
    std::vector<int64_t> order;
    std::vector<mmid_row_mapping> mapping;
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> groups; // (base, batch, rows, row_start)
    int32_t * d_ri1 = nullptr;
    int32_t * d_ri2 = nullptr;
    int64_t * d_ep  = nullptr;
    size_t d_rows_cap = 0;
    size_t d_ep_cap   = 0;
};

// GGML_SYCL_XE2MX_MOE_DEVROUTE=1 routes XE2MX4 MoE prefill fully on the device
// (atomic counting sort + tile list built by the GPU), so the op never waits on the host.
struct Xe2mxDevRoute {
    sycl::queue * q = nullptr;
    const void * ids_data = nullptr;
    int64_t E = 0, routed = 0;
    int32_t * counts = nullptr;   // [E]
    int32_t * offs   = nullptr;   // [E]
    int32_t * cursor = nullptr;   // [E]
    int32_t * tiles  = nullptr;   // [4*max_tiles]
    int32_t * ntiles = nullptr;   // [1]
    int32_t * ri1 = nullptr;
    int32_t * ri2 = nullptr;
    size_t cap_rows = 0, cap_E = 0, cap_T = 0;
    int64_t span = 0;
};

static bool ggml_sycl_mul_mat_id_xe2mx_devroute(ggml_backend_sycl_context & ctx, ggml_tensor * dst, int dev_id) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int64_t K = w->ne[0], N = w->ne[1], E = w->ne[2];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    const int64_t routed = n_used * n_tok, channels = x->ne[1];
    const bool is_iq2 = w->type == GGML_TYPE_IQ2_XXS;
    if ((w->type != GGML_TYPE_XE2MX4 && !is_iq2) || !g_ggml_sycl_enable_esimd || (K % 64) != 0 || K < 128 ||
        (N % 16) != 0 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (is_iq2) {
        // only the keeper IQ2 kernel (DPAS + DIRECT + SLM=48) has a tiled variant
        if (!ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DPAS", 0) || !ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DIRECT", 0) ||
            ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_SLM", 0) != 48 || (K % 256) != 0 || (N % 64) != 0) {
            return false;
        }
    } else if (!ggml_sycl_xe2mx_pack_moe_soa(ctx, w)) {
        return false;
    }
    const queue_ptr stream = ctx.stream();
    const int64_t span = is_iq2 ? ggml_sycl_iq2_moe_tile_span() : ggml_sycl_xe2mx_moe_tile_span();
    const int64_t ub_rows = routed + E * 7;                  // each expert padded to 8 rows
    const int64_t max_tiles = (ub_rows + span - 1) / span + E;

    static Xe2mxDevRoute st[GGML_SYCL_MAX_DEVICES];
    Xe2mxDevRoute & r = st[dev_id];
    const bool hit = strstr(w->name, "ffn_down_exps") && r.q == stream && r.ids_data == ids->data &&
                     r.E == E && r.routed == routed && r.span == span;
    if (!hit) {
        auto grow = [&](int32_t *& ptr, size_t & cap, size_t need) {
            if (cap < need) {
                if (ptr) ggml_sycl_free_device(ptr, *stream);
                const size_t c = std::max(need, (size_t) 256);
                ptr = (int32_t *) ggml_sycl_malloc_device(c * sizeof(int32_t), *stream);
                cap = c;
            }
        };
        if (r.cap_rows < (size_t) ub_rows) {
            size_t c1 = r.cap_rows, c2 = r.cap_rows;
            grow(r.ri1, c1, ub_rows);
            grow(r.ri2, c2, ub_rows);
            r.cap_rows = c1;
        }
        if (r.cap_E < (size_t) E) {
            size_t a = r.cap_E, b = r.cap_E, c = r.cap_E;
            grow(r.counts, a, E); grow(r.offs, b, E); grow(r.cursor, c, E);
            r.cap_E = a;
        }
        size_t ct = r.cap_T;
        grow(r.tiles, ct, (size_t) 4 * max_tiles);
        r.cap_T = ct;
        if (!r.ntiles) r.ntiles = (int32_t *) ggml_sycl_malloc_device(sizeof(int32_t) * 4, *stream);

        int32_t * counts = r.counts; int32_t * offs = r.offs; int32_t * cursor = r.cursor;
        int32_t * tiles = r.tiles; int32_t * ntiles = r.ntiles; int32_t * ri1 = r.ri1; int32_t * ri2 = r.ri2;
        const char * idp = (const char *) ids->data;
        const size_t inb0 = ids->nb[0], inb1 = ids->nb[1];
        const int32_t sp = (int32_t) span;
        stream->memset(counts, 0, E * sizeof(int32_t));
        stream->memset(ri1, 0xff, ub_rows * sizeof(int32_t));
        stream->parallel_for(sycl::range<1>((size_t) routed), [=](sycl::id<1> it) {
            const int64_t i = it[0];
            const int64_t tok = i / n_used, slot = i % n_used;
            const int32_t e = *(const int32_t *) (idp + tok * inb1 + slot * inb0);
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> a(counts[e]);
            a.fetch_add(1);
        });
        stream->single_task([=]() {
            int32_t off = 0, t = 0;
            for (int32_t e = 0; e < (int32_t) E; ++e) {
                const int32_t c = counts[e];
                const int32_t padded = (c + 7) & ~7;
                offs[e] = off;
                cursor[e] = 0;
                for (int32_t m = 0; m < padded; m += sp) {
                    tiles[4 * t + 0] = e;
                    tiles[4 * t + 1] = off;
                    tiles[4 * t + 2] = padded;
                    tiles[4 * t + 3] = m;
                    ++t;
                }
                off += padded;
            }
            ntiles[0] = t;
        });
        stream->parallel_for(sycl::range<1>((size_t) routed), [=](sycl::id<1> it) {
            const int64_t i = it[0];
            const int64_t tok = i / n_used, slot = i % n_used;
            const int32_t e = *(const int32_t *) (idp + tok * inb1 + slot * inb0);
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> a(cursor[e]);
            const int32_t pos = offs[e] + a.fetch_add(1);
            ri1[pos] = (int32_t) slot;
            ri2[pos] = (int32_t) tok;
        });
        r.q = stream; r.ids_data = ids->data; r.E = E; r.routed = routed; r.span = span;
    }

    const int64_t nb = K / 32;
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool(), (size_t) ub_rows * (size_t) K);
    ggml_sycl_pool_alloc<float> Ad(ctx.pool(), (size_t) ub_rows * (size_t) nb);
    // GGML_SYCL_MOE_FUSE_SCATTER=1 writes GEMM rows straight to dst (no temp buffer, no scatter kernel)
    static const bool fuse_sc = ggml_sycl_get_env("GGML_SYCL_MOE_FUSE_SCATTER", 0) != 0;
    // fuse only when the tile variant the dispatcher will pick actually stores rows to dst
    const bool fused = fuse_sc && dst->nb[0] == sizeof(float) &&
                       (is_iq2 ? ggml_sycl_iq2_moe_tiles_can_fuse((int) N)
                               : ggml_sycl_xe2mx_moe_tiles_can_fuse((int) N));
    ggml_sycl_pool_alloc<float> result(ctx.pool(), fused ? 1 : (size_t) ub_rows * (size_t) N);
    int8_t * A = Aq.get(); float * da = Ad.get(); float * yp = result.get();
    const int32_t * ri1 = r.ri1; const int32_t * ri2 = r.ri2;
    const char * xin = (const char *) x->data;
    char * yout = (char *) dst->data;
    const size_t x0 = x->nb[0], x1 = x->nb[1], x2 = x->nb[2];
    const size_t y0 = dst->nb[0], y1 = dst->nb[1], y2 = dst->nb[2];

    stream->parallel_for(sycl::range<1>((size_t) ub_rows * (size_t) nb), [=](sycl::id<1> it) {
        const int64_t idx = (int64_t) it[0];
        const int64_t pr = idx / nb;
        const int64_t ib = idx % nb;
        int8_t * out = A + (size_t) pr * (size_t) K + (size_t) ib * 32;
        const int32_t i1 = ri1[pr];
        if (i1 < 0) {
            da[idx] = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) out[k] = 0;
            return;
        }
        const int32_t i2 = ri2[pr];
        const char * row_ptr = xin + (size_t)(i1 % channels) * x1 + (size_t) i2 * x2 + (size_t) ib * 32 * x0;
        float af[32];
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float val = *(const float *)(row_ptr + (size_t) k * x0);
            af[k] = val;
            amax = sycl::fmax(amax, sycl::fabs(val));
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        da[idx] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float xq = af[k] * id;
            int q = (int)(xq + (xq >= 0.0f ? 0.5f : -0.5f));
            if (q < -127) q = -127;
            if (q > 127)  q = 127;
            out[k] = (int8_t) q;
        }
    });
    bool launched;
    if (is_iq2) {
        static uint32_t * grid_dev[GGML_SYCL_MAX_DEVICES] = {};
        uint32_t *& g32 = grid_dev[dev_id];
        if (!g32) {
            g32 = (uint32_t *) ggml_sycl_malloc_device(256 * sizeof(uint64_t), *stream);
            stream->memcpy(g32, iq2xxs_grid, 256 * sizeof(uint64_t)).wait();   // once per device
        }
        launched = ggml_sycl_iq2_moe_tiles(ctx, (const uint8_t *) w->data, w->nb[2], w->nb[1], g32, r.tiles, r.ntiles,
                                           A, da, yp, (int) K, (int) N, (int) max_tiles,
                                           fused ? ri1 : nullptr, ri2, fused ? yout : nullptr, y1, y2);
    } else {
        launched = ggml_sycl_xe2mx_moe_tiles(ctx, (const uint8_t *) w->data, r.tiles, r.ntiles, A, da, yp,
                                             (int) K, (int) N, (int) max_tiles,
                                             fused ? ri1 : nullptr, ri2, fused ? yout : nullptr, y1, y2);
    }
    if (!launched) {
        GGML_ABORT("xe2mx devroute tiles launch failed");
    }
    if (!fused) {
    stream->parallel_for(sycl::range<1>((size_t) ub_rows * (size_t) N), [=](sycl::id<1> it) {
        const int64_t flat = it[0];
        const int64_t n = flat % N;
        const int64_t pr = flat / N;
        const int32_t i1 = ri1[pr];
        if (i1 >= 0) {
            const int32_t i2 = ri2[pr];
            *(float *)(yout + (size_t) i1 * y1 + (size_t) i2 * y2 + (size_t) n * y0) = yp[flat];
        }
    });
    }
    return true;
}

// Bounded expert-local reconstruction plus Intel matrix GEMM, without a whole-model FP16 copy.
// Routing is counted once per operation; all gathers, reconstruction, GEMMs and scatters run on Intel.
static void ggml_sycl_mul_mat_id_xe2mx_prefill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const int64_t K=w->ne[0], N=w->ne[1], E=w->ne[2];
    const int64_t routed=ids->ne[0]*ids->ne[1], channels=x->ne[1];
    const queue_ptr stream=ctx.stream();
    const int64_t group=std::max(1,std::min(32,ggml_sycl_get_env("GGML_SYCL_XE2MX_MOE_GROUP",16)));

    // per-device: two prefill slots on different cards must not serialize on one global mutex
    static std::mutex rcache_mu[GGML_SYCL_MAX_DEVICES];
    static Xe2mxRoutingCache rcache[GGML_SYCL_MAX_DEVICES];
    const int dev_id = std::max(0, std::min(GGML_SYCL_MAX_DEVICES - 1, ctx.device));

    std::lock_guard<std::mutex> lk(rcache_mu[dev_id]);
    static const bool devroute = ggml_sycl_get_env("GGML_SYCL_XE2MX_MOE_DEVROUTE", 0) != 0;
    if (devroute && ggml_sycl_mul_mat_id_xe2mx_devroute(ctx, dst, dev_id)) {
        return;
    }
    Xe2mxRoutingCache & rc = rcache[dev_id];
    const bool hit = (channels > 1 && w && strstr(w->name, "ffn_down_exps") &&
                      rc.q == stream && rc.ids_data == ids->data && rc.E == E &&
                      rc.routed == routed && rc.group == group &&
                      rc.d_ri1 != nullptr && rc.d_ri2 != nullptr && rc.d_ep != nullptr);

    if (!hit) {
        std::vector<char> host_ids(ggml_nbytes(ids));
        stream->memcpy(host_ids.data(),ids->data,host_ids.size()).wait();
        std::vector<int64_t> raw_counts, raw_offsets;
        rc.mapping.clear();
        rc.groups.clear();
        mmid_counting_sort_rows(ids,host_ids.data(),ids->ne[0],E,routed,raw_counts,raw_offsets,rc.mapping);
        rc.order.resize(E);
        rc.counts.resize(E);
        rc.offsets.resize(E);
        for (int64_t e=0;e<E;++e) { rc.order[e]=e; }
        std::sort(rc.order.begin(),rc.order.end(),[&](int64_t a,int64_t b) { return raw_counts[a]>raw_counts[b]; });
        for (int64_t e=0;e<E;++e) {
            rc.counts[e]=raw_counts[rc.order[e]];
            rc.offsets[e]=raw_offsets[rc.order[e]];
        }
        // IQ2 s8 DPAS wants M>=32 (gemm_s8_vnni); XE2MX4 DPAS pads to 8.
        const int tight = w->type == GGML_TYPE_IQ2_XXS && ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DIRECT", 0)
                        ? ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_TIGHT", 0) : 0;
        const int row_align = tight ? 8 : ((w->type == GGML_TYPE_IQ2_XXS && ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DPAS", 0)) ? 32 : 8);
        int64_t total_padded_rows = 0, max_group_rows = 0, max_batch = 1;
        for (int64_t base=0;base<E && rc.counts[base]>0;) {
            const int64_t rows=GGML_PAD(rc.counts[base], row_align);
            int64_t batch=std::min(group,E-base);
            // s8 pack is K*N bytes/expert; fp16 path is 2*K*N. Cap scratch ~256 MiB.
            const int64_t w_bytes = (w->type == GGML_TYPE_IQ2_XXS) ? (K * N) : (K * N * 2);
            while (batch>1 && (batch*rows*(K*2+N*4)+batch*w_bytes)>256*1024*1024) { batch=(batch+1)/2; }
            while (batch>1 && rc.counts[base+batch-1]==0) { --batch; }
            if (tight == 1) {
                while (batch > 1 && GGML_PAD(rc.counts[base + batch - 1], row_align) != rows) --batch;
            } else if (tight == 2) {
                while (batch > 1 && rc.counts[base + batch - 1] * 10 < rows * 9) --batch;
            }
            rc.groups.emplace_back(base, batch, rows, total_padded_rows);
            total_padded_rows += batch * rows;
            max_group_rows = std::max(max_group_rows, batch * rows);
            max_batch = std::max(max_batch, batch);
            base += batch;
        }
        rc.q = stream;
        rc.ids_data = ids->data;
        rc.E = E;
        rc.routed = routed;
        rc.group = group;
        rc.total_padded_rows = total_padded_rows;
        rc.max_group_rows = max_group_rows;
        rc.max_batch = max_batch;

        if (!rc.groups.empty()) {
            std::vector<int32_t> host_i1(total_padded_rows, -1);
            std::vector<int32_t> host_i2(total_padded_rows, -1);
            for (const auto & g : rc.groups) {
                const int64_t base = std::get<0>(g);
                const int64_t batch = std::get<1>(g);
                const int64_t rows = std::get<2>(g);
                const int64_t row_start = std::get<3>(g);
                for (int64_t e = 0; e < batch; ++e) {
                    const int64_t cnt = rc.counts[base + e];
                    const int64_t off = rc.offsets[base + e];
                    const int64_t dst_row0 = row_start + e * rows;
                    for (int64_t r = 0; r < cnt; ++r) {
                        const mmid_row_mapping m = rc.mapping[off + r];
                        host_i1[dst_row0 + r] = m.i1;
                        host_i2[dst_row0 + r] = m.i2;
                    }
                }
            }
            if (rc.d_rows_cap < (size_t) total_padded_rows) {
                if (rc.d_ri1) ggml_sycl_free_device(rc.d_ri1, *stream);
                if (rc.d_ri2) ggml_sycl_free_device(rc.d_ri2, *stream);
                const size_t cap = std::max((size_t) total_padded_rows, (size_t) 65536);
                rc.d_ri1 = (int32_t *) ggml_sycl_malloc_device(cap * sizeof(int32_t), *stream);
                rc.d_ri2 = (int32_t *) ggml_sycl_malloc_device(cap * sizeof(int32_t), *stream);
                rc.d_rows_cap = (rc.d_ri1 && rc.d_ri2) ? cap : 0;
            }
            if (rc.d_ep_cap < (size_t) E) {
                if (rc.d_ep) ggml_sycl_free_device(rc.d_ep, *stream);
                const size_t cap = std::max((size_t) E, (size_t) 256);
                rc.d_ep = (int64_t *) ggml_sycl_malloc_device(cap * sizeof(int64_t), *stream);
                rc.d_ep_cap = rc.d_ep ? cap : 0;
            }
            stream->memcpy(rc.d_ri1, host_i1.data(), total_padded_rows * sizeof(int32_t));
            stream->memcpy(rc.d_ri2, host_i2.data(), total_padded_rows * sizeof(int32_t));
            stream->memcpy(rc.d_ep, rc.order.data(), E * sizeof(int64_t)).wait();
        }
    }

    if (rc.groups.empty()) {
        return;
    }

    const int64_t total_padded_rows = rc.total_padded_rows;
    const int64_t max_batch = rc.max_batch;

    ggml_sycl_pool_alloc<float> result(ctx.pool(), total_padded_rows * N);

    int32_t * ri1 = rc.d_ri1;
    int32_t * ri2 = rc.d_ri2;
    int64_t * ep  = rc.d_ep;
    float * yp    = result.get();

    const char * xin = (const char *) x->data;
    const char * win = (const char *) w->data;
    char * yout = (char *) dst->data;
    const size_t x0 = x->nb[0], x1 = x->nb[1], x2 = x->nb[2], we = w->nb[2], wr = w->nb[1];
    const size_t y0 = dst->nb[0], y1 = dst->nb[1], y2 = dst->nb[2];

    const int64_t Npad = GGML_PAD(N, 64);
    const bool use_dpas = w->type == GGML_TYPE_XE2MX4 &&
                          g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_XE2MX_MOE_DPAS", 1) != 0 &&
                          (K % 64) == 0 && K >= 128;
    // IQ2_XXS codebook -> int8 weights (grid values 8/25/43 with signs) + fp16 per-32 scale,
    // then existing s8x s8 DPAS. Avoids materializing full fp16 expert weights.
    const bool use_iq2_dpas = w->type == GGML_TYPE_IQ2_XXS &&
                              g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DPAS", 0) != 0 &&
                              (K % 64) == 0 && (N % 16) == 0 && K >= 128 && N >= 16;

    if (use_iq2_dpas) {
        const int64_t nb = K / 32;
        // gemm_s8_soa: W8 is [K][pitch] (pitch>=N), d16 is [nb][N]
        ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool(), (size_t) total_padded_rows * (size_t) K);
        ggml_sycl_pool_alloc<float> Ad(ctx.pool(), (size_t) total_padded_rows * (size_t) nb);
        const bool iq2_direct_alloc = ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DIRECT", 0) != 0;
        ggml_sycl_pool_alloc<int8_t> W8p(ctx.pool(), iq2_direct_alloc ? 1 : (size_t) max_batch * (size_t) K * (size_t) N);
        ggml_sycl_pool_alloc<uint16_t> D16p(ctx.pool(), iq2_direct_alloc ? 1 : (size_t) max_batch * (size_t) nb * (size_t) N);

        int8_t * A = Aq.get();
        float * da = Ad.get();
        int8_t * W8 = W8p.get();
        uint16_t * D16 = D16p.get();

        // 1. Gather & quantize activations to int8 (same as XE2MX4 MoE DPAS)
        stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) nb), [=](sycl::id<1> it) {
            const int64_t idx = (int64_t) it[0];
            const int64_t pr = idx / nb;
            const int64_t ib = idx % nb;
            int8_t * out = A + (size_t) pr * (size_t) K + (size_t) ib * 32;
            const int32_t i1 = ri1[pr];
            if (i1 < 0) {
                da[idx] = 0.0f;
#pragma unroll
                for (int k = 0; k < 32; ++k) {
                    out[k] = 0;
                }
                return;
            }
            const int32_t i2 = ri2[pr];
            const char * row_ptr = xin + (size_t)(i1 % channels) * x1 + (size_t) i2 * x2 + (size_t) ib * 32 * x0;
            float af[32];
            float amax = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float val = *(const float *)(row_ptr + (size_t) k * x0);
                af[k] = val;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }
            const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
            da[idx] = s;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float xq = af[k] * id;
                int q = (int)(xq + (xq >= 0.0f ? 0.5f : -0.5f));
                if (q < -127) q = -127;
                if (q > 127)  q = 127;
                out[k] = (int8_t) q;
            }
        });

        // 2a. Direct: IQ2 blocks decoded inside the DPAS kernel, no weight scratch.
        static const bool iq2_direct = ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DIRECT", 0) != 0;
        if (iq2_direct) {
            static uint32_t * grid_dev[GGML_SYCL_MAX_DEVICES] = {};
            uint32_t *& g32 = grid_dev[dev_id];
            if (!g32) {
                g32 = (uint32_t *) ggml_sycl_malloc_device(256 * sizeof(uint64_t), *stream);
                stream->memcpy(g32, iq2xxs_grid, 256 * sizeof(uint64_t)).wait();
            }
            bool ok = true;
            for (const auto & g : rc.groups) {
                const int64_t base      = std::get<0>(g);
                const int64_t batch     = std::get<1>(g);
                const int64_t rows      = std::get<2>(g);
                const int64_t row_start = std::get<3>(g);
                ok = ok && ggml_sycl_iq2_moe_dpas(ctx, (const uint8_t *) win, ep + base, we, wr, g32,
                    A + (size_t) row_start * (size_t) K, da + (size_t) row_start * (size_t) nb,
                    yp + (size_t) row_start * (size_t) N, (int) K, (int) N, (int) rows, (int) batch);
            }
            GGML_ASSERT(ok && "iq2 direct moe dpas launch failed");
        }
        // 2b. IQ2 -> s8 [K][N] + d16 [nb][N]. Tile N by 16 so each store is contiguous.
        for (const auto & g : (iq2_direct ? decltype(rc.groups){} : rc.groups)) {
            const int64_t base      = std::get<0>(g);
            const int64_t batch     = std::get<1>(g);
            const int64_t rows      = std::get<2>(g);
            const int64_t row_start = std::get<3>(g);
            const int64_t ntiles    = N / 16;

            stream->parallel_for(sycl::range<1>((size_t) batch * (size_t) ntiles * (size_t) nb), [=](sycl::id<1> it) {
                const int64_t idx = (int64_t) it[0];
                const int64_t ib  = idx % nb;
                const int64_t nt  = (idx / nb) % ntiles;
                const int64_t e   = idx / (nb * ntiles);
                const int64_t n0  = nt * 16;
                const int kb  = (int) (ib / 8);
                const int ib8 = (int) (ib % 8);
                int8_t tile[32][16];
                uint16_t dtile[16];
#pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const int64_t row = n0 + t;
                    const block_iq2_xxs * q = (const block_iq2_xxs *)(win + ep[base + e] * we + row * wr) + kb;
                    const uint16_t * q2 = q->qs + 4 * ib8;
                    const uint8_t * aux8 = (const uint8_t *) q2;
                    const uint32_t aux32 = q2[2] | ((uint32_t) q2[3] << 16);
                    const float d = (float) q->d * (0.5f + (float) (aux32 >> 28)) * 0.25f;
                    dtile[t] = sycl::bit_cast<uint16_t>((sycl::half) d);
#pragma unroll
                    for (int il = 0; il < 4; ++il) {
                        const int8_t * grid = (const int8_t *)(iq2xxs_grid + aux8[il]);
                        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            int v = (int) grid[j];
                            if (signs & kmask_iq2xs[j]) {
                                v = -v;
                            }
                            tile[il * 8 + j][t] = (int8_t) v;
                        }
                    }
                }
                // d16[ib][n0:n0+16]
                uint16_t * dp = D16 + ((size_t) e * (size_t) nb + (size_t) ib) * (size_t) N + (size_t) n0;
#pragma unroll
                for (int t = 0; t < 16; ++t) {
                    dp[t] = dtile[t];
                }
                // W8[k][n0:n0+16] for k in ib*32.. — contiguous 16-byte stores
                int8_t * wp0 = W8 + ((size_t) e * (size_t) K + (size_t) ib * 32) * (size_t) N + (size_t) n0;
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    int8_t * wp = wp0 + (size_t) j * (size_t) N;
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        wp[t] = tile[j][t];
                    }
                }
            });

            for (int64_t e = 0; e < batch; ++e) {
                const int8_t * We = W8 + (size_t) e * (size_t) K * (size_t) N;
                const uint16_t * De = D16 + (size_t) e * (size_t) nb * (size_t) N;
                const int8_t * Ae = A + (size_t) (row_start + e * rows) * (size_t) K;
                const float * dae = da + (size_t) (row_start + e * rows) * (size_t) nb;
                float * Ce = yp + (size_t) (row_start + e * rows) * (size_t) N;
                if (!ggml_sycl_xe2mx_gemm_s8(ctx, We, De, Ae, dae, Ce, K, N, rows, N)) {
                    static int nfail = 0;
                    if (nfail < 4) {
                        nfail++;
                        GGML_SYCL_XE2_LOG("iq2 moe dpas: gemm_s8 failed N=%ld M=%ld K=%ld\n",
                                          (long) N, (long) rows, (long) K);
                    }
                }
            }
        }

        stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) N), [=](sycl::id<1> it) {
            const int64_t flat = it[0];
            const int64_t n = flat % N;
            const int64_t pr = flat / N;
            const int32_t i1 = ri1[pr];
            if (i1 >= 0) {
                const int32_t i2 = ri2[pr];
                *(float *)(yout + (size_t) i1 * y1 + (size_t) i2 * y2 + (size_t) n * y0) = yp[flat];
            }
        });
        return;
    }

    if (use_dpas) {
        const bool in_place_soa = (N % 16) == 0 && ggml_sycl_xe2mx_pack_moe_soa(ctx, w);
        const int64_t nb = K / 32;
        const size_t pitch = (size_t) nb * 16;
        const int64_t Ndpas = in_place_soa ? N : Npad;
        const size_t expert_w_stride = (size_t) Ndpas * pitch + (size_t) nb * (size_t) Ndpas * sizeof(uint16_t);
        ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool(), (size_t) total_padded_rows * (size_t) K);
        ggml_sycl_pool_alloc<float> Ad(ctx.pool(), (size_t) total_padded_rows * (size_t) nb);
        ggml_sycl_pool_alloc<uint8_t> Wsoa(ctx.pool(), in_place_soa ? 1 : (size_t) max_batch * expert_w_stride);
        ggml_sycl_pool_alloc<float> result_pad(ctx.pool(), (Ndpas > N) ? (size_t) total_padded_rows * (size_t) Ndpas : 1);

        int8_t * A = Aq.get();
        float * da = Ad.get();
        uint8_t * wp_soa = in_place_soa ? (uint8_t *) w->data : Wsoa.get();
        float * yp_dpas = (Ndpas > N) ? result_pad.get() : yp;

        // 1. Gather & quantize all padded rows across all groups into int8 A + float da
        stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) nb), [=](sycl::id<1> it) {
            const int64_t idx = (int64_t) it[0];
            const int64_t pr = idx / nb;
            const int64_t ib = idx % nb;
            int8_t * out = A + (size_t) pr * (size_t) K + (size_t) ib * 32;
            const int32_t i1 = ri1[pr];
            if (i1 < 0) {
                da[idx] = 0.0f;
#pragma unroll
                for (int k = 0; k < 32; ++k) {
                    out[k] = 0;
                }
                return;
            }
            const int32_t i2 = ri2[pr];
            const char * row_ptr = xin + (size_t)(i1 % channels) * x1 + (size_t) i2 * x2 + (size_t) ib * 32 * x0;
            float af[32];
            float amax = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float val = *(const float *)(row_ptr + (size_t) k * x0);
                af[k] = val;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }
            const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
            da[idx] = s;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float xq = af[k] * id;
                int q = (int)(xq + (xq >= 0.0f ? 0.5f : -0.5f));
                if (q < -127) q = -127;
                if (q > 127)  q = 127;
                out[k] = (int8_t) q;
            }
        });

        // 2. Per-group ESIMD s4 x s8 DPAS (zero weight repack when in_place_soa is active)
        for (const auto & g : rc.groups) {
            const int64_t base      = std::get<0>(g);
            const int64_t batch     = std::get<1>(g);
            const int64_t rows      = std::get<2>(g);
            const int64_t row_start = std::get<3>(g);

            if (!in_place_soa) {
                stream->parallel_for(sycl::range<1>((size_t) batch * (size_t) Ndpas * (size_t) nb * 4), [=](sycl::id<1> it) {
                    const int64_t idx = (int64_t) it[0];
                    const int64_t j4  = idx & 3;
                    const int64_t b   = idx >> 2;
                    const int64_t kb  = b % nb;
                    const int64_t row = (b / nb) % Ndpas;
                    const int64_t e   = b / (nb * Ndpas);
                    uint8_t * dst_e = wp_soa + (size_t) e * expert_w_stride;
                    if (row >= N) {
                        if (j4 == 0) {
                            uint16_t * d16 = (uint16_t *)(dst_e + (size_t) Ndpas * pitch);
                            d16[(size_t) kb * (size_t) Ndpas + (size_t) row] = 0;
                        }
                        uint32_t * qs32 = (uint32_t *)(dst_e + (size_t) row * pitch + (size_t) kb * 16);
                        qs32[j4] = 0u;
                        return;
                    }
                    const uint8_t * blk = (const uint8_t *)(win + ep[base + e] * we + row * wr) + kb * 18;
                    if (j4 == 0) {
                        uint16_t * d16 = (uint16_t *)(dst_e + (size_t) Ndpas * pitch);
                        uint16_t dval;
                        memcpy(&dval, blk, sizeof(uint16_t));
                        d16[(size_t) kb * (size_t) Ndpas + (size_t) row] = dval;
                    }
                    uint32_t qv;
                    memcpy(&qv, blk + 2 + j4 * 4, sizeof(uint32_t));
                    uint32_t * qs32 = (uint32_t *)(dst_e + (size_t) row * pitch + (size_t) kb * 16);
                    qs32[j4] = qv;
                });
            }

            ggml_sycl_xe2mx_moe_dpas(ctx,
                wp_soa,
                in_place_soa ? (ep + base) : nullptr,
                A + (size_t) row_start * (size_t) K,
                da + (size_t) row_start * (size_t) nb,
                yp_dpas + (size_t) row_start * (size_t) Ndpas,
                (int) K, (int) Ndpas, (int) rows, (int) batch);
        }

        if (Ndpas > N) {
            stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) N), [=](sycl::id<1> it) {
                const int64_t flat = it[0];
                const int64_t n = flat % N;
                const int64_t pr = flat / N;
                const int32_t i1 = ri1[pr];
                if (i1 >= 0) {
                    const int32_t i2 = ri2[pr];
                    *(float *)(yout + (size_t) i1 * y1 + (size_t) i2 * y2 + (size_t) n * y0) = yp_dpas[pr * Ndpas + n];
                }
            });
            return;
        }
    } else {
        ggml_sycl_pool_alloc<sycl::half> gathered(ctx.pool(), total_padded_rows * K);
        ggml_sycl_pool_alloc<sycl::half> weights(ctx.pool(), max_batch * K * N);
        sycl::half * xp = gathered.get();
        sycl::half * wp = weights.get();

        // 1. Single kernel launch to gather all padded rows across all groups into xp
        stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) K), [=](sycl::id<1> it) {
            const int64_t flat = it[0];
            const int64_t k = flat % K;
            const int64_t pr = flat / K;
            const int32_t i1 = ri1[pr];
            sycl::half v = 0;
            if (i1 >= 0) {
                const int32_t i2 = ri2[pr];
                v = (sycl::half) *(const float *)(xin + (i1 % channels) * x1 + (size_t) i2 * x2 + (size_t) k * x0);
            }
            xp[flat] = v;
        });

        // 2. Per-group expert -> FP16 dequant + cached oneDNN batched GEMM
        sycl::half2 * wp2 = reinterpret_cast<sycl::half2 *>(wp);
        const bool iq2 = w->type == GGML_TYPE_IQ2_XXS;
        for (const auto & g : rc.groups) {
            const int64_t base      = std::get<0>(g);
            const int64_t batch     = std::get<1>(g);
            const int64_t rows      = std::get<2>(g);
            const int64_t row_start = std::get<3>(g);

            if (iq2) {
                // one 32-lane work-group per 256-weight block, same math as dequantize_block_iq2_xxs
                const int64_t nbk = K / QK_K;
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t) batch * (size_t) N * (size_t) nbk * 32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> it) {
                        const int64_t b   = (int64_t) it.get_group(0);
                        const int     tid = (int) it.get_local_id(0);
                        const int64_t kb  = b % nbk;
                        const int64_t row = (b / nbk) % N;
                        const int64_t e   = b / (nbk * N);
                        const block_iq2_xxs * q = (const block_iq2_xxs *)(win + ep[base + e] * we + row * wr) + kb;
                        const int il = tid / 8;
                        const int ib = tid % 8;
                        const uint16_t * q2 = q->qs + 4 * ib;
                        const uint8_t * aux8 = (const uint8_t *) q2;
                        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
                        const uint32_t aux32 = q2[2] | ((uint32_t) q2[3] << 16);
                        const float d = (float) q->d * (0.5f + (float) (aux32 >> 28)) * 0.25f;
                        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
                        sycl::half * y = wp + ((size_t) e * (size_t) N + (size_t) row) * (size_t) K + (size_t) kb * QK_K + 32 * ib + 8 * il;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            y[j] = (sycl::half) (d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
                        }
                    });
            } else {
                stream->parallel_for(sycl::range<1>((size_t) batch * (size_t) N * (size_t) (K / 2)), [=](sycl::id<1> it) {
                    const int64_t idx = it[0];
                    const int64_t j = idx & 15;
                    const int64_t b = idx >> 4;
                    const int64_t kb = b % (K / 32);
                    const int64_t row = (b / (K / 32)) % N;
                    const int64_t e = b / (K * N / 32);
                    const block_xe2mx4 * q = (const block_xe2mx4 *)(win + ep[base + e] * we + row * wr) + kb;
                    const float d = (float) q->d;
                    const uint8_t v = q->qs[j];
                    wp2[idx] = sycl::half2((sycl::half)(((int)(v & 15) - 8) * d),
                                           (sycl::half)(((int)(v >> 4) - 8) * d));
                });
            }

            DnnlGemmWrapper::gemm(ctx, (int) N, (int) rows, (int) K,
                wp, DnnlGemmWrapper::to_dt<sycl::half>(), 1, K, N * K,
                xp + row_start * K, DnnlGemmWrapper::to_dt<sycl::half>(), 1, K, rows * K,
                yp + row_start * N, DnnlGemmWrapper::to_dt<float>(), stream, batch, batch);
        }
    }

    // 3. Single kernel launch to scatter all valid rows from yp into yout
    stream->parallel_for(sycl::range<1>((size_t) total_padded_rows * (size_t) N), [=](sycl::id<1> it) {
        const int64_t flat = it[0];
        const int64_t n = flat % N;
        const int64_t pr = flat / N;
        const int32_t i1 = ri1[pr];
        if (i1 >= 0) {
            const int32_t i2 = ri2[pr];
            *(float *)(yout + (size_t) i1 * y1 + (size_t) i2 * y2 + (size_t) n * y0) = yp[flat];
        }
    });
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    if (src0->type == GGML_TYPE_IQ2_XXS && src1->ne[2] > 8 && (src0->ne[0] % QK_K) == 0 &&
        src1->type == GGML_TYPE_F32 &&
        ggml_sycl_get_env("GGML_SYCL_IQ2_MOE_DPAS", 0)) {
        ggml_sycl_mul_mat_id_xe2mx_prefill(ctx, dst);
        return;
    }
    if (src0->type == GGML_TYPE_XE2MX4) {
        // 8 rows or fewer (decode, speculative verify) stay on DP4A, which is faster there
        if (src1->ne[2] > 8 && ggml_sycl_get_env("GGML_SYCL_XE2MX_MOE_GEMM", 0)) {
            ggml_sycl_mul_mat_id_xe2mx_prefill(ctx, dst);
            return;
        }
        if (ggml_sycl_get_env("GGML_SYCL_XE2MX_MOE_DP4A", 0) &&
            ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, dst->src[2], dst)) {
            return;
        }
        ggml_sycl_mul_mat_id_xe2mx_native(ctx, dst);
        return;
    }
    if (ggml_is_exl3(src0->type)) {
        GGML_ABORT("EXL3 mul_mat_id is CUDA-only (packed expert tiles); SYCL has no MoE EXL3 path");
    }
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
        return;
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host is drained before we use it again
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*sizeof(mmid_row_mapping))));

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                const char *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                const mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;
    initialize_sycl_begining();
    xe2mx_scratch_bind(ctx.device, true);

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    const ggml_tensor * s0 = dst->src[0];
    std::cerr << e.what() << " OP " << ggml_op_name(dst->op)
              << " dev " << ctx.device
              << " dst " << (dst->name ? dst->name : "")
              << " " << dst->ne[0] << "x" << dst->ne[1]
              << " src0 " << (s0 && s0->name ? s0->name : "")
              << " " << (s0 ? s0->ne[0] : 0) << "x" << (s0 ? s0->ne[1] : 0)
              << std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    bool res = get_memory_size(dpct::dev_mgr::instance().get_device(device), *free, *total,
                               (MemoryAPIType) g_ggml_sycl_get_mem_api);
    if (!res) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *free, *total);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    if (tensor && tensor->type == GGML_TYPE_XE2MX4 && !ggml_sycl_xe2mx_lo_bytes_are_soa(tensor)) {
        fprintf(stderr, "xe2mx: refuse get_tensor_async of LO_INLINE weight %s\n",
                tensor->name ? tensor->name : "?");
        fflush(stderr);
        GGML_ABORT("xe2mx LO_INLINE get_tensor_async unsupported");
    }
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
static int ggml_sycl_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data;
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static const ggml_tensor * ggml_sycl_unwrap_tensor(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
                 t->op == GGML_OP_TRANSPOSE)) {
        t = t->view_src ? t->view_src : t->src[0];
    }
    return t;
}

// Per compute call: who reads each GET_ROWS result. One pass over the graph instead of one full scan per GET_ROWS
// node (that was ~50 scans x ~3000 nodes x 10 srcs per verify graph, ~1.5 ms host per step).
static uint64_t g_sycl_compute_epoch = 0;
struct ggml_sycl_gr_use {
    int  n    = 0;
    bool gdn  = false;
    bool conv = false;
};
static std::unordered_map<const ggml_tensor *, ggml_sycl_gr_use> g_sycl_gr_use;
static uint64_t g_sycl_gr_use_epoch = ~(uint64_t) 0;
static const ggml_cgraph * g_sycl_gr_use_graph = nullptr;

static void ggml_sycl_gr_use_build(const ggml_cgraph * cgraph) {
    g_sycl_gr_use.clear();
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!n->src[s]) {
                continue;
            }
            const ggml_tensor * t = ggml_sycl_unwrap_tensor(n->src[s]);
            if (!t || t->op != GGML_OP_GET_ROWS) {
                continue;
            }
            auto & u = g_sycl_gr_use[t];
            u.n++;
            if (n->op == GGML_OP_GATED_DELTA_NET && s == 5 && n->src[2] && t->src[0]) {
                const int64_t S_v = n->src[2]->ne[0];
                const int64_t H   = n->src[2]->ne[1];
                if (t->src[0]->ne[0] == S_v * S_v * H) {
                    u.gdn = true;
                }
            }
            if (n->op == GGML_OP_CONCAT && ggml_sycl_is_gdn_conv_concat(n)) {
                u.conv = true;
            }
        }
    }
    g_sycl_gr_use_epoch = g_sycl_compute_epoch;
    g_sycl_gr_use_graph = cgraph;
}


// Bolt Mamba-2: concat(state, transpose(xBC view)) -> ssm_conv -> add(bias) -> silu, with xBC a strided view.
// Writes only the last 64 concat columns (the conv-state CPY reads the last 3) and the silu output.
static const ggml_tensor * g_bolt_conv_done[256];
static int g_bolt_conv_done_n = 0;

static bool ggml_sycl_bolt_conv_skip(const ggml_tensor * node) {
    for (int k = 0; k < g_bolt_conv_done_n; ++k) {
        if (g_bolt_conv_done[k] == node) {
            g_bolt_conv_done[k] = nullptr;
            return true;
        }
    }
    return false;
}

static bool ggml_sycl_try_bolt_conv(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_BOLT_CONV", 0);
    ggml_tensor * cc = cgraph->nodes[i];
    if (!on || cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ((const int32_t *) cc->op_params)[0] != 0) {
        return false;
    }
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * x  = cc->src[1];
    static int ndbg = 0;
    if (ndbg++ < 1 && st && x) {
        GGML_SYCL_XE2_LOG("bolt conv probe st ne=%ld,%ld,%ld nb=%zu,%zu x ne=%ld,%ld,%ld,%ld nb=%zu,%zu,%zu xdata%%16=%zu cc_cont=%d st_cont=%d\n",
            (long) st->ne[0], (long) st->ne[1], (long) st->ne[2], st->nb[0], st->nb[1], (long) x->ne[0], (long) x->ne[1], (long) x->ne[2], (long) x->ne[3],
            x->nb[0], x->nb[1], x->nb[2], (size_t) (((uintptr_t) x->data) % 16), (int) ggml_is_contiguous(cc), (int) ggml_is_contiguous(st));
        fflush(stderr);
    }
    if (!st || !x || st->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || st->ne[0] != 3 ||
        !ggml_is_contiguous(st) || !ggml_is_contiguous(cc) || x->nb[1] != sizeof(float) ||
        (x->nb[0] % 16) != 0 || x->ne[1] != st->ne[1] || x->ne[2] != st->ne[2] || x->ne[3] != 1 ||
        (st->ne[1] % 4) != 0 || (((uintptr_t) x->data) % 16) != 0) {
        return false;
    }
    ggml_tensor * conv = nullptr; ggml_tensor * add = nullptr; ggml_tensor * silu = nullptr;
    int jc = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 64; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SSM_CONV && cgraph->nodes[j]->src[0] == cc) { conv = cgraph->nodes[j]; jc = j; break; }
    }
    if (ndbg < 3) {
        ndbg++;
        for (int j = i + 1; j < cgraph->n_nodes && j < i + 8; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            GGML_SYCL_XE2_LOG("  bolt conv seq %d %s %s src0=%s\n", j - i, ggml_op_desc(n), n->name, n->src[0] ? n->src[0]->name : "-");
        }
        fflush(stderr);
    }
    if (!conv || jc + 2 >= cgraph->n_nodes) return false;
    {
        const int ncs_chk = 3 + (int) x->ne[0];
        const int keep_chk = ncs_chk < 64 ? ncs_chk : 64;
        for (int j = i + 1; j < jc; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_VIEW && n->view_src == cc) {
                const int col0 = (int) ((n->view_offs % cc->nb[1]) / sizeof(float));
                if (n->ne[0] > 3 || col0 < ncs_chk - keep_chk || n->nb[1] != cc->nb[1]) return false;
            } else if (n->op != GGML_OP_VIEW && n->op != GGML_OP_CPY && n->op != GGML_OP_RESHAPE) {
                for (int k = 0; k < GGML_MAX_SRC; ++k) if (n->src[k] == cc) return false;
            }
        }
    }
    add = cgraph->nodes[jc + 1]; silu = cgraph->nodes[jc + 2];
    if (add->op != GGML_OP_ADD || add->src[0] != conv || silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != add) return false;
    const ggml_tensor * w = conv->src[1];
    const ggml_tensor * bias = add->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 4 || !ggml_is_contiguous(w) || bias->type != GGML_TYPE_F32 ||
        ggml_nelements(bias) != st->ne[1] || !ggml_is_contiguous(silu) || !ggml_is_contiguous(conv)) return false;
    if (g_bolt_conv_done_n + 3 > 256) return false;

    const int C = (int) st->ne[1];
    const int n_t = (int) x->ne[0];
    const int n_s = (int) st->ne[2];
    const int ncs = 3 + n_t;
    const size_t xt = x->nb[0] / sizeof(float);
    const size_t xs = x->nb[2] / sizeof(float);
    const float * sp = (const float *) st->data;
    const float * xp = (const float *) x->data;
    const float * wp = (const float *) w->data;
    const float * bp = (const float *) bias->data;
    float * cp = (float *) cc->data;
    float * yp = (float *) silu->data;
    queue_ptr q = ctx.stream();
    const int n_keep = ncs < 64 ? ncs : 64;
    const int tok0 = ncs - n_keep;
    q->parallel_for(sycl::range<1>((size_t) n_s * C * n_keep), [=](sycl::id<1> id) {
        size_t idx = id[0];
        const int tl = (int) (idx % n_keep); idx /= n_keep;
        const int c = (int) (idx % C);
        const int s = (int) (idx / C);
        const int k = tok0 + tl;
        const float v = k < 3 ? sp[(size_t) s * 3 * C + (size_t) c * 3 + k]
                              : xp[(size_t) s * xs + (size_t) (k - 3) * xt + c];
        cp[(size_t) s * ncs * C + (size_t) c * ncs + k] = v;
    });
    constexpr int TOK = 64;
    const int n_c4 = C / 4;
    const int n_tb = (n_t + TOK - 1) / TOK;
    const size_t tot = (size_t) n_c4 * n_tb * n_s;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((tot + 255) / 256 * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_id(0);
        if (idx >= tot) return;
        const int c = (int) (idx % n_c4) * 4;
        const int tb = (int) ((idx / n_c4) % n_tb);
        const int s = (int) (idx / ((size_t) n_c4 * n_tb));
        const int t0 = tb * TOK;
        const int t1 = t0 + TOK < n_t ? t0 + TOK : n_t;
        auto ld = [&](int k) -> sycl::float4 {
            if (k < 3) {
                const float * b = sp + (size_t) s * 3 * C;
                return sycl::float4(b[(c + 0) * 3 + k], b[(c + 1) * 3 + k], b[(c + 2) * 3 + k], b[(c + 3) * 3 + k]);
            }
            return *reinterpret_cast<const sycl::float4 *>(xp + (size_t) s * xs + (size_t) (k - 3) * xt + c);
        };
        sycl::float4 wv[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) wv[r] = *reinterpret_cast<const sycl::float4 *>(wp + (size_t) (c + r) * 4);
        const sycl::float4 w0(wv[0].x(), wv[1].x(), wv[2].x(), wv[3].x());
        const sycl::float4 w1(wv[0].y(), wv[1].y(), wv[2].y(), wv[3].y());
        const sycl::float4 w2(wv[0].z(), wv[1].z(), wv[2].z(), wv[3].z());
        const sycl::float4 w3(wv[0].w(), wv[1].w(), wv[2].w(), wv[3].w());
        const sycl::float4 bb = *reinterpret_cast<const sycl::float4 *>(bp + c);
        sycl::float4 x0 = ld(t0), x1 = ld(t0 + 1), x2 = ld(t0 + 2);
        float * yb = yp + (size_t) s * C * n_t;
        for (int t = t0; t < t1; ++t) {
            const sycl::float4 x3 = ld(t + 3);
            sycl::float4 v = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + bb;
            v = v / (1.0f + sycl::native::exp(-v));
            *reinterpret_cast<sycl::float4 *>(yb + (size_t) t * C + c) = v;
            x0 = x1; x1 = x2; x2 = x3;
        }
    });
    g_bolt_conv_done[g_bolt_conv_done_n++] = conv;
    g_bolt_conv_done[g_bolt_conv_done_n++] = add;
    g_bolt_conv_done[g_bolt_conv_done_n++] = silu;
    static int nlog = 0;
    if (nlog++ < 2) { GGML_SYCL_XE2_LOG("bolt conv fuse %s C=%d n_t=%d xt=%zu\n", cc->name, C, n_t, xt); fflush(stderr); }
    return true;
}

static bool ggml_sycl_skip_gdn_state_rows(const ggml_cgraph * cgraph, int node_idx) {
    const int skipgr = ggml_sycl_get_env("GGML_SYCL_SKIPGR", 1);
    const int skipconv = ggml_sycl_get_env("GGML_SYCL_SKIPCONV", 0);
    if (skipgr == 0 && skipconv == 0) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    if (gr->op != GGML_OP_GET_ROWS || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) || !gr->src[0] || !gr->src[1]) {
        return false;
    }
    if (gr->src[0]->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32 ||
        gr->src[0]->nb[0] != sizeof(float) || (gr->src[0]->nb[1] % sizeof(float)) != 0) {
        return false;
    }
    int nuser = 0;
    int gdn_ok = 0;
    int conv_ok = 0;
    if (ggml_sycl_get_env("GGML_SYCL_GR_SCAN_SLOW", 0) == 0) {
        if (g_sycl_gr_use_epoch != g_sycl_compute_epoch || g_sycl_gr_use_graph != cgraph) {
            ggml_sycl_gr_use_build(cgraph);
        }
        const auto it = g_sycl_gr_use.find(gr);
        if (it != g_sycl_gr_use.end()) {
            nuser = it->second.n;
            gdn_ok = it->second.gdn;
            conv_ok = it->second.conv;
        }
    } else
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!n->src[s]) {
                continue;
            }
            if (ggml_sycl_unwrap_tensor(n->src[s]) != gr) {
                continue;
            }
            nuser++;
            if (n->op == GGML_OP_GATED_DELTA_NET && s == 5 && n->src[2]) {
                const int64_t S_v = n->src[2]->ne[0];
                const int64_t H   = n->src[2]->ne[1];
                if (gr->src[0]->ne[0] == S_v * S_v * H) {
                    gdn_ok = 1;
                }
            }
            if (n->op == GGML_OP_CONCAT && ggml_sycl_is_gdn_conv_concat(n)) {
                conv_ok = 1;
            }
        }
    }
    if (nuser == 1 && gdn_ok && skipgr) {
        return true;
    }
    if (nuser == 1 && conv_ok && skipconv) {
        static int nhit = 0;
        if (nhit < 4) {
            nhit++;
            GGML_SYCL_XE2_LOG( "xe2mx skipconv hit ne0=%ld ne1=%ld\n", (long) gr->ne[0], (long) gr->ne[1]);
            fflush(stderr);
        }
        return true;
    }
    return false;
}

static bool ggml_sycl_skip_gdn_l2(const ggml_cgraph * cgraph, int node_idx) {
    if (ggml_sycl_get_env("GGML_SYCL_GDN_TCOL", 1) == 0) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_GDN_L2", 0) == 0 &&
        ggml_sycl_get_env("GGML_SYCL_FUSE_CONV_L2", 0) == 0) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[node_idx];
    if (!node || node->op != GGML_OP_L2_NORM || (node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !node->src[0] || node->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(node->src[0])) {
        return false;
    }
    int nuser  = 0;
    int gdn_ok = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!n || ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!n->src[s] || ggml_sycl_unwrap_tensor(n->src[s]) != node) {
                continue;
            }
            nuser++;
            if (n->op != GGML_OP_GATED_DELTA_NET || (s != 0 && s != 1) || !n->src[2] ||
                n->src[2]->ne[0] != 128 || n->src[2]->ne[2] < 16) {
                continue;
            }
            const ggml_tensor * uq = ggml_sycl_unwrap_tensor(n->src[0]);
            const ggml_tensor * uk = ggml_sycl_unwrap_tensor(n->src[1]);
            if (uq && uk && uq->op == GGML_OP_L2_NORM && uk->op == GGML_OP_L2_NORM &&
                uq->src[0] && uk->src[0] && ggml_is_contiguous_rows(uq->src[0]) &&
                ggml_is_contiguous_rows(uk->src[0])) {
                gdn_ok = 1;
            }
        }
    }
    if (nuser == 1 && gdn_ok) {
        static int nskipl2 = 0;
        if (nskipl2 < 4 && ggml_sycl_get_env("GGML_SYCL_FUSE_CONV_L2", 0)) {
            nskipl2++;
            GGML_SYCL_XE2_LOG( "xe2mx skip-l2 hit %s\n", node->name);
            fflush(stderr);
        }
        return true;
    }
    return false;
}

static bool ggml_sycl_skip_gdn_pro_sigmoid(const ggml_cgraph * cgraph, int node_idx) {
    if ((ggml_sycl_get_env("GGML_SYCL_FUSE_GDN_PRO", 0) & 1) == 0) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[node_idx];
    if (!node || node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    int nuser = 0;
    int gdn_ok = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!n->src[s]) {
                continue;
            }
            if (ggml_sycl_unwrap_tensor(n->src[s]) != node) {
                continue;
            }
            nuser++;
            if (n->op == GGML_OP_GATED_DELTA_NET && s == 4 && n->src[2] &&
                (n->src[2]->ne[2] <= 4 || n->src[2]->ne[2] >= 16) &&
                n->src[3] && n->src[3]->ne[0] != n->src[2]->ne[0]) {
                gdn_ok = 1;
            }
        }
    }
    if (nuser == 1 && gdn_ok) {
        static int nhit = 0;
        if (nhit < 4) {
            nhit++;
            GGML_SYCL_XE2_LOG( "xe2mx skippro sigmoid hit\n");
            fflush(stderr);
        }
        return true;
    }
    return false;
}

static int ggml_sycl_skip_gdn_pro_gate(const ggml_cgraph * cgraph, int node_idx) {
    if ((ggml_sycl_get_env("GGML_SYCL_FUSE_GDN_PRO", 0) & 2) == 0) {
        return 0;
    }
    if (!ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { GGML_UNARY_OP_SOFTPLUS })) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[node_idx + 2];
    int nuser = 0;
    int gdn_ok = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!n->src[s]) {
                continue;
            }
            if (ggml_sycl_unwrap_tensor(n->src[s]) != mul) {
                continue;
            }
            nuser++;
            if (n->op == GGML_OP_GATED_DELTA_NET && s == 3 && n->src[2] &&
                (n->src[2]->ne[2] <= 4 || n->src[2]->ne[2] >= 16) &&
                n->src[3] && n->src[3]->ne[0] != n->src[2]->ne[0]) {
                gdn_ok = 1;
            }
        }
    }
    if (nuser == 1 && gdn_ok) {
        static int nhit = 0;
        if (nhit < 4) {
            nhit++;
            GGML_SYCL_XE2_LOG( "xe2mx skippro gate hit\n");
            fflush(stderr);
        }
        return 3;
    }
    return 0;
}


struct xe2mx_optime_scope {
    ggml_backend_sycl_context * ctx;
    int                         on;
    int                         op;
    double *                    acc;
    int *                       nacc;
    double *                    t_mm;
    int *                       n_mm;
    const ggml_tensor *         node;
    std::chrono::steady_clock::time_point t0;
    xe2mx_optime_scope(ggml_backend_sycl_context * c, int o, int on_, double * a, int * n,
                       double * tm, int * nm, const ggml_tensor * nd) :
        ctx(c), on(on_), op(o), acc(a), nacc(n), t_mm(tm), n_mm(nm), node(nd) {
        if (!on) {
            return;
        }
        ctx->stream()->wait();
        t0 = std::chrono::steady_clock::now();
    }
    ~xe2mx_optime_scope() {
        if (!on) {
            return;
        }
        ctx->stream()->wait();
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        acc[op] += dt;
        nacc[op]++;
        {
            static const int optnode = ggml_sycl_get_env("GGML_SYCL_XE2MX_OPTNODE", 0);
            if (optnode && op == GGML_OP_MUL_MAT && node && node->src[0] && node->src[1]) {
                GGML_SYCL_XE2_LOG( "optn %s t=%s N=%ld K=%ld M=%ld ms=%.3f\n", node->name,
                        ggml_type_name(node->src[0]->type), (long) node->src[0]->ne[1], (long) node->src[0]->ne[0],
                        (long) ggml_nrows(node->src[1]), dt);
            }
        }
        if (op == GGML_OP_MUL_MAT && node && node->src[0] && t_mm && n_mm) {
            const int t = (int) node->src[0]->type;
            if (t >= 0 && t < GGML_TYPE_COUNT) {
                t_mm[t] += dt;
                n_mm[t]++;
            }
        }
    }
};

struct xe2mx_gtime_scope {
    ggml_backend_sycl_context * ctx;
    int on;
    int M;
    int nmm;
    int nodes;
    std::chrono::steady_clock::time_point t0;
    xe2mx_gtime_scope(ggml_backend_sycl_context * c, const ggml_cgraph * g) : ctx(c), on(0), M(0), nmm(0), nodes(0) {
        static const int gt = ggml_sycl_get_env("GGML_SYCL_XE2MX_GTIME", 0);
        if (!gt) {
            return;
        }
        on = 1;
        nodes = g->n_nodes;
        for (int i = 0; i < g->n_nodes; ++i) {
            const ggml_tensor * n = g->nodes[i];
            if (n && n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[1] && n->src[0]->type == GGML_TYPE_XE2MX4) {
                if (M == 0) {
                    M = (int) ggml_nrows(n->src[1]);
                }
                nmm++;
            }
        }
        ctx->stream()->wait();
        t0 = std::chrono::steady_clock::now();
    }
    ~xe2mx_gtime_scope() {
        if (!on) {
            return;
        }
        const auto ts = std::chrono::steady_clock::now();
        ctx->stream()->wait();
        const auto te = std::chrono::steady_clock::now();
        const double sub = std::chrono::duration<double, std::milli>(ts - t0).count();
        const double tot = std::chrono::duration<double, std::milli>(te - t0).count();
        GGML_SYCL_XE2_LOG( "xe2mx gtime M=%d mm=%d nodes=%d submit=%.2f tot=%.2f\n", M, nmm, nodes, sub, tot);
        fflush(stderr);
    }
};

#include "bolt-tail.hpp"

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);
    ggml_sycl_env_cache_enable();
    xe2mx_gtime_scope gtime_sc(sycl_ctx, cgraph);
    g_sycl_compute_epoch++;
    g_sycl_compute_started = true;
    ggml_sycl_xe2mx_graph_begin();
    ggml_sycl_xe2mx_qbatch_prepare(*sycl_ctx, cgraph);

    const int xe2mx_optime = ggml_sycl_get_env("GGML_SYCL_XE2MX_OPTIME", 0);
    const int dumpgr = ggml_sycl_get_env("GGML_SYCL_XE2MX_DUMPGR", 0);
    int nsk0 = 0;
    int nsk1 = 0;
    int mm_mmax = 0;
    int n_xsmall = 0;
    if (xe2mx_optime) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (ggml_sycl_is_view_or_noop(n)) {
                continue;
            }
            if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[1]) {
                const int m = (int) ggml_nrows(n->src[1]);
                if (m > mm_mmax) {
                    mm_mmax = m;
                }
                if (n->src[0]->type == GGML_TYPE_XE2MX4 && m > 0 && m < 32) {
                    n_xsmall++;
                }
            }
        }
    }
    const int opt_decode = xe2mx_optime && n_xsmall >= 200;
    if (xe2mx_optime) {
        static int ng_all = 0;
        ng_all++;
        if (ng_all <= 3 || opt_decode) {
            GGML_SYCL_XE2_LOG( "xe2mx optime Mmax=%d xsmall=%d decode=%d\n", mm_mmax, n_xsmall, opt_decode ? 1 : 0);
            fflush(stderr);
        }
    }
    static int n_dec_timed = 0;
    static int n_dec_seen = 0;
    static const int dec_skip = ggml_sycl_get_env("GGML_SYCL_XE2MX_OPTIME_DSKIP", 0);
    static int n_pf_timed = 0;
    int opt_this = 0;
    if (opt_decode) {
        n_dec_seen++;
    }
    if (opt_decode && n_dec_seen > dec_skip && n_dec_timed < 3) {
        opt_this = 1;
        n_dec_timed++;
    }
    if (xe2mx_optime && mm_mmax >= 1024 && n_pf_timed < 14) {
        opt_this = 1;
        n_pf_timed++;
    }
    static int n_tp_seen = 0;
    const int tp_optime = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_OPTIME", 0);
    if (tp_optime && sycl_ctx->device == 0) {
        n_tp_seen++;
        if (n_tp_seen >= 260 && n_tp_seen < 264) {
            opt_this = 1;
        }
    }
    static int n_tp_seen_d[8] = {};
    if (tp_optime == 2 && sycl_ctx->device < 8 && xe2mx_optime) {
        // count every per-device graph call; load + two prefill graphs cover ~260
        // calls per device, so 262..320 lands inside the first decode token
        const int igr = n_tp_seen_d[sycl_ctx->device]++;
        if (igr >= 30 && igr < 90) {
            opt_this = 1;
        }
    }
    double t_op[GGML_OP_COUNT] = {};
    int n_op[GGML_OP_COUNT] = {};
    double t_mm[GGML_TYPE_COUNT] = {};
    int n_mm[GGML_TYPE_COUNT] = {};

    // one-time census: which nodes each device actually computes in TP chunks
    {
        static int ncalls = 0;
        ncalls++;
        if (ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_CENSUS", 0) && ncalls >= 262 && ncalls < 300) {
            int ncomp = 0;
            int nskip = 0;
            int nview = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * n = cgraph->nodes[i];
                if (ggml_sycl_is_view_or_noop(n)) {
                    nview++;
                    continue;
                }
                if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    nskip++;
                    continue;
                }
                ncomp++;
            }
            GGML_SYCL_XE2_LOG( "xe2mx tp census d=%d call=%d nodes=%d comp=%d skip=%d view=%d\n",
                    sycl_ctx->device, ncalls, cgraph->n_nodes, ncomp, nskip, nview);
            fflush(stderr);
        }
    }

    ggml_sycl_conv_concat_fuse_begin();
    g_bolt_conv_done_n = 0;
    // fused XE2MX paths below bypass compute_forward; bind this device's scratch first
    xe2mx_scratch_bind(sycl_ctx->device);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        xe2mx_scratch_bind(sycl_ctx->device);
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        {
            static const bool node_trace = ggml_sycl_get_env("GGML_SYCL_XE2MX_NODETRACE", 0) != 0;
            if (node_trace) {
                GGML_SYCL_XE2_LOG( "sycl node dev=%d i=%d %s op=%s %dx%d\n", (int) sycl_ctx->device, i,
                        node->name ? node->name : "", ggml_op_name(node->op),
                        (int) node->ne[0], (int) node->ne[1]);
            }
        }

        xe2mx_optime_scope opt_sc(sycl_ctx, (int) node->op, opt_this, t_op, n_op, t_mm, n_mm, node);
        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            i += nodes_to_skip;
            continue;
        }
        if (node->op == GGML_OP_MUL) {
            const int exl3_skip = ggml_sycl_try_exl3_fuse(*sycl_ctx, cgraph, i);
            if (exl3_skip > 0) {
                i += exl3_skip;
                continue;
            }
        }
#ifndef NDEBUG
        {
            auto sycl_buf_here = [](const ggml_tensor * t, int device) {
                ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
                if (!b) {
                    return false;
                }
                if (b->buft == ggml_backend_sycl_buffer_type(device)) {
                    return true;
                }
                if (!ggml_backend_buffer_is_sycl_split(b) || !b->buft->device) {
                    return false;
                }
                auto * dctx = (ggml_backend_sycl_device_context *) b->buft->device->context;
                return dctx && dctx->device == device;
            };
            assert(sycl_buf_here(node, sycl_ctx->device));
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (node->src[j] != nullptr) {
                    assert(sycl_buf_here(node->src[j], sycl_ctx->device));
                }
            }
        }
#endif
        static const int lab_no_conv_fuse = ggml_sycl_get_env("XE2_LAB_NO_CONV_FUSE", 0);
        // XE2_LAB_NO_CONV_FUSE: 1 = never fuse; 2 = auto, skip only deep-rollback verify batches (DFlash2 block,
        // 3..64 tokens: the fused path corrupts it); cheap-MTP verify (<= 2) and prefill keep the fusion
        bool conv_fuse_off = lab_no_conv_fuse == 1;
        if (lab_no_conv_fuse == 2 && node->op == GGML_OP_CONCAT && node->src[0] && node->src[1]) {
            const int64_t n_tok = node->ne[0] - node->src[0]->ne[0];
            conv_fuse_off = n_tok > 2 && n_tok <= 64;
        }
        if (g_bolt_conv_done_n > 0 && ggml_sycl_bolt_conv_skip(node)) {
            continue;
        }
        if (node->op == GGML_OP_CONCAT && ggml_sycl_try_bolt_conv(*sycl_ctx, cgraph, i)) {
            continue;
        }
        if (!conv_fuse_off && node->op == GGML_OP_CONCAT && ggml_sycl_try_conv_concat_fuse(*sycl_ctx, cgraph, i)) {
            continue;
        }
        if (node->op == GGML_OP_SSM_CONV) {
            const int ssm_skip = ggml_sycl_ssm_conv_skip_done(cgraph, i);
            if (ssm_skip > 0) {
                i += ssm_skip - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_CPY && ggml_sycl_conv_cpy_skip_done(node)) {
            continue;
        }
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            const int gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            if (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, node, fused_state_cpy);
                i += gdn_nodes_to_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_CONT) {
            const int tail = ggml_sycl_try_bolt_tail(*sycl_ctx, cgraph, i);
            if (tail) { i += tail - 1; continue; }
        }
        if (ggml_sycl_get_env("GGML_SYCL_BOLT_EXPERT_SUM", 0) && node->op == GGML_OP_ADD &&
            i + 7 < cgraph->n_nodes &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
                                          GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD }, { i + 7 })) {
            bool valid = true;
            struct sum_inputs { const float * p[9]; } inputs{};
            const int64_t n0 = node->ne[0], n1 = node->ne[1];
            size_t strides[9] = {};
            inputs.p[0] = (const float *) node->src[0]->data;
            strides[0] = node->src[0]->nb[1] / sizeof(float);
            for (int k = 0; k < 8; ++k) {
                ggml_tensor * a = cgraph->nodes[i + k];
                const ggml_tensor * s = a->src[1];
                valid = valid && a->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 &&
                    a->ne[0] == n0 && a->ne[1] == n1 && a->ne[2] == 1 && a->ne[3] == 1 &&
                    s->ne[0] == n0 && s->ne[1] == n1 && s->nb[0] == sizeof(float) &&
                    (k == 0 || a->src[0] == cgraph->nodes[i + k - 1]);
                inputs.p[k + 1] = (const float *) s->data;
                strides[k + 1] = s->nb[1] / sizeof(float);
            }
            ggml_tensor * last = cgraph->nodes[i + 7];
            valid = valid && node->src[0]->type == GGML_TYPE_F32 &&
                node->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous(last);
            if (valid) {
                static int nsum = 0;
                if (nsum++ < 2) { GGML_SYCL_XE2_LOG("bolt expert-sum bind %s\n", last->name); fflush(stderr); }
                struct sum_strides { size_t s[9]; } st{};
                for (int k = 0; k < 9; ++k) st.s[k] = strides[k];
                float * y = (float *) last->data;
                sycl_ctx->stream()->parallel_for(sycl::range<1>(n0 * n1), [=](sycl::id<1> id) {
                    const size_t col = id[0] % n0, row = id[0] / n0;
                    float v = inputs.p[0][row * st.s[0] + col];
                    for (int k = 1; k < 9; ++k) v += inputs.p[k][row * st.s[k] + col];
                    y[id] = v;
                });
                i += 7;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD) {
            const int npro = ggml_sycl_skip_gdn_pro_gate(cgraph, i);
            if (npro > 0) {
                i += npro - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { GGML_UNARY_OP_SOFTPLUS })) {
            ggml_sycl_op_add_softplus_mul(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            static int naddrms = 0;
            naddrms++;
            const int drop_env = ggml_sycl_get_env("GGML_SYCL_FUSE_ADD_RMS_DROPADD", 0);
            const int dropadd = drop_env && ggml_node_get_use_count(cgraph, i) == 1;
            if (naddrms <= 8 && (drop_env || ggml_sycl_get_env("GGML_SYCL_XE2MX_OPTIME", 0))) {
                GGML_SYCL_XE2_LOG( "xe2mx fuse add-rms hit %s nuser=%d drop=%d\n", node->name,
                        ggml_node_get_use_count(cgraph, i), dropadd ? 1 : 0);
                fflush(stderr);
            }
            ggml_sycl_op_add_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                                            dropadd ? 1 : 0);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_ROPE && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_VIEW && cgraph->nodes[i + 2]->op == GGML_OP_SET_ROWS &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
            static int nropeset = 0;
            if (nropeset < 8) {
                nropeset++;
                GGML_SYCL_XE2_LOG( "xe2mx rope-set bind %s\n", node->name);
                fflush(stderr);
            }
            ggml_sycl_rope_fused(*sycl_ctx, node, cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_CONT && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY && cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_tensor * un = cgraph->nodes[i + 1];
            ggml_tensor * mu = cgraph->nodes[i + 2];
            static int ncontmul = 0;
            if (ncontmul < 8) {
                ncontmul++;
                GGML_SYCL_XE2_LOG( "xe2mx contunary-mul bind %s\n", node->name);
                fflush(stderr);
            }
            ggml_sycl_unary_mul_from(*sycl_ctx, un, node->src[0], mu);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_CONT && i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_CONT, GGML_OP_UNARY },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_tensor * un = cgraph->nodes[i + 1];
            ggml_sycl_unary_from(*sycl_ctx, un, node->src[0]);
            i++;
            continue;
        }
        if (ggml_sycl_get_env("GGML_SYCL_BOLT_RELU_SQR", 0) &&
            node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU &&
            i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_SQR &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { i + 1 }) &&
            node->src[0]->type == GGML_TYPE_F32 && cgraph->nodes[i + 1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(cgraph->nodes[i + 1])) {
            const float * x = (const float *) node->src[0]->data;
            float * y = (float *) cgraph->nodes[i + 1]->data;
            const size_t n = ggml_nelements(cgraph->nodes[i + 1]);
            sycl_ctx->stream()->parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
                const float v = x[j];
                y[j] = v > 0.0f ? v * v : 0.0f;
            });
            ++i;
            continue;
        }
        if (node->op == GGML_OP_L2_NORM && ggml_sycl_skip_gdn_l2(cgraph, i)) {
            continue;
        }
        if (node->op == GGML_OP_UNARY && ggml_sycl_skip_gdn_pro_sigmoid(cgraph, i)) {
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(node) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

        const bool xe_row_split = node->op == GGML_OP_MUL_MAT && node->src[0]
                && ggml_backend_buffer_is_sycl_split(node->src[0]->buffer);
        if ((node->op == GGML_OP_MUL || node->op == GGML_OP_MUL_MAT) && !xe_row_split) {
            const int fwht_n = ggml_sycl_try_fwht_scale_fuse(*sycl_ctx, cgraph, i);
            if (fwht_n > 0) {
                i += fwht_n - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split) {
            const int nov = ggml_sycl_xe2mx_ovl_run(*sycl_ctx, cgraph, i);
            if (nov > 0) {
                i += nov - 1;
                continue;
            }
            const int ns4 = ggml_sycl_xe2mx_s4ovl_run(*sycl_ctx, cgraph, i);
            if (ns4 > 0) {
                i += ns4 - 1;
                continue;
            }
            const int nlo = ggml_sycl_xe2mx_loovl_run(*sycl_ctx, cgraph, i);
            if (nlo > 0) {
                i += nlo - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split && ggml_sycl_xe2mx_mul_mat_add(*sycl_ctx, cgraph, i)) {
            i += 1;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split) {
            const int nzsilu = ggml_sycl_xe2mx_mul_mat_silu_mul(*sycl_ctx, cgraph, i);
            if (nzsilu > 0) {
                i += nzsilu - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split) {
            const int ngmlp = ggml_sycl_xe2mx_s4gmlp_run(*sycl_ctx, cgraph, i);
            if (ngmlp > 0) {
                i += ngmlp - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split && ggml_sycl_xe2mx_mul_mat_glu(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && !xe_row_split) {
            // GGML_SYCL_XE2MX_BATCH=1: one launch for several small-M XE2MX4 MUL_MATs (own src1 each). Returns 1
            // for the head it ran and for every later member it already computed. Off: always 0.
            const int nbt = ggml_sycl_xe2mx_batch_run(*sycl_ctx, cgraph, i);
            if (nbt > 0) {
                i += nbt - 1;
                continue;
            }
            const int nmg = ggml_sycl_xe2mx_mg_run(*sycl_ctx, cgraph, i);
            if (nmg > 0) {
                continue;
            }
            const int nfn = ggml_sycl_xe2mx_fn_run(*sycl_ctx, cgraph, i);
            if (nfn > 0) {
                i += nfn - 1;
                continue;
            }
            const int nf2 = ggml_sycl_xe2mx_f2_run(*sycl_ctx, cgraph, i);
            if (nf2 > 0) {
                i += nf2 - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_mmvq_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && node->src[0] && ggml_is_exl3(node->src[0]->type) &&
                node->src[1] && ggml_nrows(node->src[1]) > EXL3_GEMV_MAX_M) {
            const ggml_tensor * nxt = nullptr;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * n = cgraph->nodes[j];
                if (n->op == GGML_OP_MUL_MAT && n->src[0] && ggml_is_exl3(n->src[0]->type) &&
                        n->src[1] && ggml_nrows(n->src[1]) > EXL3_GEMV_MAX_M) {
                    nxt = n;
                    break;
                }
            }
            ggml_sycl_exl3_set_lookahead(nxt);
        }
        if (node->op == GGML_OP_GET_ROWS) {
            const bool skipgrn = ggml_sycl_skip_gdn_state_rows(cgraph, i);
            if (dumpgr) {
                if (skipgrn) {
                    nsk1++;
                } else {
                    nsk0++;
                    static int ndump = 0;
                    if (ndump < 24) {
                        ndump++;
                        GGML_SYCL_XE2_LOG( "xe2mx dumpgr skip=0 name=%s ne0=%ld ne1=%ld\n",
                                node->name, (long) node->ne[0], (long) node->ne[1]);
                        fflush(stderr);
                    }
                }
            }
            if (skipgrn) {
                continue;
            }
        }
        bool ok = false;
        try {
            ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        } catch (const std::exception & e) {
            GGML_SYCL_XE2_LOG( "sycl node THROW dev=%d node=%s op=%s %dx%d type=%s what=%s\n",
                    (int) sycl_ctx->device, node->name ? node->name : "", ggml_op_name(node->op),
                    (int) node->ne[0], (int) node->ne[1], typeid(e).name(), e.what());
            fflush(stderr);
            throw;
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
    if (dumpgr) {
        static int ngd = 0;
        ngd++;
        if (ngd <= 4) {
            GGML_SYCL_XE2_LOG( "xe2mx dumpgr graph %d skip0=%d skip1=%d\n", ngd, nsk0, nsk1);
            fflush(stderr);
        }
    }
    if (opt_this) {
        static int ng = 0;
        ng++;
        if (ng <= 3 || (mm_mmax >= 2048 && ng <= 16) || mm_mmax <= 16) {
            int ord[GGML_OP_COUNT];
            for (int o = 0; o < GGML_OP_COUNT; ++o) {
                ord[o] = o;
            }
            std::sort(ord, ord + GGML_OP_COUNT, [&](int a, int b) { return t_op[a] > t_op[b]; });
            double tot = 0;
            for (int o = 0; o < GGML_OP_COUNT; ++o) {
                tot += t_op[o];
            }
            GGML_SYCL_XE2_LOG( "xe2mx optime graph %d d=%d tot=%.1f ms Mmax=%d xsmall=%d nodes=%d\n",
                    ng, sycl_ctx->device, tot, mm_mmax, n_xsmall, cgraph->n_nodes);
            const double floor_ms = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_OPTIME", 0) ? 0.05 : 0.5;
            for (int k = 0; k < 16; ++k) {
                const int o = ord[k];
                if (t_op[o] < floor_ms) {
                    break;
                }
                GGML_SYCL_XE2_LOG( "  %s %.1f ms n=%d\n", ggml_op_name((ggml_op) o), t_op[o], n_op[o]);
            }
            if (t_op[GGML_OP_MUL_MAT] >= 0.5) {
                for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                    if (t_mm[t] < 0.5) {
                        continue;
                    }
                    GGML_SYCL_XE2_LOG( "  MUL_MAT %s %.1f ms n=%d\n", ggml_type_name((ggml_type) t), t_mm[t], n_mm[t]);
                }
            }
            if (ng == 2) {
                auto dump_op = [&](ggml_op op, const char * tag, int cap) {
                    int n = 0;
                    for (int i = 0; i < cgraph->n_nodes && n < cap; ++i) {
                        const ggml_tensor * t = cgraph->nodes[i];
                        if (!t || t->op != op) {
                            continue;
                        }
                        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            continue;
                        }
                        const char * nxt = (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1])
                            ? ggml_op_name(cgraph->nodes[i + 1]->op) : "-";
                        if (op == GGML_OP_UNARY) {
                            GGML_SYCL_XE2_LOG( "  %s %s %s ne=%ld,%ld,%ld nxt=%s\n", tag, t->name,
                                    ggml_unary_op_name(ggml_get_unary_op(t)),
                                    (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], nxt);
                        } else {
                            GGML_SYCL_XE2_LOG( "  %s %s ne=%ld,%ld,%ld nxt=%s\n", tag, t->name,
                                    (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], nxt);
                        }
                        n++;
                    }
                };
                dump_op(GGML_OP_ADD, "ADD", 16);
                dump_op(GGML_OP_UNARY, "UNARY", 12);
                dump_op(GGML_OP_CONCAT, "CONCAT", 8);
                dump_op(GGML_OP_MUL, "MUL", 12);
                dump_op(GGML_OP_CONT, "CONT", 8);
                dump_op(GGML_OP_RMS_NORM, "RMS", 8);
                dump_op(GGML_OP_L2_NORM, "L2", 8);
                dump_op(GGML_OP_SCALE, "SCALE", 8);
                int nmmf = 0;
                for (int i = 0; i < cgraph->n_nodes && nmmf < 12; ++i) {
                    const ggml_tensor * t = cgraph->nodes[i];
                    if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0]) {
                        continue;
                    }
                    if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    if (t->src[0]->type != GGML_TYPE_F32) {
                        continue;
                    }
                    GGML_SYCL_XE2_LOG( "  MM_F32 %s w=%s ne=%ld,%ld,%ld K=%ld N=%ld M=%ld\n", t->name,
                            t->src[0]->name, (long) t->ne[0], (long) t->ne[1], (long) t->ne[2],
                            (long) t->src[0]->ne[0], (long) t->src[0]->ne[1],
                            (long) (t->src[1] ? ggml_nrows(t->src[1]) : 0));
                    nmmf++;
                }
            }
            fflush(stderr);
        }
    }
}

#ifdef GGML_SYCL_GRAPH
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1 && ggml_sycl_get_env("GGML_SYCL_GRAPH_MULTI", 0) == 0) {
        GGML_SYCL_XE2_LOG( "sycl-graph: skip multi-device count=%d\n", ggml_sycl_info().device_count);
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT:
                if (ggml_sycl_get_env("GGML_SYCL_GRAPH_CONCAT", 0) == 0) {
                    GGML_SYCL_XE2_LOG( "sycl-graph: skip node CONCAT\n");
                    return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(node_op));
                return false;
            case GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
        }
    }
    return true;
}
#endif


static enum ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    if (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    } else if (g_ggml_sycl_enable_graph == 0) {
        static int nskip = 0;
        if (nskip++ < 3) {
            GGML_SYCL_XE2_LOG( "sycl-graph: ENABLE_GRAPH is 0\n");
        }
    }
    if (use_sycl_graph && !sycl_ctx->moe_q8) {
        // The fused MoE matvec allocates its q8 scratch on first use; never do that while recording
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i] && cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                use_sycl_graph = false;
                break;
            }
        }
    }
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Prefill is a fat XE2MX token batch (ne[1]), not ggml_nrows. A decode graph
        // still has MUL_MATs whose nrows count heads, and that was skipping capture.
        bool prefill_matmul = false;
        const ggml_tensor * fat = nullptr;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (!n || n->op != GGML_OP_MUL_MAT || !n->src[0] || !n->src[1]) {
                continue;
            }
            if (n->src[0]->type != GGML_TYPE_XE2MX4) {
                continue;
            }
            if (n->src[1]->ne[1] > 8) {
                prefill_matmul = true;
                fat = n;
                break;
            }
        }
        if (prefill_matmul) {
            static int nskip_pf = 0;
            if (nskip_pf++ < 4) {
                GGML_SYCL_XE2_LOG("sycl-graph: skip prefill %s ne1=%ld nodes=%d\n",
                                 fat && fat->name ? fat->name : "?",
                                 fat && fat->src[1] ? (long) fat->src[1]->ne[1] : -1,
                                 cgraph->n_nodes);
            }
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Record every step; executable graph replay is not supported here.
        {
            GGML_SYCL_XE2_LOG("sycl-graph: record decode nodes=%d\n", cgraph->n_nodes);
        }
        ggml_sycl_xe2mx_graph_record_prepare(*sycl_ctx, cgraph);

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        if (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } else {
            try {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        // Wait on the recorded event rather than draining the destination queue.
        // A queue wait here serializes independent work already submitted to that device.
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_info().device_count;
}


// backend device

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    bool res = get_memory_size(dpct::dev_mgr::instance().get_device(ctx->device), *free, *total,
                               (MemoryAPIType) g_ggml_sycl_get_mem_api);
    if (!res) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *free, *total);
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    const sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    if (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }

                if (src0_type == GGML_TYPE_XE2MX4 && op->op == GGML_OP_MUL_MAT_ID) {
                    const ggml_tensor * ids = op->src[2];
                    return ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[device].hw_info.arch) &&
                           (!a->buffer || !ggml_backend_buffer_is_sycl_split(a->buffer)) &&
                           a->ne[3] == 1 && b->ne[3] == 1 && b->type == GGML_TYPE_F32 &&
                           op->type == GGML_TYPE_F32 && ids && ids->type == GGML_TYPE_I32 &&
                           a->ne[0] == b->ne[0] && a->ne[0] % QK_XE2MX4 == 0 &&
                           (b->ne[1] == 1 || b->ne[1] == ids->ne[0]) && b->ne[2] == ids->ne[1];
                }

                if (src0_type == GGML_TYPE_TQ2_0) {
                    return false;
                }

                if (ggml_is_exl3(src0_type)) {
                    if (op->op == GGML_OP_MUL_MAT_ID) {
                        return false;
                    }
                    return ggml_sycl_exl3_available(device) &&
                           ggml_sycl_exl3_shape_ok(op->src[0], op->src[1], op);
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_XE2MX4:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                if (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_XE2MX4) {
                    return false;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                return res;
            }
            break;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0) {
                    return false;
                }

                return true;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= 32;
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(device, op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name) {
        return buft->device == dev;
    }
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;
    struct state_copy { void * dst; const void * src; size_t bytes; };
    std::vector<state_copy> pending_state;
    struct peer_weight {
        const void * source; void * resident; size_t bytes;
        std::unique_ptr<ggml_tensor> tensor;
        std::unique_ptr<ggml_tensor_extra_gpu> extra;
    };
    std::vector<peer_weight> peer_ffn;
    bool peer_ffn_ready = false;
    // LAYEROWN: full-width FFN weights for each card's owned layers, owner-written KV/state slices
    struct own_copy { void * dst; const void * src; size_t bytes; int from; size_t row_bytes = 0; int64_t rows = 0; };
    struct own_storage {
        std::unordered_map<std::string, peer_weight> full[2];
        std::vector<own_copy> pending;
        bool preloaded = false;
        // causal-contiguous valid KV length seen by this request's owned FA nodes (op_params[5]); KV rows >= it are empty
        int32_t kv_valid = 0;
        int64_t kv_cells = 0;
        bool    kv_valid_bad = false;
    };
    std::shared_ptr<own_storage> own = std::make_shared<own_storage>();
    // per-slot scratch so pipelined slots do not alias the same device buffers
    void * own_tmp[2][3] = {};
    size_t own_tmp_cap[2] = {};
    char * own_bounce = nullptr;
};

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    if (n_backends != 2) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

static bool ggml_backend_sycl_comm_share_owner(void * src, void * dst) {
    auto * a = (ggml_backend_sycl_comm_context *) src;
    auto * b = (ggml_backend_sycl_comm_context *) dst;
    if (!a || !b || !b->own->full[0].empty() || !b->own->full[1].empty()) return false;
    for (int c = 0; c < 2; ++c) {
        auto * ca = (ggml_backend_sycl_context *) a->backends[c]->context;
        auto * cb = (ggml_backend_sycl_context *) b->backends[c]->context;
        if (ca->device != cb->device || ca->stream() != cb->stream()) return false;
    }
    b->own = a->own;
    return true;
}

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    if (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        try {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            for (const auto & w : comm_ctx->peer_ffn) sycl::free(w.resident, *sctx0->stream());
            comm_ctx->peer_ffn.clear();
            ggml_backend_sycl_context * owners[2] = { sctx0, sctx1 };
            for (int c = 0; c < 2; ++c) {
                if (comm_ctx->own.use_count() == 1) {
                    for (const auto & entry : comm_ctx->own->full[c]) {
                        sycl::free(entry.second.resident, *owners[c]->stream());
                    }
                    comm_ctx->own->full[c].clear();
                }
                for (auto & p : comm_ctx->own_tmp[c]) {
                    if (p) sycl::free(p, *owners[c]->stream());
                    p = nullptr;
                }
                comm_ctx->own_tmp_cap[c] = 0;
            }
            if (comm_ctx->own_bounce) sycl::free(comm_ctx->own_bounce, *sctx0->stream());
            if (comm_ctx->own.use_count() == 1) comm_ctx->own->pending.clear();
        } catch (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

// Structural preconditions shared by the allreduce entry point and the
// collapsed executor preflight: F32/F16, contiguous, matching element counts.
static bool ggml_sycl_tp_ar_tensors_ok(struct ggml_tensor * t0, struct ggml_tensor * t1) {
    if (t0->type != t1->type) {
        return false;
    }
    // F16 takes the direct 2-byte memcpy + add path below; other types are
    // rejected so the meta-backend uses its generic all-reduce.
    if (t0->type != GGML_TYPE_F32 && t0->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(t0) || !ggml_is_contiguous(t1)) {
        return false;
    }
    if (ggml_nelements(t0) != ggml_nelements(t1)) {
        return false;
    }
    return true;
}

// Body of the N=2 F32/F16 allreduce. wait_queues=false (collapsed executor)
// drops the front q->wait() calls: the D2H memcpys ride each card's in-order
// stream directly behind its chunk kernels, so the copy events also cover
// chunk completion and the host blocks only once per chunk, on the copies.
static bool ggml_sycl_tp_ar_impl(void * comm_ctx_v, struct ggml_tensor ** tensors, bool wait_queues) try {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. Cross-device copies
    // go through dev2dev_memcpy because the two devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op).
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        q0->wait();
        q1->wait();
        dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, f16_tmp0, tensors[1]->data, nbytes);
        dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        static char * bounce0 = nullptr;
        static char * bounce1 = nullptr;
        static size_t bounce_cap = 0;
        static bool bounce_is_pinned = false;
        const bool ar_pinned = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_ARPINNED", 0) != 0;
        char * b0 = nullptr;
        char * b1 = nullptr;
        if (ar_pinned) {
            // each buffer belongs to the context of the queue that writes it:
            // cross-context copy-engine writes into foreign host USM are not
            // coherent on this stack (Hi loop), CPU-side reads of both are fine
            if (bounce_cap < (size_t) nbytes || !bounce_is_pinned) {
                // a resize frees buffers the previous H2D may still read
                q0->wait();
                q1->wait();
                if (bounce_is_pinned) {
                    if (bounce0) {
                        sycl::free(bounce0, *q0);
                    }
                    if (bounce1) {
                        sycl::free(bounce1, *q1);
                    }
                    bounce0 = bounce1 = nullptr;
                } else {
                    free(bounce0);
                    free(bounce1);
                    bounce0 = bounce1 = nullptr;
                }
                bounce0 = (char *) sycl::malloc_host((size_t) nbytes, *q0);
                bounce1 = (char *) sycl::malloc_host((size_t) nbytes, *q1);
                bounce_is_pinned = true;
                bounce_cap = (bounce0 && bounce1) ? (size_t) nbytes : 0;
            }
        } else {
            if (bounce_cap < (size_t) nbytes || bounce_is_pinned) {
                // a resize frees buffers the previous H2D may still read
                q0->wait();
                q1->wait();
                if (bounce_is_pinned) {
                    if (bounce0) {
                        sycl::free(bounce0, *q0);
                    }
                    if (bounce1) {
                        sycl::free(bounce1, *q1);
                    }
                    bounce0 = bounce1 = nullptr;
                    bounce_is_pinned = false;
                } else {
                    free(bounce0);
                    free(bounce1);
                }
                bounce0 = (char *) malloc(nbytes);
                bounce1 = (char *) malloc(nbytes);
                bounce_cap = (bounce0 && bounce1) ? (size_t) nbytes : 0;
            }
        }
        if (!bounce0 || !bounce1) {
            return false;
        }
        b0 = bounce0;
        b1 = bounce1;
        const bool ar_prof = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_ARPROFILE", 0) != 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (wait_queues) {
            q0->wait();
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (wait_queues) {
            q1->wait();
        }
        const auto t2 = std::chrono::steady_clock::now();
        // in-order queue + event waits; per-context buffers keep every DMA
        // inside the context that allocated it. Pinned path: D2H as compute
        // kernels (each writes its own queue-context bounce buffer) so the
        // rendezvous adds no compute<->copy engine transition on the chunk's
        // critical path.
        sycl::event ed2h0;
        sycl::event ed2h1;
        if (bounce_is_pinned) {
            const float * s0p = (const float *) tensors[0]->data;
            const float * s1p = (const float *) tensors[1]->data;
            float * h0p = (float *) b0;
            float * h1p = (float *) b1;
            ed2h0 = q0->submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { h0p[i] = s0p[i]; });
            });
            ed2h1 = q1->submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { h1p[i] = s1p[i]; });
            });
        } else {
            ed2h0 = q0->memcpy(b0, tensors[0]->data, nbytes);
            ed2h1 = q1->memcpy(b1, tensors[1]->data, nbytes);
        }
        const auto t3 = std::chrono::steady_clock::now();
        // wait card1's copy first: the host arrives here right after submitting,
        // so w1b reports card1's own chunk+copy wall instead of reading as 0
        ed2h1.wait();
        ed2h0.wait();
        const auto t4 = std::chrono::steady_clock::now();
        const auto t5 = std::chrono::steady_clock::now();
        float * h0 = (float *) b0;
        float * h1 = (float *) b1;
        for (int64_t i = 0; i < nelem; ++i) {
            h0[i] += h1[i];
        }
        if (bounce_is_pinned) {
            // mirror the sum so each H2D reads its own context's buffer
            memcpy(b1, b0, nbytes);
        }
        const auto t6 = std::chrono::steady_clock::now();
        if (bounce_is_pinned) {
            const float * src = (const float *) b0;
            q0->submit([&](sycl::handler & h) {
                float * dst = (float *) tensors[0]->data;
                h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) { dst[i] = src[i]; });
            });
            q1->submit([&](sycl::handler & h) {
                float * dst = (float *) tensors[1]->data;
                const float * src1 = (const float *) b1;
                h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) { dst[i] = src1[i]; });
            });
        } else {
            q0->memcpy(tensors[0]->data, b0, nbytes);
            q1->memcpy(tensors[1]->data, b0, nbytes);
        }
        const auto t7 = std::chrono::steady_clock::now();
        if (ar_prof) {
            static double sw0, sw1, sdh, sw0b, sw1b, sadd, sh2d, stot;
            static int sar = 0;
            sw0   += std::chrono::duration<double, std::micro>(t1 - t0).count();
            sw1   += std::chrono::duration<double, std::micro>(t2 - t1).count();
            sdh   += std::chrono::duration<double, std::micro>(t3 - t2).count();
            sw0b  += std::chrono::duration<double, std::micro>(t4 - t3).count();
            sw1b  += std::chrono::duration<double, std::micro>(t5 - t4).count();
            sadd  += std::chrono::duration<double, std::micro>(t6 - t5).count();
            sh2d  += std::chrono::duration<double, std::micro>(t7 - t6).count();
            stot  += std::chrono::duration<double, std::micro>(t7 - t0).count();
            if (++sar == 64) {
                GGML_SYCL_XE2_LOG(
                        "xe2mx tp ar n=64 w0=%.0f w1=%.0f d2h=%.0f w0b=%.0f w1b=%.0f add=%.0f h2d=%.0f tot=%.0f pin=%d\n",
                        sw0 / sar, sw1 / sar, sdh / sar, sw0b / sar, sw1b / sar,
                        sadd / sar, sh2d / sar, stot / sar, (int) bounce_is_pinned);
                fflush(stderr);
                sar = 0;
                sw0 = sw1 = sdh = sw0b = sw1b = sadd = sh2d = stot = 0;
            }
        }
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    sycl::event c0 = q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    sycl::event c1 = q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: COMM-D2D-FIX-BF16 cross-device copy of compressed bytes via
    // dev2dev_memcpy (separate SYCL contexts; sync copy after compress).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    c0.wait();
    c1.wait();
    dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, inbox0, outbox1, bf16_bytes);
    dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr || tensors == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (comm_ctx->backends.size() != 2) {
        return false;
    }
    if (!ggml_sycl_tp_ar_tensors_ok(tensors[0], tensors[1])) {
        return false;
    }

    return ggml_sycl_tp_ar_impl(comm_ctx_v, tensors, /*wait_queues=*/ true);
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

// Fused device-side add for the collapsed executor (F32): after both chunks'
// partials are copied to the per-context pinned bounces, each card's own kernel
// computes dst = own_partial + peer_bounce in place (kernels may READ
// foreign-context host USM; the host waited on the peer's copy event before
// submitting). Replaces the CPU add + mirror + separate H2D kernels, and the
// backends[0] add queues before its own chunk ends, so the post-copy host round
// trip disappears from the critical path. Bounces are double-buffered per chunk
// parity so the next chunk's peer copy cannot race the previous add.
static bool ggml_sycl_tp_fused_ar(ggml_backend_sycl_comm_context * comm_ctx,
                                  ggml_tensor * t0, ggml_tensor * t1) try {
    const int64_t nelem = ggml_nelements(t0);
    const size_t nbytes = ggml_nbytes(t0);
    if (nelem == 0) {
        return true;
    }
    static char * bounce[2][2] = { { nullptr, nullptr }, { nullptr, nullptr } };
    static size_t bounce_cap = 0;
    static int parity = 0;
    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();
    if (bounce_cap < nbytes) {
        q0->wait();
        q1->wait();
        for (int pi = 0; pi < 2; ++pi) {
            if (bounce[0][pi]) {
                sycl::free(bounce[0][pi], *q0);
            }
            if (bounce[1][pi]) {
                sycl::free(bounce[1][pi], *q1);
            }
            bounce[0][pi] = (char *) sycl::malloc_host(nbytes, *q0);
            bounce[1][pi] = (char *) sycl::malloc_host(nbytes, *q1);
        }
        bounce_cap = (bounce[0][0] && bounce[0][1] && bounce[1][0] && bounce[1][1]) ? nbytes : 0;
        if (bounce_cap == 0) {
            return false;
        }
        parity = 0;
    }
    const int pi = parity & 1;
    parity++;
    float * b0 = (float *) bounce[0][pi];
    float * b1 = (float *) bounce[1][pi];
    float * d0 = (float *) t0->data;
    float * d1 = (float *) t1->data;
    // peer partials to pinned (each write stays in its own queue context)
    sycl::event e0 = q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { b0[i] = d0[i]; });
    });
    sycl::event e1 = q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { b1[i] = d1[i]; });
    });
    // backends[1]'s copy completes first (fast slot): queue backends[0]'s fused
    // add immediately - the in-order queue places it after its own chunk + copy
    e1.wait();
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { d0[i] = d0[i] + b1[i]; });
    });
    // backends[0]'s copy completes last: queue backends[1]'s fused add
    e0.wait();
    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>((size_t) nelem), [=](sycl::id<1> i) { d1[i] = d1[i] + b0[i]; });
    });
    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

// Collapsed TP executor (GGML_SYCL_XE2MX_TP_EXEC=1 gates the meta-backend
// call): run all per-backend chunk graphs from one loop with the allreduce
// rendezvous inline. Each chunk is submitted to backend 0 then backend 1
// exactly like the default meta loop; the reduction after each chunk (except
// the last, like meta) is the pinned bounce with the front queue waits
// dropped, so the D2H rides each card's in-order stream behind its chunk and
// the only host block per chunk is the two copy events.
bool ggml_backend_sycl_comm_graph_compute(void * comm_ctx_v, struct ggml_cgraph ** cgraphs, size_t n_graphs) try {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr || cgraphs == nullptr) {
        return false;
    }
    if (comm_ctx->backends.size() != 2) {
        return false;
    }
    if (n_graphs < 2 || n_graphs % 2 != 0) {
        return false;
    }
    const size_t n_chunks = n_graphs / 2;

    // preflight every chunk's rendezvous tensors so an unsupported reduce can
    // never leave the graph half-executed (false => meta re-runs its loop)
    for (size_t i = 0; i < n_chunks; i++) {
        ggml_tensor * red[2];
        for (size_t j = 0; j < 2; j++) {
            ggml_cgraph * g = cgraphs[2*i + j];
            if (g == nullptr || g->n_nodes < 1 || g->nodes[g->n_nodes - 1] == nullptr) {
                static int npre0 = 0;
                if (npre0++ < 4) {
                    GGML_SYCL_XE2_LOG( "xe2mx tp exec preflight null chunk=%zu slot=%zu\n", i, j);
                    fflush(stderr);
                }
                return false;
            }
            red[j] = g->nodes[g->n_nodes - 1];
        }
        if (!ggml_sycl_tp_ar_tensors_ok(red[0], red[1])) {
            static int npre1 = 0;
            if (npre1++ < 4) {
                GGML_SYCL_XE2_LOG( "xe2mx tp exec preflight tensor chunk=%zu type0=%d type1=%d ne=%ld\n",
                        i, (int) red[0]->type, (int) red[1]->type, (long) ggml_nelements(red[0]));
                fflush(stderr);
            }
            return false;
        }
    }

    {
        static int nexec = 0;
        if (nexec++ < 2) {
            GGML_SYCL_XE2_LOG( "xe2mx tp exec chunks=%zu inline rendezvous\n", n_chunks);
            fflush(stderr);
        }
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();
    const bool ex_prof = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_ARPROFILE", 0) != 0;
    // diagnosis: submit backends[1]'s chunk first when set; default order matches meta
    const bool ex_rev = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_REV", 0) != 0;
    const bool ex_probe2 = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_PROBE2", 0) != 0;
    const bool ex_fused = ggml_sycl_get_env("GGML_SYCL_XE2MX_TP_FUSED_AR", 1) != 0;
    static double s_sub0, s_sub1, s_subd;
    static int s_n = 0;
    static double p_w0, p_w1, p_r0, p_r1;
    static int p_n = 0;

    for (size_t i = 0; i < n_chunks; i++) {
        if (ex_probe2 && i > 0) {
            // residual drain: whatever of the previous rendezvous's H2D (and any
            // pre-chunk work) is still queued on each card at iteration start
            const auto tra = std::chrono::steady_clock::now();
            q0->wait();
            const auto trb = std::chrono::steady_clock::now();
            q1->wait();
            const auto trc = std::chrono::steady_clock::now();
            p_r0 += std::chrono::duration<double, std::micro>(trb - tra).count();
            p_r1 += std::chrono::duration<double, std::micro>(trc - trb).count();
        }
        ggml_cgraph * ga = ex_rev ? cgraphs[2*i + 1] : cgraphs[2*i + 0];
        ggml_cgraph * gb = ex_rev ? cgraphs[2*i + 0] : cgraphs[2*i + 1];
        ggml_backend_t ba = ex_rev ? comm_ctx->backends[1] : comm_ctx->backends[0];
        ggml_backend_t bb = ex_rev ? comm_ctx->backends[0] : comm_ctx->backends[1];
        const auto ts0 = std::chrono::steady_clock::now();
        ggml_status s0 = GGML_STATUS_SUCCESS;
        try {
            s0 = ggml_backend_sycl_graph_compute(ba, ga);
        } catch (const std::exception & e) {
            GGML_SYCL_XE2_LOG( "xe2mx tp exec THROW card0 chunk=%zu what=%s\n", i, e.what());
            fflush(stderr);
            q0->wait();
            q1->wait();
            return false;
        }
        const auto ts1 = std::chrono::steady_clock::now();
        if (s0 != GGML_STATUS_SUCCESS) {
            static int nsub0 = 0;
            if (nsub0++ < 4) {
                fprintf(stderr, "xe2mx tp exec submit0 fail chunk=%zu status=%d\n", i, (int) s0);
                fflush(stderr);
            }
            q0->wait();
            q1->wait();
            return false;
        }
        ggml_status s1 = GGML_STATUS_SUCCESS;
        try {
            s1 = ggml_backend_sycl_graph_compute(bb, gb);
        } catch (const std::exception & e) {
            GGML_SYCL_XE2_LOG( "xe2mx tp exec THROW card1 chunk=%zu what=%s\n", i, e.what());
            fflush(stderr);
            q0->wait();
            q1->wait();
            return false;
        }
        const auto ts2 = std::chrono::steady_clock::now();
        if (s1 != GGML_STATUS_SUCCESS) {
            static int nsub1 = 0;
            if (nsub1++ < 4) {
                fprintf(stderr, "xe2mx tp exec submit1 fail chunk=%zu status=%d\n", i, (int) s1);
                fflush(stderr);
            }
            q0->wait();
            q1->wait();
            return false;
        }
        // diagnosis: block on each queue right after its chunk submission to get
        // true per-card chunk drain walls, independent of copy-event semantics
        if (ex_probe2) {
            const auto tw0 = std::chrono::steady_clock::now();
            q0->wait();
            const auto tw1 = std::chrono::steady_clock::now();
            q1->wait();
            const auto tw2 = std::chrono::steady_clock::now();
            p_w0 += std::chrono::duration<double, std::micro>(tw1 - tw0).count();
            p_w1 += std::chrono::duration<double, std::micro>(tw2 - tw1).count();
            if (++p_n == 64) {
                GGML_SYCL_XE2_LOG( "xe2mx tp probe2 n=64 resid0=%.0f resid1=%.0f drain0=%.0f drain1=%.0f\n",
                        p_r0 / p_n, p_r1 / p_n, p_w0 / p_n, p_w1 / p_n);
                fflush(stderr);
                p_n = 0; p_w0 = p_w1 = p_r0 = p_r1 = 0;
            }
        }
        if (ex_prof) {
            s_sub0 += std::chrono::duration<double, std::micro>(ts1 - ts0).count();
            s_sub1 += std::chrono::duration<double, std::micro>(ts2 - ts1).count();
            if (++s_n == 64) {
                GGML_SYCL_XE2_LOG( "xe2mx tp exec n=64 sub0=%.0f sub1=%.0f\n",
                        s_sub0 / s_n, s_sub1 / s_n);
                fflush(stderr);
                s_n = 0;
                s_sub0 = s_sub1 = s_subd = 0;
            }
        }
        if (i + 1 >= n_chunks) {
            break;
        }
        // tensors[] stay keyed to backends (0 -> q0/bounce0) whatever the submit order
        ggml_cgraph * ga2 = cgraphs[2*i + 0];
        ggml_cgraph * gb2 = cgraphs[2*i + 1];
        ggml_tensor * red[2] = { ga2->nodes[ga2->n_nodes - 1], gb2->nodes[gb2->n_nodes - 1] };
        bool fused_done = false;
        if (ex_fused && red[0]->type == GGML_TYPE_F32 && red[1]->type == GGML_TYPE_F32) {
            fused_done = ggml_sycl_tp_fused_ar(comm_ctx, red[0], red[1]);
        }
        if (!fused_done && !ggml_sycl_tp_ar_impl(comm_ctx_v, red, /*wait_queues=*/ false)) {
            static int nrend = 0;
            if (nrend++ < 4) {
                fprintf(stderr, "xe2mx tp exec rendezvous fail chunk=%zu fused=%d ne=%ld\n",
                        i, (int) fused_done, (long) ggml_nelements(red[0]));
                fflush(stderr);
            }
            q0->wait();
            q1->wait();
            return false;
        }
    }
    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { GGML_ABORT("TP executor failed with an unknown exception"); }

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_share_owner") == 0) {
        return (void *) ggml_backend_sycl_comm_share_owner;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_graph_compute") == 0) {
        return (void *)ggml_backend_sycl_comm_graph_compute;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            initialize_sycl_begining();
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
