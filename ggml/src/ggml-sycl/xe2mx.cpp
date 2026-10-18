#include "xe2mx.hpp"
#include "xe2mx-a8.hpp"

#include "convert.hpp"
#include "gemm.hpp"
#include "sycl_hw.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#endif

#include <chrono>
#include <cmath>
#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#define GGML_SYCL_XE2MX_HAS_ESIMD
#endif

extern int g_ggml_sycl_enable_esimd;

// Activations are cast to f16 for the XMX paths. A value past the f16 range would become inf and then NaN
// in the dot product (a single outlier in a drafter FFN row did this at M >= 32), so saturate instead.
static void xe2mx_f32_to_f16_sat(const void * __restrict__ vx, sycl::half * __restrict__ y, int64_t k, dpct::queue_ptr stream) {
    const float * x = (const float *) vx;
    stream->parallel_for(sycl::range<1>((size_t) k), [=](sycl::id<1> i) {
        const float v = x[i];
        y[i] = sycl::half(sycl::fmin(sycl::fmax(v, -65504.0f), 65504.0f));
    });
}

static to_fp16_sycl_t xe2mx_get_to_x(ggml_type type, ggml_tensor * dst) {
    return type == GGML_TYPE_F32 ? xe2mx_f32_to_f16_sat : ggml_get_to_fp16_sycl(type, dst);
}

static const char * xe2mx_dump_root(void) {
    const char * e = std::getenv("GGML_SYCL_XE2MX_DUMP_DIR");
    if (e && e[0]) {
        return e;
    }
    // No baked-in path: activation dumps and the runtime rotation sidecar stay
    // off unless the caller names a directory.
    return nullptr;
}

// Disk cache of packed LO weights (SOA + s4+scale). Speeds restart after wedge;
// default off until GGML_SYCL_XE2MX_LO_CACHE=1 (or path).
static const char * xe2mx_lo_cache_dir(void) {
    const char * e = std::getenv("GGML_SYCL_XE2MX_LO_CACHE");
    if (!e || !e[0] || e[0] == '0') {
        return nullptr;
    }
    if (e[0] == '1' && e[1] == '\0') {
        // "1" means "use the cache" but no directory was named. Off is safer
        // than writing into a guessed path.
        return nullptr;
    }
    return e;
}

static void xe2mx_lo_cache_mkdir(const char * dir) {
    if (!dir || !dir[0]) {
        return;
    }
    mkdir(dir, 0755);
}

static std::string xe2mx_lo_cache_path(const char * dir, const char * name, int64_t N, int64_t K,
                                       int wg, int lo_h, size_t aos, size_t pbytes) {
    // sanitize tensor name
    std::string safe;
    if (name) {
        for (const char * p = name; *p; ++p) {
            char c = *p;
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-') {
                safe.push_back(c);
            } else {
                safe.push_back('_');
            }
        }
    }
    if (safe.empty()) {
        safe = "anon";
    }
    char buf[1024];
    const int mse = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_MSE", 0);
    char mtag[32] = "";
    if (mse > 0) {
        snprintf(mtag, sizeof(mtag), "_mse%d", mse);
    }
    snprintf(buf, sizeof(buf), "%s/%s_N%ld_K%ld_g%d_h%d_a%zu_p%zu%s.bin",
             dir, safe.c_str(), (long) N, (long) K, wg, lo_h, aos, pbytes, mtag);
    return std::string(buf);
}

// header + load/store live after g_xe2mx_lo maps (see below)

static void xe2mx_time_begin(int on, const queue_ptr & stream, std::chrono::steady_clock::time_point * t) {
    if (!on) {
        return;
    }
    stream->wait();
    *t = std::chrono::steady_clock::now();
}

static double xe2mx_time_ms(int on, const queue_ptr & stream, std::chrono::steady_clock::time_point t0) {
    if (!on) {
        return 0;
    }
    stream->wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void xe2mx_expand_s8(queue_ptr stream, const uint8_t * qs, int8_t * w8,
                              int64_t N, int64_t nb, size_t qpitch) {
    const int nw = 32;
    const int n32 = (int) (N / nw);
    if (n32 > 0) {
        stream->parallel_for(sycl::range<2>((size_t) nb, (size_t) n32), [=](sycl::id<2> gid) {
            const int ib = (int) gid[0];
            const int n0 = (int) gid[1] * nw;
            int8_t * wbase = w8 + (size_t) ib * 32u * (size_t) N + (size_t) n0;
            for (int jb = 0; jb < 16; ++jb) {
                sycl::vec<int8_t, 16> lo0;
                sycl::vec<int8_t, 16> lo1;
                sycl::vec<int8_t, 16> hi0;
                sycl::vec<int8_t, 16> hi1;
                for (int i = 0; i < 16; ++i) {
                    const uint8_t b0 = qs[(size_t) (n0 + i) * qpitch + (size_t) ib * 16u + (size_t) jb];
                    const uint8_t b1 = qs[(size_t) (n0 + 16 + i) * qpitch + (size_t) ib * 16u + (size_t) jb];
                    lo0[i] = (int8_t) ((int) (b0 & 0x0F) - 8);
                    hi0[i] = (int8_t) ((int) (b0 >> 4) - 8);
                    lo1[i] = (int8_t) ((int) (b1 & 0x0F) - 8);
                    hi1[i] = (int8_t) ((int) (b1 >> 4) - 8);
                }
                int8_t * wlo = wbase + (size_t) (2 * jb) * (size_t) N;
                int8_t * whi = wbase + (size_t) (2 * jb + 1) * (size_t) N;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(wlo) = lo0;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(wlo + 16) = lo1;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(whi) = hi0;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(whi + 16) = hi1;
            }
        });
    }
    const int n0t = n32 * nw;
    const int n16t = (int) ((N - n0t) / 16);
    if (n16t > 0) {
        stream->parallel_for(sycl::range<2>((size_t) nb, (size_t) n16t), [=](sycl::id<2> gid) {
            const int ib = (int) gid[0];
            const int n0 = n0t + (int) gid[1] * 16;
            int8_t * wbase = w8 + (size_t) ib * 32u * (size_t) N + (size_t) n0;
            for (int jb = 0; jb < 16; ++jb) {
                sycl::vec<int8_t, 16> lo;
                sycl::vec<int8_t, 16> hi;
                for (int i = 0; i < 16; ++i) {
                    const uint8_t b = qs[(size_t) (n0 + i) * qpitch + (size_t) ib * 16u + (size_t) jb];
                    lo[i] = (int8_t) ((int) (b & 0x0F) - 8);
                    hi[i] = (int8_t) ((int) (b >> 4) - 8);
                }
                int8_t * wlo = wbase + (size_t) (2 * jb) * (size_t) N;
                int8_t * whi = wbase + (size_t) (2 * jb + 1) * (size_t) N;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(wlo) = lo;
                *reinterpret_cast<sycl::vec<int8_t, 16> *>(whi) = hi;
            }
        });
    }
}

static void xe2mx_deq_soa_f16(ggml_backend_sycl_context & ctx, const uint8_t * packed, sycl::half * W,
                               int64_t K, int64_t N) {
    const int64_t nb = K / 32;
    const size_t pitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * pitch);
    queue_ptr stream = ctx.stream();
    const int n16 = (int) (N / 16);
    stream->parallel_for(sycl::range<2>((size_t) nb, (size_t) n16), [=](sycl::id<2> gid) {
        const int ib = (int) gid[0];
        const int n0 = (int) gid[1] * 16;
        const sycl::vec<sycl::half, 16> dh =
            *reinterpret_cast<const sycl::vec<sycl::half, 16> *>(dpl + (size_t) ib * (size_t) N + (size_t) n0);
        float d[16];
        uint8_t q[16][16];
        for (int i = 0; i < 16; ++i) {
            d[i] = (float) dh[i];
            const uint8_t * qsrc = qs + (size_t) (n0 + i) * pitch + (size_t) ib * 16u;
            *reinterpret_cast<sycl::vec<uint8_t, 16> *>(q[i]) =
                *reinterpret_cast<const sycl::vec<uint8_t, 16> *>(qsrc);
        }
        for (int j = 0; j < 16; ++j) {
            sycl::vec<sycl::half, 16> lo;
            sycl::vec<sycl::half, 16> hi;
            for (int i = 0; i < 16; ++i) {
                const int v0 = (int) (q[i][j] & 0x0F) - 8;
                const int v1 = (int) (q[i][j] >> 4) - 8;
                lo[i] = sycl::half(d[i] * (float) v0);
                hi[i] = sycl::half(d[i] * (float) v1);
            }
            sycl::half * wlo = W + (size_t) (ib * 32 + 2 * j) * (size_t) N + (size_t) n0;
            sycl::half * whi = W + (size_t) (ib * 32 + 2 * j + 1) * (size_t) N + (size_t) n0;
            *reinterpret_cast<sycl::vec<sycl::half, 16> *>(wlo) = lo;
            *reinterpret_cast<sycl::vec<sycl::half, 16> *>(whi) = hi;
        }
    });
}

struct xe2mx_wcache {
    void * packed;
    size_t bytes;
    int64_t K;
    int64_t N;
    int vnni;
    // host mirror of .packed (LO_HOSTPACK). Pinned host memory, not a device pointer.
    void * host = nullptr;
    size_t host_bytes = 0;
    int host_vnni = 0;
    int host_seq = 0;
    int use_seq = 0;
    bool host_pending = false;
    bool in_arena = false;
    bool pinned = false;
    sycl::event host_ev;
    // ggml device that owns .packed. Hostpack upload must not place the other
    // card's weights in this card's arena (cross-device USM, UR 39).
    int hp_dev = -1;
};

static std::unordered_map<const void *, xe2mx_wcache> g_xe2mx_wcache;
static std::unordered_set<const void *> g_xe2mx_lo;
static std::unordered_map<const void *, xe2mx_wcache> g_xe2mx_lo_w;
static size_t g_xe2mx_wcache_used = 0;
static size_t g_xe2mx_s8r_ex_used = 0;

// LO_HOSTPACK: the packed prefill copy of each LO weight lives in pinned host
// memory. It is uploaded before the first large GEMM of a prompt and dropped
// when decode starts. Decode reads the original tensor bytes only.
static bool xe2mx_lo_hp_on(void) {
    static const int on = [] {
        const char * e = getenv("GGML_SYCL_XE2MX_LO_HOSTPACK");
        if (e) {
            return atoi(e);
        }
        return ggml_sycl_info().device_count > 1 ? 0 : 1;
    }();
    return on != 0;
}
static bool g_xe2mx_lo_hp_resident = true;
// Per-device residency. The process-global flag uploaded every card's packs
// into whichever queue called ensure first.
static bool g_xe2mx_lo_hp_res_dev[GGML_SYCL_MAX_DEVICES];
static int g_xe2mx_lo_hp_res_init = 0;
static void * g_xe2mx_lo_hp_arena_dev[GGML_SYCL_MAX_DEVICES] = {};
static sycl::queue * g_xe2mx_lo_hp_cq_dev[GGML_SYCL_MAX_DEVICES] = {};
static int xe2mx_q_dev(sycl::queue & q) {
    const unsigned int id = dpct::dev_mgr::instance().get_device_id(q.get_device());
    if (id >= (unsigned int) GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return (int) id;
}
static bool & xe2mx_hp_res(int dev) {
    if (!g_xe2mx_lo_hp_res_init) {
        for (int i = 0; i < GGML_SYCL_MAX_DEVICES; ++i) {
            g_xe2mx_lo_hp_res_dev[i] = true;
        }
        g_xe2mx_lo_hp_res_init = 1;
    }
    if (dev < 0 || dev >= GGML_SYCL_MAX_DEVICES) {
        dev = 0;
    }
    return g_xe2mx_lo_hp_res_dev[dev];
}

// GGML_SYCL_XE2MX_Q32WG8_N="6144,1024": decode GEMV shapes that run on the WG=8 KB=4 grid.
static bool xe2mx_q32wg8_n_hit(int64_t N) {
    static std::vector<int64_t> ns = [] {
        std::vector<int64_t> v;
        const char * e = getenv("GGML_SYCL_XE2MX_Q32WG8_N");
        while (e && *e) {
            char * end = nullptr;
            const long long x = strtoll(e, &end, 10);
            if (end == e) {
                break;
            }
            if (x > 0) {
                v.push_back((int64_t) x);
            }
            e = (*end == ',') ? end + 1 : end;
        }
        return v;
    }();
    for (int64_t x : ns) {
        if (x == N) {
            return true;
        }
    }
    return false;
}
static long g_xe2mx_lo_hp_ups = 0;
static int g_xe2mx_lo_hp_seq = 0;
static int g_xe2mx_lo_hp_use = 0;
static void * g_xe2mx_lo_hp_arena = nullptr;

static long xe2mx_hp_key(const xe2mx_wcache * v) {
    // graph use order from the first prompt, then load order
    return v->use_seq ? v->use_seq : 1000000L + v->host_seq;
}
static sycl::queue * g_xe2mx_lo_hp_cq = nullptr;
// NF_KEEP FFN packs (g_xe2mx_wcache) that also live in host RAM. Opt-in via
// GGML_SYCL_XE2MX_LO_HOSTPACK=2.
static std::vector<xe2mx_wcache *> g_xe2mx_hp_extra;
static bool xe2mx_nf_hp_on(void) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK", 0);
    return on == 2;
}
// Mode 3: NF_KEEP FFN packs are not copied to RAM. They are baked on the card
// at the first prefill of a prompt (from weights already on the card) and freed
// when decode starts.
static bool xe2mx_nf_rebake_on(void) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK", 0);
    return on == 3;
}
struct xe2mx_nf_keep_ent {
    const void * key;
    size_t bytes;
    bool extra;
    int dev;
};
static std::vector<xe2mx_nf_keep_ent> g_xe2mx_nf_keep_ents;
static size_t g_xe2mx_nf_keep_used = 0;

// Separate in-order queue so the upload runs on a copy engine while the
// compute queue keeps working on earlier layers.
static sycl::queue & xe2mx_lo_hp_copyq(sycl::queue & q) {
    const int dev = xe2mx_q_dev(q);
    if (!g_xe2mx_lo_hp_cq_dev[dev]) {
        g_xe2mx_lo_hp_cq_dev[dev] = new sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order());
    }
    return *g_xe2mx_lo_hp_cq_dev[dev];
}

static void xe2mx_lo_hostpack_store(sycl::queue & q, xe2mx_wcache & val, const void * src, size_t bytes, bool src_dev) {
    if (!xe2mx_lo_hp_on() || !src || bytes == 0) {
        return;
    }
    if (val.host && val.host_bytes != bytes) {
        sycl::free(val.host, q);
        val.host = nullptr;
    }
    if (!val.host) {
        val.host = sycl::malloc_host(bytes, q);
        if (!val.host) {
            fprintf(stderr, "xe2mx lo hostpack pin fail %zu\n", bytes);
            fflush(stderr);
            return;
        }
    }
    if (src_dev) {
        q.memcpy(val.host, src, bytes).wait();
    } else {
        std::memcpy(val.host, src, bytes);
    }
    val.host_bytes = bytes;
    val.host_vnni = val.vnni;
    if (val.host_seq == 0) {
        val.host_seq = ++g_xe2mx_lo_hp_seq;
    }
    static size_t logged = 0;
    logged += bytes;
    if (logged == bytes || (logged >> 30) != ((logged - bytes) >> 30)) {
        GGML_SYCL_XE2_LOG( "xe2mx lo hostpack resident %zu MiB\n", logged >> 20);
        fflush(stderr);
    }
}

// Queue every pack that is not on the device, in load (layer) order, on the
// copy queue. Returns at once; each GEMM waits only on its own tensor.
static void xe2mx_lo_hostpack_ensure(sycl::queue & q) {
    const int dev = xe2mx_q_dev(q);
    if (!xe2mx_lo_hp_on() || xe2mx_hp_res(dev)) {
        return;
    }
    std::vector<xe2mx_wcache *> order;
    order.reserve(g_xe2mx_lo_w.size());
    for (auto & kv : g_xe2mx_lo_w) {
        xe2mx_wcache & v = kv.second;
        if (v.hp_dev != dev) {
            continue;
        }
        if (!v.packed && v.host && v.host_bytes) {
            order.push_back(&v);
        }
    }
    for (xe2mx_wcache * v : g_xe2mx_hp_extra) {
        if (v->hp_dev != dev) {
            continue;
        }
        if (!v->packed && v->host && v->host_bytes) {
            order.push_back(v);
        }
    }
    std::sort(order.begin(), order.end(),
              [](const xe2mx_wcache * a, const xe2mx_wcache * b) { return xe2mx_hp_key(a) < xe2mx_hp_key(b); });
    static const int sync_up = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK_SYNC", 0);
    static const int use_arena = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK_ARENA", 1);
    sycl::queue & cq = sync_up ? q : xe2mx_lo_hp_copyq(q);
    const int64_t t0 = ggml_time_us();
    size_t up = 0;
    char * arena = nullptr;
    size_t aoff = 0;
    if (use_arena && !order.empty()) {
        size_t total = 0;
        for (xe2mx_wcache * v : order) {
            total += (v->host_bytes + 255) & ~(size_t) 255;
        }
        GGML_ASSERT(!g_xe2mx_lo_hp_arena_dev[dev]);
        arena = (char *) ggml_sycl_malloc_device(total, q);
        g_xe2mx_lo_hp_arena_dev[dev] = arena;
    }
    for (xe2mx_wcache * v : order) {
        void * d = nullptr;
        if (arena) {
            d = arena + aoff;
            aoff += (v->host_bytes + 255) & ~(size_t) 255;
            v->in_arena = true;
        } else {
            d = ggml_sycl_malloc_device(v->host_bytes, q);
            v->in_arena = false;
        }
        if (!d) {
            fprintf(stderr, "xe2mx lo hostpack upload fail %zu\n", v->host_bytes);
            fflush(stderr);
            GGML_ABORT("xe2mx LO_HOSTPACK: device alloc failed");
        }
        v->host_ev = cq.memcpy(d, v->host, v->host_bytes);
        v->host_pending = !sync_up;
        v->packed = d;
        v->bytes = v->host_bytes;
        v->vnni = v->host_vnni;
        up += v->host_bytes;
    }
    if (sync_up) {
        q.wait();
    }
    xe2mx_hp_res(dev) = true;
    g_xe2mx_lo_hp_ups++;
    if (g_xe2mx_lo_hp_ups <= 4 || (g_xe2mx_lo_hp_ups % 32) == 0) {
        GGML_SYCL_XE2_LOG( "xe2mx lo hostpack upload #%ld dev=%d %zu MiB queued (%s%s) host %.2f ms\n", g_xe2mx_lo_hp_ups,
                dev, up >> 20, sync_up ? "sync" : "async", arena ? ",arena" : "", (ggml_time_us() - t0) / 1000.0);
        fflush(stderr);
    }
}

// Make the compute queue wait for this pack's copy without blocking the host.
static void xe2mx_lo_hostpack_fence(sycl::queue & q, xe2mx_wcache & v) {
    if (v.host && v.use_seq == 0) {
        v.use_seq = ++g_xe2mx_lo_hp_use;
    }
    if (!v.host_pending) {
        return;
    }
    q.ext_oneapi_submit_barrier({ v.host_ev });
    v.host_pending = false;
}

static void xe2mx_nf_hp_register(sycl::queue & q, xe2mx_wcache & v, const void * key = nullptr,
                                  bool extra = false) {
    const int dev = xe2mx_q_dev(q);
    v.hp_dev = dev;
    if (xe2mx_nf_rebake_on() && key && v.packed
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0)) {
        g_xe2mx_nf_keep_ents.push_back({ key, v.bytes, extra, dev });
        return;
    }
    if (!xe2mx_nf_hp_on() || !v.packed) {
        return;
    }
    xe2mx_lo_hostpack_store(q, v, v.packed, v.bytes, true);
    if (v.host) {
        v.use_seq = ++g_xe2mx_lo_hp_use;
        g_xe2mx_hp_extra.push_back(&v);
    }
}

// Free the device packs. Called once at load and when decode starts.
static void xe2mx_lo_hostpack_drop(sycl::queue & q) {
    const int dev = xe2mx_q_dev(q);
    if (!xe2mx_lo_hp_on() || !xe2mx_hp_res(dev)) {
        return;
    }
    bool any = false;
    for (auto & kv : g_xe2mx_lo_w) {
        if (kv.second.hp_dev == dev && kv.second.host && kv.second.packed && !kv.second.pinned) {
            any = true;
            break;
        }
    }
    for (xe2mx_wcache * v : g_xe2mx_hp_extra) {
        if (v->hp_dev == dev && v->host && v->packed) {
            any = true;
            break;
        }
    }
    if (g_xe2mx_lo_hp_arena_dev[dev]) {
        any = true;
    }
    if (!any) {
        return;
    }
    if (g_xe2mx_lo_hp_cq_dev[dev]) {
        g_xe2mx_lo_hp_cq_dev[dev]->wait();
    }
    q.wait();
    // First pass only: pin the earliest packs (in use order) on the device so the
    // first layers of every prompt do not wait for the copy.
    static bool pin_done[GGML_SYCL_MAX_DEVICES] = {};
    if (!pin_done[dev]) {
        pin_done[dev] = true;
        const size_t pin_cap = (size_t) ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK_KEEP_MB", 0) << 20;
        std::vector<xe2mx_wcache *> all;
        for (auto & kv : g_xe2mx_lo_w) {
            if (kv.second.hp_dev == dev && kv.second.host && kv.second.packed && kv.second.packed != kv.first) {
                all.push_back(&kv.second);
            }
        }
        std::sort(all.begin(), all.end(),
                  [](const xe2mx_wcache * a, const xe2mx_wcache * b) { return xe2mx_hp_key(a) < xe2mx_hp_key(b); });
        size_t pinned = 0;
        for (xe2mx_wcache * v : all) {
            if (pinned + v->bytes > pin_cap) {
                break;
            }
            pinned += v->bytes;
            v->pinned = true;
        }
        if (pin_cap) {
            GGML_SYCL_XE2_LOG( "xe2mx lo hostpack pinned %zu MiB on device\n", pinned >> 20);
            fflush(stderr);
        }
    }
    size_t freed = 0;
    for (auto & kv : g_xe2mx_lo_w) {
        xe2mx_wcache & v = kv.second;
        if (v.hp_dev != dev) {
            continue;
        }
        if (!v.host || !v.packed || v.packed == kv.first || v.pinned) {
            continue;
        }
        v.host_pending = false;
        if (!v.in_arena) {
            ggml_sycl_free_device(v.packed, q);
        }
        v.in_arena = false;
        freed += v.bytes;
        v.packed = nullptr;
    }
    for (xe2mx_wcache * v : g_xe2mx_hp_extra) {
        if (v->hp_dev != dev || !v->host || !v->packed) {
            continue;
        }
        v->host_pending = false;
        if (!v->in_arena) {
            ggml_sycl_free_device(v->packed, q);
        }
        v->in_arena = false;
        freed += v->bytes;
        v->packed = nullptr;
    }
    if (g_xe2mx_lo_hp_arena_dev[dev]) {
        ggml_sycl_free_device(g_xe2mx_lo_hp_arena_dev[dev], q);
        g_xe2mx_lo_hp_arena_dev[dev] = nullptr;
    }
    for (auto it_ent = g_xe2mx_nf_keep_ents.begin(); it_ent != g_xe2mx_nf_keep_ents.end(); ) {
        const xe2mx_nf_keep_ent & e = *it_ent;
        if (e.dev != dev) {
            ++it_ent;
            continue;
        }
        auto it = g_xe2mx_wcache.find(e.key);
        if (it != g_xe2mx_wcache.end()) {
            if (it->second.packed) {
                ggml_sycl_free_device(it->second.packed, q);
            }
            freed += e.bytes;
            g_xe2mx_nf_keep_used -= e.bytes;
            if (e.extra) {
                g_xe2mx_s8r_ex_used -= e.bytes;
            } else {
                g_xe2mx_wcache_used -= e.bytes;
            }
            g_xe2mx_wcache.erase(it);
        }
        it_ent = g_xe2mx_nf_keep_ents.erase(it_ent);
    }
    xe2mx_hp_res(dev) = false;
    static int nlog = 0;
    if (nlog < 8) {
        nlog++;
        GGML_SYCL_XE2_LOG( "xe2mx lo hostpack dropped dev=%d %zu MiB\n", dev, freed >> 20);
        fflush(stderr);
    }
}

static std::unordered_map<const void *, xe2mx_wcache> g_xe2mx_scb;
static std::unordered_map<const void *, xe2mx_wcache> g_xe2mx_s44any;
static uint8_t * g_xe2mx_s44_tmp = nullptr;
static float * g_xe2mx_s44_tsc = nullptr;
static int8_t * g_xe2mx_s44_t8 = nullptr;
static size_t g_xe2mx_s44_tcap = 0;
static size_t g_xe2mx_s44_tsccap = 0;
static size_t g_xe2mx_s44_t8cap = 0;
static const size_t XE2MX_WCACHE_CAP = (size_t) 12 * 1024 * 1024 * 1024;
static const size_t XE2MX_S8R_EX_CAP = (size_t) 4 * 1024 * 1024 * 1024;

static size_t xe2mx_s8r_ex_cap() {
    const int gb = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8REX", 4);
    if (gb == 0) {
        return 0;
    }
    if (gb < 1 || gb > 16) {
        return XE2MX_S8R_EX_CAP;
    }
    return (size_t) gb * 1024 * 1024 * 1024;
}

static int8_t * xe2mx_s8k_cached(queue_ptr stream, const uint8_t * packed, int64_t K, int64_t N) {
    const int64_t pitch = ((N + 63) / 64) * 64;
    if (pitch != N) {
        return nullptr;
    }
    const size_t bytes = (size_t) K * (size_t) pitch;
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == 3 && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        return (int8_t *) it->second.packed;
    }
    if (it != g_xe2mx_wcache.end()) {
        return nullptr;
    }
    if (g_xe2mx_wcache_used + bytes > XE2MX_WCACHE_CAP) {
        return nullptr;
    }
    void * persist = ggml_sycl_malloc_device(bytes, *stream);
    if (!persist) {
        return nullptr;
    }
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    xe2mx_expand_s8(stream, packed, (int8_t *) persist, N, nb, qpitch);
    g_xe2mx_wcache_used += bytes;
    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, 3 };
    return (int8_t *) persist;
}

struct xe2mx_s8r_view {
    int8_t * w = nullptr;
    float * sc = nullptr;
    float * aux = nullptr;
};

static bool xe2mx_s8dw_shape(int64_t K, int64_t N) {
    return ggml_sycl_get_env("GGML_SYCL_XE2MX_S8DW", 0) && K == 17408 && N == 5120;
}

bool ggml_sycl_xe2mx_s8dw_on(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_XE2MX4 && xe2mx_s8dw_shape(t->ne[0], t->ne[1]);
}

static int xe2mx_lo_enabled(void) {
    return ggml_sycl_get_env("GGML_SYCL_XE2MX_LO", 0);
}

bool ggml_sycl_xe2mx_lo_on(const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_XE2MX4 || !xe2mx_lo_enabled()) {
        return false;
    }
    if (ggml_sycl_xe2mx_s8dw_on(t)) {
        return false;
    }
    const int64_t K = t->ne[0];
    const int64_t N = t->ne[1];
    int nmin = ggml_sycl_get_env("GGML_SYCL_XE2MX_LON", 4096);
    if (nmin < 16) {
        nmin = 4096;
    }
    const bool ffn = N >= 16384 || K >= 16384;
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0) && ffn) {
        // no LO pack for FFN: exact SOA decode, prefill via SOA/S44 temp path, frees ~8.7 GiB
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_MID", 0) && !ffn) {
        // no LO pack for attn/mid: exact original decode, prefill via plain xe2mx path, frees ~3.35 GiB
        return false;
    }
    return t->ne[2] == 1 && t->ne[3] == 1 && (K % 128) == 0 && (N % 16) == 0 && N >= nmin;
}

size_t ggml_sycl_xe2mx_lo_nbytes(const ggml_tensor * t) {
    // Dual-store default: arena keeps block_xe2mx4 SOA size; LO pack is a separate persist.
    // LO_INLINE overwrites SOA slot in-place (pbytes <= aos) so alloc size stays ggml_nbytes.
    // Returning pbytes here would shrink the slot but needs every consumer LO-only before load -
    // not default. Parent may opt into LO_INLINE without changing get_alloc_size.
    (void) t;
    return 0;
}

// vnni tags in g_xe2mx_lo_w:
//   51 = dual-store: map key is SOA t->data, .packed is separate s4+scale persist
//   52 = inline:     map key is t->data, .packed == t->data (LO bytes in tensor slot)
// GGML_SYCL_XE2MX_LO_INLINE:
//   0 = off (dual-store, default)
//   1 = all LO tensors
//   2 = FFN-like only (N>=16384 || K>=16384) ~8.1 GiB pack reclaim
//   3 = attn/mid only (not FFN-like) ~3.35 GiB pack reclaim
static int xe2mx_lo_inline_mode(void) {
    const int m = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_INLINE", 0);
    if (m < 0 || m > 3) {
        return 0;
    }
    return m;
}

static bool xe2mx_lo_shape_ffn_like(int64_t N, int64_t K) {
    return N >= 16384 || K >= 16384;
}

static bool xe2mx_lo_want_inline(int64_t N, int64_t K) {
    const int m = xe2mx_lo_inline_mode();
    if (m == 0) {
        return false;
    }
    if (m == 1) {
        return true;
    }
    if (m == 2) {
        return xe2mx_lo_shape_ffn_like(N, K);
    }
    // m == 3: attn/mid
    return !xe2mx_lo_shape_ffn_like(N, K);
}

static bool xe2mx_lo_entry_inline(const xe2mx_wcache & e) {
    return e.vnni == 52 && e.packed != nullptr;
}

bool ggml_sycl_xe2mx_lo_inline_on(const ggml_tensor * t) {
    if (!t || !t->data) {
        return false;
    }
    auto it = g_xe2mx_lo_w.find((const void *) t->data);
    if (it == g_xe2mx_lo_w.end()) {
        return false;
    }
    return xe2mx_lo_entry_inline(it->second);
}

bool ggml_sycl_xe2mx_lo_bytes_are_soa(const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_XE2MX4) {
        return true;
    }
    // Non-LO XE2MX4 is always block/SOA (or AoS before pack_xmx).
    if (!t->data || !g_xe2mx_lo_w.count((const void *) t->data)) {
        return true;
    }
    return !ggml_sycl_xe2mx_lo_inline_on(t);
}

// After dual LO bake (cache hit or miss): optionally move pack into t->data and free persist.
// Requires pbytes <= ggml_nbytes(t). Marks vnni=52. Cache hit/miss share this path.
static void xe2mx_lo_maybe_inline(ggml_tensor * t, sycl::queue & q, int id) {
    if (!t || !t->data) {
        return;
    }
    const int64_t K = t->ne[0];
    const int64_t N = t->ne[1];
    if (!xe2mx_lo_want_inline(N, K)) {
        return;
    }
    {
        // GGML_SYCL_XE2MX_LO_INLINE_SKIP=sub1,sub2: keep matching tensors dual (original decode)
        static const std::string skip = [] {
            const char * e = getenv("GGML_SYCL_XE2MX_LO_INLINE_SKIP");
            return std::string(e ? e : "");
        }();
        if (!skip.empty() && t->name[0]) {
            size_t p = 0;
            while (p <= skip.size()) {
                size_t c = skip.find(',', p);
                if (c == std::string::npos) {
                    c = skip.size();
                }
                const std::string tok = skip.substr(p, c - p);
                if (!tok.empty() && strstr(t->name, tok.c_str())) {
                    return;
                }
                p = c + 1;
            }
        }
    }
    auto it = g_xe2mx_lo_w.find((const void *) t->data);
    if (it == g_xe2mx_lo_w.end() || !it->second.packed) {
        return;
    }
    if (xe2mx_lo_entry_inline(it->second)) {
        return;
    }
    void * pack = it->second.packed;
    const size_t pb = it->second.bytes;
    const size_t aos = ggml_nbytes(t);
    if (pb == 0 || pb > aos) {
        static int nref = 0;
        if (nref < 4) {
            nref++;
            fprintf(stderr, "xe2mx lo inline REFUSE #%d %s pb=%zu aos=%zu (need pb<=aos)\n",
                    id, t->name ? t->name : "?", pb, aos);
            fflush(stderr);
        }
        return;
    }
    // Dual only: pack is a separate persist (vnni 51). Never free t->data.
    if (pack != t->data && it->second.vnni == 51) {
        q.memcpy(t->data, pack, pb).wait();
        ggml_sycl_free_device(pack, q);
    } else if (pack != t->data) {
        fprintf(stderr, "xe2mx lo inline REFUSE #%d %s unexpected pack!=data vnni=%d\n",
                id, t->name ? t->name : "?", it->second.vnni);
        fflush(stderr);
        return;
    }
    it->second.packed = t->data;
    it->second.vnni = 52;
    // map key stays t->data; identity = device ptr (not weight-byte hash)
    static size_t g_lo_inline_reclaimed = 0;
    g_lo_inline_reclaimed += pb;
    GGML_SYCL_XE2_LOG( "xe2mx lo inline #%d %s N=%ld K=%ld free_pack=%zu total_reclaimed=%zu\n",
            id, t->name ? t->name : "?", (long) N, (long) K, pb, g_lo_inline_reclaimed);
    fflush(stderr);
}

size_t ggml_sycl_xe2mx_s8dw_nbytes(const ggml_tensor * t) {
    if (!ggml_sycl_xe2mx_s8dw_on(t)) {
        return 0;
    }
    return (size_t) t->ne[0] * (size_t) t->ne[1] + (size_t) t->ne[1] * sizeof(float);
}

bool ggml_sycl_xe2mx_s8dw_upload(ggml_tensor * t, const void * host, size_t size, sycl::queue & q) {
    if (!ggml_sycl_xe2mx_s8dw_on(t) || !host || !t->data || size != ggml_nbytes(t)) {
        return false;
    }
    const int64_t K = t->ne[0];
    const int64_t N = t->ne[1];
    const int64_t nb = K / 32;
    const size_t wbytes = (size_t) K * (size_t) N;
    const size_t bytes = wbytes + (size_t) N * sizeof(float);
    int8_t * hw = (int8_t *) malloc(bytes);
    if (!hw) {
        return false;
    }
    float * hsc = (float *) ((char *) hw + wbytes);
    const uint8_t * src = (const uint8_t *) host;
    for (int64_t n = 0; n < N; ++n) {
        float amax = 0.0f;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = src + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
            ggml_fp16_t dh;
            memcpy(&dh, blk, sizeof(dh));
            const float d = ggml_fp16_to_fp32(dh);
            for (int j = 0; j < 16; ++j) {
                const int vui = (int) blk[2 + j];
                const float x0 = ((float) (vui & 0xF) - 8.0f) * d;
                const float x1 = ((float) (vui >> 4) - 8.0f) * d;
                amax = fmaxf(amax, fabsf(x0));
                amax = fmaxf(amax, fabsf(x1));
            }
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        hsc[n] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t * col = hw + (size_t) n * (size_t) K;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = src + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
            ggml_fp16_t dh;
            memcpy(&dh, blk, sizeof(dh));
            const float d = ggml_fp16_to_fp32(dh);
            for (int j = 0; j < 16; ++j) {
                const int vui = (int) blk[2 + j];
                float x0 = ((float) (vui & 0xF) - 8.0f) * d * id;
                float x1 = ((float) (vui >> 4) - 8.0f) * d * id;
                int q0 = (int) (x0 + (x0 >= 0.0f ? 0.5f : -0.5f));
                int q1 = (int) (x1 + (x1 >= 0.0f ? 0.5f : -0.5f));
                if (q0 < -127) {
                    q0 = -127;
                }
                if (q0 > 127) {
                    q0 = 127;
                }
                if (q1 < -127) {
                    q1 = -127;
                }
                if (q1 > 127) {
                    q1 = 127;
                }
                col[ib * 32 + 2 * j] = (int8_t) q0;
                col[ib * 32 + 2 * j + 1] = (int8_t) q1;
            }
        }
    }
    q.memcpy(t->data, hw, bytes).wait();
    free(hw);
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) t->extra;
    if (extra) {
        extra->optimized_feature.reorder = true;
    }
    static int nup = 0;
    if (nup < 2) {
        nup++;
        GGML_SYCL_XE2_LOG( "xe2mx s8dw upload K=%ld N=%ld bytes=%zu\n", (long) K, (long) N, bytes);
        fflush(stderr);
    }
    return true;
}

static void xe2mx_s8r_pack(queue_ptr stream, const uint8_t * packed, int8_t * w, float * sc,
                             int64_t K, int64_t N, int vnni_layout) {
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float amax = 0.0f;
            for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui & 0xF) - 8.0f) * d));
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui >> 4) - 8.0f) * d));
                }
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sc[n] = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            int8_t * col = w + (size_t) n * (size_t) K;
            for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    float x0 = ((float) (vui & 0xF) - 8.0f) * d * id;
                    float x1 = ((float) (vui >> 4) - 8.0f) * d * id;
                    int q0 = (int) (x0 + (x0 >= 0.0f ? 0.5f : -0.5f));
                    int q1 = (int) (x1 + (x1 >= 0.0f ? 0.5f : -0.5f));
                    if (q0 < -127) {
                        q0 = -127;
                    }
                    if (q0 > 127) {
                        q0 = 127;
                    }
                    if (q1 < -127) {
                        q1 = -127;
                    }
                    if (q1 > 127) {
                        q1 = 127;
                    }
                    if (vnni_layout) {
                        const int k0 = ib * 32 + 2 * j;
                        const int k1 = ib * 32 + 2 * j + 1;
                        w[(size_t) (k0 / 4) * (size_t) N * 4u + (size_t) n * 4u + (size_t) (k0 % 4)] = (int8_t) q0;
                        w[(size_t) (k1 / 4) * (size_t) N * 4u + (size_t) n * 4u + (size_t) (k1 % 4)] = (int8_t) q1;
                    } else {
                        col[ib * 32 + 2 * j] = (int8_t) q0;
                        col[ib * 32 + 2 * j + 1] = (int8_t) q1;
                    }
                }
            }
        });
}

static bool xe2mx_s8r_cached(queue_ptr stream, const uint8_t * packed, int64_t K, int64_t N, xe2mx_s8r_view * out) {
    const int64_t s8rn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RN", 10240);
    if (!packed || !out || (K % 32) != 0 || N < s8rn) {
        return false;
    }
    const int extra = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RDN", 0) ? (K >= 16384) : (N < 16384);
    const int tag = 17;
    const size_t wbytes = (size_t) K * (size_t) N;
    const size_t bytes = wbytes + (size_t) N * sizeof(float);
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        out->w = (int8_t *) it->second.packed;
        out->sc = (float *) ((char *) it->second.packed + wbytes);
        return true;
    }
    if (it != g_xe2mx_wcache.end()) {
        return false;
    }
    if (bytes <= (1ull << 20)) {
        return false;
    }
    if (extra) {
        if (g_xe2mx_s8r_ex_used + bytes > xe2mx_s8r_ex_cap()) {
            return false;
        }
    } else if (g_xe2mx_wcache_used + bytes > (ggml_sycl_get_env("GGML_SYCL_XE2MX_S48DN", 0)
                   ? ((size_t) 10 * 1024 * 1024 * 1024) : XE2MX_WCACHE_CAP)) {
        return false;
    }
    void * persist = ggml_sycl_malloc_device(bytes, *stream);
    if (!persist) {
        return false;
    }
    int8_t * w = (int8_t *) persist;
    float * sc = (float *) ((char *) persist + wbytes);
    xe2mx_s8r_pack(stream, packed, w, sc, K, N, 0);
    if (extra) {
        g_xe2mx_s8r_ex_used += bytes;
    } else {
        g_xe2mx_wcache_used += bytes;
    }
    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
    out->w = w;
    out->sc = sc;
    return true;
}

static std::unordered_map<const void *, int> g_xe2mx_s48_idx;
static int g_xe2mx_s48_next = 0;
static std::vector<sycl::half> g_xe2mx_s48_uhost[64];
static int g_xe2mx_s48_uhost_ok[64];
static sycl::half * g_xe2mx_s48_udev = nullptr;
static int g_xe2mx_s48_udev_layer = -1;
static sycl::half * g_xe2mx_s48_ulayer[64];

static int xe2mx_s48_bind(const uint8_t * packed) {
    auto it = g_xe2mx_s48_idx.find((const void *) packed);
    if (it != g_xe2mx_s48_idx.end()) {
        return it->second / 2;
    }
    const int id = g_xe2mx_s48_next++;
    g_xe2mx_s48_idx[(const void *) packed] = id;
    return id / 2;
}

static int xe2mx_blk_layer(const ggml_tensor * t) {
    if (!t) {
        return -1;
    }
    const char * p = strstr(t->name, "blk.");
    if (!p) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (int) (*p - '0');
        p++;
    }
    return layer;
}

static const sycl::half * xe2mx_s48_U(queue_ptr stream, int layer, int64_t K) {
    if (layer < 0 || layer >= 64 || K != 5120) {
        return nullptr;
    }
    const size_t n = (size_t) K * (size_t) K;
    if (!g_xe2mx_s48_uhost_ok[layer]) {
        char path[512];
        const char * root = xe2mx_dump_root();
        FILE * f = nullptr;
        if (root) {
            snprintf(path, sizeof(path), "%s/s48_U/%02d.f16", root, layer);
            f = fopen(path, "rb");
        }
        if (!f) {
            return nullptr;
        }
        g_xe2mx_s48_uhost[layer].resize(n);
        const size_t got = fread(g_xe2mx_s48_uhost[layer].data(), sizeof(sycl::half), n, f);
        fclose(f);
        if (got != n) {
            g_xe2mx_s48_uhost[layer].clear();
            return nullptr;
        }
        g_xe2mx_s48_uhost_ok[layer] = 1;
    }
    if (g_xe2mx_s48_ulayer[layer]) {
        return g_xe2mx_s48_ulayer[layer];
    }
    sycl::half * slot = (sycl::half *) ggml_sycl_malloc_device(n * sizeof(sycl::half), *stream);
    if (slot) {
        stream->memcpy(slot, g_xe2mx_s48_uhost[layer].data(), n * sizeof(sycl::half));
        stream->wait();
        g_xe2mx_s48_ulayer[layer] = slot;
        return slot;
    }
    if (!g_xe2mx_s48_udev) {
        g_xe2mx_s48_udev = (sycl::half *) ggml_sycl_malloc_device(n * sizeof(sycl::half), *stream);
        if (!g_xe2mx_s48_udev) {
            return nullptr;
        }
    }
    if (g_xe2mx_s48_udev_layer != layer) {
        stream->memcpy(g_xe2mx_s48_udev, g_xe2mx_s48_uhost[layer].data(), n * sizeof(sycl::half));
        stream->wait();
        g_xe2mx_s48_udev_layer = layer;
    }
    return g_xe2mx_s48_udev;
}

struct xe2mx_s4_key {
    int M;
    int N;
    int K;
    int fl;
    uintptr_t q;
    bool operator==(const xe2mx_s4_key & o) const {
        return M == o.M && N == o.N && K == o.K && fl == o.fl && q == o.q;
    }
};

struct xe2mx_s4_key_hash {
    size_t operator()(const xe2mx_s4_key & k) const {
        return (size_t) k.M * 2654435761u ^ (size_t) k.N * 40503u ^ (size_t) k.K ^ ((size_t) k.fl << 17) ^ k.q;
    }
};

struct xe2mx_s4_prim {
    dnnl::matmul prim;
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::memory::desc sc_md;
    dnnl::memory::desc zp_md;
    dnnl::memory::desc sa_md;
    dnnl::memory::desc sp_md;
    dnnl::memory::desc bin_md;
};

static std::unordered_map<xe2mx_s4_key, xe2mx_s4_prim, xe2mx_s4_key_hash> g_xe2mx_s4_prims;
static std::unordered_map<xe2mx_s4_key, xe2mx_s4_prim, xe2mx_s4_key_hash> g_xe2mx_s48_prims;

static xe2mx_s4_prim * xe2mx_s48_get(ggml_backend_sycl_context & ctx, queue_ptr stream,
                                      int64_t M, int64_t N, int64_t K, int kind, int64_t g) {
    dnnl::engine eng = ctx.engine_dnnl(stream);
    xe2mx_s4_key key{ (int) M, (int) N, (int) K, kind + ((int) g << 8), (uintptr_t) stream };
    auto pit = g_xe2mx_s48_prims.find(key);
    if (pit != g_xe2mx_s48_prims.end()) {
        return &pit->second;
    }
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (kind == 0) {
        a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f16, {K, 1});
        b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f16, {N, 1});
        c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f16, {N, 1});
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    } else if (kind == 11) {
        a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f16, {K, 1});
        b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
        c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) K, 1},
            dnnl::memory::data_type::f32);
    } else if (kind == 12) {
        a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s8, {K, 1});
        b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
        c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
        attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) K, 1},
            dnnl::memory::data_type::f32);
    } else if (kind == 1 || kind == 4 || kind == 7 || kind == 8) {
        a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s8, {K, 1});
        b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ba);
        c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
        attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {K, 1}, dnnl::memory::data_type::f32);
        dnnl::post_ops po;
        if (kind == 7 || kind == 8) {
            po.append_sum(1.f);
        }
        if (kind == 4 || kind == 8) {
            dnnl::memory::desc bin_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
            po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 0.f);
            po.append_binary(dnnl::algorithm::binary_mul, bin_md);
            attr.set_post_ops(po);
        } else if (kind == 7) {
            attr.set_post_ops(po);
        }
    } else {
        a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s4, {K, 1});
        b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
        c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
        if (kind == 10) {
            attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {K, 1}, dnnl::memory::data_type::f32);
        } else if (kind == 9) {
            attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
            attr.set_scales(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::f32);
        } else if (kind == 5 || kind == 6) {
            attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {g, 1}, dnnl::memory::data_type::f32);
        } else {
            attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, g}, dnnl::memory::data_type::f32);
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {g, 1}, dnnl::memory::data_type::f32);
        }
        if (kind == 3 || kind == 6) {
            dnnl::memory::desc bin_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
            dnnl::post_ops po;
            po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 0.f);
            po.append_binary(dnnl::algorithm::binary_mul, bin_md);
            attr.set_post_ops(po);
        }
    }
    auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    xe2mx_s4_prim e;
    e.prim = dnnl::matmul(pd);
    e.a_md = a_md;
    e.b_md = b_md;
    e.c_md = c_md;
    e.sp_md = pd.scratchpad_desc();
    if (kind == 11) {
        e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
    } else if (kind == 12) {
        e.sa_md = dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1});
        e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
    } else if (kind == 1 || kind == 4 || kind == 7 || kind == 8 || kind == 10) {
        e.sa_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32, {1});
        e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
        if (kind == 10) {
            e.sa_md = dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1});
        }
    } else if (kind == 9) {
        e.sa_md = dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1});
        e.sc_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32, {1});
    } else if (kind == 5 || kind == 6) {
        e.sa_md = dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1});
        e.sc_md = dnnl::memory::desc({K / g, N}, dnnl::memory::data_type::f32, {N, 1});
    } else if (kind == 2 || kind == 3) {
        e.sa_md = dnnl::memory::desc({M, K / g}, dnnl::memory::data_type::f32, {K / g, 1});
        e.sc_md = dnnl::memory::desc({K / g, N}, dnnl::memory::data_type::f32, {N, 1});
    }
    if (kind == 3 || kind == 4 || kind == 6 || kind == 8) {
        e.bin_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
    }
    return &g_xe2mx_s48_prims.insert({ key, std::move(e) }).first->second;
}

static bool xe2mx_f16_mm(ggml_backend_sycl_context & ctx, queue_ptr stream,
                           const sycl::half * A, const sycl::half * B, sycl::half * C,
                           int64_t M, int64_t N, int64_t K) {
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        xe2mx_s4_prim * pe = xe2mx_s48_get(ctx, stream, M, N, K, 0, 1);
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pe->sp_md, eng, stream);
        pe->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(pe->a_md, eng, (void *) A)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(pe->b_md, eng, (void *) B)},
            {DNNL_ARG_DST, dnnl::memory(pe->c_md, eng, C)},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        });
        return true;
    } catch (...) {
        return false;
    }
}

static void xe2mx_s8_quant_f16(queue_ptr stream, const sycl::half * src, int8_t * w, float * sc,
                                 int64_t K, int64_t N) {
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * col = src + (size_t) n * (size_t) K;
            int8_t * dst = w + (size_t) n * (size_t) K;
            float amax = 0.0f;
            for (int k = lid; k < (int) K; k += (int) wg) {
                amax = sycl::fmax(amax, sycl::fabs((float) col[k]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sc[n] = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            for (int k = lid; k < (int) K; k += (int) wg) {
                float x = (float) col[k] * id;
                int q = (int) sycl_rint_i(x);
                if (q < -127) {
                    q = -127;
                }
                if (q > 127) {
                    q = 127;
                }
                dst[k] = (int8_t) q;
            }
        });
}

static void xe2mx_s44_unpack_f32(queue_ptr stream, const uint8_t * packed, float * dst, int64_t K, int64_t N);
static void xe2mx_silu_mul(queue_ptr stream, float * gate, const float * up, int64_t n);
static void xe2mx_s44_quant_f32(queue_ptr stream, const float * src, int8_t * w, float * sc, float * bsum,
                                  int64_t K, int64_t N, int64_t g);
static void xe2mx_s44_requant(queue_ptr stream, const uint8_t * packed, int8_t * w, float * sc,
                                int64_t K, int64_t N, int64_t g);
static void xe2mx_pack_s8_s4(queue_ptr stream, const int8_t * s8, uint8_t * s4, int64_t K, int64_t N);

static sycl::half * g_xe2mx_s48_xrot = nullptr;
static size_t g_xe2mx_s48_xrot_cap = 0;
static const sycl::half * g_xe2mx_s48_xrot_xh = nullptr;
static int64_t g_xe2mx_s48_xrot_M = 0;
static int64_t g_xe2mx_s48_xrot_K = 0;
static int g_xe2mx_s48_xrot_layer = -1;
static uint64_t g_xe2mx_s48_xrot_epoch = 0;
static uint8_t * g_xe2mx_s48_us4[64];
static float * g_xe2mx_s48_usc4[64];
static int g_xe2mx_s48_us4g[64];
static float * g_xe2mx_s48_xrotf = nullptr;
static size_t g_xe2mx_s48_xrotf_cap = 0;
static uint8_t * g_xe2mx_s48_ras4 = nullptr;
static size_t g_xe2mx_s48_ras4_cap = 0;
static float * g_xe2mx_s48_rsas4 = nullptr;
static size_t g_xe2mx_s48_rsas4_cap = 0;
static int8_t * g_xe2mx_s48_ra8 = nullptr;
static size_t g_xe2mx_s48_ra8_cap = 0;
static float * g_xe2mx_s48_rtmp = nullptr;
static size_t g_xe2mx_s48_rtmp_cap = 0;
static int8_t * g_xe2mx_s48_as8 = nullptr;
static size_t g_xe2mx_s48_as8_cap = 0;
static uint8_t * g_xe2mx_s48_as4 = nullptr;
static size_t g_xe2mx_s48_as4_cap = 0;
static float * g_xe2mx_s48_sas8 = nullptr;
static float * g_xe2mx_s48_sas4 = nullptr;
static size_t g_xe2mx_s48_sas4_cap = 0;
static const sycl::half * g_xe2mx_s48_axh = nullptr;
static const uint8_t * g_xe2mx_s48_apacked = nullptr;
static int64_t g_xe2mx_s48_aM = 0;
static int64_t g_xe2mx_s48_aK = 0;
static int g_xe2mx_s48_ar = 0;
static int g_xe2mx_s48_ag = 0;
static int g_xe2mx_s48_alayer = -1;
static uint64_t g_xe2mx_s48_a_epoch = 0;
static int8_t * g_xe2mx_s48_a8lo = nullptr;
static size_t g_xe2mx_s48_a8lo_cap = 0;
static float * g_xe2mx_s48_qtmp = nullptr;
static size_t g_xe2mx_s48_qtmp_cap = 0;

static int xe2mx_s48_fused() {
    return ggml_sycl_get_env("GGML_SYCL_XE2MX_S48FUSED", 0);
}

static int xe2mx_s48_r(int64_t K, int layer) {
    const int from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48FROM", 32);
    int r = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48R", 640);
    if (layer >= 0 && layer < from) {
        r = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48REARLY", 0);
    }
    if (r == 0) {
        return 0;
    }
    if (r < 64 || (r % 64) != 0 || r >= (int) K) {
        r = 640;
    }
    if (((K - (int64_t) r) % 128) != 0) {
        r = 640;
    }
    return r;
}

static int xe2mx_s48_g(int64_t Kl) {
    int g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48G", 128);
    if (g < 128 || (g % 16) != 0 || Kl < (int64_t) g || (Kl % (int64_t) g) != 0) {
        g = 128;
    }
    return g;
}

static int xe2mx_s48_dng(int64_t K) {
    int g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48DNG", 256);
    if (g < 64 || (g % 32) != 0 || K < (int64_t) g || (K % (int64_t) g) != 0) {
        g = 128;
    }
    return g;
}

static void xe2mx_s48_quant_w4_tens(queue_ptr stream, const float * src, int8_t * w, float * sc,
                                      int64_t K, int64_t N) {
    const size_t nlo = (size_t) K * (size_t) N;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            float a = 0.0f;
            for (size_t t = (size_t) lid; t < nlo; t += 256u) {
                a = sycl::fmax(a, sycl::fabs(src[t]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), a, sycl::maximum<float>());
            if (lid == 0) {
                sc[0] = gmax > 0.0f ? gmax / 7.0f : 0.0f;
            }
        });
    stream->parallel_for(sycl::range<1>(nlo), [=](sycl::id<1> gid) {
        const float s = sc[0];
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int q = (int) sycl_rint_i(src[gid[0]] * id);
        if (q < -8) {
            q = -8;
        }
        if (q > 7) {
            q = 7;
        }
        w[gid[0]] = (int8_t) q;
    });
}

static void xe2mx_s48_quant_w8(queue_ptr stream, const sycl::half * wr, int8_t * w, float * sc,
                                 int64_t K, int64_t N, int64_t r) {
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * col = wr + (size_t) n * (size_t) K;
            int8_t * dst = w + (size_t) n * (size_t) r;
            float amax = 0.0f;
            for (int k = lid; k < (int) r; k += (int) wg) {
                amax = sycl::fmax(amax, sycl::fabs((float) col[k]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sc[n] = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            for (int k = lid; k < (int) r; k += (int) wg) {
                int q = (int) sycl_rint_i((float) col[k] * id);
                if (q < -127) {
                    q = -127;
                }
                if (q > 127) {
                    q = 127;
                }
                dst[k] = (int8_t) q;
            }
        });
}

static bool xe2mx_s48_cached(ggml_backend_sycl_context & ctx, queue_ptr stream, const uint8_t * packed,
                               int64_t K, int64_t N, xe2mx_s8r_view * out, int layer) {
    if (!packed || !out || (K % 32) != 0) {
        return false;
    }
    const int64_t r = xe2mx_s48_r(K, layer);
    const int64_t Kl = r == 0 ? K : (K - r);
    const int64_t g = xe2mx_s48_g(Kl);
    if (Kl < g || (Kl % g) != 0) {
        return false;
    }
    const int pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48PN", 0);
    const int tw = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48TW", 0);
    const int tag = pn ? 51 : (tw ? 50 : (r == 0 ? 52 : 49));
    const size_t w8b = (size_t) N * (size_t) r;
    const size_t sc8b = r == 0 ? 0 : ((size_t) N * sizeof(float));
    const size_t w4b = (size_t) N * (size_t) (Kl / 2);
    const size_t sc4b = pn ? ((size_t) N * sizeof(float))
        : (tw ? sizeof(float) : ((size_t) (Kl / g) * (size_t) N * sizeof(float)));
    const size_t bytes = w8b + sc8b + w4b + sc4b;
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        out->w = (int8_t *) it->second.packed;
        out->sc = (float *) ((char *) it->second.packed + w8b);
        out->aux = (float *) ((char *) it->second.packed + w8b + sc8b + w4b);
        return true;
    }
    if (it != g_xe2mx_wcache.end()) {
        return false;
    }
    const int fused = xe2mx_s48_fused();
    const sycl::half * U = fused ? nullptr : xe2mx_s48_U(stream, layer, K);
    if (!fused && !U) {
        static int nu = 0;
        if (nu < 4) {
            nu++;
            GGML_SYCL_XE2_LOG( "xe2mx s48 cached no U layer=%d\n", layer);
            fflush(stderr);
        }
        return false;
    }
    if (g_xe2mx_wcache_used + bytes > XE2MX_WCACHE_CAP) {
        return false;
    }
    void * persist = ggml_sycl_malloc_device(bytes, *stream);
    if (!persist) {
        return false;
    }
    const size_t nk = (size_t) K * (size_t) N;
    sycl::half * wf = (sycl::half *) ggml_sycl_malloc_device(nk * sizeof(sycl::half), *stream);
    sycl::half * wr = (sycl::half *) ggml_sycl_malloc_device(nk * sizeof(sycl::half), *stream);
    float * f32 = (float *) ggml_sycl_malloc_device(nk * sizeof(float), *stream);
    if (!wf || !wr || !f32) {
        if (wf) {
            ggml_sycl_free_device(wf, *stream);
        }
        if (wr) {
            ggml_sycl_free_device(wr, *stream);
        }
        if (f32) {
            ggml_sycl_free_device(f32, *stream);
        }
        ggml_sycl_free_device(persist, *stream);
        return false;
    }
    xe2mx_s44_unpack_f32(stream, packed, f32, K, N);
    if (fused) {
        stream->parallel_for(sycl::range<1>(nk), [=](sycl::id<1> gid) {
            wr[gid[0]] = sycl::half(f32[gid[0]]);
        });
        stream->wait();
    } else {
        stream->parallel_for(sycl::range<1>(nk), [=](sycl::id<1> gid) {
            wf[gid[0]] = sycl::half(f32[gid[0]]);
        });
        stream->wait();
        if (!xe2mx_f16_mm(ctx, stream, wf, U, wr, N, K, K)) {
            static int nw = 0;
            if (nw < 4) {
                nw++;
                fprintf(stderr, "xe2mx s48 cached W@U fail layer=%d N=%ld K=%ld\n", layer, (long) N, (long) K);
                fflush(stderr);
            }
            ggml_sycl_free_device(wf, *stream);
            ggml_sycl_free_device(wr, *stream);
            ggml_sycl_free_device(f32, *stream);
            ggml_sycl_free_device(persist, *stream);
            return false;
        }
        stream->wait();
    }
    int8_t * w8 = (int8_t *) persist;
    float * sc8 = r == 0 ? nullptr : (float *) ((char *) persist + w8b);
    uint8_t * w4 = (uint8_t *) ((char *) persist + w8b + sc8b);
    float * sc4 = (float *) ((char *) persist + w8b + sc8b + w4b);
    if (r > 0) {
        xe2mx_s48_quant_w8(stream, wr, w8, sc8, K, N, r);
    }
    const size_t nlo = (size_t) N * (size_t) Kl;
    stream->parallel_for(sycl::range<1>(nlo), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        const size_t k = t % (size_t) Kl;
        const size_t n = t / (size_t) Kl;
        f32[n * (size_t) Kl + k] = (float) wr[n * (size_t) K + (size_t) r + k];
    });
    int8_t * t8 = (int8_t *) wf;
    if (pn) {
        xe2mx_s44_quant_f32(stream, f32, t8, sc4, nullptr, Kl, N, Kl);
    } else if (tw) {
        xe2mx_s48_quant_w4_tens(stream, f32, t8, sc4, Kl, N);
    } else {
        xe2mx_s44_quant_f32(stream, f32, t8, sc4, nullptr, Kl, N, g);
    }
    xe2mx_pack_s8_s4(stream, t8, w4, Kl, N);
    stream->wait();
    ggml_sycl_free_device(wf, *stream);
    ggml_sycl_free_device(wr, *stream);
    ggml_sycl_free_device(f32, *stream);
    g_xe2mx_wcache_used += bytes;
    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
    out->w = w8;
    out->sc = sc8;
    out->aux = sc4;
    return true;
}

static void xe2mx_s4dn_requant(queue_ptr stream, const uint8_t * packed, int8_t * w, float * sc,
                                 int64_t K, int64_t N, int64_t g) {
    const int64_t nb = K / 32;
    const int64_t ng = K / g;
    const int bpg = (int) (g / 32);
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const size_t wg = 32;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * (size_t) ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int blk = (int) it.get_group(0);
            const int n = blk / (int) ng;
            const int ig = blk - n * (int) ng;
            const int lid = (int) it.get_local_id(0);
            int8_t * col = w + (size_t) n * (size_t) K;
            float amax = 0.0f;
            const int ib0 = ig * bpg;
            const int ib1 = ib0 + bpg;
            for (int ib = ib0 + lid; ib < ib1; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui & 0xF) - 8.0f) * d));
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui >> 4) - 8.0f) * d));
                }
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sc[(size_t) ig * (size_t) N + (size_t) n] = gmax > 0.0f ? gmax / 8.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 8.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            for (int ib = ib0 + lid; ib < ib1; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    float x0 = ((float) (vui & 0xF) - 8.0f) * d * id;
                    float x1 = ((float) (vui >> 4) - 8.0f) * d * id;
                    int q0 = (int) sycl_rint_i(x0);
                    int q1 = (int) sycl_rint_i(x1);
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
                    col[ib * 32 + 2 * j] = (int8_t) q0;
                    col[ib * 32 + 2 * j + 1] = (int8_t) q1;
                }
            }
        });
}

static void * g_xe2mx_s4dn_slot = nullptr;
static size_t g_xe2mx_s4dn_slot_cap = 0;
static const uint8_t * g_xe2mx_s4dn_src = nullptr;
static int64_t g_xe2mx_s4dn_g = 0;
static int8_t * g_xe2mx_s4dn_t8 = nullptr;
static size_t g_xe2mx_s4dn_t8_cap = 0;
static uint8_t * g_xe2mx_s4dn_psrc = nullptr;
static size_t g_xe2mx_s4dn_psrc_cap = 0;

static bool xe2mx_s4dn_cached(queue_ptr stream, const uint8_t * packed,
                                int64_t K, int64_t N, xe2mx_s8r_view * out) {
    if (!packed || !out || N != 5120 || K != 17408 || (K % 32) != 0) {
        return false;
    }
    const int64_t g = xe2mx_s48_dng(K);
    if ((K % g) != 0 || (g % 32) != 0) {
        return false;
    }
    const size_t w4b = (size_t) N * (size_t) (K / 2);
    const size_t sc4b = (size_t) (K / g) * (size_t) N * sizeof(float);
    const size_t bytes = w4b + sc4b;
    const size_t nk = (size_t) K * (size_t) N;
    const int tag = 53;
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        out->w = (int8_t *) it->second.packed;
        out->sc = nullptr;
        out->aux = (float *) ((char *) it->second.packed + w4b);
        return true;
    }
    if (!g_xe2mx_s4dn_t8 || g_xe2mx_s4dn_t8_cap < nk) {
        if (g_xe2mx_s4dn_t8) {
            ggml_sycl_free_device(g_xe2mx_s4dn_t8, *stream);
            g_xe2mx_s4dn_t8 = nullptr;
            g_xe2mx_s4dn_t8_cap = 0;
        }
        g_xe2mx_s4dn_t8 = (int8_t *) ggml_sycl_malloc_device(nk, *stream);
        if (!g_xe2mx_s4dn_t8) {
            return false;
        }
        g_xe2mx_s4dn_t8_cap = nk;
    }
    const size_t pb = (size_t) N * (size_t) (K / 32) * 16u + (size_t) (K / 32) * (size_t) N * sizeof(sycl::half);
    if (!g_xe2mx_s4dn_psrc || g_xe2mx_s4dn_psrc_cap < pb) {
        if (g_xe2mx_s4dn_psrc) {
            ggml_sycl_free_device(g_xe2mx_s4dn_psrc, *stream);
            g_xe2mx_s4dn_psrc = nullptr;
            g_xe2mx_s4dn_psrc_cap = 0;
        }
        g_xe2mx_s4dn_psrc = (uint8_t *) ggml_sycl_malloc_device(pb, *stream);
        if (!g_xe2mx_s4dn_psrc) {
            return false;
        }
        g_xe2mx_s4dn_psrc_cap = pb;
    }
    stream->memcpy(g_xe2mx_s4dn_psrc, packed, pb);
    void * dst = nullptr;
    int extra = 0;
    int mainc = 0;
    if (it == g_xe2mx_wcache.end() && g_xe2mx_wcache_used + bytes <= XE2MX_WCACHE_CAP) {
        dst = ggml_sycl_malloc_device(bytes, *stream);
        mainc = dst ? 1 : 0;
    }
    if (!dst && it == g_xe2mx_wcache.end() && g_xe2mx_s8r_ex_used + bytes <= xe2mx_s8r_ex_cap()) {
        dst = ggml_sycl_malloc_device(bytes, *stream);
        extra = dst ? 1 : 0;
    }
    if (!dst) {
        if (!g_xe2mx_s4dn_slot || g_xe2mx_s4dn_slot_cap < bytes) {
            if (g_xe2mx_s4dn_slot) {
                ggml_sycl_free_device(g_xe2mx_s4dn_slot, *stream);
                g_xe2mx_s4dn_slot = nullptr;
                g_xe2mx_s4dn_slot_cap = 0;
            }
            g_xe2mx_s4dn_slot = ggml_sycl_malloc_device(bytes, *stream);
            if (!g_xe2mx_s4dn_slot) {
                return false;
            }
            g_xe2mx_s4dn_slot_cap = bytes;
        }
        dst = g_xe2mx_s4dn_slot;
    }
    uint8_t * w4 = (uint8_t *) dst;
    float * sc4 = (float *) ((char *) dst + w4b);
    xe2mx_s4dn_requant(stream, g_xe2mx_s4dn_psrc, g_xe2mx_s4dn_t8, sc4, K, N, g);
    xe2mx_pack_s8_s4(stream, g_xe2mx_s4dn_t8, w4, K, N);
    if (mainc) {
        g_xe2mx_wcache_used += bytes;
        g_xe2mx_wcache[(const void *) packed] = { dst, bytes, K, N, tag };
    } else if (extra) {
        g_xe2mx_s8r_ex_used += bytes;
        g_xe2mx_wcache[(const void *) packed] = { dst, bytes, K, N, tag };
    } else {
        g_xe2mx_s4dn_src = packed;
        g_xe2mx_s4dn_g = g;
    }
    out->w = (int8_t *) dst;
    out->sc = nullptr;
    out->aux = sc4;
    return true;
}

static void xe2mx_s48_quant_a(queue_ptr stream, const sycl::half * xh, int8_t * As8, float * sas8,
                                uint8_t * As4, float * sas4, int8_t * A8lo, float * tmx,
                                int64_t M, int64_t K, int64_t r, int64_t g, int row_a) {
    const int64_t Kl = r == 0 ? K : (K - r);
    const int64_t ng = Kl / g;
    const size_t wg = 128;
    if (r > 0) {
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            float amax = 0.0f;
            for (int k = lid; k < (int) r; k += (int) wg) {
                amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
            }
            tmx[m] = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        });
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            float a = 0.0f;
            for (int m = lid; m < (int) M; m += 256) {
                a = sycl::fmax(a, tmx[m]);
            }
            const float gv = sycl::reduce_over_group(it.get_group(), a, sycl::maximum<float>());
            if (lid == 0) {
                sas8[0] = gv > 0.0f ? gv / 127.0f : 0.0f;
            }
        });
    stream->parallel_for(sycl::range<1>((size_t) M * (size_t) r), [=](sycl::id<1> gid) {
        const float s = sas8[0];
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        const size_t t = (size_t) gid[0];
        const size_t k = t % (size_t) r;
        const size_t m = t / (size_t) r;
        int q = (int) sycl_rint_i((float) xh[m * (size_t) K + k] * id);
        if (q < -127) {
            q = -127;
        }
        if (q > 127) {
            q = 127;
        }
        As8[t] = (int8_t) q;
    });
    }
    if (row_a) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int m = (int) it.get_group(0);
                const int lid = (int) it.get_local_id(0);
                const sycl::half * row = xh + (size_t) m * (size_t) K + (size_t) r;
                int8_t * arow = A8lo + (size_t) m * (size_t) Kl;
                float amax = 0.0f;
                for (int k = lid; k < (int) Kl; k += (int) wg) {
                    amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float s = gmax > 0.0f ? gmax / 8.0f : 0.0f;
                if (lid == 0) {
                    sas4[m] = s;
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = lid; k < (int) Kl; k += (int) wg) {
                    int q = (int) sycl_rint_i((float) row[k] * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k] = (int8_t) q;
                }
            });
        stream->parallel_for(sycl::range<1>((size_t) M * (size_t) (Kl / 2)), [=](sycl::id<1> gid) {
            const size_t t = (size_t) gid[0];
            const size_t k2 = t % (size_t) (Kl / 2);
            const size_t m = t / (size_t) (Kl / 2);
            const uint8_t u0 = (uint8_t) A8lo[m * (size_t) Kl + k2 * 2u];
            const uint8_t u1 = (uint8_t) A8lo[m * (size_t) Kl + k2 * 2u + 1u];
            As4[t] = (uint8_t) ((u0 & 0x0Fu) | (uint8_t) ((u1 & 0x0Fu) << 4));
        });
    } else {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int blk = (int) it.get_group(0);
                const int m = blk / (int) ng;
                const int ig = blk - m * (int) ng;
                const int lid = (int) it.get_local_id(0);
                const sycl::half * row = xh + (size_t) m * (size_t) K + (size_t) r + (size_t) ig * (size_t) g;
                volatile int8_t * arow = A8lo + (size_t) m * (size_t) Kl + (size_t) ig * (size_t) g;
                float amax = 0.0f;
                for (int k = lid; k < (int) g; k += (int) wg) {
                    amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float s = gmax > 0.0f ? gmax / 8.0f : 0.0f;
                if (lid == 0) {
                    sas4[(size_t) m * (size_t) ng + (size_t) ig] = s;
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = lid; k < (int) g; k += (int) wg) {
                    int q = (int) sycl_rint_i((float) row[k] * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k] = (int8_t) q;
                }
                sycl::group_barrier(it.get_group());
                const int np = (int) g / 2;
                uint8_t * dst = As4 + (size_t) m * (size_t) (Kl / 2) + (size_t) ig * (size_t) np;
                for (int k2 = lid; k2 < np; k2 += (int) wg) {
                    const uint8_t u0 = (uint8_t) arow[k2 * 2];
                    const uint8_t u1 = (uint8_t) arow[k2 * 2 + 1];
                    dst[k2] = (uint8_t) ((u0 & 0x0Fu) | (uint8_t) ((u1 & 0x0Fu) << 4));
                }
            });
    }
}

static bool xe2mx_s48_two(ggml_backend_sycl_context & ctx, queue_ptr stream,
                            const int8_t * As8, const float * sas8, const int8_t * Ws8, const float * sws8,
                            const uint8_t * As4, const float * sas4, const uint8_t * Ws4, const float * sws4,
                            float * C, int64_t M, int64_t N, int64_t r, int64_t Kl, int64_t g,
                            const float * glu_up) {
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        const int hi_only = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48HI", 0);
        const int lo_only = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48LO", 0);
        const int pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48PN", 0);
        const int tw = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48TW", 0);
        const int row_a = (pn || tw || (r == 0 && Kl == 17408)) ? 1 : ggml_sycl_get_env("GGML_SYCL_XE2MX_S48RA", 0);
        const int k4 = pn ? 10 : (tw ? 9 : (row_a ? 5 : 2));
        if ((lo_only || r == 0) && !hi_only) {
            int k4l = k4;
            if (glu_up && k4 == 2) {
                k4l = 3;
            } else if (glu_up && k4 == 5) {
                k4l = 6;
            }
            xe2mx_s4_prim * p4 = xe2mx_s48_get(ctx, stream, M, N, Kl, k4l, g);
            if (!p4) {
                return false;
            }
            dnnl::memory sp4 = ctx.get_scratchpad_mem(p4->sp_md, eng, stream);
            std::unordered_map<int, dnnl::memory> a4s = {
                {DNNL_ARG_SRC, dnnl::memory(p4->a_md, eng, (void *) As4)},
                {DNNL_ARG_WEIGHTS, dnnl::memory(p4->b_md, eng, (void *) Ws4)},
                {DNNL_ARG_DST, dnnl::memory(p4->c_md, eng, C)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p4->sa_md, eng, (void *) sas4)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p4->sc_md, eng, (void *) sws4)},
                {DNNL_ARG_SCRATCHPAD, sp4},
            };
            if (k4l == 3 || k4l == 6) {
                a4s.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                    dnnl::memory(p4->bin_md, eng, (void *) glu_up)});
            }
            p4->prim.execute(strm, a4s);
            if (glu_up && k4l != 3 && k4l != 6) {
                xe2mx_silu_mul(stream, C, glu_up, M * N);
            }
            return true;
        }
        const int k8 = hi_only ? (glu_up ? 4 : 1) : (glu_up ? 8 : 7);
        xe2mx_s4_prim * p8 = xe2mx_s48_get(ctx, stream, M, N, r, k8, g);
        xe2mx_s4_prim * p4 = hi_only ? nullptr : xe2mx_s48_get(ctx, stream, M, N, Kl, k4, g);
        if (!p8 || (!hi_only && !p4)) {
            return false;
        }
        if (p4 && p4->sp_md.get_size() > p8->sp_md.get_size()) {
            ctx.get_scratchpad_mem(p4->sp_md, eng, stream);
        }
        if (p4) {
            dnnl::memory sp4 = ctx.get_scratchpad_mem(p4->sp_md, eng, stream);
            p4->prim.execute(strm, {
                {DNNL_ARG_SRC, dnnl::memory(p4->a_md, eng, (void *) As4)},
                {DNNL_ARG_WEIGHTS, dnnl::memory(p4->b_md, eng, (void *) Ws4)},
                {DNNL_ARG_DST, dnnl::memory(p4->c_md, eng, C)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p4->sa_md, eng, (void *) sas4)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p4->sc_md, eng, (void *) sws4)},
                {DNNL_ARG_SCRATCHPAD, sp4},
            });
        }
        dnnl::memory sp8 = ctx.get_scratchpad_mem(p8->sp_md, eng, stream);
        std::unordered_map<int, dnnl::memory> a8s = {
            {DNNL_ARG_SRC, dnnl::memory(p8->a_md, eng, (void *) As8)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p8->b_md, eng, (void *) Ws8)},
            {DNNL_ARG_DST, dnnl::memory(p8->c_md, eng, C)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p8->sa_md, eng, (void *) sas8)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p8->sc_md, eng, (void *) sws8)},
            {DNNL_ARG_SCRATCHPAD, sp8},
        };
        if (k8 == 4) {
            a8s.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                dnnl::memory(p8->bin_md, eng, (void *) glu_up)});
        } else if (k8 == 8) {
            a8s.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                dnnl::memory(p8->bin_md, eng, (void *) glu_up)});
        }
        p8->prim.execute(strm, a8s);
        return true;
    } catch (const dnnl::error & e) {
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx s48 two dnnl: %s M=%d N=%d r=%d Kl=%d glu=%d\n",
                    e.what(), (int) M, (int) N, (int) r, (int) Kl, glu_up ? 1 : 0);
            fflush(stderr);
        }
        return false;
    } catch (...) {
        return false;
    }
}

static int g_xe2mx_splitk_dead = 0;

struct xe2mx_sk_key {
    int M;
    int N;
    int Ks;
    int Kfull;
    int g;
    int sum;
    int acc;
    uintptr_t q;
    bool operator==(const xe2mx_sk_key & o) const {
        return M == o.M && N == o.N && Ks == o.Ks && Kfull == o.Kfull
            && g == o.g && sum == o.sum && acc == o.acc && q == o.q;
    }
};

struct xe2mx_sk_key_hash {
    size_t operator()(const xe2mx_sk_key & k) const {
        return (size_t) k.M * 2654435761u ^ (size_t) k.N * 40503u ^ (size_t) k.Ks
            ^ ((size_t) k.Kfull << 8) ^ ((size_t) k.g << 16) ^ ((size_t) k.sum << 3)
            ^ ((size_t) k.acc << 5) ^ k.q;
    }
};

static std::unordered_map<xe2mx_sk_key, xe2mx_s4_prim, xe2mx_sk_key_hash> g_xe2mx_sk_prims;

static xe2mx_s4_prim * xe2mx_s44_sk_get(ggml_backend_sycl_context & ctx, queue_ptr stream,
                                         int64_t M, int64_t N, int64_t Ks, int64_t Kfull,
                                         int64_t g, int sum, int acc) {
    dnnl::engine eng = ctx.engine_dnnl(stream);
    xe2mx_sk_key key{ (int) M, (int) N, (int) Ks, (int) Kfull, (int) g, sum, acc, (uintptr_t) stream };
    auto pit = g_xe2mx_sk_prims.find(key);
    if (pit != g_xe2mx_sk_prims.end()) {
        return &pit->second;
    }
    dnnl::memory::desc a_md({M, Ks}, dnnl::memory::data_type::s4, {Ks, 1});
    dnnl::memory::desc b_md({Ks, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
    dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, g}, dnnl::memory::data_type::f32);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {g, 1}, dnnl::memory::data_type::f32);
    if (acc == 1) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::s32);
    } else if (acc == 2) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::any);
    } else if (acc == 3) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::relaxed);
    }
    if (sum) {
        dnnl::post_ops po;
        po.append_sum(1.f);
        attr.set_post_ops(po);
    }
    auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    const size_t w4b = (size_t) N * (size_t) (Ks / 2);
    if (pd.weights_desc().get_size() > w4b) {
        GGML_SYCL_XE2_LOG( "xe2mx splitk skip psz=%zu w4b=%zu Ks=%ld N=%ld\n",
                pd.weights_desc().get_size(), w4b, (long) Ks, (long) N);
        fflush(stderr);
        return nullptr;
    }
    xe2mx_s4_prim e;
    e.prim = dnnl::matmul(pd);
    e.a_md = a_md;
    e.b_md = b_md;
    e.c_md = c_md;
    e.sp_md = pd.scratchpad_desc();
    e.sa_md = dnnl::memory::desc({M, Ks / g}, dnnl::memory::data_type::f32, {Ks / g, 1});
    e.sc_md = dnnl::memory::desc({Ks / g, N}, dnnl::memory::data_type::f32, {N, 1});
    static int nsk = 0;
    if (nsk < 8) {
        nsk++;
        GGML_SYCL_XE2_LOG( "xe2mx splitk impl=%s M=%ld N=%ld Ks=%ld K=%ld g=%ld sum=%d psz=%zu\n",
                pd.impl_info_str(), (long) M, (long) N, (long) Ks, (long) Kfull, (long) g, sum,
                pd.weights_desc().get_size());
        fflush(stderr);
    }
    return &g_xe2mx_sk_prims.insert({ key, std::move(e) }).first->second;
}

static bool xe2mx_s44_splitk(ggml_backend_sycl_context & ctx, queue_ptr stream,
                              const float * xf, const sycl::half * xh,
                              const uint8_t * w4, const float * scw, float * C,
                              int64_t M, int64_t N, int64_t K, int64_t g, int acc) {
    if (g_xe2mx_splitk_dead || !stream || !w4 || !scw || !C || (!xf && !xh)) {
        return false;
    }
    if (M < 1024 || K < 16384 || g < 64 || (K % g) != 0 || (N % 16) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    const int64_t ng0 = ng / 2;
    if (ng0 < 1 || ng0 >= ng) {
        return false;
    }
    const int64_t K0 = ng0 * g;
    const int64_t K1 = K - K0;
    const int ovl = ggml_sycl_get_env("GGML_SYCL_XE2MX_LOSPLITK", 0) >= 2 ? 1 : 0;
    queue_ptr q1 = stream;
    if (ovl) {
        static sycl::queue * side = nullptr;
        static queue_ptr side_seen = nullptr;
        if (!side || side_seen != stream) {
            delete side;
            side = new sycl::queue(stream->get_context(), stream->get_device(),
                                    sycl::property_list{ sycl::property::queue::in_order() });
            side_seen = stream;
        }
        q1 = side;
    }
    const size_t a0n = (size_t) M * (size_t) (K0 / 2);
    const size_t a1n = (size_t) M * (size_t) (K1 / 2);
    const size_t w0n = (size_t) N * (size_t) (K0 / 2);
    const size_t w1n = (size_t) N * (size_t) (K1 / 2);
    ggml_sycl_pool_alloc<uint8_t> A0(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> A1(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> W0(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> W1(ctx.pool());
    ggml_sycl_pool_alloc<float> S0(ctx.pool());
    ggml_sycl_pool_alloc<float> S1(ctx.pool());
    uint8_t * a0 = A0.alloc(a0n);
    uint8_t * a1 = A1.alloc(a1n);
    uint8_t * cw0 = W0.alloc(w0n);
    uint8_t * cw1 = W1.alloc(w1n);
    float * sa0 = S0.alloc((size_t) M * (size_t) ng0);
    float * sa1 = S1.alloc((size_t) M * (size_t) (ng - ng0));
    if (!a0 || !a1 || !cw0 || !cw1 || !sa0 || !sa1) {
        return false;
    }
    const size_t rowf = (size_t) (K / 2);
    const size_t rows0 = (size_t) (K0 / 2);
    const size_t rows1 = (size_t) (K1 / 2);
    int pok0 = 0;
    if (xf) {
        pok0 = xe2mx_quant_a_s4_nibble_had_f32_rte_off(stream, xf, a0, sa0, M, K, 0, K0, g) ? 1 : 0;
    } else {
        pok0 = xe2mx_quant_a_s4_nibble_had_off(stream, xh, a0, sa0, M, K, 0, K0, g) ? 1 : 0;
    }
    if (!pok0) {
        return false;
    }
    stream->parallel_for(sycl::range<1>(w0n), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        const size_t n = t / rows0;
        const size_t i = t - n * rows0;
        cw0[n * rows0 + i] = w4[n * rowf + i];
    });
    try {
        xe2mx_s4_prim * p0 = xe2mx_s44_sk_get(ctx, stream, M, N, K0, K, g, 0, acc);
        xe2mx_s4_prim * p1 = xe2mx_s44_sk_get(ctx, stream, M, N, K1, K, g, 1, acc);
        if (!p0 || !p1) {
            return false;
        }
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        if (p1->sp_md.get_size() > p0->sp_md.get_size()) {
            ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
        }
        dnnl::memory sp0 = ctx.get_scratchpad_mem(p0->sp_md, eng, stream);
        p0->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(p0->a_md, eng, a0)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p0->b_md, eng, cw0)},
            {DNNL_ARG_DST, dnnl::memory(p0->c_md, eng, C)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p0->sa_md, eng, sa0)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p0->sc_md, eng, (void *) scw)},
            {DNNL_ARG_SCRATCHPAD, sp0},
        });
        int pok1 = 0;
        if (xf) {
            pok1 = xe2mx_quant_a_s4_nibble_had_f32_rte_off(q1, xf, a1, sa1, M, K, K0, K1, g) ? 1 : 0;
        } else {
            pok1 = xe2mx_quant_a_s4_nibble_had_off(q1, xh, a1, sa1, M, K, K0, K1, g) ? 1 : 0;
        }
        if (!pok1) {
            if (ovl) {
                q1->wait();
            }
            return false;
        }
        q1->parallel_for(sycl::range<1>(w1n), [=](sycl::id<1> gid) {
            const size_t t = (size_t) gid[0];
            const size_t n = t / rows1;
            const size_t i = t - n * rows1;
            cw1[n * rows1 + i] = w4[n * rowf + rows0 + i];
        });
        if (ovl) {
            q1->wait();
        }
        void * sc1 = (void *) (scw + (size_t) ng0 * (size_t) N);
        dnnl::memory sp1 = ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
        p1->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(p1->a_md, eng, a1)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p1->b_md, eng, cw1)},
            {DNNL_ARG_DST, dnnl::memory(p1->c_md, eng, C)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p1->sa_md, eng, sa1)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p1->sc_md, eng, sc1)},
            {DNNL_ARG_SCRATCHPAD, sp1},
        });
        return true;
    } catch (const dnnl::error & e) {
        g_xe2mx_splitk_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx splitk dnnl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (const sycl::exception & e) {
        g_xe2mx_splitk_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx splitk sycl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (...) {
        g_xe2mx_splitk_dead = 1;
        return false;
    }
}

static int g_xe2mx_grpacc_dead = 0;

struct xe2mx_grp_key {
    int M;
    int N;
    int g;
    int Kfull;
    int sum;
    int acc;
    int arow;
    uintptr_t q;
    bool operator==(const xe2mx_grp_key & o) const {
        return M == o.M && N == o.N && g == o.g && Kfull == o.Kfull
            && sum == o.sum && acc == o.acc && arow == o.arow && q == o.q;
    }
};

struct xe2mx_grp_key_hash {
    size_t operator()(const xe2mx_grp_key & k) const {
        return (size_t) k.M * 2654435761u ^ (size_t) k.N * 40503u ^ ((size_t) k.g << 8)
            ^ ((size_t) k.Kfull << 16) ^ ((size_t) k.sum << 3) ^ ((size_t) k.acc << 5)
            ^ ((size_t) k.arow << 7) ^ k.q;
    }
};

static std::unordered_map<xe2mx_grp_key, xe2mx_s4_prim, xe2mx_grp_key_hash> g_xe2mx_grp_prims;
static std::unordered_set<xe2mx_grp_key, xe2mx_grp_key_hash> g_xe2mx_grp_fail;

static xe2mx_s4_prim * xe2mx_s44_grp_get(ggml_backend_sycl_context & ctx, queue_ptr stream,
                                          int64_t M, int64_t N, int64_t g, int64_t Kfull,
                                          int sum, int acc, int arow) {
    dnnl::engine eng = ctx.engine_dnnl(stream);
    xe2mx_grp_key key{ (int) M, (int) N, (int) g, (int) Kfull, sum, acc, arow, (uintptr_t) stream };
    if (g_xe2mx_grp_fail.count(key)) {
        return nullptr;
    }
    auto pit = g_xe2mx_grp_prims.find(key);
    if (pit != g_xe2mx_grp_prims.end()) {
        return &pit->second;
    }
    dnnl::memory::desc a_md({M, g}, dnnl::memory::data_type::s4, {Kfull, 1});
    dnnl::memory::desc b_md({g, N}, dnnl::memory::data_type::s4, {1, Kfull});
    dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {g, 1}, dnnl::memory::data_type::f32);
    if (acc == 1) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::s32);
    } else if (acc == 2) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::any);
    } else if (acc == 3) {
        attr.set_accumulation_mode(dnnl::accumulation_mode::relaxed);
    }
    if (sum) {
        dnnl::post_ops po;
        po.append_sum(1.f);
        attr.set_post_ops(po);
    }
    try {
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        xe2mx_s4_prim e;
        e.prim = dnnl::matmul(pd);
        e.a_md = a_md;
        e.b_md = b_md;
        e.c_md = c_md;
        e.sp_md = pd.scratchpad_desc();
        const int64_t ng = Kfull / g;
        e.sa_md = arow
            ? dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1})
            : dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {ng});
        e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
        static int ngrp = 0;
        if (ngrp < 8) {
            ngrp++;
            GGML_SYCL_XE2_LOG( "xe2mx grpacc impl=%s M=%ld N=%ld g=%ld K=%ld arow=%d sum=%d psz=%zu\n",
                    pd.impl_info_str(), (long) M, (long) N, (long) g, (long) Kfull, arow, sum,
                    pd.weights_desc().get_size());
            fflush(stderr);
        }
        return &g_xe2mx_grp_prims.insert({ key, std::move(e) }).first->second;
    } catch (const dnnl::error & e) {
        static int nfail = 0;
        if (nfail < 8) {
            nfail++;
            GGML_SYCL_XE2_LOG( "xe2mx grpacc pd: %s M=%ld N=%ld g=%ld K=%ld arow=%d sum=%d\n",
                    e.what(), (long) M, (long) N, (long) g, (long) Kfull, arow, sum);
            fflush(stderr);
        }
        g_xe2mx_grp_fail.insert(key);
        return nullptr;
    }
}

static bool xe2mx_s44_grpacc(ggml_backend_sycl_context & ctx, queue_ptr stream,
                              const uint8_t * Ap, const float * sap,
                              const uint8_t * w4, const float * scw, float * C,
                              int64_t M, int64_t N, int64_t K, int64_t g, int acc,
                              int arow, const float * glu_up) {
    if (g_xe2mx_grpacc_dead || !stream || !Ap || !sap || !w4 || !scw || !C) {
        return false;
    }
    if (M < 1024 || g < 128 || (K % g) != 0 || (N % 16) != 0 || (g % 2) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    if (ng < 2) {
        return false;
    }
    try {
        xe2mx_s4_prim * p0 = xe2mx_s44_grp_get(ctx, stream, M, N, g, K, 0, acc, arow);
        xe2mx_s4_prim * p1 = xe2mx_s44_grp_get(ctx, stream, M, N, g, K, 1, acc, arow);
        if (!p0 || !p1) {
            return false;
        }
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        if (p1->sp_md.get_size() > p0->sp_md.get_size()) {
            ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
        }
        dnnl::memory sp = ctx.get_scratchpad_mem(p0->sp_md, eng, stream);
        const size_t goff = (size_t) (g / 2);
        for (int64_t ig = 0; ig < ng; ig++) {
            uint8_t * ap = (uint8_t *) Ap + (size_t) ig * goff;
            uint8_t * bp = (uint8_t *) w4 + (size_t) ig * goff;
            float * sa = arow ? (float *) sap : ((float *) sap + (size_t) ig);
            xe2mx_s4_prim * pe = (ig == 0) ? p0 : p1;
            if (ig == 1) {
                sp = ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
            }
            pe->prim.execute(strm, {
                {DNNL_ARG_SRC, dnnl::memory(pe->a_md, eng, ap)},
                {DNNL_ARG_WEIGHTS, dnnl::memory(pe->b_md, eng, bp)},
                {DNNL_ARG_DST, dnnl::memory(pe->c_md, eng, C)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(pe->sa_md, eng, sa)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(pe->sc_md, eng, (void *) scw)},
                {DNNL_ARG_SCRATCHPAD, sp},
            });
        }
        if (glu_up) {
            xe2mx_silu_mul(stream, C, glu_up, M * N);
        }
        return true;
    } catch (const dnnl::error & e) {
        g_xe2mx_grpacc_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx grpacc dnnl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (const sycl::exception & e) {
        g_xe2mx_grpacc_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx grpacc sycl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (...) {
        g_xe2mx_grpacc_dead = 1;
        return false;
    }
}

static int g_xe2mx_splitm_dead = 0;

static bool xe2mx_s44_splitm(ggml_backend_sycl_context & ctx, queue_ptr stream,
                              const float * xf, const sycl::half * xh,
                              const uint8_t * w4, const float * scw, float * C,
                              int64_t M, int64_t N, int64_t K, int64_t g, int acc) {
    if (g_xe2mx_splitm_dead || !stream || !w4 || !scw || !C || (!xf && !xh)) {
        return false;
    }
    if (M < 1024 || K < 16384 || g < 64 || (K % g) != 0 || (N % 16) != 0) {
        return false;
    }
    const int64_t M0 = M / 2;
    const int64_t M1 = M - M0;
    if (M0 < 16 || M1 < 16) {
        return false;
    }
    const int ovl = ggml_sycl_get_env("GGML_SYCL_XE2MX_LOSPLITM", 0) >= 2 ? 1 : 0;
    queue_ptr q1 = stream;
    if (ovl) {
        static sycl::queue * side = nullptr;
        static queue_ptr side_seen = nullptr;
        if (!side || side_seen != stream) {
            delete side;
            side = new sycl::queue(stream->get_context(), stream->get_device(),
                                    sycl::property_list{ sycl::property::queue::in_order() });
            side_seen = stream;
        }
        q1 = side;
    }
    const size_t ng = (size_t) (K / g);
    const size_t a0n = (size_t) M0 * (size_t) (K / 2);
    const size_t a1n = (size_t) M1 * (size_t) (K / 2);
    ggml_sycl_pool_alloc<uint8_t> A0(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> A1(ctx.pool());
    ggml_sycl_pool_alloc<float> S0(ctx.pool());
    ggml_sycl_pool_alloc<float> S1(ctx.pool());
    uint8_t * a0 = A0.alloc(a0n);
    uint8_t * a1 = A1.alloc(a1n);
    float * sa0 = S0.alloc((size_t) M0 * ng);
    float * sa1 = S1.alloc((size_t) M1 * ng);
    if (!a0 || !a1 || !sa0 || !sa1) {
        return false;
    }
    int pok0 = 0;
    if (xf) {
        pok0 = xe2mx_quant_a_s4_nibble_had_f32_rte(stream, xf, a0, sa0, M0, K, g) ? 1 : 0;
    } else {
        pok0 = xe2mx_quant_a_s4_nibble_had(stream, xh, a0, sa0, M0, K, g) ? 1 : 0;
    }
    if (!pok0) {
        return false;
    }
    try {
        xe2mx_s4_prim * p0 = xe2mx_s44_sk_get(ctx, stream, M0, N, K, K, g, 0, acc);
        xe2mx_s4_prim * p1 = xe2mx_s44_sk_get(ctx, stream, M1, N, K, K, g, 0, acc);
        if (!p0 || !p1) {
            return false;
        }
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        if (p1->sp_md.get_size() > p0->sp_md.get_size()) {
            ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
        }
        dnnl::memory sp0 = ctx.get_scratchpad_mem(p0->sp_md, eng, stream);
        p0->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(p0->a_md, eng, a0)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p0->b_md, eng, (void *) w4)},
            {DNNL_ARG_DST, dnnl::memory(p0->c_md, eng, C)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p0->sa_md, eng, sa0)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p0->sc_md, eng, (void *) scw)},
            {DNNL_ARG_SCRATCHPAD, sp0},
        });
        int pok1 = 0;
        if (xf) {
            pok1 = xe2mx_quant_a_s4_nibble_had_f32_rte(q1, xf + (size_t) M0 * (size_t) K, a1, sa1, M1, K, g) ? 1 : 0;
        } else {
            pok1 = xe2mx_quant_a_s4_nibble_had(q1, xh + (size_t) M0 * (size_t) K, a1, sa1, M1, K, g) ? 1 : 0;
        }
        if (!pok1) {
            if (ovl) {
                q1->wait();
            }
            return false;
        }
        if (ovl) {
            q1->wait();
        }
        dnnl::memory sp1 = ctx.get_scratchpad_mem(p1->sp_md, eng, stream);
        p1->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(p1->a_md, eng, a1)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p1->b_md, eng, (void *) w4)},
            {DNNL_ARG_DST, dnnl::memory(p1->c_md, eng, C + (size_t) M0 * (size_t) N)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p1->sa_md, eng, sa1)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p1->sc_md, eng, (void *) scw)},
            {DNNL_ARG_SCRATCHPAD, sp1},
        });
        return true;
    } catch (const dnnl::error & e) {
        g_xe2mx_splitm_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx splitm dnnl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (const sycl::exception & e) {
        g_xe2mx_splitm_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx splitm sycl: %s M=%ld N=%ld K=%ld\n",
                    e.what(), (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return false;
    } catch (...) {
        g_xe2mx_splitm_dead = 1;
        return false;
    }
}

static void xe2mx_s4t_pack(queue_ptr stream, const uint8_t * packed, uint8_t * w, float * sc,
                             int64_t K, int64_t N, int xor88) {
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float amax = 0.0f;
            for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui & 0xF) - 8.0f) * d));
                    amax = sycl::fmax(amax, sycl::fabs(((float) (vui >> 4) - 8.0f) * d));
                }
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sc[n] = gmax > 0.0f ? gmax / 7.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 7.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            uint8_t * col = w + (size_t) n * qpitch;
            for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    float x0 = ((float) (vui & 0xF) - 8.0f) * d * id;
                    float x1 = ((float) (vui >> 4) - 8.0f) * d * id;
                    int u0 = (int) (x0 + 8.0f + (x0 >= 0.0f ? 0.5f : -0.5f));
                    int u1 = (int) (x1 + 8.0f + (x1 >= 0.0f ? 0.5f : -0.5f));
                    if (u0 < 0) {
                        u0 = 0;
                    }
                    if (u0 > 15) {
                        u0 = 15;
                    }
                    if (u1 < 0) {
                        u1 = 0;
                    }
                    if (u1 > 15) {
                        u1 = 15;
                    }
                    col[ib * 16 + j] = (uint8_t) (((u1 << 4) | u0) ^ (xor88 ? 0x88u : 0u));
                }
            }
        });
}

static bool xe2mx_s4t_cached(queue_ptr stream, const uint8_t * packed, int64_t K, int64_t N, xe2mx_s8r_view * out) {
    const int s4tn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TN", 17408);
    if (!packed || !out || (K % 32) != 0 || (K / 32) % 2 != 0 || N < s4tn || (N % 16) != 0) {
        return false;
    }
    const int extra = N < 16384;
    const int tag = 20;
    const int64_t nb = K / 32;
    const size_t wbytes = (size_t) N * (size_t) nb * 16u;
    const size_t bytes = wbytes + (size_t) N * sizeof(float);
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        out->w = (int8_t *) it->second.packed;
        out->sc = (float *) ((char *) it->second.packed + wbytes);
        return true;
    }
    if (it != g_xe2mx_wcache.end()) {
        return false;
    }
    if (bytes <= (1ull << 20)) {
        return false;
    }
    if (extra) {
        if (g_xe2mx_s8r_ex_used + bytes > xe2mx_s8r_ex_cap()) {
            return false;
        }
    } else if (g_xe2mx_wcache_used + bytes > XE2MX_WCACHE_CAP) {
        return false;
    }
    void * persist = ggml_sycl_malloc_device(bytes, *stream);
    if (!persist) {
        return false;
    }
    uint8_t * w = (uint8_t *) persist;
    float * sc = (float *) ((char *) persist + wbytes);
    xe2mx_s4t_pack(stream, packed, w, sc, K, N, 1);
    if (extra) {
        g_xe2mx_s8r_ex_used += bytes;
    } else {
        g_xe2mx_wcache_used += bytes;
    }
    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
    out->w = (int8_t *) w;
    out->sc = sc;
    return true;
}

static int xe2mx_s44_g() {
    const int g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44G", 128);
    if (g != 32 && g != 64 && g != 128 && g != 256 && g != 512 && g != 1024) {
        return 128;
    }
    return g;
}

// Activation group width per projection role. Falls back to S44G when unset.
// DN = K >= 16384 (ffn_down), UP = N >= 16384 (ffn gate/up), MID = everything else.
// S44G_LFROM/S44G_LTO limit the role override to layers [from, to); unknown layer keeps it.
static int xe2mx_s44_g_kn(int64_t K, int64_t N, int layer = -1) {
    if (layer >= 0) {
        const int lf = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44G_LFROM", 0);
        const int lt = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44G_LTO", 1 << 20);
        if (layer < lf || layer >= lt) {
            return xe2mx_s44_g();
        }
    }
    const char * name = K >= 16384 ? "GGML_SYCL_XE2MX_S44G_DN"
        : (N >= 16384 ? "GGML_SYCL_XE2MX_S44G_UP" : "GGML_SYCL_XE2MX_S44G_MID");
    const int g = ggml_sycl_get_env(name, 0);
    if ((g == 32 || g == 64 || g == 128 || g == 256 || g == 512 || g == 1024) && (K % g) == 0) {
        return g;
    }
    return xe2mx_s44_g();
}

static int xe2mx_s44_hs() {
    const int hs = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HS", 128);
    if (hs != 32 && hs != 64 && hs != 128) {
        return 128;
    }
    return hs;
}

static int xe2mx_s44_qmode(int s8ra, int s44h, int s44zp, int s44g, int s44clip, int s44adiv,
                           int s44arow, int s44rms, int pack_pn, int s44hrow, int s44hocc,
                           int s44hwide, int s44hilp) {
    return s8ra + (s44h ? 2 : 0) + (s44zp ? 4 : 0) + s44g
        + s44clip * 2048 + s44adiv * 262144 + s44arow * 1048576 + (s44rms ? 2097152 : 0)
        + xe2mx_s44_hs() * 4194304 + (pack_pn ? 8388608 : 0)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0) ? 16777216 : 0)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44MEAN", 0) ? 33554432 : 0)
        + (s44hrow ? 67108864 : 0)
        + (s44hocc ? 134217728 : 0)
        + (s44hwide ? (1073741824 + s44hwide) : 0)
        + (s44hilp * 99991)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AS8", 0) ? 536870912 : 0)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HF16", 0) ? 268435456 : 0)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44WCEN", 0) ? 99989 : 0)
        + (ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_MSE", 0) * 7919);
}

static int xe2mx_s44_h_on(int layer, int64_t K, int64_t N = 0) {
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return 0;
    }
    const int hs = xe2mx_s44_hs();
    if ((K % hs) != 0) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HFFN", 0) && N < 16384 && K < 16384) {
        return 0;
    }
    const int hfrom = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HFROM", 0);
    const int hto = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HTO", 256);
    if (layer < 0) {
        if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HSKIPDN", 0) && K >= 16384) {
            return 0;
        }
        return 1;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HSKIPDN", 0) && K >= 16384) {
        return 0;
    }
    return layer >= hfrom && layer < hto;
}

static void xe2mx_had128_f32(queue_ptr stream, float * x, int64_t rows, int64_t K, const float * src = nullptr) {
    const float * in = src ? src : x;
    const int hs = xe2mx_s44_hs();
    const int nb = (int) (K / hs);
    const size_t ng = (size_t) rows * (size_t) nb;
    const size_t wg = (size_t) hs;
    const float inv = (hs == 32) ? 0.1767766952966369f : ((hs == 64) ? 0.125f : 0.08838834764831845f);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            const int blk = (int) it.get_group(0);
            const int r = blk / nb;
            const int b = blk - r * nb;
            auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(it.get_group());
            float * s = *slm;
            s[lid] = in[(size_t) r * (size_t) K + (size_t) b * (size_t) hs + (size_t) lid];
            for (int h = 1; h < hs; h *= 2) {
                it.barrier(sycl::access::fence_space::local_space);
                const int pair = lid ^ h;
                const float me = s[lid];
                const float you = s[pair];
                it.barrier(sycl::access::fence_space::local_space);
                if ((lid & h) == 0) {
                    s[lid] = me + you;
                    s[pair] = me - you;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            x[(size_t) r * (size_t) K + (size_t) b * (size_t) hs + (size_t) lid] = s[lid] * inv;
        });
}

static void xe2mx_had128_f16(queue_ptr stream, sycl::half * x, int64_t rows, int64_t K) {
    const int hs = xe2mx_s44_hs();
    const int nb = (int) (K / hs);
    const size_t ng = (size_t) rows * (size_t) nb;
    const size_t wg = (size_t) hs;
    const float inv = (hs == 32) ? 0.1767766952966369f : ((hs == 64) ? 0.125f : 0.08838834764831845f);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            const int blk = (int) it.get_group(0);
            const int r = blk / nb;
            const int b = blk - r * nb;
            auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(it.get_group());
            float * s = *slm;
            s[lid] = (float) x[(size_t) r * (size_t) K + (size_t) b * (size_t) hs + (size_t) lid];
            for (int h = 1; h < hs; h *= 2) {
                it.barrier(sycl::access::fence_space::local_space);
                const int pair = lid ^ h;
                const float me = s[lid];
                const float you = s[pair];
                it.barrier(sycl::access::fence_space::local_space);
                if ((lid & h) == 0) {
                    s[lid] = me + you;
                    s[pair] = me - you;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            x[(size_t) r * (size_t) K + (size_t) b * (size_t) hs + (size_t) lid] =
                sycl::half(s[lid] * inv);
        });
}

static bool xe2mx_had128_sa(queue_ptr stream, sycl::half * x, float * sa,
                            int64_t M, int64_t K, int64_t g) {
    if (!stream || !x || !sa || M < 1 || K < 128 || g < 128) {
        return false;
    }
    if ((K % g) != 0 || (g % 128) != 0) {
        return false;
    }
    const int64_t ng = K / g;
    const int nblk = (int) (g / 128);
    const float inv = 0.08838834764831845f;
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * 128u), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const int lane = lid & 15;
            const int sgid = lid >> 4;
            sycl::half * row = x + (size_t) m * (size_t) K;
            const int k0 = ig * (int) g;
            const sycl::sub_group sg = it.get_sub_group();
            constexpr int EL = 8;
            float amax = 0.0f;
            for (int blk = sgid; blk < nblk; blk += 8) {
                float a[EL];
                const int base = k0 + blk * 128 + lane * EL;
                const sycl::vec<sycl::half, 8> vh =
                    *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(row + base);
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
                sycl::vec<sycl::half, 8> ho;
#pragma unroll
                for (int i = 0; i < EL; ++i) {
                    a[i] = a[i] * inv;
                    amax = sycl::fmax(amax, sycl::fabs(a[i]));
                    ho[i] = sycl::half(a[i]);
                }
                *reinterpret_cast<sycl::vec<sycl::half, 8> *>(row + base) = ho;
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
            if (lid == 0) {
                sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
            }
        });
    return true;
}

static void xe2mx_quant_a_s4_had(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                                  int64_t M, int64_t K, int64_t g) {
    const int hs = xe2mx_s44_hs();
    const int64_t ng = K / g;
    const int nblk = (int) (g / hs);
    const size_t wg = (size_t) hs;
    const float inv = (hs == 32) ? 0.1767766952966369f : ((hs == 64) ? 0.125f : 0.08838834764831845f);
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            int8_t * arow = A + (size_t) m * (size_t) K;
            auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
            float * s = *slm;
            for (int ig = 0; ig < (int) ng; ig++) {
                const int k0 = ig * (int) g;
                for (int b = 0; b < nblk; b++) {
                    s[b * hs + lid] = (float) row[k0 + b * hs + lid];
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int h = 1; h < hs; h *= 2) {
                        it.barrier(sycl::access::fence_space::local_space);
                        const int pair = lid ^ h;
                        const float me = s[b * hs + lid];
                        const float you = s[b * hs + pair];
                        it.barrier(sycl::access::fence_space::local_space);
                        if ((lid & h) == 0) {
                            s[b * hs + lid] = me + you;
                            s[b * hs + pair] = me - you;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    s[b * hs + lid] = s[b * hs + lid] * inv;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float amax = 0.0f;
                for (int b = 0; b < nblk; b++) {
                    amax = sycl::fmax(amax, sycl::fabs(s[b * hs + lid]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float sc = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = sc;
                }
                const float id = sc > 0.0f ? 1.0f / sc : 0.0f;
                for (int b = 0; b < nblk; b++) {
                    int q = (int) sycl_rint_i(s[b * hs + lid] * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k0 + b * hs + lid] = (int8_t) q;
                }
            }
        });
}

static void xe2mx_s44_unpack_f32(queue_ptr stream, const uint8_t * packed, float * dst, int64_t K, int64_t N) {
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float * col = dst + (size_t) n * (size_t) K;
            for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; j++) {
                    const int vui = (int) q[j];
                    col[ib * 32 + 2 * j] = ((float) (vui & 0xF) - 8.0f) * d;
                    col[ib * 32 + 2 * j + 1] = ((float) (vui >> 4) - 8.0f) * d;
                }
            }
        });
}

static void xe2mx_s44_quant_f32(queue_ptr stream, const float * src, int8_t * w, float * sc, float * bsum,
                                  int64_t K, int64_t N, int64_t g) {
    const int64_t ng = K / g;
    const size_t wg = 128;
    const int wcen = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44WCEN", 0);
    const int mse_steps = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_MSE", 0);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float * col = src + (size_t) n * (size_t) K;
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
                float s = gmax > 0.0f ? gmax / 7.0f : 0.0f;
                if (mse_steps > 0 && s > 0.0f) {
                    // pick clip ratio with lowest squared error, search on GPU
                    float best_e = 0.0f;
                    float best_s = s;
                    for (int t = 0; t <= mse_steps; ++t) {
                        const float st = s * (1.0f - 0.4f * (float) t / (float) mse_steps);
                        const float it_ = 1.0f / st;
                        float e = 0.0f;
                        for (int k = k0 + lid; k < k1; k += (int) wg) {
                            const float x = col[k] - mean;
                            float q = sycl_rint_i(x * it_);
                            q = sycl::fmin(sycl::fmax(q, -8.0f), 7.0f);
                            const float d = x - q * st;
                            e += d * d;
                        }
                        e = sycl::reduce_over_group(it.get_group(), e, sycl::plus<float>());
                        if (t == 0 || e < best_e) {
                            best_e = e;
                            best_s = st;
                        }
                    }
                    s = best_s;
                }
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
                }
            }
        });
}

static void xe2mx_s44_zp_corr(queue_ptr stream, float * C, const float * za, const float * bsum,
                                int64_t M, int64_t N, int64_t ng) {
    stream->parallel_for(sycl::range<1>((size_t) M * (size_t) N), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        const size_t n = t % (size_t) N;
        const size_t m = t / (size_t) N;
        float s = 0.0f;
        for (int ig = 0; ig < (int) ng; ig++) {
            s += za[m * (size_t) ng + (size_t) ig] * bsum[(size_t) ig * (size_t) N + n];
        }
        C[t] += s;
    });
}

static void xe2mx_s44_requant(queue_ptr stream, const uint8_t * packed, int8_t * w, float * sc,
                                int64_t K, int64_t N, int64_t g) {
    const int64_t nb = K / 32;
    const int64_t ng = K / g;
    const int bpg = (int) (g / 32);
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const size_t wg = 128;
    if (g == 32) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int n = (int) it.get_group(0);
                const int lid = (int) it.get_local_id(0);
                int8_t * col = w + (size_t) n * (size_t) K;
                for (int ib = lid; ib < (int) nb; ib += (int) wg) {
                    sc[(size_t) ib * (size_t) N + (size_t) n] = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                    const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                    for (int j = 0; j < 16; j++) {
                        const int vui = (int) q[j];
                        col[ib * 32 + 2 * j] = (int8_t) ((vui & 0xF) - 8);
                        col[ib * 32 + 2 * j + 1] = (int8_t) ((vui >> 4) - 8);
                    }
                }
            });
        return;
    }
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int n = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            int8_t * col = w + (size_t) n * (size_t) K;
            for (int ig = 0; ig < (int) ng; ig++) {
                float amax = 0.0f;
                const int ib0 = ig * bpg;
                const int ib1 = ib0 + bpg;
                for (int ib = ib0 + lid; ib < ib1; ib += (int) wg) {
                    const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                    const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                    for (int j = 0; j < 16; j++) {
                        const int vui = (int) q[j];
                        amax = sycl::fmax(amax, sycl::fabs(((float) (vui & 0xF) - 8.0f) * d));
                        amax = sycl::fmax(amax, sycl::fabs(((float) (vui >> 4) - 8.0f) * d));
                    }
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                if (lid == 0) {
                    sc[(size_t) ig * (size_t) N + (size_t) n] = gmax > 0.0f ? gmax / 7.0f : 0.0f;
                }
                const float s = gmax > 0.0f ? gmax / 7.0f : 0.0f;
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int ib = ib0 + lid; ib < ib1; ib += (int) wg) {
                    const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                    const uint8_t * q = qs + (size_t) n * qpitch + (size_t) ib * 16u;
                    for (int j = 0; j < 16; j++) {
                        const int vui = (int) q[j];
                        float x0 = ((float) (vui & 0xF) - 8.0f) * d * id;
                        float x1 = ((float) (vui >> 4) - 8.0f) * d * id;
                        int q0 = (int) sycl_rint_i(x0);
                        int q1 = (int) sycl_rint_i(x1);
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
                        col[ib * 32 + 2 * j] = (int8_t) q0;
                        col[ib * 32 + 2 * j + 1] = (int8_t) q1;
                    }
                }
            }
        });
}

static void xe2mx_pack_s8_s4(queue_ptr stream, const int8_t * s8, uint8_t * s4, int64_t K, int64_t N) {
    const size_t nk2 = (size_t) (K / 2);
    stream->parallel_for(sycl::range<1>((size_t) N * nk2), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        const size_t k2 = t % nk2;
        const size_t n = t / nk2;
        const uint8_t u0 = (uint8_t) s8[(size_t) n * (size_t) K + k2 * 2u];
        const uint8_t u1 = (uint8_t) s8[(size_t) n * (size_t) K + k2 * 2u + 1u];
        s4[(size_t) n * nk2 + k2] = (uint8_t) ((u0 & 0x0Fu) | (uint8_t) ((u1 & 0x0Fu) << 4));
    });
}

static int8_t * g_xe2mx_s44_b8 = nullptr;
static size_t g_xe2mx_s44_b8_cap = 0;

static int8_t * xe2mx_s44_b8_expand(queue_ptr stream, const uint8_t * s4, int64_t K, int64_t N) {
    if (!stream || !s4 || K < 2 || (K & 1) || N < 1) {
        return nullptr;
    }
    const size_t nk = (size_t) K * (size_t) N;
    if (g_xe2mx_s44_b8_cap < nk) {
        g_xe2mx_s44_b8 = (int8_t *) ggml_sycl_malloc_device(nk, *stream);
        g_xe2mx_s44_b8_cap = g_xe2mx_s44_b8 ? nk : 0;
    }
    if (!g_xe2mx_s44_b8) {
        return nullptr;
    }
    int8_t * dstb = g_xe2mx_s44_b8;
    const size_t nk2 = (size_t) (K / 2);
    const int64_t Kc = K;
    stream->parallel_for(sycl::range<1>((size_t) N * nk2), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        const size_t k2 = t % nk2;
        const size_t n = t / nk2;
        const uint8_t p = s4[(size_t) n * nk2 + k2];
        const uint8_t lo = (uint8_t) (p & 0x0Fu);
        const uint8_t hi = (uint8_t) ((p >> 4) & 0x0Fu);
        int8_t * dst = dstb + n * (size_t) Kc + k2 * 2u;
        dst[0] = (int8_t) (lo | ((lo & 8u) ? 0xF0u : 0u));
        dst[1] = (int8_t) (hi | ((hi & 8u) ? 0xF0u : 0u));
    });
    return dstb;
}

static void xe2mx_f32_to_f16(queue_ptr stream, const float * x, sycl::half * y, int64_t n) {
    if (!stream || !x || !y || n < 1) {
        return;
    }
    const int64_t n8 = n / 8;
    const int64_t tail = n - n8 * 8;
    if (n8 > 0) {
        stream->parallel_for(sycl::range<1>((size_t) n8), [=](sycl::id<1> gid) {
            const size_t i = (size_t) gid[0] * 8u;
            const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8> *>(x + i);
            const sycl::vec<sycl::half, 8> h = v.convert<sycl::half, sycl::rounding_mode::rte>();
            *reinterpret_cast<sycl::vec<sycl::half, 8> *>(y + i) = h;
        });
    }
    if (tail > 0) {
        stream->parallel_for(sycl::range<1>((size_t) tail), [=](sycl::id<1> gid) {
            const size_t i = (size_t) n8 * 8u + (size_t) gid[0];
            y[i] = (sycl::half) x[i];
        });
    }
}

static void xe2mx_lo_contract_env(void) {
    setenv("GGML_SYCL_XE2MX_S44", "1", 0);
    setenv("GGML_SYCL_XE2MX_S44N", "4096", 0);
    setenv("GGML_SYCL_XE2MX_S44FROM", "0", 0);
    setenv("GGML_SYCL_XE2MX_S44A", "8", 0);
    setenv("GGML_SYCL_XE2MX_S8RA", "0", 1);
    setenv("GGML_SYCL_XE2MX_S44AROW", "0", 1);
    setenv("GGML_SYCL_XE2MX_S44CLIP", "80", 0);
    setenv("GGML_SYCL_XE2MX_S44ADIV", "8", 0);
    setenv("GGML_SYCL_XE2MX_S44FPACK", "1", 0);
    setenv("GGML_SYCL_XE2MX_S44HILP", "4", 0);
    setenv("GGML_SYCL_XE2MX_S4ACC", "3", 0);
    setenv("GGML_SYCL_XE2MX_Q32V", "1", 0);
    setenv("GGML_SYCL_XE2MX_S44H", "1", 0);
    setenv("GGML_SYCL_XE2MX_S44PN", "1", 0);
    setenv("GGML_SYCL_XE2MX_S44G", "1024", 0);
    setenv("GGML_SYCL_XE2MX_S44RMSHILP", "1", 0);
    unsetenv("GGML_SYCL_XE2MX_S44ACACHE");
    unsetenv("GGML_SYCL_XE2MX_S44ASTICK");
    ggml_sycl_env_cache_invalidate();
}

void ggml_sycl_xe2mx_lo_hostpack_release(sycl::queue & q) {
    xe2mx_lo_hostpack_drop(q);
}

void * ggml_sycl_xe2mx_lo_packed(const void * key) {
    auto it = g_xe2mx_lo_w.find(key);
    if (it == g_xe2mx_lo_w.end()) {
        return nullptr;
    }
    return it->second.packed;
}

void ggml_sycl_xe2mx_lo_rekey(const void * oldk, const void * newk) {
    auto it = g_xe2mx_lo_w.find(oldk);
    if (it == g_xe2mx_lo_w.end() || oldk == newk) {
        return;
    }
    xe2mx_wcache val = it->second;
    // Inline LO: packed lives in the tensor slot (oldk). After rekey, pack pointer must track newk.
    if (val.vnni == 52 && val.packed == oldk) {
        val.packed = const_cast<void *>(newk);
    }
    g_xe2mx_lo_w.erase(it);
    g_xe2mx_lo_w[newk] = val;
    if (g_xe2mx_lo.erase(oldk)) {
        g_xe2mx_lo.insert(newk);
    }
}

struct xe2mx_lo_cache_hdr {
    char magic[8];
    uint32_t ver;
    uint32_t pad;
    uint64_t N;
    uint64_t K;
    uint64_t wg;
    uint64_t lo_h;
    uint64_t aos;
    uint64_t pbytes;
};

static bool xe2mx_lo_cache_try_load(sycl::queue & q, ggml_tensor * t,
                                    int64_t N, int64_t K, int wg, int lo_h, size_t aos, size_t pbytes,
                                    bool force, int id) {
    const char * dir = xe2mx_lo_cache_dir();
    if (!dir) {
        return false;
    }
    const std::string path = xe2mx_lo_cache_path(dir, t->name, N, K, wg, lo_h, aos, pbytes);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    xe2mx_lo_cache_hdr hdr {};
    in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!in || std::memcmp(hdr.magic, "XE2MXLO1", 8) != 0 || hdr.ver != 1) {
        return false;
    }
    if (hdr.N != (uint64_t) N || hdr.K != (uint64_t) K || hdr.wg != (uint64_t) wg
        || hdr.lo_h != (uint64_t) lo_h || hdr.aos != aos || hdr.pbytes != pbytes) {
        return false;
    }
    std::vector<uint8_t> soa(aos);
    std::vector<uint8_t> pack(pbytes);
    in.read(reinterpret_cast<char *>(soa.data()), (std::streamsize) aos);
    in.read(reinterpret_cast<char *>(pack.data()), (std::streamsize) pbytes);
    if (!in) {
        return false;
    }
    void * persist = ggml_sycl_malloc_device(pbytes, q);
    if (!persist) {
        fprintf(stderr, "xe2mx lo cache load fail malloc #%d\n", id);
        fflush(stderr);
        return false;
    }
    q.memcpy(persist, pack.data(), pbytes).wait();
    if (!force) {
        q.memcpy(t->data, soa.data(), aos).wait();
    }
    g_xe2mx_lo.insert(t->data);
    g_xe2mx_lo_w[t->data] = { persist, pbytes, K, N, 51, nullptr };
    g_xe2mx_lo_w[t->data].hp_dev = xe2mx_q_dev(q);
    xe2mx_lo_hostpack_store(q, g_xe2mx_lo_w[t->data], pack.data(), pbytes, false);
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) t->extra;
    if (extra) {
        extra->optimized_feature.reorder = true;
        extra->optimized_feature.xe2mx_lo = true;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo cache HIT #%d %s N=%ld K=%ld\n", id, t->name ? t->name : "?", (long) N, (long) K);
    fflush(stderr);
    // force=true (split): caller may free SOA slice and rekey to pack; skip inline here.
    if (!force) {
        xe2mx_lo_maybe_inline(t, q, id);
    }
    return true;
}

static void xe2mx_lo_cache_store(const char * name, int64_t N, int64_t K, int wg, int lo_h,
                                 const uint8_t * soa, size_t aos, const uint8_t * pack, size_t pbytes, int id) {
    const char * dir = xe2mx_lo_cache_dir();
    if (!dir || !soa || !pack) {
        return;
    }
    xe2mx_lo_cache_mkdir(dir);
    const std::string path = xe2mx_lo_cache_path(dir, name, N, K, wg, lo_h, aos, pbytes);
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    xe2mx_lo_cache_hdr hdr {};
    std::memcpy(hdr.magic, "XE2MXLO1", 8);
    hdr.ver = 1;
    hdr.N = (uint64_t) N;
    hdr.K = (uint64_t) K;
    hdr.wg = (uint64_t) wg;
    hdr.lo_h = (uint64_t) lo_h;
    hdr.aos = aos;
    hdr.pbytes = pbytes;
    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char *>(soa), (std::streamsize) aos);
    out.write(reinterpret_cast<const char *>(pack), (std::streamsize) pbytes);
    out.close();
    if (!out) {
        ::unlink(tmp.c_str());
        return;
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        return;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo cache STORE #%d %s\n", id, name ? name : "?");
    fflush(stderr);
}

bool ggml_sycl_xe2mx_lo_upload(ggml_tensor * t, const void * host, size_t size, sycl::queue & q, bool force) {
    const bool shape = t && t->type == GGML_TYPE_XE2MX4 && t->ne[2] == 1 && t->ne[3] == 1
        && (t->ne[0] % 128) == 0 && (t->ne[1] % 16) == 0 && t->ne[1] >= 16;
    if ((force ? !shape : !ggml_sycl_xe2mx_lo_on(t)) || !host || !t->data || size != ggml_nbytes(t)) {
        return false;
    }
    const int64_t K = t->ne[0];
    const int64_t N = t->ne[1];
    xe2mx_lo_contract_env();
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int lo_h = xe2mx_s44_h_on(-1, K, N);
    int wg = (int) K;
    if (!lo_h || !s44pn) {
        const int gnh = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GNH", 0);
        if (!lo_h && (gnh == 32 || gnh == 64 || gnh == 128) && (K % (int64_t) gnh) == 0) {
            wg = gnh;
        } else {
            wg = xe2mx_s44_g();
            if (wg <= 0 || (K % (int64_t) wg) != 0) {
                wg = (int) K;
            }
        }
    }
    const int64_t nb = K / 32;
    const size_t pitch = (size_t) nb * 16u;
    const size_t aos = ggml_nbytes(t);
    const size_t w4b = (size_t) N * (size_t) (K / 2);
    const size_t scb = (size_t) (K / wg) * (size_t) N * sizeof(float);
    const int s44zp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0);
    const int s44wcen = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44WCEN", 0);
    const size_t zpb = (s44zp || (s44wcen && !lo_h)) ? scb : 0;
    const size_t fbytes = (size_t) N * (size_t) K * sizeof(float);
    const size_t s8b = (size_t) N * (size_t) K;
    queue_ptr stream = &q;
    static int nup = 0;
    const int id = ++nup;
    GGML_SYCL_XE2_LOG( "xe2mx lo upload start #%d %s N=%ld K=%ld\n",
            id, t->name ? t->name : "?", (long) N, (long) K);
    fflush(stderr);
    const size_t pbytes = w4b + scb + zpb;
    if (xe2mx_lo_cache_try_load(q, t, N, K, wg, lo_h, aos, pbytes, force, id)) {
        xe2mx_lo_contract_env();
        return true;
    }
    // Host-side LO bake: no device scratch (device path OOMs when Chrome/compositor holds Arc).
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOST", 0)) {
        void * aos_d = nullptr;
        void * soa_d = nullptr;
        void * f_d = nullptr;
        void * s8_d = nullptr;
        std::vector<uint8_t> soa((size_t) aos);
        std::vector<float> f32((size_t) N * (size_t) K);
        std::vector<int8_t> s8((size_t) N * (size_t) K);
        std::vector<uint8_t> out(w4b + scb + zpb);
        const uint8_t * hsrc = (const uint8_t *) host;
        uint8_t * qs = soa.data();
        uint16_t * dpl = (uint16_t *) (soa.data() + (size_t) N * pitch);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ib = 0; ib < nb; ++ib) {
                const uint8_t * blk = hsrc + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
                uint8_t * qn = qs + (size_t) n * pitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; ++j) qn[j] = blk[2 + j];
                dpl[(size_t) ib * (size_t) N + (size_t) n] = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            }
        }
        const sycl::half * dph = (const sycl::half *) dpl;
        #pragma omp parallel for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            float * col = f32.data() + (size_t) n * (size_t) K;
            for (int64_t ib = 0; ib < nb; ++ib) {
                const float d = (float) dph[(size_t) ib * (size_t) N + (size_t) n];
                const uint8_t * qq = qs + (size_t) n * pitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; ++j) {
                    const int vui = (int) qq[j];
                    col[ib * 32 + 2 * j] = ((float) (vui & 0xF) - 8.0f) * d;
                    col[ib * 32 + 2 * j + 1] = ((float) (vui >> 4) - 8.0f) * d;
                }
            }
        }
        if (lo_h) {
            const float inv = 0.08838834764831845f; // 1/sqrt(128)
            #pragma omp parallel for schedule(static)
            for (int64_t n = 0; n < N; ++n) {
                float * row = f32.data() + (size_t) n * (size_t) K;
                for (int64_t k0 = 0; k0 + 128 <= K; k0 += 128) {
                    float tmp[128];
                    for (int i = 0; i < 128; ++i) tmp[i] = row[k0 + i];
                    for (int h = 1; h < 128; h *= 2) {
                        float nxt[128];
                        for (int lid = 0; lid < 128; ++lid) {
                            const int pair = lid ^ h;
                            const float me = tmp[lid], you = tmp[pair];
                            if ((lid & h) == 0) {
                                nxt[lid] = me + you;
                                nxt[pair] = me - you;
                            }
                        }
                        // fill pairs written by the if branch only - redo full like device
                        for (int lid = 0; lid < 128; ++lid) {
                            const int pair = lid ^ h;
                            const float me = tmp[lid], you = tmp[pair];
                            if ((lid & h) == 0) {
                                nxt[lid] = me + you;
                                nxt[pair] = me - you;
                            }
                        }
                        for (int i = 0; i < 128; ++i) tmp[i] = nxt[i];
                    }
                    for (int i = 0; i < 128; ++i) row[k0 + i] = tmp[i] * inv;
                }
            }
        }
        float * sc = (float *) (out.data() + w4b);
        const int64_t ng = K / (int64_t) wg;
        #pragma omp parallel for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            const float * col = f32.data() + (size_t) n * (size_t) K;
            int8_t * dst = s8.data() + (size_t) n * (size_t) K;
            for (int64_t ig = 0; ig < ng; ++ig) {
                const int k0 = (int) (ig * wg);
                const int k1 = k0 + wg;
                float amax = 0.f;
                for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(col[k]));
                // Match xe2mx_s44_quant_f32: scale = amax/7, q in [-8,7]
                const float s = (amax > 0.f) ? (amax / 7.f) : 0.f;
                sc[(size_t) ig * (size_t) N + (size_t) n] = s;
                const float inv = (s > 0.f) ? (1.f / s) : 0.f;
                for (int k = k0; k < k1; ++k) {
                    int v = (int) std::lround(col[k] * inv);
                    if (v > 7) v = 7;
                    if (v < -8) v = -8;
                    dst[k] = (int8_t) v;
                }
            }
        }
        uint8_t * s4 = out.data();
        #pragma omp parallel for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            const int8_t * src8 = s8.data() + (size_t) n * (size_t) K;
            uint8_t * dst4 = s4 + (size_t) n * (size_t) (K / 2);
            for (int64_t k2 = 0; k2 < K / 2; ++k2) {
                const int8_t a = src8[k2 * 2], b = src8[k2 * 2 + 1];
                dst4[k2] = (uint8_t) (((uint8_t) a & 0x0f) | (((uint8_t) b & 0x0f) << 4));
            }
        }
        void * persist = ggml_sycl_malloc_device(out.size(), q);
        if (!persist) {
            fprintf(stderr, "xe2mx lo upload HOST fail malloc #%d\n", id);
            fflush(stderr);
            return false;
        }
        q.memcpy(persist, out.data(), out.size()).wait();
        if (!force) {
            // Match device path: weight view becomes SOA layout
            q.memcpy(t->data, soa.data(), aos).wait();
        }
        xe2mx_lo_cache_store(t->name, N, K, wg, lo_h, soa.data(), aos, out.data(), out.size(), id);
        g_xe2mx_lo.insert(t->data);
        g_xe2mx_lo_w[t->data] = { persist, out.size(), K, N, 51, nullptr };
        g_xe2mx_lo_w[t->data].hp_dev = xe2mx_q_dev(q);
        xe2mx_lo_hostpack_store(q, g_xe2mx_lo_w[t->data], out.data(), out.size(), false);
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) t->extra;
        if (extra) {
            extra->optimized_feature.reorder = true;
            extra->optimized_feature.xe2mx_lo = true;
        }
        xe2mx_lo_contract_env();
        GGML_SYCL_XE2_LOG( "xe2mx lo upload HOST #%d %s N=%ld K=%ld ok\n", id, t->name ? t->name : "?", (long) N, (long) K);
        fflush(stderr);
        // free unused device scratch if allocated below is skipped
        if (aos_d) ggml_sycl_free_device(aos_d, q);
        if (soa_d) ggml_sycl_free_device(soa_d, q);
        if (f_d) ggml_sycl_free_device(f_d, q);
        if (s8_d) ggml_sycl_free_device(s8_d, q);
        if (!force) {
            xe2mx_lo_maybe_inline(t, q, id);
        }
        return true;
    }
    void * aos_d = ggml_sycl_malloc_device(aos, q);
    void * soa_d = ggml_sycl_malloc_device(aos, q);
    void * f_d = ggml_sycl_malloc_device(fbytes, q);
    void * s8_d = ggml_sycl_malloc_device(s8b, q);
    void * persist = ggml_sycl_malloc_device(w4b + scb + zpb, q);
    if (!aos_d || !soa_d || !f_d || !s8_d || !persist) {
        if (aos_d) {
            ggml_sycl_free_device(aos_d, q);
        }
        if (soa_d) {
            ggml_sycl_free_device(soa_d, q);
        }
        if (f_d) {
            ggml_sycl_free_device(f_d, q);
        }
        if (s8_d) {
            ggml_sycl_free_device(s8_d, q);
        }
        if (persist) {
            ggml_sycl_free_device(persist, q);
        }
        fprintf(stderr, "xe2mx lo upload fail malloc #%d N=%ld K=%ld\n", id, (long) N, (long) K);
        fflush(stderr);
        return false;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d malloc ok aos=%zu f=%zu\n", id, aos, fbytes);
    fflush(stderr);
    q.memcpy(aos_d, host, aos).wait();
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d h2d ok\n", id);
    fflush(stderr);
    const uint8_t * src = (const uint8_t *) aos_d;
    uint8_t * qs = (uint8_t *) soa_d;
    sycl::half * dpl = (sycl::half *) ((uint8_t *) soa_d + (size_t) N * pitch);
    // Host SOA: AOT device soa parallel_for SEGV'd on bmg_g31 (EXIT 139 after h2d).
    {
        std::vector<uint8_t> hsoa((size_t) aos);
        const uint8_t * hsrc = (const uint8_t *) host;
        uint8_t * hqs = hsoa.data();
        uint16_t * hdpl = (uint16_t *) (hsoa.data() + (size_t) N * pitch);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ib = 0; ib < nb; ++ib) {
                const uint8_t * blk = hsrc + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
                uint8_t * qn = hqs + (size_t) n * pitch + (size_t) ib * 16u;
                for (int j = 0; j < 16; ++j) qn[j] = blk[2 + j];
                hdpl[(size_t) ib * (size_t) N + (size_t) n] =
                    (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
            }
        }
        q.memcpy(soa_d, hsoa.data(), aos).wait();
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d soa ok (host)\n", id);
    fflush(stderr);
    xe2mx_s44_unpack_f32(stream, (const uint8_t *) soa_d, (float *) f_d, K, N);
    try { q.wait_and_throw(); } catch (const std::exception & e) {
        fprintf(stderr, "xe2mx lo upload #%d unpack EXC %s\n", id, e.what());
        fflush(stderr);
        return false;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d unpack ok lo_h=%d\n", id, lo_h);
    fflush(stderr);
    if (lo_h) {
        xe2mx_had128_f32(stream, (float *) f_d, N, K);
        try { q.wait_and_throw(); } catch (const std::exception & e) {
            fprintf(stderr, "xe2mx lo upload #%d had EXC %s\n", id, e.what());
            fflush(stderr);
            return false;
        }
        GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d had ok\n", id);
        fflush(stderr);
    }
    float * sc = (float *) ((char *) persist + w4b);
    float * bsum = (s44zp || (s44wcen && !lo_h)) ? (float *) ((char *) persist + w4b + scb) : nullptr;
    const int gptq = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GPTQ", 0) && !lo_h;
    if (gptq) {
        xe2mx_s44_quant_w_gptq(stream, (float *) f_d, (int8_t *) s8_d, sc, bsum, K, N, (int64_t) wg);
    } else {
        xe2mx_s44_quant_f32(stream, (const float *) f_d, (int8_t *) s8_d, sc, bsum, K, N, (int64_t) wg);
    }
    try { q.wait_and_throw(); } catch (const std::exception & e) {
        fprintf(stderr, "xe2mx lo upload #%d quant EXC %s\n", id, e.what());
        fflush(stderr);
        return false;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d quant ok\n", id);
    fflush(stderr);
    xe2mx_pack_s8_s4(stream, (const int8_t *) s8_d, (uint8_t *) persist, K, N);
    try { q.wait_and_throw(); } catch (const std::exception & e) {
        fprintf(stderr, "xe2mx lo upload #%d pack EXC %s\n", id, e.what());
        fflush(stderr);
        return false;
    }
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d pack ok\n", id);
    fflush(stderr);
    if (!force) {
        q.memcpy(t->data, soa_d, aos);
    }
    q.wait();
    // Disk cache: D2H SOA + packed once per tensor name/shape
    if (xe2mx_lo_cache_dir()) {
        std::vector<uint8_t> hsoa(aos);
        std::vector<uint8_t> hpack(w4b + scb + zpb);
        q.memcpy(hsoa.data(), soa_d, aos).wait();
        q.memcpy(hpack.data(), persist, w4b + scb + zpb).wait();
        xe2mx_lo_cache_store(t->name, N, K, wg, lo_h, hsoa.data(), aos, hpack.data(), hpack.size(), id);
    }
    ggml_sycl_free_device(aos_d, q);
    ggml_sycl_free_device(soa_d, q);
    ggml_sycl_free_device(f_d, q);
    ggml_sycl_free_device(s8_d, q);
    g_xe2mx_lo.insert(t->data);
    g_xe2mx_lo_w[t->data] = { persist, w4b + scb + zpb, K, N, 51, nullptr };
    g_xe2mx_lo_w[t->data].hp_dev = xe2mx_q_dev(q);
    xe2mx_lo_hostpack_store(q, g_xe2mx_lo_w[t->data], persist, w4b + scb + zpb, true);
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) t->extra;
    if (extra) {
        extra->optimized_feature.reorder = true;
        extra->optimized_feature.xe2mx_lo = true;
    }
    xe2mx_lo_contract_env();
    GGML_SYCL_XE2_LOG( "xe2mx lo upload #%d %s N=%ld K=%ld dual\n",
            id, t->name ? t->name : "?", (long) N, (long) K);
    fflush(stderr);
    if (!force) {
        xe2mx_lo_maybe_inline(t, q, id);
    }
    return true;
}

static bool xe2mx_lo_small(ggml_backend_sycl_context & ctx,
                           const ggml_tensor * src0,
                           const ggml_tensor * src1,
                           ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(src1);
    if (M < 1 || (K % 2) != 0 || !src0->data || !src1->data || !dst->data) {
        return false;
    }
    // LO prefill pack only (dual .packed or inline t->data). Never read SOA as s4.
    auto lo_it = g_xe2mx_lo_w.find((const void *) src0->data);
    if (lo_it != g_xe2mx_lo_w.end() && lo_it->second.host && !lo_it->second.packed) {
        // small-M never uploads; caller falls back to the original-bytes path
        return false;
    }
    if (lo_it == g_xe2mx_lo_w.end() || !lo_it->second.packed) {
        return false;
    }
    const size_t w4b = (size_t) N * (size_t) (K / 2);
    const size_t sc_pn = (size_t) N * sizeof(float);
    const size_t lb = lo_it->second.bytes;
    // Per-N scales (wg==K / s44pn): w4 + N*f32 [, + aux N*f32]. Group scales unsupported here.
    if (lb != w4b + sc_pn && lb != w4b + sc_pn + sc_pn) {
        return false;
    }
    // Act path: F32 memcpy or F16/other via to_fp32. Hadamard matches LO prefill (lo_h packs).
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        const to_fp32_sycl_t probe = ggml_get_to_fp32_sycl(src1->type, dst);
        if (!probe) {
            return false;
        }
    }
    queue_ptr stream = ctx.stream();
    const uint8_t * w4 = (const uint8_t *) lo_it->second.packed;
    const float * sc = (const float *) (w4 + w4b);
    ggml_sycl_pool_alloc<float> Af(ctx.pool());
    float * a = Af.alloc((size_t) M * (size_t) K);
    const float * had_src = nullptr;
    if (src1->type == GGML_TYPE_F32) {
        had_src = (const float *) src1->data;
    } else if (src1->type == GGML_TYPE_F16) {
        const to_fp32_sycl_t to_a = ggml_get_to_fp32_sycl(GGML_TYPE_F16, dst);
        if (!to_a) {
            return false;
        }
        to_a(src1->data, a, M * K, stream);
    } else {
        const to_fp32_sycl_t to_a = ggml_get_to_fp32_sycl(src1->type, dst);
        if (!to_a) {
            return false;
        }
        to_a(src1->data, a, M * K, stream);
    }
    xe2mx_had128_f32(stream, a, M, K, had_src);
    float * C = (float *) dst->data;
    if (M <= 4 && (K % 256) == 0 && ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_GEMV", 1)) {
        // one sub-group per RR output rows (act vec reused RR times), 8 sub-groups per work-group
        const int64_t nw = K / 8;
        const int Mi = (int) M;
        constexpr int RR = 4;
        const int rr_env = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_GEMV_RR", 1);
        if (rr_env && (N % (8 * RR)) == 0) {
            const size_t nrg = (size_t) N / (8u * RR);
            stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nrg * 128u), sycl::range<1>(128)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const int64_t n0 = ((int64_t) it.get_group(0) * 8 + (int64_t) (it.get_local_id(0) >> 4)) * RR;
                    const int lane = (int) (it.get_local_id(0) & 15);
                    const size_t rs = (size_t) (K / 2);
                    float acc[RR][4];
#pragma unroll
                    for (int r = 0; r < RR; ++r) {
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            acc[r][m] = 0.0f;
                        }
                    }
                    for (int64_t w = lane; w < nw; w += 16) {
                        uint32_t v[RR];
#pragma unroll
                        for (int r = 0; r < RR; ++r) {
                            v[r] = ((const uint32_t *) (w4 + (size_t) (n0 + r) * rs))[w];
                        }
                        for (int m = 0; m < Mi; ++m) {
                            const sycl::vec<float, 8> av = *reinterpret_cast<const sycl::vec<float, 8> *>(a + (size_t) m * (size_t) K + (size_t) w * 8u);
#pragma unroll
                            for (int r = 0; r < RR; ++r) {
                                float s = 0.0f;
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    s += av[i] * (float) ((int32_t) (v[r] << (28 - 4 * i)) >> 28);
                                }
                                acc[r][m] += s;
                            }
                        }
                    }
#pragma unroll
                    for (int r = 0; r < RR; ++r) {
                        const float sn = sc[n0 + r];
                        for (int m = 0; m < Mi; ++m) {
                            const float red = sycl::reduce_over_group(it.get_sub_group(), acc[r][m], sycl::plus<float>());
                            if (lane == 0) {
                                C[(size_t) m * (size_t) N + (size_t) (n0 + r)] = red * sn;
                            }
                        }
                    }
                });
            return true;
        }
        const size_t nrg = ((size_t) N + 7u) / 8u;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nrg * 128u), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int64_t n = (int64_t) it.get_group(0) * 8 + (int64_t) (it.get_local_id(0) >> 4);
                const int lane = (int) (it.get_local_id(0) & 15);
                if (n >= N) {
                    return;
                }
                const uint32_t * row = (const uint32_t *) (w4 + (size_t) n * (size_t) (K / 2));
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int64_t w = lane; w < nw; w += 16) {
                    const uint32_t v = row[w];
                    float q[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        q[i] = (float) ((int32_t) (v << (28 - 4 * i)) >> 28);
                    }
                    for (int m = 0; m < Mi; ++m) {
                        const sycl::vec<float, 8> av = *reinterpret_cast<const sycl::vec<float, 8> *>(a + (size_t) m * (size_t) K + (size_t) w * 8u);
                        float s = 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            s += av[i] * q[i];
                        }
                        acc[m] += s;
                    }
                }
                const float sn = sc[n];
                for (int m = 0; m < Mi; ++m) {
                    const float r = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                    if (lane == 0) {
                        C[(size_t) m * (size_t) N + (size_t) n] = r * sn;
                    }
                }
            });
        return true;
    }
#if GGML_SYCL_DNNL
    if (N < 16384 && K < 16384) {
    try {
        ggml_sycl_pool_alloc<int8_t> As(ctx.pool());
        ggml_sycl_pool_alloc<float> Sa(ctx.pool());
        int8_t * as8 = As.alloc((size_t) M * (size_t) K);
        float * sa = Sa.alloc((size_t) M);
        stream->parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> id) {
            const size_t m = (size_t) id[0];
            const float * ar = a + m * (size_t) K;
            float amax = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                amax = sycl::fmax(amax, sycl::fabs(ar[k]));
            }
            const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
            sa[m] = s;
            const float idv = s > 0.0f ? 1.0f / s : 0.0f;
            int8_t * dr = as8 + m * (size_t) K;
            for (int64_t k = 0; k < K; ++k) {
                int q = (int) sycl_rint_i(ar[k] * idv);
                if (q < -127) {
                    q = -127;
                }
                if (q > 127) {
                    q = 127;
                }
                dr[k] = (int8_t) q;
            }
        });
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        xe2mx_s4_prim * pe = xe2mx_s48_get(ctx, stream, M, N, K, 12, K);
        dnnl::memory a_mem(pe->a_md, eng, as8);
        dnnl::memory b_mem(pe->b_md, eng, const_cast<uint8_t *>(w4));
        dnnl::memory c_mem(pe->c_md, eng, C);
        dnnl::memory sc_mem(pe->sc_md, eng, const_cast<float *>(sc));
        dnnl::memory sa_mem(pe->sa_md, eng, sa);
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pe->sp_md, eng, stream);
        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_mem},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        };
        pe->prim.execute(strm, args);
        return true;
    } catch (const dnnl::error & e) {
        static int nerr = 0;
        if (nerr < 2) {
            nerr++;
            fprintf(stderr, "xe2mx: s8s4 fail %s\n", e.what());
            fflush(stderr);
        }
    }
    }
#endif
    const size_t nk2 = (size_t) (K / 2);
    const size_t wg = 16;
    const size_t ng = ((size_t) N + wg - 1) / wg;
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const size_t n = (size_t) it.get_global_id(0);
            if (n >= (size_t) N) {
                return;
            }
            const uint8_t * row = w4 + n * nk2;
            const float sn = sc[n];
            for (int64_t m = 0; m < M; ++m) {
                const float * ar = a + (size_t) m * (size_t) K;
                float acc = 0.0f;
                size_t k2 = 0;
                for (; k2 + 4 <= nk2; k2 += 4) {
                    const uint8_t b0 = row[k2];
                    const uint8_t b1 = row[k2 + 1];
                    const uint8_t b2 = row[k2 + 2];
                    const uint8_t b3 = row[k2 + 3];
                    int q0 = (int) (b0 & 0x0Fu); int q1 = (int) (b0 >> 4);
                    int q2 = (int) (b1 & 0x0Fu); int q3 = (int) (b1 >> 4);
                    int q4 = (int) (b2 & 0x0Fu); int q5 = (int) (b2 >> 4);
                    int q6 = (int) (b3 & 0x0Fu); int q7 = (int) (b3 >> 4);
                    if (q0 >= 8) { q0 -= 16; } if (q1 >= 8) { q1 -= 16; }
                    if (q2 >= 8) { q2 -= 16; } if (q3 >= 8) { q3 -= 16; }
                    if (q4 >= 8) { q4 -= 16; } if (q5 >= 8) { q5 -= 16; }
                    if (q6 >= 8) { q6 -= 16; } if (q7 >= 8) { q7 -= 16; }
                    const float * ak = ar + k2 * 2u;
                    acc += ak[0] * (float) q0 + ak[1] * (float) q1
                        + ak[2] * (float) q2 + ak[3] * (float) q3
                        + ak[4] * (float) q4 + ak[5] * (float) q5
                        + ak[6] * (float) q6 + ak[7] * (float) q7;
                }
                for (; k2 < nk2; ++k2) {
                    const uint8_t b = row[k2];
                    int q0 = (int) (b & 0x0Fu);
                    int q1 = (int) (b >> 4);
                    if (q0 >= 8) { q0 -= 16; }
                    if (q1 >= 8) { q1 -= 16; }
                    acc += ar[k2 * 2u] * (float) q0 + ar[k2 * 2u + 1u] * (float) q1;
                }
                C[(size_t) m * (size_t) N + n] = acc * sn;
            }
        });
    return true;
}

static bool xe2mx_s48_rot_s4(ggml_backend_sycl_context & ctx, queue_ptr stream,
                               const sycl::half * xh, const sycl::half * U, sycl::half * xrot,
                               int64_t M, int64_t K, int layer, int64_t g) {
    if (!xh || !U || !xrot || layer < 0 || layer >= 64 || K != 5120 || g < 128 || (K % g) != 0) {
        return false;
    }
    const size_t nk = (size_t) K * (size_t) K;
    const size_t w4b = nk / 2u;
    const size_t scb = (size_t) (K / g) * (size_t) K * sizeof(float);
    const size_t xn = (size_t) M * (size_t) K;
    const size_t xa4 = (size_t) M * (size_t) (K / 2);
    const size_t xas = (size_t) M * (size_t) (K / g) * sizeof(float);
    if (!g_xe2mx_s48_us4[layer] || g_xe2mx_s48_us4g[layer] != (int) g) {
        if (g_xe2mx_s48_us4[layer]) {
            ggml_sycl_free_device(g_xe2mx_s48_us4[layer], *stream);
            g_xe2mx_s48_us4[layer] = nullptr;
        }
        if (g_xe2mx_s48_usc4[layer]) {
            ggml_sycl_free_device(g_xe2mx_s48_usc4[layer], *stream);
            g_xe2mx_s48_usc4[layer] = nullptr;
        }
        uint8_t * us4 = (uint8_t *) ggml_sycl_malloc_device(w4b, *stream);
        float * usc = (float *) ggml_sycl_malloc_device(scb, *stream);
        float * uf = (float *) ggml_sycl_malloc_device(nk * sizeof(float), *stream);
        int8_t * t8 = (int8_t *) ggml_sycl_malloc_device(nk, *stream);
        if (!us4 || !usc || !uf || !t8) {
            if (us4) {
                ggml_sycl_free_device(us4, *stream);
            }
            if (usc) {
                ggml_sycl_free_device(usc, *stream);
            }
            if (uf) {
                ggml_sycl_free_device(uf, *stream);
            }
            if (t8) {
                ggml_sycl_free_device(t8, *stream);
            }
            return false;
        }
        stream->parallel_for(sycl::range<1>(nk), [=](sycl::id<1> gid) {
            const size_t t = (size_t) gid[0];
            const size_t k = t % (size_t) K;
            const size_t n = t / (size_t) K;
            uf[t] = (float) U[(size_t) k * (size_t) K + n];
        });
        xe2mx_s44_quant_f32(stream, uf, t8, usc, nullptr, K, K, g);
        xe2mx_pack_s8_s4(stream, t8, us4, K, K);
        stream->wait();
        ggml_sycl_free_device(uf, *stream);
        ggml_sycl_free_device(t8, *stream);
        g_xe2mx_s48_us4[layer] = us4;
        g_xe2mx_s48_usc4[layer] = usc;
        g_xe2mx_s48_us4g[layer] = (int) g;
    }
    if (g_xe2mx_s48_ras4_cap < xa4) {
        if (g_xe2mx_s48_ras4) {
            ggml_sycl_free_device(g_xe2mx_s48_ras4, *stream);
        }
        g_xe2mx_s48_ras4 = (uint8_t *) ggml_sycl_malloc_device(xa4, *stream);
        g_xe2mx_s48_ras4_cap = g_xe2mx_s48_ras4 ? xa4 : 0;
    }
    if (g_xe2mx_s48_rsas4_cap < xas) {
        if (g_xe2mx_s48_rsas4) {
            ggml_sycl_free_device(g_xe2mx_s48_rsas4, *stream);
        }
        g_xe2mx_s48_rsas4 = (float *) ggml_sycl_malloc_device(xas, *stream);
        g_xe2mx_s48_rsas4_cap = g_xe2mx_s48_rsas4 ? xas : 0;
    }
    if (g_xe2mx_s48_ra8_cap < xn) {
        if (g_xe2mx_s48_ra8) {
            ggml_sycl_free_device(g_xe2mx_s48_ra8, *stream);
        }
        g_xe2mx_s48_ra8 = (int8_t *) ggml_sycl_malloc_device(xn, *stream);
        g_xe2mx_s48_ra8_cap = g_xe2mx_s48_ra8 ? xn : 0;
    }
    if (g_xe2mx_s48_rtmp_cap < (size_t) M * sizeof(float)) {
        if (g_xe2mx_s48_rtmp) {
            ggml_sycl_free_device(g_xe2mx_s48_rtmp, *stream);
        }
        g_xe2mx_s48_rtmp = (float *) ggml_sycl_malloc_device((size_t) M * sizeof(float), *stream);
        g_xe2mx_s48_rtmp_cap = g_xe2mx_s48_rtmp ? (size_t) M * sizeof(float) : 0;
    }
    if (g_xe2mx_s48_xrotf_cap < xn * sizeof(float)) {
        if (g_xe2mx_s48_xrotf) {
            ggml_sycl_free_device(g_xe2mx_s48_xrotf, *stream);
        }
        g_xe2mx_s48_xrotf = (float *) ggml_sycl_malloc_device(xn * sizeof(float), *stream);
        g_xe2mx_s48_xrotf_cap = g_xe2mx_s48_xrotf ? xn * sizeof(float) : 0;
    }
    if (!g_xe2mx_s48_ras4 || !g_xe2mx_s48_rsas4 || !g_xe2mx_s48_ra8 || !g_xe2mx_s48_rtmp || !g_xe2mx_s48_xrotf) {
        return false;
    }
    xe2mx_s48_quant_a(stream, xh, nullptr, nullptr, g_xe2mx_s48_ras4, g_xe2mx_s48_rsas4,
                      g_xe2mx_s48_ra8, g_xe2mx_s48_rtmp, M, K, 0, g, 0);
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        xe2mx_s4_prim * p4 = xe2mx_s48_get(ctx, stream, M, K, K, 2, g);
        if (!p4) {
            return false;
        }
        dnnl::memory sp4 = ctx.get_scratchpad_mem(p4->sp_md, eng, stream);
        p4->prim.execute(strm, {
            {DNNL_ARG_SRC, dnnl::memory(p4->a_md, eng, (void *) g_xe2mx_s48_ras4)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(p4->b_md, eng, (void *) g_xe2mx_s48_us4[layer])},
            {DNNL_ARG_DST, dnnl::memory(p4->c_md, eng, g_xe2mx_s48_xrotf)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(p4->sa_md, eng, (void *) g_xe2mx_s48_rsas4)},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(p4->sc_md, eng, (void *) g_xe2mx_s48_usc4[layer])},
            {DNNL_ARG_SCRATCHPAD, sp4},
        });
    } catch (...) {
        return false;
    }
    float * xf = g_xe2mx_s48_xrotf;
    stream->parallel_for(sycl::range<1>(xn), [=](sycl::id<1> gid) {
        xrot[gid[0]] = sycl::half(xf[gid[0]]);
    });
    return true;
}

// NOPACK_FFN fused bake: original SOA -> dequant -> had128 -> per-row amax/7 -> s4 pack, one row per WG.
// Reads the 4-bit source twice, writes only K/2 bytes; no f32 / s8 temporaries.
static void xe2mx_nf_bake_row(queue_ptr stream, const uint8_t * packed, uint8_t * w4, float * sc, int64_t K, int64_t N, bool sc_known = false) {
    // one row per WG (2 sub-groups x 16 lanes; bake_search winner). Each sub-group does whole 128-blocks: lane holds 8 elems,
    // FWHT bits 1,2,4 in registers, bits 8..64 via shuffle_xor. No SLM, no barriers except the amax reduce.
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    const int nblk = (int) (K / 128);
    const float inv = 0.08838834764831845f;
    static const int nf_w32 = ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_W32", 1);
    if (sc_known && nf_w32) {
        // w32: one lane = one 32-elem block (16B load/store), 2 sub-groups per row. FWHT bits 0..4 in registers,
        // bits 5,6 via 2 shuffles; identical stage order to the per-8 kernel => bit-exact (bake_search 1.35x).
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) N * 32), sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp reassociate(off) contract(off)
                const int64_t n = (int64_t) it.get_group(0);
                auto sg = it.get_sub_group();
                const int L = (int) sg.get_local_linear_id();
                const int sgi = (int) (it.get_local_id(0) >> 4);
                const float s = sc[n];
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                const uint8_t * qrow = qs + (size_t) n * qpitch;
                uint8_t * orow = w4 + (size_t) n * (size_t) (K / 2);
                const int sub = L & 3;
                for (int b0 = sgi * 4; b0 < nblk; b0 += 8) {
                    const int b = b0 + (L >> 2);
                    const bool live = b < nblk;
                    const int ib = (live ? b : b0) * 4 + sub;
                    const sycl::vec<uint32_t, 4> u = *(const sycl::vec<uint32_t, 4> *) (qrow + (size_t) ib * 16u);
                    const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
                    float v[32];
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t uw = u[w];
#pragma unroll
                        for (int j = 0; j < 8; ++j) v[w * 8 + j] = ((float) ((uw >> (4 * j)) & 0xFu) - 8.0f) * d;
                    }
#pragma unroll
                    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if ((j & h) == 0) { const float a = v[j], c = v[j + h]; v[j] = a + c; v[j + h] = a - c; }
                        }
                    }
#pragma unroll
                    for (int hl = 1; hl < 4; hl *= 2) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            const float pv = sycl::permute_group_by_xor(sg, v[j], hl);
                            v[j] = (sub & hl) ? (pv - v[j]) : (v[j] + pv);
                        }
                    }
                    sycl::vec<uint32_t, 4> o;
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        uint32_t ow = 0;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            int q = (int) sycl_rint_i((v[w * 8 + j] * inv) * id);
                            q = q < -8 ? -8 : (q > 7 ? 7 : q);
                            ow |= ((uint32_t) q & 0xFu) << (4 * j);
                        }
                        o[w] = ow;
                    }
                    if (live) *(sycl::vec<uint32_t, 4> *) (orow + (size_t) b * 64u + (size_t) sub * 16u) = o;
                }
            });
        return;
    }
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) N * 32), sycl::range<1>(32)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp reassociate(off) contract(off)
            const int n = (int) it.get_group(0);
            auto sg = it.get_sub_group();
            const int L = (int) sg.get_local_linear_id();
            const int sgi = (int) (it.get_local_id(0) >> 4);
            auto had = [&](int b, float * v) {
                const int ib = b * 4 + (L >> 2);
                const uint32_t u = *(const uint32_t *) (qs + (size_t) n * qpitch + (size_t) ib * 16u + (size_t) ((L & 3) * 4));
                const float d = (float) dpl[(size_t) ib * (size_t) N + (size_t) n];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] = ((float) ((u >> (4 * j)) & 0xFu) - 8.0f) * d;
                }
#pragma unroll
                for (int h = 1; h < 8; h *= 2) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if ((j & h) == 0) {
                            const float a = v[j], c = v[j + h];
                            v[j] = a + c;
                            v[j + h] = a - c;
                        }
                    }
                }
#pragma unroll
                for (int hl = 1; hl < 16; hl *= 2) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float p = sycl::permute_group_by_xor(sg, v[j], hl);
                        v[j] = (L & hl) ? (p - v[j]) : (v[j] + p);
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] *= inv;
                }
            };
            float s;
            if (sc_known) {
                s = sc[n];
            } else {
                float amax = 0.0f;
                for (int b = sgi; b < nblk; b += 2) {
                    float v[8];
                    had(b, v);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        amax = sycl::fmax(amax, sycl::fabs(v[j]));
                    }
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                s = gmax > 0.0f ? gmax / 7.0f : 0.0f;
                if (it.get_local_id(0) == 0) {
                    sc[n] = s;
                }
            }
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            for (int b = sgi; b < nblk; b += 2) {
                float v[8];
                had(b, v);
                uint32_t o = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int q = (int) sycl_rint_i(v[j] * id);
                    q = q < -8 ? -8 : (q > 7 ? 7 : q);
                    o |= ((uint32_t) q & 0xFu) << (4 * j);
                }
                *(uint32_t *) (w4 + (size_t) n * (size_t) (K / 2) + (size_t) b * 64u + (size_t) L * 4u) = o;
            }
        });
}

static bool xe2mx_s44_cached(queue_ptr stream, const uint8_t * packed, int64_t K, int64_t N, xe2mx_s8r_view * out,
                             int layer = -1, int64_t g_ov = 0) {
    const int64_t s44n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44N", 10240);
    const int dn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN", 0);
    const int s44on = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44", 0);
    const int64_t g = g_ov > 0 ? g_ov : (int64_t) xe2mx_s44_g();
    const int ok_ffn = s44on && N >= s44n;
    const int ok_dn = dn && N >= 4096 && K >= 16384;
    if (!packed || !out || (K % 32) != 0 || (K % g) != 0 || (N % 16) != 0 || (!ok_ffn && !ok_dn)) {
        return false;
    }
    const int extra = N < 16384;
    const int h = xe2mx_s44_h_on(layer, K, N);
    const int zp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0);
    const int tag = 25 + (h ? 2 : 0) + (zp ? 1 : 0) + (int) (g / 32) + (h ? xe2mx_s44_hs() * 8 : 0);
    const int64_t ng = K / g;
    const size_t wbytes = (size_t) N * (size_t) (K / 2);
    const size_t scb = (size_t) ng * (size_t) N * sizeof(float);
    const size_t bytes = wbytes + scb + (zp ? scb : 0);
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.host && !it->second.packed
        && it->second.vnni == tag && it->second.host_bytes == bytes && it->second.K == K && it->second.N == N) {
        xe2mx_lo_hostpack_ensure(*stream);
    }
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        xe2mx_lo_hostpack_fence(*stream, it->second);
        out->w = (int8_t *) it->second.packed;
        out->sc = (float *) ((char *) it->second.packed + wbytes);
        out->aux = zp ? (float *) ((char *) it->second.packed + wbytes + scb) : nullptr;
        return true;
    }
    if (it != g_xe2mx_wcache.end()) {
        return false;
    }
    if (bytes <= (1ull << 20)) {
        return false;
    }
    int room = 1;
    if (((extra && ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_MID", 0)) || ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0)) && !ggml_sycl_get_env("GGML_SYCL_XE2MX_NPM_S44KEEP", 0)) {
        // NOPACK_MID: do not re-grow a persistent s44 copy for mid tensors (would eat the freed VRAM)
        // NOPACK_FFN: optional small persistent budget (NF_KEEP_MB) so some FFN packs skip the per-ubatch re-bake.
        const size_t nf_keep_cap = (size_t) ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_KEEP_MB", 0) << 20;
        if (ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0) && g_xe2mx_nf_keep_used + bytes <= nf_keep_cap) {
            g_xe2mx_nf_keep_used += bytes;
        } else {
            room = 0;
        }
    }
    if (extra) {
        if (g_xe2mx_s8r_ex_used + bytes > xe2mx_s8r_ex_cap()) {
            room = 0;
        }
    } else if (g_xe2mx_wcache_used + bytes > XE2MX_WCACHE_CAP) {
        room = 0;
    }
    void * persist = nullptr;
    if (room) {
        persist = ggml_sycl_malloc_device(bytes, *stream);
    }
    const size_t tbytes = (size_t) N * (size_t) K;
    if (persist) {
        uint8_t * w4 = (uint8_t *) persist;
        float * sc = (float *) ((char *) persist + wbytes);
        float * bsum = zp ? (float *) ((char *) persist + wbytes + scb) : nullptr;
        void * tmp = ggml_sycl_malloc_device(tbytes, *stream);
        if (!tmp) {
            ggml_sycl_free_device(persist, *stream);
            persist = nullptr;
        } else {
            int8_t * w8 = (int8_t *) tmp;
            if (h || zp) {
                const size_t fbytes = tbytes * sizeof(float);
                void * ftmp = ggml_sycl_malloc_device(fbytes, *stream);
                if (!ftmp) {
                    ggml_sycl_free_device(tmp, *stream);
                    ggml_sycl_free_device(persist, *stream);
                    persist = nullptr;
                } else {
                    xe2mx_s44_unpack_f32(stream, packed, (float *) ftmp, K, N);
                    if (h) {
                        xe2mx_had128_f32(stream, (float *) ftmp, N, K);
                    }
                    xe2mx_s44_quant_f32(stream, (const float *) ftmp, w8, sc, bsum, K, N, g);
                    xe2mx_pack_s8_s4(stream, w8, w4, K, N);
                    stream->wait();
                    ggml_sycl_free_device(ftmp, *stream);
                    ggml_sycl_free_device(tmp, *stream);
                    if (extra) {
                        g_xe2mx_s8r_ex_used += bytes;
                    } else {
                        g_xe2mx_wcache_used += bytes;
                    }
                    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
                    xe2mx_nf_hp_register(*stream, g_xe2mx_wcache[(const void *) packed], (const void *) packed, extra);
                    out->w = (int8_t *) w4;
                    out->sc = sc;
                    out->aux = bsum;
                    return true;
                }
            } else {
                xe2mx_s44_requant(stream, packed, w8, sc, K, N, g);
                xe2mx_pack_s8_s4(stream, w8, w4, K, N);
                stream->wait();
                ggml_sycl_free_device(tmp, *stream);
                if (extra) {
                    g_xe2mx_s8r_ex_used += bytes;
                } else {
                    g_xe2mx_wcache_used += bytes;
                }
                g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
                xe2mx_nf_hp_register(*stream, g_xe2mx_wcache[(const void *) packed], (const void *) packed, extra);
                out->w = (int8_t *) w4;
                out->sc = sc;
                out->aux = nullptr;
                return true;
            }
        }
    }
    if (!persist && h && !zp && ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0)
        && (N >= 16384 || K >= 16384)) {
        // NOPACK_FFN transient bake: FFN has no persistent LO pack (decode reads exact original SOA);
        // prefill re-bakes the s4 pack into reusable scratch each call. In-order queue keeps it safe.
        struct nf_buffers {
            void * f = nullptr, * w8 = nullptr, * w4 = nullptr;
            size_t fcap = 0, cap8 = 0, cap4 = 0;
            std::unordered_map<const void *, float *> scales;
        };
        static nf_buffers buffers[GGML_SYCL_MAX_DEVICES];
        int device = -1;
        for (int d = 0; d < ggml_sycl_info().device_count; ++d) {
            if (stream->get_device() == dpct::get_device(d)) { device = d; break; }
        }
        GGML_ASSERT(device >= 0);
        auto & b = buffers[device];
        auto & nf_f = b.f; auto & nf_fcap = b.fcap;
        auto & nf_8 = b.w8; auto & nf_8cap = b.cap8;
        auto & nf_w = b.w4; auto & nf_wcap = b.cap4;
        const size_t fb = tbytes * sizeof(float);
        const size_t wb = wbytes + scb;
        const bool nf_fast = g == K && xe2mx_s44_hs() == 128 && (K % 128) == 0 && !ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_SLOW", 0);
        if (!nf_fast && nf_fcap < fb) { if (nf_f) { stream->wait(); ggml_sycl_free_device(nf_f, *stream); } nf_f = ggml_sycl_malloc_device(fb, *stream); nf_fcap = nf_f ? fb : 0; }
        if (!nf_fast && nf_8cap < tbytes) { if (nf_8) { stream->wait(); ggml_sycl_free_device(nf_8, *stream); } nf_8 = ggml_sycl_malloc_device(tbytes, *stream); nf_8cap = nf_8 ? tbytes : 0; }
        if (nf_wcap < wb) { if (nf_w) { stream->wait(); ggml_sycl_free_device(nf_w, *stream); } nf_w = ggml_sycl_malloc_device(wb, *stream); nf_wcap = nf_w ? wb : 0; }
        if ((nf_fast || (nf_f && nf_8)) && nf_w) {
            uint8_t * w4 = (uint8_t *) nf_w;
            float * sc = (float *) ((char *) nf_w + wbytes);
            static int nf_chk = ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_CHECK", 0);
            if (nf_chk > 0) {
                nf_chk--;
                const size_t wb2 = wbytes + (size_t) N * 4;
                std::vector<uint8_t> a(wb2), b(wb2);
                void * t1 = ggml_sycl_malloc_device(wb2, *stream);
                void * f1 = ggml_sycl_malloc_device(fb, *stream);
                void * s1 = ggml_sycl_malloc_device(tbytes, *stream);
                xe2mx_nf_bake_row(stream, packed, (uint8_t *) t1, (float *) ((char *) t1 + wbytes), K, N);
                stream->memcpy(a.data(), t1, wb2).wait();
                xe2mx_s44_unpack_f32(stream, packed, (float *) f1, K, N);
                xe2mx_had128_f32(stream, (float *) f1, N, K);
                xe2mx_s44_quant_f32(stream, (const float *) f1, (int8_t *) s1, (float *) ((char *) t1 + wbytes), nullptr, K, N, g);
                xe2mx_pack_s8_s4(stream, (const int8_t *) s1, (uint8_t *) t1, K, N);
                stream->memcpy(b.data(), t1, wb2).wait();
                size_t dw = 0, ds = 0, first = (size_t) -1;
                for (size_t i = 0; i < wbytes; ++i) if (a[i] != b[i]) { dw++; if (first == (size_t) -1) first = i; }
                const float * fa = (const float *) (a.data() + wbytes);
                const float * fb2 = (const float *) (b.data() + wbytes);
                float mr = 0.f;
                for (int64_t i = 0; i < N; ++i) { if (fa[i] != fb2[i]) ds++; mr = std::max(mr, std::fabs(fa[i] - fb2[i]) / (std::fabs(fb2[i]) + 1e-20f)); }
                GGML_SYCL_XE2_LOG( "NF_CHECK N=%ld K=%ld g=%ld diff_w=%zu/%zu first=%zd diff_sc=%zu maxrel=%g a0=%02x b0=%02x sc0 %g %g\n",
                    (long) N, (long) K, (long) g, dw, wbytes, (ssize_t) first, ds, mr,
                    first != (size_t) -1 ? a[first] : 0, first != (size_t) -1 ? b[first] : 0, fa[0], fb2[0]);
                fflush(stderr);
                ggml_sycl_free_device(t1, *stream); ggml_sycl_free_device(f1, *stream); ggml_sycl_free_device(s1, *stream);
            }
            if (g == K && xe2mx_s44_hs() == 128 && (K % 128) == 0 && !ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_SLOW", 0)) {
                auto & nf_sc = b.scales;
                auto sit = nf_sc.find((const void *) packed);
                if (sit != nf_sc.end() && !ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_NOSC", 0)) {
                    stream->memcpy(sc, sit->second, (size_t) N * sizeof(float));
                    xe2mx_nf_bake_row(stream, packed, w4, sc, K, N, true);
                } else {
                    xe2mx_nf_bake_row(stream, packed, w4, sc, K, N, false);
                    float * keep = (float *) ggml_sycl_malloc_device((size_t) N * sizeof(float), *stream);
                    if (keep) {
                        stream->memcpy(keep, sc, (size_t) N * sizeof(float));
                        nf_sc[(const void *) packed] = keep;
                    }
                }
            } else {
                xe2mx_s44_unpack_f32(stream, packed, (float *) nf_f, K, N);
                xe2mx_had128_f32(stream, (float *) nf_f, N, K);
                xe2mx_s44_quant_f32(stream, (const float *) nf_f, (int8_t *) nf_8, sc, nullptr, K, N, g);
                xe2mx_pack_s8_s4(stream, (const int8_t *) nf_8, w4, K, N);
            }
            static int nf_log = 0;
            if (nf_log < 3) { nf_log++; GGML_SYCL_XE2_LOG( "xe2mx NOPACK_FFN transient bake N=%ld K=%ld g=%ld\n", (long) N, (long) K, (long) g); fflush(stderr); }
            out->w = (int8_t *) w4;
            out->sc = sc;
            out->aux = nullptr;
            return true;
        }
    }
    if (h || zp || !dn) {
        return false;
    }
    if (g_xe2mx_s44_t8cap < tbytes) {
        void * p = ggml_sycl_malloc_device(tbytes, *stream);
        if (!p) {
            return false;
        }
        g_xe2mx_s44_t8 = (int8_t *) p;
        g_xe2mx_s44_t8cap = tbytes;
    }
    if (g_xe2mx_s44_tcap < wbytes) {
        void * p = ggml_sycl_malloc_device(wbytes, *stream);
        if (!p) {
            return false;
        }
        g_xe2mx_s44_tmp = (uint8_t *) p;
        g_xe2mx_s44_tcap = wbytes;
    }
    if (g_xe2mx_s44_tsccap < scb) {
        void * p = ggml_sycl_malloc_device(scb, *stream);
        if (!p) {
            return false;
        }
        g_xe2mx_s44_tsc = (float *) p;
        g_xe2mx_s44_tsccap = scb;
    }
    if (!g_xe2mx_s44_t8 || !g_xe2mx_s44_tmp || !g_xe2mx_s44_tsc) {
        return false;
    }
    xe2mx_s44_requant(stream, packed, g_xe2mx_s44_t8, g_xe2mx_s44_tsc, K, N, g);
    xe2mx_pack_s8_s4(stream, g_xe2mx_s44_t8, g_xe2mx_s44_tmp, K, N);
    out->w = (int8_t *) g_xe2mx_s44_tmp;
    out->sc = g_xe2mx_s44_tsc;
    out->aux = nullptr;
    return true;
}

static bool xe2mx_s8pc_cached(queue_ptr stream, const uint8_t * packed, int64_t K, int64_t N, xe2mx_s8r_view * out) {
    if (!packed || !out || (K % 32) != 0 || N < 4096 || (N % 16) != 0) {
        return false;
    }
    const int tag = 22;
    const int64_t nb = K / 32;
    const size_t wbytes = (size_t) N * (size_t) nb * 16u;
    const size_t bytes = wbytes + (size_t) N * sizeof(float);
    auto it = g_xe2mx_wcache.find((const void *) packed);
    if (it != g_xe2mx_wcache.end() && it->second.vnni == tag && it->second.bytes == bytes
        && it->second.K == K && it->second.N == N && it->second.packed) {
        out->w = (int8_t *) it->second.packed;
        out->sc = (float *) ((char *) it->second.packed + wbytes);
        return true;
    }
    if (it != g_xe2mx_wcache.end()) {
        return false;
    }
    if (bytes <= (1ull << 20)) {
        return false;
    }
    if (g_xe2mx_wcache_used + bytes > XE2MX_WCACHE_CAP) {
        return false;
    }
    void * persist = ggml_sycl_malloc_device(bytes, *stream);
    if (!persist) {
        return false;
    }
    uint8_t * w = (uint8_t *) persist;
    float * sc = (float *) ((char *) persist + wbytes);
    xe2mx_s4t_pack(stream, packed, w, sc, K, N, 0);
    g_xe2mx_wcache_used += bytes;
    g_xe2mx_wcache[(const void *) packed] = { persist, bytes, K, N, tag };
    out->w = (int8_t *) w;
    out->sc = sc;
    return true;
}
static double g_xe2mx_sum_deq = 0;
static double g_xe2mx_sum_x = 0;
static double g_xe2mx_sum_gemm = 0;
static int g_xe2mx_sum_n = 0;
static int g_xe2mx_sum_reg = 0;
static const ggml_tensor * g_xe2mx_xh_src = nullptr;
static sycl::half * g_xe2mx_xh_buf = nullptr;
static size_t g_xe2mx_xh_cap = 0;
// device the xh cache buffer was allocated on (see g_xe2mx_ar_dev).
static int g_xe2mx_xh_dev = -1;
static int64_t g_xe2mx_xh_M = 0;
static int64_t g_xe2mx_xh_K = 0;
static uint64_t g_xe2mx_xh_epoch = 0;
// One entry per early-computed matmul per graph (MG on a 64-layer Qwen3.8 verify graph needs 176).
constexpr int XE2MX_FN_SKIP_MAX = 512;
static const ggml_tensor * g_xe2mx_fn_skip[XE2MX_FN_SKIP_MAX];
static int g_xe2mx_fn_nskip = 0;
// GGML_SYCL_XE2MX_BATCH: members already computed by an earlier batch head. Reset per graph.
constexpr int XE2MX_BT_SKIP_MAX = 1024;
static const ggml_tensor * g_xe2mx_bt_skip[XE2MX_BT_SKIP_MAX];
static int g_xe2mx_bt_nskip = 0;
struct xe2mx_bt_stats {
    int heads;    // batched launches
    int members;  // MUL_MATs folded into those launches, heads included
    int seen;     // XE2MX4 MUL_MATs scanned in windows
    int r_shape;  // not the plain q32 launch_soa shape / M differs from head
    int r_fuse;   // owned by GLU / MG / FN / FADD / ZSILU
    int r_src;    // src1 not final at the head
    int r_dst;    // dst bytes still live between head and member
};
static xe2mx_bt_stats g_xe2mx_bt_st = {};
// FN deferred landing: b's fused result waits here until the graph reaches b.
// Slots are per queue: the MTP draft context has its own backend queue and also fuses (attn_k+attn_v), and a shared
// slot let the draft's FN kernel overwrite a target slot before the target's landing copy ran (nondeterministic text).
struct xe2mx_fn_defer_slot {
    void * buf = nullptr;
    size_t cap = 0;
    size_t bytes = 0;
};
static std::unordered_map<const void *, std::vector<xe2mx_fn_defer_slot>> g_xe2mx_fn_defer_q;
static xe2mx_fn_defer_slot * xe2mx_fn_slots(const void * q) {
    auto & v = g_xe2mx_fn_defer_q[q];
    if (v.empty()) {
        v.resize(XE2MX_FN_SKIP_MAX);
    }
    return v.data();
}

// Called while the queue is NOT recording. Grows every defer slot to the largest
// XE2MX4 matmul in this graph so mg_run's wait+malloc path does not run during capture.
void ggml_sycl_xe2mx_graph_record_prepare(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    if (!cgraph || !ctx.stream()) {
        return;
    }
    xe2mx_lo_hostpack_drop(*ctx.stream());
    size_t need = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
            continue;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || t->type != GGML_TYPE_F32) {
            continue;
        }
        const size_t bytes = (size_t) t->src[0]->ne[1] * (size_t) ggml_nrows(t->src[1]) * sizeof(float);
        if (bytes > need) {
            need = bytes;
        }
    }
    if (need == 0) {
        return;
    }
    xe2mx_fn_defer_slot * slots = xe2mx_fn_slots((const void *) ctx.stream());
    for (int s = 0; s < XE2MX_FN_SKIP_MAX; ++s) {
        if (slots[s].cap >= need) {
            continue;
        }
        if (slots[s].buf) {
            ctx.stream()->wait();
            ggml_sycl_free_device(slots[s].buf, *ctx.stream());
            slots[s].buf = nullptr;
            slots[s].cap = 0;
        }
        slots[s].buf = ggml_sycl_malloc_device(need, *ctx.stream());
        if (!slots[s].buf) {
            return;
        }
        slots[s].cap = need;
    }
}
static int g_xe2mx_fn_landed = 0;
static long g_xe2mx_fn_graphs = 0;
static uint64_t g_xe2mx_xh_hit_epoch = 0;
static const ggml_tensor * g_xe2mx_aq_src = nullptr;
static int8_t * g_xe2mx_aq_buf = nullptr;
static float * g_xe2mx_daq_buf = nullptr;
static size_t g_xe2mx_aq_cap = 0;
static size_t g_xe2mx_daq_cap = 0;
static int64_t g_xe2mx_aq_M = 0;
static int64_t g_xe2mx_aq_K = 0;
static uint64_t g_xe2mx_aq_hit_epoch = 0;
static const sycl::half * g_xe2mx_ar_xh = nullptr;
static uint8_t * g_xe2mx_ar_buf = nullptr;
static size_t g_xe2mx_ar_cap = 0;
static size_t g_xe2mx_ar_bytes = 0;
static int64_t g_xe2mx_ar_M = 0;
static int64_t g_xe2mx_ar_K = 0;
static uint64_t g_xe2mx_ar_hit_epoch = 0;
// device the ar cache buffer was allocated on: the cache is process-global
// and a dual-GPU run must not hand card0's reorder output to card1's gemm
// (cross-device USM in kernel args -> L0 execute fails, UR 39).
static int g_xe2mx_ar_dev = -1;

static uint8_t * g_xe2mx_loa = nullptr;
static float * g_xe2mx_losa = nullptr;
static const void * g_xe2mx_loa_src = nullptr;
static const sycl::half * g_xe2mx_loa_xh = nullptr;
static size_t g_xe2mx_loa_cap = 0;
static size_t g_xe2mx_losa_cap = 0;
static int64_t g_xe2mx_loa_M = 0;
static int64_t g_xe2mx_loa_K = 0;
static int64_t g_xe2mx_loa_g = 0;
static uint64_t g_xe2mx_loa_epoch = 0;

static int g_xe2mx_s44_pack_n = 0;
static int g_xe2mx_s44_pack_hit = 0;
static std::unordered_set<const void *> g_xe2mx_s44_uniq;

enum { XE2MX_S44_SLOT_MAX = 32 };
struct xe2mx_s44_aslot {
    uint8_t * a4 = nullptr;
    float * sa = nullptr;
    float * za = nullptr;
    size_t a4_cap = 0;
    size_t sa_cap = 0;
    size_t za_cap = 0;
    int64_t M = 0;
    int64_t K = 0;
    uint64_t epoch = 0;
    uint64_t tick = 0;
    const sycl::half * xh = nullptr;
    const float * xf = nullptr;
    const void * src = nullptr;
    int sa_row = -1;
    int qmode = -1;
    // device the a4/sa/za buffers live on: the slot array is process-global,
    // and on a dual-GPU run the second card's victim path must NOT reuse the
    // first card's buffers (cross-device USM in kernel args -> L0 execute
    // fails with UR 39 OUT_OF_DEVICE_MEMORY at M>=1024 where the pack fires).
    int dev = -1;
};
static xe2mx_s44_aslot g_xe2mx_s44_slot[XE2MX_S44_SLOT_MAX];
static uint64_t g_xe2mx_s44_tick = 0;

static int xe2mx_s44_nslot(void) {
    int n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44SLOTS", 1);
    if (n < 1) {
        n = 1;
    }
    if (n > XE2MX_S44_SLOT_MAX) {
        n = XE2MX_S44_SLOT_MAX;
    }
    return n;
}

static int xe2mx_s44_slot_find(const void * asrc, const sycl::half * xh, const float * xf, int64_t M, int64_t K,
                               int s8ra, int qmode, size_t an4, size_t sa_need, size_t za_need, int dev_id) {
    if (!asrc) {
        return -1;
    }
    const int dev = dev_id;
    const int n = xe2mx_s44_nslot();
    for (int i = 0; i < n; i++) {
        xe2mx_s44_aslot & s = g_xe2mx_s44_slot[i];
        if (!s.a4 || !s.sa || s.src != asrc || s.dev != dev) {
            continue;
        }
        if (s.M != M || s.K != K || s.qmode != qmode || s.sa_row != s8ra) {
            continue;
        }
        if (s.epoch != g_xe2mx_xh_epoch) {
            continue;
        }
        if (s.a4_cap < an4 || s.sa_cap < sa_need) {
            continue;
        }
        if (za_need && (!s.za || s.za_cap < za_need)) {
            continue;
        }
        const int same = (xh && s.xh == xh) || (!xh && xf && s.xf == xf);
        if (!same) {
            continue;
        }
        return i;
    }
    return -1;
}

static int xe2mx_s44_slot_victim(void) {
    const int n = xe2mx_s44_nslot();
    int v = 0;
    uint64_t best = g_xe2mx_s44_slot[0].tick;
    for (int i = 0; i < n; i++) {
        if (!g_xe2mx_s44_slot[i].a4) {
            return i;
        }
        if (g_xe2mx_s44_slot[i].tick < best) {
            best = g_xe2mx_s44_slot[i].tick;
            v = i;
        }
    }
    return v;
}

static int xe2mx_s44_pack_slot_xf(queue_ptr stream, const void * asrc, const float * xf,
                                  int64_t M, int64_t K) {
    if (!asrc || !xf || M < 32 || (K % 128) != 0) {
        return 0;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return 0;
    }
    const int s44g = xe2mx_s44_g();
    const int hs = xe2mx_s44_hs();
    if (s44g < hs || (K % s44g) != 0 || (s44g % hs) != 0 || s44g > 1024) {
        return 0;
    }
    int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 2 && s44hilp != 4) {
        s44hilp = 0;
    }
    int s44hwide = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWIDE", 0);
    if (s44hwide != 2 && s44hwide != 4 && s44hwide != 8) {
        s44hwide = 0;
    }
    const int s44h = 1;
    const int s44zp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0);
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int s44_pack_a = s44from >= 0;
    const int s44hrow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HROW", 0) && s44h && s44_pack_a;
    const int s44hocc = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HOCC", 0) && s44h && s44_pack_a && !s44hrow;
    int s8ra = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0);
    if (s44_pack_a && s44arow && !s44hrow) {
        s8ra = 1;
    }
    if (s8ra || s44zp || s44hrow || s44hocc) {
        return 0;
    }
    const int qmode = xe2mx_s44_qmode(s8ra, s44h, s44zp, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44_pack_a && s44pn, s44hrow, s44hocc, s44hwide, s44hilp);
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t sa_need = (size_t) M * (size_t) (K / s44g) * sizeof(float);
    const int dev_id = xe2mx_q_dev(*stream);
    if (xe2mx_s44_slot_find(asrc, nullptr, xf, M, K, s8ra, qmode, an4, sa_need, 0, dev_id) >= 0) {
        return 1;
    }
    const int vi = xe2mx_s44_slot_victim();
    xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
    if (vs.a4_cap < an4 || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return 0;
        }
        vs.a4 = (uint8_t *) p;
        vs.a4_cap = an4;
    }
    if (vs.sa_cap < sa_need || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(sa_need, *stream);
        if (!p) {
            return 0;
        }
        vs.sa = (float *) p;
        vs.sa_cap = sa_need;
    }
    vs.dev = dev_id;
    if (!xe2mx_quant_a_s4_nibble_had_f32_rte(stream, xf, vs.a4, vs.sa, M, K, (int64_t) s44g)) {
        return 0;
    }
    vs.xh = nullptr;
    vs.xf = xf;
    vs.src = asrc;
    vs.M = M;
    vs.K = K;
    vs.epoch = g_xe2mx_xh_epoch;
    vs.sa_row = s8ra;
    vs.qmode = qmode;
    vs.tick = ++g_xe2mx_s44_tick;
    return 1;
}

bool ggml_sycl_xe2mx_s44_pack_xf(queue_ptr stream, const void * asrc, const float * xf,
                                 int64_t M, int64_t K) {
    return xe2mx_s44_pack_slot_xf(stream, asrc, xf, M, K) != 0;
}

bool ggml_sycl_xe2mx_s44_rms_hilp(queue_ptr stream, const void * asrc, const float * x, const float * mul,
                                  float * y, int64_t M, int64_t K, float eps) {
    if (!asrc || !x || !mul || !y || M < 32 || (K % 128) != 0) {
        return false;
    }
    // S44 slot state is per-process, keyed on scratch pointers + a global epoch
    // that bumps per graph_compute. With >1 device the two cards interleave
    // graph_computes (TP executor), thrashing the epoch mid-graph and leaving
    // packs half-consumed on the next ubatch: async UR 40 on the 2nd prefill
    // graph (S45). One-card product cell keeps the path.
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMSHILP", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return false;
    }
    const int s44g = xe2mx_s44_g();
    const int hs = xe2mx_s44_hs();
    if (hs != 128 || s44g != 1024 || (K % s44g) != 0) {
        return false;
    }
    int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 4) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0)) {
        return false;
    }
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int qmode = xe2mx_s44_qmode(0, 1, 0, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44from >= 0 && s44pn, 0, 0, 0, s44hilp);
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t sa_need = (size_t) M * (size_t) (K / s44g) * sizeof(float);
    const int dev_id = xe2mx_q_dev(*stream);
    if (xe2mx_s44_slot_find(asrc, nullptr, y, M, K, 0, qmode, an4, sa_need, 0, dev_id) >= 0) {
        return false;
    }
    const int vi = xe2mx_s44_slot_victim();
    xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
    if (vs.a4_cap < an4 || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return false;
        }
        vs.a4 = (uint8_t *) p;
        vs.a4_cap = an4;
    }
    if (vs.sa_cap < sa_need || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(sa_need, *stream);
        if (!p) {
            return false;
        }
        vs.sa = (float *) p;
        vs.sa_cap = sa_need;
    }
    vs.dev = dev_id;
    if (!xe2mx_rms_mul_hilp_pack(stream, x, mul, y, vs.a4, vs.sa, M, K, (int64_t) s44g, eps)) {
        return false;
    }
    vs.xh = nullptr;
    vs.xf = y;
    vs.src = asrc;
    vs.M = M;
    vs.K = K;
    vs.epoch = g_xe2mx_xh_epoch;
    vs.sa_row = 0;
    vs.qmode = qmode;
    vs.tick = ++g_xe2mx_s44_tick;
    static int nrms = 0;
    if (nrms < 4) {
        nrms++;
        GGML_SYCL_XE2_LOG( "xe2mx s44 rmshilp M=%ld K=%ld\n", (long) M, (long) K);
        fflush(stderr);
    }
    return true;
}

bool ggml_sycl_xe2mx_s44_add_rms_hilp(queue_ptr stream, const void * asrc, const float * a, const float * b,
                                      float * add_dst, const float * mul, float * y, int64_t M, int64_t K,
                                      float eps, int drop_add) {
    if (!asrc || !a || !b || !mul || !y || M < 32 || (K % 128) != 0) {
        return false;
    }
    // multi-device guard, see ggml_sycl_xe2mx_s44_rms_hilp (S45)
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMSHILP", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return false;
    }
    const int s44g = xe2mx_s44_g();
    const int hs = xe2mx_s44_hs();
    if (hs != 128 || s44g != 1024 || (K % s44g) != 0) {
        return false;
    }
    int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 4) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0)) {
        return false;
    }
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int qmode = xe2mx_s44_qmode(0, 1, 0, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44from >= 0 && s44pn, 0, 0, 0, s44hilp);
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t sa_need = (size_t) M * (size_t) (K / s44g) * sizeof(float);
    const int dev_id = xe2mx_q_dev(*stream);
    if (xe2mx_s44_slot_find(asrc, nullptr, y, M, K, 0, qmode, an4, sa_need, 0, dev_id) >= 0) {
        return true;
    }
    const int vi = xe2mx_s44_slot_victim();
    xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
    if (vs.a4_cap < an4 || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return false;
        }
        vs.a4 = (uint8_t *) p;
        vs.a4_cap = an4;
    }
    if (vs.sa_cap < sa_need || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(sa_need, *stream);
        if (!p) {
            return false;
        }
        vs.sa = (float *) p;
        vs.sa_cap = sa_need;
    }
    vs.dev = dev_id;
    if (!xe2mx_add_rms_mul_hilp_pack(stream, a, b, add_dst, mul, y, vs.a4, vs.sa, M, K, (int64_t) s44g, eps, drop_add)) {
        return false;
    }
    vs.xh = nullptr;
    vs.xf = y;
    vs.src = asrc;
    vs.M = M;
    vs.K = K;
    vs.epoch = g_xe2mx_xh_epoch;
    vs.sa_row = 0;
    vs.qmode = qmode;
    vs.tick = ++g_xe2mx_s44_tick;
    static int nadd = 0;
    if (nadd < 6) {
        nadd++;
        GGML_SYCL_XE2_LOG( "xe2mx s44 addrms M=%ld K=%ld\n", (long) M, (long) K);
        fflush(stderr);
    }
    return true;
}

bool ggml_sycl_xe2mx_s44_add_rms_bufs(queue_ptr stream, const void * asrc, const float * y, int64_t M, int64_t K,
                                      uint8_t ** A4, float ** sa, int * gout) {
    if (A4) {
        *A4 = nullptr;
    }
    if (sa) {
        *sa = nullptr;
    }
    if (!stream || !asrc || !y || !A4 || !sa || M < 32 || K > 8192 || (K % 128) != 0) {
        return false;
    }
    // multi-device guard, see ggml_sycl_xe2mx_s44_rms_hilp (S45)
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMSHILP", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return false;
    }
    const int s44g = xe2mx_s44_g_kn(K, K);
    const int hs = xe2mx_s44_hs();
    if (hs != 128 || s44g < 128 || s44g > 1024 || (K % s44g) != 0) {
        return false;
    }
    const int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 4) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILPM", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN64", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWG", 128) != 128) {
        return false;
    }
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int qmode = xe2mx_s44_qmode(0, 1, 0, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44from >= 0 && s44pn, 0, 0, 0, s44hilp);
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t sa_need = (size_t) M * (size_t) (K / s44g) * sizeof(float);
    const int dev_id = xe2mx_q_dev(*stream);
    if (xe2mx_s44_slot_find(asrc, nullptr, y, M, K, 0, qmode, an4, sa_need, 0, dev_id) >= 0) {
        return false;
    }
    const int vi = xe2mx_s44_slot_victim();
    xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
    if (vs.a4_cap < an4 || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return false;
        }
        vs.a4 = (uint8_t *) p;
        vs.a4_cap = an4;
    }
    if (vs.sa_cap < sa_need || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(sa_need, *stream);
        if (!p) {
            return false;
        }
        vs.sa = (float *) p;
        vs.sa_cap = sa_need;
    }
    vs.dev = dev_id;
    vs.xh = nullptr;
    vs.xf = y;
    vs.src = asrc;
    vs.M = M;
    vs.K = K;
    vs.epoch = g_xe2mx_xh_epoch;
    vs.sa_row = 0;
    vs.qmode = qmode;
    vs.tick = ++g_xe2mx_s44_tick;
    *A4 = vs.a4;
    *sa = vs.sa;
    if (gout) {
        *gout = s44g;
    }
    static int nbuf = 0;
    if (nbuf < 4) {
        nbuf++;
        GGML_SYCL_XE2_LOG( "xe2mx s44 addrmsbuf M=%ld K=%ld\n", (long) M, (long) K);
        fflush(stderr);
    }
    return true;
}

bool ggml_sycl_xe2mx_s44_silu_hilp(queue_ptr stream, const void * asrc, const float * gate, const float * up,
                                   float * y, int64_t M, int64_t K) {
    if (!asrc || !gate || !up || !y || M < 32 || (K % 1024) != 0) {
        return false;
    }
    // multi-device guard, see ggml_sycl_xe2mx_s44_rms_hilp (S45)
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S44SILUHILP", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
        || !ggml_sycl_get_env("GGML_SYCL_XE2MX_S44H", 0)) {
        return false;
    }
    const int s44g = xe2mx_s44_g();
    const int hs = xe2mx_s44_hs();
    if (hs != 128 || s44g != 1024 || (K % s44g) != 0) {
        return false;
    }
    int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 4) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILPM", 0)
        || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN64", 0)) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWG", 128) != 128) {
        return false;
    }
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int qmode = xe2mx_s44_qmode(0, 1, 0, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44from >= 0 && s44pn, 0, 0, 0, s44hilp);
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t sa_need = (size_t) M * (size_t) (K / s44g) * sizeof(float);
    const int dev_id = xe2mx_q_dev(*stream);
    if (xe2mx_s44_slot_find(asrc, nullptr, y, M, K, 0, qmode, an4, sa_need, 0, dev_id) >= 0) {
        return true;
    }
    const int vi = xe2mx_s44_slot_victim();
    xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
    if (vs.a4_cap < an4 || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return false;
        }
        vs.a4 = (uint8_t *) p;
        vs.a4_cap = an4;
    }
    if (vs.sa_cap < sa_need || vs.dev != dev_id) {
        void * p = ggml_sycl_malloc_device(sa_need, *stream);
        if (!p) {
            return false;
        }
        vs.sa = (float *) p;
        vs.sa_cap = sa_need;
    }
    vs.dev = dev_id;
    if (!xe2mx_silu_up_hilp_pack(stream, gate, up, y, vs.a4, vs.sa, M, K, (int64_t) s44g)) {
        return false;
    }
    vs.xh = nullptr;
    vs.xf = y;
    vs.src = asrc;
    vs.M = M;
    vs.K = K;
    vs.epoch = g_xe2mx_xh_epoch;
    vs.sa_row = 0;
    vs.qmode = qmode;
    vs.tick = ++g_xe2mx_s44_tick;
    static int nsilu = 0;
    if (nsilu < 6) {
        nsilu++;
        GGML_SYCL_XE2_LOG( "xe2mx s44 siluhilp M=%ld K=%ld\n", (long) M, (long) K);
        fflush(stderr);
    }
    return true;
}

void ggml_sycl_xe2mx_graph_begin() {
    if (g_xe2mx_s44_pack_n > 0) {
        static int npr = 0;
        if (npr < 4) {
            npr++;
            GGML_SYCL_XE2_LOG( "xe2mx s44 pack hit=%d / %d uniq=%zu slots=%d\n",
                    g_xe2mx_s44_pack_hit, g_xe2mx_s44_pack_n, g_xe2mx_s44_uniq.size(),
                    xe2mx_s44_nslot());
            fflush(stderr);
        }
        g_xe2mx_s44_pack_n = 0;
        g_xe2mx_s44_pack_hit = 0;
        g_xe2mx_s44_uniq.clear();
    }
    g_xe2mx_xh_epoch++;
    if (g_xe2mx_fn_nskip > 0 && ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_DEBUG", 0)) {
        static int nprn = 0;
        g_xe2mx_fn_graphs++;
        if (g_xe2mx_fn_landed != g_xe2mx_fn_nskip || nprn < 6) {
            nprn++;
            GGML_SYCL_XE2_LOG( "xe2mx fn graph %ld pairs=%d landed=%d\n", g_xe2mx_fn_graphs, g_xe2mx_fn_nskip,
                    g_xe2mx_fn_landed);
            fflush(stderr);
        }
    }
    g_xe2mx_fn_landed = 0;
    g_xe2mx_fn_nskip = 0;
    if (g_xe2mx_bt_st.seen > 0 && ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH_LOG", 0)) {
        static int nbt = 0;
        if (nbt < 8) {
            nbt++;
            const xe2mx_bt_stats & s = g_xe2mx_bt_st;
            GGML_SYCL_XE2_LOG( "xe2mx batch graph launches=%d members=%d seen=%d rej shape=%d fuse=%d src=%d dst=%d\n",
                    s.heads, s.members, s.seen, s.r_shape, s.r_fuse, s.r_src, s.r_dst);
            fflush(stderr);
        }
    }
    g_xe2mx_bt_st = {};
    g_xe2mx_bt_nskip = 0;
}

struct xe2mx_nk_acc {
    int64_t N;
    int64_t K;
    double gemm;
    double deq;
    int n;
};
static xe2mx_nk_acc g_xe2mx_nk[24];
static int g_xe2mx_nk_n = 0;

static void xe2mx_nk_add(int64_t N, int64_t K, double gemm, double deq) {
    for (int i = 0; i < g_xe2mx_nk_n; ++i) {
        if (g_xe2mx_nk[i].N == N && g_xe2mx_nk[i].K == K) {
            g_xe2mx_nk[i].gemm += gemm;
            g_xe2mx_nk[i].deq += deq;
            g_xe2mx_nk[i].n++;
            return;
        }
    }
    if (g_xe2mx_nk_n >= 24) {
        return;
    }
    g_xe2mx_nk[g_xe2mx_nk_n++] = {N, K, gemm, deq, 1};
}

static bool xe2mx_nf_xf_on(int64_t K, int64_t N) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_NOPACK_FFN", 0) && ggml_sycl_get_env("GGML_SYCL_XE2MX_NF_XF2", 1);
    return on && (N >= 16384 || K >= 16384);
}

static void xe2mx_time_dump() {
    GGML_SYCL_XE2_LOG( "xe2mx p0 sum deq=%.1f x=%.1f gemm=%.1f ms n=%d used=%zu\n",
            g_xe2mx_sum_deq, g_xe2mx_sum_x, g_xe2mx_sum_gemm, g_xe2mx_sum_n, g_xe2mx_wcache_used);
    for (int i = 0; i < g_xe2mx_nk_n; ++i) {
        const double ng = (double) g_xe2mx_nk[i].n;
        GGML_SYCL_XE2_LOG( "xe2mx p0 nk N=%ld K=%ld n=%d gemm=%.2f deq=%.2f tot=%.1f\n",
                (long) g_xe2mx_nk[i].N, (long) g_xe2mx_nk[i].K, g_xe2mx_nk[i].n,
                g_xe2mx_nk[i].gemm / ng, g_xe2mx_nk[i].deq / ng, g_xe2mx_nk[i].gemm);
    }
    fflush(stderr);
}

#if GGML_SYCL_DNNL
#include "oneapi/dnnl/dnnl_graph.hpp"
static int g_xe2mx_s4_dead = 0;
static int g_xe2mx_fadd_dead = 0;
static int g_xe2mx_s4acc_dead = 0;
static int g_xe2mx_s8u4_dead = 0;
static int g_xe2mx_s8r_dead = 0;
static int g_xe2mx_s4t_dead = 0;
static int g_xe2mx_s44_dead = 0;
static int g_xe2mx_s44ax_dead = 0;
static int g_xe2mx_s44any_dead = 0;
static int g_xe2mx_s44act_dead = 0;
static int g_xe2mx_s44as8_dead = 0;
static int g_xe2mx_s44f16a_dead = 0;
static sycl::half * g_xe2mx_f16a_ah = nullptr;
static float * g_xe2mx_f16a_sa = nullptr;
static float * g_xe2mx_f16a_sa2[2] = { nullptr, nullptr };
static uint8_t * g_xe2mx_f16a_ap2[2] = { nullptr, nullptr };
static size_t g_xe2mx_f16a_ah_cap = 0;
static size_t g_xe2mx_f16a_sa_cap = 0;
static size_t g_xe2mx_f16a_sa2_cap[2] = { 0, 0 };
static size_t g_xe2mx_f16a_ap2_cap[2] = { 0, 0 };
static int g_xe2mx_f16a_sel = 0;
static int8_t * g_xe2mx_s8ra = nullptr;
static float * g_xe2mx_s8rsa = nullptr;
static size_t g_xe2mx_s8rsa_cap = 0;
static int g_xe2mx_s8rsa_row = -1;
static const sycl::half * g_xe2mx_s8rxh = nullptr;
static const void * g_xe2mx_s8r_src = nullptr;
static size_t g_xe2mx_s8ra_cap = 0;
static int64_t g_xe2mx_s8rM = 0;
static int64_t g_xe2mx_s8rK = 0;
static uint64_t g_xe2mx_s8r_epoch = 0;
static uint8_t * g_xe2mx_s44a = nullptr;
static float * g_xe2mx_s44sa = nullptr;
static const sycl::half * g_xe2mx_s44xh = nullptr;
static const void * g_xe2mx_s44_src = nullptr;
static size_t g_xe2mx_s44a_cap = 0;
static size_t g_xe2mx_s44sa_cap = 0;
static int64_t g_xe2mx_s44M = 0;
static int64_t g_xe2mx_s44K = 0;
static uint64_t g_xe2mx_s44_epoch = 0;
static int g_xe2mx_s44sa_row = -1;
static int g_xe2mx_s44_qmode = -1;
static float * g_xe2mx_s44za = nullptr;
static size_t g_xe2mx_s44za_cap = 0;
struct xe2mx_acent {
    uint8_t * a4 = nullptr;
    float * sa = nullptr;
    size_t a4_cap = 0;
    size_t sa_cap = 0;
    int64_t M = 0;
    int64_t K = 0;
    int qmode = -1;
    uint64_t epoch = 0;
    const sycl::half * xh = nullptr;
};
static std::unordered_map<std::string, xe2mx_acent> g_xe2mx_amap;
static size_t g_xe2mx_amap_used = 0;
static int g_xe2mx_s8g_dead = 0;
static int g_xe2mx_s4f32_dead = 0;
static int g_xe2mx_s4f4_dead = 0;
static int g_xe2mx_s4fhs_dead = 0;
static int g_xe2mx_s4bf16_dead = 0;
static int g_xe2mx_s4tf32_dead = 0;
static int g_xe2mx_s4glu_dead = 0;
static int g_xe2mx_zsilu_dead = 0;
static int g_xe2mx_p0_zsilu = 0;
static int g_xe2mx_gluf16_dead = 0;
static int g_xe2mx_p0_gluf16 = 0;
static int g_xe2mx_p0_glu = 0;
static int g_xe2mx_s4gmlp_dead = 0;
static int8_t * g_xe2mx_zp = nullptr;
static size_t g_xe2mx_zp_cap = 0;

static void xe2mx_quant_a_one(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                                float * tmx, int64_t M, int64_t K);
static void xe2mx_quant_a_row(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                                int64_t M, int64_t K);
static void xe2mx_quant_a_s4(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                               float * tmx, int64_t M, int64_t K, int64_t g, float * za, int zp);
static bool xe2mx_s8r_jm2(queue_ptr stream, const int8_t * W, const float * scw,
                           const int8_t * A, const float * sa, float * C,
                           int64_t K, int64_t N, int64_t M);
static void xe2mx_quant_a(ggml_backend_sycl_context & ctx, const sycl::half * xh, int8_t * A, float * da,
                           int64_t M, int64_t K);
static void xe2mx_quant_a_f32(ggml_backend_sycl_context & ctx, const float * x, int8_t * A, float * da,
                                int64_t M, int64_t K);

static int8_t * xe2mx_zp8(queue_ptr stream, size_t n) {
    if (g_xe2mx_zp_cap < n) {
        const size_t cap = n < (4ull << 20) ? (4ull << 20) : n;
        void * p = ggml_sycl_malloc_device(cap, *stream);
        if (!p) {
            return nullptr;
        }
        stream->fill((int8_t *) p, (int8_t) 8, cap);
        g_xe2mx_zp = (int8_t *) p;
        g_xe2mx_zp_cap = cap;
    }
    return g_xe2mx_zp;
}

static bool xe2mx_p0_f4(ggml_backend_sycl_context & ctx, const uint8_t * packed,
                         const sycl::half * xh, float * C,
                         int64_t K, int64_t N, int64_t M, int time_on,
                         double * t_deq, double * t_gemm) {
    if (g_xe2mx_s4f4_dead || xh == nullptr || packed == nullptr || C == nullptr) {
        return false;
    }
    const int mode = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F4", 0);
    if (mode < 1 || mode > 3) {
        return false;
    }
    if ((K % 32) != 0) {
        return false;
    }
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    queue_ptr stream = ctx.stream();
    const bool use_e8 = (mode == 1 || mode == 3);
    const bool use_any = (mode == 3);
    ggml_sycl_pool_alloc<uint8_t> E8(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> Bro(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> Un(ctx.pool());
    const uint8_t * sc8 = nullptr;
    const size_t usz = (size_t) K * (size_t) N;
    const size_t npb = (size_t) N * qpitch;
    uint8_t * unp = nullptr;
    std::chrono::steady_clock::time_point t0;
    xe2mx_time_begin(time_on, stream, &t0);
    if (use_e8) {
        const size_t nsc = (size_t) nb * (size_t) N;
        uint8_t * dst = E8.alloc(nsc);
        const sycl::half * srcd = dpl;
        stream->parallel_for(sycl::range<1>(nsc), [=](sycl::id<1> gid) {
            const float f = (float) srcd[gid[0]];
            const uint32_t u = sycl::bit_cast<uint32_t>(f);
            dst[gid[0]] = (uint8_t) ((u >> 23) & 0xff);
        });
        sc8 = dst;
    }
    {
        auto wcit = g_xe2mx_wcache.find((const void *) packed);
        if (wcit != g_xe2mx_wcache.end() && wcit->second.vnni == 13
            && wcit->second.bytes == usz && wcit->second.K == K && wcit->second.N == N) {
            unp = (uint8_t *) wcit->second.packed;
        } else {
            void * persist = nullptr;
            if (!use_any && usz > (1ull << 20) && g_xe2mx_wcache_used + usz <= XE2MX_WCACHE_CAP
                && !g_xe2mx_wcache.count((const void *) packed)) {
                persist = ggml_sycl_malloc_device(usz, *stream);
                if (persist) {
                    unp = (uint8_t *) persist;
                }
            }
            if (!unp) {
                unp = Un.alloc(usz);
            }
            const uint8_t * qsrc = qs;
            stream->parallel_for(sycl::range<1>(npb), [=](sycl::id<1> gid) {
                const uint8_t b = qsrc[gid[0]];
                unp[gid[0] * 2u] = (uint8_t) (b & 0x0Fu);
                unp[gid[0] * 2u + 1u] = (uint8_t) (b >> 4);
            });
            if (persist) {
                g_xe2mx_wcache_used += usz;
                g_xe2mx_wcache[(const void *) packed] = { persist, usz, K, N, 13 };
            }
        }
    }
    *t_deq = xe2mx_time_ms(time_on, stream, t0);
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        const int fl = 0x400000 | (mode & 7);
        xe2mx_s4_key key{ (int) M, (int) N, (int) K, fl, (uintptr_t) stream };
        xe2mx_s4_prim * pe = nullptr;
        auto pit = g_xe2mx_s4_prims.find(key);
        if (pit == g_xe2mx_s4_prims.end()) {
            dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::f16, {K, 1});
            dnnl::memory::desc b_user({K, N}, dnnl::memory::data_type::f4_e2m1, dnnl::memory::format_tag::ba);
            dnnl::memory::desc b_in = use_any
                ? dnnl::memory::desc({K, N}, dnnl::memory::data_type::f4_e2m1, dnnl::memory::format_tag::any)
                : b_user;
            dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            if (use_e8) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::e8m0);
            } else {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f16);
            }
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_in, c_md, attr);
            if (time_on) {
                static int nimplf4 = 0;
                if (nimplf4 < 8) {
                    nimplf4++;
                    GGML_SYCL_XE2_LOG( "xe2mx f4 impl=%s mode=%d psz=%zu N=%ld M=%ld K=%ld\n",
                            pd.impl_info_str(), mode, pd.weights_desc().get_size(),
                            (long) N, (long) M, (long) K);
                }
            }
            xe2mx_s4_prim e;
            e.prim = dnnl::matmul(pd);
            e.a_md = a_md;
            e.b_md = pd.weights_desc();
            e.c_md = c_md;
            if (use_e8) {
                e.sc_md = dnnl::memory::desc({nb, N}, dnnl::memory::data_type::e8m0, {N, 1});
            } else {
                e.sc_md = dnnl::memory::desc({nb, N}, dnnl::memory::data_type::f16, {N, 1});
            }
            e.sp_md = pd.scratchpad_desc();
            pe = &g_xe2mx_s4_prims.insert({ key, std::move(e) }).first->second;
        } else {
            pe = &pit->second;
        }
        dnnl::memory::desc b_user({K, N}, dnnl::memory::data_type::f4_e2m1, dnnl::memory::format_tag::ba);
        void * bptr = unp;
        if (use_any && pe->b_md != b_user) {
            const size_t psz = pe->b_md.get_size();
            uint8_t * ro = nullptr;
            auto wcit = g_xe2mx_wcache.find((const void *) packed);
            if (wcit != g_xe2mx_wcache.end() && wcit->second.vnni == 12
                && wcit->second.bytes == psz && wcit->second.K == K && wcit->second.N == N) {
                ro = (uint8_t *) wcit->second.packed;
            } else {
                void * persist = nullptr;
                if (psz > (1ull << 20) && g_xe2mx_wcache_used + psz <= XE2MX_WCACHE_CAP
                    && !g_xe2mx_wcache.count((const void *) packed)) {
                    persist = ggml_sycl_malloc_device(psz, *stream);
                    if (persist) {
                        ro = (uint8_t *) persist;
                    }
                }
                if (!ro) {
                    ro = Bro.alloc(psz);
                }
                dnnl::memory b_src(b_user, eng, unp);
                dnnl::memory b_dst(pe->b_md, eng, ro);
                dnnl::reorder(b_src, b_dst).execute(strm, b_src, b_dst);
                if (persist) {
                    g_xe2mx_wcache_used += psz;
                    g_xe2mx_wcache[(const void *) packed] = { persist, psz, K, N, 12 };
                }
            }
            bptr = ro;
        }
        dnnl::memory a_mem(pe->a_md, eng, const_cast<sycl::half *>(xh));
        dnnl::memory b_mem(pe->b_md, eng, bptr);
        dnnl::memory c_mem(pe->c_md, eng, C);
        dnnl::memory sc_mem(pe->sc_md, eng,
            use_e8 ? (void *) const_cast<uint8_t *>(sc8) : (void *) const_cast<sycl::half *>(dpl));
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pe->sp_md, eng, stream);
        xe2mx_time_begin(time_on, stream, &t0);
        pe->prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_mem},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        });
        *t_gemm = xe2mx_time_ms(time_on, stream, t0);
        return true;
    } catch (const dnnl::error & e) {
        g_xe2mx_s4f4_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx f4 dnnl: %s\n", e.what());
            fflush(stderr);
        }
        return false;
    } catch (const sycl::exception & e) {
        g_xe2mx_s4f4_dead = 1;
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx f4 sycl: %s\n", e.what());
            fflush(stderr);
        }
        return false;
    }
}

static void xe2mx_silu_mul(queue_ptr stream, float * gate, const float * up, int64_t n) {
    stream->parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> gid) {
        const int64_t i = (int64_t) gid[0];
        const float g = gate[i];
        gate[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
    });
}

static int g_xe2mx_dumpa_n = 0;
static int g_xe2mx_dumpa_gemm = 0;

static void xe2mx_dump_ffn_a(ggml_backend_sycl_context & ctx, const sycl::half * xh,
                               const uint8_t * packed, int64_t K, int64_t N, int64_t M) {
    const int want = ggml_sycl_get_env("GGML_SYCL_XE2MX_DUMPA", 0);
    if (want < 1 || g_xe2mx_dumpa_n >= want) {
        return;
    }
    if (!xh || !packed || N != 17408 || K != 5120 || M < 32) {
        return;
    }
    if ((g_xe2mx_dumpa_gemm++ % 2) == 1) {
        return;
    }
    queue_ptr stream = ctx.stream();
    const size_t xsz = (size_t) M * (size_t) K;
    sycl::half * xhost = (sycl::half *) malloc(xsz * sizeof(sycl::half));
    if (!xhost) {
        return;
    }
    stream->memcpy(xhost, xh, xsz * sizeof(sycl::half));
    const int dumpw = ggml_sycl_get_env("GGML_SYCL_XE2MX_DUMPAW", 0);
    const int ncur = g_xe2mx_dumpa_n;
    const int wantw = dumpw == 1 || (dumpw == 2 && (ncur == 0 || ncur == 31 || ncur == 63));
    const size_t wsz = (size_t) N * (size_t) K;
    void * wd = nullptr;
    sycl::half * whost = nullptr;
    if (wantw) {
        wd = ggml_sycl_malloc_device(wsz * sizeof(sycl::half), *stream);
        if (!wd) {
            free(xhost);
            return;
        }
        xe2mx_deq_soa_f16(ctx, packed, (sycl::half *) wd, K, N);
        whost = (sycl::half *) malloc(wsz * sizeof(sycl::half));
        if (!whost) {
            free(xhost);
            ggml_sycl_free_device(wd, *stream);
            return;
        }
        stream->memcpy(whost, wd, wsz * sizeof(sycl::half));
    }
    stream->wait();
    const char * root = xe2mx_dump_root();
    if (!root) {
        return;
    }
    char path[512];
    if (wantw) {
        snprintf(path, sizeof(path), "%s/dumpa/xe2mx_ffn%d.bin", root, g_xe2mx_dumpa_n);
    } else {
        snprintf(path, sizeof(path), "%s/dumpa/xe2mx_x%d.bin", root, g_xe2mx_dumpa_n);
    }
    FILE * f = fopen(path, "wb");
    if (f) {
        const int64_t hdr[3] = { M, K, N };
        fwrite(hdr, sizeof(int64_t), 3, f);
        fwrite(xhost, sizeof(sycl::half), xsz, f);
        if (wantw && whost) {
            fwrite(whost, sizeof(sycl::half), wsz, f);
        }
        fclose(f);
        GGML_SYCL_XE2_LOG( "xe2mx dumpa %s M=%ld K=%ld N=%ld w=%d\n",
                path, (long) M, (long) K, (long) N, wantw ? dumpw : 0);
        fflush(stderr);
    }
    free(xhost);
    free(whost);
    if (wd) {
        ggml_sycl_free_device(wd, *stream);
    }
    g_xe2mx_dumpa_n++;
}

static bool xe2mx_p0_u4(ggml_backend_sycl_context & ctx, const uint8_t * packed,
                         const sycl::half * xh, const float * xf, float * C,
                         int64_t K, int64_t N, int64_t M, int time_on,
                         double * t_deq, double * t_gemm, const float * residual = nullptr,
                         const float * glu_up = nullptr, const void * asrc = nullptr,
                         const ggml_tensor * wt = nullptr) {
    auto lo_it = packed ? g_xe2mx_lo_w.find((const void *) packed) : g_xe2mx_lo_w.end();
    const int lo_w = lo_it != g_xe2mx_lo_w.end();
    if (lo_w && !lo_it->second.packed) {
        // Prompts too short to hide the upload stay on the original-bytes path.
        static const int min_m = ggml_sycl_get_env("GGML_SYCL_XE2MX_LO_HOSTPACK_MIN_M", 0);
        if (M < min_m) {
            return false;
        }
        xe2mx_lo_hostpack_ensure(*ctx.stream());
    }
    if (lo_w) {
        xe2mx_lo_hostpack_fence(*ctx.stream(), lo_it->second);
    }
    if (lo_w) {
        static int lo_p0 = 0;
        if (lo_p0 < 4) {
            lo_p0++;
            GGML_SYCL_XE2_LOG( "xe2mx: p0 in M=%ld N=%ld K=%ld xh=%d\n",
                    (long) M, (long) N, (long) K, xh ? 1 : 0);
            fflush(stderr);
        }
    }
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    queue_ptr stream = ctx.stream();
    int s4g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4G", 32);
    int s4fhs = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4FHS", 0) && !g_xe2mx_s4fhs_dead;
    int s4bf16 = !s4fhs && ggml_sycl_get_env("GGML_SYCL_XE2MX_S4BF16", 0) && !g_xe2mx_s4bf16_dead;
    if (s4fhs) {
        s4g = 64;
    }
    if (s4g != 64 && s4g != 128) {
        s4g = 32;
    }
    if ((K % s4g) != 0) {
        s4g = 32;
        s4fhs = 0;
    }
    const int64_t nbg = K / s4g;
    const size_t zpn = (size_t) nbg * (size_t) N;
    const int s8g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8G", 0) && !g_xe2mx_s8g_dead;
    const int s8u4_env = !s8g && ggml_sycl_get_env("GGML_SYCL_XE2MX_S8U4", 0) && !g_xe2mx_s8u4_dead;
    int s8u4 = s8u4_env;
    const int s4t0 = !s8g && !s8u4_env && ggml_sycl_get_env("GGML_SYCL_XE2MX_S4T", 0) && !g_xe2mx_s4t_dead
        && xh != nullptr && M >= 32 && N >= ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TN", 17408);
    int s44g = xe2mx_s44_g_kn(K, N, xe2mx_blk_layer(wt));
    const int s8rn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RN", 10240);
    const int s44dn = !s8g && !s8u4_env && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44DN", 0) && !g_xe2mx_s44_dead
        && xh != nullptr && M >= 32 && (K % s44g) == 0 && N < s8rn && K >= 16384 && N >= 4096;
    int s44layer = -1;
    if (!s44dn && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44", 0)
        && N >= ggml_sycl_get_env("GGML_SYCL_XE2MX_S44N", 10240)) {
        s44layer = xe2mx_blk_layer(wt);
        if (s44layer < 0) {
            s44layer = xe2mx_s48_bind(packed);
        }
    }
    const int s44from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FROM", -1);
    const int s44to = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44TO", 256);
    const int s44ffnfrom = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FFNFROM", -1);
    const int s44skip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44SKIP", -1);
    const int s44skip2 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44SKIP2", -1);
    int s44_pack_a = s44from >= 0 && s44layer >= s44from && s44layer < s44to;
    if (!s44_pack_a && s44ffnfrom >= 0 && s44layer >= s44ffnfrom && s44layer < s44to
        && (N >= 16384 || K >= 16384)) {
        s44_pack_a = 1;
    }
    if (s44_pack_a && s44skip >= 0 && s44layer == s44skip) {
        s44_pack_a = 0;
    }
    if (s44_pack_a && s44skip2 >= 0 && s44layer == s44skip2) {
        s44_pack_a = 0;
    }
    int s44a8 = !s44dn && !s44_pack_a
        && (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44A", 4) == 8 || s44g == 32);
    const int s44h = xe2mx_s44_h_on(s44layer, K, N);
    const int s44zp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0);
    const int s44wcen = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44WCEN", 0);
    if (!s44h) {
        const int gnh = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GNH", 0);
        if ((gnh == 32 || gnh == 64 || gnh == 128) && (K % (int64_t) gnh) == 0) {
            s44g = gnh;
        }
    }
    const int s44hrow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HROW", 0) && s44h && s44_pack_a;
    const int s44hocc = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HOCC", 0) && s44h && s44_pack_a && !s44hrow;
    int s44hwide = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HWIDE", 0);
    if (s44hwide != 2 && s44hwide != 4 && s44hwide != 8) {
        s44hwide = 0;
    }
    int s44hilp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44HILP", 0);
    if (s44hilp != 2 && s44hilp != 4) {
        s44hilp = 0;
    }
    int s8ra = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RA", 0);
    if (s44dn) {
        s8ra = 1;
    }
    if (s44_pack_a && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1) && !s44hrow) {
        s8ra = 1;
    }
    int s44pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44PN", 0);
    const int s44clip = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const int s44adiv = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const int s44arow = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AROW", 1);
    const int s44rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    const int s44qmode = xe2mx_s44_qmode(s8ra, s44h, s44zp, s44g, s44clip, s44adiv, s44arow, s44rms,
        s44_pack_a && s44pn, s44hrow, s44hocc, s44hwide, s44hilp);
    int s440 = !s8g && !s8u4_env && !g_xe2mx_s44_dead
        && (xh != nullptr || ((lo_w || (xe2mx_nf_xf_on(K, N) && s44_pack_a)) && xf))
        && (M >= 32 || lo_w) && (K % s44g) == 0
        && (s44dn || (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44", 0)
            && N >= ggml_sycl_get_env("GGML_SYCL_XE2MX_S44N", 10240))
            || lo_w);
    const int s8r0 = !s8g && !s8u4_env && ggml_sycl_get_env("GGML_SYCL_XE2MX_S8R", 1) && !g_xe2mx_s8r_dead
        && xh != nullptr && N >= s8rn;
    int s8r = 0;
    int s4t = 0;
    int s44 = 0;
    int s44b8 = 0;
    int s44ax = 0;
    int8_t * s44ax_ptr = nullptr;
    int s8pc = 0;
    int s48on = 0;
    int s48dn = 0;
    int s48layer = -1;
    const int s8rh = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RH", 0);
    xe2mx_s8r_view s8rv{};
    xe2mx_s8r_view s4tv{};
    xe2mx_s8r_view s44v{};
    xe2mx_s8r_view s8pcv{};
    int f16d = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F16", 0);
    if (s4fhs) {
        f16d = 1;
    }
    const int packx = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0);
    int s4s = packx || ggml_sycl_get_env("GGML_SYCL_XE2MX_S4S", 0);
    const int s4f32 = !s8g && xf && ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F32", 0) && !g_xe2mx_s4f32_dead;
    const bool fadd = residual != nullptr && glu_up == nullptr;
    const bool s4glu = glu_up != nullptr && !g_xe2mx_s4glu_dead;
    int s4acc_use = (!fadd && !g_xe2mx_s4acc_dead && (!s4glu || s8u4 || lo_w))
        ? ggml_sycl_get_env("GGML_SYCL_XE2MX_S4ACC", 0) : 0;
    int s44any = s440 && !g_xe2mx_s44any_dead && N >= 16384 && s44_pack_a
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ANY", 0);
    int s4any_use = (ggml_sycl_get_env("GGML_SYCL_XE2MX_S4ANY", 0)
        && !s8g && !s8u4 && !s8r0 && !s4t0 && !s440 && !s4s && !s4f32 && !fadd && !s4glu)
        || s44any;
    const int s4actp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4ACT", 0);
    const int s4act_use = s4actp >= 1
        && !s8g && !s8u4 && !s8r0 && !s4t0 && !s440 && !s4s && !s4f32 && !fadd && !s4glu && xh != nullptr;
    if (s8g || s8u4 || s8r0 || s4t0 || s440 || s4s || s4f32 || fadd || s4glu || s4any_use || s4act_use) {
        s4fhs = 0;
        s4bf16 = 0;
        f16d = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F16", 0);
    }
    if (fadd && (s8g || s8u4 || s8r0 || s4t0 || s4f32 || s4s || f16d || g_xe2mx_fadd_dead)) {
        return false;
    }
    if (s4glu && (s8g || s4f32 || s4s || f16d || fadd)) {
        return false;
    }
    if (!s8g && !s4f32 && xh == nullptr && !((lo_w || (xe2mx_nf_xf_on(K, N) && s44_pack_a)) && xf)) {
        return false;
    }
    if (s8g && xh == nullptr && xf == nullptr) {
        return false;
    }
    xe2mx_dump_ffn_a(ctx, xh, packed, K, N, M);
    ggml_sycl_pool_alloc<int8_t> zp_fb(ctx.pool());
    ggml_sycl_pool_alloc<int8_t> As(ctx.pool());
    ggml_sycl_pool_alloc<float> sar(ctx.pool());
    ggml_sycl_pool_alloc<float> tmx(ctx.pool());
    ggml_sycl_pool_alloc<float> zar(ctx.pool());
    ggml_sycl_pool_alloc<float> dag(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> Scg(ctx.pool());
    ggml_sycl_pool_alloc<sycl::ext::oneapi::bfloat16> Scb(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> Cf(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> Xhad(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> Xpad(ctx.pool());
    ggml_sycl_pool_alloc<float> Cpad(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> Qx(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> Bro(ctx.pool());
    ggml_sycl_pool_alloc<uint8_t> Aar(ctx.pool());
    const sycl::half * scp = dpl;
    float * scf = nullptr;
    sycl::ext::oneapi::bfloat16 * scb = nullptr;
    int8_t * Ap = nullptr;
    float * sap = nullptr;
    float * zap = nullptr;
    float * dap = nullptr;
    const uint8_t * bq = qs;
    int s44act = 0;
    int s44f16a = 0;
    int s44f16apack = 0;
    int s44as8 = 0;
    sycl::half * Ah = nullptr;
    double t_f16a_copy = 0;
    double t_f16a_had = 0;
    float * C_out = C;
    const int64_t M_in = M;
    std::chrono::steady_clock::time_point t0;
    xe2mx_time_begin(time_on, stream, &t0);
    if (lo_w) {
        static int lo_gemm_n = 0;
        if (lo_gemm_n < 4) {
            lo_gemm_n++;
            GGML_SYCL_XE2_LOG( "xe2mx: lo gemm #%d M=%ld N=%ld K=%ld\n",
                    lo_gemm_n, (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        s44 = 1;
        s44_pack_a = 1;
        s44a8 = 0;
        {
            const size_t w4b = (size_t) N * (size_t) (K / 2);
            const size_t sc_pn = (size_t) N * sizeof(float);
            const size_t sc_g = (size_t) (K / s44g) * (size_t) N * sizeof(float);
            const size_t lb = lo_it->second.bytes;
            s44v.w = (int8_t *) lo_it->second.packed;
            s44v.sc = (float *) ((char *) lo_it->second.packed + w4b);
            s44v.aux = nullptr;
            s44pn = 0;
            if (lb == w4b + sc_pn) {
                s44pn = 1;
            } else if (lb == w4b + sc_pn + sc_pn) {
                s44pn = 1;
                s44v.aux = (float *) ((char *) lo_it->second.packed + w4b + sc_pn);
            } else if (lb == w4b + sc_g + sc_g) {
                s44v.aux = (float *) ((char *) lo_it->second.packed + w4b + sc_g);
            }
        }
        s440 = 1;
        s44act = !g_xe2mx_s44act_dead && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ACT", 0);
        s44f16a = !g_xe2mx_s44f16a_dead
            && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44F16A", 0) && (xf || xh);
        if (s44f16a) {
            s44act = 0;
            s44f16apack = (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44F16APACK", 0)
                || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44F16AESIMD", 0)) ? 1 : 0;
        }
        s44as8 = !s44f16a && !g_xe2mx_s44as8_dead
            && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AS8", 0) && (xf || xh);
        if (s44as8) {
            s44act = 0;
        }
        s44any = !g_xe2mx_s44any_dead && (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ANY", 0)
            || ggml_sycl_get_env("GGML_SYCL_XE2MX_LOANY", 0));
        if (s44any) {
            s4any_use = 1;
        }
    } else if (s440) {
        s44 = xe2mx_s44_cached(stream, packed, K, N, &s44v, s44layer,
                               (s44_pack_a && s44pn) ? K : 0) ? 1 : 0;
        if (!s44) {
            s44any = 0;
            s4any_use = 0;
        } else {
            s44act = s44_pack_a && !s44a8 && !g_xe2mx_s44act_dead
                && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ACT", 0);
        }
    }
    const int gluf16 = g_xe2mx_p0_gluf16 && s44 && N >= 16384 && !g_xe2mx_gluf16_dead;
    if (g_xe2mx_p0_gluf16 && !gluf16) {
        return false;
    }
    if (s4t0 && !s44) {
        s4t = xe2mx_s4t_cached(stream, packed, K, N, &s4tv) ? 1 : 0;
    }
    if (!s4t && !s44 && ggml_sycl_get_env("GGML_SYCL_XE2MX_S48DN", 0)
        && N == 5120 && K == 17408 && xh != nullptr && M >= 32 && !fadd) {
        s48dn = xe2mx_s4dn_cached(stream, packed, K, N, &s8rv) ? 1 : 0;
    }
    if (s8r0 && !s4t && !s44) {
        const int s48 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48", 0);
        if (s48 && N == 17408 && K == 5120 && xh != nullptr && M >= 32 && !fadd) {
            s48layer = xe2mx_s48_bind(packed);
            const int from = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48FROM", 32);
            const int re = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48REARLY", 0);
            if (s48layer >= from && s48layer < 64) {
                s48on = xe2mx_s48_cached(ctx, stream, packed, K, N, &s8rv, s48layer) ? 1 : 0;
            } else if (re > 0 && s48layer >= 0 && s48layer < from) {
                s48on = xe2mx_s48_cached(ctx, stream, packed, K, N, &s8rv, s48layer) ? 1 : 0;
            }
        }
        if (!s48on && !s48dn) {
            s8r = xe2mx_s8r_cached(stream, packed, K, N, &s8rv) ? 1 : 0;
        }
    }
    if (!s8r && !s4t && !s44 && !s48on && !s48dn && xe2mx_s8dw_shape(K, N) && xh != nullptr && packed) {
        s8r = 1;
        s8rv.w = (int8_t *) packed;
        s8rv.sc = (float *) ((char *) packed + (size_t) K * (size_t) N);
    }
    if (s8u4) {
        s8pc = xe2mx_s8pc_cached(stream, packed, K, N, &s8pcv) ? 1 : 0;
        if (!s8pc) {
            s8u4 = 0;
            s4acc_use = 0;
        } else if (s4acc_use == 0) {
            s4acc_use = 1;
        }
    }
    int8_t * zpp = nullptr;
    if (!s4s && !s8r && !s44) {
        const size_t zneed = s8pc ? (size_t) N : zpn;
        zpp = xe2mx_zp8(stream, zneed);
        if (!zpp) {
            zpp = zp_fb.alloc(zneed);
            stream->fill(zpp, (int8_t) 8, zneed);
        }
    }
    if (s4s && !packx && !s8pc) {
        const size_t qn = (size_t) N * qpitch;
        auto wcit = g_xe2mx_wcache.find((const void *) packed);
        if (wcit != g_xe2mx_wcache.end() && wcit->second.vnni == 5
            && wcit->second.bytes == qn && wcit->second.K == K && wcit->second.N == N) {
            bq = (const uint8_t *) wcit->second.packed;
        } else {
            uint8_t * qdst = nullptr;
            void * persist = nullptr;
            if (qn > (1ull << 20) && g_xe2mx_wcache_used + qn <= XE2MX_WCACHE_CAP
                && !g_xe2mx_wcache.count((const void *) packed)) {
                persist = ggml_sycl_malloc_device(qn, *stream);
                if (persist) {
                    qdst = (uint8_t *) persist;
                }
            }
            if (!qdst) {
                qdst = Qx.alloc(qn);
            }
            const uint8_t * qsrc = qs;
            stream->parallel_for(sycl::range<1>(qn), [=](sycl::id<1> gid) {
                qdst[gid[0]] = (uint8_t) (qsrc[gid[0]] ^ 0x88u);
            });
            bq = qdst;
            if (persist) {
                g_xe2mx_wcache_used += qn;
                g_xe2mx_wcache[(const void *) packed] = { persist, qn, K, N, 5 };
            }
        }
    }
    if (s8g) {
        Ap = As.alloc((size_t) M * (size_t) K);
        dap = dag.alloc((size_t) M * (size_t) nb);
        if (xf) {
            xe2mx_quant_a_f32(ctx, xf, Ap, dap, M, K);
        } else {
            xe2mx_quant_a(ctx, xh, Ap, dap, M, K);
        }
    } else if (s44 && !s44a8) {
        if (s44f16a) {
            const size_t an8h = (size_t) M * (size_t) K;
            const size_t ahb = an8h * sizeof(sycl::half);
            if (ahb > g_xe2mx_f16a_ah_cap) {
                if (g_xe2mx_f16a_ah) {
                    ggml_sycl_free_device(g_xe2mx_f16a_ah, *stream);
                    g_xe2mx_f16a_ah = nullptr;
                    g_xe2mx_f16a_ah_cap = 0;
                }
                g_xe2mx_f16a_ah = (sycl::half *) ggml_sycl_malloc_device(ahb, *stream);
                if (!g_xe2mx_f16a_ah) {
                    return false;
                }
                g_xe2mx_f16a_ah_cap = ahb;
            }
            {
                const size_t ng = (size_t) (K / s44g);
                const size_t sab = (ng ? ((size_t) M * ng) : (size_t) M) * sizeof(float);
                if (s44f16apack) {
                    g_xe2mx_f16a_sel ^= 1;
                    const int sl = g_xe2mx_f16a_sel;
                    if (sab > g_xe2mx_f16a_sa2_cap[sl]) {
                        if (g_xe2mx_f16a_sa2[sl]) {
                            ggml_sycl_free_device(g_xe2mx_f16a_sa2[sl], *stream);
                            g_xe2mx_f16a_sa2[sl] = nullptr;
                            g_xe2mx_f16a_sa2_cap[sl] = 0;
                        }
                        g_xe2mx_f16a_sa2[sl] = (float *) ggml_sycl_malloc_device(sab, *stream);
                        if (!g_xe2mx_f16a_sa2[sl]) {
                            return false;
                        }
                        g_xe2mx_f16a_sa2_cap[sl] = sab;
                    }
                    sap = g_xe2mx_f16a_sa2[sl];
                } else if (sab > g_xe2mx_f16a_sa_cap) {
                    if (g_xe2mx_f16a_sa) {
                        ggml_sycl_free_device(g_xe2mx_f16a_sa, *stream);
                        g_xe2mx_f16a_sa = nullptr;
                        g_xe2mx_f16a_sa_cap = 0;
                    }
                    g_xe2mx_f16a_sa = (float *) ggml_sycl_malloc_device(sab, *stream);
                    if (!g_xe2mx_f16a_sa) {
                        return false;
                    }
                    g_xe2mx_f16a_sa_cap = sab;
                    stream->fill(g_xe2mx_f16a_sa, 1.f, sab / sizeof(float));
                    sap = g_xe2mx_f16a_sa;
                } else {
                    sap = g_xe2mx_f16a_sa;
                }
            }
            {
                static int nsplit = 0;
                const int split = nsplit < 8 && !s44f16apack;
                std::chrono::steady_clock::time_point tc0;
                if (split) {
                    stream->wait();
                    tc0 = std::chrono::steady_clock::now();
                }
                if (xf) {
                    Ah = g_xe2mx_f16a_ah;
                    xe2mx_f32_to_f16(stream, xf, Ah, (int64_t) an8h);
                } else {
                    Ah = g_xe2mx_f16a_ah;
                    stream->memcpy(Ah, xh, ahb);
                }
                if (split) {
                    stream->wait();
                    t_f16a_copy = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tc0).count();
                    tc0 = std::chrono::steady_clock::now();
                }
                xe2mx_had128_sa(stream, Ah, sap, M, K, (int64_t) s44g);
                if (s44f16apack) {
                    const size_t apb = (size_t) M * (size_t) (K / 2);
                    const int sl = g_xe2mx_f16a_sel;
                    if (apb > g_xe2mx_f16a_ap2_cap[sl]) {
                        if (g_xe2mx_f16a_ap2[sl]) {
                            ggml_sycl_free_device(g_xe2mx_f16a_ap2[sl], *stream);
                            g_xe2mx_f16a_ap2[sl] = nullptr;
                            g_xe2mx_f16a_ap2_cap[sl] = 0;
                        }
                        g_xe2mx_f16a_ap2[sl] = (uint8_t *) ggml_sycl_malloc_device(apb, *stream);
                        if (!g_xe2mx_f16a_ap2[sl]) {
                            return false;
                        }
                        g_xe2mx_f16a_ap2_cap[sl] = apb;
                    }
                    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S44F16AESIMD", 0)) {
                        if (!xe2mx_pack_hf_s4(stream, Ah, sap, g_xe2mx_f16a_ap2[sl], M, K, (int64_t) s44g)) {
                            return false;
                        }
                    } else if (!xe2mx_quant_a_s4_nibble(stream, Ah, g_xe2mx_f16a_ap2[sl], sap, M, K,
                            (int64_t) s44g, nullptr)) {
                        return false;
                    }
                    Ap = (int8_t *) g_xe2mx_f16a_ap2[sl];
                }
                if (split) {
                    stream->wait();
                    t_f16a_had = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tc0).count();
                    nsplit++;
                }
            }
            static int nf16a = 0;
            if (nf16a < 4) {
                nf16a++;
                GGML_SYCL_XE2_LOG( "xe2mx s44 f16a fuse pack=%d M=%ld N=%ld K=%ld g=%d ah=%p ap=%p\n",
                        s44f16apack, (long) M, (long) N, (long) K, s44g, (void *) Ah, (void *) Ap);
                fflush(stderr);
            }
        } else if (lo_w && !s4glu && !fadd && !s44act && !g_xe2mx_p0_glu && !g_xe2mx_splitm_dead
            && ggml_sycl_get_env("GGML_SYCL_XE2MX_LOSPLITM", 0)
            && M >= 1024 && K >= 16384 && s44v.w && s44v.sc && (xf || xh)) {
            std::chrono::steady_clock::time_point tsm;
            xe2mx_time_begin(time_on, stream, &tsm);
            if (xe2mx_s44_splitm(ctx, stream, xf, xh, (const uint8_t *) s44v.w, s44v.sc,
                                 C, M, N, K, (int64_t) s44g, s4acc_use)) {
                *t_gemm = xe2mx_time_ms(time_on, stream, tsm);
                static int nsmh = 0;
                if (nsmh < 8) {
                    nsmh++;
                    GGML_SYCL_XE2_LOG( "xe2mx splitm hit M=%ld N=%ld K=%ld g=%d tot=%.2f\n",
                            (long) M, (long) N, (long) K, s44g, *t_gemm);
                    fflush(stderr);
                }
                return true;
            }
        }
        if (!s44f16a) {
        if (lo_w && !s4glu && !fadd && !s44act && !g_xe2mx_p0_glu && !g_xe2mx_splitk_dead
            && ggml_sycl_get_env("GGML_SYCL_XE2MX_LOSPLITK", 0)
            && M >= 1024 && K >= 16384 && s44v.w && s44v.sc && (xf || xh)) {
            std::chrono::steady_clock::time_point tsk;
            xe2mx_time_begin(time_on, stream, &tsk);
            if (xe2mx_s44_splitk(ctx, stream, xf, xh, (const uint8_t *) s44v.w, s44v.sc,
                                 C, M, N, K, (int64_t) s44g, s4acc_use)) {
                *t_gemm = xe2mx_time_ms(time_on, stream, tsk);
                static int nskh = 0;
                if (nskh < 8) {
                    nskh++;
                    GGML_SYCL_XE2_LOG( "xe2mx splitk hit M=%ld N=%ld K=%ld g=%d tot=%.2f\n",
                            (long) M, (long) N, (long) K, s44g, *t_gemm);
                    fflush(stderr);
                }
                return true;
            }
        }
        const size_t an8 = (size_t) M * (size_t) K;
        const size_t an4 = (size_t) M * (size_t) (K / 2);
        const size_t apack = s44as8 ? an8 : an4;
        const size_t ng = (size_t) (K / s44g);
        const size_t sa_need = (s8ra || s44hrow) ? ((size_t) M * sizeof(float))
            : ((size_t) M * ng * sizeof(float));
        const size_t za_need = (!s8ra && (s44zp || (s44wcen && !s44h))) ? sa_need : 0;
        const int acache = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ACACHE", 0);
        const int astick = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ASTICK", 0);
        std::string akey;
        if ((acache || astick) && asrc) {
            const ggml_tensor * at = (const ggml_tensor *) asrc;
            char buf[320];
            std::snprintf(buf, sizeof(buf), "%d/%s/%ld/%ld/%d", ctx.device, at->name, (long) M, (long) K, s44qmode);
            akey.assign(buf);
        }
        if (asrc) {
            g_xe2mx_s44_uniq.insert(asrc);
        }
        int amap_hit = 0;
        if (!akey.empty()) {
            auto it = g_xe2mx_amap.find(akey);
            if (it != g_xe2mx_amap.end() && it->second.M == M && it->second.K == K
                && it->second.qmode == s44qmode && it->second.a4 && it->second.sa
                && it->second.a4_cap >= apack && it->second.sa_cap >= sa_need
                && (astick || (acache && it->second.epoch == g_xe2mx_xh_epoch
                    && it->second.xh == xh))) {
                Ap = (int8_t *) it->second.a4;
                sap = it->second.sa;
                amap_hit = 1;
            }
        }
        if (amap_hit) {
            g_xe2mx_s44_pack_hit++;
            g_xe2mx_s44_pack_n++;
        } else {
            const int hit = xe2mx_s44_slot_find(asrc, xh, xf, M, K, s8ra, s44qmode, apack, sa_need, za_need, ctx.device);
            if (hit >= 0) {
                xe2mx_s44_aslot & hs = g_xe2mx_s44_slot[hit];
                Ap = (int8_t *) hs.a4;
                sap = hs.sa;
                zap = za_need ? hs.za : nullptr;
                hs.tick = ++g_xe2mx_s44_tick;
                g_xe2mx_s44_pack_hit++;
                g_xe2mx_s44_pack_n++;
            } else {
            g_xe2mx_s44_pack_n++;
            if (g_xe2mx_s8ra_cap < an8) {
                void * p = ggml_sycl_malloc_device(an8, *stream);
                if (p) {
                    g_xe2mx_s8ra = (int8_t *) p;
                    g_xe2mx_s8ra_cap = an8;
                }
            }
            const int vi = xe2mx_s44_slot_victim();
            xe2mx_s44_aslot & vs = g_xe2mx_s44_slot[vi];
            // Buffers are device-bound: on a dual-GPU run the shared slot may
            // hold the OTHER card's allocation (capacity fits but the device
            // differs) - reuse would put cross-device USM into kernel args and
            // the L0 execute fails (UR 39). Reallocate on THIS stream instead;
            // the foreign buffer is leaked once per size, bounded.
            const int dev_id = ctx.device;
            if (vs.a4_cap < apack || vs.dev != dev_id) {
                void * p = ggml_sycl_malloc_device(apack, *stream);
                if (p) {
                    vs.a4 = (uint8_t *) p;
                    vs.a4_cap = apack;
                }
            }
            if (vs.sa_cap < sa_need || vs.dev != dev_id) {
                void * p = ggml_sycl_malloc_device(sa_need, *stream);
                if (p) {
                    vs.sa = (float *) p;
                    vs.sa_cap = sa_need;
                }
            }
            if (za_need && (vs.za_cap < za_need || vs.dev != dev_id)) {
                void * p = ggml_sycl_malloc_device(za_need, *stream);
                if (p) {
                    vs.za = (float *) p;
                    vs.za_cap = za_need;
                }
            }
            vs.dev = dev_id;
            int8_t * A8 = g_xe2mx_s8ra;
            uint8_t * A4 = vs.a4;
            sap = vs.sa;
            zap = za_need ? vs.za : nullptr;
            if (!akey.empty()) {
                const int acap_gb = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ACAP", 3);
                const size_t acap = (acap_gb < 1 || acap_gb > 16)
                    ? ((size_t) 3 * 1024 * 1024 * 1024)
                    : ((size_t) acap_gb * 1024 * 1024 * 1024);
                xe2mx_acent & e = g_xe2mx_amap[akey];
                if (e.a4_cap < apack && g_xe2mx_amap_used + apack <= acap) {
                    void * p = ggml_sycl_malloc_device(apack, *stream);
                    if (p) {
                        e.a4 = (uint8_t *) p;
                        e.a4_cap = apack;
                        g_xe2mx_amap_used += apack;
                    }
                }
                if (e.sa_cap < sa_need && g_xe2mx_amap_used + sa_need <= acap) {
                    void * p = ggml_sycl_malloc_device(sa_need, *stream);
                    if (p) {
                        e.sa = (float *) p;
                        e.sa_cap = sa_need;
                        g_xe2mx_amap_used += sa_need;
                    }
                }
                if (e.a4 && e.sa && e.a4_cap >= an4 && e.sa_cap >= sa_need) {
                    A4 = e.a4;
                    sap = e.sa;
                }
            }
            if (g_xe2mx_s44sa && !sap) {
                sap = g_xe2mx_s44sa;
            }
            if (za_need && g_xe2mx_s44za) {
                zap = g_xe2mx_s44za;
            }
            if (!A8) {
                A8 = As.alloc(an8);
            }
            if (!A4) {
                A4 = Qx.alloc(apack);
            }
            if (!sap) {
                sap = sar.alloc((s8ra || s44hrow) ? (size_t) M : ((size_t) M * ng));
            }
            if (za_need && !zap) {
                zap = zar.alloc((size_t) M * ng);
            }
            // Device LO + large n_ctx leaves little spare; pool/device alloc can return null.
            // Unchecked use here SEGVs mid-prefill after KV has grown.
            if (!A4 || !sap) {
                static int nnull = 0;
                if (nnull < 4) {
                    nnull++;
                    fprintf(stderr, "xe2mx s44 A pack OOM A4=%p sap=%p M=%ld K=%ld apack=%zu\n",
                            (void *) A4, (void *) sap, (long) M, (long) K, apack);
                    fflush(stderr);
                }
                return false;
            }
            const sycl::half * xq = xh;
            const int s44g_a = (s8ra || s44hrow) ? (int) K : s44g;
            const int fpack = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FPACK", 0)
                && !s44zp && !s8ra && !s44rms;
            int packed4 = 0;
            if (fpack) {
                if (s44as8 && s44h && (s44g_a % xe2mx_s44_hs()) == 0
                    && s44g_a >= xe2mx_s44_hs() && s44g_a <= 1024) {
                    if (xh) {
                        packed4 = xe2mx_quant_a_s8_had(stream, xh, A4, sap, M, K, (int64_t) s44g_a) ? 1 : 0;
                    } else if (xf) {
                        packed4 = xe2mx_quant_a_s8_had_f32_rte(stream, xf, A4, sap, M, K, (int64_t) s44g_a) ? 1 : 0;
                    }
                    static int nas8 = 0;
                    if (packed4 && nas8 < 4) {
                        nas8++;
                        GGML_SYCL_XE2_LOG( "xe2mx s44 as8 M=%ld N=%ld K=%ld g=%d\n",
                                (long) M, (long) N, (long) K, s44g_a);
                        fflush(stderr);
                    }
                } else if (s44hrow && (K % xe2mx_s44_hs()) == 0) {
                    if (xh) {
                        packed4 = xe2mx_quant_a_s4_nibble_had(stream, xh, A4, sap, M, K, K) ? 1 : 0;
                    } else if (xf) {
                        packed4 = xe2mx_quant_a_s4_nibble_had_f32_rte(stream, xf, A4, sap, M, K, K) ? 1 : 0;
                    }
                    static int nhrow = 0;
                    if (packed4 && nhrow < 4) {
                        nhrow++;
                        GGML_SYCL_XE2_LOG( "xe2mx s44 hrow M=%ld N=%ld K=%ld\n",
                                (long) M, (long) N, (long) K);
                        fflush(stderr);
                    }
                } else if (s44hocc && (K % xe2mx_s44_hs()) == 0
                    && (s44g_a % xe2mx_s44_hs()) == 0 && s44g_a >= xe2mx_s44_hs()) {
                    float * xf32 = tmx.alloc(an8);
                    const sycl::half * xs = xh;
                    stream->parallel_for(sycl::range<1>(an8), [=](sycl::id<1> gid) {
                        xf32[gid[0]] = (float) xs[gid[0]];
                    });
                    xe2mx_had128_f32(stream, xf32, M, K);
                    packed4 = xe2mx_quant_a_s4_nibble_f32(stream, xf32, A4, sap, M, K, (int64_t) s44g_a) ? 1 : 0;
                } else if (s44h && (s44g_a % xe2mx_s44_hs()) == 0 && s44g_a >= xe2mx_s44_hs() && s44g_a <= 1024) {
                    if (xh) {
                        packed4 = xe2mx_quant_a_s4_nibble_had(stream, xh, A4, sap, M, K, (int64_t) s44g_a) ? 1 : 0;
                    } else if (xf) {
                        packed4 = xe2mx_quant_a_s4_nibble_had_f32_rte(stream, xf, A4, sap, M, K, (int64_t) s44g_a) ? 1 : 0;
                    }
                } else if (!s44h) {
                    if (xh) {
                        packed4 = xe2mx_quant_a_s4_nibble(stream, xh, A4, sap, M, K, (int64_t) s44g_a, zap) ? 1 : 0;
                    } else if (xf) {
                        packed4 = xe2mx_quant_a_s4_nibble_f32(stream, xf, A4, sap, M, K, (int64_t) s44g_a, zap) ? 1 : 0;
                    }
                }
            }
            if (!packed4 && !xh && xf && !s44zp) {
                return false;
            }
            if (!packed4) {
            if (s44as8) {
                return false;
            }
            if (s44h && !s44zp && !s8ra && xh && (s44g_a % xe2mx_s44_hs()) == 0 && s44g_a >= xe2mx_s44_hs() && s44g_a <= 1024) {
                xe2mx_quant_a_s4_had(stream, xh, A8, sap, M, K, (int64_t) s44g_a);
            } else {
                if (!xq && xf) {
                    sycl::half * xw = Xhad.alloc(an8);
                    if (!xw) {
                        return false;
                    }
                    xe2mx_f32_to_f16(stream, xf, xw, (int64_t) an8);
                    xq = xw;
                }
                if (s44h) {
                    sycl::half * xw = nullptr;
                    if (xq && xq != xh) {
                        xw = (sycl::half *) xq;
                    } else if (xh) {
                        xw = Xhad.alloc(an8);
                        if (xw) {
                            const sycl::half * xs = xh;
                            stream->parallel_for(sycl::range<1>(an8), [=](sycl::id<1> gid) {
                                xw[gid[0]] = xs[gid[0]];
                            });
                        }
                    }
                    if (!xw) {
                        return false;
                    }
                    xe2mx_had128_f16(stream, xw, M, K);
                    xq = xw;
                }
                if (!xq) {
                    return false;
                }
                xe2mx_quant_a_s4(stream, xq, A8, sap, nullptr, M, K, (int64_t) s44g_a, zap, s44zp && !s8ra ? 1 : 0);
            }
            {
                const int8_t * src8 = A8;
                stream->parallel_for(sycl::range<1>(an4), [=](sycl::id<1> gid) {
                    const size_t t = (size_t) gid[0];
                    const size_t k2 = t % (size_t) (K / 2);
                    const size_t m = t / (size_t) (K / 2);
                    const uint8_t u0 = (uint8_t) src8[m * (size_t) K + k2 * 2u];
                    const uint8_t u1 = (uint8_t) src8[m * (size_t) K + k2 * 2u + 1u];
                    A4[t] = (uint8_t) ((u0 & 0x0Fu) | (uint8_t) ((u1 & 0x0Fu) << 4));
                });
            }
            }
            Ap = (int8_t *) A4;
            if (A4 == vs.a4) {
                vs.xh = xh;
                vs.xf = xf;
                vs.src = asrc;
                vs.M = M;
                vs.K = K;
                vs.epoch = g_xe2mx_xh_epoch;
                vs.sa_row = s8ra;
                vs.qmode = s44qmode;
                vs.tick = ++g_xe2mx_s44_tick;
            }
            if (!akey.empty()) {
                auto it = g_xe2mx_amap.find(akey);
                if (it != g_xe2mx_amap.end() && it->second.a4 == A4) {
                    it->second.M = M;
                    it->second.K = K;
                    it->second.qmode = s44qmode;
                    it->second.epoch = g_xe2mx_xh_epoch;
                    it->second.xh = xh;
                }
            }
            }
        }
        }
    }
    if (s48on && xh) {
        const int skipu = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48SKIPU", 0) || xe2mx_s48_fused();
        int rot_ok = 0;
        if (skipu) {
            rot_ok = 1;
        } else {
            const sycl::half * U = xe2mx_s48_U(stream, s48layer, K);
            const size_t xn = (size_t) M * (size_t) K;
            const int hit = U && g_xe2mx_s48_xrot && g_xe2mx_s48_xrot_xh == xh
                && g_xe2mx_s48_xrot_M == M && g_xe2mx_s48_xrot_K == K
                && g_xe2mx_s48_xrot_layer == s48layer
                && g_xe2mx_s48_xrot_epoch == g_xe2mx_xh_epoch
                && g_xe2mx_s48_xrot_cap >= xn * sizeof(sycl::half);
            if (hit) {
                xh = g_xe2mx_s48_xrot;
                rot_ok = 1;
            } else if (U) {
                if (g_xe2mx_s48_xrot_cap < xn * sizeof(sycl::half)) {
                    if (g_xe2mx_s48_xrot) {
                        ggml_sycl_free_device(g_xe2mx_s48_xrot, *stream);
                        g_xe2mx_s48_xrot = nullptr;
                        g_xe2mx_s48_xrot_cap = 0;
                    }
                    g_xe2mx_s48_xrot = (sycl::half *) ggml_sycl_malloc_device(xn * sizeof(sycl::half), *stream);
                    if (g_xe2mx_s48_xrot) {
                        g_xe2mx_s48_xrot_cap = xn * sizeof(sycl::half);
                    }
                }
                if (g_xe2mx_s48_xrot) {
                    const int s48ru = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48RU", 0);
                    int rok = 0;
                    if (s48ru) {
                        const int64_t rg = xe2mx_s48_g(K);
                        rok = xe2mx_s48_rot_s4(ctx, stream, xh, U, g_xe2mx_s48_xrot, M, K, s48layer, rg) ? 1 : 0;
                    }
                    if (!rok) {
                        rok = xe2mx_f16_mm(ctx, stream, xh, U, g_xe2mx_s48_xrot, M, K, K) ? 1 : 0;
                    }
                    if (rok) {
                    g_xe2mx_s48_xrot_xh = xh;
                    g_xe2mx_s48_xrot_M = M;
                    g_xe2mx_s48_xrot_K = K;
                    g_xe2mx_s48_xrot_layer = s48layer;
                    g_xe2mx_s48_xrot_epoch = g_xe2mx_xh_epoch;
                    xh = g_xe2mx_s48_xrot;
                    rot_ok = 1;
                    }
                }
            }
        }
        if (!rot_ok) {
            s48on = 0;
        }
    }
    if ((s48on || s48dn) && xh) {
        const int64_t r = s48dn ? 0 : xe2mx_s48_r(K, s48layer);
        const int64_t Kl = s48dn ? K : (K - r);
        const int64_t g = s48dn ? xe2mx_s48_dng(K) : xe2mx_s48_g(Kl);
        const size_t nb8 = (size_t) M * (size_t) r;
        const size_t nb4 = (size_t) M * (size_t) (Kl / 2);
        const size_t ns4 = (size_t) M * (size_t) (Kl / g);
        const int pn = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48PN", 0);
        const int tw = ggml_sycl_get_env("GGML_SYCL_XE2MX_S48TW", 0);
        const int row_a = (pn || tw || s48dn) ? 1 : ggml_sycl_get_env("GGML_SYCL_XE2MX_S48RA", 0);
        const size_t sa4n = row_a ? (size_t) M * sizeof(float) : ns4 * sizeof(float);
        int8_t * As8 = nullptr;
        uint8_t * As4 = nullptr;
        float * sas8 = nullptr;
        float * sas4 = nullptr;
        int ahit = g_xe2mx_s48_as4 && g_xe2mx_s48_sas4
            && (r == 0 || (g_xe2mx_s48_as8 && g_xe2mx_s48_sas8 && g_xe2mx_s48_as8_cap >= nb8))
            && g_xe2mx_s48_alayer == s48layer && g_xe2mx_s48_aM == M && g_xe2mx_s48_aK == K
            && g_xe2mx_s48_axh == xh
            && (!s48dn || g_xe2mx_s48_apacked == packed)
            && g_xe2mx_s48_ar == (int) r && g_xe2mx_s48_ag == (int) g && g_xe2mx_s48_a_epoch == g_xe2mx_xh_epoch
            && g_xe2mx_s48_as4_cap >= nb4
            && g_xe2mx_s48_sas4_cap >= sa4n;
        int s48ok = 0;
        if (ahit) {
            As8 = g_xe2mx_s48_as8;
            As4 = g_xe2mx_s48_as4;
            sas8 = g_xe2mx_s48_sas8;
            sas4 = g_xe2mx_s48_sas4;
            s48ok = 1;
        } else {
            if (nb8 > 0 && g_xe2mx_s48_as8_cap < nb8) {
                if (g_xe2mx_s48_as8) {
                    ggml_sycl_free_device(g_xe2mx_s48_as8, *stream);
                }
                g_xe2mx_s48_as8 = (int8_t *) ggml_sycl_malloc_device(nb8, *stream);
                g_xe2mx_s48_as8_cap = g_xe2mx_s48_as8 ? nb8 : 0;
            }
            if (g_xe2mx_s48_as4_cap < nb4) {
                if (g_xe2mx_s48_as4) {
                    ggml_sycl_free_device(g_xe2mx_s48_as4, *stream);
                }
                g_xe2mx_s48_as4 = (uint8_t *) ggml_sycl_malloc_device(nb4, *stream);
                g_xe2mx_s48_as4_cap = g_xe2mx_s48_as4 ? nb4 : 0;
            }
            if (!g_xe2mx_s48_sas8) {
                g_xe2mx_s48_sas8 = (float *) ggml_sycl_malloc_device(sizeof(float), *stream);
            }
            if (g_xe2mx_s48_sas4_cap < sa4n) {
                if (g_xe2mx_s48_sas4) {
                    ggml_sycl_free_device(g_xe2mx_s48_sas4, *stream);
                }
                g_xe2mx_s48_sas4 = (float *) ggml_sycl_malloc_device(sa4n, *stream);
                g_xe2mx_s48_sas4_cap = g_xe2mx_s48_sas4 ? sa4n : 0;
            }
            As8 = g_xe2mx_s48_as8;
            As4 = g_xe2mx_s48_as4;
            sas8 = g_xe2mx_s48_sas8;
            sas4 = g_xe2mx_s48_sas4;
            if (g_xe2mx_s48_qtmp_cap < (size_t) M * sizeof(float)) {
                if (g_xe2mx_s48_qtmp) {
                    ggml_sycl_free_device(g_xe2mx_s48_qtmp, *stream);
                }
                g_xe2mx_s48_qtmp = (float *) ggml_sycl_malloc_device((size_t) M * sizeof(float), *stream);
                g_xe2mx_s48_qtmp_cap = g_xe2mx_s48_qtmp ? (size_t) M * sizeof(float) : 0;
            }
            const size_t nlo = (size_t) M * (size_t) Kl;
            if (g_xe2mx_s48_a8lo_cap < nlo) {
                if (g_xe2mx_s48_a8lo) {
                    ggml_sycl_free_device(g_xe2mx_s48_a8lo, *stream);
                }
                g_xe2mx_s48_a8lo = (int8_t *) ggml_sycl_malloc_device(nlo, *stream);
                g_xe2mx_s48_a8lo_cap = g_xe2mx_s48_a8lo ? nlo : 0;
            }
            float * tmp = g_xe2mx_s48_qtmp;
            int8_t * A8lo = g_xe2mx_s48_a8lo;
            s48ok = As4 && sas4 && tmp && A8lo && (r == 0 || (As8 && sas8));
            if (s48ok) {
                xe2mx_s48_quant_a(stream, xh, As8, sas8, As4, sas4, A8lo, tmp, M, K, r, g, row_a);
                g_xe2mx_s48_axh = xh;
                g_xe2mx_s48_apacked = packed;
                g_xe2mx_s48_aM = M;
                g_xe2mx_s48_aK = K;
                g_xe2mx_s48_ar = (int) r;
                g_xe2mx_s48_ag = (int) g;
                g_xe2mx_s48_alayer = s48layer;
                g_xe2mx_s48_a_epoch = g_xe2mx_xh_epoch;
            }
        }
        if (s48ok) {
            int8_t * Ws8 = s8rv.w;
            float * sws8 = s8rv.sc;
            uint8_t * Ws4 = (r == 0) ? (uint8_t *) s8rv.w
                : (sws8 ? (uint8_t *) (sws8 + N) : nullptr);
            float * sws4 = s8rv.aux;
            *t_deq = xe2mx_time_ms(time_on, stream, t0);
            xe2mx_time_begin(time_on, stream, &t0);
            s48ok = Ws4 && sws4 && (r == 0 || (Ws8 && sws8))
                && xe2mx_s48_two(ctx, stream, As8, sas8, Ws8, sws8, As4, sas4, Ws4, sws4,
                                 C, M, N, r, Kl, g, s4glu ? glu_up : nullptr);
            if (s48ok) {
                *t_gemm = xe2mx_time_ms(time_on, stream, t0);
                static int ns48 = 0;
                if (time_on && ns48 < 80) {
                    ns48++;
                    GGML_SYCL_XE2_LOG( "xe2mx s48 split N=%ld M=%ld K=%ld r=%ld glu=%d ahit=%d deq=%.2f gemm=%.2f\n",
                            (long) N, (long) M, (long) K, (long) r, s4glu ? 1 : 0, ahit, *t_deq, *t_gemm);
                    fflush(stderr);
                }
                return true;
            }
        }
        s48on = 0;
        s48dn = 0;
    }
    if (s8u4 || (s8r && !s8rh) || s4t || (s44 && s44a8)) {
        const size_t an = (size_t) M * (size_t) K;
        const int s8rp = !s48on && !s48dn && ggml_sycl_get_env("GGML_SYCL_XE2MX_S8RP", 1);
        if (s8rp && !s8ra && asrc && asrc == g_xe2mx_s8r_src
            && (s8r || s4t || s8u4 || (s44 && s44a8)) && g_xe2mx_s8ra && g_xe2mx_s8rsa && g_xe2mx_s8rxh == xh
            && g_xe2mx_s8rM == M && g_xe2mx_s8rK == K && g_xe2mx_s8r_epoch == g_xe2mx_xh_epoch
            && g_xe2mx_s8ra_cap >= an && g_xe2mx_s8rsa_cap >= sizeof(float) && g_xe2mx_s8rsa_row == 0) {
            Ap = g_xe2mx_s8ra;
            sap = g_xe2mx_s8rsa;
        } else if (s8ra) {
            Ap = As.alloc(an);
            sap = sar.alloc((size_t) M);
            xe2mx_quant_a_row(stream, xh, Ap, sap, M, K);
        } else {
            if (s8rp && (s8r || s4t || s8u4 || (s44 && s44a8))) {
                if (g_xe2mx_s8ra_cap < an) {
                    void * p = ggml_sycl_malloc_device(an, *stream);
                    if (p) {
                        g_xe2mx_s8ra = (int8_t *) p;
                        g_xe2mx_s8ra_cap = an;
                    }
                }
                if (g_xe2mx_s8rsa_cap < sizeof(float)) {
                    void * p = ggml_sycl_malloc_device(sizeof(float), *stream);
                    if (p) {
                        g_xe2mx_s8rsa = (float *) p;
                        g_xe2mx_s8rsa_cap = sizeof(float);
                    }
                }
                if (g_xe2mx_s8ra && g_xe2mx_s8rsa && g_xe2mx_s8ra_cap >= an && g_xe2mx_s8rsa_cap >= sizeof(float)) {
                    Ap = g_xe2mx_s8ra;
                    sap = g_xe2mx_s8rsa;
                }
            }
            if (!Ap) {
                Ap = As.alloc(an);
            }
            if (!sap) {
                sap = sar.alloc(1);
            }
            float * tmp = tmx.alloc((size_t) M);
            xe2mx_quant_a_one(stream, xh, Ap, sap, tmp, M, K);
            if (s8rp && asrc && (s8r || s4t || s8u4 || (s44 && s44a8)) && Ap == g_xe2mx_s8ra) {
                g_xe2mx_s8rxh = xh;
                g_xe2mx_s8r_src = asrc;
                g_xe2mx_s8rM = M;
                g_xe2mx_s8rK = K;
                g_xe2mx_s8r_epoch = g_xe2mx_xh_epoch;
                g_xe2mx_s8rsa_row = 0;
            }
        }
    }
    if (s8r) {
        bq = (const uint8_t *) s8rv.w;
        scf = s8rv.sc;
    }
    if (s4t) {
        bq = (const uint8_t *) s4tv.w;
        scf = s4tv.sc;
    }
    if (s44) {
        bq = (const uint8_t *) s44v.w;
        scf = s44v.sc;
        if (bq && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44B8", 0)) {
            int8_t * b8 = xe2mx_s44_b8_expand(stream, bq, K, N);
            if (b8) {
                bq = (const uint8_t *) b8;
                s44b8 = 1;
                s4any_use = 0;
                static int nb8 = 0;
                if (nb8 < 2) {
                    nb8++;
                    GGML_SYCL_XE2_LOG( "xe2mx s44b8 expand N=%ld K=%ld\n", (long) N, (long) K);
                }
            }
        }
    }
    if (s8pc) {
        bq = (const uint8_t *) s8pcv.w;
        scf = s8pcv.sc;
    }
    if (s4fhs) {
        const int ratio = s4g / 32;
        scb = Scb.alloc((size_t) nbg * (size_t) N);
        const sycl::half * srcd = dpl;
        const int64_t Nn = N;
        stream->parallel_for(sycl::range<1>((size_t) nbg * (size_t) N), [=](sycl::id<1> gid) {
            const int64_t t = (int64_t) gid[0];
            const int64_t n = t % Nn;
            const int64_t ib = t / Nn;
            scb[t] = sycl::ext::oneapi::bfloat16((float) srcd[(ib * (int64_t) ratio) * Nn + n]);
        });
    } else if (s4bf16) {
        const size_t scn = (size_t) nbg * (size_t) N;
        const size_t scb_bytes = scn * sizeof(sycl::ext::oneapi::bfloat16);
        auto sit = g_xe2mx_scb.find((const void *) packed);
        if (sit != g_xe2mx_scb.end() && sit->second.bytes == scb_bytes
            && sit->second.K == K && sit->second.N == N) {
            scb = (sycl::ext::oneapi::bfloat16 *) sit->second.packed;
        } else {
            sycl::ext::oneapi::bfloat16 * dst = nullptr;
            void * persist = nullptr;
            if (scb_bytes > (1ull << 20) && g_xe2mx_wcache_used + scb_bytes <= XE2MX_WCACHE_CAP
                && !g_xe2mx_scb.count((const void *) packed)) {
                persist = ggml_sycl_malloc_device(scb_bytes, *stream);
                if (persist) {
                    dst = (sycl::ext::oneapi::bfloat16 *) persist;
                }
            }
            if (!dst) {
                dst = Scb.alloc(scn);
            }
            const sycl::half * srcd = dpl;
            stream->parallel_for(sycl::range<1>(scn), [=](sycl::id<1> gid) {
                dst[gid[0]] = sycl::ext::oneapi::bfloat16((float) srcd[gid[0]]);
            });
            scb = dst;
            if (persist) {
                g_xe2mx_wcache_used += scb_bytes;
                g_xe2mx_scb[(const void *) packed] = { persist, scb_bytes, K, N, 14 };
            }
        }
    } else if (s4g != 32) {
        const int ratio = s4g / 32;
        sycl::half * scd = Scg.alloc((size_t) nbg * (size_t) N);
        const sycl::half * srcd = dpl;
        const int64_t Nn = N;
        stream->parallel_for(sycl::range<1>((size_t) nbg * (size_t) N), [=](sycl::id<1> gid) {
            const int64_t t = (int64_t) gid[0];
            const int64_t n = t % Nn;
            const int64_t ib = t / Nn;
            scd[t] = srcd[(ib * (int64_t) ratio) * Nn + n];
        });
        scp = scd;
    }
    *t_deq = xe2mx_time_ms(time_on, stream, t0);
    if (s4t && !fadd && Ap && sap && scf && bq) {
        xe2mx_time_begin(time_on, stream, &t0);
        if (ggml_sycl_xe2mx_gemm_s4t(ctx, bq, scf, Ap, sap, C, K, N, M, s4glu ? glu_up : nullptr)) {
            *t_gemm = xe2mx_time_ms(time_on, stream, t0);
            static int ns4t = 0;
            if (time_on && ns4t < 8) {
                ns4t++;
                GGML_SYCL_XE2_LOG( "xe2mx s4t dpas N=%ld M=%ld K=%ld glu=%d\n",
                        (long) N, (long) M, (long) K, s4glu ? 1 : 0);
                fflush(stderr);
            }
            return true;
        }
        s4t = 0;
        bq = qs;
        scf = nullptr;
    }
    if (s8r && !s8rh && !s4glu && !fadd && Ap && sap && scf && bq
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_JM2", 0)) {
        xe2mx_time_begin(time_on, stream, &t0);
        if (xe2mx_s8r_jm2(stream, (const int8_t *) bq, scf, Ap, sap, C, K, N, M)) {
            *t_gemm = xe2mx_time_ms(time_on, stream, t0);
            static int njm2 = 0;
            if (njm2 < 8) {
                njm2++;
                GGML_SYCL_XE2_LOG( "xe2mx jm2 N=%ld M=%ld K=%ld\n", (long) N, (long) M, (long) K);
                fflush(stderr);
            }
            return true;
        }
    }
    if (lo_w && s44 && s44pn && s44_pack_a && !s44a8 && !s44as8 && !s44f16a && !s44hrow
        && !s44act && !fadd && Ap && sap && s44v.w && s44v.sc
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GRPACC", 0)) {
        xe2mx_time_begin(time_on, stream, &t0);
        const int64_t gs = s8ra ? (int64_t) 1024 : (int64_t) s44g;
        if (xe2mx_s44_grpacc(ctx, stream, (const uint8_t *) Ap, sap,
                (const uint8_t *) s44v.w, s44v.sc, C, M, N, K, gs, s4acc_use, s8ra, glu_up)) {
            *t_gemm = xe2mx_time_ms(time_on, stream, t0);
            static int ngrpacc = 0;
            if (ngrpacc < 8) {
                ngrpacc++;
                GGML_SYCL_XE2_LOG( "xe2mx grpacc M=%ld N=%ld K=%ld g=%ld arow=%d glu=%d gemm=%.2f\n",
                        (long) M, (long) N, (long) K, (long) gs, s8ra, glu_up ? 1 : 0, *t_gemm);
                fflush(stderr);
            }
            return true;
        }
    }
    if (lo_w && s44 && !s44a8 && !s44as8 && !s4glu && !fadd && !s44act && !g_xe2mx_p0_glu
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPAS", 0)
        && M >= ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPASN", 1024)
        && (ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPASDN", 0) == 0
            || (N == 5120 && K == 17408))
        && Ap && sap && s44v.w && s44v.sc) {
        xe2mx_time_begin(time_on, stream, &t0);
        if (ggml_sycl_xe2mx_lo_s44(ctx, (const uint8_t *) Ap, sap,
                (const uint8_t *) s44v.w, s44v.sc, C, K, N, M, (int64_t) s44g)) {
            *t_gemm = xe2mx_time_ms(time_on, stream, t0);
            static int nlo = 0;
            if (nlo < 4) {
                nlo++;
                GGML_SYCL_XE2_LOG( "xe2mx lodpas M=%ld N=%ld K=%ld g=%d gemm=%.2f\n",
                        (long) M, (long) N, (long) K, s44g, *t_gemm);
                fflush(stderr);
            }
            return true;
        }
    }
    if (lo_w && s44 && s44f16a && !s44f16apack && !s44a8 && !s44as8 && !s4glu && !fadd && !s44act
        && !g_xe2mx_p0_glu
        && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44F16AESIMDGEMM", 0)
        && M >= ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPASN", 1024)
        && Ah && sap && s44v.w && s44v.sc) {
        xe2mx_time_begin(time_on, stream, &t0);
        if (ggml_sycl_xe2mx_lo_hf_s44(ctx, Ah, sap,
                (const uint8_t *) s44v.w, s44v.sc, C, K, N, M, (int64_t) s44g)) {
            *t_gemm = xe2mx_time_ms(time_on, stream, t0);
            static int nlohf = 0;
            if (nlohf < 4) {
                nlohf++;
                GGML_SYCL_XE2_LOG( "xe2mx lohf M=%ld N=%ld K=%ld g=%d gemm=%.2f\n",
                        (long) M, (long) N, (long) K, s44g, *t_gemm);
                fflush(stderr);
            }
            return true;
        }
    }
    if (s44 && Ap && !s44a8 && !g_xe2mx_s44ax_dead
            && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44AX", 0)) {
        int8_t * ax = xe2mx_s44_b8_expand(stream, (const uint8_t *) Ap, K, M);
        if (ax) {
            s44ax = 1;
            s44ax_ptr = ax;
            s44act = 0;
            static int nax = 0;
            if (nax < 2) {
                nax++;
                GGML_SYCL_XE2_LOG( "xe2mx s44ax expand M=%ld K=%ld\n", (long) M, (long) K);
            }
        }
    }
    const int s44fp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FP", 0);
    const int s4fp0 = (s4fhs || s4bf16) ? 3 : (s44 && s44fp ? s44fp : ggml_sycl_get_env("GGML_SYCL_XE2MX_S4FP", 1));
    const int s4fp = (s4fp0 == 5 && g_xe2mx_s4tf32_dead) ? 1 : s4fp0;
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        const int fl = (s8u4 ? 1 : 0) | (f16d ? 2 : 0) | (s4s ? 4 : 0) | (s8g ? 8 : 0) | (s4f32 ? 16 : 0)
            | (gluf16 ? (1 << 5) : 0)
            | ((s4fp & 7) << 5) | (fadd ? (1 << 11) : 0)
            | ((s4acc_use & 7) << 12) | (s4any_use ? (1 << 15) : 0) | (s4act_use ? (1 << 10) : 0)
            | (s44act ? (1 << 14) : 0)
            | (((s44 ? (s44f16apack ? 7 : (s44f16a ? 5 : (s44as8 ? 6 : (s44hrow ? 2 : ((s44_pack_a && s44pn) ? 3 : s44g))))) : s4g) & 255) << 16) | (s4fhs ? (1 << 24) : 0) | (s4bf16 ? (1 << 25) : 0)
            | (s4glu ? (1 << 26) : 0) | (s8r ? (1 << 27) : 0) | ((s8r && s8rh) ? (1 << 28) : 0)
            | (s4t ? (1 << 29) : 0) | (s8pc ? (1 << 30) : 0) | (s44 ? (1 << 8) : 0) | (s44 ? (1 << 9) : 0) | (s44b8 ? (1 << 23) : 0) | (s44ax ? (1 << 22) : 0)
            | ((s44 && s44a8) ? (1 << 7) : 0) | ((s44 && !s44a8) ? (1 << 10) : 0) | (s8ra ? 4 : 0);
        xe2mx_s4_key key{ (int) M, (int) N, (int) K, fl, (uintptr_t) stream };
        xe2mx_s4_prim * pe = nullptr;
        const auto th0 = std::chrono::steady_clock::now();
        auto pit = g_xe2mx_s4_prims.find(key);
        const int pd_hit = (pit != g_xe2mx_s4_prims.end()) ? 1 : 0;
        if (pit == g_xe2mx_s4_prims.end()) {
            dnnl::memory::data_type adt = dnnl::memory::data_type::f16;
            if (s44 && s44f16apack) {
                adt = dnnl::memory::data_type::s4;
            } else if (s44 && s44f16a) {
                adt = dnnl::memory::data_type::f16;
            } else if (s44 && s44as8) {
                adt = dnnl::memory::data_type::s8;
            } else if (s44 && s44ax) {
                adt = dnnl::memory::data_type::s8;
            } else if (s44 && !s44a8) {
                adt = dnnl::memory::data_type::s4;
            } else if (s8g || s8u4 || (s8r && !s8rh) || s4t || (s44 && s44a8)) {
                adt = dnnl::memory::data_type::s8;
            } else if (s4f32) {
                adt = dnnl::memory::data_type::f32;
            }
            dnnl::memory::desc a_user({M, K}, adt, {K, 1});
            dnnl::memory::desc a_md = (s4act_use || s44act)
                ? dnnl::memory::desc({M, K}, adt, dnnl::memory::format_tag::any)
                : a_user;
            dnnl::memory::data_type bdt = dnnl::memory::data_type::u4;
            if (s8r || s44b8) {
                bdt = dnnl::memory::data_type::s8;
            } else if (s4s || s44) {
                bdt = dnnl::memory::data_type::s4;
            }
            dnnl::memory::desc b_user({K, N}, bdt, dnnl::memory::format_tag::ba);
            dnnl::memory::desc b_md = s4any_use
                ? dnnl::memory::desc({K, N}, bdt, dnnl::memory::format_tag::any)
                : b_user;
            dnnl::memory::desc c_md({M, N},
                (f16d || (gluf16 && !s4glu)) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32, {N, 1});
            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            if (s8g) {
                attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, 32}, dnnl::memory::data_type::f32);
            } else if (s8ra && (s8r || s4t || s8u4 || s44)) {
                attr.set_scales(DNNL_ARG_SRC, 1 << 0, {}, dnnl::memory::data_type::f32);
            } else if (s8u4 || s4t) {
                attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32);
            } else if (s44 && s44a8) {
                attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32);
            } else if (s44 && s44hrow) {
                attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, (int64_t) K},
                    dnnl::memory::data_type::f32);
            } else if (s44) {
                attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, (int64_t) s44g},
                    dnnl::memory::data_type::f32);
            } else if (s8r && !s8rh) {
                attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32);
            }
            // Per-column weight scale declared as per-channel (N mask only) so the kernel does not
            // multiply by the same column scale at every activation group boundary.
            static const int s44w1d = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44W1D", 0);
            if (s44 && s44_pack_a && s44pn && s44w1d) {
                attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 1, {}, dnnl::memory::data_type::f32);
            } else if (s44 && s44_pack_a && s44pn) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) K, 1},
                    dnnl::memory::data_type::f32);
            } else if (s44) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) s44g, 1},
                    dnnl::memory::data_type::f32);
            } else if (s8r) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) K, 1},
                    dnnl::memory::data_type::f32);
            }
            if (s4t || s8pc) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {(int64_t) K, 1},
                    dnnl::memory::data_type::f32);
            } else if (!s8r && !s44) {
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {s4g, 1},
                    (s4fhs || s4bf16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f16);
            }
            if (s8pc) {
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 1), {}, dnnl::memory::data_type::s8);
            } else if (!s4s && !s8r && !s44) {
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {s4g, 1}, dnnl::memory::data_type::s8);
            }
            if (!s8g && !s8u4 && !(s8r && !s8rh) && !s4t
                && (!s44 || ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FP", 0))) {
                if (s4fp == 1) {
                    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
                } else if (s4fp == 2) {
                    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, false);
                } else if (s4fp == 3) {
                    attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
                } else if (s4fp == 4) {
                    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
                } else if (s4fp == 5) {
                    attr.set_fpmath_mode(dnnl::fpmath_mode::tf32, true);
                }
            }
            if (s4acc_use == 1) {
                attr.set_accumulation_mode(dnnl::accumulation_mode::s32);
            } else if (s4acc_use == 2) {
                attr.set_accumulation_mode(dnnl::accumulation_mode::any);
            } else if (s4acc_use == 3) {
                attr.set_accumulation_mode(dnnl::accumulation_mode::relaxed);
            }
            dnnl::memory::desc bin_md;
            if (fadd) {
                bin_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, {N, 1});
                dnnl::post_ops po;
                po.append_binary(dnnl::algorithm::binary_add, bin_md);
                attr.set_post_ops(po);
            } else if (s4glu) {
                bin_md = dnnl::memory::desc({M, N},
                    gluf16 ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32, {N, 1});
                dnnl::post_ops po;
                po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 0.f);
                po.append_binary(dnnl::algorithm::binary_mul, bin_md);
                attr.set_post_ops(po);
            }
            auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            if (s4any_use && lo_w) {
                const size_t w4b = (size_t) N * (size_t) (K / 2);
                if (pd.weights_desc().get_size() > w4b
                    && !ggml_sycl_get_env("GGML_SYCL_XE2MX_LOANY", 0)) {
                    pd = dnnl::matmul::primitive_desc(eng, a_md, b_user, c_md, attr);
                    s4any_use = 0;
                    s44any = 0;
                }
            }
            if (time_on || s8r || s8u4 || s44) {
                static int nimpl = 0;
                if (nimpl < 32) {
                    nimpl++;
                    GGML_SYCL_XE2_LOG( "xe2mx s4 impl=%s fl=%d psz=%zu asz=%zu N=%ld M=%ld K=%ld g=%d s8r=%d s8u4=%d s44=%d s44g=%d a8=%d pack=%d pn=%d acc=%d any=%d act=%d\n",
                            pd.impl_info_str(), fl, pd.weights_desc().get_size(), pd.src_desc().get_size(),
                            (long) N, (long) M, (long) K, s4g, s8r, s8u4, s44, s44g, s44a8, s44_pack_a, s44pn, s4acc_use, s4any_use, s44act);
                    fflush(stderr);
                }
            }
            xe2mx_s4_prim e;
            e.prim = dnnl::matmul(pd);
            e.a_md = (s4act_use || s44act) ? pd.src_desc() : a_md;
            e.b_md = s4any_use ? pd.weights_desc() : b_user;
            e.c_md = c_md;
            if (s44 && s44_pack_a && s44pn && s44w1d) {
                e.sc_md = dnnl::memory::desc({N}, dnnl::memory::data_type::f32, {1});
            } else if (s44 && s44_pack_a && s44pn) {
                e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
            } else if (s44) {
                e.sc_md = dnnl::memory::desc({K / s44g, N}, dnnl::memory::data_type::f32, {N, 1});
            } else if (s8r || s4t || s8pc) {
                e.sc_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, {N, 1});
            } else {
                e.sc_md = dnnl::memory::desc({nbg, N},
                    (s4fhs || s4bf16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f16, {N, 1});
            }
            if (s8pc) {
                e.zp_md = dnnl::memory::desc({1, N}, dnnl::memory::data_type::s8, {N, 1});
            } else {
                e.zp_md = dnnl::memory::desc({nbg, N}, dnnl::memory::data_type::s8, {N, 1});
            }
            if (s8ra && (s8r || s4t || s8u4 || s44)) {
                e.sa_md = dnnl::memory::desc({M}, dnnl::memory::data_type::f32, {1});
            } else if (s44 && s44hrow) {
                e.sa_md = dnnl::memory::desc({M, 1}, dnnl::memory::data_type::f32, {1, 1});
            } else if (s44 && !s44a8) {
                e.sa_md = dnnl::memory::desc({M, K / s44g}, dnnl::memory::data_type::f32, {K / s44g, 1});
            } else if (s8g) {
                e.sa_md = dnnl::memory::desc({M, nb}, dnnl::memory::data_type::f32, {nb, 1});
            } else {
                e.sa_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32, {1});
            }
            e.sp_md = pd.scratchpad_desc();
            e.bin_md = bin_md;
            pe = &g_xe2mx_s4_prims.insert({ key, std::move(e) }).first->second;
        } else {
            pe = &pit->second;
        }
        sycl::half * cf = nullptr;
        void * cptr = C;
        if (f16d && !gluf16) {
            cf = Cf.alloc((size_t) M * (size_t) N);
            cptr = cf;
        }
        void * aptr = nullptr;
        if (s44f16apack && Ap) {
            aptr = Ap;
        } else if (s44f16a && Ah) {
            aptr = Ah;
        } else if (s8g || s8u4 || (s8r && !s8rh) || s4t || s44) {
            aptr = (s44ax && s44ax_ptr) ? (void *) s44ax_ptr : (void *) Ap;
        } else if (s4f32) {
            aptr = const_cast<float *>(xf);
        } else {
            aptr = const_cast<sycl::half *>(xh);
        }
        if (s44act && Ap) {
            dnnl::memory::desc a_user4({M, K}, dnnl::memory::data_type::s4, {K, 1});
            if (pe->a_md != a_user4) {
                const size_t psz = pe->a_md.get_size();
                uint8_t * ro = nullptr;
                if (g_xe2mx_ar_buf && g_xe2mx_ar_dev == ctx.device
                    && g_xe2mx_ar_xh == (const sycl::half *) Ap
                    && g_xe2mx_ar_M == M && g_xe2mx_ar_K == K && g_xe2mx_ar_bytes == psz
                    && g_xe2mx_ar_hit_epoch == g_xe2mx_xh_epoch) {
                    ro = g_xe2mx_ar_buf;
                } else {
                    if (g_xe2mx_ar_cap < psz || g_xe2mx_ar_dev != ctx.device) {
                        void * p = ggml_sycl_malloc_device(psz, *stream);
                        if (p) {
                            g_xe2mx_ar_buf = (uint8_t *) p;
                            g_xe2mx_ar_cap = psz;
                        }
                    }
                    if (g_xe2mx_ar_buf && g_xe2mx_ar_cap >= psz) {
                        ro = g_xe2mx_ar_buf;
                        g_xe2mx_ar_dev = ctx.device;
                    }
                    if (!ro) {
                        ro = Aar.alloc(psz);
                    }
                    dnnl::memory a_src(a_user4, eng, Ap);
                    dnnl::memory a_dst(pe->a_md, eng, ro);
                    dnnl::reorder(a_src, a_dst).execute(strm, a_src, a_dst);
                    if (ro == g_xe2mx_ar_buf) {
                        g_xe2mx_ar_xh = (const sycl::half *) Ap;
                        g_xe2mx_ar_M = M;
                        g_xe2mx_ar_K = K;
                        g_xe2mx_ar_bytes = psz;
                        g_xe2mx_ar_hit_epoch = g_xe2mx_xh_epoch;
                    }
                }
                aptr = ro;
            }
        } else if (s4act_use && xh) {
            dnnl::memory::desc a_user({M, K}, dnnl::memory::data_type::f16, {K, 1});
            if (pe->a_md != a_user) {
                const size_t psz = pe->a_md.get_size();
                uint8_t * ro = nullptr;
                if (s4actp >= 2 && g_xe2mx_ar_buf && g_xe2mx_ar_dev == ctx.device
                    && g_xe2mx_ar_xh == xh
                    && g_xe2mx_ar_M == M && g_xe2mx_ar_K == K && g_xe2mx_ar_bytes == psz
                    && g_xe2mx_ar_hit_epoch == g_xe2mx_xh_epoch) {
                    ro = g_xe2mx_ar_buf;
                } else {
                    if (s4actp >= 2) {
                        if (g_xe2mx_ar_cap < psz || g_xe2mx_ar_dev != ctx.device) {
                            void * p = ggml_sycl_malloc_device(psz, *stream);
                            if (p) {
                                g_xe2mx_ar_buf = (uint8_t *) p;
                                g_xe2mx_ar_cap = psz;
                            }
                        }
                        if (g_xe2mx_ar_buf && g_xe2mx_ar_cap >= psz) {
                            ro = g_xe2mx_ar_buf;
                            g_xe2mx_ar_dev = ctx.device;
                        }
                    }
                    if (!ro) {
                        ro = Aar.alloc(psz);
                    }
                    dnnl::memory a_src(a_user, eng, const_cast<sycl::half *>(xh));
                    dnnl::memory a_dst(pe->a_md, eng, ro);
                    dnnl::reorder(a_src, a_dst).execute(strm, a_src, a_dst);
                    if (s4actp >= 2 && ro == g_xe2mx_ar_buf) {
                        g_xe2mx_ar_xh = xh;
                        g_xe2mx_ar_M = M;
                        g_xe2mx_ar_K = K;
                        g_xe2mx_ar_bytes = psz;
                        g_xe2mx_ar_hit_epoch = g_xe2mx_xh_epoch;
                    }
                }
                aptr = ro;
            }
        }
        dnnl::memory a_mem(pe->a_md, eng, aptr);
        void * bptr = const_cast<uint8_t *>(bq);
        if (s4any_use) {
            const dnnl::memory::data_type bdt =
                (s4s || s44) ? dnnl::memory::data_type::s4 : dnnl::memory::data_type::u4;
            dnnl::memory::desc b_user({K, N}, bdt, dnnl::memory::format_tag::ba);
            if (pe->b_md != b_user) {
                const size_t psz = pe->b_md.get_size();
                uint8_t * ro = nullptr;
                const size_t w4b = (size_t) N * (size_t) (K / 2);
                if (lo_w) {
                    uint8_t * dst = (uint8_t *) lo_it->second.packed;
                    const size_t scb = (size_t) N * sizeof(float);
                    if (lo_it->second.vnni == 40 && lo_it->second.bytes >= psz + scb) {
                        ro = dst;
                    } else {
                        uint8_t * tmp = Bro.alloc(psz);
                        dnnl::memory b_src(b_user, eng, dst);
                        dnnl::memory b_dst(pe->b_md, eng, tmp);
                        dnnl::reorder(b_src, b_dst).execute(strm, b_src, b_dst);
                        void * np = dst;
                        if (psz > w4b) {
                            np = ggml_sycl_malloc_device(psz + scb, *stream);
                            if (np) {
                                stream->memcpy((char *) np + psz, (char *) dst + (lo_it->second.bytes - scb), scb);
                            } else {
                                np = tmp;
                            }
                        }
                        stream->memcpy(np, tmp, psz);
                        if (np != dst && np != tmp) {
                            stream->wait();
                            ggml_sycl_free_device(dst, *stream);
                            lo_it->second.packed = np;
                            lo_it->second.bytes = psz + scb;
                            s44v.w = (int8_t *) np;
                            s44v.sc = (float *) ((char *) np + psz);
                            scf = s44v.sc;
                            lo_it->second.vnni = 40;
                            xe2mx_lo_hostpack_store(*stream, lo_it->second, np, psz + scb, true);
                        } else if (np == dst) {
                            lo_it->second.vnni = 40;
                            xe2mx_lo_hostpack_store(*stream, lo_it->second, np, lo_it->second.bytes, true);
                        }
                        ro = (uint8_t *) np;
                    }
                } else {
                    auto * cmap = s44 ? &g_xe2mx_s44any : &g_xe2mx_wcache;
                    const int ctag = s44 ? 40 : 9;
                    auto wcit = cmap->find((const void *) packed);
                    if (wcit != cmap->end() && wcit->second.vnni == ctag
                        && wcit->second.bytes == psz && wcit->second.K == K && wcit->second.N == N) {
                        ro = (uint8_t *) wcit->second.packed;
                    } else {
                    void * persist = nullptr;
                    const int extra = N < 16384;
                    int room = !s44 && psz > (1ull << 20)
                        && cmap->find((const void *) packed) == cmap->end();
                    if (room && extra) {
                        room = g_xe2mx_s8r_ex_used + psz <= xe2mx_s8r_ex_cap();
                    } else if (room) {
                        room = g_xe2mx_wcache_used + psz <= XE2MX_WCACHE_CAP;
                    }
                    if (room) {
                        persist = ggml_sycl_malloc_device(psz, *stream);
                        if (persist) {
                            ro = (uint8_t *) persist;
                        }
                    }
                    if (!ro) {
                        ro = Bro.alloc(psz);
                    }
                    dnnl::memory b_src(b_user, eng, const_cast<uint8_t *>(bq));
                    dnnl::memory b_dst(pe->b_md, eng, ro);
                    dnnl::reorder(b_src, b_dst).execute(strm, b_src, b_dst);
                    if (persist) {
                        if (extra) {
                            g_xe2mx_s8r_ex_used += psz;
                        } else {
                            g_xe2mx_wcache_used += psz;
                        }
                        (*cmap)[(const void *) packed] = { persist, psz, K, N, ctag };
                    }
                    }
                }
                bptr = ro;
            }
        }
        dnnl::memory b_mem(pe->b_md, eng, bptr);
        dnnl::memory c_mem(pe->c_md, eng, cptr);
        void * scptr = nullptr;
        if (s4fhs || s4bf16) {
            scptr = scb;
        } else if (s8r || s4t || s8pc || s44) {
            scptr = scf;
        } else {
            scptr = const_cast<sycl::half *>(scp);
        }
        dnnl::memory sc_mem(pe->sc_md, eng, scptr);
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pe->sp_md, eng, stream);
        // get_scratchpad_mem may free-then-fail-realloc under device pressure; null handle SEGVs in execute.
        if (pe->sp_md.get_size() > 0 && sp_mem.get_data_handle() == nullptr) {
            static int nsp = 0;
            if (nsp < 4) {
                nsp++;
                fprintf(stderr, "xe2mx s4 scratch OOM psz=%zu M=%ld N=%ld K=%ld s44=%d\n",
                        pe->sp_md.get_size(), (long) M, (long) N, (long) K, s44);
                fflush(stderr);
            }
            return false;
        }
        if ((s44 || s8r || s4t || s8u4 || s8g) && (aptr == nullptr || (s44 && sap == nullptr && !s44f16a))) {
            return false;
        }
        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_mem},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        };
        if (!s4s && !s8r && !s44 && zpp) {
            dnnl::memory zp_mem(pe->zp_md, eng, zpp);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});
        }
        if (s8g) {
            dnnl::memory sa_mem(pe->sa_md, eng, dap);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem});
            } else if (s8u4 || (s8r && !s8rh) || s4t || s44) {
            dnnl::memory sa_mem(pe->sa_md, eng, sap);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem});
        }
        if (fadd) {
            dnnl::memory bin_mem(pe->bin_md, eng, const_cast<float *>(residual));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, bin_mem});
        } else if (s4glu) {
            dnnl::memory bin_mem(pe->bin_md, eng, (void *) glu_up);
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, bin_mem});
        }
        xe2mx_time_begin(time_on, stream, &t0);
        {
            static int nge = 0;
            std::chrono::steady_clock::time_point tg0;
            nge++;
            const auto th1 = std::chrono::steady_clock::now();
            const int split = s44f16a && !s44f16apack && (nge <= 16 || (nge % 50) == 0);
            if (split) {
                stream->wait();
                tg0 = std::chrono::steady_clock::now();
            }
            {
                // Dual prefill: the pointer that is identical on both cards (or
                // allocated on the other card) is the UR 39 at M>=1024.
                static int narg[GGML_SYCL_MAX_DEVICES] = {};
                const int d = ctx.device;
                if (M >= 512 && d >= 0 && d < GGML_SYCL_MAX_DEVICES && narg[d] < 6) {
                    narg[d]++;
                    auto onq = [&](const void * p) -> int {
                        if (!p) {
                            return -1;
                        }
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
                        try {
                            auto zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(stream->get_context());
                            auto zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(stream->get_device());
                            ze_memory_allocation_properties_t props = {};
                            props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
                            ze_device_handle_t mem_dev = nullptr;
                            if (zeMemGetAllocProperties(zc, p, &props, &mem_dev) != ZE_RESULT_SUCCESS) {
                                return -2;
                            }
                            if (props.type == ZE_MEMORY_TYPE_HOST) {
                                return -3;
                            }
                            if (props.type == ZE_MEMORY_TYPE_UNKNOWN) {
                                return -5;
                            }
                            return mem_dev == zd ? 1 : 0;
                        } catch (...) {
                            return -4;
                        }
#else
                        (void) p;
                        return -6;
#endif
                    };
                    GGML_SYCL_XE2_LOG(
                            "xe2mx s4args dev=%d sycldev=%d n=%d M=%ld N=%ld K=%ld "
                            "A=%p/%d B=%p/%d C=%p/%d SC=%p/%d SA=%p/%d SP=%p/%d "
                            "loa=%p s8ra=%p ar=%p xh=%p RES=%p/%d GLU=%p/%d name=%s\n",
                            d, ggml_sycl_get_device(), narg[d],
                            (long) M, (long) N, (long) K,
                            aptr, onq(aptr), bptr, onq(bptr), cptr, onq(cptr),
                            scptr, onq(scptr), (void *) sap, onq(sap),
                            sp_mem.get_data_handle(), onq(sp_mem.get_data_handle()),
                            (void *) g_xe2mx_loa, (void *) g_xe2mx_s8ra,
                            (void *) g_xe2mx_ar_buf, (void *) g_xe2mx_xh_buf,
                            (void *) residual, onq(residual), (void *) glu_up, onq(glu_up),
                            (wt && wt->name) ? wt->name : "?");
                    fflush(stderr);
                }
            }
            pe->prim.execute(strm, args);
            const auto th2 = std::chrono::steady_clock::now();
            if (s44f16a && !s44f16apack) {
                strm.wait();
            }
            if (split) {
                stream->wait();
                const double tgemm = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tg0).count();
                const double tpd = std::chrono::duration<double, std::milli>(th1 - th0).count();
                const double tex = std::chrono::duration<double, std::milli>(th2 - th1).count();
                GGML_SYCL_XE2_LOG( "xe2mx f16a split n=%d hit=%d copy=%.2f had=%.2f pd=%.2f exec=%.2f wait=%.2f M=%ld N=%ld K=%ld ah=%p\n",
                        nge, pd_hit, t_f16a_copy, t_f16a_had, tpd, tex, tgemm, (long) M, (long) N, (long) K, (void *) Ah);
                fflush(stderr);
            }
        }
        if (C != C_out) {
            stream->memcpy(C_out, C, (size_t) M_in * (size_t) N * sizeof(float));
        }
        if (f16d && cf && !gluf16) {
            const size_t nn = (size_t) M * (size_t) N;
            stream->parallel_for(sycl::range<1>(nn), [=](sycl::id<1> gid) {
                C[gid[0]] = (float) cf[gid[0]];
            });
        }
        if (s44 && ((s44zp && zap) || (s44wcen && !s44h && zap)) && !s8ra && s44v.aux && !gluf16 && (!s4glu || glu_up == nullptr)) {
            const int64_t ng_zp = s44pn ? (int64_t) 1 : (K / s44g);
            xe2mx_s44_zp_corr(stream, C, zap, s44v.aux, M, N, ng_zp);
            static int nwcen = 0;
            if (s44wcen && nwcen < 4) {
                nwcen++;
                GGML_SYCL_XE2_LOG( "xe2mx wcen corr M=%ld N=%ld K=%ld ng=%ld\n",
                        (long) M, (long) N, (long) K, (long) ng_zp);
                fflush(stderr);
            }
        }
        *t_gemm = xe2mx_time_ms(time_on, stream, t0);
        return true;
    } catch (const dnnl::error & e) {
        if (fadd) {
            g_xe2mx_fadd_dead = 1;
        } else if (s8u4) {
            g_xe2mx_s8u4_dead = 1;
        } else if (g_xe2mx_p0_gluf16) {
            g_xe2mx_gluf16_dead = 1;
        } else if (s4glu) {
            if (g_xe2mx_p0_zsilu) {
                g_xe2mx_zsilu_dead = 1;
            } else {
                g_xe2mx_s4glu_dead = 1;
            }
        } else if (s4fhs) {
            g_xe2mx_s4fhs_dead = 1;
        } else if (s4bf16) {
            g_xe2mx_s4bf16_dead = 1;
        } else if (s4fp == 5) {
            g_xe2mx_s4tf32_dead = 1;
        } else if (ctx.ovl_q) {
        } else if (s4acc_use) {
            g_xe2mx_s4acc_dead = 1;
        } else if (s8g) {
            g_xe2mx_s8g_dead = 1;
        } else if (s8r) {
            g_xe2mx_s8r_dead = 1;
        } else if (s44f16a) {
            g_xe2mx_s44f16a_dead = 1;
        } else if (s44as8) {
            g_xe2mx_s44as8_dead = 1;
        } else if (s44act) {
            g_xe2mx_s44act_dead = 1;
        } else if (s44any) {
            g_xe2mx_s44any_dead = 1;
        } else if (s44ax) {
            g_xe2mx_s44ax_dead = 1;
        } else if (s44) {
            g_xe2mx_s44_dead = 1;
        } else if (s4t) {
            g_xe2mx_s4t_dead = 1;
        } else if (s4f32) {
            g_xe2mx_s4f32_dead = 1;
        } else {
            g_xe2mx_s4_dead = 1;
        }
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx s4 dnnl: %s M=%d N=%d K=%d glu=%d fadd=%d s44=%d\n",
                    e.what(), (int) M, (int) N, (int) K, s4glu, fadd, s44);
            fflush(stderr);
        }
        return false;
    } catch (const sycl::exception & e) {
        if (fadd) {
            g_xe2mx_fadd_dead = 1;
        } else if (s8u4) {
            g_xe2mx_s8u4_dead = 1;
        } else if (g_xe2mx_p0_gluf16) {
            g_xe2mx_gluf16_dead = 1;
        } else if (s4glu) {
            if (g_xe2mx_p0_zsilu) {
                g_xe2mx_zsilu_dead = 1;
            } else {
                g_xe2mx_s4glu_dead = 1;
            }
        } else if (s4fhs) {
            g_xe2mx_s4fhs_dead = 1;
        } else if (s4bf16) {
            g_xe2mx_s4bf16_dead = 1;
        } else if (s4fp == 5) {
            g_xe2mx_s4tf32_dead = 1;
        } else if (ctx.ovl_q) {
        } else if (s4acc_use) {
            g_xe2mx_s4acc_dead = 1;
        } else if (s8g) {
            g_xe2mx_s8g_dead = 1;
        } else if (s8r) {
            g_xe2mx_s8r_dead = 1;
        } else if (s44f16a) {
            g_xe2mx_s44f16a_dead = 1;
        } else if (s44as8) {
            g_xe2mx_s44as8_dead = 1;
        } else if (s44act) {
            g_xe2mx_s44act_dead = 1;
        } else if (s44any) {
            g_xe2mx_s44any_dead = 1;
        } else if (s44ax) {
            g_xe2mx_s44ax_dead = 1;
        } else if (s44) {
            g_xe2mx_s44_dead = 1;
        } else if (s4t) {
            g_xe2mx_s4t_dead = 1;
        } else if (s4f32) {
            g_xe2mx_s4f32_dead = 1;
        } else {
            g_xe2mx_s4_dead = 1;
        }
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx s4 sycl: %s M=%d N=%d K=%d glu=%d fadd=%d s44=%d\n",
                    e.what(), (int) M, (int) N, (int) K, s4glu, fadd, s44);
            fflush(stderr);
        }
        return false;
    }
}
#endif

static bool xe2mx_p0_dnnl(ggml_backend_sycl_context & ctx,
                           const ggml_tensor * src0,
                           const ggml_tensor * src1,
                           ggml_tensor * dst,
                           int time_on,
                           bool packed,
                           const float * residual = nullptr,
                           bool allow_f16p = true,
                           const float * glu_up = nullptr) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(src1);
    if (residual && glu_up) {
        return false;
    }
    queue_ptr stream = ctx.stream();

    std::chrono::steady_clock::time_point t0;
    const int s4 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4", 1);
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);

    ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
    const sycl::half * xh = nullptr;
    const float * xf = nullptr;
    const int s8g = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8G", 0);
    const int s4f32 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F32", 0);
    const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
    xe2mx_time_begin(time_on, stream, &t0);
    if ((s8g || s4f32) && src1->type == GGML_TYPE_F32) {
        xf = (const float *) src1->data;
    } else if (src0->data && (g_xe2mx_lo_w.count((const void *) src0->data) || xe2mx_nf_xf_on(K, N))
               && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && M >= 16) {
        xf = (const float *) src1->data;
    } else if (src1->type == GGML_TYPE_F16) {
        xh = (const sycl::half *) src1->data;
    } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_dev == ctx.device
               && g_xe2mx_xh_src == src1 && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
               && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
        xh = g_xe2mx_xh_buf;
    } else {
        const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
        GGML_ASSERT(to_x);
        sycl::half * xdst = nullptr;
        if (xcache) {
            const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
            if (g_xe2mx_xh_cap < need || g_xe2mx_xh_dev != ctx.device) {
                void * p = ggml_sycl_malloc_device(need, *stream);
                if (p) {
                    g_xe2mx_xh_buf = (sycl::half *) p;
                    g_xe2mx_xh_cap = need;
                }
            }
            if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                xdst = g_xe2mx_xh_buf;
                g_xe2mx_xh_dev = ctx.device;
            }
        }
        if (!xdst) {
            X.alloc((size_t) M * (size_t) K);
            xdst = X.get();
        }
        if (src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1)) {
            xe2mx_f32_to_f16(stream, (const float *) src1->data, xdst, M * K);
        } else {
            to_x(src1->data, xdst, M * K, stream);
        }
        xh = xdst;
        if (xcache && xdst == g_xe2mx_xh_buf) {
            g_xe2mx_xh_src = src1;
            g_xe2mx_xh_M = M;
            g_xe2mx_xh_K = K;
            g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
            g_xe2mx_xh_dev = ctx.device;
        } else {
            g_xe2mx_xh_src = nullptr;
        }
    }
    const double t_x = xe2mx_time_ms(time_on, stream, t0);

#if GGML_SYCL_DNNL
    if (s4 && packed && N >= s4n && !g_xe2mx_s4_dead) {
        double t_deq4 = 0;
        double t_gemm4 = 0;
        auto s4ok = [&]() {
            if (time_on) {
                if (!g_xe2mx_sum_reg) {
                    g_xe2mx_sum_reg = 1;
                    atexit(xe2mx_time_dump);
                }
                g_xe2mx_sum_deq += t_deq4;
                g_xe2mx_sum_x += t_x;
                g_xe2mx_sum_gemm += t_gemm4;
                g_xe2mx_sum_n++;
                xe2mx_nk_add(N, K, t_gemm4, t_deq4);
                static int nlog4 = 0;
                if (nlog4 < 8) {
                    nlog4++;
                    GGML_SYCL_XE2_LOG( "xe2mx p0 deq=%.2f x=%.2f gemm=%.2f ms N=%ld M=%ld K=%ld packed=4 used=%zu\n",
                            t_deq4, t_x, t_gemm4, (long) N, (long) M, (long) K, g_xe2mx_wcache_used);
                }
            }
        };
        const int s4f4 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4F4", 0);
        if (s4f4 && !g_xe2mx_s4f4_dead && xh && !glu_up) {
            if (xe2mx_p0_f4(ctx, (const uint8_t *) src0->data, xh, (float *) dst->data,
                                K, N, M, time_on, &t_deq4, &t_gemm4)) {
                s4ok();
                return true;
            }
        }
        if (xe2mx_p0_u4(ctx, (const uint8_t *) src0->data, xh, xf, (float *) dst->data,
                           K, N, M, time_on, &t_deq4, &t_gemm4, residual, glu_up, src1, src0)) {
            s4ok();
            return true;
        }
        if (!g_xe2mx_s4_dead) {
            if (!xh) {
                if (src1->type == GGML_TYPE_F16) {
                    xh = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    X.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, X.get(), M * K, stream);
                    xh = X.get();
                }
            }
            if (s4f4 && !g_xe2mx_s4f4_dead && xh && !glu_up) {
                if (xe2mx_p0_f4(ctx, (const uint8_t *) src0->data, xh, (float *) dst->data,
                                    K, N, M, time_on, &t_deq4, &t_gemm4)) {
                    s4ok();
                    return true;
                }
            }
            if (xe2mx_p0_u4(ctx, (const uint8_t *) src0->data, xh, nullptr, (float *) dst->data,
                               K, N, M, time_on, &t_deq4, &t_gemm4, residual, glu_up, src1, src0)) {
                s4ok();
                return true;
            }
        }
    }
#endif
    if (residual || glu_up) {
        return false;
    }
    if (!allow_f16p) {
        return false;
    }
    // LO_INLINE: t->data is s4 pack, not block_xe2mx4 / XMX SOA - never dequant-as-SOA.
    if (src0->data && !ggml_sycl_xe2mx_lo_bytes_are_soa(src0)) {
        return false;
    }

    if (!xh) {
        if (src1->type == GGML_TYPE_F16) {
            xh = (const sycl::half *) src1->data;
        } else {
            const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
            GGML_ASSERT(to_x);
            X.alloc((size_t) M * (size_t) K);
            to_x(src1->data, X.get(), M * K, stream);
            xh = X.get();
        }
    }

    xe2mx_time_begin(time_on, stream, &t0);

    const sycl::half * Wp = nullptr;
    ggml_sycl_pool_alloc<sycl::half> W(ctx.pool());
    if (packed) {
        auto wcit = g_xe2mx_wcache.find((const void *) src0->data);
        if (wcit != g_xe2mx_wcache.end() && wcit->second.vnni == 2
            && wcit->second.K == K && wcit->second.N == N
            && wcit->second.bytes == (size_t) N * (size_t) K * sizeof(sycl::half)) {
            Wp = (const sycl::half *) wcit->second.packed;
        }
    }
    if (!Wp) {
        const size_t abn = (size_t) N * (size_t) K * sizeof(sycl::half);
        sycl::half * Wdst = nullptr;
        void * persist = nullptr;
        if (packed && N >= 1024 && abn > (1ull << 20) && g_xe2mx_wcache_used + abn <= XE2MX_WCACHE_CAP
            && !g_xe2mx_wcache.count((const void *) src0->data)) {
            persist = ggml_sycl_malloc_device(abn, *stream);
            if (persist) {
                Wdst = (sycl::half *) persist;
            }
        }
        if (!Wdst) {
            Wdst = W.alloc((size_t) N * (size_t) K);
        }
        if (packed) {
            xe2mx_deq_soa_f16(ctx, (const uint8_t *) src0->data, Wdst, K, N);
        } else {
            const to_fp16_sycl_t to_w = ggml_get_to_fp16_sycl(GGML_TYPE_XE2MX4, dst);
            GGML_ASSERT(to_w);
            to_w(src0->data, Wdst, N * K, stream);
        }
        Wp = Wdst;
        if (persist) {
            g_xe2mx_wcache_used += abn;
            g_xe2mx_wcache[(const void *) src0->data] = { persist, abn, K, N, 2 };
        }
    }
    const double t_deq = xe2mx_time_ms(time_on, stream, t0);

    xe2mx_time_begin(time_on, stream, &t0);
#if GGML_SYCL_DNNL
    if (packed) {
        DnnlGemmWrapper::gemm(ctx, (int) N, (int) M, (int) K,
                               Wp, DnnlGemmWrapper::to_dt<sycl::half>(),
                               (dnnl_dim_t) N, 1, (dnnl_dim_t) N * (dnnl_dim_t) K,
                               xh, DnnlGemmWrapper::to_dt<sycl::half>(),
                               1, (dnnl_dim_t) K, (dnnl_dim_t) M * (dnnl_dim_t) K,
                               dst->data, DnnlGemmWrapper::to_dt<float>(), stream, 1, 1);
    } else {
        DnnlGemmWrapper::row_gemm(ctx, (int) N, (int) M, (int) K,
                                  Wp, DnnlGemmWrapper::to_dt<sycl::half>(),
                                  xh, DnnlGemmWrapper::to_dt<sycl::half>(),
                                  dst->data, DnnlGemmWrapper::to_dt<float>(), stream);
    }
#else
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        (int) N, (int) M, (int) K, &alpha, Wp, dpct::library_data_t::real_half, (int) K,
        xh, dpct::library_data_t::real_half, (int) K, &beta,
        (float *) dst->data, dpct::library_data_t::real_float, (int) N,
        dpct::library_data_t::real_float)));
#endif
    const double t_gemm = xe2mx_time_ms(time_on, stream, t0);
    if (time_on) {
        if (!g_xe2mx_sum_reg) {
            g_xe2mx_sum_reg = 1;
            atexit(xe2mx_time_dump);
        }
        g_xe2mx_sum_deq += t_deq;
        g_xe2mx_sum_x += t_x;
        g_xe2mx_sum_gemm += t_gemm;
        g_xe2mx_sum_n++;
        xe2mx_nk_add(N, K, t_gemm, t_deq);
        static int nlog = 0;
        if (nlog < 8) {
            nlog++;
            GGML_SYCL_XE2_LOG( "xe2mx p0 deq=%.2f x=%.2f gemm=%.2f ms N=%ld M=%ld K=%ld packed=%d used=%zu\n",
                    t_deq, t_x, t_gemm, (long) N, (long) M, (long) K, packed ? 1 : 0, g_xe2mx_wcache_used);
        }
    }
    return true;
}


#if GGML_SYCL_DNNL

static void xe2mx_quant_a_one(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                                float * tmx, int64_t M, int64_t K) {
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            float amax = 0.0f;
            for (int k = lid; k < (int) K; k += (int) wg) {
                amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
            }
            tmx[m] = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        });
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            float a = 0.0f;
            for (int m = lid; m < (int) M; m += 256) {
                a = sycl::fmax(a, tmx[m]);
            }
            const float g = sycl::reduce_over_group(it.get_group(), a, sycl::maximum<float>());
            if (lid == 0) {
                sa[0] = g > 0.0f ? g / 127.0f : 0.0f;
            }
        });
    stream->parallel_for(sycl::range<1>((size_t) M * (size_t) K), [=](sycl::id<1> gid) {
        const float s = sa[0];
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        float xq = (float) xh[gid[0]] * id;
        int q = (int) sycl_rint_i(xq);
        if (q < -127) {
            q = -127;
        }
        if (q > 127) {
            q = 127;
        }
        A[gid[0]] = (int8_t) q;
    });
}

static void xe2mx_quant_a_row(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                                int64_t M, int64_t K) {
    const size_t wg = 128;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int m = (int) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            int8_t * arow = A + (size_t) m * (size_t) K;
            float amax = 0.0f;
            for (int k = lid; k < (int) K; k += (int) wg) {
                amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
            }
            const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            if (lid == 0) {
                sa[m] = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            }
            const float s = gmax > 0.0f ? gmax / 127.0f : 0.0f;
            const float id = s > 0.0f ? 1.0f / s : 0.0f;
            for (int k = lid; k < (int) K; k += (int) wg) {
                int q = (int) sycl_rint_i((float) row[k] * id);
                if (q < -127) {
                    q = -127;
                }
                if (q > 127) {
                    q = 127;
                }
                arow[k] = (int8_t) q;
            }
        });
}

static void xe2mx_quant_a_s4(queue_ptr stream, const sycl::half * xh, int8_t * A, float * sa,
                               float * tmx, int64_t M, int64_t K, int64_t g, float * za, int zp) {
    (void) tmx;
    const int64_t ng = K / g;
    const size_t wg = 128;
    const int clipp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44CLIP", 100);
    const float clip = (clipp > 0 && clipp <= 100) ? (float) clipp / 100.0f : 1.0f;
    const int adivp = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ADIV", 7);
    const float adiv = (adivp >= 4 && adivp <= 8) ? (float) adivp : 7.0f;
    const int rms = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44RMS", 0);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) M * (size_t) ng * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
            const int gid = (int) it.get_group(0);
            const int m = gid / (int) ng;
            const int ig = gid - m * (int) ng;
            const int lid = (int) it.get_local_id(0);
            const sycl::half * row = xh + (size_t) m * (size_t) K;
            int8_t * arow = A + (size_t) m * (size_t) K;
            const int k0 = ig * (int) g;
            const int k1 = k0 + (int) g;
            if (zp) {
                float mn = 1.0e30f;
                float mx = -1.0e30f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    const float v = (float) row[k];
                    mn = sycl::fmin(mn, v);
                    mx = sycl::fmax(mx, v);
                }
                mn = sycl::reduce_over_group(it.get_group(), mn, sycl::minimum<float>());
                mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
                const float mid = 0.5f * (mn + mx);
                const float half = 0.5f * (mx - mn);
                const float s = half > 0.0f ? half / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = s;
                    if (za) {
                        za[(size_t) m * (size_t) ng + (size_t) ig] = mid;
                    }
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    int q = (int) sycl_rint_i(((float) row[k] - mid) * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k] = (int8_t) q;
                }
            } else if (rms) {
                float ss = 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    const float v = (float) row[k];
                    ss += v * v;
                }
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                const float rmsv = sycl::sqrt(ss / (float) g);
                const float s = rmsv > 0.0f ? (rmsv * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = s;
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    int q = (int) sycl_rint_i((float) row[k] * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k] = (int8_t) q;
                }
            } else {
                float amax = 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    amax = sycl::fmax(amax, sycl::fabs((float) row[k]));
                }
                const float gmax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
                const float s = gmax > 0.0f ? (gmax * clip) / adiv : 0.0f;
                if (lid == 0) {
                    sa[(size_t) m * (size_t) ng + (size_t) ig] = s;
                }
                const float id = s > 0.0f ? 1.0f / s : 0.0f;
                for (int k = k0 + lid; k < k1; k += (int) wg) {
                    int q = (int) sycl_rint_i((float) row[k] * id);
                    if (q < -8) {
                        q = -8;
                    }
                    if (q > 7) {
                        q = 7;
                    }
                    arow[k] = (int8_t) q;
                }
            }
        });
}

static void xe2mx_pack_vnni(queue_ptr stream, const int8_t * W, int8_t * P, int64_t K, int64_t N) {
    const int k4n = (int) (K / 4);
    const int n16 = (int) (N / 16);
    stream->parallel_for(sycl::range<2>((size_t) k4n, (size_t) n16), [=](sycl::id<2> gid) {
        const int k4 = (int) gid[0];
        const int n0 = (int) gid[1] * 16;
        const int8_t * s0 = W + (size_t) (k4 * 4) * (size_t) N + (size_t) n0;
        int8_t * d = P + (size_t) k4 * (size_t) N * 4u + (size_t) n0 * 4u;
        for (int i = 0; i < 16; ++i) {
            d[i * 4 + 0] = s0[0 * N + i];
            d[i * 4 + 1] = s0[1 * N + i];
            d[i * 4 + 2] = s0[2 * N + i];
            d[i * 4 + 3] = s0[3 * N + i];
        }
    });
}

template <int NT>
static bool xe2mx_gemm_jm_nt(queue_ptr stream, const int8_t * P, const float * scp,
                              const int8_t * A, const float * da, float * C,
                              int64_t K, int64_t N, int64_t M) {
    constexpr int TM = 8;
    constexpr int TN = 16;
    constexpr int TK = 32;
    constexpr int SG = 16;
    constexpr int PF = 4;
    const int nstep = TN * NT;
    if ((M % TM) != 0 || (N % nstep) != 0 || (K % TK) != 0 || M < 1) {
        return false;
    }
    const int nb = (int) (K / TK);
    const int m_tiles = (int) (M / TM);
    const int n_tiles = (int) (N / nstep);
    const auto props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>
    };
    try {
        stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) m_tiles * (size_t) SG, (size_t) n_tiles),
                              sycl::range<2>((size_t) SG, 1)),
            props,
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                using namespace sycl::ext::oneapi::experimental::matrix;
                auto sg = it.get_sub_group();
                const int m0 = (int) it.get_group(0) * TM;
                const int n0 = (int) it.get_group(1) * nstep;
                float facc[NT][8];
                int wr[NT][8];
                int wc[NT][8];
                int nel = 0;
                int first = 1;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        facc[t][i] = 0.0f;
                    }
                }
                for (int ib = 0; ib < nb; ++ib) {
                    const int k0 = ib * TK;
                    joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK, layout::row_major> afrag;
                    auto pa = sycl::address_space_cast<sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(const_cast<int8_t *>(A + (size_t) m0 * (size_t) K + (size_t) k0));
                    joint_matrix_load(sg, afrag, pa, (size_t) K);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN, layout::ext_intel_packed> bfrag;
                        joint_matrix<sycl::sub_group, int, use::accumulator, TM, TN> acc;
                        joint_matrix_fill(sg, acc, 0);
                        const int nt = n0 + t * TN;
                        auto pb = sycl::address_space_cast<sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(
                                const_cast<int8_t *>(P + (size_t) (k0 / PF) * (size_t) N * (size_t) PF + (size_t) nt * (size_t) PF));
                        joint_matrix_load(sg, bfrag, pb, (size_t) N * (size_t) PF);
                        joint_matrix_mad(sg, acc, afrag, bfrag, acc);
                        int i = 0;
                        sycl::ext::intel::experimental::matrix::joint_matrix_apply(
                            sg, acc, [&](auto & x, size_t row, size_t col) {
                                const float dw = scp[(size_t) ib * (size_t) N + (size_t) nt + col];
                                const float dar = da[(size_t) (m0 + (int) row) * (size_t) nb + (size_t) ib];
                                const float v = (float) (int) x * dw * dar;
                                if (first) {
                                    wr[t][i] = (int) row;
                                    wc[t][i] = (int) col;
                                    facc[t][i] = v;
                                } else {
                                    facc[t][i] += v;
                                }
                                i++;
                            });
                        if (first) {
                            nel = i;
                        }
                    }
                    first = 0;
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int nt = n0 + t * TN;
                    for (int i = 0; i < nel; ++i) {
                        C[(size_t) (m0 + wr[t][i]) * (size_t) N + (size_t) nt + (size_t) wc[t][i]] = facc[t][i];
                    }
                }
            });
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx jm: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

static bool xe2mx_gemm_jm(queue_ptr stream, const int8_t * P, const float * scp,
                             const int8_t * A, const float * da, float * C,
                             int64_t K, int64_t N, int64_t M) {
    if ((N % 64) == 0) {
        return xe2mx_gemm_jm_nt<4>(stream, P, scp, A, da, C, K, N, M);
    }
    return xe2mx_gemm_jm_nt<1>(stream, P, scp, A, da, C, K, N, M);
}

static int g_xe2mx_jm2_dead = 0;

static bool xe2mx_s8r_jm2(queue_ptr stream, const int8_t * W, const float * scw,
                           const int8_t * A, const float * sa, float * C,
                           int64_t K, int64_t N, int64_t M) {
    (void) g_xe2mx_jm2_dead;
    GGML_UNUSED(stream);
    GGML_UNUSED(W);
    GGML_UNUSED(scw);
    GGML_UNUSED(A);
    GGML_UNUSED(sa);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
}

static bool xe2mx_dnnl_woq(ggml_backend_sycl_context & ctx,
                            const uint8_t * packed,
                            const sycl::half * xh,
                            const int8_t * As8,
                            const float * da,
                            float * C,
                            int64_t K,
                            int64_t N,
                            int64_t M,
                            int time_on,
                            double * t_e,
                            double * t_k) {
    if (xh == nullptr || packed == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || N < 1) {
        return false;
    }
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const uint8_t * qs = packed;
    const sycl::half * dpl = (const sycl::half *) (packed + (size_t) N * qpitch);
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<float> sc(ctx.pool());
    float * scp = sc.alloc((size_t) nb * (size_t) N);
    const int u4_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_U4", 0);
    std::chrono::steady_clock::time_point t0;
    xe2mx_time_begin(time_on, stream, &t0);
    stream->parallel_for(sycl::range<1>((size_t) N * (size_t) nb), [=](sycl::id<1> gid) {
        const size_t t = (size_t) gid[0];
        scp[t] = (float) dpl[t];
    });
    if (u4_on) {
        ggml_sycl_pool_alloc<int32_t> zp(ctx.pool());
        int32_t * zpp = zp.alloc(1);
        stream->fill(zpp, (int32_t) 8, 1);
        try {
            dnnl::engine eng = ctx.engine_dnnl(stream);
            dnnl::stream strm = ctx.stream_dnnl(stream);
            dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::f16, {K, 1});
            dnnl::memory::desc b4_md({K, N}, dnnl::memory::data_type::u4, {1, K});
            dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
            dnnl::primitive_attr attr4;
            attr4.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            attr4.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f32);
            attr4.set_zero_points_mask(DNNL_ARG_WEIGHTS, 0);
            attr4.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto pd4 = dnnl::matmul::primitive_desc(eng, a_md, b4_md, c_md, attr4);
            if (time_on) {
                static int nimpl4 = 0;
                if (nimpl4 < 4) {
                    nimpl4++;
                    GGML_SYCL_XE2_LOG( "xe2mx woq4 impl=%s N=%ld M=%ld K=%ld\n", pd4.impl_info_str(), (long) N, (long) M, (long) K);
                }
            }
            dnnl::memory a_mem(a_md, eng, const_cast<sycl::half *>(xh));
            dnnl::memory b_mem(b4_md, eng, const_cast<uint8_t *>(qs));
            dnnl::memory c_mem(c_md, eng, C);
            dnnl::memory sc_mem({{nb, N}, dnnl::memory::data_type::f32, {N, 1}}, eng, scp);
            dnnl::memory zp_mem({{1}, dnnl::memory::data_type::s32, {1}}, eng, zpp);
            dnnl::memory sp_mem = ctx.get_scratchpad_mem(pd4.scratchpad_desc(), eng, stream);
            dnnl::matmul prim(pd4);
            *t_e = xe2mx_time_ms(time_on, stream, t0);
            xe2mx_time_begin(time_on, stream, &t0);
            prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem},
                {DNNL_ARG_SCRATCHPAD, sp_mem},
            });
            *t_k = xe2mx_time_ms(time_on, stream, t0);
            return true;
        } catch (const dnnl::error & e) {
            static int nerr4 = 0;
            if (nerr4 < 2) {
                nerr4++;
                GGML_SYCL_XE2_LOG( "xe2mx woq4 dnnl: %s\n", e.what());
                fflush(stderr);
            }
        } catch (const sycl::exception & e) {
            static int nerr4 = 0;
            if (nerr4 < 2) {
                nerr4++;
                GGML_SYCL_XE2_LOG( "xe2mx woq4 sycl: %s\n", e.what());
                fflush(stderr);
            }
        }
    }
    const int nocache = ggml_sycl_get_env("GGML_SYCL_XE2MX_NOCACHE", 0);
    const int noany = ggml_sycl_get_env("GGML_SYCL_XE2MX_ANY", 0) ? 0 : 1;
    xe2mx_wcache * wc = nullptr;
    auto wcit = g_xe2mx_wcache.find((const void *) packed);
    if (!nocache && wcit != g_xe2mx_wcache.end() && wcit->second.K == K && wcit->second.N == N && wcit->second.packed) {
        wc = &wcit->second;
    }
    ggml_sycl_pool_alloc<int8_t> W8(ctx.pool());
    int8_t * w8 = nullptr;
    if (!wc) {
        w8 = W8.alloc((size_t) K * (size_t) N);
        xe2mx_time_begin(time_on, stream, &t0);
        xe2mx_expand_s8(stream, qs, w8, N, nb, qpitch);
        *t_e = xe2mx_time_ms(time_on, stream, t0);
    } else {
        *t_e = 0;
    }
    ggml_sycl_pool_alloc<int8_t> Pv(ctx.pool());
    const int jm = ggml_sycl_get_env("GGML_SYCL_XE2MX_JM", 0);
    if (jm && As8 && da && (M % 8) == 0 && (N % 16) == 0) {
        int8_t * km = w8;
        if (!km && wc && wc->vnni == 0 && wc->bytes == (size_t) K * (size_t) N) {
            km = (int8_t *) wc->packed;
        }
        int8_t * vn = nullptr;
        if (wc && wc->vnni && wc->bytes == (size_t) K * (size_t) N) {
            vn = (int8_t *) wc->packed;
        } else if (km) {
            const size_t abn = (size_t) K * (size_t) N;
            xe2mx_time_begin(time_on, stream, &t0);
            vn = Pv.alloc(abn);
            xe2mx_pack_vnni(stream, km, vn, K, N);
            *t_e += xe2mx_time_ms(time_on, stream, t0);
        }
        if (vn) {
            xe2mx_time_begin(time_on, stream, &t0);
            if (xe2mx_gemm_jm(stream, vn, scp, As8, da, C, K, N, M)) {
                try {
                    *t_k = xe2mx_time_ms(time_on, stream, t0);
                    if (time_on) {
                        static int njm = 0;
                        if (njm < 8) {
                            njm++;
                            GGML_SYCL_XE2_LOG( "xe2mx jm e=%.2f k=%.2f ms N=%ld M=%ld K=%ld hit=%d\n",
                                    *t_e, *t_k, (long) N, (long) M, (long) K, (wc && wc->vnni) ? 1 : 0);
                        }
                    }
                    const size_t abn = (size_t) K * (size_t) N;
                    if (!nocache && vn && (!wc || wc->vnni == 0)) {
                        if (wc && wc->bytes == abn) {
                            stream->memcpy(wc->packed, vn, abn);
                            wc->vnni = 1;
                        } else if (abn > (1ull << 20) && g_xe2mx_wcache_used + abn <= XE2MX_WCACHE_CAP
                                   && !g_xe2mx_wcache.count((const void *) packed)) {
                            void * persist = ggml_sycl_malloc_device(abn, *stream);
                            if (persist) {
                                stream->memcpy(persist, vn, abn);
                                g_xe2mx_wcache_used += abn;
                                g_xe2mx_wcache[(const void *) packed] = { persist, abn, K, N, 1 };
                            }
                        }
                    }
                    return true;
                } catch (const sycl::exception & e) {
                    static int nerrj = 0;
                    if (nerrj < 2) {
                        nerrj++;
                        GGML_SYCL_XE2_LOG( "xe2mx jm wait: %s\n", e.what());
                        fflush(stderr);
                    }
                }
            }
        }
    }
    xe2mx_time_begin(time_on, stream, &t0);
    static int i8_group_dead = 0;
    const int i8g = ggml_sycl_get_env("GGML_SYCL_XE2MX_I8G", 0);
    int8_t * b8g = w8;
    if (!b8g && wc && wc->vnni == 0 && wc->bytes == (size_t) K * (size_t) N) {
        b8g = (int8_t *) wc->packed;
    }
    if (As8 && da && i8g && !i8_group_dead && b8g) {
        try {
            dnnl::engine eng = ctx.engine_dnnl(stream);
            dnnl::stream strm = ctx.stream_dnnl(stream);
            dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
            dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8, {N, 1});
            dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, 32}, dnnl::memory::data_type::f32);
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f32);
            auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            if (time_on) {
                static int nimpl8 = 0;
                if (nimpl8 < 4) {
                    nimpl8++;
                    GGML_SYCL_XE2_LOG( "xe2mx i8 impl=%s N=%ld M=%ld K=%ld\n", pd.impl_info_str(), (long) N, (long) M, (long) K);
                }
            }
            dnnl::memory a_mem(a_md, eng, const_cast<int8_t *>(As8));
            dnnl::memory b_mem(b_md, eng, b8g);
            dnnl::memory c_mem(c_md, eng, C);
            dnnl::memory sa_mem({{M, nb}, dnnl::memory::data_type::f32, {nb, 1}}, eng, const_cast<float *>(da));
            dnnl::memory sw_mem({{nb, N}, dnnl::memory::data_type::f32, {N, 1}}, eng, scp);
            dnnl::memory sp_mem = ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, stream);
            dnnl::matmul prim(pd);
            prim.execute(strm, {
                {DNNL_ARG_SRC, a_mem},
                {DNNL_ARG_WEIGHTS, b_mem},
                {DNNL_ARG_DST, c_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sw_mem},
                {DNNL_ARG_SCRATCHPAD, sp_mem},
            });
            *t_k = xe2mx_time_ms(time_on, stream, t0);
            return true;
        } catch (const dnnl::error & e) {
            i8_group_dead = 1;
            static int nerr8 = 0;
            if (nerr8 < 2) {
                nerr8++;
                GGML_SYCL_XE2_LOG( "xe2mx i8 dnnl: %s\n", e.what());
                fflush(stderr);
            }
        } catch (const sycl::exception & e) {
            i8_group_dead = 1;
            static int nerr8 = 0;
            if (nerr8 < 2) {
                nerr8++;
                GGML_SYCL_XE2_LOG( "xe2mx i8 sycl: %s\n", e.what());
                fflush(stderr);
            }
        }
    }
    static int i8a_dead = 0;
    const int i8a = ggml_sycl_get_env("GGML_SYCL_XE2MX_I8A", 0);
    if (i8a && !i8a_dead && xh) {
        int8_t * b8 = w8;
        if (!b8 && wc && wc->bytes == (size_t) K * (size_t) N) {
            b8 = (int8_t *) wc->packed;
        }
        if (b8) {
            ggml_sycl_pool_alloc<int8_t> Ai(ctx.pool());
            ggml_sycl_pool_alloc<float> sar(ctx.pool());
            ggml_sycl_pool_alloc<float> tmx(ctx.pool());
            int8_t * Ap = Ai.alloc((size_t) M * (size_t) K);
            float * sap = sar.alloc(1);
            float * tmp = tmx.alloc((size_t) M);
            xe2mx_quant_a_one(stream, xh, Ap, sap, tmp, M, K);
            try {
                dnnl::engine eng = ctx.engine_dnnl(stream);
                dnnl::stream strm = ctx.stream_dnnl(stream);
                dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
                dnnl::memory::desc b_user({K, N}, dnnl::memory::data_type::s8, {N, 1});
                dnnl::memory::desc b_in({K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
                dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
                dnnl::primitive_attr attr;
                attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f32);
                auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_in, c_md, attr);
                if (time_on) {
                    static int nimpla = 0;
                    if (nimpla < 4) {
                        nimpla++;
                        GGML_SYCL_XE2_LOG( "xe2mx i8a impl=%s psz=%zu N=%ld M=%ld K=%ld\n",
                                pd.impl_info_str(), pd.weights_desc().get_size(), (long) N, (long) M, (long) K);
                    }
                }
                void * bptr = b8;
                ggml_sycl_pool_alloc<uint8_t> Btmp(ctx.pool());
                const int need_ro = (pd.weights_desc() != b_user);
                if (need_ro) {
                    const size_t psz = pd.weights_desc().get_size();
                    bptr = Btmp.alloc(psz);
                    dnnl::memory b_src(b_user, eng, b8);
                    dnnl::memory b_dst(pd.weights_desc(), eng, bptr);
                    dnnl::reorder(b_src, b_dst).execute(strm, b_src, b_dst);
                }
                dnnl::memory a_mem(a_md, eng, Ap);
                dnnl::memory b_mem(pd.weights_desc(), eng, bptr);
                dnnl::memory c_mem(c_md, eng, C);
                dnnl::memory sa_mem({{1}, dnnl::memory::data_type::f32, {1}}, eng, sap);
                dnnl::memory sw_mem({{nb, N}, dnnl::memory::data_type::f32, {N, 1}}, eng, scp);
                dnnl::memory sp_mem = ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, stream);
                dnnl::matmul prim(pd);
                prim.execute(strm, {
                    {DNNL_ARG_SRC, a_mem},
                    {DNNL_ARG_WEIGHTS, b_mem},
                    {DNNL_ARG_DST, c_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sw_mem},
                    {DNNL_ARG_SCRATCHPAD, sp_mem},
                });
                *t_k = xe2mx_time_ms(time_on, stream, t0);
                return true;
            } catch (const dnnl::error & e) {
                i8a_dead = 1;
                static int nerra = 0;
                if (nerra < 2) {
                    nerra++;
                    GGML_SYCL_XE2_LOG( "xe2mx i8a dnnl: %s\n", e.what());
                    fflush(stderr);
                }
            } catch (const sycl::exception & e) {
                i8a_dead = 1;
                static int nerra = 0;
                if (nerra < 2) {
                    nerra++;
                    GGML_SYCL_XE2_LOG( "xe2mx i8a sycl: %s\n", e.what());
                    fflush(stderr);
                }
            }
        }
    }
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::f16, {K, 1});
        dnnl::memory::desc b_user({K, N}, dnnl::memory::data_type::s8, {N, 1});
        dnnl::memory::desc b_in = b_user;
        if (!noany) {
            b_in = dnnl::memory::desc({K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
        }
        dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f32);
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_in, c_md, attr);
        if (wc && wc->bytes != pd.weights_desc().get_size()) {
            wc = nullptr;
        }
        if (!wc && w8 == nullptr) {
            w8 = W8.alloc((size_t) K * (size_t) N);
            xe2mx_expand_s8(stream, qs, w8, N, nb, qpitch);
        }
        if (time_on) {
            static int nimpl = 0;
            nimpl++;
            if (nimpl <= 4 || nimpl == 64 || nimpl == 200) {
                GGML_SYCL_XE2_LOG( "xe2mx woq impl=%s any=%d psz=%zu used=%zu N=%ld M=%ld K=%ld hit=%d n=%d\n",
                        pd.impl_info_str(), noany ? 0 : 1, pd.weights_desc().get_size(),
                        g_xe2mx_wcache_used, (long) N, (long) M, (long) K, wc ? 1 : 0, nimpl);
            }
        }
        dnnl::memory a_mem(a_md, eng, const_cast<sycl::half *>(xh));
        dnnl::memory c_mem(c_md, eng, C);
        dnnl::memory sc_mem({{nb, N}, dnnl::memory::data_type::f32, {N, 1}}, eng, scp);
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, stream);
        void * bptr = nullptr;
        ggml_sycl_pool_alloc<uint8_t> Btmp(ctx.pool());
        if (wc) {
            bptr = wc->packed;
        } else {
            const size_t psz = pd.weights_desc().get_size();
            const int need_ro = !noany && (pd.weights_desc() != b_user);
            if (!need_ro) {
                bptr = w8;
            } else {
                void * persist = nullptr;
                if (!nocache && psz > (1ull << 20) && psz <= 4ull * (size_t) K * (size_t) N
                    && g_xe2mx_wcache_used + psz <= XE2MX_WCACHE_CAP) {
                    persist = ggml_sycl_malloc_device(psz, *stream);
                }
                if (persist) {
                    bptr = persist;
                    g_xe2mx_wcache_used += psz;
                    g_xe2mx_wcache[(const void *) packed] = { persist, psz, K, N, 0 };
                } else {
                    bptr = Btmp.alloc(psz);
                }
                dnnl::memory b_src(b_user, eng, w8);
                dnnl::memory b_dst(pd.weights_desc(), eng, bptr);
                dnnl::reorder(b_src, b_dst).execute(strm, b_src, b_dst);
            }
            if (!need_ro && !nocache && !g_xe2mx_wcache.count((const void *) packed)) {
                const size_t abn = (size_t) K * (size_t) N;
                if (abn > (1ull << 20) && g_xe2mx_wcache_used + abn <= XE2MX_WCACHE_CAP) {
                    void * persist = ggml_sycl_malloc_device(abn, *stream);
                    if (persist) {
                        stream->memcpy(persist, w8, abn);
                        g_xe2mx_wcache_used += abn;
                        g_xe2mx_wcache[(const void *) packed] = { persist, abn, K, N, 0 };
                        bptr = persist;
                    }
                }
            }
        }
        dnnl::memory b_mem(pd.weights_desc(), eng, bptr);
        dnnl::matmul prim(pd);
        prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_mem},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        });
    } catch (const dnnl::error & e) {
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx woq dnnl: %s\n", e.what());
            fflush(stderr);
        }
        *t_k = xe2mx_time_ms(time_on, stream, t0);
        return false;
    } catch (const sycl::exception & e) {
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx woq sycl: %s\n", e.what());
            fflush(stderr);
        }
        *t_k = xe2mx_time_ms(time_on, stream, t0);
        return false;
    }
    *t_k = xe2mx_time_ms(time_on, stream, t0);
    return true;
}
#endif

static bool xe2mx_pack_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0) {
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
    // Dual LO: t->data already SOA qs+dpl (same as pack output). Inline: s4 in slot - never AoS pack.
    if (extra && extra->optimized_feature.xe2mx_lo) {
        return true;
    }
    if (!ggml_sycl_xe2mx_lo_bytes_are_soa(src0)) {
        return true;
    }
    if (!extra || extra->optimized_feature.reorder) {
        return extra && extra->optimized_feature.reorder;
    }
    if (ggml_sycl_xe2mx_s8dw_on(src0)) {
        extra->optimized_feature.reorder = true;
        return true;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if ((K % 32) != 0 || (N % 16) != 0) {
        return false;
    }
    const int64_t nb = K / 32;
    if ((unsigned) nb * 16u < 64u) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(src0);
    const size_t pitch = (size_t) nb * 16u;
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<uint8_t> tmp(ctx.pool());
    tmp.alloc(nbytes);
    stream->memcpy(tmp.get(), src0->data, nbytes);
    const uint8_t * src = tmp.get();
    uint8_t * dst = (uint8_t *) src0->data;
    uint8_t * qs = dst;
    sycl::half * dpl = (sycl::half *) (dst + (size_t) N * pitch);
    stream->parallel_for(sycl::range<1>((size_t) N * (size_t) nb), [=](sycl::id<1> gid) {
        const int64_t t = (int64_t) gid[0];
        const int n = (int) (t / nb);
        const int ib = (int) (t % nb);
        const uint8_t * blk = src + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
        uint8_t * q = qs + (size_t) n * pitch + (size_t) ib * 16u;
        for (int j = 0; j < 16; ++j) {
            q[j] = blk[2 + j];
        }
        const uint16_t db = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
        dpl[(size_t) ib * (size_t) N + (size_t) n] = *reinterpret_cast<const sycl::half *>(&db);
    });
    extra->optimized_feature.reorder = true;
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0)) {
        const size_t qn = (size_t) N * pitch;
        stream->parallel_for(sycl::range<1>(qn), [=](sycl::id<1> gid) {
            qs[gid[0]] = (uint8_t) (qs[gid[0]] ^ 0x88u);
        });
    }
    return true;
}

static void xe2mx_quant_a_f32(ggml_backend_sycl_context & ctx, const float * x, int8_t * A, float * da,
                                int64_t M, int64_t K) {
    const int64_t nb = K / 32;
    queue_ptr stream = ctx.stream();
    stream->parallel_for(sycl::range<1>((size_t) M * (size_t) nb), [=](sycl::id<1> gid) {
        const int64_t idx = (int64_t) gid[0];
        const int m = (int) (idx / nb);
        const int ib = (int) (idx % nb);
        const float * row = x + (size_t) m * (size_t) K + (size_t) ib * 32;
        float amax = 0.0f;
        float af[32];
        for (int k = 0; k < 32; k += 8) {
            const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8> *>(row + k);
            for (int j = 0; j < 8; ++j) {
                const float val = v[j];
                af[k + j] = val;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        da[(size_t) m * (size_t) nb + (size_t) ib] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t * out = A + (size_t) m * (size_t) K + (size_t) ib * 32;
        for (int k = 0; k < 32; ++k) {
            float xq = af[k] * id;
            int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
            if (q < -127) {
                q = -127;
            }
            if (q > 127) {
                q = 127;
            }
            out[k] = (int8_t) q;
        }
    });
}

// GGML_SYCL_XE2MX_BATCH: quantize several distinct small-M activations in one launch. Segment w covers
// [end[w-1], end[w]) blocks of x[w]; the per-block body is xe2mx_quant_a_f32 verbatim.
struct xe2mx_btq_args {
    const float * x[XE2MX_BATCH_MAX];
    int8_t *      A[XE2MX_BATCH_MAX];
    float *       da[XE2MX_BATCH_MAX];
    int           end[XE2MX_BATCH_MAX];
    int           nb[XE2MX_BATCH_MAX];
    int           K[XE2MX_BATCH_MAX];
    int           n;
};

[[maybe_unused]] static void xe2mx_quant_a_f32_multi(ggml_backend_sycl_context & ctx, const xe2mx_btq_args & q) {
    const int total = q.end[q.n - 1];
    queue_ptr stream = ctx.stream();
    stream->parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> gid) {
        const int g = (int) gid[0];
        int w = 0;
        int base = 0;
#pragma unroll
        for (int j = 0; j < XE2MX_BATCH_MAX - 1; ++j) {
            if (j + 1 < q.n && g >= q.end[j]) {
                w = j + 1;
                base = q.end[j];
            }
        }
        const float * x = q.x[w];
        int8_t * A = q.A[w];
        float * da = q.da[w];
        const int64_t nb = q.nb[w];
        const int64_t K = q.K[w];
        const int64_t idx = (int64_t) (g - base);
        const int m = (int) (idx / nb);
        const int ib = (int) (idx % nb);
        const float * row = x + (size_t) m * (size_t) K + (size_t) ib * 32;
        float amax = 0.0f;
        float af[32];
        for (int k = 0; k < 32; k += 8) {
            const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8> *>(row + k);
            for (int j = 0; j < 8; ++j) {
                const float val = v[j];
                af[k + j] = val;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        da[(size_t) m * (size_t) nb + (size_t) ib] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t * out = A + (size_t) m * (size_t) K + (size_t) ib * 32;
        for (int k = 0; k < 32; ++k) {
            float xq = af[k] * id;
            int qv = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
            if (qv < -127) {
                qv = -127;
            }
            if (qv > 127) {
                qv = 127;
            }
            out[k] = (int8_t) qv;
        }
    });
}

// One launch for every small-M XE2MX4 activation quant in a decode graph. Per-call quant is ~3 us of dispatch
// each (258/step). GGML_SYCL_XE2MX_QBATCH=1. Prefill (any M>4) skips this. Lookup is exact (src, M, K).
struct xe2mx_qb_ent {
    const void * src;
    int64_t      M;
    int64_t      K;
    int8_t *     A;
    float *      da;
};
struct xe2mx_qb_k {
    const float * x;
    int8_t *      A;
    float *       da;
    int           off;
    int           nblk;
    int           nb;
    int           K;
};
static std::vector<xe2mx_qb_ent> g_xe2mx_qb;
static uint64_t g_xe2mx_qb_epoch = ~(uint64_t) 0;
static int8_t * g_xe2mx_qb_A = nullptr;
static size_t g_xe2mx_qb_Acap = 0;
static float * g_xe2mx_qb_da = nullptr;
static size_t g_xe2mx_qb_dacap = 0;
static xe2mx_qb_k * g_xe2mx_qb_dev = nullptr;
static int g_xe2mx_qb_devcap = 0;

static bool xe2mx_qbatch_hit(const void * src, int64_t M, int64_t K, int8_t ** A, float ** da) {
    if (!src || g_xe2mx_qb_epoch != g_xe2mx_xh_epoch) {
        return false;
    }
    for (const auto & e : g_xe2mx_qb) {
        if (e.src == src && e.M == M && e.K == K) {
            *A = e.A;
            *da = e.da;
            return true;
        }
    }
    return false;
}

void ggml_sycl_xe2mx_qbatch_prepare(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    g_xe2mx_qb.clear();
    g_xe2mx_qb_epoch = ~(uint64_t) 0;
    if (!cgraph || !ggml_sycl_get_env("GGML_SYCL_XE2MX_QBATCH", 0)) {
        return;
    }
    struct item { const float * x; int64_t M; int64_t K; };
    std::vector<item> items;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
            continue;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || t->src[1]->type != GGML_TYPE_F32) {
            continue;
        }
        if (!ggml_is_contiguous(t->src[1])) {
            continue;
        }
        const int64_t M = ggml_nrows(t->src[1]);
        const int64_t K = t->src[0]->ne[0];
        if (M > 4) {
            return;
        }
        if (M < 1 || K < 32 || (K % 32) != 0) {
            continue;
        }
        const float * x = (const float *) t->src[1]->data;
        bool seen = false;
        for (const auto & it : items) {
            seen = seen || (it.x == x && it.M == M && it.K == K);
        }
        if (!seen) {
            items.push_back(item{ x, M, K });
        }
    }
    if (items.size() < 2 || items.size() > 512) {
        return;
    }
    size_t abytes = 0, dbytes = 0, nwork = 0;
    for (const auto & it : items) {
        abytes += (size_t) it.M * (size_t) it.K;
        dbytes += (size_t) it.M * (size_t) (it.K / 32) * sizeof(float);
        nwork += (size_t) it.M * (size_t) (it.K / 32);
    }
    queue_ptr stream = ctx.stream();
    if (g_xe2mx_qb_Acap < abytes) {
        void * p = ggml_sycl_malloc_device(abytes, *stream);
        if (!p) {
            return;
        }
        g_xe2mx_qb_A = (int8_t *) p;
        g_xe2mx_qb_Acap = abytes;
    }
    if (g_xe2mx_qb_dacap < dbytes) {
        void * p = ggml_sycl_malloc_device(dbytes, *stream);
        if (!p) {
            return;
        }
        g_xe2mx_qb_da = (float *) p;
        g_xe2mx_qb_dacap = dbytes;
    }
    if (g_xe2mx_qb_devcap < (int) items.size()) {
        void * p = ggml_sycl_malloc_device(items.size() * sizeof(xe2mx_qb_k), *stream);
        if (!p) {
            return;
        }
        g_xe2mx_qb_dev = (xe2mx_qb_k *) p;
        g_xe2mx_qb_devcap = (int) items.size();
    }
    std::vector<xe2mx_qb_k> host(items.size());
    size_t ao = 0, dfo = 0;
    int off = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const int nb = (int) (items[i].K / 32);
        const int nblk = (int) items[i].M * nb;
        host[i] = xe2mx_qb_k{ items[i].x, g_xe2mx_qb_A + ao, g_xe2mx_qb_da + (dfo / sizeof(float)), off, nblk, nb,
                              (int) items[i].K };
        g_xe2mx_qb.push_back(xe2mx_qb_ent{ items[i].x, items[i].M, items[i].K, host[i].A, host[i].da });
        ao += (size_t) items[i].M * (size_t) items[i].K;
        dfo += (size_t) items[i].M * (size_t) nb * sizeof(float);
        off += nblk;
    }
    stream->memcpy(g_xe2mx_qb_dev, host.data(), host.size() * sizeof(xe2mx_qb_k)).wait();
    const xe2mx_qb_k * dev = g_xe2mx_qb_dev;
    const int nitems = (int) items.size();
    stream->parallel_for(sycl::range<1>(nwork), [=](sycl::id<1> gid) {
        const int g = (int) gid[0];
        int lo = 0, hi = nitems;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (dev[mid].off <= g) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const xe2mx_qb_k it = dev[lo];
        const int local = g - it.off;
        if (local < 0 || local >= it.nblk) {
            return;
        }
        const int m = local / it.nb;
        const int ib = local % it.nb;
        const float * row = it.x + (size_t) m * (size_t) it.K + (size_t) ib * 32;
        float amax = 0.0f;
        float af[32];
        for (int k = 0; k < 32; k += 8) {
            const sycl::vec<float, 8> v = *reinterpret_cast<const sycl::vec<float, 8> *>(row + k);
            for (int j = 0; j < 8; ++j) {
                const float val = v[j];
                af[k + j] = val;
                amax = sycl::fmax(amax, sycl::fabs(val));
            }
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        it.da[(size_t) m * (size_t) it.nb + (size_t) ib] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t * out = it.A + (size_t) m * (size_t) it.K + (size_t) ib * 32;
        for (int k = 0; k < 32; ++k) {
            float xq = af[k] * id;
            int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
            if (q < -127) {
                q = -127;
            }
            if (q > 127) {
                q = 127;
            }
            out[k] = (int8_t) q;
        }
    });
    g_xe2mx_qb_epoch = g_xe2mx_xh_epoch;
    static int nlog = 0;
    if (nlog < 3) {
        nlog++;
        GGML_SYCL_XE2_LOG( "xe2mx qbatch n=%d work=%zu A=%zu MiB\n", nitems, nwork, abytes >> 20);
        fflush(stderr);
    }
}

static void xe2mx_quant_a(ggml_backend_sycl_context & ctx, const sycl::half * xh, int8_t * A, float * da,
                           int64_t M, int64_t K) {
    const int64_t nb = K / 32;
    queue_ptr stream = ctx.stream();
    stream->parallel_for(sycl::range<1>((size_t) M * (size_t) nb), [=](sycl::id<1> gid) {
        const int64_t idx = (int64_t) gid[0];
        const int m = (int) (idx / nb);
        const int ib = (int) (idx % nb);
        const sycl::half * row = xh + (size_t) m * (size_t) K + (size_t) ib * 32;
        float amax = 0.0f;
        float af[32];
        for (int k = 0; k < 32; ++k) {
            const float v = (float) row[k];
            af[k] = v;
            const float av = v < 0.0f ? -v : v;
            if (av > amax) {
                amax = av;
            }
        }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        da[(size_t) m * (size_t) nb + (size_t) ib] = s;
        const float id = s > 0.0f ? 1.0f / s : 0.0f;
        int8_t * out = A + (size_t) m * (size_t) K + (size_t) ib * 32;
        for (int k = 0; k < 32; ++k) {
            float xq = af[k] * id;
            int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
            if (q < -127) {
                q = -127;
            }
            if (q > 127) {
                q = 127;
            }
            out[k] = (int8_t) q;
        }
    });
}

#ifdef GGML_SYCL_XE2MX_HAS_ESIMD

namespace {

using namespace sycl::ext::intel::esimd;

ESIMD_INLINE float xe2mx_dot32(simd<float, 32> a, simd<float, 32> b) {
    simd<float, 32> p = a * b;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        s += p[i];
    }
    return s;
}

ESIMD_INLINE simd<float, 32> xe2mx_deq32(const uint8_t * wb, int64_t nb, int n, int ib) {
    simd<float, 32> w(0.0f);
    const size_t boff = ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
    simd<uint16_t, 1> db;
    db.copy_from((const uint16_t *) (wb + boff));
    simd<sycl::half, 1> dh = db.bit_cast_view<sycl::half>();
    const float d = convert<float>(dh)[0];
    simd<uint8_t, 16> qs;
    qs.copy_from(wb + boff + 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t q = qs[j];
        w[2 * j + 0] = d * float((int) (q & 0x0F) - 8);
        w[2 * j + 1] = d * float((int) (q >> 4) - 8);
    }
    return w;
}

} // namespace

static void xe2mx_gemv_aos(ggml_backend_sycl_context & ctx,
                             const uint8_t * wb,
                             const sycl::half * xh,
                             float * C,
                             int64_t K, int64_t N, int64_t M) {
    queue_ptr stream = ctx.stream();
    const int64_t nb = K / QK_XE2MX4;
    const int Mi = (int) M;
    constexpr int VN = 1;
    const int ngrp = (int) ((N + VN - 1) / VN);
    stream->parallel_for(sycl::range<1>((size_t) ngrp), [=](sycl::id<1> gid) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int n0 = (int) gid[0] * VN;
        simd<float, 8 * VN> acc(0.0f);
        // prefetch each output's next block record (scale + nibbles) so the
        // serial acc chain is not load-latency bound; values and order unchanged
        simd<uint16_t, 1> nd[VN];
        simd<uint8_t, 16> nq[VN];
        for (int nn = 0; nn < VN; ++nn) {
            const int n = n0 + nn;
            if (n < (int) N) {
                const size_t boff = ((size_t) n * (size_t) nb) * 18u;
                nd[nn].copy_from((const uint16_t *) (wb + boff));
                nq[nn].copy_from(wb + boff + 2);
            }
        }
        // activation rows for block ib+1 prefetched alongside the weight records
        simd<sycl::half, 32> na[8];
        for (int r = 0; r < 8; ++r) {
            if (r < Mi) {
                na[r].copy_from(xh + (int64_t) r * K);
            }
        }
        for (int ib = 0; ib < (int) nb; ++ib) {
            simd<float, 32> arow[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                arow[r] = simd<float, 32>(0.0f);
                if (r < Mi) {
                    simd<sycl::half, 32> ah = na[r];
                    arow[r] = convert<float>(ah);
                    if (ib + 1 < (int) nb) {
                        na[r].copy_from(xh + (int64_t) r * K + (int64_t) (ib + 1) * QK_XE2MX4);
                    }
                }
            }
#pragma unroll
            for (int nn = 0; nn < VN; ++nn) {
                const int n = n0 + nn;
                if (n < (int) N) {
                    const simd<uint16_t, 1> cd = nd[nn];
                    simd<uint16_t, 1> cdv = cd;
                    const simd<uint8_t, 16> cq = nq[nn];
                    if (ib + 1 < (int) nb) {
                        const size_t boff = ((size_t) n * (size_t) nb + (size_t) ib + 1) * 18u;
                        nd[nn].copy_from((const uint16_t *) (wb + boff));
                        nq[nn].copy_from(wb + boff + 2);
                    }
                    simd<sycl::half, 1> dh = cdv.bit_cast_view<sycl::half>();
                    const float d = convert<float>(dh)[0];
                    simd<float, 32> w(0.0f);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const uint8_t q = cq[j];
                        w[2 * j + 0] = d * float((int) (q & 0x0F) - 8);
                        w[2 * j + 1] = d * float((int) (q >> 4) - 8);
                    }
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        if (r < Mi) {
                            acc[r * VN + nn] += xe2mx_dot32(w, arow[r]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < Mi) {
#pragma unroll
                for (int nn = 0; nn < VN; ++nn) {
                    const int n = n0 + nn;
                    if (n < (int) N) {
                        C[(int64_t) r * N + n] = acc[r * VN + nn];
                    }
                }
            }
        }
    });
}

#endif

static bool xe2mx_lo_pack_act(ggml_backend_sycl_context & ctx, const ggml_tensor * act,
                               ggml_tensor * dst, int64_t M, int64_t K,
                               uint8_t ** a4, float ** sa, int64_t * gout) {
    const int64_t g = xe2mx_s44_g();
    if (!act || !a4 || !sa || !gout || M < 1 || (K % 64) != 0 || (g % 64) != 0 || (K % g) != 0) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    const sycl::half * xh = nullptr;
    if (act->type == GGML_TYPE_F16) {
        xh = (const sycl::half *) act->data;
    } else if (g_xe2mx_xh_buf && g_xe2mx_xh_dev == ctx.device && g_xe2mx_xh_src == act && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
               && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
        xh = g_xe2mx_xh_buf;
    } else {
        const to_fp16_sycl_t to_x = xe2mx_get_to_x(act->type, dst);
        if (!to_x) {
            return false;
        }
        const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
        if (g_xe2mx_xh_cap < need || g_xe2mx_xh_dev != ctx.device) {
            void * p = ggml_sycl_malloc_device(need, *stream);
            if (p) {
                g_xe2mx_xh_buf = (sycl::half *) p;
                g_xe2mx_xh_cap = need;
                g_xe2mx_xh_dev = ctx.device;
            }
        }
        if (!g_xe2mx_xh_buf || g_xe2mx_xh_cap < need) {
            return false;
        }
        if (act->type == GGML_TYPE_F32 && ggml_is_contiguous(act)) {
            const size_t an4e = (size_t) M * (size_t) (K / 2);
            const size_t sane = (size_t) M * (size_t) (K / g) * sizeof(float);
            if (g_xe2mx_loa_cap < an4e) {
                void * p = ggml_sycl_malloc_device(an4e, *stream);
                if (!p) {
                    return false;
                }
                g_xe2mx_loa = (uint8_t *) p;
                g_xe2mx_loa_cap = an4e;
            }
            if (g_xe2mx_losa_cap < sane) {
                void * p = ggml_sycl_malloc_device(sane, *stream);
                if (!p) {
                    return false;
                }
                g_xe2mx_losa = (float *) p;
                g_xe2mx_losa_cap = sane;
            }
            if (!xe2mx_quant_a_s4_nibble_had_f32_rte(stream, (const float *) act->data,
                                                     g_xe2mx_loa, g_xe2mx_losa, M, K, g)) {
                return false;
            }
            g_xe2mx_loa_src = act;
            g_xe2mx_loa_xh = nullptr;
            g_xe2mx_loa_M = M;
            g_xe2mx_loa_K = K;
            g_xe2mx_loa_g = g;
            g_xe2mx_loa_epoch = g_xe2mx_xh_epoch;
            *a4 = g_xe2mx_loa;
            *sa = g_xe2mx_losa;
            *gout = g;
            return true;
        } else {
            to_x(act->data, g_xe2mx_xh_buf, M * K, stream);
        }
        xh = g_xe2mx_xh_buf;
        g_xe2mx_xh_src = act;
        g_xe2mx_xh_M = M;
        g_xe2mx_xh_K = K;
        g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
    }
    const size_t an4 = (size_t) M * (size_t) (K / 2);
    const size_t san = (size_t) M * (size_t) (K / g) * sizeof(float);
    if (g_xe2mx_loa && g_xe2mx_losa && g_xe2mx_loa_src == act && g_xe2mx_loa_xh == xh
        && g_xe2mx_loa_M == M && g_xe2mx_loa_K == K && g_xe2mx_loa_g == g
        && g_xe2mx_loa_epoch == g_xe2mx_xh_epoch
        && g_xe2mx_loa_cap >= an4 && g_xe2mx_losa_cap >= san) {
        *a4 = g_xe2mx_loa;
        *sa = g_xe2mx_losa;
        *gout = g;
        return true;
    }
    if (g_xe2mx_loa_cap < an4) {
        void * p = ggml_sycl_malloc_device(an4, *stream);
        if (!p) {
            return false;
        }
        g_xe2mx_loa = (uint8_t *) p;
        g_xe2mx_loa_cap = an4;
    }
    if (g_xe2mx_losa_cap < san) {
        void * p = ggml_sycl_malloc_device(san, *stream);
        if (!p) {
            return false;
        }
        g_xe2mx_losa = (float *) p;
        g_xe2mx_losa_cap = san;
    }
    if (!xe2mx_quant_a_s4_nibble_had(stream, xh, g_xe2mx_loa, g_xe2mx_losa, M, K, g)) {
        return false;
    }
    g_xe2mx_loa_src = act;
    g_xe2mx_loa_xh = xh;
    g_xe2mx_loa_M = M;
    g_xe2mx_loa_K = K;
    g_xe2mx_loa_g = g;
    g_xe2mx_loa_epoch = g_xe2mx_xh_epoch;
    *a4 = g_xe2mx_loa;
    *sa = g_xe2mx_losa;
    *gout = g;
    return true;
}

void ggml_sycl_mul_mat_xe2mx(ggml_backend_sycl_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst) {
    xe2mx_scratch_bind(ctx.device);
    GGML_ASSERT(src0->type == GGML_TYPE_XE2MX4);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % QK_XE2MX4 == 0);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(src1);
    static int nenter = 0;
    nenter++;
    const bool logp = nenter >= 400 && nenter <= 403;
    const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
    const int dpas_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_DPAS", 1);
    const int s8_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8", 0);
    const int f32a = ggml_sycl_get_env("GGML_SYCL_XE2MX_F32A", 0);
    const int q32 = ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32", 1);
    const int f16p = ggml_sycl_get_env("GGML_SYCL_XE2MX_F16P", 1);
    const int pf_early = ggml_sycl_get_env("GGML_SYCL_XE2MX_PF", 2);
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    const bool s4_shape = (N >= s4n) && ((K % QK_XE2MX4) == 0) && ((N % 16) == 0);
    const bool xe2 = ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[ctx.device].hw_info.arch);
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
    const bool packed = extra && extra->optimized_feature.reorder;
    const bool want_xmx = xe2 && dpas_on;
    const int is_lo = src0->data && g_xe2mx_lo_w.count((const void *) src0->data);
    if (is_lo && M < 16 && xe2mx_hp_res(ctx.device) && xe2mx_lo_hp_on()) {
        xe2mx_lo_hostpack_drop(*ctx.stream());
    }
    const int lo_inline = is_lo && ggml_sycl_xe2mx_lo_inline_on(src0);
    // LO_INLINE: t->data is s4 pack only - never ESIMD/XMX/AoS/SOA dequant. Fail loud if no LO consumer.
    // Dual-store (flag off): SOA routes below unless LOEARLY forces LO pack path.
    if (lo_inline) {
        static int nlo = 0;
        if (nlo < 4) {
            nlo++;
            GGML_SYCL_XE2_LOG( "xe2mx: LO_INLINE mul_mat M=%ld N=%ld K=%ld\n",
                    (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        // Small-M: lo_small (validated pack+had+pn scales) before p0/S44 M=1 path.
        if (M < 16 && xe2mx_lo_small(ctx, src0, src1, dst)) {
            return;
        }
        if (xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, true)) {
            return;
        }
        if (xe2mx_lo_small(ctx, src0, src1, dst)) {
            return;
        }
        GGML_SYCL_XE2_LOG( "xe2mx: LO_INLINE FATAL no consumer M=%ld N=%ld K=%ld (dst undefined)\n",
                (long) M, (long) N, (long) K);
        fflush(stderr);
        GGML_ABORT("xe2mx LO_INLINE: no safe mul_mat consumer");
    }
    if (is_lo && M >= 1 && ggml_sycl_get_env("GGML_SYCL_XE2MX_LOEARLY", 0)) {
        static int nlo = 0;
        if (nlo < 4) {
            nlo++;
            GGML_SYCL_XE2_LOG( "xe2mx: lo mul_mat M=%ld N=%ld K=%ld\n",
                    (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        if (!xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, true)) {
            fprintf(stderr, "xe2mx: lo dnnl fail M=%ld N=%ld K=%ld\n",
                    (long) M, (long) N, (long) K);
            fflush(stderr);
        }
        return;
    }

#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    if (g_ggml_sycl_enable_esimd && want_xmx && (packed || xe2mx_pack_xmx(ctx, src0))) {
        queue_ptr stream = ctx.stream();
        std::chrono::steady_clock::time_point t0;
        xe2mx_time_begin(time_on, stream, &t0);
        if (xe2mx_s8dw_shape(K, N) && f16p && s4_shape) {
            xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, true);
            return;
        }
        if (pf_early == 3 && M >= 32) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_prefill(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                       (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog3 = 0;
                    if (nlog3 < 8) {
                        nlog3++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf3 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 4 && M >= 32) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_xmx8(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                      (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog4 = 0;
                    if (nlog4 < 8) {
                        nlog4++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf4 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 6 && M >= 32) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_xmx_nt(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                       (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog6 = 0;
                    if (nlog6 < 8) {
                        nlog6++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf6 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 7 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_blk(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                    (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog7 = 0;
                    if (nlog7 < 8) {
                        nlog7++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf7 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 8 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_wg(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                    (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog8 = 0;
                    if (nlog8 < 8) {
                        nlog8++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf8 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 9 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_prefill_n8(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                          (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog9 = 0;
                    if (nlog9 < 8) {
                        nlog9++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf9 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 10 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> Apf(ctx.pool());
            ggml_sycl_pool_alloc<float> dapf(ctx.pool());
            bool qok = false;
            if (src1->type == GGML_TYPE_F32) {
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Apf.get(), dapf.get(), M, K);
                qok = true;
            } else {
                const sycl::half * xhpf = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhpf = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xpf.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xpf.get(), M * K, stream);
                    xhpf = Xpf.get();
                }
                Apf.alloc((size_t) M * (size_t) K);
                dapf.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a(ctx, xhpf, Apf.get(), dapf.get(), M, K);
                qok = true;
            }
            if (qok) {
                bool okpf = ggml_sycl_xe2mx_isa(ctx, (const uint8_t *) src0->data, Apf.get(), dapf.get(),
                                                     (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog10 = 0;
                    if (nlog10 < 8) {
                        nlog10++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf10 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 11 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            const sycl::half * xhpf = nullptr;
            const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
            if (src1->type == GGML_TYPE_F16) {
                xhpf = (const sycl::half *) src1->data;
            } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_src == src1 && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
                       && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
                xhpf = g_xe2mx_xh_buf;
            } else {
                const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                GGML_ASSERT(to_x);
                sycl::half * xdst = nullptr;
                if (xcache) {
                    const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
                    if (g_xe2mx_xh_cap < need) {
                        void * p = ggml_sycl_malloc_device(need, *stream);
                        if (p) {
                            g_xe2mx_xh_buf = (sycl::half *) p;
                            g_xe2mx_xh_cap = need;
                        }
                    }
                    if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                        xdst = g_xe2mx_xh_buf;
                    }
                }
                if (!xdst) {
                    Xpf.alloc((size_t) M * (size_t) K);
                    xdst = Xpf.get();
                }
                to_x(src1->data, xdst, M * K, stream);
                xhpf = xdst;
                if (xcache && xdst == g_xe2mx_xh_buf) {
                    g_xe2mx_xh_src = src1;
                    g_xe2mx_xh_M = M;
                    g_xe2mx_xh_K = K;
                    g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
                    g_xe2mx_xh_dev = ctx.device;
                }
            }
            if (xhpf) {
                bool okpf = ggml_sycl_xe2mx_isa_h(ctx, (const uint8_t *) src0->data, xhpf,
                                                         (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog11 = 0;
                    if (nlog11 < 8) {
                        nlog11++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf11 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        if (pf_early == 13 && M >= 1024) {
            ggml_sycl_pool_alloc<sycl::half> Xpf(ctx.pool());
            const sycl::half * xhpf = nullptr;
            const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
            if (src1->type == GGML_TYPE_F16) {
                xhpf = (const sycl::half *) src1->data;
            } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_src == src1 && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
                       && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
                xhpf = g_xe2mx_xh_buf;
            } else {
                const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                GGML_ASSERT(to_x);
                sycl::half * xdst = nullptr;
                if (xcache) {
                    const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
                    if (g_xe2mx_xh_cap < need) {
                        void * p = ggml_sycl_malloc_device(need, *stream);
                        if (p) {
                            g_xe2mx_xh_buf = (sycl::half *) p;
                            g_xe2mx_xh_cap = need;
                        }
                    }
                    if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                        xdst = g_xe2mx_xh_buf;
                    }
                }
                if (!xdst) {
                    Xpf.alloc((size_t) M * (size_t) K);
                    xdst = Xpf.get();
                }
                to_x(src1->data, xdst, M * K, stream);
                xhpf = xdst;
                if (xcache && xdst == g_xe2mx_xh_buf) {
                    g_xe2mx_xh_src = src1;
                    g_xe2mx_xh_M = M;
                    g_xe2mx_xh_K = K;
                    g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
                    g_xe2mx_xh_dev = ctx.device;
                }
            }
            if (xhpf) {
                bool okpf = ggml_sycl_xe2mx_isa_hf(ctx, (const uint8_t *) src0->data, xhpf,
                                                         (float *) dst->data, K, N, M);
                if (time_on) {
                    static int nlog13 = 0;
                    if (nlog13 < 8) {
                        nlog13++;
                        const double tk = xe2mx_time_ms(time_on, stream, t0);
                        GGML_SYCL_XE2_LOG( "xe2mx pf13 k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, okpf ? 1 : 0);
                    }
                }
                if (okpf) {
                    return;
                }
            }
        }
        const int s8x = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8X", 0);
        if (s8x && M >= 32) {
            ggml_sycl_pool_alloc<sycl::half> Xs(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> As(ctx.pool());
            ggml_sycl_pool_alloc<float> das(ctx.pool());
            As.alloc((size_t) M * (size_t) K);
            das.alloc((size_t) M * (size_t) (K / 32));
            if (src1->type == GGML_TYPE_F32) {
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, As.get(), das.get(), M, K);
            } else {
                const sycl::half * xhs = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhs = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xs.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xs.get(), M * K, stream);
                    xhs = Xs.get();
                }
                xe2mx_quant_a(ctx, xhs, As.get(), das.get(), M, K);
            }
            int8_t * w8 = xe2mx_s8k_cached(stream, (const uint8_t *) src0->data, K, N);
            ggml_sycl_pool_alloc<int8_t> W8p(ctx.pool());
            if (!w8) {
                w8 = W8p.alloc((size_t) K * (size_t) N);
                const size_t qpitch_e = (size_t) (K / 32) * 16u;
                xe2mx_expand_s8(stream, (const uint8_t *) src0->data, w8, N, K / 32, qpitch_e);
            }
            const size_t qpitch = (size_t) (K / 32) * 16u;
            const uint16_t * d16 = (const uint16_t *) ((const uint8_t *) src0->data + (size_t) N * qpitch);
            xe2mx_time_begin(time_on, stream, &t0);
            bool oks = ggml_sycl_xe2mx_gemm_s8_vnni(ctx, w8, d16, As.get(), das.get(),
                                                       (float *) dst->data, K, N, M);
            const double tk = xe2mx_time_ms(time_on, stream, t0);
            if (time_on) {
                static int nlogx = 0;
                if (nlogx < 8) {
                    nlogx++;
                    GGML_SYCL_XE2_LOG( "xe2mx s8x k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            tk, (long) N, (long) M, (long) K, oks ? 1 : 0);
                }
            }
            if (oks) {
                return;
            }
        }
        if (s8_on && M >= 32 && (N >= 8192 || K >= 8192)) {
            ggml_sycl_pool_alloc<sycl::half> Xs(ctx.pool());
            ggml_sycl_pool_alloc<int8_t> As(ctx.pool());
            ggml_sycl_pool_alloc<float> das(ctx.pool());
            As.alloc((size_t) M * (size_t) K);
            das.alloc((size_t) M * (size_t) (K / 32));
            if (src1->type == GGML_TYPE_F32) {
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, As.get(), das.get(), M, K);
            } else {
                const sycl::half * xhs = nullptr;
                if (src1->type == GGML_TYPE_F16) {
                    xhs = (const sycl::half *) src1->data;
                } else {
                    const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
                    GGML_ASSERT(to_x);
                    Xs.alloc((size_t) M * (size_t) K);
                    to_x(src1->data, Xs.get(), M * K, stream);
                    xhs = Xs.get();
                }
                xe2mx_quant_a(ctx, xhs, As.get(), das.get(), M, K);
            }
            int8_t * w8 = xe2mx_s8k_cached(stream, (const uint8_t *) src0->data, K, N);
            if (w8) {
                const size_t qpitch = (size_t) (K / 32) * 16u;
                const uint16_t * d16 = (const uint16_t *) ((const uint8_t *) src0->data + (size_t) N * qpitch);
                xe2mx_time_begin(time_on, stream, &t0);
                bool oks = ggml_sycl_xe2mx_gemm_s8(ctx, w8, d16, As.get(), das.get(),
                                                     (float *) dst->data, K, N, M, N);
                const double tk = xe2mx_time_ms(time_on, stream, t0);
                if (time_on) {
                    static int nlogs = 0;
                    if (nlogs < 8) {
                        nlogs++;
                        GGML_SYCL_XE2_LOG( "xe2mx s8p k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                                tk, (long) N, (long) M, (long) K, oks ? 1 : 0);
                    }
                }
                if (oks) {
                    return;
                }
            }
        }
        if (f16p && M >= 1024 && s4_shape) {
            xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, true);
            return;
        }
        const int s4m = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4M", 1024);
        if (f16p && M >= s4m && M < 1024) {
            if (xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, true, nullptr, false)) {
                return;
            }
        }
        if (M == 1 && src1->type == GGML_TYPE_F32 && ggml_sycl_get_env("GGML_SYCL_XE2MX_GEMV", 0)) {
            bool okv = ggml_sycl_xe2mx_gemv_f32(ctx, (const uint8_t *) src0->data,
                                                  (const float *) src1->data,
                                                  (float *) dst->data, K, N);
            if (time_on) {
                static int nlogv = 0;
                if (nlogv < 8) {
                    nlogv++;
                    const double tk = xe2mx_time_ms(time_on, stream, t0);
                    GGML_SYCL_XE2_LOG( "xe2mx gemv k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            tk, (long) N, (long) M, (long) K, okv ? 1 : 0);
                }
            }
            if (okv) {
                return;
            }
        }
        if (f32a && M < 32 && src1->type == GGML_TYPE_F32) {
            bool okf = ggml_sycl_xe2mx_xmx_f32(ctx, (const uint8_t *) src0->data,
                                                (const float *) src1->data,
                                                (float *) dst->data, K, N, M);
            if (time_on) {
                static int nlogf = 0;
                if (nlogf < 8) {
                    nlogf++;
                    const double tk = xe2mx_time_ms(time_on, stream, t0);
                    GGML_SYCL_XE2_LOG( "xe2mx f32a k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            tk, (long) N, (long) M, (long) K, okf ? 1 : 0);
                }
            }
            if (okf) {
                return;
            }
        }
        if (q32 && M < 32 && src1->type == GGML_TYPE_F32) {
            const int acache = ggml_sycl_get_env("GGML_SYCL_XE2MX_ACACHE", 0);
            ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
            ggml_sycl_pool_alloc<float> daq(ctx.pool());
            int8_t * Ap = nullptr;
            float * dap = nullptr;
            int ahit = 0;
            int8_t * qbA = nullptr;
            float * qbda = nullptr;
            if (xe2mx_qbatch_hit(src1->data, M, K, &qbA, &qbda)) {
                Ap = qbA;
                dap = qbda;
                ahit = 1;
            } else if (acache && g_xe2mx_aq_buf && g_xe2mx_daq_buf && g_xe2mx_aq_src == src1
                && g_xe2mx_aq_M == M && g_xe2mx_aq_K == K
                && g_xe2mx_aq_hit_epoch == g_xe2mx_xh_epoch) {
                Ap = g_xe2mx_aq_buf;
                dap = g_xe2mx_daq_buf;
                ahit = 1;
            } else {
                if (acache) {
                    const size_t an = (size_t) M * (size_t) K;
                    const size_t dn = (size_t) M * (size_t) (K / 32) * sizeof(float);
                    if (g_xe2mx_aq_cap < an) {
                        void * p = ggml_sycl_malloc_device(an, *stream);
                        if (p) {
                            g_xe2mx_aq_buf = (int8_t *) p;
                            g_xe2mx_aq_cap = an;
                        }
                    }
                    if (g_xe2mx_daq_cap < dn) {
                        void * p = ggml_sycl_malloc_device(dn, *stream);
                        if (p) {
                            g_xe2mx_daq_buf = (float *) p;
                            g_xe2mx_daq_cap = dn;
                        }
                    }
                    if (g_xe2mx_aq_buf && g_xe2mx_aq_cap >= an
                        && g_xe2mx_daq_buf && g_xe2mx_daq_cap >= dn) {
                        Ap = g_xe2mx_aq_buf;
                        dap = g_xe2mx_daq_buf;
                    }
                }
                if (!Ap) {
                    Aq.alloc((size_t) M * (size_t) K);
                    daq.alloc((size_t) M * (size_t) (K / 32));
                    Ap = Aq.get();
                    dap = daq.get();
                }
                xe2mx_quant_a_f32(ctx, (const float *) src1->data, Ap, dap, M, K);
                if (acache && Ap == g_xe2mx_aq_buf) {
                    g_xe2mx_aq_src = src1;
                    g_xe2mx_aq_M = M;
                    g_xe2mx_aq_K = K;
                    g_xe2mx_aq_hit_epoch = g_xe2mx_xh_epoch;
                } else {
                    g_xe2mx_aq_src = nullptr;
                }
            }
            bool okq = false;
            if (ggml_sycl_get_env("GGML_SYCL_XE2MX_NT", 0)) {
                okq = ggml_sycl_xe2mx_xmx_nt(ctx, (const uint8_t *) src0->data, Ap, dap,
                                                (float *) dst->data, K, N, M);
            }
            const int q32wg = ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32WG", 0);
            // WG=8 KB=4 for shapes whose WG=4 grid ends in a ragged wave (e.g. GDN gate N=6144 is 2.4 waves on B65).
            // Q32WG=8 = every shape; Q32WG8_N = comma list of N.
            if (!okq && (q32wg == 8 || xe2mx_q32wg8_n_hit(N))) {
                okq = ggml_sycl_xe2mx_xmx8(ctx, (const uint8_t *) src0->data, Ap, dap,
                                              (float *) dst->data, K, N, M);
            }
            if (!okq && q32wg == 2) {
                okq = ggml_sycl_xe2mx_xmx2(ctx, (const uint8_t *) src0->data, Ap, dap,
                                              (float *) dst->data, K, N, M);
            }
            if (!okq && q32wg == 1) {
                okq = ggml_sycl_xe2mx_xmx1(ctx, (const uint8_t *) src0->data, Ap, dap,
                                              (float *) dst->data, K, N, M);
            }
            if (!okq) {
                okq = ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) src0->data, Ap, dap,
                                             (float *) dst->data, K, N, M);
            }
            if (time_on) {
                static int nlogq = 0;
                if (nlogq < 8) {
                    nlogq++;
                    const double tk = xe2mx_time_ms(time_on, stream, t0);
                    GGML_SYCL_XE2_LOG( "xe2mx q32 k=%.2f ms N=%ld M=%ld K=%ld ok=%d hit=%d\n",
                            tk, (long) N, (long) M, (long) K, okq ? 1 : 0, ahit);
                }
            }
            if (okq) {
                if (logp) {
                    GGML_SYCL_XE2_LOG( "xe2mx mm q32 M=%ld N=%ld K=%ld\n", (long) M, (long) N, (long) K);
                }
                return;
            }
        }
        if (logp) {
            GGML_SYCL_XE2_LOG( "xe2mx mm fall M=%ld N=%ld K=%ld src1=%d\n", (long) M, (long) N, (long) K, (int) src1->type);
        }
        ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
        const sycl::half * xh = nullptr;
        if (src1->type == GGML_TYPE_F16) {
            xh = (const sycl::half *) src1->data;
        } else {
            const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
            GGML_ASSERT(to_x);
            X.alloc((size_t) M * (size_t) K);
            to_x(src1->data, X.get(), M * K, stream);
            xh = X.get();
        }
        bool ok = false;
        double t_e = 0;
        double t_k = 0;
        double t_x = 0;
        ggml_sycl_pool_alloc<int8_t> A(ctx.pool());
        ggml_sycl_pool_alloc<float> da(ctx.pool());
        int have_a = 0;
        const int i8g = ggml_sycl_get_env("GGML_SYCL_XE2MX_I8G", 0);
        const int jm = ggml_sycl_get_env("GGML_SYCL_XE2MX_JM", 0);
        const int soa_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_SOA", 0);
        const int ks_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_KS", 0);
        const int pf_on = !soa_on && (M >= 32) && (pf_early == 1 || (pf_early == 2 && (M < 1024 || !s4_shape)));
#if GGML_SYCL_DNNL
        if (M >= 32 && pf_on) {
            A.alloc((size_t) M * (size_t) K);
            da.alloc((size_t) M * (size_t) (K / 32));
            xe2mx_time_begin(time_on, stream, &t0);
            xe2mx_quant_a(ctx, xh, A.get(), da.get(), M, K);
            t_x = xe2mx_time_ms(time_on, stream, t0);
            have_a = 1;
            xe2mx_time_begin(time_on, stream, &t0);
            ok = ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) src0->data, A.get(), da.get(),
                                           (float *) dst->data, K, N, M);
            t_k = xe2mx_time_ms(time_on, stream, t0);
            if (time_on) {
                static int nlogp = 0;
                if (nlogp < 8) {
                    nlogp++;
                    GGML_SYCL_XE2_LOG( "xe2mx soa x=%.2f k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            t_x, t_k, (long) N, (long) M, (long) K, ok ? 1 : 0);
                }
            }
            if (ok) {
                return;
            }
        }
        if (ks_on && M >= 1024) {
            if (!have_a) {
                A.alloc((size_t) M * (size_t) K);
                da.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_time_begin(time_on, stream, &t0);
                xe2mx_quant_a(ctx, xh, A.get(), da.get(), M, K);
                t_x = xe2mx_time_ms(time_on, stream, t0);
                have_a = 1;
            }
            xe2mx_time_begin(time_on, stream, &t0);
            ok = ggml_sycl_xe2mx_ksplit(ctx, (const uint8_t *) src0->data, A.get(), da.get(),
                                          (float *) dst->data, K, N, M);
            t_k = xe2mx_time_ms(time_on, stream, t0);
            if (time_on) {
                static int nlogk = 0;
                if (nlogk < 8) {
                    nlogk++;
                    GGML_SYCL_XE2_LOG( "xe2mx ks x=%.2f k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            t_x, t_k, (long) N, (long) M, (long) K, ok ? 1 : 0);
                }
            }
            if (ok) {
                return;
            }
        }
        if (M >= 32 && !soa_on) {
            const int8_t * As8 = nullptr;
            const float * dap = nullptr;
            if (i8g || jm) {
                A.alloc((size_t) M * (size_t) K);
                da.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_time_begin(time_on, stream, &t0);
                xe2mx_quant_a(ctx, xh, A.get(), da.get(), M, K);
                t_x = xe2mx_time_ms(time_on, stream, t0);
                As8 = A.get();
                dap = da.get();
                have_a = 1;
            }
            ok = xe2mx_dnnl_woq(ctx, (const uint8_t *) src0->data, xh, As8, dap,
                                  (float *) dst->data, K, N, M, time_on, &t_e, &t_k);
            if (time_on) {
                static int nlog = 0;
                if (nlog < 8) {
                    nlog++;
                    GGML_SYCL_XE2_LOG( "xe2mx woq x=%.2f e=%.2f k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                            t_x, t_e, t_k, (long) N, (long) M, (long) K, ok ? 1 : 0);
                }
            }
            if (ok) {
                return;
            }
        }
#endif
        if (!have_a) {
            A.alloc((size_t) M * (size_t) K);
            da.alloc((size_t) M * (size_t) (K / 32));
            xe2mx_time_begin(time_on, stream, &t0);
            xe2mx_quant_a(ctx, xh, A.get(), da.get(), M, K);
            t_x = xe2mx_time_ms(time_on, stream, t0);
        }
        if (s8_on && M >= 32) {
            const int64_t pitch = ((N + 63) / 64) * 64;
            ggml_sycl_pool_alloc<int8_t> W8(ctx.pool());
            int8_t * w8 = W8.alloc((size_t) K * (size_t) pitch);
            const size_t qpitch = (size_t) (K / 32) * 16u;
            const uint8_t * qs = (const uint8_t *) src0->data;
            const uint16_t * d16 = (const uint16_t *) (qs + (size_t) N * qpitch);
            const int n16 = (int) (N / 16);
            xe2mx_time_begin(time_on, stream, &t0);
            stream->parallel_for(sycl::range<2>((size_t) K, (size_t) n16), [=](sycl::id<2> gid) {
                const int k = (int) gid[0];
                const int n0 = (int) gid[1] * 16;
                const int ib = k / 32;
                const int kn = k - ib * 32;
                const int jb = kn / 2;
                const int hi = kn & 1;
                int8_t * w = w8 + (size_t) k * (size_t) pitch + (size_t) n0;
                for (int i = 0; i < 16; ++i) {
                    const uint8_t b = qs[(size_t) (n0 + i) * qpitch + (size_t) ib * 16u + (size_t) jb];
                    const int v = (int) (hi ? (b >> 4) : (b & 0x0F)) - 8;
                    w[i] = (int8_t) v;
                }
            });
            t_e = xe2mx_time_ms(time_on, stream, t0);
            xe2mx_time_begin(time_on, stream, &t0);
            ok = ggml_sycl_xe2mx_gemm_s8(ctx, w8, d16, A.get(), da.get(),
                                          (float *) dst->data, K, N, M, pitch);
            t_k = xe2mx_time_ms(time_on, stream, t0);
        }
        if (!ok) {
            xe2mx_time_begin(time_on, stream, &t0);
            ok = ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) src0->data, A.get(), da.get(),
                                      (float *) dst->data, K, N, M);
            t_k = xe2mx_time_ms(time_on, stream, t0);
        }
        if (time_on) {
            static int nlog = 0;
            if (nlog < 8) {
                nlog++;
                GGML_SYCL_XE2_LOG( "xe2mx xmx x=%.2f e=%.2f k=%.2f ms N=%ld M=%ld K=%ld ok=%d\n",
                        t_x, t_e, t_k, (long) N, (long) M, (long) K, ok ? 1 : 0);
            }
        }
        if (ok) {
            return;
        }
    } else if (g_ggml_sycl_enable_esimd && xe2 && !packed && M < 32
               && ggml_sycl_xe2mx_lo_bytes_are_soa(src0)) {
        queue_ptr stream = ctx.stream();
        ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
        const sycl::half * xh = nullptr;
        if (src1->type == GGML_TYPE_F16) {
            xh = (const sycl::half *) src1->data;
        } else {
            const to_fp16_sycl_t to_x = xe2mx_get_to_x(src1->type, dst);
            GGML_ASSERT(to_x);
            X.alloc((size_t) M * (size_t) K);
            to_x(src1->data, X.get(), M * K, stream);
            xh = X.get();
        }
        {
            static std::map<std::tuple<long, long, long, int>, int> naos;
            auto key = std::make_tuple((long) M, (long) N, (long) K, (int) src1->type);
            auto & cnt = naos[key];
            if (cnt < 2 && naos.size() < 48) {
                cnt++;
                GGML_SYCL_XE2_LOG( "xe2mx aos M=%ld N=%ld K=%ld nb=%ld src1type=%d\n",
                        (long) M, (long) N, (long) K, (long) (K / QK_XE2MX4), (int) src1->type);
                fflush(stderr);
            }
        }
        xe2mx_gemv_aos(ctx, (const uint8_t *) src0->data, xh, (float *) dst->data, K, N, M);
        return;
    }
#else
    GGML_UNUSED(xe2);
    GGML_UNUSED(dpas_on);
    GGML_UNUSED(s8_on);
    GGML_UNUSED(f32a);
    GGML_UNUSED(q32);
    GGML_UNUSED(f16p);
    GGML_UNUSED(want_xmx);
#endif
    extra = (ggml_tensor_extra_gpu *) src0->extra;
    xe2mx_p0_dnnl(ctx, src0, src1, dst, time_on, extra && extra->optimized_feature.reorder);
}

int ggml_sycl_xe2mx_ovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_OVL", 0) == 0) {
        return 0;
    }
    if (!cgraph || node_idx < 0 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    if (!g_ggml_sycl_enable_esimd) {
        return 0;
    }
    ggml_tensor * nodes[8];
    int nrun = 0;
    const ggml_tensor * act = nullptr;
    int64_t K0 = 0;
    int64_t M0 = 0;
    for (int j = node_idx; j < cgraph->n_nodes && nrun < 8; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32) {
            break;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || !ggml_is_contiguous(t->src[0]) ||
            !ggml_is_contiguous(t->src[1]) || !ggml_is_contiguous(t)) {
            break;
        }
        const int64_t M = ggml_nrows(t->src[1]);
        const int64_t K = t->src[0]->ne[0];
        const int64_t N = t->src[0]->ne[1];
        if (M < 1 || M >= 32 || (K % 32) != 0 || (N % 16) != 0) {
            break;
        }
        if (t->src[1]->type != GGML_TYPE_F32) {
            break;
        }
        if (nrun == 0) {
            act = t->src[1];
            K0 = K;
            M0 = M;
        } else if (t->src[1] != act || K != K0 || M != M0) {
            break;
        }
        int alias = 0;
        for (int p = 0; p < nrun; ++p) {
            if (nodes[p]->data == t->data) {
                alias = 1;
                break;
            }
        }
        if (alias) {
            break;
        }
        nodes[nrun++] = t;
    }
    if (nrun < 2) {
        return 0;
    }
    for (int i = 0; i < nrun; ++i) {
        if (!xe2mx_pack_xmx(ctx, nodes[i]->src[0])) {
            return 0;
        }
    }
    queue_ptr main = ctx.stream();
    static sycl::queue * side = nullptr;
    static queue_ptr side_seen = nullptr;
    if (!side || side_seen != main) {
        delete side;
        side = new sycl::queue(main->get_context(), main->get_device(),
                                sycl::property_list{ sycl::property::queue::in_order() });
        side_seen = main;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    Aq.alloc((size_t) M0 * (size_t) K0);
    daq.alloc((size_t) M0 * (size_t) (K0 / 32));
    xe2mx_quant_a_f32(ctx, (const float *) act->data, Aq.get(), daq.get(), M0, K0);
    {
        sycl::event evq = main->ext_oneapi_submit_barrier();
        side->ext_oneapi_submit_barrier({ evq });
    }
    int okall = 1;
    for (int i = 0; i < nrun; ++i) {
        ctx.ovl_q = (i & 1) ? side : nullptr;
        const int64_t N = nodes[i]->src[0]->ne[1];
        if (!ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) nodes[i]->src[0]->data, Aq.get(), daq.get(),
                                 (float *) nodes[i]->data, K0, N, M0)) {
            okall = 0;
            break;
        }
    }
    ctx.ovl_q = nullptr;
    side->wait();
    main->wait();
    if (!okall) {
        return 0;
    }
    return nrun;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

int ggml_sycl_xe2mx_f2_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_F2", 0) == 0) {
        return 0;
    }
    if (!cgraph || node_idx < 0 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    if (!g_ggml_sycl_enable_esimd) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    ggml_tensor * b = cgraph->nodes[node_idx + 1];
    if (!a || !b || a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    if (!a->src[0] || !a->src[1] || !b->src[0] || !b->src[1]) {
        return 0;
    }
    if (a->src[0]->type != GGML_TYPE_XE2MX4 || b->src[0]->type != GGML_TYPE_XE2MX4) {
        return 0;
    }
    if (a->src[1] != b->src[1] || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return 0;
    }
    if (a->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(a->src[0]) || !ggml_is_contiguous(b->src[0]) ||
        !ggml_is_contiguous(a->src[1]) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return 0;
    }
    if (a->data == b->data) {
        return 0;
    }
    const int64_t K = a->src[0]->ne[0];
    const int64_t N = a->src[0]->ne[1];
    const int64_t M = ggml_nrows(a->src[1]);
    if (b->src[0]->ne[0] != K || b->src[0]->ne[1] != N) {
        return 0;
    }
    if (M < 1 || M >= 32 || (K % 32) != 0 || (N % 16) != 0) {
        return 0;
    }
    if (!xe2mx_pack_xmx(ctx, a->src[0]) || !xe2mx_pack_xmx(ctx, b->src[0])) {
        return 0;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    Aq.alloc((size_t) M * (size_t) K);
    daq.alloc((size_t) M * (size_t) (K / 32));
    xe2mx_quant_a_f32(ctx, (const float *) a->src[1]->data, Aq.get(), daq.get(), M, K);
    if (!ggml_sycl_xe2mx_xmx_pair(ctx, (const uint8_t *) a->src[0]->data, (const uint8_t *) b->src[0]->data,
                                   Aq.get(), daq.get(), (float *) a->data, (float *) b->data, K, N, M)) {
        return 0;
    }
    return 2;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

static bool xe2mx_bytes_overlap(const ggml_tensor * x, const ggml_tensor * y) {
    if (!x || !y || !x->data || !y->data) {
        return false;
    }
    const char * a0 = (const char *) x->data;
    const char * a1 = a0 + ggml_nbytes(x);
    const char * b0 = (const char *) y->data;
    const char * b1 = b0 + ggml_nbytes(y);
    return a0 < b1 && b0 < a1;
}

// Writing t's result at node ia (t itself sits at ib > ia) is safe when no node in (ia, ib) writes or reads
// memory that overlaps t. ggml-alloc may hand t's bytes to tensors that die before ib.
static bool xe2mx_early_write_safe(const ggml_cgraph * g, int ia, int ib, const ggml_tensor * t) {
    for (int j = ia + 1; j < ib; ++j) {
        const ggml_tensor * n = g->nodes[j];
        if (!n || n == t) {
            continue;
        }
        if (xe2mx_bytes_overlap(n, t)) {
            return false;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] && n->src[s] != t && xe2mx_bytes_overlap(n->src[s], t)) {
                return false;
            }
        }
    }
    return true;
}

// N not a multiple of 16 (alpha/beta shards are 24) cannot join MG. Pack a side SOA
// buffer padded up to the next multiple of 16 and store only the true columns.
struct xe2mx_mgpad {
    uint8_t * packed = nullptr;
    int64_t Npad = 0;
    int64_t Ntrue = 0;
    int64_t K = 0;
};
static std::unordered_map<const void *, xe2mx_mgpad> g_xe2mx_mgpad;

static bool xe2mx_mg_pad_ensure(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                int64_t * n_phys, const uint8_t ** vx) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_XE2MX_MGPAD", 1);
    if (!on || !src0 || !src0->data || !n_phys || !vx || src0->type != GGML_TYPE_XE2MX4) {
        return false;
    }
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
    if ((extra && (extra->optimized_feature.xe2mx_lo || extra->optimized_feature.reorder))
        || !ggml_sycl_xe2mx_lo_bytes_are_soa(src0)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (N <= 0 || N >= 64 || (N % 16) == 0 || (K % 32) != 0) {
        return false;
    }
    const int64_t nb = K / 32;
    if ((unsigned) nb * 16u < 64u) {
        return false;
    }
    const int64_t Npad = (N + 15) & ~(int64_t) 15;
    auto it = g_xe2mx_mgpad.find(src0->data);
    if (it != g_xe2mx_mgpad.end() && it->second.packed && it->second.Ntrue == N
        && it->second.K == K && it->second.Npad == Npad) {
        *n_phys = it->second.Npad;
        *vx = it->second.packed;
        return true;
    }
    const size_t pitch = (size_t) nb * 16u;
    const size_t nbytes = (size_t) Npad * pitch + (size_t) Npad * (size_t) nb * sizeof(sycl::half);
    queue_ptr stream = ctx.stream();
    uint8_t * dst = (uint8_t *) ggml_sycl_malloc_device(nbytes, *stream);
    if (!dst) {
        return false;
    }
    stream->memset(dst, 0, nbytes);
    const size_t src_bytes = (size_t) N * (size_t) nb * 18u;
    ggml_sycl_pool_alloc<uint8_t> tmp(ctx.pool());
    uint8_t * src = tmp.alloc(src_bytes);
    if (!src) {
        return false;
    }
    stream->memcpy(src, src0->data, src_bytes);
    uint8_t * qs = dst;
    sycl::half * dpl = (sycl::half *) (dst + (size_t) Npad * pitch);
    const int64_t Nt = N;
    const int64_t Np = Npad;
    stream->parallel_for(sycl::range<1>((size_t) Nt * (size_t) nb), [=](sycl::id<1> gid) {
        const int64_t t = (int64_t) gid[0];
        const int n = (int) (t / nb);
        const int ib = (int) (t % nb);
        const uint8_t * blk = src + ((size_t) n * (size_t) nb + (size_t) ib) * 18u;
        uint8_t * q = qs + (size_t) n * pitch + (size_t) ib * 16u;
        for (int j = 0; j < 16; ++j) {
            q[j] = blk[2 + j];
        }
        const uint16_t db = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
        dpl[(size_t) ib * (size_t) Np + (size_t) n] = *reinterpret_cast<const sycl::half *>(&db);
    });
    g_xe2mx_mgpad[src0->data] = { dst, Npad, N, K };
    *n_phys = Npad;
    *vx = dst;
    static int nlog = 0;
    if (nlog < 4) {
        nlog++;
        GGML_SYCL_XE2_LOG( "xe2mx mg pad %s N=%ld -> %ld K=%ld\n",
                src0->name ? src0->name : "?", (long) N, (long) Npad, (long) K);
        fflush(stderr);
    }
    return true;
}

// One-shot: padded N=24 store mask must match the unmasked N=32 kernel on the real columns.
static void xe2mx_mg_pad_selftest(ggml_backend_sycl_context & ctx) {
    static int done = 0;
    if (done || ggml_sycl_get_env("GGML_SYCL_XE2MX_MGPAD_TEST", 0) == 0) {
        return;
    }
    done = 1;
    const int64_t K = 256;
    const int64_t N = 24;
    const int64_t M = 2;
    const int64_t nb = K / 32;
    const size_t aos_bytes = (size_t) N * (size_t) nb * 18u;
    queue_ptr stream = ctx.stream();
    std::vector<uint8_t> haos(aos_bytes);
    for (size_t i = 0; i < aos_bytes; i += 18) {
        haos[i] = 0x00;
        haos[i + 1] = 0x3c;
        for (int j = 0; j < 16; ++j) {
            haos[i + 2 + j] = (uint8_t) (0x11 * (((i / 18) + (size_t) j) & 7));
        }
    }
    uint8_t * aos = (uint8_t *) ggml_sycl_malloc_device(aos_bytes, *stream);
    if (!aos) {
        fprintf(stderr, "xe2mx mg pad TEST alloc fail\n");
        fflush(stderr);
        return;
    }
    stream->memcpy(aos, haos.data(), aos_bytes).wait();
    ggml_tensor fake = {};
    fake.type = GGML_TYPE_XE2MX4;
    fake.ne[0] = K;
    fake.ne[1] = N;
    fake.data = aos;
    int64_t nphys = 0;
    const uint8_t * vx = nullptr;
    if (!xe2mx_mg_pad_ensure(ctx, &fake, &nphys, &vx) || nphys != 32 || !vx) {
        fprintf(stderr, "xe2mx mg pad TEST pack fail nphys=%ld\n", (long) nphys);
        fflush(stderr);
        return;
    }
    std::vector<float> hx((size_t) M * (size_t) K, 0.25f);
    float * x = (float *) ggml_sycl_malloc_device(hx.size() * sizeof(float), *stream);
    if (!x) {
        fprintf(stderr, "xe2mx mg pad TEST x alloc fail\n");
        fflush(stderr);
        return;
    }
    stream->memcpy(x, hx.data(), hx.size() * sizeof(float)).wait();
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    int8_t * A = Aq.alloc((size_t) M * (size_t) K);
    float * da = daq.alloc((size_t) M * (size_t) nb);
    if (!A || !da) {
        fprintf(stderr, "xe2mx mg pad TEST quant alloc fail\n");
        fflush(stderr);
        return;
    }
    xe2mx_quant_a_f32(ctx, x, A, da, M, K);
    std::vector<float> h24((size_t) M * 24, 12345.f);
    std::vector<float> h32((size_t) M * 32, 12345.f);
    float * c24 = (float *) ggml_sycl_malloc_device(h24.size() * sizeof(float), *stream);
    float * c32 = (float *) ggml_sycl_malloc_device(h32.size() * sizeof(float), *stream);
    if (!c24 || !c32) {
        fprintf(stderr, "xe2mx mg pad TEST C alloc fail\n");
        fflush(stderr);
        return;
    }
    stream->memcpy(c24, h24.data(), h24.size() * sizeof(float)).wait();
    stream->memcpy(c32, h32.data(), h32.size() * sizeof(float)).wait();
    const uint8_t * vxs[1] = { vx };
    float * cs24[1] = { c24 };
    float * cs32[1] = { c32 };
    const int64_t Ns[1] = { 32 };
    const int64_t Nd24[1] = { 24 };
    const int64_t Nd32[1] = { 32 };
    const bool ok24 = ggml_sycl_xe2mx_xmx_mg(ctx, vxs, cs24, Ns, Nd24, 1, A, da, K, M);
    stream->wait();
    const bool ok32 = ggml_sycl_xe2mx_xmx_mg(ctx, vxs, cs32, Ns, Nd32, 1, A, da, K, M);
    stream->wait();
    stream->memcpy(h24.data(), c24, h24.size() * sizeof(float)).wait();
    stream->memcpy(h32.data(), c32, h32.size() * sizeof(float)).wait();
    float maxdiff = 0.f;
    int sentinel = 0;
    for (int m = 0; m < (int) M; ++m) {
        for (int n = 0; n < 24; ++n) {
            const float a = h24[(size_t) m * 24 + (size_t) n];
            const float b = h32[(size_t) m * 32 + (size_t) n];
            maxdiff = std::max(maxdiff, fabsf(a - b));
            if (a == 12345.f) {
                sentinel++;
            }
        }
    }
    float tail = 0.f;
    for (int m = 0; m < (int) M; ++m) {
        for (int n = 24; n < 32; ++n) {
            tail = std::max(tail, fabsf(h32[(size_t) m * 32 + (size_t) n]));
        }
    }
    GGML_SYCL_XE2_LOG( "xe2mx mg pad TEST ok=%d/%d maxdiff=%g sentinel=%d tail=%g\n",
            (int) ok24, (int) ok32, maxdiff, sentinel, tail);
    fflush(stderr);
}

// MG: one launch for up to 4 XE2MX4 matmuls that share src1 (GDN qkv + gate + beta + alpha at the MTP verify
// batch). Later nodes are written directly when xe2mx_early_write_safe, else into a per-queue slot that lands when
// the graph loop reaches them. Off unless GGML_SYCL_XE2MX_MG=1.
int ggml_sycl_xe2mx_mg_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    static const int mg_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_MG", 0);
    if (!mg_on || !cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes || !g_ggml_sycl_enable_esimd) {
        return 0;
    }
    xe2mx_mg_pad_selftest(ctx);
    ggml_tensor * a = cgraph->nodes[node_idx];
    if (!a) {
        return 0;
    }
    for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
        if (g_xe2mx_fn_skip[s] == a) {
            xe2mx_fn_defer_slot * slots = xe2mx_fn_slots((const void *) ctx.stream());
            if (slots[s].bytes > 0) {
                ctx.stream()->memcpy(a->data, slots[s].buf, slots[s].bytes);
            }
            g_xe2mx_fn_landed++;
            return 1;
        }
    }
    if (a->op != GGML_OP_MUL_MAT || !a->src[0] || !a->src[1] || a->src[0]->type != GGML_TYPE_XE2MX4 ||
        a->type != GGML_TYPE_F32 || a->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(a->src[0]) || !ggml_is_contiguous(a->src[1]) || !ggml_is_contiguous(a)) {
        return 0;
    }
    static const int mg_minm = ggml_sycl_get_env("GGML_SYCL_XE2MX_MG_MINM", 2);
    static const int mg_maxm = ggml_sycl_get_env("GGML_SYCL_XE2MX_MG_MAXM", 2);
    static const int mg_max  = std::max(2, std::min(8, ggml_sycl_get_env("GGML_SYCL_XE2MX_MG_MAX", 8)));
    static const int mg_direct = ggml_sycl_get_env("GGML_SYCL_XE2MX_MG_DIRECT", 1);
    const int64_t K = a->src[0]->ne[0];
    const int64_t M = ggml_nrows(a->src[1]);
    if (M < mg_minm || M > mg_maxm || (K % 32) != 0 || (a->src[0]->ne[1] % 16) != 0 || a->ne[0] != a->src[0]->ne[1]) {
        return 0;
    }
    ggml_tensor * mem[8] = { a, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    int idx[8] = { node_idx, 0, 0, 0, 0, 0, 0, 0 };
    bool force_defer[8] = { false, false, false, false, false, false, false, false };
    int nm = 1;
    const int j1 = node_idx + 160 < cgraph->n_nodes ? node_idx + 160 : cgraph->n_nodes;
    for (int j = node_idx + 1; j < j1 && nm < mg_max; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || t->src[1] != a->src[1]) {
            continue;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[0]) ||
            !ggml_is_contiguous(t) || t->src[0]->ne[0] != K ||
            t->ne[0] != t->src[0]->ne[1]) {
            continue;
        }
        if ((t->src[0]->ne[1] % 16) != 0) {
            int64_t np = 0;
            const uint8_t * pv = nullptr;
            if (!xe2mx_mg_pad_ensure(ctx, t->src[0], &np, &pv)) {
                continue;
            }
        }
        bool dup = false;
        for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
            dup = dup || g_xe2mx_fn_skip[s] == t;
        }
        for (int s = 0; s < g_xe2mx_bt_nskip; ++s) {
            dup = dup || g_xe2mx_bt_skip[s] == t;
        }
        if (dup) {
            continue;
        }
        // Sharing bytes with an earlier member is how ggml-alloc reuses a dead output (alpha dies before beta);
        // such a member lands from its slot at its own node instead of being dropped.
        for (int m = 0; m < nm; ++m) {
            force_defer[nm] = force_defer[nm] || xe2mx_bytes_overlap(mem[m], t);
        }
        mem[nm] = t;
        idx[nm] = j;
        nm++;
    }
    if (nm < 2 || g_xe2mx_fn_nskip + nm - 1 > XE2MX_FN_SKIP_MAX) {
        return 0;
    }
    {
        static int ncap = 0;
        if (ncap < 6) {
            int avail = 1;
            for (int j = node_idx + 1; j < j1; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                if (t && t->op == GGML_OP_MUL_MAT && t->src[1] == a->src[1] && t->src[0] &&
                    t->src[0]->type == GGML_TYPE_XE2MX4 && t->src[0]->ne[0] == K) {
                    avail++;
                }
            }
            ncap++;
            GGML_SYCL_XE2_LOG( "xe2mx mg avail=%d took=%d M=%ld K=%ld win=%d\n",
                    avail, nm, (long) M, (long) K, j1 - node_idx);
            fflush(stderr);
        }
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_DEBUG", 0) >= 2) {
        static int nwin = 0;
        if (nwin < 3) {
            nwin++;
            for (int j = node_idx; j < j1; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
                    continue;
                }
                GGML_SYCL_XE2_LOG( "xe2mx mg win j=%d %s w=%s(%s) K=%ld N=%ld src1=%s%s cont=%d/%d\n", j, t->name,
                        t->src[0]->name, ggml_type_name(t->src[0]->type), (long) t->src[0]->ne[0],
                        (long) t->src[0]->ne[1], t->src[1]->name, t->src[1] == a->src[1] ? " [same]" : "",
                        (int) ggml_is_contiguous(t), (int) ggml_is_contiguous(t->src[1]));
            }
            fflush(stderr);
        }
    }
    int64_t Nphys[8] = {};
    const uint8_t * vxpad[8] = {};
    for (int m = 0; m < nm; ++m) {
        const ggml_tensor * w = mem[m]->src[0];
        if ((w->ne[1] % 16) == 0) {
            if (!xe2mx_pack_xmx(ctx, w)) {
                return 0;
            }
            Nphys[m] = w->ne[1];
            vxpad[m] = (const uint8_t *) w->data;
        } else if (!xe2mx_mg_pad_ensure(ctx, w, &Nphys[m], &vxpad[m])) {
            return 0;
        }
    }
    xe2mx_fn_defer_slot * slots = xe2mx_fn_slots((const void *) ctx.stream());
    const uint8_t * vx[8];
    float * C[8];
    int64_t N[8];
    int64_t Ndst[8];
    size_t slot_bytes[8] = {};
    for (int m = 0; m < nm; ++m) {
        vx[m] = vxpad[m];
        N[m] = Nphys[m];
        Ndst[m] = mem[m]->src[0]->ne[1];
        C[m] = (float *) mem[m]->data;
        if (m > 0 && (force_defer[m] || !(mg_direct && xe2mx_early_write_safe(cgraph, node_idx, idx[m], mem[m])))) {
            xe2mx_fn_defer_slot & slot = slots[g_xe2mx_fn_nskip + m - 1];
            const size_t bytes = (size_t) Ndst[m] * (size_t) M * sizeof(float);
            if (slot.cap < bytes) {
                if (slot.buf) {
                    ctx.stream()->wait();
                    ggml_sycl_free_device(slot.buf, *ctx.stream());
                    slot.buf = nullptr;
                    slot.cap = 0;
                }
                slot.buf = ggml_sycl_malloc_device(bytes, *ctx.stream());
                if (!slot.buf) {
                    return 0;
                }
                slot.cap = bytes;
            }
            C[m] = (float *) slot.buf;
            slot_bytes[m] = bytes;
        }
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    int8_t * qbA = nullptr;
    float * qbda = nullptr;
    if (!xe2mx_qbatch_hit(a->src[1]->data, M, K, &qbA, &qbda)) {
        Aq.alloc((size_t) M * (size_t) K);
        daq.alloc((size_t) M * (size_t) (K / 32));
        xe2mx_quant_a_f32(ctx, (const float *) a->src[1]->data, Aq.get(), daq.get(), M, K);
        qbA = Aq.get();
        qbda = daq.get();
    }
    if (!ggml_sycl_xe2mx_xmx_mg(ctx, vx, C, N, Ndst, nm, qbA, qbda, K, M)) {
        return 0;
    }
    {
        static int nmg = 0;
        if (nmg < 4) {
            nmg++;
            GGML_SYCL_XE2_LOG( "xe2mx mg nw=%d M=%ld K=%ld phys/dst=%ld/%ld,%ld/%ld,%ld/%ld,%ld/%ld\n",
                    nm, (long) M, (long) K,
                    (long) N[0], (long) Ndst[0],
                    nm > 1 ? (long) N[1] : 0L, nm > 1 ? (long) Ndst[1] : 0L,
                    nm > 2 ? (long) N[2] : 0L, nm > 2 ? (long) Ndst[2] : 0L,
                    nm > 3 ? (long) N[3] : 0L, nm > 3 ? (long) Ndst[3] : 0L);
            fflush(stderr);
        }
    }
    for (int m = 1; m < nm; ++m) {
        slots[g_xe2mx_fn_nskip].bytes = slot_bytes[m];
        g_xe2mx_fn_skip[g_xe2mx_fn_nskip++] = mem[m];
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_DEBUG", 0)) {
        static int nlog = 0;
        if (nlog < 4) {
            nlog++;
            GGML_SYCL_XE2_LOG( "xe2mx mg nw=%d M=%ld K=%ld N=%ld,%ld,%ld,%ld deferred=%d,%d,%d\n", nm, (long) M, (long) K,
                    (long) N[0], nm > 1 ? (long) N[1] : 0L, nm > 2 ? (long) N[2] : 0L, nm > 3 ? (long) N[3] : 0L,
                    slot_bytes[1] > 0, slot_bytes[2] > 0, slot_bytes[3] > 0);
            fflush(stderr);
        }
    }
    return 1;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
static bool xe2mx_bt_empty_op(const ggml_tensor * n) {
    return n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
}

static bool xe2mx_bt_reads(const ggml_tensor * n, const ggml_tensor * t) {
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (n->src[s] == t) {
            return true;
        }
    }
    return false;
}

// Old-path knobs that would send a small-M XE2MX4 MUL_MAT anywhere but the q32 launch_soa body.
static bool xe2mx_bt_env_ok(ggml_backend_sycl_context & ctx) {
    static const bool env_ok = [] {
        const int q32wg = ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32WG", 0);
        return ggml_sycl_get_env("GGML_SYCL_XE2MX_DPAS", 1) != 0 && ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32", 1) != 0 &&
               ggml_sycl_get_env("GGML_SYCL_XE2MX_F32A", 0) == 0 && ggml_sycl_get_env("GGML_SYCL_XE2MX_NT", 0) == 0 &&
               ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32KB", 8) != 16 && q32wg != 1 && q32wg != 2 && q32wg != 8 &&
               !(ggml_sycl_get_env("GGML_SYCL_XE2MX_F16P", 1) && ggml_sycl_get_env("GGML_SYCL_XE2MX_S4M", 1024) <= 8);
    }();
    return env_ok && g_ggml_sycl_enable_esimd &&
           ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[ctx.device].hw_info.arch);
}

// t would take the q32 launch_soa body in ggml_sycl_mul_mat_xe2mx, with 2 <= M <= 8 (and M == M_need if set).
static bool xe2mx_bt_shape_ok(ggml_backend_sycl_context & ctx, const ggml_tensor * t, int64_t M_need) {
    if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
        return false;
    }
    const ggml_tensor * w = t->src[0];
    const ggml_tensor * x = t->src[1];
    if (w->type != GGML_TYPE_XE2MX4 || x->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32) {
        return false;
    }
    if (!w->data || !x->data || !t->data || !w->buffer) {
        return false;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(t) || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    const int64_t M = ggml_nrows(x);
    if (M < 2 || M > 8 || (M_need > 0 && M != M_need)) {
        return false;
    }
    if (x->ne[0] != K || t->ne[0] != N || ggml_nrows(t) != M) {
        return false;
    }
    if ((K % 32) != 0 || (N % 16) != 0 || ((K / 32) % 8) != 0 || (K / 32) * 16 < 64) {
        return false;
    }
    if (g_xe2mx_lo_w.count((const void *) w->data) || xe2mx_s8dw_shape(K, N) || xe2mx_q32wg8_n_hit(N)) {
        return false;
    }
    if (!ggml_sycl_xe2mx_lo_bytes_are_soa(w)) {
        return false;
    }
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) w->extra;
    if (!extra || extra->optimized_feature.xe2mx_lo) {
        return false;
    }
    return extra->optimized_feature.reorder || xe2mx_pack_xmx(ctx, w);
}

// A member that a later fuse would take at its own node (GLU gate/up, FADD, ZSILU) or that MG/FN already owns.
static bool xe2mx_bt_fuse_owned(const ggml_cgraph * g, int j) {
    const ggml_tensor * t = g->nodes[j];
    for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
        if (g_xe2mx_fn_skip[s] == t) {
            return true;
        }
    }
    static const int fadd = ggml_sycl_get_env("GGML_SYCL_XE2MX_FADD", 0);
    static const int zsilu = ggml_sycl_get_env("GGML_SYCL_FUSE_ZSILU", 0);
    for (int d = 1; d <= 2 && j + d < g->n_nodes; ++d) {
        const ggml_tensor * n = g->nodes[j + d];
        if (!n || !xe2mx_bt_reads(n, t)) {
            continue;
        }
        if (n->op == GGML_OP_GLU) {
            return true;
        }
        if (d == 1 && fadd && n->op == GGML_OP_ADD) {
            return true;
        }
        if (d == 1 && zsilu && (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_UNARY)) {
            return true;
        }
    }
    return false;
}
#endif

// GGML_SYCL_XE2MX_BATCH=1 (default off). When node_idx is an XE2MX4 MUL_MAT with 2 <= M <= 8, gather up to
// GGML_SYCL_XE2MX_BATCH_MAX (<= 16) more XE2MX4 MUL_MATs with the same M from the next GGML_SYCL_XE2MX_BATCH_WIN
// (32) nodes and run them all in one ggml_sycl_xe2mx_xmx_batch launch. Members need not share src1. A member
// joins only when (a) no non-view node in [head, member) writes bytes of its src1, so src1 is final now, and
// (b) xe2mx_early_write_safe holds for its dst and it does not overlap another member's dst. Each distinct
// src1 is quantized once (qbatch hit if present). Members are recorded in g_xe2mx_bt_skip and this function
// returns 1 for them when the graph loop reaches them. Returns 1 when node_idx was handled, else 0.
int ggml_sycl_xe2mx_batch_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    static const int bt_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH", 0);
    if (!bt_on || !cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    if (!a || a->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
        if (g_xe2mx_fn_skip[s] == a) {
            return 0;  // MG/FN lands it
        }
    }
    for (int s = 0; s < g_xe2mx_bt_nskip; ++s) {
        if (g_xe2mx_bt_skip[s] == a) {
            return 1;
        }
    }
    if (!xe2mx_bt_env_ok(ctx) || !xe2mx_bt_shape_ok(ctx, a, 0)) {
        return 0;
    }
    static const int bt_win = std::max(1, std::min(256, ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH_WIN", 32)));
    static const int bt_max = std::max(2, std::min(XE2MX_BATCH_MAX, ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH_MAX", 16)));
    static const int bt_qsplit = ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH_QSPLIT", 0);
    static const int bt_log = ggml_sycl_get_env("GGML_SYCL_XE2MX_BATCH_LOG", 0);
    const int64_t M = ggml_nrows(a->src[1]);
    const ggml_backend_buffer_type_t wbt = ggml_backend_buffer_get_type(a->src[0]->buffer);

    ggml_tensor * mem[XE2MX_BATCH_MAX] = { a };
    int idx[XE2MX_BATCH_MAX] = { node_idx };
    int nm = 1;
    const int j1 = std::min(cgraph->n_nodes, node_idx + 1 + bt_win);
    for (int j = node_idx + 1; j < j1 && nm < bt_max; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || t->src[0]->type != GGML_TYPE_XE2MX4) {
            continue;
        }
        g_xe2mx_bt_st.seen++;
        if (!t->src[0]->buffer || ggml_backend_buffer_get_type(t->src[0]->buffer) != wbt ||
            !xe2mx_bt_shape_ok(ctx, t, M)) {
            g_xe2mx_bt_st.r_shape++;
            continue;
        }
        bool owned = xe2mx_bt_fuse_owned(cgraph, j);
        for (int s = 0; s < g_xe2mx_bt_nskip && !owned; ++s) {
            owned = g_xe2mx_bt_skip[s] == t;
        }
        if (owned) {
            g_xe2mx_bt_st.r_fuse++;
            continue;
        }
        bool ready = true;
        for (int k = node_idx; k < j && ready; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            ready = !n || xe2mx_bt_empty_op(n) || !xe2mx_bytes_overlap(n, t->src[1]);
        }
        if (!ready) {
            g_xe2mx_bt_st.r_src++;
            continue;
        }
        bool safe = xe2mx_early_write_safe(cgraph, node_idx, j, t);
        for (int m = 0; m < nm && safe; ++m) {
            safe = !xe2mx_bytes_overlap(mem[m], t);
        }
        if (!safe) {
            g_xe2mx_bt_st.r_dst++;
            continue;
        }
        mem[nm] = t;
        idx[nm] = j;
        nm++;
    }
    if (nm < 2 || g_xe2mx_bt_nskip + nm - 1 > XE2MX_BT_SKIP_MAX) {
        return 0;
    }

    // One quantized A per distinct (src1 bytes, K). Prefer the graph-wide qbatch copy.
    const int8_t * Au[XE2MX_BATCH_MAX] = {};
    const float * dau[XE2MX_BATCH_MAX] = {};
    const void * ukey[XE2MX_BATCH_MAX] = {};
    int64_t uK[XE2MX_BATCH_MAX] = {};
    int uq[XE2MX_BATCH_MAX] = {};  // 1: needs a quant here
    int umap[XE2MX_BATCH_MAX] = {};
    int nu = 0;
    size_t abytes = 0;
    size_t dfloats = 0;
    for (int m = 0; m < nm; ++m) {
        const void * key = mem[m]->src[1]->data;
        const int64_t K = mem[m]->src[0]->ne[0];
        int u = 0;
        while (u < nu && !(ukey[u] == key && uK[u] == K)) {
            u++;
        }
        if (u == nu) {
            ukey[u] = key;
            uK[u] = K;
            int8_t * qA = nullptr;
            float * qda = nullptr;
            if (xe2mx_qbatch_hit(key, M, K, &qA, &qda)) {
                Au[u] = qA;
                dau[u] = qda;
            } else {
                uq[u] = 1;
                abytes += (size_t) M * (size_t) K;
                dfloats += (size_t) M * (size_t) (K / 32);
            }
            nu++;
        }
        umap[m] = u;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    if (abytes > 0) {
        Aq.alloc(abytes);
        daq.alloc(dfloats);
        xe2mx_btq_args q = {};
        size_t ao = 0;
        size_t dofs = 0;
        int blk = 0;
        for (int u = 0; u < nu; ++u) {
            if (!uq[u]) {
                continue;
            }
            int8_t * A = Aq.get() + ao;
            float * da = daq.get() + dofs;
            Au[u] = A;
            dau[u] = da;
            ao += (size_t) M * (size_t) uK[u];
            dofs += (size_t) M * (size_t) (uK[u] / 32);
            if (bt_qsplit) {
                xe2mx_quant_a_f32(ctx, (const float *) ukey[u], A, da, M, uK[u]);
                continue;
            }
            blk += (int) (M * (uK[u] / 32));
            q.x[q.n] = (const float *) ukey[u];
            q.A[q.n] = A;
            q.da[q.n] = da;
            q.nb[q.n] = (int) (uK[u] / 32);
            q.K[q.n] = (int) uK[u];
            q.end[q.n] = blk;
            q.n++;
        }
        if (q.n == 1) {
            xe2mx_quant_a_f32(ctx, q.x[0], q.A[0], q.da[0], M, q.K[0]);
        } else if (q.n > 1) {
            for (int w = q.n; w < XE2MX_BATCH_MAX; ++w) {
                q.x[w] = q.x[0];
                q.A[w] = q.A[0];
                q.da[w] = q.da[0];
                q.nb[w] = q.nb[0];
                q.K[w] = q.K[0];
                q.end[w] = blk;
            }
            xe2mx_quant_a_f32_multi(ctx, q);
        }
    }

    xe2mx_batch_ent ents[XE2MX_BATCH_MAX];
    for (int m = 0; m < nm; ++m) {
        ents[m].vx = (const uint8_t *) mem[m]->src[0]->data;
        ents[m].A = Au[umap[m]];
        ents[m].da = dau[umap[m]];
        ents[m].C = (float *) mem[m]->data;
        ents[m].K = mem[m]->src[0]->ne[0];
        ents[m].N = mem[m]->src[0]->ne[1];
        ents[m].M = M;
    }
    if (!ggml_sycl_xe2mx_xmx_batch(ctx, ents, nm)) {
        return 0;
    }
    for (int m = 1; m < nm; ++m) {
        g_xe2mx_bt_skip[g_xe2mx_bt_nskip++] = mem[m];
    }
    g_xe2mx_bt_st.heads++;
    g_xe2mx_bt_st.members += nm;
    if (bt_log >= 2) {
        static int nlog = 0;
        if (nlog < 16) {
            nlog++;
            GGML_SYCL_XE2_LOG( "xe2mx batch head=%d nw=%d uniq_src1=%d M=%ld last=%d\n", node_idx, nm, nu, (long) M,
                    idx[nm - 1]);
            fflush(stderr);
        }
    }
    return 1;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

int ggml_sycl_xe2mx_fn_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_FN", 0) == 0) {
        return 0;
    }
    if (!cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return 0;
    }
    if (!g_ggml_sycl_enable_esimd) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    if (!a) {
        return 0;
    }
    for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
        if (g_xe2mx_fn_skip[s] == a) {
            // b was computed at a's slot into a private buffer; land it now that a->data is b's own memory.
            xe2mx_fn_defer_slot * slots = xe2mx_fn_slots((const void *) ctx.stream());
            if (slots[s].bytes > 0) {
                ctx.stream()->memcpy(a->data, slots[s].buf, slots[s].bytes);
            }
            g_xe2mx_fn_landed++;
            return 1;
        }
    }
    if (a->op != GGML_OP_MUL_MAT || !a->src[0] || !a->src[1]) {
        return 0;
    }
    if (a->src[0]->type != GGML_TYPE_XE2MX4 || a->type != GGML_TYPE_F32 || a->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(a->src[0]) || !ggml_is_contiguous(a->src[1]) || !ggml_is_contiguous(a)) {
        return 0;
    }
    const int64_t K = a->src[0]->ne[0];
    const int64_t N0 = a->src[0]->ne[1];
    const int64_t M = ggml_nrows(a->src[1]);
    // FN_MAXM: largest decode batch to fuse (2 = cheap MTP verify). Needs the deferred copy below for M > 1.
    static const int fn_maxm = ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_MAXM", 1);
    if (M < 1 || M > fn_maxm || (K % 32) != 0 || (N0 % 16) != 0) {
        return 0;
    }
    ggml_tensor * b = nullptr;
    ggml_tensor * bsmall = nullptr;
    const int j1 = node_idx + 64 < cgraph->n_nodes ? node_idx + 64 : cgraph->n_nodes;
    for (int j = node_idx + 1; j < j1; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
            continue;
        }
        if (t->src[1] != a->src[1] || t->src[0]->type != GGML_TYPE_XE2MX4 || t->type != GGML_TYPE_F32) {
            continue;
        }
        if (!ggml_is_contiguous(t->src[0]) || !ggml_is_contiguous(t) || t->data == a->data) {
            continue;
        }
        if (t->src[0]->ne[0] != K || (t->src[0]->ne[1] % 16) != 0 || ggml_nrows(t->src[1]) != M) {
            continue;
        }
        int seen = 0;
        for (int s = 0; s < g_xe2mx_fn_nskip; ++s) {
            if (g_xe2mx_fn_skip[s] == t) {
                seen = 1;
                break;
            }
        }
        for (int s = 0; s < g_xe2mx_bt_nskip && !seen; ++s) {
            seen = g_xe2mx_bt_skip[s] == t;
        }
        if (seen) {
            continue;
        }
        if (t->src[0]->ne[1] >= 256) {
            if (!b || t->src[0]->ne[1] > b->src[0]->ne[1]) {
                b = t;
            }
        } else if (!bsmall || t->src[0]->ne[1] > bsmall->src[0]->ne[1]) {
            bsmall = t;
        }
    }
    if (!b) {
        b = bsmall;
    }
    if (!b) {
        return 0;
    }
    const int64_t N1 = b->src[0]->ne[1];
    const int64_t nlo = N0 < N1 ? N0 : N1;
    const int64_t nhi = N0 < N1 ? N1 : N0;
    static const int64_t fn_minn = ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_MINN", 256);  // 2048 = GDN qkv+gate only
    if (nlo < fn_minn || nhi > nlo * 4 || a->ne[0] != N0 || b->ne[0] != N1) {
        return 0;
    }
    if (!xe2mx_pack_xmx(ctx, a->src[0]) || !xe2mx_pack_xmx(ctx, b->src[0])) {
        return 0;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    Aq.alloc((size_t) M * (size_t) K);
    daq.alloc((size_t) M * (size_t) (K / 32));
    xe2mx_quant_a_f32(ctx, (const float *) a->src[1]->data, Aq.get(), daq.get(), M, K);
    if (g_xe2mx_fn_nskip >= XE2MX_FN_SKIP_MAX) {
        return 0;
    }
    // b->data may still belong to tensors that are live between a and b (graph allocator reuse), so b's
    // rows go to a private slot and get copied into b->data when the loop reaches b. FN_DEFER=0 = old direct write.
    static const int fn_defer = ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_DEFER", 1);
    xe2mx_fn_defer_slot & slot = xe2mx_fn_slots((const void *) ctx.stream())[g_xe2mx_fn_nskip];
    const size_t bbytes = (size_t) N1 * (size_t) M * sizeof(float);
    float * c1 = (float *) b->data;
    if (fn_defer) {
        if (slot.cap < bbytes) {
            if (slot.buf) {
                ctx.stream()->wait();  // a queued landing copy may still read the old buffer
                ggml_sycl_free_device(slot.buf, *ctx.stream());
                slot.buf = nullptr;
                slot.cap = 0;
            }
            slot.buf = ggml_sycl_malloc_device(bbytes, *ctx.stream());
            if (!slot.buf) {
                return 0;
            }
            slot.cap = bbytes;
        }
        c1 = (float *) slot.buf;
    }
    if (!ggml_sycl_xe2mx_xmx_fn(ctx, (const uint8_t *) a->src[0]->data, (const uint8_t *) b->src[0]->data,
                                Aq.get(), daq.get(), (float *) a->data, c1, K, N0, N1, M)) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_FN_VERIFY", 0)) {
        static int nver = 0;
        if (nver < 6) {
            nver++;
            ggml_sycl_pool_alloc<float> r0(ctx.pool()), r1(ctx.pool());
            r0.alloc((size_t) N0 * M);
            r1.alloc((size_t) N1 * M);
            ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) a->src[0]->data, Aq.get(), daq.get(), r0.get(), K, N0, M);
            ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) b->src[0]->data, Aq.get(), daq.get(), r1.get(), K, N1, M);
            std::vector<float> h0((size_t) N0 * M), g0((size_t) N0 * M), h1((size_t) N1 * M), g1((size_t) N1 * M);
            ctx.stream()->memcpy(h0.data(), r0.get(), h0.size() * 4);
            ctx.stream()->memcpy(g0.data(), a->data, g0.size() * 4);
            ctx.stream()->memcpy(h1.data(), r1.get(), h1.size() * 4);
            ctx.stream()->memcpy(g1.data(), c1, g1.size() * 4).wait();
            double d0 = 0, d1 = 0;
            long n0 = 0, n1 = 0;
            for (size_t i = 0; i < h0.size(); ++i) { const double d = fabs((double) h0[i] - g0[i]); if (d > 0) { n0++; } if (d > d0) { d0 = d; } }
            for (size_t i = 0; i < h1.size(); ++i) { const double d = fabs((double) h1[i] - g1[i]); if (d > 0) { n1++; } if (d > d1) { d1 = d; } }
            GGML_SYCL_XE2_LOG( "xe2mx fn verify N0=%ld N1=%ld M=%ld K=%ld: a maxdiff %.3g (%ld/%zu) b maxdiff %.3g (%ld/%zu)\n",
                    (long) N0, (long) N1, (long) M, (long) K, d0, n0, h0.size(), d1, n1, h1.size());
            fflush(stderr);
        }
    }
    slot.bytes = fn_defer ? bbytes : 0;
    g_xe2mx_fn_skip[g_xe2mx_fn_nskip++] = b;
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0)) {
        static int nlogfn = 0;
        if (nlogfn < 8) {
            nlogfn++;
            GGML_SYCL_XE2_LOG( "xe2mx fn N0=%ld N1=%ld M=%ld K=%ld\n", (long) N0, (long) N1, (long) M, (long) K);
        }
    }
    return 1;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

int ggml_sycl_xe2mx_s4ovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S4OVL", 0) == 0) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0)) {
        return 0;
    }
    if (!cgraph || node_idx < 0 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    ggml_tensor * nodes[8];
    int nrun = 0;
    const ggml_tensor * act = nullptr;
    int64_t K0 = 0;
    int64_t M0 = 0;
    for (int j = node_idx; j < cgraph->n_nodes && nrun < 8; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32) {
            break;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || !ggml_is_contiguous(t->src[0]) ||
            !ggml_is_contiguous(t->src[1]) || !ggml_is_contiguous(t)) {
            break;
        }
        const int64_t M = ggml_nrows(t->src[1]);
        const int64_t K = t->src[0]->ne[0];
        const int64_t N = t->src[0]->ne[1];
        if (M < 1024 || N < s4n || (K % 32) != 0 || (N % 16) != 0) {
            break;
        }
        if (nrun == 0) {
            act = t->src[1];
            K0 = K;
            M0 = M;
        } else if (t->src[1] != act || K != K0 || M != M0) {
            break;
        }
        int alias = 0;
        for (int p = 0; p < nrun; ++p) {
            if (nodes[p]->data == t->data) {
                alias = 1;
                break;
            }
        }
        if (alias) {
            break;
        }
        nodes[nrun++] = t;
    }
    if (nrun < 2) {
        return 0;
    }
    for (int i = 0; i < nrun; ++i) {
        if (!xe2mx_pack_xmx(ctx, nodes[i]->src[0])) {
            return 0;
        }
    }
    queue_ptr main = ctx.stream();
    static sycl::queue * side = nullptr;
    static queue_ptr side_seen = nullptr;
    if (!side || side_seen != main) {
        delete side;
        side = new sycl::queue(main->get_context(), main->get_device(),
                                sycl::property_list{ sycl::property::queue::in_order() });
        side_seen = main;
    }
    const sycl::half * xh = nullptr;
    if (act->type == GGML_TYPE_F16) {
        xh = (const sycl::half *) act->data;
    } else {
        const size_t need = (size_t) M0 * (size_t) K0 * sizeof(sycl::half);
        if (g_xe2mx_xh_cap < need) {
            void * p = ggml_sycl_malloc_device(need, *main);
            if (!p) {
                return 0;
            }
            g_xe2mx_xh_buf = (sycl::half *) p;
            g_xe2mx_xh_cap = need;
        }
        const to_fp16_sycl_t to_x = xe2mx_get_to_x(act->type, nodes[0]);
        if (!to_x) {
            return 0;
        }
        to_x(act->data, g_xe2mx_xh_buf, M0 * K0, main);
        xh = g_xe2mx_xh_buf;
        g_xe2mx_xh_src = act;
        g_xe2mx_xh_M = M0;
        g_xe2mx_xh_K = K0;
        g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
    }
    main->wait();
    {
        dnnl::engine eng0 = ctx.engine_dnnl(main);
        dnnl::engine eng1 = ctx.engine_dnnl(side);
        dnnl::memory::desc sp({ 256 * 1024 * 1024 }, dnnl::memory::data_type::u8, { 1 });
        ctx.get_scratchpad_mem(sp, eng0, main);
        ctx.get_scratchpad_mem(sp, eng1, side);
        main->wait();
        side->wait();
    }
    {
        sycl::event evq = main->ext_oneapi_submit_barrier();
        side->ext_oneapi_submit_barrier({ evq });
    }
    int okall = 1;
    for (int i = 0; i < nrun; ++i) {
        ctx.ovl_q = (i & 1) ? side : nullptr;
        const int64_t N = nodes[i]->src[0]->ne[1];
        double t_deq = 0;
        double t_gemm = 0;
        if (!xe2mx_p0_u4(ctx, (const uint8_t *) nodes[i]->src[0]->data, xh, nullptr,
                         (float *) nodes[i]->data, K0, N, M0, 0, &t_deq, &t_gemm,
                         nullptr, nullptr, nodes[i]->src[1], nodes[i]->src[0])) {
            okall = 0;
            break;
        }
    }
    ctx.ovl_q = nullptr;
    side->wait();
    main->wait();
    if (!okall) {
        return 0;
    }
    return nrun;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

// LO dual-queue: consecutive MUL_MAT with LO weights + shared A.
// S4OVL used p0_u4 (wrong path). This calls xe2mx_p0_dnnl LO.
int ggml_sycl_xe2mx_loovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_LOOVL", 0) == 0) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0)) {
        return 0;
    }
    if (!cgraph || node_idx < 0 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    ggml_tensor * nodes[4];
    int nrun = 0;
    const ggml_tensor * act = nullptr;
    int64_t K0 = 0;
    int64_t M0 = 0;
    for (int j = node_idx; j < cgraph->n_nodes && nrun < 4; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32) {
            break;
        }
        if (t->src[0]->type != GGML_TYPE_XE2MX4 || !ggml_is_contiguous(t->src[0]) ||
            !ggml_is_contiguous(t->src[1]) || !ggml_is_contiguous(t)) {
            break;
        }
        if (!t->src[0]->data || !g_xe2mx_lo_w.count((const void *) t->src[0]->data)) {
            break;
        }
        const int64_t M = ggml_nrows(t->src[1]);
        const int64_t K = t->src[0]->ne[0];
        const int64_t N = t->src[0]->ne[1];
        if (M < 1024 || N < s4n || (K % 32) != 0 || (N % 16) != 0) {
            break;
        }
        if (t->src[1]->type != GGML_TYPE_F32) {
            break;
        }
        if (nrun == 0) {
            act = t->src[1];
            K0  = K;
            M0  = M;
        } else if (t->src[1] != act || K != K0 || M != M0) {
            break;
        }
        int alias = 0;
        for (int p = 0; p < nrun; ++p) {
            if (nodes[p]->data == t->data || nodes[p]->src[0]->data == t->src[0]->data) {
                alias = 1;
                break;
            }
        }
        if (alias) {
            break;
        }
        nodes[nrun++] = t;
    }
    if (nrun < 2) {
        return 0;
    }
    queue_ptr main = ctx.stream();
    static sycl::queue * side = nullptr;
    static queue_ptr side_seen = nullptr;
    if (!side || side_seen != main) {
        delete side;
        side = new sycl::queue(main->get_context(), main->get_device(),
                                sycl::property_list{ sycl::property::queue::in_order() });
        side_seen = main;
    }
    // Pack LO A + first gemm on main (fills shared LO A cache).
    ctx.ovl_q = nullptr;
    if (!xe2mx_p0_dnnl(ctx, nodes[0]->src[0], nodes[0]->src[1], nodes[0], 0, true)) {
        return 0;
    }
    main->wait();
    {
        dnnl::engine eng0 = ctx.engine_dnnl(main);
        dnnl::engine eng1 = ctx.engine_dnnl(side);
        dnnl::memory::desc sp({ 256 * 1024 * 1024 }, dnnl::memory::data_type::u8, { 1 });
        ctx.get_scratchpad_mem(sp, eng0, main);
        ctx.get_scratchpad_mem(sp, eng1, side);
        main->wait();
        side->wait();
    }
    {
        sycl::event evq = main->ext_oneapi_submit_barrier();
        side->ext_oneapi_submit_barrier({ evq });
    }
    int okall = 1;
    for (int i = 1; i < nrun; ++i) {
        ctx.ovl_q = (i & 1) ? side : nullptr;
        if (!xe2mx_p0_dnnl(ctx, nodes[i]->src[0], nodes[i]->src[1], nodes[i], 0, true)) {
            okall = 0;
            break;
        }
    }
    ctx.ovl_q = nullptr;
    side->wait();
    main->wait();
    if (!okall) {
        return 0;
    }
    static int nloovl = 0;
    if (nloovl < 4) {
        nloovl++;
        GGML_SYCL_XE2_LOG( "xe2mx loovl hit nrun=%d M=%ld K=%ld\n", nrun, (long) M0, (long) K0);
        fflush(stderr);
    }
    return nrun;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

static int g_xe2mx_q8dnnl_dead = 0;

#if GGML_SYCL_DNNL
static bool xe2mx_q8_dnnl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    if (ggml_sycl_get_env("GGML_SYCL_Q8_DNNL", 0) == 0 || g_xe2mx_q8dnnl_dead) {
        return false;
    }
    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(src1);
    if (M < 1 || M >= 32 || N < 65536) {
        return false;
    }
    if ((K % 32) != 0 || (N % 16) != 0) {
        return false;
    }
    const int64_t nb = K / 32;
    const size_t qn = (size_t) K * (size_t) N;
    const size_t sn = (size_t) nb * (size_t) N * sizeof(float);
    const size_t bytes = qn + sn;
    queue_ptr stream = ctx.stream();
    int8_t * b8 = nullptr;
    float * sw = nullptr;
    auto wcit = g_xe2mx_wcache.find((const void *) src0->data);
    if (wcit != g_xe2mx_wcache.end() && wcit->second.vnni == 14
        && wcit->second.bytes == bytes && wcit->second.K == K && wcit->second.N == N) {
        b8 = (int8_t *) wcit->second.packed;
        sw = (float *) (b8 + qn);
    } else if (bytes > (1ull << 20) && g_xe2mx_wcache_used + bytes <= XE2MX_WCACHE_CAP
               && !g_xe2mx_wcache.count((const void *) src0->data)) {
        void * persist = ggml_sycl_malloc_device(bytes, *stream);
        if (persist) {
            b8 = (int8_t *) persist;
            sw = (float *) (b8 + qn);
            const block_q8_0 * src = (const block_q8_0 *) src0->data;
            stream->parallel_for(sycl::range<1>((size_t) N * (size_t) nb), [=](sycl::id<1> gid) {
                const int64_t t = (int64_t) gid[0];
                const int n = (int) (t / nb);
                const int ib = (int) (t % nb);
                const block_q8_0 * blk = src + (size_t) n * (size_t) nb + (size_t) ib;
                int8_t * q = b8 + (size_t) n * (size_t) K + (size_t) ib * 32;
                for (int j = 0; j < 32; ++j) {
                    q[j] = blk->qs[j];
                }
                sw[(size_t) n * (size_t) nb + (size_t) ib] =
                    (float) (*reinterpret_cast<const sycl::half *>(&blk->d));
            });
            g_xe2mx_wcache_used += bytes;
            g_xe2mx_wcache[(const void *) src0->data] = { persist, bytes, K, N, 14 };
        }
    }
    if (!b8 || !sw) {
        return false;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    Aq.alloc((size_t) M * (size_t) K);
    daq.alloc((size_t) M * (size_t) nb);
    xe2mx_quant_a_f32(ctx, (const float *) src1->data, Aq.get(), daq.get(), M, K);
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
        dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8, {1, K});
        dnnl::memory::desc c_md({M, N}, dnnl::memory::data_type::f32, {N, 1});
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {1, 32}, dnnl::memory::data_type::f32);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {32, 1}, dnnl::memory::data_type::f32);
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        dnnl::memory a_mem(a_md, eng, Aq.get());
        dnnl::memory b_mem(b_md, eng, b8);
        dnnl::memory c_mem(c_md, eng, (float *) dst->data);
        dnnl::memory sa_mem({{M, nb}, dnnl::memory::data_type::f32, {nb, 1}}, eng, daq.get());
        dnnl::memory sw_mem({{nb, N}, dnnl::memory::data_type::f32, {1, nb}}, eng, sw);
        dnnl::memory sp_mem = ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, stream);
        dnnl::matmul prim(pd);
        prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sa_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sw_mem},
            {DNNL_ARG_SCRATCHPAD, sp_mem},
        });
        return true;
    } catch (const dnnl::error & e) {
        g_xe2mx_q8dnnl_dead = 1;
        GGML_SYCL_XE2_LOG( "xe2mx q8dnnl: %s\n", e.what());
        fflush(stderr);
        return false;
    } catch (const sycl::exception & e) {
        g_xe2mx_q8dnnl_dead = 1;
        GGML_SYCL_XE2_LOG( "xe2mx q8dnnl sycl: %s\n", e.what());
        fflush(stderr);
        return false;
    }
}
#endif

bool ggml_sycl_q8_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       ggml_tensor * dst) {
#if GGML_SYCL_DNNL
    if (xe2mx_q8_dnnl(ctx, src0, src1, dst)) {
        return true;
    }
#endif
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    if (ggml_sycl_get_env("GGML_SYCL_Q8_XMX", 0) == 0) {
        return false;
    }
    if (!g_ggml_sycl_enable_esimd) {
        return false;
    }
    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(src1);
    if (M < 1 || M >= 32 || N < 65536) {
        return false;
    }
    if ((K % 32) != 0 || (N % 16) != 0) {
        return false;
    }
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
    if (!extra || !extra->optimized_feature.reorder) {
        return false;
    }
    ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
    ggml_sycl_pool_alloc<float> daq(ctx.pool());
    Aq.alloc((size_t) M * (size_t) K);
    daq.alloc((size_t) M * (size_t) (K / 32));
    xe2mx_quant_a_f32(ctx, (const float *) src1->data, Aq.get(), daq.get(), M, K);
    return ggml_sycl_xe2mx_gemm_q8(ctx, src0->data, Aq.get(), daq.get(), (float *) dst->data, K, N, M);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}

#if GGML_SYCL_DNNL
struct xe2mx_gmlp_part {
    dnnl::graph::compiled_partition cp;
    std::vector<dnnl::graph::logical_tensor> ins;
    dnnl::graph::logical_tensor out;
    size_t id_src = 0;
    size_t id_wg = 0;
    size_t id_sg = 0;
    size_t id_zg = 0;
    size_t id_wu = 0;
    size_t id_su = 0;
    size_t id_zu = 0;
    size_t id_wd = 0;
    size_t id_sd = 0;
    size_t id_zd = 0;
    int ok = 0;
};

static std::unordered_map<xe2mx_s4_key, xe2mx_gmlp_part, xe2mx_s4_key_hash> g_xe2mx_gmlp;
static uint8_t * g_xe2mx_gmlp_zp = nullptr;
static size_t g_xe2mx_gmlp_zp_cap = 0;
static sycl::half * g_xe2mx_gmlp_ho = nullptr;
static size_t g_xe2mx_gmlp_ho_cap = 0;

static uint8_t * xe2mx_gmlp_zp8(queue_ptr stream, size_t n) {
    if (g_xe2mx_gmlp_zp_cap < n) {
        const size_t cap = n < (4ull << 20) ? (4ull << 20) : n;
        void * p = ggml_sycl_malloc_device(cap, *stream);
        if (!p) {
            return nullptr;
        }
        stream->fill((uint8_t *) p, (uint8_t) 8, cap);
        g_xe2mx_gmlp_zp = (uint8_t *) p;
        g_xe2mx_gmlp_zp_cap = cap;
    }
    return g_xe2mx_gmlp_zp;
}


static bool xe2mx_gmlp_run(ggml_backend_sycl_context & ctx, const uint8_t * wg, const uint8_t * wu,
                            const uint8_t * wd, const sycl::half * xh, float * C, int64_t K, int64_t N, int64_t M,
                            int time_on, double * t_gemm) {
    using dt = dnnl::graph::logical_tensor::data_type;
    using pt = dnnl::graph::logical_tensor::property_type;
    if (g_xe2mx_s4gmlp_dead || xh == nullptr || wg == nullptr || wu == nullptr || wd == nullptr || C == nullptr) {
        return false;
    }
    if ((K % 32) != 0 || (N % 32) != 0 || M < 1 || N < 1) {
        return false;
    }
    const int64_t nb = K / 32;
    const int64_t nbd = N / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const size_t qpd = (size_t) nbd * 16u;
    const uint8_t * qg = wg;
    const uint8_t * qu = wu;
    const uint8_t * qd = wd;
    const sycl::half * dg = (const sycl::half *) (wg + (size_t) N * qpitch);
    const sycl::half * du = (const sycl::half *) (wu + (size_t) N * qpitch);
    const sycl::half * dd = (const sycl::half *) (wd + (size_t) K * qpd);
    queue_ptr stream = ctx.stream();
    uint8_t * zpp = xe2mx_gmlp_zp8(stream, (size_t) nb * (size_t) N);
    if (!zpp) {
        return false;
    }
    xe2mx_s4_key key{ (int) M, (int) N, (int) K, 2, (uintptr_t) stream };
    xe2mx_gmlp_part * pe = nullptr;
    auto pit = g_xe2mx_gmlp.find(key);
    if (pit == g_xe2mx_gmlp.end()) {
        xe2mx_gmlp_part e;
        try {
            dnnl::engine eng = ctx.engine_dnnl(stream);
            size_t id = 0;
            const dnnl::graph::logical_tensor::dims wei_sz = {K, N};
            const dnnl::graph::logical_tensor::dims wei_st = {1, K};
            const dnnl::graph::logical_tensor::dims sc_sz = {nb, N};
            const dnnl::graph::logical_tensor::dims sc_st = {N, 1};
            const dnnl::graph::logical_tensor::dims src_sz = {M, K};
            const dnnl::graph::logical_tensor::dims src_st = {K, 1};
            const dnnl::graph::logical_tensor::dims hd_sz = {M, N};
            const dnnl::graph::logical_tensor::dims hd_st = {N, 1};
            auto wei0 = dnnl::graph::logical_tensor(id++, dt::u4, wei_sz, wei_st, pt::constant);
            auto sc0 = dnnl::graph::logical_tensor(id++, dt::f16, sc_sz, sc_st, pt::constant);
            auto zp0 = dnnl::graph::logical_tensor(id++, dt::u8, sc_sz, sc_st, pt::constant);
            auto wei0d = dnnl::graph::logical_tensor(id++, dt::f16, wei_sz, wei_st);
            auto deq_g = dnnl::graph::op(id++, dnnl::graph::op::kind::DynamicDequantize, "deq_g");
            deq_g.set_attr<std::string>(dnnl::graph::op::attr::qtype, "per_group");
            deq_g.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::group_shape, {32, 1});
            deq_g.set_attr<int64_t>(dnnl::graph::op::attr::axis, 0);
            deq_g.add_inputs({wei0, sc0, zp0});
            deq_g.add_outputs({wei0d});
            auto src = dnnl::graph::logical_tensor(id++, dt::f16, src_sz, src_st);
            auto out0 = dnnl::graph::logical_tensor(id++, dt::f16, hd_sz, hd_st);
            auto fc_g = dnnl::graph::op(id++, dnnl::graph::op::kind::MatMul, "fc_g");
            fc_g.add_inputs({src, wei0d});
            fc_g.add_outputs({out0});
            auto wei1 = dnnl::graph::logical_tensor(id++, dt::u4, wei_sz, wei_st, pt::constant);
            auto sc1 = dnnl::graph::logical_tensor(id++, dt::f16, sc_sz, sc_st, pt::constant);
            auto zp1 = dnnl::graph::logical_tensor(id++, dt::u8, sc_sz, sc_st, pt::constant);
            auto wei1d = dnnl::graph::logical_tensor(id++, dt::f16, wei_sz, wei_st);
            auto deq_u = dnnl::graph::op(id++, dnnl::graph::op::kind::DynamicDequantize, "deq_u");
            deq_u.set_attr<std::string>(dnnl::graph::op::attr::qtype, "per_group");
            deq_u.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::group_shape, {32, 1});
            deq_u.set_attr<int64_t>(dnnl::graph::op::attr::axis, 0);
            deq_u.add_inputs({wei1, sc1, zp1});
            deq_u.add_outputs({wei1d});
            auto out1 = dnnl::graph::logical_tensor(id++, dt::f16, hd_sz, hd_st);
            auto fc_u = dnnl::graph::op(id++, dnnl::graph::op::kind::MatMul, "fc_u");
            fc_u.add_inputs({src, wei1d});
            fc_u.add_outputs({out1});
            auto out2 = dnnl::graph::logical_tensor(id++, dt::f16, hd_sz, hd_st);
            auto sig = dnnl::graph::op(id++, dnnl::graph::op::kind::Sigmoid, "sig");
            sig.add_inputs({out0});
            sig.add_outputs({out2});
            auto out3 = dnnl::graph::logical_tensor(id++, dt::f16, hd_sz, hd_st);
            auto swm = dnnl::graph::op(id++, dnnl::graph::op::kind::Multiply, "swm");
            swm.add_inputs({out0, out2});
            swm.add_outputs({out3});
            auto out4 = dnnl::graph::logical_tensor(id++, dt::f16, hd_sz, hd_st);
            auto mul = dnnl::graph::op(id++, dnnl::graph::op::kind::Multiply, "mul");
            mul.add_inputs({out3, out1});
            mul.add_outputs({out4});
            const dnnl::graph::logical_tensor::dims down_sz = {N, K};
            const dnnl::graph::logical_tensor::dims down_st = {1, N};
            const dnnl::graph::logical_tensor::dims scd_sz = {nbd, K};
            const dnnl::graph::logical_tensor::dims scd_st = {K, 1};
            const dnnl::graph::logical_tensor::dims out_sz = {M, K};
            const dnnl::graph::logical_tensor::dims out_st = {K, 1};
            auto wei2 = dnnl::graph::logical_tensor(id++, dt::u4, down_sz, down_st, pt::constant);
            auto sc2 = dnnl::graph::logical_tensor(id++, dt::f16, scd_sz, scd_st, pt::constant);
            auto zp2 = dnnl::graph::logical_tensor(id++, dt::u8, scd_sz, scd_st, pt::constant);
            auto wei2d = dnnl::graph::logical_tensor(id++, dt::f16, down_sz, down_st);
            auto deq_d = dnnl::graph::op(id++, dnnl::graph::op::kind::DynamicDequantize, "deq_d");
            deq_d.set_attr<std::string>(dnnl::graph::op::attr::qtype, "per_group");
            deq_d.set_attr<dnnl::graph::logical_tensor::dims>(dnnl::graph::op::attr::group_shape, {32, 1});
            deq_d.set_attr<int64_t>(dnnl::graph::op::attr::axis, 0);
            deq_d.add_inputs({wei2, sc2, zp2});
            deq_d.add_outputs({wei2d});
            auto dst = dnnl::graph::logical_tensor(id++, dt::f16, out_sz, out_st);
            auto fc_d = dnnl::graph::op(id++, dnnl::graph::op::kind::MatMul, "fc_d");
            fc_d.add_inputs({out4, wei2d});
            fc_d.add_outputs({dst});
            dnnl::graph::graph g(eng.get_kind());
            g.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            g.add_op(deq_g);
            g.add_op(deq_u);
            g.add_op(fc_g);
            g.add_op(fc_u);
            g.add_op(sig);
            g.add_op(swm);
            g.add_op(mul);
            g.add_op(deq_d);
            g.add_op(fc_d);
            g.finalize();
            auto parts = g.get_partitions();
            static int nlog = 0;
            if (time_on && nlog < 2) {
                nlog++;
                GGML_SYCL_XE2_LOG( "xe2mx gmlp npart=%zu supp=%d M=%ld N=%ld K=%ld\n",
                        parts.size(), (parts.size() == 1 && parts[0].is_supported()) ? 1 : 0,
                        (long) M, (long) N, (long) K);
                fflush(stderr);
            }
            if (parts.size() != 1 || !parts[0].is_supported()) {
                g_xe2mx_s4gmlp_dead = 1;
                g_xe2mx_gmlp.insert({ key, std::move(e) });
                return false;
            }
            e.ins = parts[0].get_input_ports();
            e.out = parts[0].get_output_ports()[0];
            e.cp = parts[0].compile(e.ins, {e.out}, eng);
            e.out = e.cp.query_logical_tensor(e.out.get_id());
            e.id_src = src.get_id();
            e.id_wg = wei0.get_id();
            e.id_sg = sc0.get_id();
            e.id_zg = zp0.get_id();
            e.id_wu = wei1.get_id();
            e.id_su = sc1.get_id();
            e.id_zu = zp1.get_id();
            e.id_wd = wei2.get_id();
            e.id_sd = sc2.get_id();
            e.id_zd = zp2.get_id();
            e.ok = 1;
            pe = &g_xe2mx_gmlp.insert({ key, std::move(e) }).first->second;
        } catch (const std::exception & ex) {
            static int nerr = 0;
            if (nerr < 4) {
                nerr++;
                GGML_SYCL_XE2_LOG( "xe2mx gmlp build: %s\n", ex.what());
                fflush(stderr);
            }
            g_xe2mx_s4gmlp_dead = 1;
            return false;
        }
    } else {
        pe = &pit->second;
    }
    if (!pe || !pe->ok) {
        return false;
    }
    try {
        dnnl::engine eng = ctx.engine_dnnl(stream);
        dnnl::stream strm = ctx.stream_dnnl(stream);
        auto id2ptr = [&](size_t r) -> void * {
            if (r == pe->id_src) {
                return const_cast<sycl::half *>(xh);
            }
            if (r == pe->id_wg) {
                return const_cast<uint8_t *>(qg);
            }
            if (r == pe->id_sg) {
                return const_cast<sycl::half *>(dg);
            }
            if (r == pe->id_zg) {
                return zpp;
            }
            if (r == pe->id_wu) {
                return const_cast<uint8_t *>(qu);
            }
            if (r == pe->id_su) {
                return const_cast<sycl::half *>(du);
            }
            if (r == pe->id_zu) {
                return zpp;
            }
            if (r == pe->id_wd) {
                return const_cast<uint8_t *>(qd);
            }
            if (r == pe->id_sd) {
                return const_cast<sycl::half *>(dd);
            }
            if (r == pe->id_zd) {
                return zpp;
            }
            return nullptr;
        };
        std::vector<dnnl::graph::tensor> ti;
        ti.reserve(pe->ins.size());
        for (auto & t : pe->ins) {
            void * p = id2ptr(t.get_id());
            if (!p) {
                g_xe2mx_s4gmlp_dead = 1;
                return false;
            }
            ti.emplace_back(t, eng, p);
        }
        ggml_sycl_pool_alloc<sycl::half> Ho(ctx.pool());
        sycl::half * ho = nullptr;
        const size_t hob = (size_t) M * (size_t) K * sizeof(sycl::half);
        if (g_xe2mx_gmlp_ho_cap < hob) {
            void * p = ggml_sycl_malloc_device(hob, *stream);
            if (p) {
                g_xe2mx_gmlp_ho = (sycl::half *) p;
                g_xe2mx_gmlp_ho_cap = hob;
            }
        }
        if (g_xe2mx_gmlp_ho && g_xe2mx_gmlp_ho_cap >= hob) {
            ho = g_xe2mx_gmlp_ho;
        } else {
            ho = Ho.alloc((size_t) M * (size_t) K);
        }
        dnnl::graph::tensor to(pe->out, eng, ho);
        std::chrono::steady_clock::time_point t0;
        xe2mx_time_begin(time_on, stream, &t0);
        pe->cp.execute(strm, ti, {to});
        float * dstf = C;
        const int64_t nn = M * K;
        stream->parallel_for(sycl::range<1>((size_t) nn), [=](sycl::id<1> gid) {
            dstf[gid[0]] = (float) ho[gid[0]];
        });
        *t_gemm = xe2mx_time_ms(time_on, stream, t0);
        return true;
    } catch (const std::exception & ex) {
        static int nerr = 0;
        if (nerr < 4) {
            nerr++;
            GGML_SYCL_XE2_LOG( "xe2mx gmlp exec: %s\n", ex.what());
            fflush(stderr);
        }
        g_xe2mx_s4gmlp_dead = 1;
        try {
            stream->wait();
        } catch (...) {
        }
        return false;
    }
}
#endif

#if GGML_SYCL_DNNL
static bool xe2mx_s4t_glu_run(ggml_backend_sycl_context & ctx, const uint8_t * wg, const uint8_t * wu,
                               const sycl::half * xh, float * C, int64_t K, int64_t N, int64_t M,
                               int time_on, double * t_deq, double * t_gemm) {
    if (!xh || !wg || !wu || !C) {
        return false;
    }
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S4T", 0) || g_xe2mx_s4t_dead) {
        return false;
    }
    static int dual_dead = 0;
    if (!ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TDUAL", 0) || dual_dead) {
        return false;
    }
    if (M < 32 || N < ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TN", 17408) || (K % 32) != 0 || (N % 16) != 0) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    xe2mx_s8r_view gv{};
    xe2mx_s8r_view uv{};
    std::chrono::steady_clock::time_point t0;
    xe2mx_time_begin(time_on, stream, &t0);
    if (!xe2mx_s4t_cached(stream, wg, K, N, &gv) || !xe2mx_s4t_cached(stream, wu, K, N, &uv)) {
        return false;
    }
    const size_t an = (size_t) M * (size_t) K;
    int8_t * Ap = nullptr;
    float * sap = nullptr;
    ggml_sycl_pool_alloc<int8_t> As(ctx.pool());
    ggml_sycl_pool_alloc<float> sar(ctx.pool());
    ggml_sycl_pool_alloc<float> tmx(ctx.pool());
    if (g_xe2mx_s8ra && g_xe2mx_s8rsa && g_xe2mx_s8rxh == xh
        && g_xe2mx_s8rM == M && g_xe2mx_s8rK == K && g_xe2mx_s8r_epoch == g_xe2mx_xh_epoch
        && g_xe2mx_s8ra_cap >= an) {
        Ap = g_xe2mx_s8ra;
        sap = g_xe2mx_s8rsa;
    } else {
        if (g_xe2mx_s8ra_cap < an) {
            void * p = ggml_sycl_malloc_device(an, *stream);
            if (p) {
                g_xe2mx_s8ra = (int8_t *) p;
                g_xe2mx_s8ra_cap = an;
            }
        }
        if (!g_xe2mx_s8rsa) {
            void * p = ggml_sycl_malloc_device(sizeof(float), *stream);
            if (p) {
                g_xe2mx_s8rsa = (float *) p;
            }
        }
        if (g_xe2mx_s8ra && g_xe2mx_s8rsa && g_xe2mx_s8ra_cap >= an) {
            Ap = g_xe2mx_s8ra;
            sap = g_xe2mx_s8rsa;
        }
        if (!Ap) {
            Ap = As.alloc(an);
        }
        if (!sap) {
            sap = sar.alloc(1);
        }
        float * tmp = tmx.alloc((size_t) M);
        xe2mx_quant_a_one(stream, xh, Ap, sap, tmp, M, K);
        if (Ap == g_xe2mx_s8ra) {
            g_xe2mx_s8rxh = xh;
            g_xe2mx_s8rM = M;
            g_xe2mx_s8rK = K;
            g_xe2mx_s8r_epoch = g_xe2mx_xh_epoch;
        }
    }
    *t_deq = xe2mx_time_ms(time_on, stream, t0);
    if (!Ap || !sap || !gv.w || !gv.sc || !uv.w || !uv.sc) {
        return false;
    }
    xe2mx_time_begin(time_on, stream, &t0);
    if (!ggml_sycl_xe2mx_gemm_s4t_glu(ctx, (const uint8_t *) gv.w, gv.sc, (const uint8_t *) uv.w, uv.sc,
                                       Ap, sap, C, K, N, M)) {
        dual_dead = 1;
        return false;
    }
    *t_gemm = xe2mx_time_ms(time_on, stream, t0);
    if (time_on) {
        static int nfire = 0;
        if (nfire < 4) {
            nfire++;
            GGML_SYCL_XE2_LOG( "xe2mx s4t glu dual=%.2f ms N=%ld M=%ld K=%ld\n",
                    *t_gemm, (long) N, (long) M, (long) K);
            fflush(stderr);
        }
    }
    return true;
}
#endif

bool ggml_sycl_xe2mx_mul_mat_glu(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }
    const int glu_s4 = ggml_sycl_get_env("GGML_SYCL_XE2MX_GLU", 0);
    const int dglu = ggml_sycl_get_env("GGML_SYCL_XE2MX_DGLU", 0);
    const int s4glu = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4GLU", 1);
    if (glu_s4 == 0 && dglu == 0 && s4glu == 0) {
        return false;
    }
    if (!cgraph || node_idx + 2 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    ggml_tensor * b = cgraph->nodes[node_idx + 1];
    ggml_tensor * glu = cgraph->nodes[node_idx + 2];
    if (!a || !b || !glu) {
        return false;
    }
    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU) {
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    if (ggml_get_op_params_i32(glu, 1)) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { node_idx + 2 })) {
        return false;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a))) {
        return false;
    }
    const ggml_tensor * wg = gate->src[0];
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * act = up->src[1];
    if (!wg || !wu || !act || act != gate->src[1]) {
        return false;
    }
    if (wg->type != GGML_TYPE_XE2MX4 || wu->type != GGML_TYPE_XE2MX4) {
        return false;
    }
    if (act->type != GGML_TYPE_F32 && act->type != GGML_TYPE_F16) {
        return false;
    }
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !ggml_is_contiguous(act)) {
        return false;
    }
    if (!ggml_is_contiguous(wg) || !ggml_is_contiguous(wu)) {
        return false;
    }
    const int64_t K = wg->ne[0];
    const int64_t N = wg->ne[1];
    const int64_t M = ggml_nrows(act);
    if (wu->ne[0] != K || wu->ne[1] != N) {
        return false;
    }
    if (glu->ne[0] != N || glu->ne[1] != act->ne[1]) {
        return false;
    }
#ifdef GGML_SYCL_XE2MX_HAS_ESIMD
    // Dual LO keeps SOA qs+dpl in t->data (FGLU/DGLU). Inline LO has s4 pack - skip XMX.
    const int glu_inline = ggml_sycl_xe2mx_lo_inline_on(wg) || ggml_sycl_xe2mx_lo_inline_on(wu);
    if (!glu_inline && ggml_sycl_get_env("GGML_SYCL_XE2MX_FGLU", 1) && M >= 1 && M < 32 && act->type == GGML_TYPE_F32
        && g_ggml_sycl_enable_esimd && (K % QK_XE2MX4) == 0 && (N % 16) == 0) {
        if (xe2mx_pack_xmx(ctx, wg) && xe2mx_pack_xmx(ctx, wu)) {
            ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
            ggml_sycl_pool_alloc<float> daq(ctx.pool());
            int8_t * qbA = nullptr;
            float * qbda = nullptr;
            if (!xe2mx_qbatch_hit(act->data, M, K, &qbA, &qbda)) {
                Aq.alloc((size_t) M * (size_t) K);
                daq.alloc((size_t) M * (size_t) (K / 32));
                xe2mx_quant_a_f32(ctx, (const float *) act->data, Aq.get(), daq.get(), M, K);
                qbA = Aq.get();
                qbda = daq.get();
            }
            if (ggml_sycl_xe2mx_xmx_glu(ctx, (const uint8_t *) wg->data, (const uint8_t *) wu->data,
                                         qbA, qbda, (float *) glu->data, K, N, M)) {
                return true;
            }
        }
    }
    if (!glu_inline && dglu && M >= 1 && M < 32 && act->type == GGML_TYPE_F32 && g_ggml_sycl_enable_esimd
        && (K % QK_XE2MX4) == 0 && (N % 16) == 0) {
        if (xe2mx_pack_xmx(ctx, wg) && xe2mx_pack_xmx(ctx, wu)) {
            ggml_sycl_pool_alloc<int8_t> Aq(ctx.pool());
            ggml_sycl_pool_alloc<float> daq(ctx.pool());
            ggml_sycl_pool_alloc<float> Cg(ctx.pool());
            ggml_sycl_pool_alloc<float> Cu(ctx.pool());
            Aq.alloc((size_t) M * (size_t) K);
            daq.alloc((size_t) M * (size_t) (K / 32));
            Cg.alloc((size_t) M * (size_t) N);
            Cu.alloc((size_t) M * (size_t) N);
            xe2mx_quant_a_f32(ctx, (const float *) act->data, Aq.get(), daq.get(), M, K);
            const bool okg = ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) wg->data, Aq.get(), daq.get(),
                                                   Cg.get(), K, N, M);
            const bool oku = ggml_sycl_xe2mx_xmx(ctx, (const uint8_t *) wu->data, Aq.get(), daq.get(),
                                                   Cu.get(), K, N, M);
            if (okg && oku) {
                float * dst = (float *) glu->data;
                const float * pg = Cg.get();
                const float * pu = Cu.get();
                const int64_t nn = M * N;
                ctx.stream()->parallel_for(sycl::range<1>((size_t) nn), [=](sycl::id<1> gid) {
                    const int64_t i = (int64_t) gid[0];
                    const float g = pg[i];
                    dst[i] = (g / (1.0f + sycl::exp(-g))) * pu[i];
                });
                return true;
            }
        }
    }
#endif
    if (s4glu && !g_xe2mx_s4glu_dead) {
        const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
        if (M >= 1024 && N >= s4n && (K % QK_XE2MX4) == 0 && (N % 16) == 0
            && xe2mx_pack_xmx(ctx, wg) && xe2mx_pack_xmx(ctx, wu)) {
            queue_ptr stream = ctx.stream();
            ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
            const sycl::half * xh = nullptr;
            const float * xf_act = nullptr;
            const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
            const int glu_lo = (wu->data && g_xe2mx_lo_w.count((const void *) wu->data))
                || (wg->data && g_xe2mx_lo_w.count((const void *) wg->data));
            if (act->type == GGML_TYPE_F16) {
                xh = (const sycl::half *) act->data;
            } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_src == act && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
                       && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
                xh = g_xe2mx_xh_buf;
            } else if (glu_lo && act->type == GGML_TYPE_F32 && ggml_is_contiguous(act)) {
                xf_act = (const float *) act->data;
            } else {
                const to_fp16_sycl_t to_x = xe2mx_get_to_x(act->type, glu);
                if (to_x) {
                    sycl::half * xdst = nullptr;
                    if (xcache) {
                        const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
                        if (g_xe2mx_xh_cap < need) {
                            void * p = ggml_sycl_malloc_device(need, *stream);
                            if (p) {
                                g_xe2mx_xh_buf = (sycl::half *) p;
                                g_xe2mx_xh_cap = need;
                            }
                        }
                        if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                            xdst = g_xe2mx_xh_buf;
                        }
                    }
                    if (!xdst) {
                        X.alloc((size_t) M * (size_t) K);
                        xdst = X.get();
                    }
                    if (act->type == GGML_TYPE_F32 && ggml_is_contiguous(act)) {
                        xe2mx_f32_to_f16(stream, (const float *) act->data, xdst, M * K);
                    } else {
                        to_x(act->data, xdst, M * K, stream);
                    }
                    xh = xdst;
                    if (xcache && xdst == g_xe2mx_xh_buf) {
                        g_xe2mx_xh_src = act;
                        g_xe2mx_xh_M = M;
                        g_xe2mx_xh_K = K;
                        g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
                    }
                }
            }
            if (xh || xf_act) {
                const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
                double t_deq = 0;
                double t_gemm = 0;
                if (xh && xe2mx_s4t_glu_run(ctx, (const uint8_t *) wg->data, (const uint8_t *) wu->data,
                                       xh, (float *) glu->data, K, N, M, time_on, &t_deq, &t_gemm)) {
                    return true;
                }
                ggml_sycl_pool_alloc<float> Cu(ctx.pool());
                ggml_sycl_pool_alloc<sycl::half> Cu16(ctx.pool());
                const int gluf16 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GLUF16", 0)
                    && N >= 16384 && !g_xe2mx_gluf16_dead;
                float * up = nullptr;
                if (gluf16) {
                    up = (float *) Cu16.alloc((size_t) M * (size_t) N);
                } else {
                    up = Cu.alloc((size_t) M * (size_t) N);
                }
                double t_deq2 = 0;
                double t_gemm2 = 0;
                const int s8u4_glu = ggml_sycl_get_env("GGML_SYCL_XE2MX_S8U4", 0) && !g_xe2mx_s8u4_dead;
                const int s44zp_glu = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44", 0)
                    && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44ZP", 0);
                const int glux = ggml_sycl_get_env("GGML_SYCL_XE2MX_S44GLUX", 0);
                const float * glu_bin = (s8u4_glu || s44zp_glu || glux) ? nullptr : up;
                g_xe2mx_p0_gluf16 = gluf16;
                g_xe2mx_p0_glu = 1;
                const bool oku = xe2mx_p0_u4(ctx, (const uint8_t *) wu->data, xh, xf_act, up,
                                              K, N, M, time_on, &t_deq, &t_gemm, nullptr, nullptr, act, wu);
                const bool okg = oku && xe2mx_p0_u4(ctx, (const uint8_t *) wg->data, xh, xf_act,
                                                     (float *) glu->data, K, N, M, time_on, &t_deq2, &t_gemm2,
                                                     nullptr, glu_bin, act, wg);
                g_xe2mx_p0_glu = 0;
                g_xe2mx_p0_gluf16 = 0;
                if (oku && okg) {
                    if (gluf16) {
                        static int ngluf16 = 0;
                        if (ngluf16 < 4) {
                            ngluf16++;
                            GGML_SYCL_XE2_LOG( "xe2mx s44gluf16 bind N=%ld M=%ld K=%ld up=%.2f gate=%.2f\n",
                                    (long) N, (long) M, (long) K, t_gemm, t_gemm2);
                            fflush(stderr);
                        }
                    }
                    if (s8u4_glu || s44zp_glu || glux) {
                        if (!glux) {
                            ctx.stream_dnnl(stream).wait();
                        }
                        xe2mx_silu_mul(stream, (float *) glu->data, up, M * N);
                        if (!glux) {
                            stream->wait();
                        }
                    }
                    if (glu_lo && N >= 16384 && ggml_sycl_get_env("GGML_SYCL_XE2MX_S44FGLUPACK", 0)
                        && glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu)) {
                        if (xe2mx_s44_pack_slot_xf(stream, glu, (const float *) glu->data, M, N)) {
                            static int nfp = 0;
                            if (nfp < 4) {
                                nfp++;
                                GGML_SYCL_XE2_LOG( "xe2mx s44 fglupack M=%ld K=%ld\n", (long) M, (long) N);
                                fflush(stderr);
                            }
                        }
                    }
                    if (time_on) {
                        static int nfire = 0;
                        if (nfire < 2) {
                            nfire++;
                            GGML_SYCL_XE2_LOG( "xe2mx s4glu fire up=%.2f gate=%.2f ms N=%ld M=%ld K=%ld\n",
                                    t_gemm, t_gemm2, (long) N, (long) M, (long) K);
                        }
                    }
                    return true;
                }
            }
        }
    }
    if (glu_s4 == 0) {
        return false;
    }
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    if (M < 1024 || N < s4n || g_xe2mx_s4_dead) {
        return false;
    }
    if ((K % QK_XE2MX4) != 0 || (N % 16) != 0) {
        return false;
    }
    if (!xe2mx_pack_xmx(ctx, wg) || !xe2mx_pack_xmx(ctx, wu)) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    const int64_t nb = K / 32;
    const size_t qpitch = (size_t) nb * 16u;
    const size_t qs = (size_t) N * qpitch;
    const size_t ds = (size_t) nb * (size_t) N * sizeof(sycl::half);
    const size_t bytes = 2 * qs + 2 * ds;
    const int64_t N2 = N * 2;
    const uint8_t * pg = (const uint8_t *) wg->data;
    const uint8_t * pu = (const uint8_t *) wu->data;
    const uint8_t * cat = nullptr;
    auto cit = g_xe2mx_wcache.find((const void *) pg);
    if (cit != g_xe2mx_wcache.end() && cit->second.vnni == 6 && cit->second.K == K
        && cit->second.N == N2 && cit->second.bytes == bytes && cit->second.packed) {
        cat = (const uint8_t *) cit->second.packed;
    } else if (bytes > (1ull << 20) && g_xe2mx_wcache_used + bytes <= XE2MX_WCACHE_CAP
               && !g_xe2mx_wcache.count((const void *) pg)) {
        void * persist = ggml_sycl_malloc_device(bytes, *stream);
        if (persist) {
            uint8_t * dst = (uint8_t *) persist;
            stream->memcpy(dst, pg, qs);
            stream->memcpy(dst + qs, pu, qs);
            const sycl::half * dg = (const sycl::half *) (pg + qs);
            const sycl::half * du = (const sycl::half *) (pu + qs);
            sycl::half * ddst = (sycl::half *) (dst + 2 * qs);
            const int n16 = (int) (N / 16);
            stream->parallel_for(sycl::range<2>((size_t) nb, (size_t) n16), [=](sycl::id<2> gid) {
                const int ib = (int) gid[0];
                const int n0 = (int) gid[1] * 16;
                const sycl::vec<sycl::half, 16> vg =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16> *>(dg + (size_t) ib * (size_t) N + (size_t) n0);
                const sycl::vec<sycl::half, 16> vu =
                    *reinterpret_cast<const sycl::vec<sycl::half, 16> *>(du + (size_t) ib * (size_t) N + (size_t) n0);
                *reinterpret_cast<sycl::vec<sycl::half, 16> *>(ddst + (size_t) ib * (size_t) N2 + (size_t) n0) = vg;
                *reinterpret_cast<sycl::vec<sycl::half, 16> *>(ddst + (size_t) ib * (size_t) N2 + (size_t) N + (size_t) n0) = vu;
            });
            g_xe2mx_wcache_used += bytes;
            g_xe2mx_wcache[(const void *) pg] = { persist, bytes, K, N2, 6 };
            cat = dst;
        }
    }
    if (!cat) {
        return false;
    }
    ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
    const sycl::half * xh = nullptr;
    const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
    if (act->type == GGML_TYPE_F16) {
        xh = (const sycl::half *) act->data;
    } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_src == act && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
               && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
        xh = g_xe2mx_xh_buf;
    } else {
        const to_fp16_sycl_t to_x = xe2mx_get_to_x(act->type, glu);
        if (!to_x) {
            return false;
        }
        sycl::half * xdst = nullptr;
        if (xcache) {
            const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
            if (g_xe2mx_xh_cap < need) {
                void * p = ggml_sycl_malloc_device(need, *stream);
                if (p) {
                    g_xe2mx_xh_buf = (sycl::half *) p;
                    g_xe2mx_xh_cap = need;
                }
            }
            if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                xdst = g_xe2mx_xh_buf;
            }
        }
        if (!xdst) {
            X.alloc((size_t) M * (size_t) K);
            xdst = X.get();
        }
        if (act->type == GGML_TYPE_F32 && ggml_is_contiguous(act)) {
            xe2mx_f32_to_f16(stream, (const float *) act->data, xdst, M * K);
        } else {
            to_x(act->data, xdst, M * K, stream);
        }
        xh = xdst;
        if (xcache && xdst == g_xe2mx_xh_buf) {
            g_xe2mx_xh_src = act;
            g_xe2mx_xh_M = M;
            g_xe2mx_xh_K = K;
            g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
        }
    }
    ggml_sycl_pool_alloc<float> C(ctx.pool());
    float * cp = C.alloc((size_t) M * (size_t) N2);
    double t_deq = 0;
    double t_gemm = 0;
    const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
    if (!xe2mx_p0_u4(ctx, cat, xh, nullptr, cp, K, N2, M, time_on, &t_deq, &t_gemm,
                     nullptr, nullptr, act)) {
        return false;
    }
    if (time_on) {
        static int nlogg = 0;
        if (nlogg < 8) {
            nlogg++;
            GGML_SYCL_XE2_LOG( "xe2mx glu gemm=%.2f ms N=%ld N2=%ld M=%ld K=%ld\n",
                    t_gemm, (long) N, (long) N2, (long) M, (long) K);
        }
    }
    float * dst = (float *) glu->data;
    const int64_t nn = M * N;
    stream->parallel_for(sycl::range<1>((size_t) nn), [=](sycl::id<1> gid) {
        const int64_t i = (int64_t) gid[0];
        const int64_t m = i / N;
        const int64_t n = i - m * N;
        const float g = cp[(size_t) m * (size_t) N2 + (size_t) n];
        const float u = cp[(size_t) m * (size_t) N2 + (size_t) N + (size_t) n];
        dst[i] = (g / (1.0f + sycl::exp(-g))) * u;
    });
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return false;
#endif
}

int ggml_sycl_xe2mx_s4gmlp_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_S4GMLP", 0) == 0 || g_xe2mx_s4gmlp_dead) {
        return 0;
    }
    if (!cgraph || node_idx + 3 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    ggml_tensor * b = cgraph->nodes[node_idx + 1];
    ggml_tensor * glu = cgraph->nodes[node_idx + 2];
    ggml_tensor * down = cgraph->nodes[node_idx + 3];
    if (!a || !b || !glu || !down) {
        return 0;
    }
    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU || down->op != GGML_OP_MUL_MAT) {
        static int nmiss = 0;
        if (nmiss < 2 && a->op == GGML_OP_MUL_MAT && b->op == GGML_OP_MUL_MAT && glu->op == GGML_OP_GLU) {
            nmiss++;
            GGML_SYCL_XE2_LOG( "xe2mx s4gmlp miss d3op=%d src1glu=%d\n", (int) down->op, down->src[1] == glu ? 1 : 0);
            fflush(stderr);
        }
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    if (ggml_get_op_params_i32(glu, 1)) {
        return 0;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx,
                               { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT },
                               { node_idx + 3 })) {
        return 0;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a))) {
        return 0;
    }
    if (down->src[1] != glu) {
        return 0;
    }
    const ggml_tensor * wg = gate->src[0];
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wd = down->src[0];
    const ggml_tensor * act = up->src[1];
    if (!wg || !wu || !wd || !act || act != gate->src[1]) {
        return 0;
    }
    if (wg->type != GGML_TYPE_XE2MX4 || wu->type != GGML_TYPE_XE2MX4 || wd->type != GGML_TYPE_XE2MX4) {
        return 0;
    }
    if (act->type != GGML_TYPE_F32 && act->type != GGML_TYPE_F16) {
        return 0;
    }
    if (glu->type != GGML_TYPE_F32 || down->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(glu) || !ggml_is_contiguous(act) || !ggml_is_contiguous(down)) {
        return 0;
    }
    if (!ggml_is_contiguous(wg) || !ggml_is_contiguous(wu) || !ggml_is_contiguous(wd)) {
        return 0;
    }
    const int64_t K = wg->ne[0];
    const int64_t N = wg->ne[1];
    const int64_t M = ggml_nrows(act);
    if (wu->ne[0] != K || wu->ne[1] != N) {
        return 0;
    }
    if (wd->ne[0] != N || wd->ne[1] != K) {
        return 0;
    }
    if (glu->ne[0] != N || glu->ne[1] != act->ne[1]) {
        return 0;
    }
    if (down->ne[0] != K || ggml_nrows(down) != M) {
        return 0;
    }
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    if (M < 1024 || N < s4n || (K % QK_XE2MX4) != 0 || (N % 16) != 0) {
        return 0;
    }
    if (ggml_sycl_xe2mx_s8dw_on(wd)) {
        return 0;
    }
    if (!xe2mx_pack_xmx(ctx, wg) || !xe2mx_pack_xmx(ctx, wu) || !xe2mx_pack_xmx(ctx, wd)) {
        return 0;
    }
    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> X(ctx.pool());
    const sycl::half * xh = nullptr;
    const int xcache = ggml_sycl_get_env("GGML_SYCL_XE2MX_XCACHE", 1);
    if (act->type == GGML_TYPE_F16) {
        xh = (const sycl::half *) act->data;
    } else if (xcache && g_xe2mx_xh_buf && g_xe2mx_xh_src == act && g_xe2mx_xh_M == M && g_xe2mx_xh_K == K
               && g_xe2mx_xh_hit_epoch == g_xe2mx_xh_epoch) {
        xh = g_xe2mx_xh_buf;
    } else {
        const to_fp16_sycl_t to_x = xe2mx_get_to_x(act->type, down);
        if (!to_x) {
            return 0;
        }
        sycl::half * xdst = nullptr;
        if (xcache) {
            const size_t need = (size_t) M * (size_t) K * sizeof(sycl::half);
            if (g_xe2mx_xh_cap < need) {
                void * p = ggml_sycl_malloc_device(need, *stream);
                if (p) {
                    g_xe2mx_xh_buf = (sycl::half *) p;
                    g_xe2mx_xh_cap = need;
                }
            }
            if (g_xe2mx_xh_buf && g_xe2mx_xh_cap >= need) {
                xdst = g_xe2mx_xh_buf;
            }
        }
        if (!xdst) {
            X.alloc((size_t) M * (size_t) K);
            xdst = X.get();
        }
        if (act->type == GGML_TYPE_F32 && ggml_is_contiguous(act)) {
            xe2mx_f32_to_f16(stream, (const float *) act->data, xdst, M * K);
        } else {
            to_x(act->data, xdst, M * K, stream);
        }
        xh = xdst;
        if (xcache && xdst == g_xe2mx_xh_buf) {
            g_xe2mx_xh_src = act;
            g_xe2mx_xh_M = M;
            g_xe2mx_xh_K = K;
            g_xe2mx_xh_hit_epoch = g_xe2mx_xh_epoch;
        }
    }
    if (!xh) {
        return 0;
    }
    const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
    double t_gemm = 0;
    if (!xe2mx_gmlp_run(ctx, (const uint8_t *) wg->data, (const uint8_t *) wu->data, (const uint8_t *) wd->data,
                          xh, (float *) down->data, K, N, M, time_on, &t_gemm)) {
        return 0;
    }
    static int nfire = 0;
    if (time_on && nfire < 2) {
        nfire++;
        GGML_SYCL_XE2_LOG( "xe2mx s4gmlp fire %.2f ms N=%ld M=%ld K=%ld\n", t_gemm, (long) N, (long) M, (long) K);
        fflush(stderr);
    }
    return 4;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

bool ggml_sycl_xe2mx_mul_mat_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_FADD", 0) == 0) {
        return false;
    }
    if (g_xe2mx_fadd_dead) {
        return false;
    }
    if (!cgraph || node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * mm = cgraph->nodes[node_idx];
    ggml_tensor * add = cgraph->nodes[node_idx + 1];
    if (!mm || !add) {
        return false;
    }
    if (mm->op != GGML_OP_MUL_MAT || add->op != GGML_OP_ADD) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_ADD }, { node_idx + 1 })) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * act = mm->src[1];
    if (!w || !act || w->type != GGML_TYPE_XE2MX4) {
        return false;
    }
    ggml_tensor * residual = nullptr;
    if (add->src[0] == mm) {
        residual = add->src[1];
    } else if (add->src[1] == mm) {
        residual = add->src[0];
    }
    if (!residual) {
        return false;
    }
    if (add->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(residual) || !ggml_is_contiguous(act) || !ggml_is_contiguous(w)) {
        return false;
    }
    if (!ggml_are_same_shape(add, residual)) {
        return false;
    }
    if (add->ne[0] != w->ne[1] || add->ne[1] != act->ne[1]) {
        return false;
    }
    const int64_t M = ggml_nrows(act);
    const int64_t N = w->ne[1];
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    if (M < 1024 || N < s4n || g_xe2mx_s4_dead) {
        return false;
    }
    if (!xe2mx_pack_xmx(ctx, w)) {
        return false;
    }
    const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
    // Keep LO s44 PD: do NOT pass residual into oneDNN (fadd bit slows the gemm).
    // LO gemm writes into `add`, then epilogue C += residual on the same stream.
    if (!xe2mx_p0_dnnl(ctx, w, act, add, time_on, true, /*residual*/ nullptr)) {
        return false;
    }
    {
        queue_ptr stream = ctx.stream();
        const float * r = (const float *) residual->data;
        float * c = (float *) add->data;
        const size_t ne = (size_t) ggml_nelements(add);
        stream->parallel_for(sycl::range<1>(ne), [=](sycl::id<1> gid) {
            const size_t i = gid[0];
            c[i] = c[i] + r[i];
        });
    }
    static int nhit = 0;
    if (nhit < 8) {
        nhit++;
        GGML_SYCL_XE2_LOG( "xe2mx fadd-epi hit N=%ld M=%ld\n", (long) N, (long) M);
        fflush(stderr);
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return false;
#endif
}

int ggml_sycl_xe2mx_mul_mat_silu_mul(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    if (ggml_sycl_get_env("GGML_SYCL_FUSE_ZSILU", 0) == 0) {
        return 0;
    }
    if (g_xe2mx_zsilu_dead) {
        return 0;
    }
    if (!cgraph || node_idx + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[node_idx];
    if (!mm || mm->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    int silu_i = -1;
    int mul_i = -1;
    ggml_tensor * mid = cgraph->nodes[node_idx + 1];
    if (mid && (mid->op == GGML_OP_RESHAPE || mid->op == GGML_OP_VIEW) && mid->src[0] == mm
        && node_idx + 3 < cgraph->n_nodes) {
        ggml_tensor * un = cgraph->nodes[node_idx + 2];
        ggml_tensor * mul = cgraph->nodes[node_idx + 3];
        if (un && mul && un->op == GGML_OP_UNARY && ggml_get_unary_op(un) == GGML_UNARY_OP_SILU
            && un->src[0] == mid && mul->op == GGML_OP_MUL) {
            silu_i = node_idx + 2;
            mul_i = node_idx + 3;
        }
    }
    if (silu_i < 0 && mid && mid->op == GGML_OP_UNARY && ggml_get_unary_op(mid) == GGML_UNARY_OP_SILU
        && mid->src[0] == mm) {
        ggml_tensor * mul = cgraph->nodes[node_idx + 2];
        if (mul && mul->op == GGML_OP_MUL) {
            silu_i = node_idx + 1;
            mul_i = node_idx + 2;
        }
    }
    if (silu_i < 0 || mul_i < 0) {
        return 0;
    }
    ggml_tensor * un = cgraph->nodes[silu_i];
    ggml_tensor * mul = cgraph->nodes[mul_i];
    ggml_tensor * y = nullptr;
    if (mul->src[0] == un) {
        y = mul->src[1];
    } else if (mul->src[1] == un) {
        y = mul->src[0];
    }
    if (!y) {
        return 0;
    }
    if (mul_i == node_idx + 3) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx,
                                    { GGML_OP_MUL_MAT, mid->op, GGML_OP_UNARY, GGML_OP_MUL },
                                    { mul_i })) {
            return 0;
        }
    } else if (!ggml_can_fuse_subgraph(cgraph, node_idx,
                                       { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL },
                                       { mul_i })) {
        return 0;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * act = mm->src[1];
    if (!w || !act || w->type != GGML_TYPE_XE2MX4) {
        return 0;
    }
    if (mul->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(mul) || !ggml_is_contiguous(y) || !ggml_is_contiguous(act) || !ggml_is_contiguous(w)) {
        return 0;
    }
    if (ggml_nelements(mul) != ggml_nelements(y) || ggml_nelements(mul) != ggml_nrows(act) * w->ne[1]) {
        return 0;
    }
    const int64_t M = ggml_nrows(act);
    const int64_t N = w->ne[1];
    const int s4n = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4N", 1024);
    if (M < 1024 || N < s4n || g_xe2mx_s4_dead) {
        return 0;
    }
    if (!xe2mx_pack_xmx(ctx, w)) {
        return 0;
    }
    const int time_on = ggml_sycl_get_env("GGML_SYCL_XE2MX_TIME", 0);
    g_xe2mx_p0_zsilu = 1;
    const bool ok = xe2mx_p0_dnnl(ctx, w, act, mul, time_on, true, nullptr, true, (const float *) y->data);
    g_xe2mx_p0_zsilu = 0;
    if (!ok) {
        return 0;
    }
    static int nhit = 0;
    if (nhit < 8) {
        nhit++;
        GGML_SYCL_XE2_LOG( "xe2mx zsilu hit N=%ld M=%ld skip=%d\n", (long) N, (long) M, mul_i - node_idx);
        fflush(stderr);
    }
    return mul_i - node_idx + 1;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(node_idx);
    return 0;
#endif
}

struct xe2mx_devscratch {
    void * lo_hp_arena;
    sycl::queue * lo_hp_cq;
    uint8_t * s44_tmp;
    float * s44_tsc;
    int8_t * s44_t8;
    size_t s44_tcap, s44_tsccap, s44_t8cap;
    sycl::half * s48_udev;
    int s48_udev_layer;
    sycl::half * s48_ulayer[64];
    sycl::half * s48_xrot;
    size_t s48_xrot_cap;
    const sycl::half * s48_xrot_xh;
    int64_t s48_xrot_M, s48_xrot_K;
    int s48_xrot_layer;
    uint64_t s48_xrot_epoch;
    uint8_t * s48_us4[64];
    float * s48_usc4[64];
    int s48_us4g[64];
    float * s48_xrotf;
    size_t s48_xrotf_cap;
    uint8_t * s48_ras4;
    size_t s48_ras4_cap;
    float * s48_rsas4;
    size_t s48_rsas4_cap;
    int8_t * s48_ra8;
    size_t s48_ra8_cap;
    float * s48_rtmp;
    size_t s48_rtmp_cap;
    int8_t * s48_as8;
    size_t s48_as8_cap;
    uint8_t * s48_as4;
    size_t s48_as4_cap;
    float * s48_sas8;
    float * s48_sas4;
    size_t s48_sas4_cap;
    const sycl::half * s48_axh;
    const uint8_t * s48_apacked;
    int64_t s48_aM, s48_aK;
    int s48_ar, s48_ag, s48_alayer;
    uint64_t s48_a_epoch;
    int8_t * s48_a8lo;
    size_t s48_a8lo_cap;
    float * s48_qtmp;
    size_t s48_qtmp_cap;
    void * s4dn_slot;
    size_t s4dn_slot_cap;
    const uint8_t * s4dn_src;
    int64_t s4dn_g;
    int8_t * s4dn_t8;
    size_t s4dn_t8_cap;
    uint8_t * s4dn_psrc;
    size_t s4dn_psrc_cap;
    int8_t * s44_b8;
    size_t s44_b8_cap;
    const ggml_tensor * xh_src;
    sycl::half * xh_buf;
    size_t xh_cap;
    int xh_dev = -1;
    int64_t xh_M, xh_K;
    uint64_t xh_epoch, xh_hit_epoch;
    const ggml_tensor * aq_src;
    int8_t * aq_buf;
    float * daq_buf;
    size_t aq_cap, daq_cap;
    int64_t aq_M, aq_K;
    uint64_t aq_hit_epoch;
    const sycl::half * ar_xh;
    uint8_t * ar_buf;
    size_t ar_cap, ar_bytes;
    int ar_dev = -1;
    int64_t ar_M, ar_K;
    uint64_t ar_hit_epoch;
    uint8_t * loa;
    float * losa;
    const void * loa_src;
    const sycl::half * loa_xh;
    size_t loa_cap, losa_cap;
    int64_t loa_M, loa_K, loa_g;
    uint64_t loa_epoch;
    xe2mx_s44_aslot s44_slot[XE2MX_S44_SLOT_MAX];
    uint64_t s44_tick;
    sycl::half * f16a_ah;
    float * f16a_sa;
    float * f16a_sa2[2];
    uint8_t * f16a_ap2[2];
    size_t f16a_ah_cap, f16a_sa_cap;
    size_t f16a_sa2_cap[2], f16a_ap2_cap[2];
    int f16a_sel;
    int8_t * s8ra;
    float * s8rsa;
    size_t s8rsa_cap;
    int s8rsa_row;
    const sycl::half * s8rxh;
    const void * s8r_src;
    size_t s8ra_cap;
    int64_t s8rM, s8rK;
    uint64_t s8r_epoch;
    uint8_t * s44a;
    float * s44sa;
    const sycl::half * s44xh;
    const void * s44_src;
    size_t s44a_cap, s44sa_cap;
    int64_t s44M, s44K;
    uint64_t s44_epoch;
    int s44sa_row, s44_qmode;
    float * s44za;
    size_t s44za_cap;
    int8_t * zp;
    size_t zp_cap;
    int8_t * qb_A;
    size_t qb_Acap;
    float * qb_da;
    size_t qb_dacap;
    xe2mx_qb_k * qb_dev;
    int qb_devcap;
    uint64_t qb_epoch;
    uint8_t * gmlp_zp;
    size_t gmlp_zp_cap;
    sycl::half * gmlp_ho;
    size_t gmlp_ho_cap;
};

static xe2mx_devscratch g_xe2mx_devscratch[GGML_SYCL_MAX_DEVICES];
static int g_xe2mx_scratch_dev = 0;

static void xe2mx_scratch_save(xe2mx_devscratch & s) {
    s.lo_hp_arena = g_xe2mx_lo_hp_arena; s.lo_hp_cq = g_xe2mx_lo_hp_cq;
    s.s44_tmp = g_xe2mx_s44_tmp; s.s44_tsc = g_xe2mx_s44_tsc; s.s44_t8 = g_xe2mx_s44_t8;
    s.s44_tcap = g_xe2mx_s44_tcap; s.s44_tsccap = g_xe2mx_s44_tsccap; s.s44_t8cap = g_xe2mx_s44_t8cap;
    s.s48_udev = g_xe2mx_s48_udev; s.s48_udev_layer = g_xe2mx_s48_udev_layer;
    std::memcpy(s.s48_ulayer, g_xe2mx_s48_ulayer, sizeof(s.s48_ulayer));
    s.s48_xrot = g_xe2mx_s48_xrot; s.s48_xrot_cap = g_xe2mx_s48_xrot_cap; s.s48_xrot_xh = g_xe2mx_s48_xrot_xh;
    s.s48_xrot_M = g_xe2mx_s48_xrot_M; s.s48_xrot_K = g_xe2mx_s48_xrot_K;
    s.s48_xrot_layer = g_xe2mx_s48_xrot_layer; s.s48_xrot_epoch = g_xe2mx_s48_xrot_epoch;
    std::memcpy(s.s48_us4, g_xe2mx_s48_us4, sizeof(s.s48_us4));
    std::memcpy(s.s48_usc4, g_xe2mx_s48_usc4, sizeof(s.s48_usc4));
    std::memcpy(s.s48_us4g, g_xe2mx_s48_us4g, sizeof(s.s48_us4g));
    s.s48_xrotf = g_xe2mx_s48_xrotf; s.s48_xrotf_cap = g_xe2mx_s48_xrotf_cap;
    s.s48_ras4 = g_xe2mx_s48_ras4; s.s48_ras4_cap = g_xe2mx_s48_ras4_cap;
    s.s48_rsas4 = g_xe2mx_s48_rsas4; s.s48_rsas4_cap = g_xe2mx_s48_rsas4_cap;
    s.s48_ra8 = g_xe2mx_s48_ra8; s.s48_ra8_cap = g_xe2mx_s48_ra8_cap;
    s.s48_rtmp = g_xe2mx_s48_rtmp; s.s48_rtmp_cap = g_xe2mx_s48_rtmp_cap;
    s.s48_as8 = g_xe2mx_s48_as8; s.s48_as8_cap = g_xe2mx_s48_as8_cap;
    s.s48_as4 = g_xe2mx_s48_as4; s.s48_as4_cap = g_xe2mx_s48_as4_cap;
    s.s48_sas8 = g_xe2mx_s48_sas8; s.s48_sas4 = g_xe2mx_s48_sas4; s.s48_sas4_cap = g_xe2mx_s48_sas4_cap;
    s.s48_axh = g_xe2mx_s48_axh; s.s48_apacked = g_xe2mx_s48_apacked;
    s.s48_aM = g_xe2mx_s48_aM; s.s48_aK = g_xe2mx_s48_aK;
    s.s48_ar = g_xe2mx_s48_ar; s.s48_ag = g_xe2mx_s48_ag; s.s48_alayer = g_xe2mx_s48_alayer;
    s.s48_a_epoch = g_xe2mx_s48_a_epoch;
    s.s48_a8lo = g_xe2mx_s48_a8lo; s.s48_a8lo_cap = g_xe2mx_s48_a8lo_cap;
    s.s48_qtmp = g_xe2mx_s48_qtmp; s.s48_qtmp_cap = g_xe2mx_s48_qtmp_cap;
    s.s4dn_slot = g_xe2mx_s4dn_slot; s.s4dn_slot_cap = g_xe2mx_s4dn_slot_cap; s.s4dn_src = g_xe2mx_s4dn_src;
    s.s4dn_g = g_xe2mx_s4dn_g; s.s4dn_t8 = g_xe2mx_s4dn_t8; s.s4dn_t8_cap = g_xe2mx_s4dn_t8_cap;
    s.s4dn_psrc = g_xe2mx_s4dn_psrc; s.s4dn_psrc_cap = g_xe2mx_s4dn_psrc_cap;
    s.s44_b8 = g_xe2mx_s44_b8; s.s44_b8_cap = g_xe2mx_s44_b8_cap;
    s.xh_src = g_xe2mx_xh_src; s.xh_buf = g_xe2mx_xh_buf; s.xh_cap = g_xe2mx_xh_cap; s.xh_dev = g_xe2mx_xh_dev;
    s.xh_M = g_xe2mx_xh_M; s.xh_K = g_xe2mx_xh_K; s.xh_epoch = g_xe2mx_xh_epoch; s.xh_hit_epoch = g_xe2mx_xh_hit_epoch;
    s.aq_src = g_xe2mx_aq_src; s.aq_buf = g_xe2mx_aq_buf; s.daq_buf = g_xe2mx_daq_buf;
    s.aq_cap = g_xe2mx_aq_cap; s.daq_cap = g_xe2mx_daq_cap;
    s.aq_M = g_xe2mx_aq_M; s.aq_K = g_xe2mx_aq_K; s.aq_hit_epoch = g_xe2mx_aq_hit_epoch;
    s.ar_xh = g_xe2mx_ar_xh; s.ar_buf = g_xe2mx_ar_buf; s.ar_cap = g_xe2mx_ar_cap; s.ar_bytes = g_xe2mx_ar_bytes; s.ar_dev = g_xe2mx_ar_dev;
    s.ar_M = g_xe2mx_ar_M; s.ar_K = g_xe2mx_ar_K; s.ar_hit_epoch = g_xe2mx_ar_hit_epoch;
    s.loa = g_xe2mx_loa; s.losa = g_xe2mx_losa; s.loa_src = g_xe2mx_loa_src; s.loa_xh = g_xe2mx_loa_xh;
    s.loa_cap = g_xe2mx_loa_cap; s.losa_cap = g_xe2mx_losa_cap;
    s.loa_M = g_xe2mx_loa_M; s.loa_K = g_xe2mx_loa_K; s.loa_g = g_xe2mx_loa_g; s.loa_epoch = g_xe2mx_loa_epoch;
    std::memcpy(s.s44_slot, g_xe2mx_s44_slot, sizeof(s.s44_slot));
    s.s44_tick = g_xe2mx_s44_tick;
    s.f16a_ah = g_xe2mx_f16a_ah; s.f16a_sa = g_xe2mx_f16a_sa;
    std::memcpy(s.f16a_sa2, g_xe2mx_f16a_sa2, sizeof(s.f16a_sa2));
    std::memcpy(s.f16a_ap2, g_xe2mx_f16a_ap2, sizeof(s.f16a_ap2));
    s.f16a_ah_cap = g_xe2mx_f16a_ah_cap; s.f16a_sa_cap = g_xe2mx_f16a_sa_cap;
    std::memcpy(s.f16a_sa2_cap, g_xe2mx_f16a_sa2_cap, sizeof(s.f16a_sa2_cap));
    std::memcpy(s.f16a_ap2_cap, g_xe2mx_f16a_ap2_cap, sizeof(s.f16a_ap2_cap));
    s.f16a_sel = g_xe2mx_f16a_sel;
    s.s8ra = g_xe2mx_s8ra; s.s8rsa = g_xe2mx_s8rsa; s.s8rsa_cap = g_xe2mx_s8rsa_cap; s.s8rsa_row = g_xe2mx_s8rsa_row;
    s.s8rxh = g_xe2mx_s8rxh; s.s8r_src = g_xe2mx_s8r_src; s.s8ra_cap = g_xe2mx_s8ra_cap;
    s.s8rM = g_xe2mx_s8rM; s.s8rK = g_xe2mx_s8rK; s.s8r_epoch = g_xe2mx_s8r_epoch;
    s.s44a = g_xe2mx_s44a; s.s44sa = g_xe2mx_s44sa; s.s44xh = g_xe2mx_s44xh; s.s44_src = g_xe2mx_s44_src;
    s.s44a_cap = g_xe2mx_s44a_cap; s.s44sa_cap = g_xe2mx_s44sa_cap;
    s.s44M = g_xe2mx_s44M; s.s44K = g_xe2mx_s44K; s.s44_epoch = g_xe2mx_s44_epoch;
    s.s44sa_row = g_xe2mx_s44sa_row; s.s44_qmode = g_xe2mx_s44_qmode;
    s.s44za = g_xe2mx_s44za; s.s44za_cap = g_xe2mx_s44za_cap;
    s.zp = g_xe2mx_zp; s.zp_cap = g_xe2mx_zp_cap;
    s.qb_A = g_xe2mx_qb_A; s.qb_Acap = g_xe2mx_qb_Acap; s.qb_da = g_xe2mx_qb_da; s.qb_dacap = g_xe2mx_qb_dacap;
    s.qb_dev = g_xe2mx_qb_dev; s.qb_devcap = g_xe2mx_qb_devcap; s.qb_epoch = g_xe2mx_qb_epoch;
    s.gmlp_zp = g_xe2mx_gmlp_zp; s.gmlp_zp_cap = g_xe2mx_gmlp_zp_cap;
    s.gmlp_ho = g_xe2mx_gmlp_ho; s.gmlp_ho_cap = g_xe2mx_gmlp_ho_cap;
}

static void xe2mx_scratch_load(const xe2mx_devscratch & s) {
    g_xe2mx_lo_hp_arena = s.lo_hp_arena; g_xe2mx_lo_hp_cq = s.lo_hp_cq;
    g_xe2mx_s44_tmp = s.s44_tmp; g_xe2mx_s44_tsc = s.s44_tsc; g_xe2mx_s44_t8 = s.s44_t8;
    g_xe2mx_s44_tcap = s.s44_tcap; g_xe2mx_s44_tsccap = s.s44_tsccap; g_xe2mx_s44_t8cap = s.s44_t8cap;
    g_xe2mx_s48_udev = s.s48_udev; g_xe2mx_s48_udev_layer = s.s48_udev_layer;
    std::memcpy(g_xe2mx_s48_ulayer, s.s48_ulayer, sizeof(s.s48_ulayer));
    g_xe2mx_s48_xrot = s.s48_xrot; g_xe2mx_s48_xrot_cap = s.s48_xrot_cap; g_xe2mx_s48_xrot_xh = s.s48_xrot_xh;
    g_xe2mx_s48_xrot_M = s.s48_xrot_M; g_xe2mx_s48_xrot_K = s.s48_xrot_K;
    g_xe2mx_s48_xrot_layer = s.s48_xrot_layer; g_xe2mx_s48_xrot_epoch = s.s48_xrot_epoch;
    std::memcpy(g_xe2mx_s48_us4, s.s48_us4, sizeof(s.s48_us4));
    std::memcpy(g_xe2mx_s48_usc4, s.s48_usc4, sizeof(s.s48_usc4));
    std::memcpy(g_xe2mx_s48_us4g, s.s48_us4g, sizeof(s.s48_us4g));
    g_xe2mx_s48_xrotf = s.s48_xrotf; g_xe2mx_s48_xrotf_cap = s.s48_xrotf_cap;
    g_xe2mx_s48_ras4 = s.s48_ras4; g_xe2mx_s48_ras4_cap = s.s48_ras4_cap;
    g_xe2mx_s48_rsas4 = s.s48_rsas4; g_xe2mx_s48_rsas4_cap = s.s48_rsas4_cap;
    g_xe2mx_s48_ra8 = s.s48_ra8; g_xe2mx_s48_ra8_cap = s.s48_ra8_cap;
    g_xe2mx_s48_rtmp = s.s48_rtmp; g_xe2mx_s48_rtmp_cap = s.s48_rtmp_cap;
    g_xe2mx_s48_as8 = s.s48_as8; g_xe2mx_s48_as8_cap = s.s48_as8_cap;
    g_xe2mx_s48_as4 = s.s48_as4; g_xe2mx_s48_as4_cap = s.s48_as4_cap;
    g_xe2mx_s48_sas8 = s.s48_sas8; g_xe2mx_s48_sas4 = s.s48_sas4; g_xe2mx_s48_sas4_cap = s.s48_sas4_cap;
    g_xe2mx_s48_axh = s.s48_axh; g_xe2mx_s48_apacked = s.s48_apacked;
    g_xe2mx_s48_aM = s.s48_aM; g_xe2mx_s48_aK = s.s48_aK;
    g_xe2mx_s48_ar = s.s48_ar; g_xe2mx_s48_ag = s.s48_ag; g_xe2mx_s48_alayer = s.s48_alayer;
    g_xe2mx_s48_a_epoch = s.s48_a_epoch;
    g_xe2mx_s48_a8lo = s.s48_a8lo; g_xe2mx_s48_a8lo_cap = s.s48_a8lo_cap;
    g_xe2mx_s48_qtmp = s.s48_qtmp; g_xe2mx_s48_qtmp_cap = s.s48_qtmp_cap;
    g_xe2mx_s4dn_slot = s.s4dn_slot; g_xe2mx_s4dn_slot_cap = s.s4dn_slot_cap; g_xe2mx_s4dn_src = s.s4dn_src;
    g_xe2mx_s4dn_g = s.s4dn_g; g_xe2mx_s4dn_t8 = s.s4dn_t8; g_xe2mx_s4dn_t8_cap = s.s4dn_t8_cap;
    g_xe2mx_s4dn_psrc = s.s4dn_psrc; g_xe2mx_s4dn_psrc_cap = s.s4dn_psrc_cap;
    g_xe2mx_s44_b8 = s.s44_b8; g_xe2mx_s44_b8_cap = s.s44_b8_cap;
    g_xe2mx_xh_src = s.xh_src; g_xe2mx_xh_buf = s.xh_buf; g_xe2mx_xh_cap = s.xh_cap; g_xe2mx_xh_dev = s.xh_dev;
    g_xe2mx_xh_M = s.xh_M; g_xe2mx_xh_K = s.xh_K; g_xe2mx_xh_epoch = s.xh_epoch; g_xe2mx_xh_hit_epoch = s.xh_hit_epoch;
    g_xe2mx_aq_src = s.aq_src; g_xe2mx_aq_buf = s.aq_buf; g_xe2mx_daq_buf = s.daq_buf;
    g_xe2mx_aq_cap = s.aq_cap; g_xe2mx_daq_cap = s.daq_cap;
    g_xe2mx_aq_M = s.aq_M; g_xe2mx_aq_K = s.aq_K; g_xe2mx_aq_hit_epoch = s.aq_hit_epoch;
    g_xe2mx_ar_xh = s.ar_xh; g_xe2mx_ar_buf = s.ar_buf; g_xe2mx_ar_cap = s.ar_cap; g_xe2mx_ar_bytes = s.ar_bytes; g_xe2mx_ar_dev = s.ar_dev;
    g_xe2mx_ar_M = s.ar_M; g_xe2mx_ar_K = s.ar_K; g_xe2mx_ar_hit_epoch = s.ar_hit_epoch;
    g_xe2mx_loa = s.loa; g_xe2mx_losa = s.losa; g_xe2mx_loa_src = s.loa_src; g_xe2mx_loa_xh = s.loa_xh;
    g_xe2mx_loa_cap = s.loa_cap; g_xe2mx_losa_cap = s.losa_cap;
    g_xe2mx_loa_M = s.loa_M; g_xe2mx_loa_K = s.loa_K; g_xe2mx_loa_g = s.loa_g; g_xe2mx_loa_epoch = s.loa_epoch;
    std::memcpy(g_xe2mx_s44_slot, s.s44_slot, sizeof(s.s44_slot));
    g_xe2mx_s44_tick = s.s44_tick;
    g_xe2mx_f16a_ah = s.f16a_ah; g_xe2mx_f16a_sa = s.f16a_sa;
    std::memcpy(g_xe2mx_f16a_sa2, s.f16a_sa2, sizeof(s.f16a_sa2));
    std::memcpy(g_xe2mx_f16a_ap2, s.f16a_ap2, sizeof(s.f16a_ap2));
    g_xe2mx_f16a_ah_cap = s.f16a_ah_cap; g_xe2mx_f16a_sa_cap = s.f16a_sa_cap;
    std::memcpy(g_xe2mx_f16a_sa2_cap, s.f16a_sa2_cap, sizeof(s.f16a_sa2_cap));
    std::memcpy(g_xe2mx_f16a_ap2_cap, s.f16a_ap2_cap, sizeof(s.f16a_ap2_cap));
    g_xe2mx_f16a_sel = s.f16a_sel;
    g_xe2mx_s8ra = s.s8ra; g_xe2mx_s8rsa = s.s8rsa; g_xe2mx_s8rsa_cap = s.s8rsa_cap; g_xe2mx_s8rsa_row = s.s8rsa_row;
    g_xe2mx_s8rxh = s.s8rxh; g_xe2mx_s8r_src = s.s8r_src; g_xe2mx_s8ra_cap = s.s8ra_cap;
    g_xe2mx_s8rM = s.s8rM; g_xe2mx_s8rK = s.s8rK; g_xe2mx_s8r_epoch = s.s8r_epoch;
    g_xe2mx_s44a = s.s44a; g_xe2mx_s44sa = s.s44sa; g_xe2mx_s44xh = s.s44xh; g_xe2mx_s44_src = s.s44_src;
    g_xe2mx_s44a_cap = s.s44a_cap; g_xe2mx_s44sa_cap = s.s44sa_cap;
    g_xe2mx_s44M = s.s44M; g_xe2mx_s44K = s.s44K; g_xe2mx_s44_epoch = s.s44_epoch;
    g_xe2mx_s44sa_row = s.s44sa_row; g_xe2mx_s44_qmode = s.s44_qmode;
    g_xe2mx_s44za = s.s44za; g_xe2mx_s44za_cap = s.s44za_cap;
    g_xe2mx_zp = s.zp; g_xe2mx_zp_cap = s.zp_cap;
    g_xe2mx_qb_A = s.qb_A; g_xe2mx_qb_Acap = s.qb_Acap; g_xe2mx_qb_da = s.qb_da; g_xe2mx_qb_dacap = s.qb_dacap;
    g_xe2mx_qb_dev = s.qb_dev; g_xe2mx_qb_devcap = s.qb_devcap; g_xe2mx_qb_epoch = s.qb_epoch;
    g_xe2mx_gmlp_zp = s.gmlp_zp; g_xe2mx_gmlp_zp_cap = s.gmlp_zp_cap;
    g_xe2mx_gmlp_ho = s.gmlp_ho; g_xe2mx_gmlp_ho_cap = s.gmlp_ho_cap;
}

void xe2mx_scratch_bind(int device, bool force) {
    if (device == g_xe2mx_scratch_dev && !force) {
        return;
    }
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    static int nlog = 0;
    if (nlog < 4) {
        nlog++;
        GGML_SYCL_XE2_LOG( "xe2mx scratch bind %d -> %d\n", g_xe2mx_scratch_dev, device);
        fflush(stderr);
    }
    xe2mx_scratch_save(g_xe2mx_devscratch[g_xe2mx_scratch_dev]);
    xe2mx_scratch_load(g_xe2mx_devscratch[device]);
    g_xe2mx_scratch_dev = device;
}
