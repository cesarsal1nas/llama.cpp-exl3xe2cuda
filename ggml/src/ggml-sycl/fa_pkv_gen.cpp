#include "fa_pkv_gen.hpp"

#include <mutex>
#include <unordered_map>

static std::mutex g_fa_pkv_gen_mu;
static std::unordered_map<const void *, uint64_t> g_fa_pkv_gen;
static uint64_t g_fa_pkv_gen_tick = 0;

void ggml_sycl_fa_pkv_note_write(const void * data) {
    if (!data) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_fa_pkv_gen_mu);
    g_fa_pkv_gen[data] = ++g_fa_pkv_gen_tick;
}

uint64_t ggml_sycl_fa_pkv_get_gen(const void * data) {
    if (!data) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(g_fa_pkv_gen_mu);
    const auto it = g_fa_pkv_gen.find(data);
    return it == g_fa_pkv_gen.end() ? 0 : it->second;
}
