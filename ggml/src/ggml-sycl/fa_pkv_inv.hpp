// Pure CPU helpers for FA PKV shadow invalidation (no SYCL).
// get_n_kv pads used_max_p1 to max(n_pad,256), so seq can stay fixed while
// set_rows updates K/V rows inside the pad window. pointer+seq alone is not a
// cache key; without a write-generation proof we must full-refresh [0,seq).
#pragma once

#include <cstdint>

struct fa_pkv_inv_in {
    int64_t  seq              = 0;
    int64_t  seq_cap          = 0;
    int64_t  filled           = 0;
    uint64_t slot_gen         = 0;  // gen observed when slot was last filled
    uint64_t live_gen         = 0;  // current host gen for K base data (0 = unknown)
    bool     have_live_gen    = false;
    bool     allow_incremental = false; // perf path; default false = always full refresh
};

struct fa_pkv_inv_out {
    bool    reject       = false; // over-cap or invalid args
    bool    release_slot = false; // free this layer shadow (over-cap)
    bool    need_refresh = false;
    int64_t t0           = 0;
    int64_t n_new        = 0;
    int64_t filled_after = 0;
    uint64_t gen_after   = 0;
};

// Decide dequant range for one PKV stage. Does not touch device memory.
inline fa_pkv_inv_out fa_pkv_decide(const fa_pkv_inv_in & in) {
    fa_pkv_inv_out o;
    o.gen_after = in.have_live_gen ? in.live_gen : in.slot_gen;

    if (in.seq < 1 || in.seq_cap < 1) {
        o.reject = true;
        return o;
    }
    if (in.seq > in.seq_cap) {
        o.reject       = true;
        o.release_slot = true;
        return o;
    }

    int64_t filled = in.filled;
    if (filled < 0) {
        filled = 0;
    }
    if (filled > in.seq_cap) {
        filled = 0;
    }

    // Correctness default: full refresh. pointer+filled cannot prove freshness when
    // padded seq is unchanged after set_rows (llama_kv_cache::get_n_kv).
    bool full = !in.allow_incremental;

    // Perf/incremental path only when explicitly enabled.
    if (in.allow_incremental) {
        if (in.have_live_gen) {
            if (in.live_gen != in.slot_gen) {
                full = true; // K/V written since last fill
            }
        } else {
            // No write-gen hook: cannot prove freshness -> full refresh even under INC.
            full = true;
        }
        if (in.seq < filled) {
            full = true; // shrink / reset
        }
    }

    if (full) {
        o.need_refresh = true;
        o.t0           = 0;
        o.n_new        = in.seq;
        o.filled_after = in.seq;
        return o;
    }

    // Incremental: gen matched and seq >= filled.
    if (in.seq == filled) {
        o.need_refresh = false;
        o.t0           = filled;
        o.n_new        = 0;
        o.filled_after = filled;
        return o;
    }
    // seq grew with gen still matching (no set_rows since last fill - unusual).
    o.t0    = filled;
    o.n_new = in.seq - filled;
    if (o.t0 < 0 || o.n_new < 1 || o.t0 + o.n_new > in.seq_cap) {
        o.reject = true;
        return o;
    }
    o.need_refresh = true;
    o.filled_after = in.seq;
    return o;
}

// Round mode-1 capacity; mode>=2 uses fixed cap (caller supplies).
inline int64_t fa_pkv_round_cap(int64_t seq, int64_t round = 2048) {
    if (seq < 1 || round < 1) {
        return 0;
    }
    return ((seq + round - 1) / round) * round;
}
