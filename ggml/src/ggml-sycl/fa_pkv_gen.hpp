// Host-side write generation for FA PKV invalidation (no GPU sync).
#pragma once

#include <cstdint>

// Bump generation for a device buffer base pointer after set_rows (or any KV write).
void ggml_sycl_fa_pkv_note_write(const void * data);

// Current generation for data, or 0 if never noted.
uint64_t ggml_sycl_fa_pkv_get_gen(const void * data);
