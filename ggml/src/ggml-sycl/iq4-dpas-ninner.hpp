// IQ4_XS x q8_1 dpas. K walked once per N-chunk of MAXN. SLM [WG][MAXN][16].
template <int NC, int MAXN>
ESIMD_INLINE void mul_mat_iq4_xs_q8_1_dpas_ninner(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_y,
        const int stride_y_bytes, const int stride_dst,
        const int16_t * coef, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    namespace ex = sycl::ext::intel::experimental::esimd;
    constexpr int WG  = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    constexpr int RPW = 16;
    slm_init<WG * MAXN * RPW * 4>();

    const int        nblocks = ncols / QK_K;
    const size_t     nb      = (size_t) nrows * nblocks;
    const uint8_t *  qs_base = (const uint8_t *) vx;
    const uint32_t * sl32    = (const uint32_t *) (qs_base + nb * (QK_K / 2));
    const uint32_t * pk32    = (const uint32_t *) (qs_base + nb * (QK_K / 2) + nb * 4);
    const int        tid     = it.get_local_id(0);
    const int        row0    = it.get_group(0) * RPW;
    const unsigned   pitch   = (unsigned) nblocks * (QK_K / 2);

    const simd<int16_t, 8> kc = block_load<int16_t, 8>(coef);
    const int16_t pc0 = kc[0], pc1 = kc[1], pc2 = kc[2], pc3 = kc[3];
    auto kpoly = [&](simd<uint32_t, 32> s) -> simd<uint32_t, 32> {
        simd<int16_t, 64> v = s.template bit_cast_view<int16_t>();
        simd<int16_t, 64> h = v + pc0;
        h = h * v + pc1;
        h = h * v + pc2;
        h = h * v + pc3;
        h = h >> 10;
        h = h + (v << 4);
        return h.template bit_cast_view<uint32_t>();
    };
    auto dequant = [&](simd<uint32_t, 32> src, simd<uint32_t, 32> & lo, simd<uint32_t, 32> & hi) {
        simd<uint32_t, 32> e_lo = src & 0x000F000Fu;
        simd<uint32_t, 32> o_lo = (src >> 8u) & 0x000F000Fu;
        simd<uint32_t, 32> e_hi = (src >> 4u) & 0x000F000Fu;
        simd<uint32_t, 32> o_hi = (src >> 12u) & 0x000F000Fu;
        lo = kpoly(e_lo) | (kpoly(o_lo) << 8u);
        hi = kpoly(e_hi) | (kpoly(o_hi) << 8u);
    };

    for (int nbase = 0; nbase < ncols_y; nbase += MAXN) {
        const int nlim   = ncols_y - nbase < MAXN ? ncols_y - nbase : MAXN;
        const int ntiles = nlim / NC;
        for (int col = tid; col < nlim; col += WG) {
#pragma unroll
            for (int t = 0; t < WG; ++t) {
                slm_block_store<float, 16>((uint32_t) ((t * MAXN + col) * 64), simd<float, 16>(0.0f));
            }
        }
        barrier();

        simd<uint32_t, RPW> soff;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int rowc = row0 + r < nrows ? row0 + r : nrows - 1;
            soff[r] = ((uint32_t) rowc * nblocks + tid) * 4u;
        }
        int xib = tid;
        for (int ib = tid; ib < nblocks; ib += WG, xib += WG, soff += (uint32_t) (WG * 4)) {
            simd<uint32_t, 128> T[4];
            simd<uint32_t, RPW> sl = gather<uint32_t, RPW>(sl32, soff);
            simd<uint32_t, RPW> pk = gather<uint32_t, RPW>(pk32, soff);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t *) qs_base, pitch - 1, nrows - 1,
                                                                       pitch - 1, xib * 32 + 8 * k, row0);
            }
            simd<uint16_t, 16>   dbits = convert<uint16_t>(pk >> 16u);
            simd<sycl::half, 16> dh    = dbits.template bit_cast_view<sycl::half>();
            simd<float, 16>      d16   = convert<float>(dh);
            simd<uint32_t, 16>   sh16  = pk & 0xFFFFu;
            simd<uint32_t, 128>  Bt[8];
#pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
#pragma unroll
                for (int ip = 0; ip < 4; ip += 2) {
                    simd<uint32_t, 32> src = T[sb / 2].template select<32, 1>(16 * ((sb % 2) * 4 + ip));
                    simd<uint32_t, 32> lo, hi;
                    dequant(src, lo, hi);
                    Bt[sb].template select<32, 1>(16 * ip)      = lo;
                    Bt[sb].template select<32, 1>(64 + 16 * ip) = hi;
                }
            }
            for (int t = 0; t < ntiles; ++t) {
                const int col0 = nbase + t * NC;
                const int slc0 = t * NC;
                simd<int8_t, 256>  y8[NC];
                simd<uint16_t, 16> yds[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const char * ycol = (const char *) vy + (size_t) (col0 + c) * stride_y_bytes;
                    y8[c]  = block_load<int8_t, 256>((const int8_t *) ycol + (size_t) ib * QK_K);
                    yds[c] = block_load<uint16_t, 16>((const uint16_t *) (ycol + ncols) + (size_t) ib * 16);
                }
                simd<float, 8> dy[NC];
                simd<float, 8> ysum[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<uint16_t, 8>   dyb = yds[c].template select<8, 2>(0);
                    simd<sycl::half, 8> dyh = dyb.template bit_cast_view<sycl::half>();
                    dy[c] = convert<float>(dyh);
                    simd<int16_t, 8> ys = yds[c].template select<8, 2>(1).template bit_cast_view<int16_t>();
                    ysum[c] = convert<float>(ys) * 127.0f;
                }
                simd<float, 16> acc[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
#pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    simd<int8_t, NC * 32> Am;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        Am.template select<32, 1>(32 * c) = y8[c].template select<32, 1>(32 * sb);
                    }
                    simd<uint8_t, 512> B8 = Bt[sb].template bit_cast_view<uint8_t>();
                    simd<int, NC * 16> C  = xmx::dpas<8, NC, int, int, uint8_t, int8_t>(simd<int, NC * 16>(0), B8, Am);
                    simd<uint32_t, 16> ls = (sl >> (uint32_t) (4 * sb)) & 0xFu;
                    ls |= ((sh16 >> (uint32_t) (2 * sb)) & 3u) << 4u;
                    simd<float, 16> lsd = (convert<float>(ls) - 32.0f) * d16;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<int, 16>   ci  = C.template select<16, 1>(16 * c);
                        const float     ys  = ysum[c][sb];
                        const float     dyc = dy[c][sb];
                        simd<float, 16> cf  = convert<float>(ci) - ys;
                        acc[c] += cf * (lsd * dyc);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint32_t off = (uint32_t) ((tid * MAXN + slc0 + c) * 64);
                    simd<float, 16> prev = slm_block_load<float, 16>(off);
                    slm_block_store<float, 16>(off, prev + acc[c]);
                }
            }
        }
        barrier();
        for (int col = tid; col < nlim; col += WG) {
            simd<float, 16> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, 16>((uint32_t) ((q * MAXN + col) * 64));
            }
            simd<uint32_t, 16> ridx(row0, 1);
            simd_mask<16>      m = ridx < (uint32_t) nrows;
            scatter<float, 16>(dst + (size_t) (nbase + col) * stride_dst, ridx * 4u, u, m);
        }
        barrier();
    }
}
