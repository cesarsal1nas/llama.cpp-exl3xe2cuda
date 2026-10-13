# IGC 2.41.5 and GDN prefill

IGC 2.41.5 (LLVM 22) does not auto-unroll two loops in
`gated_delta_net_tcol128` and `gated_delta_net_tcol128_l2`:

```
for (int i = 0; i < S_v; i += 4)
```

`float s[128]` then spills to private memory. On Arc B65 that cuts
prefill about in half. Generation is unchanged.

This tree has `#pragma unroll` on those four loops. Rebuild. No env var.

If the binary is older and still has `GGML_SYCL_GDN_ACC2`, set it to 1.
That is a different kernel, same result. IGC 2.40.x already unrolls these
loops, so that compiler does not need either fix.

`Q8_XMX`, `XE2MX_ACACHE`, and `XE2MX_MG` are separate speed knobs. They
are not this bug.
