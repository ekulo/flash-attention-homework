# fix_fa_api.ps1 - Rewrite src/fa_api.h on disk with canonical ASCII content.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\fix_fa_api.ps1
$root = Split-Path -Parent $PSScriptRoot
$dst = Join-Path $root "src\fa_api.h"
$content = @'
/*
 * fa_api.h - unified C ABI used by python/ctypes, fa_bench and all stages.
 * Layout: Q/K/V are [N, d] row-major.
 *   stage 1/2: FP32 in, FP32 out.
 *   stage 3-6: FP16 in, FP32 out (FP32 accumulate). O is N*d*4 bytes.
 *   fa_forward takes device pointers; fa5/fa6 are host-side wrappers.
 */
#ifndef FA_API_H
#define FA_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FA_OK = 0,
  FA_ERR_CUDA = 1,
  FA_ERR_PARAM = 2,
  FA_ERR_UNSUPPORTED = 3,
  FA_ERR_SMEM = 4,
  FA_ERR_MULTIGPU = 5,
  FA_ERR_NCCL = 6,
  FA_ERR_GRAPH = 7
} fa_status;

/* Unified forward entry (stages 1-4).
 * stage: 1 naive fp32, 2 tiled+online-softmax fp32,
 *        3 tensor-core fp16 (baseline), 4 tensor-core fp16 (tuned variants).
 * variant: stages 1-3 must be 0.
 *   stage4: 0=(BN64,S1) 1=(BN32,S1) 2=(BN64,S2) 3=(BN32,S2)
 *           4=(BN64,S3) 5=(BN32,S3) -1=auto pick by d.
 * stream: may be NULL (default stream); kernels launch async on it.
 * Constraints: fp16 path requires N%64==0 and N%BN==0;
 *   d in {16,32,64,128} (fp16), d in {16,32,64,128} (fp32 stage2),
 *   stage1 additionally accepts any d<=128.
 */
fa_status fa_forward(int stage, int variant,
                     const void* q, const void* k, const void* v, void* o,
                     int n, int d, void* stream);

/* Stage 5: multi-GPU (host-side, blocking).
 * q/k/v are full [N, d] FP16 on host; o is [N, d] FP32 host output.
 * N is split into nranks pieces: N % (nranks*64) == 0 required.
 */
fa_status fa5_forward_f16(const void* h_q, const void* h_k, const void* h_v,
                          void* h_o, int n, int d, int nranks);

/* Stage 6: mem pool + CUDA Graph inference wrapper (host-side, blocking).
 * create takes fixed device pointers q32/k32/v32 (fp32) and o32 (fp32).
 * run replays the graph once (blocking). destroy frees graph and pool.
 */
typedef struct fa6_handle_st* fa6_handle;
fa_status fa6_create(fa6_handle* handle, int n, int d,
                     const void* q32, const void* k32, const void* v32, void* o32);
fa_status fa6_run(fa6_handle handle);
fa_status fa6_destroy(fa6_handle handle);

/* Human readable status text. */
const char* fa_status_string(fa_status s);

#ifdef __cplusplus
}
#endif

#endif /* FA_API_H */
'@
Set-Content -Path $dst -Value $content -Encoding ASCII
Write-Host "fixed: $dst"
