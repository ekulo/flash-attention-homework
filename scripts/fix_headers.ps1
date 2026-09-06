# ============================================================================
# fix_headers.ps1  -  Repair headers on disk with canonical ASCII content.
# 背景:某些已存在文件在跨环境同步时内容会丢行(如 fa_api.h 缺枚举闭合行),
# 本脚本用 ASCII 内容直接覆写磁盘文件,绕开同步问题。
# 用法:  powershell -ExecutionPolicy Bypass -File scripts\fix_headers.ps1
# ============================================================================
$root = Split-Path -Parent $PSScriptRoot   # repo root

# ----------------------------------------------------------------------------
# src/fa_api.h
# ----------------------------------------------------------------------------
$fa_api = @'
// fa_api.h - unified C ABI (Python ctypes / fa_bench / fa_api.cu)
// Layout: Q/K/V are row-major [N, d].
//   stage1/2: FP32 in / FP32 out
//   stage3-6: FP16 in / FP32 out (FP32 accumulate), O is N*d*4 bytes
// fa_forward takes device pointers; fa5/fa6 are host-side wrappers.
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

/* Unified forward entry for stages 1-4.
   stage:   1 = naive FP32; 2 = tiled + online softmax FP32;
            3 = TensorCore FP16 baseline; 4 = TensorCore FP16 tuned variants.
   variant: stage1/2/3 must be 0;
            stage4: 0=(BN64,S1) 1=(BN32,S1) 2=(BN64,S2) 3=(BN32,S2)
                    4=(BN64,S3) 5=(BN32,S3) -1=auto pick.
   stream:  may be NULL (legacy default stream); kernels launch on it.
   Constraints: FP16 path requires N % 64 == 0 and N % BN == 0;
                FP16 path d in {16,32,64,128}; FP32 path d in {16,32,64,128}
                (stage1 additionally supports any d <= 128). */
fa_status fa_forward(int stage, int variant,
                     const void* q, const void* k, const void* v, void* o,
                     int n, int d, void* stream);

/* Stage 5: multi-GPU, host-side, blocking.
   h_q/h_k/h_v: full [N,d] FP16 on host; h_o: [N,d] FP32 on host.
   N is split into nranks chunks, requires N % (nranks*64) == 0.
   Backend: NCCL AllGather if built with FA_HAVE_NCCL, otherwise
   multi-stream cudaMemcpyPeerAsync. */
fa_status fa5_forward_f16(const void* h_q, const void* h_k, const void* h_v,
                          void* h_o, int n, int d, int nranks);

/* Stage 6: mem pool + CUDA Graph inference wrapper (host-side, blocking).
   create: fixed device pointers q32/k32/v32 (FP32), output o32 (FP32).
   Internally: create stream-ordered mem pool -> allocate FP16 staging from
   the pool -> capture [castQ, castK, castV, attn] into a CUDA Graph.
   run:    replay once (synchronous). destroy: release graph and pool.
   n/d obey the same FP16 constraints as fa_forward (N % 64 == 0). */
typedef struct fa6_handle_st* fa6_handle;
fa_status fa6_create(fa6_handle* handle, int n, int d,
                     const void* q32, const void* k32, const void* v32, void* o32);
fa_status fa6_run(fa6_handle handle);
fa_status fa6_destroy(fa6_handle handle);

/* Human readable error string. */
const char* fa_status_string(fa_status s);

#ifdef __cplusplus
}
#endif
#endif
'@
Set-Content -Path (Join-Path $root "src\fa_api.h") -Value $fa_api -Encoding Ascii
Write-Host "fixed src/fa_api.h"

# ----------------------------------------------------------------------------
# src/fa_wrappers.h
# ----------------------------------------------------------------------------
$fa_wrappers = @'
// fa_wrappers.h - internal host wrapper declarations
#ifndef FA_WRAPPERS_H
#define FA_WRAPPERS_H
#include "fa_api.h"

/* Stage 1: naive (FP32 in/out, any N, d <= 128). */
fa_status fa1_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream);

/* Stage 2: tiled + online softmax (FP32 in/out, any N, d in {16,32,64,128}). */
fa_status fa2_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream);

/* Stage 3: TensorCore (FP16 in, FP32 out; N % 64 == 0; d in {16,32,64,128}). */
fa_status fa3_fwd_f16(const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream);

/* Stage 4: tuned variants (FP16 in, FP32 out).
   variant: 0=(BN64,S1) 1=(BN32,S1) 2=(BN64,S2) 3=(BN32,S2)
            4=(BN64,S3) 5=(BN32,S3) -1 = auto pick per d. */
fa_status fa4_fwd_f16(int variant,
                      const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream);

#endif
'@
Set-Content -Path (Join-Path $root "src\fa_wrappers.h") -Value $fa_wrappers -Encoding Ascii
Write-Host "fixed src/fa_wrappers.h"

# ----------------------------------------------------------------------------
# src/fa_common.cuh  (ASCII version, same macros/helpers)
# ----------------------------------------------------------------------------
$fa_common = @'
// fa_common.cuh - shared utilities: error-check macros, small device helpers
#ifndef FA_COMMON_CUH
#define FA_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* Host-side error check (use inside functions returning int; -1 on error). */
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err_ = (call);                                                   \
    if (err_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d  %s\n", cudaGetErrorName(err_),    \
              __FILE__, __LINE__, cudaGetErrorString(err_));                     \
      return -1;                                                                 \
    }                                                                            \
  } while (0)

/* Same, but returns FA_ERR_CUDA (for functions returning fa_status). */
#define CUDA_CHECK_S(call)                                                       \
  do {                                                                           \
    cudaError_t err_ = (call);                                                   \
    if (err_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d  %s\n", cudaGetErrorName(err_),    \
              __FILE__, __LINE__, cudaGetErrorString(err_));                     \
      return FA_ERR_CUDA;                                                        \
    }                                                                            \
  } while (0)

/* Async error check right after a kernel launch. */
#define CUDA_LAST_CHECK()                                                        \
  do {                                                                           \
    cudaError_t err_ = cudaGetLastError();                                       \
    if (err_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA launch error %s at %s:%d  %s\n",                     \
              cudaGetErrorName(err_), __FILE__, __LINE__,                        \
              cudaGetErrorString(err_));                                         \
      return FA_ERR_CUDA;                                                        \
    }                                                                            \
  } while (0)

#define FA_DEV_INLINE __device__ __forceinline__

FA_DEV_INLINE int fa_ceil_div(int a, int b) { return (a + b - 1) / b; }

/* Reproducible host-side pseudo-random fill. */
static inline void fill_rand(float* p, size_t n, unsigned seed, float scale = 1.0f) {
  unsigned s = seed ? seed : 12345u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;  /* LCG */
    float u = (float)((s >> 8) & 0xFFFFFF) / (float)(1 << 24);
    p[i] = (u - 0.5f) * 2.0f * scale;
  }
}

#endif
'@
Set-Content -Path (Join-Path $root "src\fa_common.cuh") -Value $fa_common -Encoding Ascii
Write-Host "fixed src/fa_common.cuh"

Write-Host "All headers fixed. Now wipe build/ and rebuild."
