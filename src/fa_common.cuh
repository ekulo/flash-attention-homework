/*
 * fa_common.cuh - shared helpers: CUDA error-check macros, device constants.
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* Host-side CUDA error check (returns -1 on failure). */
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d  %s\n", cudaGetErrorName(err_),   \
              __FILE__, __LINE__, cudaGetErrorString(err_));                    \
      return -1;                                                                \
    }                                                                           \
  } while (0)

/* Same, but returns FA_ERR_CUDA (for functions returning fa_status). */
#define CUDA_CHECK_S(call)                                                      \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d  %s\n", cudaGetErrorName(err_),   \
              __FILE__, __LINE__, cudaGetErrorString(err_));                    \
      return FA_ERR_CUDA;                                                       \
    }                                                                           \
  } while (0)

/* Asynchronous error check right after a kernel launch. */
#define CUDA_LAST_CHECK()                                                       \
  do {                                                                          \
    cudaError_t err_ = cudaGetLastError();                                      \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA launch error %s at %s:%d  %s\n",                    \
              cudaGetErrorName(err_), __FILE__, __LINE__,                       \
              cudaGetErrorString(err_));                                        \
      return FA_ERR_CUDA;                                                       \
    }                                                                           \
  } while (0)

#define FA_DEV_INLINE __device__ __forceinline__

FA_DEV_INLINE int fa_ceil_div(int a, int b) { return (a + b - 1) / b; }

/* Reproducible pseudo-random host fill (LCG). */
static inline void fill_rand(float* p, size_t n, unsigned seed, float scale = 1.0f) {
  unsigned s = seed ? seed : 12345u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    float u = (float)((s >> 8) & 0xFFFFFF) / (float)(1 << 24);
    p[i] = (u - 0.5f) * 2.0f * scale;
  }
}

static inline float host_rand(float* p, size_t i, float scale = 1.0f) { return p[i]; }
