// ===========================================================================
// bench_util.cuh  ( bench_main.cu )
// ===========================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "fa_api.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d %s\n", cudaGetErrorName(e_),   \
              __FILE__, __LINE__, cudaGetErrorString(e_));                   \
      return -1;                                                             \
    }                                                                        \
  } while (0)

static FILE* g_json = nullptr;

// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
static unsigned g_seed = 20240601u;
static float frand(float scale) {
  g_seed = g_seed * 1664525u + 1013904223u;
  float u = (float)((g_seed >> 8) & 0xFFFFFF) / (float)(1 << 24);
  return (u - 0.5f) * 2.f * scale;
}
static void fill_f32(std::vector<float>& v, float scale) {
  for (auto& x : v) x = frand(scale);
}
static void f32_to_f16(const std::vector<float>& src, std::vector<__half>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = __float2half_rn(src[i]);
}

// ---------------------------------------------------------------------------
// : L  launch,(ms)
// ---------------------------------------------------------------------------
static double measure_avg(int L, int reps, const std::function<void()>& fn) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  double sum = 0.0, best = 1e18;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(e0);
    for (int i = 0; i < L; ++i) fn();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double per = ms / L;
    sum += per;
    if (per < best) best = per;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return best;  // 
}

// ---------------------------------------------------------------------------
// JSONL 
// ---------------------------------------------------------------------------
static void emit_row(const char* stage, const char* variant, const char* mode,
                     int N, int d, double ms, double tflops, double gbps,
                     const char* note) {
  if (g_json) {
    fprintf(g_json,
            "{\"stage\":\"%s\",\"variant\":\"%s\",\"mode\":\"%s\",\"N\":%d,"
            "\"d\":%d,\"ms\":%.5f,\"tflops\":%.2f,\"gbps\":%.2f,\"note\":\"%s\"}\n",
            stage, variant, mode, N, d, ms, tflops, gbps, note ? note : "");
    fflush(g_json);
  }
}

// flops = 4*N*N*d(); =  3Nd*2B +  Nd*4B(fp16 )
static void stats(int N, int d, double ms, double& tflops, double& gbps) {
  const double sec = ms * 1e-3;
  const double fl = 4.0 * (double)N * N * d;
  const double by = ((double)N * d) * (3 * 2 + 4);
  tflops = fl / sec / 1e12;
  gbps = by / sec / 1e9;
}

__global__ void bench_cast_kernel(const float* __restrict__ src,
                                  __half* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half_rn(src[i]);
}
