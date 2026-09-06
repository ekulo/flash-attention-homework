// ===========================================================================
// stage1_naive.cu  1:(FP32)
//
// , S/P , O(N*d):
//    max  pass: i  m_i = max_j (q_i . k_j)/sqrt(d)
//    lsum pass: i  l_i = _j exp((q_i . k_j)/sqrt(d) - m_i)
//    out  pass: O[i][j] = (1/l_i) _k exp(...)V[k][j]
//   "",
// ( O(N^2 d^2)),
//
// :( N /  d<=128)
// ===========================================================================
#include <cuda_runtime.h>
#include <cfloat>
#include "fa_common.cuh"
#include "fa_wrappers.h"

#define FA1_BLOCK 256      // 
#define FA1_MAX_D 128      // q 

namespace {

// : i  query  key  k 
__device__ __forceinline__ float fa1_dot(const float* __restrict__ qr,
                                         const float* __restrict__ Krow,
                                         int d) {
  float s = 0.f;
  for (int c = 0; c < d; ++c) s += qr[c] * Krow[c];
  return s;
}

// ---------------------------------------------------------------------------
// pass : block, m_i = max_j s_ij
// ---------------------------------------------------------------------------
__global__ void fa1_max_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               float* __restrict__ m,
                               int N, int d, float scale) {
  __shared__ float red[FA1_BLOCK];
  const int i = blockIdx.x;                 // ( 256 )
  if (i >= N) return;

  //  q ( L1 )
  float qr[FA1_MAX_D];
  const float* qrow = Q + (size_t)i * d;
#pragma unroll 4
  for (int c = 0; c < FA1_MAX_D; ++c) qr[c] = (c < d) ? qrow[c] : 0.f;

  float my = -FLT_MAX;
  for (int j = threadIdx.x; j < N; j += FA1_BLOCK) {
    float s = fa1_dot(qr, K + (size_t)j * d, d) * scale;
    my = fmaxf(my, s);
  }
  // block (max)
  red[threadIdx.x] = my;
  __syncthreads();
  for (int s = FA1_BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) m[i] = red[0];
}

// ---------------------------------------------------------------------------
// pass : block, l_i = _j exp(s_ij - m_i)
// ---------------------------------------------------------------------------
__global__ void fa1_lsum_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ m,
                                float* __restrict__ l,
                                int N, int d, float scale) {
  __shared__ float red[FA1_BLOCK];
  const int i = blockIdx.x;
  if (i >= N) return;

  float qr[FA1_MAX_D];
  const float* qrow = Q + (size_t)i * d;
#pragma unroll 4
  for (int c = 0; c < FA1_MAX_D; ++c) qr[c] = (c < d) ? qrow[c] : 0.f;

  const float mi = m[i];
  float my = 0.f;
  for (int j = threadIdx.x; j < N; j += FA1_BLOCK) {
    float s = fa1_dot(qr, K + (size_t)j * d, d) * scale;
    my += __expf(s - mi);
  }
  red[threadIdx.x] = my;
  __syncthreads();
  for (int s = FA1_BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) l[i] = red[0];
}

// ---------------------------------------------------------------------------
// pass : O[i][j] = (1/l_i) _k exp(s_ik - m_i)V[k][j]
// ---------------------------------------------------------------------------
__global__ void fa1_out_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V,
                               const float* __restrict__ m,
                               const float* __restrict__ l,
                               float* __restrict__ O,
                               int N, int d, float scale) {
  const int i = blockIdx.x;
  const int j = threadIdx.x;                // 
  if (i >= N || j >= d) return;

  float qr[FA1_MAX_D];
  const float* qrow = Q + (size_t)i * d;
#pragma unroll 4
  for (int c = 0; c < FA1_MAX_D; ++c) qr[c] = (c < d) ? qrow[c] : 0.f;

  const float mi = m[i];
  const float inv_l = 1.f / l[i];

  float acc = 0.f;
  for (int k = 0; k < N; ++k) {
    const float* krow = K + (size_t)k * d;   // warp  -> L1 
    float s = fa1_dot(qr, krow, d) * scale;
    acc += __expf(s - mi) * __ldg(&V[(size_t)k * d + j]);
  }
  O[(size_t)i * d + j] = acc * inv_l;
}

}  // namespace

// ---------------------------------------------------------------------------
// host wrapper(m/l  O(N) )
// ---------------------------------------------------------------------------
fa_status fa1_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream) {
  if (!Q || !K || !V || !O) return FA_ERR_PARAM;
  if (N <= 0 || d <= 0 || d > FA1_MAX_D) return FA_ERR_PARAM;

  float scale = 1.f / sqrtf((float)d);

  float *m_d = nullptr, *l_d = nullptr;
  CUDA_CHECK_S(cudaMallocAsync((void**)&m_d, (size_t)N * sizeof(float), stream));
  CUDA_CHECK_S(cudaMallocAsync((void**)&l_d, (size_t)N * sizeof(float), stream));

  dim3 gridA(N);
  fa1_max_kernel<<<gridA, FA1_BLOCK, 0, stream>>>(Q, K, m_d, N, d, scale);
  CUDA_LAST_CHECK();
  fa1_lsum_kernel<<<gridA, FA1_BLOCK, 0, stream>>>(Q, K, m_d, l_d, N, d, scale);
  CUDA_LAST_CHECK();
  fa1_out_kernel<<<gridA, FA1_BLOCK, 0, stream>>>(Q, K, V, m_d, l_d, O, N, d, scale);
  CUDA_LAST_CHECK();

  CUDA_CHECK_S(cudaFreeAsync(m_d, stream));
  CUDA_CHECK_S(cudaFreeAsync(l_d, stream));
  return FA_OK;
}
