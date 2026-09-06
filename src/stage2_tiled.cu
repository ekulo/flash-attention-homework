// ===========================================================================
// stage2_tiled.cu  2: +  Softmax(FP32)
//
// ( FlashAttention ):
//     Block  BR  query(Q tile), key;
//     Q/K/V  tile, O(N), S/P
//     key j:  m' = max(m, s_j);  = exp(m - m');  = exp(s_j - m');
//       o = o + v_j;  l = l + 
//      max/sum/( softmax),
//     query:d ; Q tile 
//      O ( smem)
//    K/V tile  warp  -> smem ;
//      +1 padding  Q / 32  bank conflict
//    / key  ->  N
// ===========================================================================
#include <cuda_runtime.h>
#include "fa_common.cuh"
#include "fa_wrappers.h"

namespace {

constexpr int kPad = 1;  //  padding 1  float, bank conflict

template <int D, int BR, int BN>
__global__ void __launch_bounds__(BR) fa2_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int N, int d, float scale) {
  // ( +1 padding)
  extern __shared__ float smem[];
  const int DP = D + kPad;
  float* qs   = smem;                 // BR(D+1)   Q tile, O 
  float* ks   = qs + BR * DP;         // BN(D+1)  K tile
  float* vs   = ks + BN * DP;         // BN(D+1)  V tile
  float* invl = vs + BN * DP;         // BR: 1/l

  const int row0 = blockIdx.x * BR;               //  block  query 
  const int rows = min(BR, N - row0);             // ()
  const int t    = threadIdx.x;                   //  t  row0+t

  // ---- 1.  Q tile(global ,)----
  for (int i = t; i < rows * D; i += BR) {
    int r = i / D, c = i - r * D;
    qs[r * DP + c] = Q[(size_t)row0 * D + i];
  }
  __syncthreads();  //  qs 

  // ---- 2. "" q , O ( qs )----
  float qr[D];
  float* orow = qs + (size_t)t * DP;
  if (t < rows) {
#pragma unroll
    for (int c = 0; c < D; ++c) { qr[c] = qs[t * DP + c]; orow[c] = 0.f; }
  }

  // ---- 3. : key( BN )----
  float m = -INFINITY, l = 0.f;
  const float* kbase = K;
  const float* vbase = V;
  for (int kb = 0; kb < N; kb += BN) {
    const int bn = min(BN, N - kb);              //  key 

    // 3.1  K/V tile( + smem )
    for (int i = t; i < bn * D; i += BR) {
      int r = i / D, c = i - r * D;
      ks[r * DP + c] = kbase[(size_t)kb * D + i];
    }
    for (int i = t; i < bn * D; i += BR) {
      int r = i / D, c = i - r * D;
      vs[r * DP + c] = vbase[(size_t)kb * D + i];
    }
    __syncthreads();  // K/V tile 

    // 3.2  softmax: key  (m, l, o )
    if (t < rows) {
      for (int j = 0; j < bn; ++j) {
        const float* krow = ks + (size_t)j * DP;   // 
        const float* vrow = vs + (size_t)j * DP;
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < D; ++c) s += qr[c] * krow[c];
        s *= scale;

        const float mnew = fmaxf(m, s);
        const float alpha = __expf(m - mnew);
        const float beta  = __expf(s - mnew);
#pragma unroll
        for (int c = 0; c < D; ++c) orow[c] = orow[c] * alpha + beta * vrow[c];
        l = l * alpha + beta;
        m = mnew;
      }
    }
    __syncthreads();  //  K/V tile
  }

  // ---- 4. ()----
  if (t < rows) invl[t] = 1.f / l;
  __syncthreads();
  for (int i = t; i < rows * D; i += BR) {
    int r = i / D, c = i - r * D;
    O[(size_t)row0 * D + i] = qs[r * DP + c] * invl[r];
  }
}

//  launch: smem 
template <int D, int BR, int BN>
fa_status launch2(const float* Q, const float* K, const float* V, float* O,
                  int N, cudaStream_t stream) {
  const size_t smem = ((size_t)BR + 2 * BN) * (D + kPad) * sizeof(float) + BR * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(fa2_kernel<D, BR, BN>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)smem);
    if (e != cudaSuccess) {
      fprintf(stderr, "cudaFuncSetAttribute failed: %s (smem=%zu)\n",
              cudaGetErrorString(e), smem);
      return FA_ERR_SMEM;
    }
    attr_set = true;
  }
  dim3 grid((N + BR - 1) / BR);
  fa2_kernel<D, BR, BN><<<grid, BR, smem, stream>>>(
      Q, K, V, O, N, D, 1.f / sqrtf((float)D));
  CUDA_LAST_CHECK();
  return FA_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// host wrapper: d  (BR, BN) ; N( 64 )
//   d=16/32/64 -> BR=128, BN=64;d=128 -> BR=64, BN=32(smem )
// ---------------------------------------------------------------------------
fa_status fa2_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream) {
  if (!Q || !K || !V || !O) return FA_ERR_PARAM;
  if (N <= 0) return FA_ERR_PARAM;
  switch (d) {
    case 16:  return launch2<16, 128, 64>(Q, K, V, O, N, stream);
    case 32:  return launch2<32, 128, 64>(Q, K, V, O, N, stream);
    case 64:  return launch2<64, 128, 64>(Q, K, V, O, N, stream);
    case 128: return launch2<128, 64, 32>(Q, K, V, O, N, stream);
    default:  return FA_ERR_UNSUPPORTED;
  }
}
