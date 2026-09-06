// ===========================================================================
// tc_p1.cuh   + K/V tile  + pipeline wait( tc_kernel.cuh)
//  tc_kernel.cuh  namespace fa_tc 
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// ( 16B , cp.async / ldmatrix)
// ---------------------------------------------------------------------------
template <int D, int BN, int STAGES>
struct Layout {
  static constexpr int BM      = 64;               // block  query 
  static constexpr int THREADS = 128;              // 4 warps
  static constexpr int WARPS   = 4;
  static constexpr int TILE_H  = BN * D;           //  K/V tile  half 

  static constexpr int Q_BYTES   = BM * D * 2;              // fp16 Q tile
  static constexpr int RING_BYTES = STAGES * TILE_H * 2;    // K( V)
  static constexpr int OFF_Q  = 0;
  static constexpr int OFF_K  = OFF_Q + Q_BYTES;
  static constexpr int OFF_V  = OFF_K + RING_BYTES;
  static constexpr int OFF_SOFT = OFF_V + RING_BYTES;       //  warp  soft 

  // Per-warp soft area: S/P/O stored as packed 16x16 blocks (ldm = 16),
  // matching wmma load/store semantics exactly (no padded rows).
  static constexpr int S_BYTES  = BN * 16 * 4;            // fp32 S blocks
  static constexpr int P_BYTES  = BN * 16 * 2;            // fp16 P blocks
  static constexpr int OA_BYTES = D * 16 * 4;             // fp32 O blocks
  static constexpr int LS_BYTES = 16 * 4;
  static constexpr int W_SS  = 0;
  static constexpr int W_PH  = S_BYTES;
  static constexpr int W_OA  = W_PH + P_BYTES;
  static constexpr int W_LS  = W_OA + OA_BYTES;
  static constexpr int WARP_SOFT = W_LS + LS_BYTES;

  static constexpr int TOTAL = OFF_SOFT + WARPS * WARP_SOFT;
  //  smem (SM80+  opt-in >=99KB, 96KB )
  static constexpr int SMEM_LIMIT = 96 * 1024;
  // : tile  16B  128 
  static constexpr bool CFG_OK = (TILE_H % 1024 == 0);
};

// ---------------------------------------------------------------------------
// K/V tile :STAGES>=2  cp.async + commit(sm80+); 16B 
// ---------------------------------------------------------------------------
template <int D, int BN, int STAGES>
__device__ __forceinline__ void tc_transfer_tile(int kb, int tid,
                                                 const __half* __restrict__ K,
                                                 const __half* __restrict__ V,
                                                 __half* __restrict__ kbuffers,
                                                 __half* __restrict__ vbuffers) {
  constexpr int TILE_H = BN * D;
  constexpr int UNITS  = TILE_H / 8;           //  tile  16B 
  constexpr int PER    = UNITS / 128;          // (CFG_OK>=1)
  const int slot = (STAGES == 1) ? 0 : ((kb / BN) % STAGES);
  __half* kd = kbuffers + (size_t)slot * TILE_H;
  __half* vd = vbuffers + (size_t)slot * TILE_H;
  const __half* ks = K + (size_t)kb * D;
  const __half* vs = V + (size_t)kb * D;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (STAGES >= 2) {
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int idx = tid + u * 128;
      __pipeline_memcpy_async(kd + idx * 8, ks + idx * 8, 16);
      __pipeline_memcpy_async(vd + idx * 8, vs + idx * 8, 16);
    }
    __pipeline_commit();
    return;
  }
#endif
  // (3/5 ,<sm80 )
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int idx = tid + u * 128;
    *reinterpret_cast<float4*>(kd + idx * 8) =
        *reinterpret_cast<const float4*>(ks + idx * 8);
    *reinterpret_cast<float4*>(vd + idx * 8) =
        *reinterpret_cast<const float4*>(vs + idx * 8);
  }
}

template <int STAGES>
__device__ __forceinline__ void tc_wait(int groups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_wait_prior(groups);
#endif
}
