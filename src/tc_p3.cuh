// ===========================================================================
// tc_p3.cuh  fa_tc_kernel  kernel( tc_kernel.cuh )
// BM=64  query/block,4 warps x 16 ; partial  key 
// ===========================================================================
#pragma once

template <int D, int BN, int STAGES>
__global__ void __launch_bounds__(128) fa_tc_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, float* __restrict__ O,
    float* __restrict__ m_out, float* __restrict__ l_out,
    int qrows, int kA0, int kA1, int kB0, int kB1, float scale) {
  using L = Layout<D, BN, STAGES>;
  extern __shared__ char smem[];
  __half* qs       = reinterpret_cast<__half*>(smem + L::OFF_Q);
  __half* kbuffers = reinterpret_cast<__half*>(smem + L::OFF_K);
  __half* vbuffers = reinterpret_cast<__half*>(smem + L::OFF_V);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int qrow0 = blockIdx.x * L::BM;

  char* soft = smem + L::OFF_SOFT + (size_t)warp * L::WARP_SOFT;
  float*  ss   = reinterpret_cast<float*>(soft + L::W_SS);
  __half* ph   = reinterpret_cast<__half*>(soft + L::W_PH);
  float*  oacc = reinterpret_cast<float*>(soft + L::W_OA);
  float*  ls   = reinterpret_cast<float*>(soft + L::W_LS);

  // ----  Q tile( block ,half2 )----
  const __half* gq = Q + (size_t)qrow0 * D;
#pragma unroll 4
  for (int i = tid; i < L::BM * D / 2; i += L::THREADS)
    reinterpret_cast<__half2*>(qs)[i] = reinterpret_cast<const __half2*>(gq)[i];
  __syncthreads();

  // ---- zero O accumulator rows (even lane per row, packed 16x16 blocks)----
  const int r = lane >> 1;
  if ((lane & 1) == 0) {
    for (int cc = 0; cc < D / 16; ++cc)
#pragma unroll
      for (int j = 0; j < 16; ++j) oacc[cc * 256 + r * 16 + j] = 0.f;
  }
  float mrun = -FLT_MAX, lrun = 0.f;

  if (kA1 <= kA0 && kB1 <= kB0) return;   // : key

  if (STAGES >= 2) {
    // ============ (cp.async ;)============
    const int nbA = (kA1 - kA0) / BN;
    int issued = 0;
    for (; issued < STAGES && issued < nbA; ++issued)
      tc_transfer_tile<D, BN, STAGES>(kA0 + issued * BN, tid, K, V, kbuffers, vbuffers);
    for (int blk = 0; blk < nbA; ++blk) {
      // (blk+STAGES<=nbA) wait(STAGES-1);
      // , wait(0) , tile
      if (blk + STAGES <= nbA)
        tc_wait<STAGES>(STAGES - 1);
      else
        tc_wait<STAGES>(0);
      __syncthreads();                    // 
      tc_compute_block<D, BN, STAGES>(kA0 + blk * BN, qs, kbuffers, vbuffers,
                                      ss, ph, oacc, warp, lane, scale, mrun, lrun);
      __syncthreads();                    // 
      if (blk + STAGES < nbA)
        tc_transfer_tile<D, BN, STAGES>(kA0 + (blk + STAGES) * BN, tid, K, V,
                                        kbuffers, vbuffers);
    }
  } else {
    // ============ ( key :3 / 5 partial)============
    for (int seg = 0; seg < 2; ++seg) {
      const int s0 = (seg == 0) ? kA0 : kB0;
      const int s1 = (seg == 0) ? kA1 : kB1;
      if (s1 <= s0) continue;
      for (int kb = s0; kb < s1; kb += BN) {
        tc_transfer_tile<D, BN, STAGES>(kb, tid, K, V, kbuffers, vbuffers);
        __syncthreads();
        tc_compute_block<D, BN, STAGES>(kb, qs, kbuffers, vbuffers, ss, ph, oacc,
                                        warp, lane, scale, mrun, lrun);
        __syncthreads();
      }
    }
  }

  // ---- epilogue: divide by l and write O (packed O blocks -> global) ----
  const bool partial = (m_out != nullptr && l_out != nullptr);
  if ((lane & 1) == 0) {
    ls[r] = partial ? 1.f : (1.f / lrun);
    if (partial) {
      m_out[qrow0 + warp * 16 + r] = mrun;
      l_out[qrow0 + warp * 16 + r] = lrun;
    }
  }
  __syncwarp();

  float* orow = O + (size_t)(qrow0 + warp * 16) * D;
#pragma unroll 4
  for (int i = lane; i < 16 * D; i += 32) {
    const int rr = i / D, col = i - rr * D;   // rr: row within warp (0..15)
    const int blk = col >> 4;                 // packed 16x16 block id
    orow[i] = oacc[blk * 256 + rr * 16 + (col & 15)] * ls[rr];
  }
}
