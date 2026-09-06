// tc_p2.cuh - single key-block compute (see tc_kernel.cuh).
// Per warp (16 query rows): S = QK^T -> online softmax -> O += P*V.
// ss/ph/oacc are warp-relative and stored as PACKED 16x16 blocks so every
// wmma load/store uses ldm = 16 (canonical layout).
//   ss:   fp32, block nc at ss + nc*256        (row r, col j -> r*16+j)
//   ph:   fp16, block nc at ph + nc*256
//   oacc: fp32, block cc at oacc + cc*256
// Manual (softmax) passes map: row r = lane>>1, half = lane&1,
// key col c -> block c>>4, in-block col c&15.
#pragma once

template <int D, int BN, int STAGES>
__device__ __forceinline__ void tc_compute_block(
    int kb, const __half* __restrict__ qs,
    const __half* __restrict__ kbuffers, const __half* __restrict__ vbuffers,
    float* __restrict__ ss, __half* __restrict__ ph, float* __restrict__ oacc,
    int warp, int lane, float scale, float& mrun, float& lrun) {
  constexpr int KC     = D / 16;          // d 16-col chunks
  constexpr int NCHUNK = BN / 16;         // 16-key chunks per tile
  const int slot = (kb / BN) % STAGES;
  const __half* kt = kbuffers + (size_t)slot * (BN * D);
  const __half* vt = vbuffers + (size_t)slot * (BN * D);

  // Q A fragments (m16k16 per 16-col chunk, ldm = D).
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[KC];
#pragma unroll
  for (int kc = 0; kc < KC; ++kc)
    wmma::load_matrix_sync(qf[kc], qs + (size_t)(warp * 16) * D + kc * 16, D);

  // ---- S = Q * K^T, one fp32 fragment per 16-key chunk ----
  for (int nc = 0; nc < NCHUNK; ++nc) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sF;
#pragma unroll
    for (int i = 0; i < sF.num_elements; ++i) sF.x[i] = 0.f;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bF;
      wmma::load_matrix_sync(bF, kt + (size_t)(nc * 16) * D + kc * 16, D);
      wmma::mma_sync(sF, qf[kc], bF, sF);
    }
    wmma::store_matrix_sync(ss + nc * 256, sF, 16, wmma::mem_row_major);
  }
  __syncwarp();

  // ---- row max (raw domain), merge two half lanes, online merge (scaled) ----
  const int r = lane >> 1, half = lane & 1;
  const int c0 = half * (BN / 2);
  float rawmax = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < BN / 2; ++i) {
    const int c = c0 + i;
    const int idx = (c >> 4) * 256 + r * 16 + (c & 15);
    rawmax = fmaxf(rawmax, ss[idx]);
  }
  rawmax = fmaxf(rawmax, __shfl_xor_sync(0xffffffffu, rawmax, 1));
  const float mc   = rawmax * scale;
  const float mnew = fmaxf(mrun, mc);
  const float a = __expf(mrun - mnew);
  const float b = __expf(mc - mnew);
  mrun = mnew;

  // ---- rescale whole O row by a (packed blocks; even lane only) ----
  if ((lane & 1) == 0) {
#pragma unroll
    for (int cc = 0; cc < KC; ++cc)
#pragma unroll
      for (int j = 0; j < 16; ++j) oacc[cc * 256 + r * 16 + j] *= a;
  }

  // ---- P = b * exp((S - max) * scale) into fp16 blocks; row sum l ----
  float lc = 0.f;
#pragma unroll
  for (int i = 0; i < BN / 2; ++i) {
    const int c = c0 + i;
    const int idx = (c >> 4) * 256 + r * 16 + (c & 15);
    const float p = b * __expf((ss[idx] - rawmax) * scale);
    lc += p;
    ph[idx] = __float2half_rn(p);
  }
  lc += __shfl_xor_sync(0xffffffffu, lc, 1);
  lrun = a * lrun + lc;
  __syncwarp();

  // ---- O += P * V (packed blocks, ldm = 16) ----
  for (int nc = 0; nc < NCHUNK; ++nc) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pF;
    wmma::load_matrix_sync(pF, ph + nc * 256, 16);
#pragma unroll
    for (int cc = 0; cc < KC; ++cc) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> oF;
      wmma::load_matrix_sync(oF, oacc + cc * 256, 16, wmma::mem_row_major);
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vF;
      wmma::load_matrix_sync(vF, vt + (size_t)(nc * 16) * D + cc * 16, D);
      wmma::mma_sync(oF, pF, vF, oF);
      wmma::store_matrix_sync(oacc + cc * 256, oF, 16, wmma::mem_row_major);
    }
  }
  __syncwarp();
}
