// ===========================================================================
// tc_p4.cuh  host launch ( +  smem opt-in)
// ===========================================================================
#pragma once

template <int D, int BN, int STAGES>
fa_status fa_tc_launch(const __half* Q, const __half* K, const __half* V,
                       float* O, float* m_out, float* l_out,
                       int qrows, int kA0, int kA1, int kB0, int kB1,
                       cudaStream_t stream) {
  using L = Layout<D, BN, STAGES>;
  if (!L::CFG_OK) return FA_ERR_UNSUPPORTED;
  if (L::TOTAL > L::SMEM_LIMIT) return FA_ERR_SMEM;
  if (!Q || !K || !V || !O) return FA_ERR_PARAM;
  if (qrows <= 0 || qrows % L::BM != 0) return FA_ERR_PARAM;
  if (kA0 % BN != 0 || kA1 % BN != 0 || kB0 % BN != 0 || kB1 % BN != 0)
    return FA_ERR_PARAM;
  if (STAGES >= 2 && (kA1 - kA0) <= 0) return FA_ERR_PARAM;  // 
  if (STAGES >= 2 && kB1 > kB0) return FA_ERR_PARAM;         // 

  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(fa_tc_kernel<D, BN, STAGES>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         L::TOTAL);
    if (e != cudaSuccess) {
      fprintf(stderr, "[fa_tc] cudaFuncSetAttribute failed: %s (smem %d B)\n",
              cudaGetErrorString(e), L::TOTAL);
      return FA_ERR_SMEM;
    }
    attr_done = true;
  }
  dim3 grid(qrows / L::BM);
  fa_tc_kernel<D, BN, STAGES><<<grid, L::THREADS, L::TOTAL, stream>>>(
      Q, K, V, O, m_out, l_out, qrows, kA0, kA1, kB0, kB1, 1.f / sqrtf((float)D));
  CUDA_LAST_CHECK();
  return FA_OK;
}
