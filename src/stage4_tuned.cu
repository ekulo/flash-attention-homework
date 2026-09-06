// ===========================================================================
// stage4_tuned.cu  4: host wrapper
//
//  TC kernel  (BN, STAGES) :
//   variant 0: (BN=64, STAGES=1)     3 ()
//   variant 1: (BN=32, STAGES=1)
//   variant 2: (BN=64, STAGES=2)     cp.async (,d<=64)
//   variant 3: (BN=32, STAGES=2)     cp.async (d=128 )
//   variant 4: (BN=64, STAGES=3)     cp.async 
//   variant 5: (BN=32, STAGES=3)
//   variant -1: (v2 / v3),
//
// (BN*d  1024  -> FA_ERR_UNSUPPORTED;
//  96KB -> FA_ERR_SMEM),bench 
// ===========================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_common.cuh"
#include "fa_api.h"
#include "fa_wrappers.h"
#include "tc_kernel.cuh"

namespace {

template <int D, int BN, int STAGES>
fa_status fa4_run(const void* Q, const void* K, const void* V, float* O,
                  int N, cudaStream_t stream) {
  return fa_tc::fa_tc_launch<D, BN, STAGES>(
      static_cast<const __half*>(Q), static_cast<const __half*>(K),
      static_cast<const __half*>(V), O, nullptr, nullptr, N, 0, N, 0, 0, stream);
}

template <int D>
fa_status fa4_variant(int v, const void* Q, const void* K, const void* V,
                      float* O, int N, cudaStream_t stream) {
  switch (v) {
    case 0: return fa4_run<D, 64, 1>(Q, K, V, O, N, stream);
    case 1: return fa4_run<D, 32, 1>(Q, K, V, O, N, stream);
    case 2: return fa4_run<D, 64, 2>(Q, K, V, O, N, stream);
    case 3: return fa4_run<D, 32, 2>(Q, K, V, O, N, stream);
    case 4: return fa4_run<D, 64, 3>(Q, K, V, O, N, stream);
    case 5: return fa4_run<D, 32, 3>(Q, K, V, O, N, stream);
    default: return FA_ERR_PARAM;
  }
}

}  // namespace

fa_status fa4_fwd_f16(int variant,
                      const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream) {
  if (!Q || !K || !V || !O) return FA_ERR_PARAM;
  if (N <= 0 || N % 64 != 0) return FA_ERR_PARAM;
  const int v = (variant < 0) ? ((d <= 64) ? 2 : 3) : variant;  // 
  switch (d) {
    case 16:  return fa4_variant<16>(v, Q, K, V, O, N, stream);
    case 32:  return fa4_variant<32>(v, Q, K, V, O, N, stream);
    case 64:  return fa4_variant<64>(v, Q, K, V, O, N, stream);
    case 128: return fa4_variant<128>(v, Q, K, V, O, N, stream);
    default:  return FA_ERR_UNSUPPORTED;
  }
}
