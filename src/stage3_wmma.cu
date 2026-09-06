// ===========================================================================
// stage3_wmma.cu  3:Tensor Core  FlashAttention()
//
//  FP16, FP32(FP32 ):BM=64(block=128 ,4 warps),
// BN=64(d<=64) BN=32(d=128,),STAGES=1( K/V )
//
// (2 ):
//    QK^T / PV  wmma m16n16k16,Tensor Core  ~  FP32 SIMT;
//     softmax  max/exp/,;
//    :softmax  warp  __shfl_xor(2  __syncthreads +
//     ), block 
// ===========================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_common.cuh"
#include "fa_api.h"
#include "fa_wrappers.h"
#include "tc_kernel.cuh"

namespace {

// d<=64  BN=64;d=128  BN=32()
template <int D>
struct S3 { static constexpr int BN = (D <= 64) ? 64 : 32; };

template <int D>
fa_status fa3_dispatch(const __half* Q, const __half* K, const __half* V,
                       float* O, int N, cudaStream_t stream) {
  //  [0, N)
  return fa_tc::fa_tc_launch<D, S3<D>::BN, 1>(
      Q, K, V, O, nullptr, nullptr, N, 0, N, 0, 0, stream);
}

}  // namespace

fa_status fa3_fwd_f16(const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream) {
  if (!Q || !K || !V || !O) return FA_ERR_PARAM;
  if (N <= 0 || N % 64 != 0) return FA_ERR_PARAM;      // 
  const auto* hq = static_cast<const __half*>(Q);
  const auto* hk = static_cast<const __half*>(K);
  const auto* hv = static_cast<const __half*>(V);
  switch (d) {
    case 16:  return fa3_dispatch<16>(hq, hk, hv, O, N, stream);
    case 32:  return fa3_dispatch<32>(hq, hk, hv, O, N, stream);
    case 64:  return fa3_dispatch<64>(hq, hk, hv, O, N, stream);
    case 128: return fa3_dispatch<128>(hq, hk, hv, O, N, stream);
    default:  return FA_ERR_UNSUPPORTED;
  }
}
