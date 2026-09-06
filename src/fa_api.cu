// ===========================================================================
// fa_api.cu  extern "C" : (stage, variant)  wrapper
// ===========================================================================
#include "fa_common.cuh"
#include "fa_api.h"
#include "fa_wrappers.h"

extern "C" fa_status fa_forward(int stage, int variant,
                                const void* q, const void* k, const void* v,
                                void* o, int n, int d, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (stage) {
    case 1:  // ,FP32 in/out
      return fa1_fwd_f32(static_cast<const float*>(q), static_cast<const float*>(k),
                         static_cast<const float*>(v), static_cast<float*>(o),
                         n, d, s);
    case 2:  //  +  softmax,FP32 in/out
      return fa2_fwd_f32(static_cast<const float*>(q), static_cast<const float*>(k),
                         static_cast<const float*>(v), static_cast<float*>(o),
                         n, d, s);
    case 3:  // Tensor Core(FP16 in,FP32 out)
      return fa3_fwd_f16(q, k, v, static_cast<float*>(o), n, d, s);
    case 4:  // Tensor Core (FP16 in,FP32 out)
      return fa4_fwd_f16(variant, q, k, v, static_cast<float*>(o), n, d, s);
    default:
      return FA_ERR_PARAM;
  }
}
