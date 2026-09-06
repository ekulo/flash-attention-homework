// ===========================================================================
// tc_kernel.cuh  Tensor Core  FlashAttention  kernel(3/4/5 )
//
// :FlashAttention-1 ( Algorithm 1)
//     Block  BM=64  query(warp  16 );K/V  BN 
//    QK^T  PV  nvcuda::wmma m16n16k16(FP16 FP32 )
//    S/P  softmax( fragment );
//     query,(m, l) ;
//    STAGES>=2:cp.async + ;STAGES==1:(3)
//    partial (m_out/l_out ):5  GPU 
//
//  tc_p1..p4 (/):
//   tc_p1: + K/V tile  + pipeline wait
//   tc_p2: key ( softmax)
//   tc_p3:fa_tc_kernel(__global__  kernel)
//   tc_p4:fa_tc_launch(host , + smem opt-in)
// ===========================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include "fa_common.cuh"
#include "fa_api.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif

namespace fa_tc {
using namespace nvcuda;

#include "tc_p1.cuh"
#include "tc_p2.cuh"
#include "tc_p3.cuh"
#include "tc_p4.cuh"

}  // namespace fa_tc
