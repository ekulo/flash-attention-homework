// ===========================================================================
// stage6_graph.cu  6:CUDA Graphs + 
//
// : N/d, buffer(),
// (FP32  -> 3  cast  FP16 +  FA kernel) CUDA Graph,
//  fa6_run  cudaGraphLaunch, kernel  CPU 
//
// :
//     stream-ordered mem pool(cudaDeviceCreateMemPool),FP16 staging
//     (cudaMallocFromPoolAsync), handle;
//    ( run );
//
// :capture  kernel launch(cudaMallocAsync  capture 
// ), capture ;
// ===========================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "fa_common.cuh"
#include "fa_api.h"
#include "fa_wrappers.h"

namespace {

//  cast kernel:FP32 -> FP16
__global__ void fa6_cast_kernel(const float* __restrict__ src,
                                __half* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half_rn(src[i]);
}

struct Handle {
  int N = 0, d = 0;
  cudaStream_t stream = nullptr;       // capture / replay stream
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  __half *q16 = nullptr, *k16 = nullptr, *v16 = nullptr;  // pool staging
};

}  // namespace

fa_status fa6_create(fa6_handle* handle, int n, int d,
                     const void* q32, const void* k32, const void* v32,
                     void* o32) {
  if (!handle || !q32 || !k32 || !v32 || !o32) return FA_ERR_PARAM;
  if (n <= 0 || n % 64 != 0) return FA_ERR_PARAM;
  if (d != 16 && d != 32 && d != 64 && d != 128) return FA_ERR_UNSUPPORTED;

  Handle* h = new Handle();
  h->N = n; h->d = d;

  // Allocator: device default stream-ordered mem pool
  // (cudaMallocAsync / cudaFreeAsync; custom pool not required here).
  CUDA_CHECK_S(cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking));

  // ---- 2. allocate FP16 staging from the default pool (N*d*half)----
  const size_t nbytes = (size_t)n * d * sizeof(__half);
  CUDA_CHECK_S(cudaMallocAsync((void**)&h->q16, nbytes, h->stream));
  CUDA_CHECK_S(cudaMallocAsync((void**)&h->k16, nbytes, h->stream));
  CUDA_CHECK_S(cudaMallocAsync((void**)&h->v16, nbytes, h->stream));

  // ---- 3.  [castQ | castK | castV | fa4]  ----
  const int elems = n * d;
  const int blocks = (elems + 255) / 256;
  CUDA_CHECK_S(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal));
  {
    fa6_cast_kernel<<<blocks, 256, 0, h->stream>>>(
        static_cast<const float*>(q32), h->q16, elems);
    fa6_cast_kernel<<<blocks, 256, 0, h->stream>>>(
        static_cast<const float*>(k32), h->k16, elems);
    fa6_cast_kernel<<<blocks, 256, 0, h->stream>>>(
        static_cast<const float*>(v32), h->v16, elems);
    // 4 (d<=64 -> BN64/S2;d=128 -> BN32/S2)
    fa_status s = fa4_fwd_f16(-1, h->q16, h->k16, h->v16,
                              static_cast<float*>(o32), n, d, h->stream);
    if (s != FA_OK) {
      fprintf(stderr, "[fa6] fa4 inside capture failed: %d\n", (int)s);
      cudaStreamEndCapture(h->stream, &h->graph);
      if (h->exec) cudaGraphExecDestroy(h->exec);
      if (h->graph) cudaGraphDestroy(h->graph);
      cudaStreamDestroy(h->stream);
      cudaFreeAsync(h->q16, nullptr);
      cudaFreeAsync(h->k16, nullptr);
      cudaFreeAsync(h->v16, nullptr);
      delete h;
      return s;
    }
  }
  CUDA_CHECK_S(cudaStreamEndCapture(h->stream, &h->graph));
  // ()
  CUDA_CHECK_S(cudaGraphInstantiate(&h->exec, h->graph, nullptr, nullptr, 0));

  *handle = reinterpret_cast<fa6_handle>(h);
  return FA_OK;
}

fa_status fa6_run(fa6_handle handle) {
  if (!handle) return FA_ERR_PARAM;
  Handle* h = reinterpret_cast<Handle*>(handle);
  // : 4  kernel launch + 
  CUDA_CHECK_S(cudaGraphLaunch(h->exec, h->stream));
  CUDA_CHECK_S(cudaStreamSynchronize(h->stream));
  return FA_OK;
}

fa_status fa6_destroy(fa6_handle handle) {
  if (!handle) return FA_ERR_PARAM;
  Handle* h = reinterpret_cast<Handle*>(handle);
  if (h->exec) cudaGraphExecDestroy(h->exec);
  if (h->graph) cudaGraphDestroy(h->graph);
  if (h->stream) cudaStreamDestroy(h->stream);
  // Return staging buffers to the default stream-ordered mem pool.
  cudaFreeAsync(h->q16, nullptr);
  cudaFreeAsync(h->k16, nullptr);
  cudaFreeAsync(h->v16, nullptr);
  delete h;
  return FA_OK;
}

const char* fa_status_string(fa_status s) {
  switch (s) {
    case FA_OK: return "OK";
    case FA_ERR_CUDA: return "CUDA runtime error";
    case FA_ERR_PARAM: return "invalid parameter";
    case FA_ERR_UNSUPPORTED: return "unsupported config (d/variant)";
    case FA_ERR_SMEM: return "shared memory budget exceeded";
    case FA_ERR_MULTIGPU: return "multi-GPU error (device count?)";
    case FA_ERR_NCCL: return "NCCL error";
    case FA_ERR_GRAPH: return "CUDA graph / mem pool error";
    default: return "unknown";
  }
}
