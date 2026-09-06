// s5_core.cuh - stage5 shared: merge/fill kernels, Shared registry, Ctx,
//               mgpu_partial dispatcher. Included inside namespace fa5 by
//               stage5_mgpu.cu (single TU).
#pragma once

// Merge two partial results (online-softmax math):
//   m = max(m1,m2); a=exp(m1-m); b=exp(m2-m);
//   O = (a*O1 + b*O2) / (a*l1 + b*l2)
__global__ void fa5_merge_kernel(const float* __restrict__ m1,
                                 const float* __restrict__ l1,
                                 const float* __restrict__ O1,
                                 const float* __restrict__ m2,
                                 const float* __restrict__ l2,
                                 const float* __restrict__ O2,
                                 float* __restrict__ O,
                                 int rows, int D) {
  const int total = rows * D;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int r = idx / D;
  const float m = fmaxf(m1[r], m2[r]);
  const float a = __expf(m1[r] - m);
  const float b = __expf(m2[r] - m);
  O[idx] = (a * O1[idx] + b * O2[idx]) / (a * l1[r] + b * l2[r]);
}

__global__ void fa5_fill_kernel(float* __restrict__ p, int n, float v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// ---------------------------------------------------------------------------
// Cross-thread registry for rank workers (each worker runs in its own thread
// and owns its device buffers).
// ---------------------------------------------------------------------------
struct Shared {
  std::atomic<int> ready{0};     // phase 1: alloc+upload published (nranks)
  int nranks = 0;
  std::vector<int> devs;
  std::vector<__half*> kfull;
  std::vector<__half*> vfull;
  std::vector<fa_status> err;
  std::vector<int> failed;           // set by fail(); synced via ready barrier
  std::vector<cudaEvent_t> up_ev;    // per-rank "upload done" event
#ifdef FA_HAVE_NCCL
  ncclUniqueId nccl_id{};
#endif
  std::atomic<int> ready2{0};        // phase 2: NCCL unique id published
};

struct Ctx {
  int rank, nranks, ndev, N, d;
  const void *h_q, *h_k, *h_v;
  void* h_o;
};

// Dispatch a partial launch by d (TC kernel partial mode, STAGES=1).
fa_status mgpu_partial(const Ctx& c,
                       const __half* q, const __half* kf, const __half* vf,
                       float* O, float* m, float* l,
                       int qrows, int kA0, int kA1, int kB0, int kB1,
                       cudaStream_t s) {
  switch (c.d) {
    case 16:  return fa_tc::fa_tc_launch<16, 64, 1>(q, kf, vf, O, m, l, qrows, kA0, kA1, kB0, kB1, s);
    case 32:  return fa_tc::fa_tc_launch<32, 64, 1>(q, kf, vf, O, m, l, qrows, kA0, kA1, kB0, kB1, s);
    case 64:  return fa_tc::fa_tc_launch<64, 64, 1>(q, kf, vf, O, m, l, qrows, kA0, kA1, kB0, kB1, s);
    case 128: return fa_tc::fa_tc_launch<128, 32, 1>(q, kf, vf, O, m, l, qrows, kA0, kA1, kB0, kB1, s);
    default:  return FA_ERR_UNSUPPORTED;
  }
}
