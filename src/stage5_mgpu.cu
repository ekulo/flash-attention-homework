// ===========================================================================
// stage5_mgpu.cu - stage 5: multi-GPU split (AllGather K/V + merge).
//
// Strategy (course-grade, single-host multi-GPU and "virtual multi-rank"):
//   * N is split into nranks pieces: rank r owns Q/K/V rows
//     [r*Nloc, (r+1)*Nloc) with Nloc = N / nranks.
//   * Every rank needs all keys to finish its own Q rows, so:
//       1) each rank computes a local partial (O1, m1, l1) on its own chunk
//          (same online-softmax kernel as the full path, restricted range);
//       2) compute stream runs the local partial while the comm stream
//          AllGathers K/V (pull all other ranks' chunks);
//       3) after comm, the same partial kernel processes the remote keys
//          (segments [0, r*Nloc) and [(r+1)*Nloc, N)) -> (O2, m2, l2);
//       4) merge kernel combines the two partials:
//          m=max(m1,m2); a=exp(m1-m); b=exp(m2-m);
//          O=(a*O1+b*O2)/(a*l1+b*l2)
//   * Comm backend: NCCL AllGather (FA_HAVE_NCCL, Linux) or
//     multi-stream cudaMemcpyPeerAsync pulls (Windows/single host).
//     Virtual mode: rank r uses device r % ndev (multi rank per GPU allowed).
//   * A per-rank upload event (ev_up) guarantees pulls only start after every
//     rank's H2D upload completed (fixes a cross-stream race).
//
// Implementation split into small headers (one atomic write each):
//   s5_core.cuh   - merge/fill kernels, Shared registry, Ctx, partial launch
//   s5_worker.cuh - mgpu_worker
// ===========================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <thread>
#include <vector>
#include "fa_common.cuh"
#include "fa_api.h"
#include "fa_wrappers.h"
#include "tc_kernel.cuh"

#ifdef FA_HAVE_NCCL
#include <nccl.h>
#endif

namespace fa5 {

#include "s5_core.cuh"
#include "s5_worker.cuh"

}  // namespace fa5

// ---------------------------------------------------------------------------
// Host entry (blocking; host memory in/out). Constraints:
//   N % (nranks * 64) == 0; d in {16,32,64,128}.
// ---------------------------------------------------------------------------
fa_status fa5_forward_f16(const void* h_q, const void* h_k, const void* h_v,
                          void* h_o, int n, int d, int nranks) {
  if (!h_q || !h_k || !h_v || !h_o) return FA_ERR_PARAM;
  if (nranks <= 0) return FA_ERR_PARAM;
  if (n <= 0 || n % (nranks * 64) != 0) return FA_ERR_PARAM;
  if (d != 16 && d != 32 && d != 64 && d != 128) return FA_ERR_UNSUPPORTED;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return FA_ERR_MULTIGPU;

  fa5::Shared S;
  S.nranks = nranks;
  S.devs.assign(nranks, -1);
  S.kfull.assign(nranks, nullptr);
  S.vfull.assign(nranks, nullptr);
  S.err.assign(nranks, FA_ERR_CUDA);
  S.failed.assign(nranks, 0);
  S.up_ev.assign(nranks, nullptr);

  fa5::Ctx c;
  c.nranks = nranks; c.ndev = ndev; c.N = n; c.d = d;
  c.h_q = h_q; c.h_k = h_k; c.h_v = h_v; c.h_o = h_o;

  std::vector<std::thread> th;
  th.reserve(nranks);
  for (int r = 0; r < nranks; ++r) {
    fa5::Ctx cc = c;
    cc.rank = r;
    th.emplace_back(fa5::mgpu_worker, cc, std::ref(S), r);
  }
  for (auto& t : th) t.join();
  for (int r = 0; r < nranks; ++r)
    if (S.err[r] != FA_OK) return S.err[r];
  return FA_OK;
}
