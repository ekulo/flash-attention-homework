// s5_worker.cuh - stage5 rank worker (see stage5_mgpu.cu). Included inside
// namespace fa5 by stage5_mgpu.cu (single TU). Depends on s5_core.cuh.
#pragma once

void mgpu_worker(const Ctx& c, Shared& S, int rank) {
  const int dev = rank % c.ndev;       // virtual mode: multiple ranks share a GPU
  S.err[rank] = FA_ERR_CUDA;           // placeholder; FA_OK on success
  // Unified failure exit: a failing rank still bumps both barrier counters so
  // other ranks never spin forever.
  bool phase_a = false, phase_b = false;
  auto fail = [&]() {
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
      fprintf(stderr, "[fa5] rank %d failure, last CUDA error: %s (%d)\n",
              rank, cudaGetErrorString(le), (int)le);
      fflush(stderr);
    }
    S.failed[rank] = 1;
    S.err[rank] = FA_ERR_CUDA;
    if (!phase_a) { phase_a = true; S.ready.fetch_add(1); }
    if (!phase_b) { phase_b = true; S.ready2.fetch_add(1); }
  };
  cudaError_t ce = cudaSetDevice(dev);
  if (ce != cudaSuccess) return fail(), void();
  const int Nloc = c.N / c.nranks;
  const int d = c.d;
  const size_t chunk_h = (size_t)Nloc * d;             // halves per chunk
  const size_t chunk_b = chunk_h * sizeof(__half);     // bytes per chunk

  // Best-effort P2P (failure is fine: memcpyPeer falls back to driver copies).
  if (c.ndev > 1) cudaDeviceEnablePeerAccess((dev + 1) % c.ndev, 0);

  // ---- allocate ----
  __half *q_d = nullptr, *kfull = nullptr, *vfull = nullptr;
  float *O1 = nullptr, *O2 = nullptr, *m1 = nullptr, *l1 = nullptr,
        *m2 = nullptr, *l2 = nullptr;
  cudaStream_t sc = nullptr, scomm = nullptr;
  cudaEvent_t ev_comm = nullptr, ev_up = nullptr;
  if (cudaMalloc(&q_d, chunk_b) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&kfull, (size_t)c.N * d * sizeof(__half)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&vfull, (size_t)c.N * d * sizeof(__half)) != cudaSuccess) return fail(), void();
  const size_t orow = (size_t)Nloc * d;
  if (cudaMalloc(&O1, orow * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&O2, orow * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&m1, (size_t)Nloc * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&l1, (size_t)Nloc * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&m2, (size_t)Nloc * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaMalloc(&l2, (size_t)Nloc * sizeof(float)) != cudaSuccess) return fail(), void();
  if (cudaStreamCreate(&sc) != cudaSuccess) return fail(), void();
  if (cudaStreamCreate(&scomm) != cudaSuccess) return fail(), void();
  if (cudaEventCreateWithFlags(&ev_comm, cudaEventDisableTiming) != cudaSuccess) return fail(), void();
  if (cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming) != cudaSuccess) return fail(), void();

  // ---- upload this rank's Q chunk and its own K/V slot ----
  const char* hq = static_cast<const char*>(c.h_q) + (size_t)rank * chunk_b;
  const char* hk = static_cast<const char*>(c.h_k) + (size_t)rank * chunk_b;
  const char* hv = static_cast<const char*>(c.h_v) + (size_t)rank * chunk_b;
  cudaMemcpyAsync(q_d, hq, chunk_b, cudaMemcpyHostToDevice, sc);
  cudaMemcpyAsync(kfull + (size_t)rank * chunk_h, hk, chunk_b, cudaMemcpyHostToDevice, sc);
  cudaMemcpyAsync(vfull + (size_t)rank * chunk_h, hv, chunk_b, cudaMemcpyHostToDevice, sc);
  cudaEventRecord(ev_up, sc);          // upload done (same stream ordering)

  S.devs[rank] = dev;
  S.kfull[rank] = kfull;
  S.vfull[rank] = vfull;
  S.up_ev[rank] = ev_up;
  phase_a = true;
  S.ready.fetch_add(1);
  while (S.ready.load(std::memory_order_acquire) < c.nranks)
    std::this_thread::yield();
  // If any peer already failed, abort (nothing to read from it).
  for (int p = 0; p < c.nranks; ++p)
    if (S.failed[p]) return fail(), void();
  // CRITICAL: every pull/AllGather must only start after ALL ranks' H2D
  // uploads completed. Wait on every peer's upload event (and our own) with
  // the comm stream.
  for (int p = 0; p < c.nranks; ++p) {
    if (p == rank) continue;
    cudaStreamWaitEvent(scomm, S.up_ev[p], 0);
  }
  cudaStreamWaitEvent(scomm, S.up_ev[rank], 0);

#ifdef FA_HAVE_NCCL
  if (rank == 0 && ncclGetUniqueId(&S.nccl_id) != ncclSuccess) return fail(), void();
  phase_b = true;
  S.ready2.fetch_add(1);
  while (S.ready2.load(std::memory_order_acquire) < c.nranks)
    std::this_thread::yield();
  ncclComm_t comm = nullptr;
  if (ncclCommInitRank(&comm, c.nranks, S.nccl_id, rank) != ncclSuccess) return fail(), void();
#endif

  // ---- local partial on the compute stream ----
  if (mgpu_partial(c, q_d, kfull, vfull, O1, m1, l1, Nloc,
                   rank * Nloc, (rank + 1) * Nloc, 0, 0, sc) != FA_OK) return fail(), void();

  // ---- AllGather: pull other ranks' K/V chunks (comm stream) ----
#ifdef FA_HAVE_NCCL
  {
    ncclResult_t nr =
        ncclAllGather(kfull + (size_t)rank * chunk_h, kfull, (size_t)Nloc * d,
                      ncclHalf, comm, scomm);
    if (nr != ncclSuccess) return fail(), void();
    nr = ncclAllGather(vfull + (size_t)rank * chunk_h, vfull, (size_t)Nloc * d,
                       ncclHalf, comm, scomm);
    if (nr != ncclSuccess) return fail(), void();
  }
#else
  for (int p = 0; p < c.nranks; ++p) {
    if (p == rank) continue;
    const int pdev = S.devs[p];
    const void* psrc_k = S.kfull[p] + (size_t)p * chunk_h;   // peer's own slot
    const void* psrc_v = S.vfull[p] + (size_t)p * chunk_h;
    void* pdst_k = kfull + (size_t)p * chunk_h;
    void* pdst_v = vfull + (size_t)p * chunk_h;
    if (cudaMemcpyPeerAsync(pdst_k, dev, psrc_k, pdev, chunk_b, scomm) != cudaSuccess) return fail(), void();
    if (cudaMemcpyPeerAsync(pdst_v, dev, psrc_v, pdev, chunk_b, scomm) != cudaSuccess) return fail(), void();
  }
#endif
  cudaEventRecord(ev_comm, scomm);
  cudaStreamWaitEvent(sc, ev_comm, 0);

  // ---- remote partial: two key segments in one kernel ----
  if (c.nranks > 1) {
    if (mgpu_partial(c, q_d, kfull, vfull, O2, m2, l2, Nloc,
                     0, rank * Nloc, (rank + 1) * Nloc, c.N, sc) != FA_OK) return fail(), void();
  }
  // nranks == 1: identity partial (m2=-inf, l2=0, O2=0)
  if (c.nranks == 1) {
    fa5_fill_kernel<<<1, 128, 0, sc>>>(m2, Nloc, -FLT_MAX);
    if (cudaMemsetAsync(O2, 0, orow * sizeof(float), sc) != cudaSuccess) return fail(), void();
    if (cudaMemsetAsync(l2, 0, (size_t)Nloc * sizeof(float), sc) != cudaSuccess) return fail(), void();
  }

  // ---- merge and copy back ----
  {
    const int threads = 256;
    const int blocks = (int)((orow + threads - 1) / threads);
    fa5_merge_kernel<<<blocks, threads, 0, sc>>>(m1, l1, O1, m2, l2, O2, O1, Nloc, d);
  }
  cudaMemcpyAsync(static_cast<char*>(c.h_o) + (size_t)rank * orow * sizeof(float),
                  O1, orow * sizeof(float), cudaMemcpyDeviceToHost, sc);
  cudaStreamSynchronize(sc);
  if (cudaGetLastError() != cudaSuccess) return fail(), void();

  S.err[rank] = FA_OK;
  cudaEventDestroy(ev_comm);
  cudaEventDestroy(ev_up);
  cudaStreamDestroy(sc);
  cudaStreamDestroy(scomm);
  cudaFree(q_d); cudaFree(kfull); cudaFree(vfull);
  cudaFree(O1); cudaFree(O2); cudaFree(m1); cudaFree(l1);
  cudaFree(m2); cudaFree(l2);
#ifdef FA_HAVE_NCCL
  ncclCommDestroy(comm);
#endif
}
