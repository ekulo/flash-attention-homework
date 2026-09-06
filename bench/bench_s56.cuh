// ===========================================================================
// bench_s56.cuh  5( GPU / )6(Graph vs eager)
// ===========================================================================
#pragma once

static int bench_s56(int ndev) {
  const std::vector<int> Ns = {256, 512, 1024, 2048};
  const std::vector<int> Ds = {64, 128};

  // ================= 5:host  =================
  {
    const int nranks = 2;   // ndev>=2 ; rank()
    const char* mode = (ndev >= 2) ? "real-2gpu" : "virtual-2rank";
    for (int d : Ds) {
      for (int N : Ns) {
        std::vector<float> hf((size_t)N * d), hkf((size_t)N * d), hvf((size_t)N * d);
        fill_f32(hf, 0.5f); fill_f32(hkf, 0.5f); fill_f32(hvf, 0.5f);
        std::vector<__half> hq16, hk16, hv16;
        f32_to_f16(hf, hq16); f32_to_f16(hkf, hk16); f32_to_f16(hvf, hv16);
        std::vector<float> ho((size_t)N * d, 0.f);
        const int reps = 5;

        auto run2 = [&]() {
          fa5_forward_f16(hq16.data(), hk16.data(), hv16.data(), ho.data(), N, d, nranks);
        };
        auto run1 = [&]() {
          fa5_forward_f16(hq16.data(), hk16.data(), hv16.data(), ho.data(), N, d, 1);
        };
        run2(); run1();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) run2();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms5 = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        auto b0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) run1();
        auto b1 = std::chrono::high_resolution_clock::now();
        double ms1 = std::chrono::duration<double, std::milli>(b1 - b0).count() / reps;

        double tf, gb;
        stats(N, d, ms5, tf, gb);
        printf("| stage5 %-13s | N=%-5d d=%-3d | %9.4f ms (2r) %9.4f ms (1r) | "
               "%.2fx |\n",
               mode, N, d, ms5, ms1, ms1 / ms5);
        emit_row("stage5", mode, "e2e-host", N, d, ms5, tf, gb, "");
        emit_row("stage5", "baseline-1rank", "e2e-host", N, d, ms1, tf, gb,
                 "single-rank baseline");
      }
    }
  }

  // ================= 6:eager 4-launch vs CUDA Graph =================
  {
    const int calls = 50;
    for (int d : Ds) {
      for (int N : Ns) {
        const size_t nd = (size_t)N * d;
        std::vector<float> hq(nd), hk(nd), hv(nd);
        fill_f32(hq, 0.5f); fill_f32(hk, 0.5f); fill_f32(hv, 0.5f);

        float *q32 = nullptr, *k32 = nullptr, *v32 = nullptr, *o32 = nullptr;
        __half *q16 = nullptr, *k16 = nullptr, *v16 = nullptr;
        CHECK(cudaMalloc(&q32, nd * 4)); CHECK(cudaMalloc(&k32, nd * 4));
        CHECK(cudaMalloc(&v32, nd * 4)); CHECK(cudaMalloc(&o32, nd * 4));
        CHECK(cudaMalloc(&q16, nd * 2)); CHECK(cudaMalloc(&k16, nd * 2));
        CHECK(cudaMalloc(&v16, nd * 2));
        CHECK(cudaMemcpy(q32, hq.data(), nd * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(k32, hk.data(), nd * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(v32, hv.data(), nd * 4, cudaMemcpyHostToDevice));

        // eager :3 cast + fa4(),
        const int blocks = (int)((nd + 255) / 256);
        auto eager_once = [&]() {
          bench_cast_kernel<<<blocks, 256>>>(q32, q16, (int)nd);
          bench_cast_kernel<<<blocks, 256>>>(k32, k16, (int)nd);
          bench_cast_kernel<<<blocks, 256>>>(v32, v16, (int)nd);
          fa_forward(4, -1, q16, k16, v16, o32, N, d, nullptr);
          cudaDeviceSynchronize();
        };

        fa6_handle h = nullptr;
        fa_status st = fa6_create(&h, N, d, q32, k32, v32, o32);
        if (st != FA_OK) {
          printf("[stage6] fa6_create failed at N=%d d=%d (status %d), skip\n",
                 N, d, (int)st);
          cudaFree(q32); cudaFree(k32); cudaFree(v32); cudaFree(o32);
          cudaFree(q16); cudaFree(k16); cudaFree(v16);
          continue;
        }

        // (eager vs graph ; python )
        eager_once();
        std::vector<float> oe(nd), og(nd);
        cudaMemcpy(oe.data(), o32, nd * 4, cudaMemcpyDeviceToHost);
        fa6_run(h);
        cudaMemcpy(og.data(), o32, nd * 4, cudaMemcpyDeviceToHost);
        double maxdiff = 0.0;
        for (size_t i = 0; i < nd; ++i)
          maxdiff = fmax(maxdiff, fabs((double)oe[i] - og[i]));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < calls; ++i) eager_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_eager =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / calls;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < calls; ++i) fa6_run(h);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_graph =
            std::chrono::duration<double, std::milli>(t3 - t2).count() / calls;

        printf("| stage6 | N=%-5d d=%-3d | eager %8.4f ms | graph %8.4f ms | "
               " %5.1f%% | maxdiff %.2e |\n",
               N, d, ms_eager, ms_graph, 100.0 * (1.0 - ms_graph / ms_eager),
               maxdiff);
        emit_row("stage6", "eager", "latency-call", N, d, ms_eager, 0, 0,
                 "3 cast + fa4 + sync");
        emit_row("stage6", "graph", "latency-call", N, d, ms_graph, 0, 0,
                 "cudaGraphLaunch + sync");

        fa6_destroy(h);
        cudaFree(q32); cudaFree(k32); cudaFree(v32); cudaFree(o32);
        cudaFree(q16); cudaFree(k16); cudaFree(v16);
      }
    }
  }
  return 0;
}
