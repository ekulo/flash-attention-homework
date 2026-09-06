// ===========================================================================
// bench_s34.cuh  3(TensorCore )4()
// ===========================================================================
#pragma once

static int bench_s34(void) {
  const std::vector<int> Ns = {256, 512, 1024, 2048};
  const std::vector<int> Ds = {64, 128};
  const int WARM = 5, REPS = 7, L = 20;

  auto measure_stage = [&](int stage, int variant, const void* q, const void* k,
                           const void* v, void* o, int N, int d) {
    auto fn = [&]() { fa_forward(stage, variant, q, k, v, o, N, d, nullptr); };
    for (int i = 0; i < WARM; ++i) fn();
    return measure_avg(L, REPS, fn);
  };

  for (int d : Ds) {
    for (int N : Ns) {
      // 3/4(FP16 in / FP32 out)
      std::vector<float> hq((size_t)N * d), hk((size_t)N * d), hv((size_t)N * d);
      fill_f32(hq, 0.5f); fill_f32(hk, 0.5f); fill_f32(hv, 0.5f);
      std::vector<__half> hq16, hk16, hv16;
      f32_to_f16(hq, hq16); f32_to_f16(hk, hk16); f32_to_f16(hv, hv16);
      __half *q = nullptr, *k = nullptr, *v = nullptr;
      float* o = nullptr;
      CHECK(cudaMalloc(&q, hq.size() * 2)); CHECK(cudaMalloc(&k, hk.size() * 2));
      CHECK(cudaMalloc(&v, hv.size() * 2)); CHECK(cudaMalloc(&o, hq.size() * 4));
      CHECK(cudaMemcpy(q, hq16.data(), hq.size() * 2, cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(k, hk16.data(), hk.size() * 2, cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(v, hv16.data(), hv.size() * 2, cudaMemcpyHostToDevice));

      double ms3 = measure_stage(3, 0, q, k, v, o, N, d);
      double tf3, gb3; stats(N, d, ms3, tf3, gb3);
      printf("| stage3 | N=%-5d d=%-3d | %9.4f ms | %8.2f TFLOPS |\n", N, d, ms3, tf3);
      emit_row("stage3", "BN64-S1", "device", N, d, ms3, tf3, gb3, "");

      const char* vnames[6] = {"BN64-S1", "BN32-S1", "BN64-S2",
                               "BN32-S2", "BN64-S3", "BN32-S3"};
      for (int variant = 0; variant < 6; ++variant) {
        auto fn4 = [&]() { fa_forward(4, variant, q, k, v, o, N, d, nullptr); };
        for (int i = 0; i < WARM; ++i) fn4();
        fa_status st = FA_OK;
        for (int i = 0; i < 3 && st == FA_OK; ++i)
          st = fa_forward(4, variant, q, k, v, o, N, d, nullptr);
        cudaDeviceSynchronize();
        if (st != FA_OK) continue;   // smem/
        double ms4 = measure_avg(L, REPS, fn4);
        double tf4, gb4; stats(N, d, ms4, tf4, gb4);
        printf("| stage4 v%d %-7s | N=%-5d d=%-3d | %9.4f ms | %8.2f TFLOPS |\n",
               variant, vnames[variant], N, d, ms4, tf4);
        emit_row("stage4", vnames[variant], "device", N, d, ms4, tf4, gb4, "");
      }
      cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
    }
  }
  return 0;
}
