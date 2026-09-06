// ===========================================================================
// bench_s12.cuh  1()2(+softmax)
// ===========================================================================
#pragma once

static int bench_s12(void) {
  const std::vector<int> Ns = {256, 512, 1024, 2048};
  const std::vector<int> Ds = {64, 128};
  const int WARM = 5, REPS = 7, L = 20;

  auto measure_stage = [&](int stage, const void* q, const void* k,
                           const void* v, void* o, int N, int d) {
    auto fn = [&]() { fa_forward(stage, 0, q, k, v, o, N, d, nullptr); };
    for (int i = 0; i < WARM; ++i) fn();
    return measure_avg(L, REPS, fn);
  };

  for (int d : Ds) {
    for (int N : Ns) {
      // 1:(,d=128  N>512 )
      if (d == 64 || N <= 512) {
        std::vector<float> hq((size_t)N * d), hk((size_t)N * d), hv((size_t)N * d);
        fill_f32(hq, 0.5f); fill_f32(hk, 0.5f); fill_f32(hv, 0.5f);
        float *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        CHECK(cudaMalloc(&q, hq.size() * 4)); CHECK(cudaMalloc(&k, hk.size() * 4));
        CHECK(cudaMalloc(&v, hv.size() * 4)); CHECK(cudaMalloc(&o, hq.size() * 4));
        CHECK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(k, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(v, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
        double ms = measure_stage(1, q, k, v, o, N, d);
        double tf, gb; stats(N, d, ms, tf, gb);
        printf("| stage1 | N=%-5d d=%-3d | %9.4f ms | %8.2f TFLOPS |\n", N, d, ms, tf);
        emit_row("stage1", "-", "device", N, d, ms, tf, gb, "");
        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
      }

      // 2(FP32  +  softmax)
      {
        std::vector<float> hq((size_t)N * d), hk((size_t)N * d), hv((size_t)N * d);
        fill_f32(hq, 0.5f); fill_f32(hk, 0.5f); fill_f32(hv, 0.5f);
        float *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        CHECK(cudaMalloc(&q, hq.size() * 4)); CHECK(cudaMalloc(&k, hk.size() * 4));
        CHECK(cudaMalloc(&v, hv.size() * 4)); CHECK(cudaMalloc(&o, hq.size() * 4));
        CHECK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(k, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(v, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
        double ms = measure_stage(2, q, k, v, o, N, d);
        double tf, gb; stats(N, d, ms, tf, gb);
        printf("| stage2 | N=%-5d d=%-3d | %9.4f ms | %8.2f TFLOPS |\n", N, d, ms, tf);
        emit_row("stage2", "-", "device", N, d, ms, tf, gb, "");
        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
      }
    }
  }
  return 0;
}
