// ===========================================================================
// bench_main.cu  
//
// :fa_bench [.jsonl]( bench_results.jsonl,)
//
// :
//   1/2/3/4  device (cudaEvent, launch )
//   4         (BN, STAGES)  + 
//   5        host ( H2D/D2H ); nranks=2 
//   6        eager 4-launch  vs CUDA Graph ()
//
// (bench_util/s12/s34/s56),
// ===========================================================================
#include "bench_util.cuh"
#include "bench_s12.cuh"
#include "bench_s34.cuh"
#include "bench_s56.cuh"

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "bench_results.jsonl";
  g_json = fopen(out_path.c_str(), "w");
  if (!g_json) {
    fprintf(stderr, "cannot open %s\n", out_path.c_str());
    return -1;
  }

  int ndev = 0;
  CHECK(cudaGetDeviceCount(&ndev));
  if (ndev <= 0) {
    fprintf(stderr, "no CUDA device\n");
    return -1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("== GPU: %s (CC %d.%d), SMs=%d, =%d ==\n", prop.name,
         prop.major, prop.minor, prop.multiProcessorCount, ndev);
  fprintf(g_json,
          "{\"stage\":\"header\",\"device\":\"%s\",\"cc\":\"%d.%d\","
          "\"sms\":%d,\"ndev\":%d}\n",
          prop.name, prop.major, prop.minor, prop.multiProcessorCount, ndev);

  if (bench_s12() != 0) return -1;
  if (bench_s34() != 0) return -1;
  printf("\n== 5( GPU)==\n");
  if (bench_s56(ndev) != 0) return -1;

  fclose(g_json);
  printf("==  %s ==\n", out_path.c_str());
  return 0;
}
