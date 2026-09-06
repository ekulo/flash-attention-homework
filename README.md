# 高性能 FlashAttention(简化版)—— 大型作业实现

六个阶段递进实现 FlashAttention 前向:`O = softmax(QK^T/√d)·V`,输入
`Q/K/V ∈ [N, d]`,输出 `O ∈ [N, d]`,从不显式物化 S/P(**显存 O(N)**)。

| 阶段 | 内容 | 数据类型 | 关键知识点 | 交付 |
|---|---|---|---|---|
| 1 | 朴素注意力 | FP32 | 线程组织、索引、边界、错误检查 | `src/stage1_naive.cu` |
| 2 | 分块 + 在线 softmax | FP32 | 共享内存 tile、合并访问、bank conflict、同步 | `src/stage2_tiled.cu` |
| 3 | Tensor Core(WMMA m16n16k16) | FP16→FP32 | `__shfl_xor` 归约、wmma、混合精度 | `src/stage3_wmma.cu` + `tc_kernel.cuh` |
| 4 | 调优:BN/STAGES、cp.async 流水 | FP16 | 参数扫描、异步拷贝、多级缓冲 | `src/stage4_tuned.cu` |
| 5 | 多 GPU 切分 | FP16 | partial+merge、NCCL/多流、事件 | `src/stage5_mgpu.cu` |
| 6 | 推理集成 | FP32→FP16 | stream-ordered 内存池、CUDA Graph | `src/stage6_graph.cu` |

## 目录结构

```
├── src/            六个阶段的 kernel + wrapper + 公共头(fa_api.h 为 C ABI)
├── bench/          fa_bench:计时基准,输出 JSONL
├── python/         fa_lib.py(ctypes 加载)+ test_correctness.py(对比 PyTorch SDPA)
├── scripts/        run_all.py(一键构建/测试/基准)、gen_report.py(生成报告表)
├── docs/           代码详解、面试题库、优化记录、性能报告、ncu 用法
├── CMakeLists.txt
└── build_windows.bat
```

## 环境要求

- **Windows**:Visual Studio 2022(x64)+ CUDA Toolkit **≥ 12.8**(RTX 50 系
  sm_120 需要 12.8+)+ CMake ≥ 3.20 + Python 3.8+ + PyTorch(CUDA 版)。
- Linux 同理(CUDA ≥ 12.8);阶段5 若开启 `-DFA_USE_NCCL=ON` 且装有 NCCL,
  使用 `ncclAllGather`,否则使用通用 `cudaMemcpyPeerAsync` 多流 backend。
- 本机为 RTX 5060 Ti → compute capability **12.0**,CMake 默认
  `CMAKE_CUDA_ARCHITECTURES="120"`;其它卡用 `-DFA_ARCHS=...` 覆盖
  (A100/RTX30 系 `80;86`,RTX40 `89`,H100 `90`)。

## 快速开始(Windows)

```bat
:: 1) 在 "x64 Native Tools Command Prompt for VS 2022" 中(或确保 cl/nvcc 在 PATH)
python scripts\run_all.py --arch 120
```

脚本会依次:cmake 配置/编译 → Python 正确性测试(与
`torch.nn.functional.scaled_dot_product_attention` 对比)→ `fa_bench` 计时
→ 生成 `docs/perf_results.md`。

只跑其中一步:

```bat
python scripts\run_all.py --build-only
python scripts\run_all.py --test-only --quick        REM 小尺寸快测
python scripts\run_all.py --bench-only
REM 手动调用 C ABI / 单点调试:见 python\test_correctness.py --help
```

Linux 构建方式相同(`python3 scripts/run_all.py`)。

## 正确性测试与精度

- 阶段1/2(FP32):任意 N、d≤128(模板集合 16/32/64/128),对照 SDPA FP32,
  断言 max abs diff < 2e-4(典型 ~1e-6);并专门测 N=300/d=40 等**非对齐边界**。
- 阶段3/4(FP16 入/FP32 出):要求 `N % 64 == 0`、`d ∈ {16,32,64,128}`,
  对照 SDPA FP16,断言 < 1e-3(满足作业误差要求)。
- 阶段5:单卡以“虚拟 2 rank”验证切分+AllGather+merge 数值;双卡为真实双卡。
- 阶段6:FP32 输入,图内 cast 到 FP16 再计算,对照“先转 FP16 再 SDPA”。

## 支持的配置一览

| 实现 | 输入 | 输出 | N | d | 说明 |
|---|---|---|---|---|---|
| stage1 | fp32 | fp32 | 任意 | 任意 ≤128 | O(N²d²) 朴素,小 N 用 |
| stage2 | fp32 | fp32 | 任意 | 16/32/64/128 | BR=128 或 64,BN=64 或 32 |
| stage3 | fp16 | fp32 | 64 的倍数 | 16/32/64/128 | BN=64(d≤64)/BN=32(d=128) |
| stage4 | fp16 | fp32 | 64 的倍数 | 同上 | 6 个 (BN,STAGES) 变体,-1 自动推荐 |
| stage5 | fp16(host) | fp32(host) | `64*nranks` 的倍数 | 同上 | nranks≤2 虚拟;多卡真实 |
| stage6 | fp32(device) | fp32(device) | 64 的倍数 | 同上 | 池分配 + 图回放 |

## 性能验证(你机器上运行后自动生成;本机实测结果已回填)

> 已在 **RTX 5060 Ti (sm_120, 36 SM) / CUDA 13.1** 完整跑通:76 项正确性
> 测试全部通过(FP32 ~1e-6、FP16 ~8e-5,<1e-3),性能数据已回填
> `docs/perf_results.md`、`docs/optimization_log.md`、`docs/performance_report.md`。

- `fa_bench` 输出 `build/results/bench_results.jsonl`
  (N ∈ {256,512,1024,2048},d ∈ {64,128},全部阶段与 stage4 全部变体)。
- `scripts/gen_report.py` 生成汇总表 `docs/perf_results.md`:
  各阶段耗时/吞吐、stage4 最优变体、阶段2→3→4 加速比、阶段5 多卡加速、
  阶段6 eager vs Graph 延迟。
- **ncu 深度分析方法与指标对照表**:见 `docs/ncu_guide.md`;
  三次调优迭代记录与瓶颈分析:**`docs/optimization_log.md`**(已含本机实测
  数据与真实排障经历)。
- **逐文件代码详解**(布局/公式/流水/踩坑点):**`docs/code_tutorial.md`**;
  配套**面试题库(含答案,按 A~H 分类)**:`docs/interview_qa.md`。
- 最终性能报告(含 256/512/1024/2048 的延迟/吞吐与结论):
  `docs/performance_report.md`。

## 已知限制(课程简化,写进报告更诚实)

1. 阶段3+ 的 tile 分块要求 N 为 64 的倍数(阶段1/2 支持任意 N,边界处理
   知识点在阶段1/2 覆盖);不做 causal mask。
2. 阶段5 的 AllGather-K/V 方案每 rank 需一份完整 K/V 副本,通信量
   O(N·d·nranks),适合教学规模;真正的长序列可改 ring-attention 流水。
3. softmax 的 S/P 经共享内存中转(warp 私有区域),换取对 wmma fragment
   布局零依赖的可读性,付出少量共享内存带宽;阶段4 记录中会量化该开销。
4. 单卡(如 5060 Ti)无法给出阶段5 的真实多卡加速数据——脚本会以
   “虚拟双 rank”验证数值,真实双卡数据需在 ≥2 GPU 机器上运行同一命令。
