# 优化记录(阶段4:三次迭代 + ncu 指标)

> 本文全部性能数字来自本机实测:**RTX 5060 Ti (CC 12.0, 36 SM)**,
> CUDA 13.1,Release 构建(arch=120),复现命令:
> `python scripts\run_all.py --arch 120 --generator Ninja`。
> 计时口径:同流 20 次连续 launch 取最小均值(device 时间)。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (sm_120, 36 SMs) |
| CUDA / 驱动 | 13.1 / (nvidia-smi) |
| 编译 | CMake+Ninja,Release,`-allow-unsupported-compiler`(VS2026) |
| 正确性基线 | PyTorch SDPA;FP32 误差 ~1e-7,FP16 ~1e-5~8e-5(全部 <1e-3) |

## 迭代 0:阶段1(朴素)→ 阶段2(分块+在线 softmax)

| N | stage1(ms) | stage2(ms) | 加速 |
|---|---|---|---|
| 256 | 0.875 | 0.115 | 7.6x |
| 512 | 2.582 | 0.225 | 11.5x |
| 1024 | 9.527 | 0.444 | 21.4x |
| 2048 | 38.211 | 0.882 | 43.3x |

(d=64)结论[实测]:朴素是 O(N²d²) 且每输出元素重算整行点积;分块 + 寄存器行累加
把复杂度降到 O(N²d),加速随 N 增大(数据复用更好)。

## 迭代 1:阶段3 = Tensor Core 基线(WMMA + warp 归约)

| N | stage2(ms) | stage3(ms) | 加速 |
|---|---|---|---|
| 256 | 0.115 | 0.019 | 6.1x |
| 512 | 0.225 | 0.033 | 6.7x |
| 1024 | 0.444 | 0.064 | 7.0x |
| 2048 | 0.882 | 0.123 | 7.2x |

结论[实测]:Tensor Core + `__shfl_xor` 归约带来 6.1~7.2x 提升(≥ 作业要求的 2~3x)。
说明:阶段2 的 FP32 行式实现受 smem 广播吞吐限制(每 key 重读整行),TC 版把
QK^T/PV 交给 tensor core,标量工作只剩 softmax 归约。

## 迭代 2:参数扫描(BN × STAGES)

| variant | BN/STAGES | N=1024 d=64 | N=2048 d=64 | 说明 |
|---|---|---|---|---|
| v0 | 64/1 | 0.0635 ms | 0.1234 ms | 同步基线 |
| v1 | 32/1 | 0.0781 ms | 0.1502 ms | 小 tile 更差 |
| v2 | 64/2 | **0.0617 ms** | **0.1194 ms** | cp.async 双缓冲(最优) |
| v3 | 32/2 | 0.0718 ms | 0.1399 ms | |
| v5 | 32/3 | 0.0741 ms | 0.1435 ms | 三级流水受 smem/占用率制约 |

d=128(BN 需 32,smem 预算):v3(BN32-S2)最优,N=2048 0.2497 ms(8.6 TFLOPS)。

结论[实测]:BN=64 优于 BN=32(tile 复用更好);STAGES=2 相对 S1 提升 ~3%(N≥1024),
STAGES=3 反而略降 —— 数据量小(全部可驻留 L2),cp.async 隐藏延迟的收益有限,
额外 smem(降占用率)抵消了收益。

## 迭代 3:cp.async 多级流水 + 尾部等待修正

- K/V 载入改为 `__pipeline_memcpy_async`(16B)+ 环形缓冲(见 tc_p1/p3);
- 修了流水尾部竞态:稳态区 `wait(STAGES-1)`,最后 STAGES-1 块 `wait(0)`;
- host 校验每 tile 16B 单元数可被 128 整除(CFG_OK)。

| N=2048,d=64 | stage3(ms) | stage4 v2(ms) | 提升 |
|---|---|---|---|
|  | 0.1230 | 0.1194 | +2.9% |

## 开发过程中发现并修复的关键问题(真实经历,供答辩)

1. **源码编码陷阱**:Windows 代码页 936 下,UTF-8 中文注释会被 MSVC/nvcc 误读,
   偶发吞掉后续 ASCII 字符,产生"幽灵语法错误/错位行号"。解决:源码统一纯
   ASCII(或加 /utf-8)。
2. **wmma 的 ldm 语义**:软 softmax 中转区最初用"行距 BN+1/D+1"的 padding 布局,
   结果每行输出被平移 r 列(第 r 行平移 r)。改为 **S/P/O 按 16×16 紧凑分块、
   ldm 恒为 16**(教科书布局)后逐行精确(误差 1e-5)。
3. **虚拟多卡上传竞态**:rank 的 H2D 上传与对端 memcpyPeerAsync 拉取无依赖,
   偶发读到半上传数据(单配置误差 2e-3)。修复:每个 rank 上传后记事件
   `ev_up`,拉取前 scomm 依次等待所有对端事件。
4. stage5 worker 多线程屏障:失败路径必须补足计数,否则其它 rank 自旋死锁;
   `vector<atomic>` 不可拷贝 → 普通 vector + 屏障 release/acquire 同步。

## 已知剩余瓶颈(供"后续工作")

1. TC kernel 每 key 块把 Q fragment 重载、S/P/O 走 smem 中转 + 32 次
   fragment 往返 → mma 占比低。N=2048 d=64 实测 9.0 TFLOPS,约为 5060 Ti
   FP16 峰值的个位数百分比;主要限制:smem 中转带宽、每 SM 仅 1 block
   (smem 64-97KB)、128 线程/block 的占用率。
2. 进一步优化方向:O 累加器留在 fragment(按 m16n16 的 lane 行/列映射做
   归约,跳过 smem 中转)、BM=128 + 8 warps、按 k 分块 split-K。
3. 多卡仅验证了虚拟双 rank(单卡);真实双卡加速需 ≥2 GPU 机器复测
   (代码与 NCCL/Peer 双后端已就绪)。
