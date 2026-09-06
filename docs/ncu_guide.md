# ncu / nsys 使用指南(阶段4/6 的 profiling)

## 1. 采样命令

```bash
# 只测我们的 FA kernel(避开 cast/merge 等小 kernel)
ncu --kernel-name regex:fa_tc_kernel --launch-count 3 \
    --set full build/bin/fa_bench_xxx        # 需要一个可重复调用的二进制
```

由于 fa_bench 内部自测全阶段,建议单独做一个 profile 用的入口
(或直接对 python 测试进程采样,GPU 侧 kernel 名一致):

```bash
ncu --kernel-name regex:fa_tc_kernel -c 5 --set full \
    python python/test_correctness.py --stages 4 --quick
```

关键指标(记入优化记录):

| 指标 | 意义 | 与本文实现的对应 |
|---|---|---|
| Achieved Occupancy | 实际占用率 | 受 smem/block(64~97KB)与寄存器限制,期望 12~50% |
| SM Busy / SM Issue | SM 忙碌率 | softmax 标量段与 mma 段的比例 |
| DRAM Throughput | 显存带宽利用率 | K/V tile 的 cp.async 载入;N=2048,d=64 时理论 ~
  读 2×N²d×2B + 写 Nd×4B |
| Shared Memory Throughput | smem 带宽 | 本实现的 S/P 中转与 O 累加器都在 smem,偏高是预期的 |
| L1/TEX Hit Rate | 缓存命中 | Q tile 每 block 复用、K/V 不跨 block 复用 |
| Warp Cycles Per Issued | warp 停驻原因 | 看下一节的 stall 分布 |
| smsp__warp_issue_stalled_* | 停滞原因细分 | long_scoreboard(等 global/cp)、barrier、short_scoreboard(等 smem)、
  mio_throttle、math_pipe_throttle 等 |

## 2. 从指标定位到代码(预期瓶颈,需实测验证)

1. **global 载入停滞(long_scoreboard / drain)**:阶段3 同步载入 K/V 时所有
   warp 空等;→ 阶段4 用 cp.async 双/三级流水,把 global 延迟藏到计算后面。
   预期阶段4 相对阶段3 的加速主要来自这里(数据待跑)。
2. **smem 吞吐偏高**:tc kernel 的 S(fp32,行距 BN+1)与 O(fp32,行距 D+1)
   中转 + 每 key 块 32 次 wmma B/C fragment 搬运。这是“布局无关 softmax”
   的代价;如 ncu 显示 shared 接近峰值,可尝试把 O 累加迁回寄存器 fragment
   (需要 wmma fragment 行映射知识,见优化记录第 3 轮)。
3. **bank conflict**:Ss 行距 BN+1、Oacc 行距 D+1 已经把半行扫描压到 ~2 路;
   若 ncu 显示 high conflict,可用 `--metrics
   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 量化。
4. **barrier 开销**:阶段3 每 key 块 2 次 `__syncthreads`;阶段4 流水路径
   每块 2 次 barrier + wait_group。block=128 线程只有 4 warps,占用率低时
   barrier 会放大延迟;BN/STAGES 扫描就是在这种约束下找折中。
5. **occupancy**:96KB smem 预算下 1~2 blocks/SM;如限制因素为
   `launch__registers_per_thread`,可用 `--launch-skip` 定位并考虑
   `__launch_bounds__` 收紧寄存器。

## 3. 阶段6:nsys 看启动开销

```bash
nsys profile --stats=true -o build/results/fa6 python python/test_correctness.py --stages 6 --quick
```

关注:
- eager 路径 4 个 kernel 之间的 **CPU launch gap**(每个 ~2-8 us);
- graph 路径单次 `cudaGraphLaunch` 的启动时间(~3-5 us);
- 两者相减即阶段6 报告的“每调用节省启动开销”。

## 4. 复现注意事项

- 计时以 `fa_bench` 输出为准(同流多次 launch 取最小均值),ncu 开销大,
  只用于取指标,不用 ncu 时间当性能数字。
- 数据/种子可复现(`fill_f32` LCG,seed 固定);跑 ncu 前先
  `python scripts/run_all.py --build-only` 保证二进制最新。
