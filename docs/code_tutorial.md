# 代码详解:高性能 FlashAttention(简化版)

> 本文与仓库内源码逐文件对应,建议边开代码边读。源码中的行内注释为了绕开
> Windows 代码页/编译器编码问题被精简为 ASCII(见 `docs/optimization_log.md`
> “源码编码陷阱”),**完整的设计意图、数学推导与逐段讲解以本文为准**。
>
> 配套文档:`docs/interview_qa.md`(面试问题与答案)、`docs/perf_results.md`(实测
> 数据)、`docs/optimization_log.md`(调优记录)。

---

## 1. 仓库地图与阅读顺序

```
src/fa_common.cuh   错误检查宏 ×3、LCG 随机数、ceil_div(被所有 TU 包含)
src/fa_api.h/.cu    C ABI 统一入口 fa_forward(fa_api.cu 按 stage 分发给各 wrapper)
src/fa_wrappers.h   每阶段的 host wrapper 声明(内部接口)
src/stage1_naive.cu 阶段1:朴素三遍扫描(FP32)
src/stage2_tiled.cu 阶段2:共享内存分块 + 在线 softmax(FP32)
src/tc_kernel.cuh   阶段3/4/5 共用的 Tensor Core 内核(组织头,include p1..p4)
src/tc_p1.cuh       Layout:共享内存预算元结构;K/V tile 搬运(cp.async/同步)
src/tc_p2.cuh       单 key 块计算:QK^T → 行 max/softmax → PV(wmma)
src/tc_p3.cuh       fa_tc_kernel 主循环(流水/同步/尾块/输出)
src/tc_p4.cuh       fa_tc_launch host 启动(校验 + 动态共享内存 opt-in)
src/stage3_wmma.cu  阶段3 wrapper:STAGES=1 同步基线的 (BN) 选择
src/stage4_tuned.cu 阶段4 wrapper:(BN, STAGES) × 6 变体,可自动选择
src/s5_core.cuh     阶段5:merge/fill 内核、跨线程共享注册表、partial 分发
src/s5_worker.cuh   阶段5:每个 rank 一个 host 线程的完整执行流程
src/stage5_mgpu.cu  阶段5:对外入口 + 线程池组织(含 NCCL 开关)
src/stage6_graph.cu 阶段6:FP32→FP16 cast + CUDA Graph 捕获/回放 + 内存池
bench/bench_*.cu[h] fa_bench 基准程序(JSONL 输出,各阶段计时口径不同)
python/fa_lib.py    ctypes 绑定(自动找 dll,张量友好封装)
python/test_correctness.py 与 PyTorch SDPA 对比的全部测试用例
scripts/run_all.py  一键:cmake 配置 → 编译 → 测试 → 基准 → 报告
scripts/gen_report.py 由 bench JSONL 生成 docs/perf_results.md
CMakeLists.txt      共享库 fa_api + 可执行 fa_bench;arch 默认 120
```

**推荐的阅读顺序**(也是答辩讲解顺序):

1. `fa_common.cuh` + `fa_api.h`:错误处理哲学与接口约定(最容易答的题都在这里);
2. `stage1_naive.cu` → `stage2_tiled.cu`:从“完全没优化”到“分块 + 在线 softmax”;
3. `tc_kernel.cuh` 的四个分片(p1 布局/搬运 → p2 计算 → p3 主循环 → p4 启动);
4. `stage3/stage4` wrapper:如何用模板参数拼出 6 个变体;
5. `s5_*` 与 `stage6_graph.cu`:多卡与推理部署;
6. 最后看 `bench/` 和 `python/`:理解“数字是怎么测出来的”。

贯穿全局的三个约定(面试常问):

| 约定 | 内容 | 原因 |
|---|---|---|
| 布局 | Q/K/V/O 全部 `[N, d]` 行主序 | ctypes/PyTorch 零拷贝对接 |
| 精度 | 阶段1/2 FP32→FP32;阶段3+ FP16 输入、FP32 累加与输出 | Tensor Core FP16 路径 |
| 约束 | FP16 路径要求 `N % 64 == 0`,`d ∈ {16,32,64,128}` | 分块对齐 wmma 16×16 |

---

## 2. 公共层详解

### 2.1 错误检查宏(fa_common.cuh)

三个宏对应三种“返回值形状”的函数:

```cpp
#define CUDA_CHECK(call)   /* 出错打印后 return -1(给返回 int 的函数)   */
#define CUDA_CHECK_S(call) /* 出错打印后 return FA_ERR_CUDA(给 fa_status) */
#define CUDA_LAST_CHECK()  /* launch 之后的 cudaGetLastError() 检查      */
```

要点(面试):

- CUDA Runtime 的错误分为两类:**同步错误**(参数错误等,立即返回)与**异步错误**
  (kernel 执行期错误,在下一个同步点才报告,且是“粘性”的)。所以光检查返回值
  不够,`<<<>>>` 之后要调 `cudaGetLastError()` 把本次 launch 的错误“取走”,
  否则错误会一直粘到后面的 API。
- 宏打印 `cudaGetErrorName + __FILE__ + __LINE__ + cudaGetErrorString`,
  靠 `do { } while(0)` 包裹保证在 `if/else` 里安全展开。
- 阶段 5 的多线程场景无法用宏(失败要“补足屏障计数”再退出),见 §7.3。

### 2.2 C ABI 与分发(fa_api.h / fa_api.cu / fa_wrappers.h)

`fa_forward(int stage, int variant, q, k, v, o, n, d, stream)` 是唯一对外入口:

- 为什么用 C ABI + `void*`:Python `ctypes` 不需要 C++ 名字修饰,一个
  `fa_api.dll` 就能被 `ctypes.CDLL` 加载(见 `python/fa_lib.py`,每个指针参数
  传 `tensor.data_ptr()`,全程零拷贝)。
- `stream` 参数允许外部(如 torch 的当前流)传入,内部所有 launch 都挂在该流上,
  保证与调用方代码的流序一致(阶段 6 正是靠这个把 `fa4_fwd_f16` 合法地捕获进
  CUDA Graph)。
- 返回 `fa_status` 枚举而不是裸 `int`:8 个错误码自带语义,Python 端映射成
  可读字符串(见 `fa_lib.py` 的 `_FA_STATUS`)。
- `fa_api.cu` 是薄分发层:按 `stage` 把 `void*` cast 成具体类型交给
  `fa_wrappers.h` 声明的内部函数;真正的参数校验在各阶段 wrapper 里做。

---

## 3. 阶段 1:朴素实现(stage1_naive.cu)

### 3.1 算法与三次 kernel 的组织

朴素版把注意力拆成三个 pass,每个 pass 一个 kernel(避免一个 kernel 里需要
跨线程共享中间量,先把正确性做出来):

```
pass1 (fa1_max_kernel):  对每个 query i, m_i = max_j (q_i·k_j) / sqrt(d)
pass2 (fa1_lsum_kernel): 对每个 query i, l_i = Σ_j exp((q_i·k_j)/√d - m_i)
pass3 (fa1_out_kernel):  O[i][j] = (1/l_i) Σ_k exp((q_i·k_k)/√d - m_i)·V[k][j]
```

- 每个 **block 处理一个 query 行**(`blockIdx.x == i`),256 线程。
- **j/k 的循环用跨线程步进**(`for (j = threadIdx.x; j < N; j += 256)`),让每次
  迭代的 32 个线程读同一行 K(广播),行内点积仍是串行 `d` 次 —— 这是刻意为之
  的“最朴素但正确”的形态:写起来简单,每个线程的累加完全私有,最后只需一次
  块内归约。
- **每行 q 先拷进寄存器数组 `qr[128]`**(`#pragma unroll 4`),这样内层点积不再
  反复从全局内存取 q(编译器还能向量化)。`d < 128` 时补 0,配合 `c < d` 判断,
  使模板大小固定、循环可展开。
- **块内归约**:`smem red[256]` → 线程写自己的部分结果 → 树状两两合并
  (`red[t] = max(red[t], red[t+s])`,每轮 `__syncthreads()`)。max 用
  `-FLT_MAX` 初值,sum 用 `0.f`,最后 `thread 0` 写 `m[i]/l[i]`。
- pass3 每个线程负责一列 `j`,对全部 k 累加 `acc += exp(...)·V[k][j]`
  —— 每个输出元素一行代码,没有归约,但 K 被每个线程反复读整行(L1 兜底)。

### 3.2 为什么它慢(面试必答)

- 计算量 **O(N²d²)**:每个输出元素要遍历全部 N 个 key 做 d 维点积 → 任何一行
  (i,k) 的点积 S_ik 被算了两遍(一遍在 max、一遍在 lsum、第三遍在 out);
- **DRAM 流量 O(N²d)**:K 每行每 query 读一次(没有分块复用);
- 还显式写了 m/l 两个 O(N) 中间数组(用 `cudaMallocAsync`,这是本项目刻意使用
  的 stream-ordered 内存,阶段 6 会再深用)。
- 实测(N=2048,d=64)38.2 ms,只有 0.03 TFLOPS —— 数字本身就是最好的论据。

### 3.3 支持任意边界

- `d ≤ 128` 任意(寄存器数组兜底 0);N 任意(block 数 = N,`i >= N` 直接返回)。
- Python 测试专门用 `N=300, d=40`(非 64/16 倍数)验证这条路径。

---

## 4. 阶段 2:共享内存分块 + 在线 softmax(stage2_tiled.cu)

### 4.1 在线 softmax 数学(整个项目的地基)

对一行 query,已知前 j 个 key 的局部统计 `(m, l, O_row)`,新 key 的得分 s 到达时:

$$m' = \max(m,\, s),\qquad \alpha = e^{m-m'},\qquad \beta = e^{s-m'}$$

$$l' = \alpha\cdot l + \beta,\qquad O' = \alpha\cdot O_{\text{row}} + \beta\cdot v_{\text{row}}$$

恒等关系(精确成立,不是近似):

$$O' = \alpha\,O + \beta\,v,\qquad l' = \alpha\,l + \beta,\qquad m' = \max(m,\,s)$$

$$\frac{O'}{l'} = \frac{\alpha\,O + \beta\,v}{\alpha\,l + \beta}
= \frac{e^{m-m'}O + e^{s-m'}v}{e^{m-m'}l + e^{s-m'}}
= \frac{\sum_{k\le j} e^{s_{ik}-m'}v_k}{\sum_{k\le j} e^{s_{ik}-m'}}$$

即:每个 key 到达时把已累加的 O 行整体“退火”回相对新 max 的尺度,再叠加新
贡献;**不依赖任何先验知识,也不需要第二遍扫描** —— 这是 FlashAttention 能
“一次遍历 K/V、不物化 S/P”的核心。面试时把它写成上面三行递推即可。

### 4.2 内核结构与共享内存布局

模板参数 `D, BR, BN`:一个 block 处理 **BR 个 query 行**、一次把 **BN 个 key**
装进共享内存(循环遍历全部 key 块,这就是作业要求的“循环遍历 K/V 块”)。

```
qs   [BR × (D+1)]  ①Q tile ②行累加器 O(复用!)③输出暂存
ks   [BN × (D+1)]  K tile
vs   [BN × (D+1)]  V tile
invl [BR]          行 1/l
```

四个易漏细节:

1. **Q tile 的三重用(代码里最巧的一处)**:第 2 步把每行 q 拷进寄存器 `qr[]`
   之后,立刻把 smem 里同一行清零(`orow[c]=0`),这行 smem 从此当 **O 行累加器**
   用;所有 key 处理完后它装的就是未归一化的 O;第 4 步按 `invl[r]` 缩放直接写回
   全局。等于只给 O 行花了一份 smem,没有额外 buffer。
2. **每行一个线程**(`threadIdx.x == 行号`,BR=128),点积、`alpha/beta` 缩放全在
   行私有寄存器/私有行里做 —— **行与行之间零通信**,softmax 归约问题被分块
   结构直接消掉(这是 BR 取大值的动机之一)。K/V 行读是 128 线程同时读同一行 →
   共享内存广播,1 个周期。
3. **`D+1` 的 padding**:行主序 tile 的行距若恰为 D=64(64×4B=256B),相邻行同列
   元素落在同一 bank;padding 一个 float 后行距 65 mod 32 = 1,相邻行错开 bank,
   消除跨行访问的 bank 冲突(本项目主要访问是“整行广播”和“行私有”,冲突本来就
   少,padding 更多是教科书式保险)。
4. **尾部边界**:`rows = min(BR, N-row0)`、`bn = min(BN, N-kb)`,加载与计算都用
   `i < rows` / `j < bn` 守卫;grid = `ceil(N/BR)`,所以阶段 2 支持任意 N
   (测试里的 N=300 专门走尾部路径)。

在线更新发生在**单个 key 粒度**(内层 `for j in [0,bn)` 每个 key 都做一次
m/l/O 更新),与 FA 论文 Algorithm 1 一致;浮点上 `alpha*orow + beta*vrow` 每列
一个 FMA,行累加量受 max 约束,不会溢出。

### 4.3 动态共享内存与启动参数

```cpp
extern __shared__ float smem[];              // 大小运行时才定
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
kernel<<<grid, BR, smem_bytes, stream>>>(...);
```

- 默认每块动态 smem 上限 48KB,超过要一次性 `cudaFuncSetAttribute` 提升
  (per-kernel 属性,静态变量 `attr_set` 保证只设一次)。
- 尺寸验算(d=64, BR=128, BN=64):`(128 + 2·64)·65·4 + 128·4 ≈ 66.6KB`;而
  d=128 时 BR=128 会到 ~130KB(超过 99KB 上限),所以 wrapper 对 d=128 自动降为
  BR=64/BN=32 —— **host 侧按 smem 预算选模板**,这个思路在阶段 3/4 会以更精确
  的字节级预算表重现。
- wrapper 用 `switch (d)` 分发到 `launch2<D,BR,BN>` 实例化,非法 d 返回
  `FA_ERR_UNSUPPORTED`。

阶段 2 把阶段 1 的 DRAM 流量从“每 query 重读全量 K/V”变成“每个 query 块读一次
全量 K/V”(K/V 被 BR 个 query 行复用),实测 N=2048 提速 ~44x,把复杂度从
O(N²d²) 降到 O(N²d) 的访存形态;计算仍由标量 SIMT 完成,真正的算力提升在
阶段 3 交给 Tensor Core。

## 5. Tensor Core 内核详解(tc_kernel.cuh + tc_p1..p4.cuh,阶段 3/4/5 共用)

### 5.1 任务划分:为什么是“4 warp × 16 行 × m16n16k16”

```text
一个 block:BM=64 个 query 行 × 全部 N 个 key(按 BN 分块循环)
  └─ 4 个 warp,每 warp 固定负责 16 个 query 行(恰好 = wmma 的 m=16)
       └─ softmax 行归约因此只需要 warp 内通信(__shfl_xor / __syncwarp)
grid = qrows / 64;每块 128 线程
```

把 16 行分给一个 warp 是整套设计的第一性选择:**一旦某一行(及其 m/l)只属于
一个 warp,行方向的 max/sum 归约就不需要跨 warp/跨 block 的 barrier**,阶段 2
里“每行一个线程”的思路在 Tensor Core 版里升级成“每行一群 lane,但群 = warp”。

内核签名还带两个一般化参数,让**同一个内核**服务三种用途:

```cpp
fa_tc_kernel(Q, K, V, O, m_out, l_out, qrows,
             kA0, kA1, kB0, kB1, scale);
```

- key 范围拆成 `[kA0,kA1)` + `[kB0,kB1)` 两段:阶段 3/4 用 `[0,N)` + 空段;
  阶段 5 用它跑“本地段”和“两段远程 key”(见 §7),一个 launch 搞定;
- `m_out/l_out != nullptr` 时为 **partial 模式**:不除 l、把 (m,l) 写出来,
  供阶段 5 的 merge 内核二次合并(见 §7.2)。

### 5.2 共享内存预算表(tc_p1 的 Layout)——面试最常被追的细节

布局顺序:`Q tile | K 环形缓冲 | V 环形缓冲 | 4 个 warp 各自的 soft 区`。
soft 区按 warp 隔离(warp 私有,不需要跨 warp 同步),内含:

```text
S    (fp32, BN×16×4B)   每 16 个 key 一个“16×16 压缩块”,块号 nc → nc*256B
P    (fp16, BN×16×2B)   同上,写回时转半精度
Oacc (fp32, D×16×4B)    16 行 × D 列的累加器,按 D/16 个 16×16 块存放
ls   (fp32, 16×4B)      行 1/l(epilogue 广播用)
```

**为什么 S/P/O 都按“压缩 16×16 块”(ldm 恒为 16)存放,而不是按
(行,列)大矩阵 + 行距?** 因为 `wmma::load_matrix_sync / store_matrix_sync`
要求给出一个“标准矩阵视角”,行距必须是该矩阵的行宽;一旦 smem 里是
“64 行 × 256 列、行距 256”的大矩阵,行距就和 wmma 期望的 16 对不上,
元素就会按 ldm 错位落行 —— 这正是早期版本“输出行被平移 r 列”bug 的根源。
改成“每 16×16 一块、块内 ldm=16”后,每块的行列语义自洽(教训见
`docs/optimization_log.md` 问题 2)。

手工验算一例(d=64, BN=64, STAGES=2):

```text
Q tile     64×64×2B                        =   8192 B
K/V ring   2×2 个 tile(64×64×2B,双缓冲)     = 2×16384 = 32768 B
4×warp soft 4×(4096+2048+4096+64)          = 41216 B
合计 TOTAL = 8192 + 32768 + 41216 = 82176 B ≈ 80.3 KB(≤ 96 KiB,可装)
```

各组合实测预算(字节):BN=64,d=64:S1 65792 ✓ / S2 82176 ✓ / S3 98560 ✗;
BN=32,d=64:S1 45312 ✓ / S2 53504 ✓ / S3 61696 ✓;
BN=32,d=128:S1 78080 ✓ / S2 94464 ✓ / S3 110848 ✗;BN=64,d=128 全部 >96 KiB。

对照结果表就能读懂两件事:**为什么 d=128 只有 BN=32 的行**;为什么 d=64 的
v4(BN64,S3)在结果里缺席 —— 它返回 `FA_ERR_SMEM` 被基准脚本跳过,不是没测。
`Layout::SMEM_LIMIT = 96*1024` 与 `CFG_OK`(tile 的 16B 单元数能被 128 整除,
即 `TILE_H % 1024 == 0`)是编译期门禁:装不下的组合在 host 侧第一行就拒绝。

### 5.3 K/V 搬运与环形缓冲(tc_transfer_tile)

- 每 tile 看成 `TILE_H = BN·D` 个 fp16,即 `TILE_H/8` 个 16B 单元;128 线程
  每人拷 `PER = TILE_H/8/128` 个单元(CFG_OK 保证整除,可安全展开)。
- **STAGES ≥ 2(sm_80+)**:`__pipeline_memcpy_async(16B)` + `__pipeline_commit()`
  发出异步拷贝,数据绕过寄存器直达 smem,由硬件推进,不用线程搬运;
- slot = `(kb / BN) % STAGES` 环形复用缓冲 —— 这就是“内存池管理 K/V tile
  缓存”的实现形态(作业点 2);
- STAGES==1 或 <sm_80 回退同步 `float4` 拷贝(slot 恒 0,配合 `__syncthreads`
  依然正确)。

### 5.4 主循环与流水控制(fa_tc_kernel)—— 正确性最容易错的地方

```text
prologue: 载入 Q tile(按 __half2 向量化,BM*D/2 个 half2 / 128 线程)
          → 偶数 lane 清零自己的 Oacc 行;mrun=-∞, lrun=0
STAGES≥2: 预取 min(STAGES, nbA) 个 tile(只发不等)
loop blk in [0, nbA):
  1. 等待:blk+STAGES ≤ nbA ? wait(STAGES-1) : wait(0)
     (稳态:允许 STAGES-1 个组仍在途,当前组必已完成;
      尾部:全部等在途组落地,不再有后续预取)
  2. __syncthreads()   # 所有 warp 的 cp.async 都完成、可见后再读
  3. tc_compute_block(当前 tile)
  4. __syncthreads()   # 所有 warp 算完,才允许下一轮搬运覆写本 slot
  5. blk+STAGES < nbA → 发出 tile blk+STAGES 的搬运
```

两道 `__syncthreads` 各管一件事:第 2 步是“读前等数据”,第 4 步是“写前等读完”。
wait 计数与 barrier 顺序写反、或尾部少一次 wait,都会造成读半块/覆写竞态
(开发中真实踩过,见 optimization_log 迭代 3)。STAGES==1 分支没有预取,退化为
“同步搬一块 → 算一块”,且支持 `[kA段, kB段]` 两段循环。

### 5.5 单 key 块计算(tc_compute_block)——wmma 逐步拆解

约定:`KC = d/16`(列分块数)、`NCHUNK = BN/16`(key 分块数)。

1. **Q 的 A fragment**:每列块 kc 一个 `matrix_a(row_major, m16k16)`,
   `load_matrix_sync(qf[kc], qs + warp*16*D + kc*16, ldm=D)`。注意它在
   **每个 key 块都重载** —— qs 其实不变,这是简化(正确性优先),也是
   后续优化点之一(把 Q fragment 提为块级寄存器,省 32 次 smem→reg 拷贝)。
2. **S = Q·Kᵀ**:每 16 个 key 一块(nc)。B 操作数从 smem 的 K tile 里按
   `matrix_b(col_major)` 载入,行距 D —— 为什么 K 是行主序却能当
   col_major 用?因为 `S = Q·Kᵀ` 里 Kᵀ 按列读 K 的行,而 K 在显存里
   “行 = key”,恰等于 Kᵀ 的列主序存储,`col_major + ldm=D` 就是这个转置。
   累加 fragment 清零 → `mma_sync` 累加 KC 次 → 结果整块
   `store_matrix_sync` 写回 `ss + nc*256`(fp32)。
3. **行 max(raw 域)+ 在线合并**:32 lane / 16 行 → 每行 2 lane:
   `r = lane>>1`,`half = lane&1`。偶数 lane 扫前半列、奇数 lane 扫后半列,
   各自得局部 max,`__shfl_xor_sync(...,1)` 一次交换合并成行 max;
   `mc = rawmax·scale` —— **先取 max 后乘 scale**,避免 exp 大指数溢出,
   且数学等价(scale>0 单调)。随后与运行中的 `mrun` 在线合并:
   `mnew = max(mrun, mc)`,`a = exp(mrun-mnew)`,`b = exp(mc-mnew)`。
4. **O 行退火**:偶数 lane 把该行 Oacc 的 D 个元素整体乘 a(旧行相对新
   max 的补偿)。写 Oacc 一定只有偶数 lane 做,奇数 lane 只贡献了归约值。
5. **P + 行和**:每列 `p = b·exp((ss-rawmax)·scale)` 写回 fp16 的 `ph`
   (验算:`b·e^{(s-rawmax)·scale} = e^{mc-mnew}·e^{(s-rawmax)·scale}
   = e^{s·scale - mnew}`,正好是相对 mnew 的 softmax 分子);
   `lc` 同法 xor(1) 合并 → `lrun = a·lrun + lc`。
6. **O += P·V**:P 按 `matrix_a(row_major)` 从 `ph + nc*256`(ldm=16)载入,
   V 按 `matrix_b(row_major)` 从 V 环形槽 `vt + nc*16*D + cc*16`(ldm=D)
   载入(行 = key,与 P 的 key 列配对),Oacc fragment 载入累加后整块存回。
   —— Oacc 每 chunk 都从 smem 载入/存回一趟,这是第二个可优化点。
7. `__syncwarp()` 收尾(保证下一轮搬运前本 warp 写完 soft 区)。

**布局无关性(防架构差异)**:wmma fragment 内部“哪个 lane 持哪个元素”是
架构细节(跨代不保证),所以代码从不假设 `sF.x[i]` 的下标语义,行方向只依赖
自己写回 smem 的 16×16 标准块 + lane 的几何关系(`>>1`,`&1`,`xor 1`)。
代价是 S/P/O 都要经 smem 往返一遍 —— 诚实记录在优化日志的“已知瓶颈”。

### 5.6 epilogue:partial 与整行写回

```cpp
if ((lane&1)==0) ls[r] = partial ? 1.f : 1.f/lrun;   // partial 留待 merge
__syncwarp();
// 每行 rr∈[0,16) 由 32 lane 步进扫 16*D 个元素写全局:
//   O[r][col] = oacc[blk*256 + rr*16 + (col&15)] * ls[rr]
```

`lrun` 只在偶数 lane 有意义(奇数 lane 的 lc 已 xor 合并),所以用
“偶数 lane 写 ls + syncwarp + 全体读”的模式广播行 1/l,而非 32 lane 各算各。

## 6. 阶段 3 / 阶段 4 wrapper:模板参数拼出变体

- **stage3_wmma.cu**:`STAGES=1` 的“同步基线”。`d≤64 → BN=64`,`d=128 →
  BN=32`(对照 §5.2 预算表就懂为什么);入口校验 `N % 64 == 0`。
- **stage4_tuned.cu**:变体表 `v0=(BN64,S1) v1=(BN32,S1) v2=(BN64,S2)
  v3=(BN32,S2) v4=(BN64,S3) v5=(BN32,S3)`;`variant=-1` 时按 d 自动选
  v2(d≤64)或 v3(d=128) —— 即实测最优。基准脚本逐个变体跑,装不下的
  组合(见 §5.2 的 ✗ 行)返回 `FA_ERR_SMEM`,结果表自动缺行。

这一层没有新 kernel,只是“用模板参数描述配置空间 + host 侧预算门禁”,
是后面做 auto-tune 的雏形。

## 7. 阶段 5:多 GPU(s5_core.cuh / s5_worker.cuh / stage5_mgpu.cu)

### 7.1 思路:本地 partial + 通信 + 远程 partial + merge

N 切成 nranks 段,rank r 持 Q/K/V 的 `[r·Nloc, (r+1)·Nloc)`。每个 rank 想算
完自己的 Nloc 行 query,必须看到**全部** key —— 所以把 K/V 做 AllGather。
但通信和计算可以重叠:**自己的 key 段不用等任何人**,先算;

```text
① sc 流:    用 [本地段] 跑 partial kernel → (O1, m1, l1)
② scomm 流: 后台 AllGather 别人家的 K/V(与 ① 并行)
③ 等 ② 完成 → 同一 partial kernel 处理 [0,r·Nloc) 与 [(r+1)·Nloc,N)
             两个 key 段(一次 launch,靠 kA/kB 参数)→ (O2, m2, l2)
④ merge:(O1,m1,l1) ⊕ (O2,m2,l2) → 最终 O
```

### 7.2 merge 公式(与 §4.1 同源)

$$m=\max(m_1,m_2),\quad a=e^{m_1-m},\quad b=e^{m_2-m},\qquad
O=\frac{a\,O_1 + b\,O_2}{a\,l_1 + b\,l_2}$$

partial 内核输出的 O 未除 l、m/l 原样写出 —— 这正是“让两个 kernel 的
partial 可结合”的最小接口(如果 O 已经归一化,合并就得先反推 l,数值上
更差)。`nranks==1` 时用 fill kernel 造恒等 partial(m₂=-∞, l₂=0, O₂=0),
合并式自动退化为 O₁/l₁。

### 7.3 worker 线程模型与跨 rank 同步(s5_worker.cuh)

每个 rank 一个 host 线程,独占 `sc`(计算)/`scomm`(通信)两条流:

- **发布协议**:上传完自己的 Q 与 K/V 槽位后,在 sc 上记事件 `ev_up`,
  `S.ready` 原子计数 +1,自旋等全部 rank 到达(带 acquire 语义);
- **失败协议**(易漏点):任何一步失败,`fail()` 先把自己标记 `S.failed[r]`,
  再**把两个阶段的屏障计数都补足**再退出 —— 否则别的 rank 会在自旋上
  死锁;别人看到 failed 标记后也会走失败路径退出;
- **竞态修复点**(这是真 bug):peer 的 K/V 槽位由 peer 线程在 peer 的 sc 流
  上上传;如果本 rank 的 scomm 流不等任何事件就直接 `memcpyPeerAsync` 去
  拉,可能读到“上传到一半”的数据(实测曾出现单配置误差 2e-3)。修复 =
  scomm 依次 `cudaStreamWaitEvent` 每一个 rank 的 `ev_up`(包括自己)再发
  拷贝 —— 跨流/跨线程的依赖用事件显式钉死;
- NCCL 分支:Linux + `-DFA_USE_NCCL=ON` 时用 `ncclAllGather`(K/V 各一次),
  否则用 `cudaMemcpyPeerAsync` 多流拉取(Windows/无 NCCL 通用);P2P
  enable 失败没关系,驱动会退化为走主机内存的拷贝;
- 虚拟模式:`dev = rank % ndev`(nranks 可 > 物理卡数),同一张卡上两个
  rank 通过不同 context/流“假装”双卡 —— 数值等价,但没有真实并行,
  所以基准里虚拟 2 rank 的时间没有稳定加速(甚至更慢:串行 + 每步 host
  屏障 + 拷贝),报告里如实标注,真实加速需要 ≥2 GPU 复测。

## 8. 阶段 6:推理集成(CUDA Graph + 流序内存池,stage6_graph.cu)

### 8.1 把什么抓成图

用户传入固定的 FP32 输入指针;一次调用原本需要 4 次 launch:
`castQ → castK → castV → fa4(自动选 v2/v3)`。这 4 个 kernel 之间没有任何
host 依赖,是 CUDA Graph 的理想场景:

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  fa6_cast_kernel<<<...>>>(q32→q16);  // 3 个 cast
  fa4_fwd_f16(-1, q16, k16, v16, o32, ...);   // 内核内部仍是同流 launch
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, ...);     // 编译成可回放 exec
```

fa4 之所以能“合法地”出现在 capture 里:它内部只做同流 kernel launch +
编译期就完成的属性设置,没有任何同步点、没有 `cudaMalloc`、不跨流等待
(见 §2.2 的 stream 约定) —— capture 规则就是“捕获流上只能出现 capture
合法的异步操作”。

### 8.2 流序内存池为什么适合这里

FP16 staging(q16/k16/v16)用 `cudaMallocAsync`(默认池)分配、`cudaFreeAsync`
归还:**分配/释放都是流上的异步操作**,同流内的后续 kernel 自动与分配建立
依赖,释放的内存立刻可被同流复用,不需要 CPU 侧同步、也没有每帧
malloc/free 的 driver 开销;池内部按大小分级复用 —— 这就是“内存池管理
临时缓冲区”在推理侧的形态。相对的,老式 `cudaMalloc` 每次都要同步 + 与
driver 交互,Graph 场景下会成为瓶颈。

### 8.3 回放与测量口径

`fa6_run = cudaGraphLaunch + cudaStreamSynchronize`(阻塞语义,方便宿主拿
结果)。基准里 eager 对照是“3 cast + fa4 + sync”重复 50 次取平均,graph
侧重复 `fa6_run` 50 次;两次结果逐位一致(maxdiff=0 —— 同一批 kernel
同一顺序,数学上必须完全相同)。短序列(N=256)省 39% 延迟,因为此时
4 次 CPU launch 开销占比大;N=2048 只省 ~14% —— 数据本身就在回答
“Graph 什么时候值得用”。

## 9. 支撑层:测试与基准(怎么证明“对”,数字怎么测)

### 9.1 ctypes 绑定(python/fa_lib.py)

- 声明 `fa_forward` 的 `argtypes`(指针一律 `c_void_p`),传
  `tensor.data_ptr()` 零拷贝;输出 `torch.empty` 预分配,由 kernel 直写;
- 每条路径封装一个方法(`forward_f32/f16/forward_mgpu_f16/graph_*`),
  dtype/device 用 assert 卡死 —— Python 侧的类型安全;
- 状态码映射表 `_FA_STATUS` 与 C 侧枚举一一对应(错误从 C 冒泡成
  RuntimeError,测试脚本能直接 catch)。

### 9.2 测试矩阵(python/test_correctness.py,当前 76 项全绿)

| 阶段 | 输入/参考 | 规模 | 要点 |
|---|---|---|---|
| 1 | fp32 vs SDPA fp32 | N∈{128,256,512} × d∈{16,32,64,128} + N=300/d=40 | 任意边界 |
| 2 | 同上 | N∈{128..2048} × 4d + N=300/d=64 | 尾部块路径 |
| 3/4 | fp16 入 vs SDPA fp16 | N∈{256..2048} × d∈{32,64,128} | 误差 <1e-3 |
| 5 | fp16(CPU 输入)vs SDPA | N∈{512,1024} × 3d,nranks=2 | 虚拟/真实双卡 |
| 6 | fp32 入(内部 cast)vs “先转 fp16 再 SDPA” | 同 3/4 | 图回放数值 |

要点:参考实现统一是 `F.scaled_dot_product_attention`(dropout=0,非因果);
输入用 `torch.randn × 1/√d` 压低 QK 得分动态范围;随机种子按
`stage·1000+d+n` 固定(可复现);d=128 的 fp32 容差放宽 4 倍;
`--quick` 缩成小矩阵供 CI 快跑。

### 9.3 基准口径(bench/)

- 阶段 1-4:`cudaEvent` 对包裹 **L=20 次连续同流 launch**(吞掉单次 launch
  抖动),重复 reps 轮取**最小均值**(min of per-iteration averages)——
  报的是 kernel 稳态吞吐,不含首轮时钟爬升;
- 阶段 5/6:host 墙钟(`high_resolution_clock`,含 H2D/D2H 与同步)——
  这两阶段卖点就是端到端延迟;
- 指标公式与代码一致:`tflops = 4·N²·d / t`(两次 QKᵀ + 两次 PV 的乘加
  共 4N²d),`gbps` 按 Q/K/V fp16 + O fp32 的最小字节数算 —— 注意这是
  **算法级 FLOP 口径**,不含 mask/softmax 的标量开销,所有阶段用同一把
  尺子比较才公平;
- 每行 JSONL(device 信息在 header 行),`gen_report.py` 聚合出
  `docs/perf_results.md`。

### 9.4 一键复现(scripts/run_all.py)

`cmake 配置 → 编译 → test_correctness.py → fa_bench → gen_report.py`。
本机实测命令:`python scripts\run_all.py --arch 120`(结果已回填各报告)。

## 10. 读代码自查清单(也是答辩高频追问)

1. half2 载入循环的迭代单位是 **half2 个数**,不是元素个数;
2. `ss/ph/oacc` 的 256B 块内索引 = `blk·256 + r·16 + (c & 15)`;
3. 流水稳态 `wait(STAGES-1)` vs 尾部 `wait(0)` 的边界是 `blk+STAGES<=nbA`;
4. 偶数 lane 才有完整行;奇数 lane 只参与 xor(1) 归约 —— 别在奇数 lane
   上做行级读写;
5. partial 模式(传了 m_out/l_out)只写 m/l、不除 l;epilogue 的 `ls`
   含义随之切换;
6. STAGES≥2 路径只支持单段 key(kB 必须为空),两段(远程)key 只能在
   STAGES=1 下跑 —— 阶段 5 因此固定 STAGES=1;
7. 所有 `size_t` 转型出现在行偏移乘法处,防 32 位索引溢出;
8. 三个 CUDA_CHECK 宏按“返回形状”选用;launch 后必跟 `CUDA_LAST_CHECK()`。

