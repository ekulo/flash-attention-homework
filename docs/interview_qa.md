# 面试题库(含答案)—— 高性能 FlashAttention 大作业

> 用法:先遮住答案自己讲一遍(每题 30~90 秒),再对照“要点”;带 ⭐ 的题是
> 该项目答辩/面试被追问概率最高的。答案都锚定本仓库代码,可与
> `docs/code_tutorial.md` 对读。数字口径:RTX 5060 Ti(sm_120)、CUDA 13.1,
> 阶段1→4 详见 `docs/perf_results.md`。

---

## A. 项目总览(30 秒电梯陈述)

**A1. 一句话介绍这个项目。**
用 CUDA 从零实现 FlashAttention 前向 `O = softmax(QKᵀ/√d)V`,六个阶段递进:
朴素 → 共享内存分块 + 在线 softmax → Tensor Core(wmma)→ cp.async 多级流水调优
→ 多 GPU(partial+AllGather+merge)→ CUDA Graph 推理封装;全程不物化 S/P,
显存 O(N);正确性对照 PyTorch SDPA,76 项全过(FP16 误差 ~1e-5,要求 <1e-3);
N=2048 端到端 38.2ms → 0.119ms(320x),峰值 9 TFLOPS。

**A2. 为什么分六个阶段?**
每个阶段对应一组正交知识点,且性能目标可测:①kernel 骨架与错误检查;
②共享内存分块 + 在线 softmax(算法地基);③Tensor Core + warp 归约(算力);
④调优方法论(参数扫描、异步拷贝流水);⑤多卡(数值切分与合并);⑥部署
(内存池 + 图回放)。每阶段相对上一阶段的提升就是“优化记录”的证据链。

**A3. 项目怎么分层?** 见 code_tutorial §1/§2:C ABI 库(fa_api)+ ctypes 绑定 +
测试 + 基准 + 一键脚本;kernel 层 stage1..6 + tc_ 公共头 + host wrapper 分层,
参数校验在 host,数值在 kernel,错误码跨层传递(枚举 + 字符串映射)。

**A4. 最重要的三个数字?** 精度 1e-5(要求 1e-3);加速 320x(朴素→调优后,
N=2048);吞吐 9.0 TFLOPS(约该卡 FP16 Tensor Core 峰值的个位数百分比 ——
能主动说出“效率不高、瓶颈在哪”比数字本身更能加分,见 D8)。

**A5. 重做一次,你会先改什么?** ①先花半天把 wmma fragment 的存取语义和
smem 布局画清楚再写 kernel(早期“输出平移 bug”就来自 ldm 误解);
②一开始就用 ncu 定基线而不是凭感觉调参;③公共 TC 内核先做正确性最小版
再叠加流水。

---

## B. FlashAttention 算法与论文

**B1. 标准注意力为什么显存 O(N²)?**
`S = QKᵀ/√d` 与 `P = softmax(S)` 都是 N×N 中间矩阵,必须物化;每行 softmax
还要先看到整行才能归一化,于是要么存 S、要么两遍读 K。显存与带宽都随
N² 增长 —— 长序列的墙。

**B2. ⭐ FlashAttention 前向的基本流程?**
Q 按 BM 行分块、K/V 按 BN 行分块;每个 Q 块遍历全部 K/V 块(两重分块循环):
块内算出 S 块 → 与该 Q 块已累加的 (m, l, O) 做在线 softmax 合并 → O 块
直接参与 P·V;所有中间量只在共享内存/寄存器,从不写全局 S/P。

**B3. ⭐ 在线 softmax 的数学(必须能手写)?**
处理第 j 个 key 前有 (m, l, O_row),新得分 s 到来:

$$m'=\max(m,s),\quad \alpha=e^{m-m'},\quad \beta=e^{s-m'}$$
$$l'=\alpha l+\beta,\qquad O'=\alpha\,O_{\text{row}}+\beta\,v$$

归一化后等于以 m' 为参考的完整加权和(分子分母同除 e^{m−m'} 归纳):

$$\frac{O'}{l'}=\frac{\alpha O+\beta v}{\alpha l+\beta}=\frac{\sum_{k\le j}e^{s_{ik}-m'}v_k}{\sum_{k\le j}e^{s_{ik}-m'}}$$

这是**精确恒等**,不是近似 —— 所以一次遍历 K/V 是安全的。要点:O 行每次
被新 max“退火”(乘 α),防止旧贡献在新尺度下失真。

**B4. 数值稳定性怎么保证?**
①每行有运行 max,exp 参数 ≤ 0,不可能溢出;②本项目先对原始得分取行 max,
再统一乘 scale(先 max 后乘,数学等价且避免大指数);③累加器 fp32;
④`__expf` 近似只在 (0,1] 量级求值。这就是为什么不需要“先算完整 S 再
softmax”的两遍法。

**B5. FA 到底省了什么?**
不省 FLOPs(仍是 O(N²d)),省的是 **HBM 流量**:S/P 不落显存;Q/K/V 各以
分块复用方式读取,DRAM 流量从“每 query 重读全量 K/V”的 O(N²d) 降到约
O(N²d/BM) 再叠加逐 tile 复用,论文口径 O(N²d²/M)(M=片上存储)。本项目
朴素→阶段2 实测 44x 就是这段流量的代价。

**B6. FA1 与 FA2 的主要区别?**
FA2 目标 ~2x:①去掉 forward 里不必要的 rescale 等待与分块间冗余同步,
把序列维也并行;②128/256 线程组织 + 整块 K/V 由全 block 合作搬运;③更少
的共享内存与寄存器开销;④前向不再需要按块重算,直接每块独立 softmax
技巧上更顺。对面试,记住“FA1 每处理一块都要等整块算完再合并;FA2 让
不同块/不同 warp 更少互相等待、更好流水”即可。

**B7. backward 怎么做?**
前向时只存每块的 (m, l)(O(N) 而不是 N²);反向重算 S 块:
`dS = P ⊙ (dV·Vᵀ) − D·P`? 精确写法:由 `O = P V` 得 `dV = Pᵀ dO`;
`dP = dO·Vᵀ`;softmax 反向:`dS = P ⊙ (dP − D·1ᵀ)`,其中
`D = rowsum(dO ⊙ O)`;再 `dQ = dS·K/√d`、`dK = dSᵀ·Q/√d`。全程同样
分块 + 在线技巧,不物化 S。

**B8. causal mask 如何高效实现?**
分块循环时利用三角结构:key 块在 query 块“之后”的整块跳过(不需要
计算),只有对角线附近的块需要 mask —— 把上三角元素置 −∞ 再照常
softmax(或按块拆成两种 kernel)。开销 O(N²d/2) 计算中只有约一半,
显存仍不物化 mask。本项目为课程简化未做(README 已知限制有写)。

**B9. 为什么 P 可以用 fp16 存?**
softmax 后 P ∈ (0,1],且是 mma 的 A 操作数必须 fp16;fp16 尾数 10bit,
相对误差 ~5e-4 量级,加上后续 fp32 累加,总误差远小于 1e-3 的验收线。
注意 S(softmax 之前)留在 fp32:行内 max/exp 归约在 fp32 里做更稳。

**B10. 超长序列 / 多 GPU 场景怎么做注意力?**
三档:①单卡长序列:split-K/Flash-Decoding(query 并行 × key 分块 +
二次合并,本项目 stage5 的 partial+merge 正是它的雏形);②多卡:
sequence-parallel / ring attention(K/V 环形传递,通信 O(N) 与计算重叠);
③本项目教学方案:切 N + AllGather K/V(通信 O(N·nranks),适合课程规模,
README 已诚实标注局限)。

**B11. FA 与标准实现的数值差在哪?**
来源:fp16 输入量化(约 1e-3 相对)、`__expf`(约 1e-6 相对)、求和/退火顺序
不同。本实现实测 FP16 通道最大 ~8e-5,FP32 通道 ~1e-6 —— 远低于 1e-3,
说明顺序重排没有引入系统性误差。

---

## C. CUDA 系统知识(都能在本仓库找到实例)

**C1. 为什么 block=128 线程、4 个 warp 各管 16 行 query?**
wmma m16n16k16 的 m=16:**一个 warp 管 16 行,行方向的 max/sum 归约就完全
落在 warp 内**(shuffle + syncwarp),不需要跨 warp 屏障;64 行 = 4×16。
这是“把 softmax 归约局域化”的顶层设计。

**C2. wmma 怎么用、有什么坑?**
`wmma::fragment<matrix_a/b/accumulator,16,16,16,...>` +
`load_matrix_sync/mma_sync/store_matrix_sync`;A/B 要 row/col_major 与 ldm
完全匹配,K 转置用 col_major 读行主序存储的 K(见 code_tutorial §5.5);
accumulator 必须 fp32。坑:fragment 内部 lane→元素映射是架构细节、不跨代
保证,所以**别写死 fragment.x 的下标语义**。

**C3. 本项目为什么让 S/P/O 全走共享内存中转?**
为了对 fragment 布局零假设:标量 softmax 归约只依赖“自己写回的 16×16 标准
块 + lane 的几何关系(>>1、&1、xor 1)”,任何 SM 架构都能编译且行为一致。
代价是 smem 带宽往返 —— 这是刻意的“正确性优先”决策,也是性能瓶颈之一
(与优化日志一致)。

**C4. 什么是 bank conflict? 本项目哪里有?**
共享内存 32 个 bank、4B 粒度,同一周期多条访问命中同一 bank 即冲突(串行化)。
本项目 soft 区是行私有 + 广播访问,冲突很少;K/V tile 行主序加了“行距+1”
padding 防跨行同 bank(阶段2 用 `D+1`,见代码注释)。回答时给出
`地址 % 32 == 同一 bank` 的判断方法即可。

**C5. __syncthreads 与 __syncwarp 区别?**
前者全 block 屏障(必须全 block 线程都到,否则死锁/UB),后者 warp 内屏障
(可带 mask、允许部分线程,用于 shuffle 前保证收敛)。TC 内核两道全块
`__syncthreads`:一道“读前等 cp.async 数据齐”,一道“写前等所有 warp 算完
再覆写环形槽”;行归约处只 `__syncwarp`。

**C6. shuffle 归约原理?**
`__shfl_xor_sync(mask, v, 1)` 让 lane i 与 lane i^1 交换 v —— 一次交换即把
相邻两 lane 的部分和合并;再 xor 2、4…做树状全 warp 归约。本项目每行 2 个
lane(2r 持前半、2r+1 持后半),所以**一次 xor(1) 就够**,是“行内归约只需
log₂(每行 lane 数)”的特例。

**C7. cp.async 是什么? 流水怎么写才对?**
sm_80+ 的异步拷贝指令:global→smem 不经寄存器,由硬件推进;每组操作
`__pipeline_memcpy_async`(16B 对齐)后 `__pipeline_commit()` 成组,
`__pipeline_wait_prior(k)` 等“在途组数 ≤ k”。环形缓冲 slot =
`(key块号) % STAGES`;稳态等 `wait(STAGES-1)`,尾部改 `wait(0)`;两道
`__syncthreads` 管读前/写前。顺序写反 = 读半块或覆写竞态(本项目真实踩过,
见优化日志迭代 3)。

**C8. 超过 48KB 的动态共享内存怎么用?**
默认每块动态 smem 上限 48KB;超过需先 `cudaFuncSetAttribute(kernel,
cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` 一次性提升,再在
launch 配置里传字节数。host 侧用编译期预算(Layout::TOTAL ≤ 96KiB +
CFG_OK)在 launch 前就拒绝装不下的 (BN, STAGES) 组合 —— 见 code_tutorial
§5.2 的手工验算表。

**C9. 占用率怎么算? 本项目为什么每 SM 只有 1 个 block?**
占用率 = 活跃 warp / SM 上限 warp。本项目每块 smem 65~96KB → 每 SM 至多
1 块(2 块要 130~190KB,超每 SM 可用 smem)→ 每 SM 4 warp。低占用率靠
cp.async 流水 + ILP 补延迟隐藏;这也是 9 TFLOPS 上不去的结构原因之一
(改进方向见 D9)。

**C10. 怎么测 kernel 时间才可信?**
同流连发 L 次(如 20)用 cudaEvent 对包裹取平均,重复多轮取最小均值;
要预热(时钟爬升);测端到端(含 H2D/D2H)用 host 墙钟;报告要写清口径
(本项目阶段1-4 是 device 稳态时间,阶段5/6 是墙钟,见 perf_results 表头)。

**C11. CUDA Graph 什么时候值得用?**
多次 launch 的 CPU 开销(每次 ~3-10us)占比高时:小 kernel、推理循环、
RL/解码每步多次小 launch。本项目把“3 cast + fa4”抓成一张图,N=256 端到端
省 39%,N=2048 只省 14% —— launch 开销占比随 kernel 变大而稀释。

**C12. Graph 捕获的合法/非法操作?**
捕获流上只能出现 capture 安全的异步操作(kernel launch、event record/wait、
流序池分配等);任何同步点(cudaStreamSynchronize、cudaMemcpy 同步版、
cudaMalloc、依赖外部流完成的操作)都会报错或产生依赖;捕获后
`cudaGraphInstantiate` 编译成 exec 才能回放。本项目 fa4 内部全是同流
launch,所以能被合法捕获(create 时属性已静态设好)。

**C13. stream-ordered 内存池(cudaMallocAsync)好在哪?**
分配/释放是流上异步操作:同流后续 kernel 自动与其排序,释放立即可被同流
复用,无 CPU 同步、无每次 driver 往返;池内按大小分级。旧式 cudaMalloc
是同步的,且与图捕获不兼容 —— 本项目阶段1 的 m/l 临时数组到阶段6 的
FP16 staging 都在用它(见 stage1 wrapper 与 stage6 create)。

**C14. CUDA 错误处理策略?**
Runtime 错误分同步/异步两类,异步错误粘性累积:每次 launch 后
`cudaGetLastError()` 取走错误,关键同步点再查;封装成宏打印
`错误名+文件+行`;多线程(阶段5)不能用“return 即走”的宏 —— 失败线程必须
补足屏障计数再退出,否则其他 rank 自旋死锁。

**C15. P2P 与 NCCL?**
P2P:同节点 GPU 间直连(UVA + cudaDeviceEnablePeerAccess + memcpyPeer);
NCCL:集合通信库,AllGather 等原语、拓扑感知、多卡多机。本项目 Windows/
无 NCCL 用 memcpyPeerAsync 多流实现 AllGather;Linux 开 FA_USE_NCCL 用
ncclAllGather。教学答案要点:小规模/同节点 peer 拷贝够用,大规模必须
NCCL(且要算通信量级 O(N·d·nranks),本项目 README 有 honest 分析)。

**C16. 代码里为什么到处是 (size_t) 转型?**
`N*d` 在 N=2048、d=128 只有 26 万,但框架是为大 N 写的;32 位 int 乘法在
N>2^31/d 时溢出。行偏移 `(size_t)i*d + j` 防溢出 —— 面试送分题。

**C17. 模板 + switch(d) 分发的意义?**
D/BN/STAGES 做编译期常量 → 循环全展开、smem 偏移常量折叠、寄存器数组
定长;switch 只实例化 4 个 d → 编译时间和代码体积可控(对比:全运行时
变量会让 wmma 的 ldm 无法静态优化)。

**C18. __expf vs expf、__float2half_rn?**
`__expf` 是快速近似(~1e-6 相对,无边界检查),在 softmax 里配合 max 平移
足够;`__float2half_rn` 做 round-to-nearest-even 的 fp32→fp16 转换 ——
本项目 P 的量化入口,精度选择直接决定误差上限。

---

## D. 本项目实现细节(被追问概率最高的 20 问)

**D1. ⭐ 完整讲一遍 TC kernel 对单个 key 块做了什么(7 步)?**
①载 Q fragment(每 16 列一个);②mma 算 S 块(fp32)存 smem;③行 max:
每行 2 lane 分管两半列 → xor(1) 合并 → 乘 scale;④与运行 (m,l) 在线合并,
偶数 lane 把 Oacc 整行乘 α;⑤算 P = β·exp((S−max)·scale) 写 fp16、累加行和;
⑥P·V 的 mma 累进 Oacc(fp32);⑦syncwarp。见 code_tutorial §5.5。

**D2. 为什么 S 是 fp32、P 是 fp16、O 累加是 fp32?**
S:max/exp 归约前需要高精度且无量化损失;P:是 mma 输入,fp16 是硬件要求,
且 P∈(0,1] 误差可控;O:跨 N 个 key 块累加,fp32 防误差累积 —— 与
FlashAttention 论文的精度设计一致,也是“混合精度”知识点的完整答案。

**D3. ⭐ 口算一份共享内存预算(背这个例子)?**
d=64、BN=64、STAGES=2:Q tile 64·64·2=8192B;K/V 双缓冲各 2 tile、
每 tile 8192B → 2·2·8192=32768B;soft 区每 warp:S 4096+P 2048+Oacc 4096+
ls 64=10304B,×4=41216B;合计 82176B(≈80.3KB ≤ 96KiB ✓)。由此可知
BN64+S3=98560B 装不下(d64 的 v4 因此缺席结果表)、d=128 只能用 BN=32。

**D4. 偶数 lane 持行、奇数 lane 只做归约 —— 为什么?**
32 lane / 16 行 → 每行 2 lane(r = lane>>1,奇偶 = 左右半列)。行级写
(Oacc 清零/退火、ls 写 1/l)只在偶数 lane 做,奇数 lane 把局部 max/和
xor(1) 合并过来;最后 epilogue 用 syncwarp 广播。行归约开销 = 1 次
shuffle,没有 smem 归约 —— 这是该布局的“性能账”。

**D5. partial 模式是什么? 为什么 stage5 需要两个 partial?**
TC 内核可输出未归一化的 (O, m, l) 而不除 l(epilogue 分支)。stage5 中
每个 rank 先算本地 key 段的 partial,通信完成后算远程 key 段的第二个
partial,再用 merge 内核合并 —— 因为 softmax 结果只有以 (m,l) 为伴的
“未归一化形式”才能无损结合(见 B3 公式)。如果输出已归一化,合并时必须
反推 l,数值与实现都更差。

**D6. ⭐ merge 内核的公式?**
m=max(m₁,m₂),a=e^{m₁−m},b=e^{m₂−m};O=(aO₁+bO₂)/(al₁+bl₂)。nranks=1
时 fill 一个“恒等 partial”(m₂=−∞, l₂=0, O₂=0),公式自动退化为 O₁/l₁
—— 用同一代码路径覆盖单卡。

**D7. “虚拟多卡”怎么骗过代码的? 性能为什么没提升?**
dev = rank % ndev:两个 rank 线程各自 cudaSetDevice 同一张卡、各自一套
流与缓冲,通过 peer 拷贝“假装”跨卡通信。数值路径与真实双卡完全一致,
所以测试能验证正确性;但两个 rank 在同一卡上串行执行,还叠加了 host
屏障与拷贝开销,所以端到端不加速甚至变慢 —— 报告里如实标注“需 ≥2 GPU
复测真实加速”。面试主动讲清这点 = 加分。

**D8. ⭐ 为什么 9 TFLOPS 只有峰值个位数百分比? 瓶颈链是什么?**
①占用率:smem 65-96KB/块 → 每 SM 1 块、4 warp(约 6-12%),延迟隐藏靠流水;
②S/P/O 经 smem 中转:每个 key 块 S 写读一遍、P 写读一遍、Oacc 载入存回
KC 次 → smem 带宽成为主瓶颈;③Q fragment 每 key 块重载;④标量 softmax
段穿插在 mma 之间。另外测试规模小(N≤2048,数据全驻留 L2),kernel 本身
就很短 —— 该卡 FP16 峰值 ~190 TFLOPS 级,9 TFLOPS 说明是访存/指令
混合瓶颈而非 mma 吞吐瓶颈。以上每条都能用 ncu 指标对上
(见 docs/ncu_guide.md)。

**D9. 怎么把性能再往上推?(说 3-5 条具体的)**
①Oacc 与 Q fragment 常驻寄存器、按 fragment 的行/列映射直接做退火与归约,
彻底去掉 soft 区 smem 往返;②BM=128、8 warps 或 2 block/SM 提占用率;
③FA2 式分工:整块 K/V 搬运与计算解耦、减少 barrier;④对 d=128 用
BN=64 需要把 soft 区压缩(如 P 就地转置)或降 S 精度策略;⑤split-K 并行
key 维 + 二次 merge(复用 stage5 的 partial 机制);⑥试 ldmatrix/手工
mma PTX 与更大 tile、L2 persistence。任何一条都先 ncu 验证假设。

**D10. stage2 为什么把 O 行“藏”在 Q tile 的 smem 里?**
Q 行拷进寄存器后,该行 smem 的生命周期就结束了,正好复用为 O 累加器
(最终输出也从这里写回,乘 1/l 即可)。一份 smem 干三件事(Q tile、O 行、
输出暂存),省掉独立 O buffer —— 面试讲这个细节能证明你真读过代码。

**D11. 为什么输入/输出精度是“FP16 进、FP32 出”?**
Tensor Core 路径输入必须 FP16(wmma m16n16k16 的 A/B),但跨 key 块的累加
与 softmax 行和用 FP32 保精度;输出 FP32 便于与 PyTorch 参考对比、误差
验收。阶段1/2 则全 FP32(教学基线,无 TC 约束)。

**D12. 为什么要求 N 是 64 的倍数? 任意 N 怎么办?**
块高 BM=64、wmma 16、K/V tile 16B 单元整除 128 线程 —— 模板化内核避免
边界分支换性能。要支持任意 N:参考阶段1/2 已示范的尾部处理(rows/bn 取
min + 守卫)或 pad 到 64 的倍数(如 FA2 里对边界块专门处理)。

**D13. STAGES≥2 为什么不能处理两段 key(kB)?**
流水路径只预取/等待单一连续范围,双段需要两次流水重启;阶段5 固定
STAGES=1 + 顺序段循环,就是为了复用同一内核支持“本地段 + 两段远程段”
(一次 launch 两个范围)。fa_tc_launch 里对“STAGES≥2 且 kB 非空”显式
拒绝 —— 这是 API 设计里“把非法组合挡在 host 侧”的例子。

**D14. 阶段3 与阶段4 的关系?**
同一内核、不同 (BN, STAGES):stage3 = v0(BN64,S1) 或 d=128 的 (BN32,S1)
同步基线;stage4 提供 6 变体 + auto 选择(-1 → v2/v3)。结果:cp.async
流水在 N≥1024 再 +3% 左右,STAGES=3 因 smem 挤压占用率反而回落 ——
“调参要有预算约束意识”就是这节的结论。

**D15. 测试为什么可信?(测试设计四问)**
①参考实现:PyTorch `scaled_dot_product_attention`(同一批算子生态里公认
正确);②口径匹配:FP16 通道与“先转 FP16 的 SDPA”比,不拿 FP32 参考欺负
FP16 kernel;③覆盖边界:N=300/d=40 非对齐、尾部块、d 全模板集合、掩码
开关、每阶段 × 全尺寸矩阵;④可复现:LCG/固定种子,76 项全绿打印
max_abs 与 rel。见 code_tutorial §9.2。

**D16. 你踩过的最深的坑?(准备 2 个故事)**
①wmma 的 ldm 语义坑:soft 区按“行距 BN+1”的 padding 布局存 S/P/O,
`store_matrix_sync` 把第 r 行写到了 r 列错位,输出行被“平移” —— 教训:
wmma 的 load/store 要求给出标准矩阵视角,行距必须自洽(后改为 16×16
压缩块);②虚拟多卡上传竞态:peer 上传与拉取无事件依赖,偶发误差 2e-3,
修复 = 每 rank 记上传事件、拉取前等齐全部事件;③Windows 编码问题把
中文注释吃掉导致幽灵编译错(教训:工具链差异要早暴露)。每个故事都能
接“你怎么验证修复了?”(全量重测 + 逐位对比)。

**D17. 为什么还要 stage5/6? 作业不是只要求 kernel 吗?**
stage5 把“数值可切分”讲透(partial + merge 是 flash-decoding 的原型),
stage6 回答“kernel 写完怎么进推理系统”(内存池 + 图)。答辩时能把
kernel 放到系统视角讲,是区分度所在。

**D18. 你最后悔的设计决定?**
诚实答案示例:早期在“fragment 布局无关的软 softmax”与“fragment 直算
softmax”之间选了前者,正确性稳但 smem 中转成了天花板 —— 若先读明白
fragment 布局(当时官方文档抓取受限),阶段4 可能直接到位。承认权衡、
给出可执行改进,比自夸更有说服力。

---

## E. 高性能计算概念快答(30 秒每题)

E1. **IO-bound 还是 compute-bound 怎么判?** 算术强度(FLOP/Byte)与机器
ridge point 比;实践:看 ncu 的 dram 吞吐 vs 计算 pipe 利用率,本项目
阶段4 的 smem 吞吐与低占用率是主因,不是 DRAM。
E2. **Roofline?** 纵轴性能、横轴算术强度,斜线=带宽墙、平线=算力墙。
E3. **Tiling 消除了什么?** DRAM 重复读取:同一数据块被多个输出复用前先
留在片上(smem/reg),把“每输出读一次”变“每块读一次”。
E4. **延迟隐藏靠什么?** 多 warp 轮转 + 指令级并行;低占用率时用异步
拷贝流水把“等数据”变成“等的同时算别的”。
E5. **归约三种实现?** shuffle(≤32 元素/warp 内)、smem 树(block 内)、
atomic(跨 block);本项目行归约是 shuffle、阶段1 块归约是 smem 树。
E6. **fp16/tf32/bf16 区别?** fp16:10bit 尾数 + 5bit 指数;tf32:19bit 尾数
截断(fp32 输入直接算);bf16:8bit 尾数(大范围低精度)。训练常用 bf16/tf32,
本项目前向推理用 fp16 足够。
E7. **L2 cache persistence?** 可选送分:把常驻数据(如长序列的 K/V 块)
钉在 L2 的一部分,减少 DRAM 往返。
E8. **异步三件套?** stream(流序执行)、event(跨流依赖)、graph(整图一次
提交);本项目 stage5 用 event 钉跨 rank 依赖、stage6 用 graph 消 launch。

---

## F. 场景设计题(高频)

**F1. 8×H100 训 1M token 的 LLM,注意力怎么做?**
分段回答:单卡内 FA2/FA3 内核(因果、分组 query 并行);跨卡 sequence
parallel/ring attention(K/V 环形流动,计算通信重叠);必要时 Flash-Decoding
(split-K)压解码;数据并行之外还要考虑 head 维并行 —— 再落到本项目:
partial+merge 机制可以原样搬到 split-K。

**F2. 给现有 FA kernel 加 causal mask,最少改动?**
分块循环里 key 块整体在 query 块之后 → 跳过;对角线块:把 S 上三角
(列 > 行)置 −∞ 后照常 softmax(数学上等于截断行和)。若用掩码矩阵,
坚持“掩码随块生成,不物化”。

**F3. 一个 kernel 很慢,你前三步?**
①ncu 全指标快照(占用率、DRAM/smem 吞吐、warp stall 原因)定位是
launch/带宽/延迟还是吞吐;②对照 roofline 判断理论天花板;③按瓶颈选
招:tile 尺寸/向量化/流水/并行度/算法重构,一次只改一个变量并回测。
最忌讳:没 profile 就调 block 数。

**F4. kernel 崩了(非法地址),怎么查?**
compute-sanitizer(--tool memcheck)定位访存;先缩小到单 kernel/单块;
查三类:越界索引(行偏移没乘 d)、dangling smem 指针(offset 算错)、
同步缺失导致的覆写;本项目早期 bug 全都这三类。

**F5. 只有一张卡怎么验证多卡逻辑?**
本项目答案就是“虚拟 rank”:多线程 + 每 rank 独立流 + peer 拷贝路径,
单卡就能跑通全部同步/事件/屏障逻辑;真实多卡差异主要是带宽与真并行。
另外可提 CUDA_VISIBLE_DEVICES、MPS、多 context 等技巧。

**F6. FP16 误差超标怎么办?**
检查顺序:输入动态范围(先看 QK 得分幅度,必要时预缩放)、参考口径
(是否该比 FP32)、P 量化是否 round-nearest、累加是否 FP32、max 平移是否
生效;病态数据(个别行 softmax 极尖)用小样本 FP32 对照定位;最后再考虑
Kahan/分段计算。本项目早期“平移 bug”就是靠逐行对照定位的。

---

## G. 反问环节(准备 2-3 个)

1. 团队目前的 kernel 是自研还是以 cuDNN/FlashAttention 库为主,分工在哪?
2. 这个岗位实际产出是训练优化、推理部署还是新算子研发?用的 GPU 型号?
3. 组里有没有自己的性能基线/ncu 流程,新人怎么上手第一个 kernel?
4. 面试官如何看待 FA3(Blackwell tcgen05)对上层框架的影响?

---

## H. 30 秒速答卡(考前背这 12 条)

1. FA = 分块遍历 + 在线 softmax + 片上累加,不物化 S/P,显存 O(N)。
2. 在线 softmax 是精确恒等:O'=αO+βv,l'=αl+β,m'=max(m,s)。
3. 省的是 HBM 流量,不是 FLOPs。
4. softmax 行归约局域化:warp=16 行 → 只需要 shuffle。
5. fragment 布局不可移植 → S/P/O 走 16×16 压缩块中转。
6. cp.async 流水:slot 环形、wait(STAGES-1)/wait(0) 尾部、双 syncthreads。
7. smem 预算(背 82176B 例子)决定 (BN,STAGES) 哪些组合能装下。
8. P fp16、S fp32、Oacc fp32 —— 混合精度三原则。
9. 多卡 = partial(O,m,l) + AllGather + merge(m=max,a,b,除法合并)。
10. 虚拟 rank 验数值不验性能;真实加速要双卡。
11. Graph 收益 ∝ launch 开销占比:小 kernel 39%,大 kernel 14%。
12. 9 TFLOPS 的天花板:smem 中转 + 1 block/SM 占用率 + fragment 重载,
    改进 = Oacc/Q fragment 常驻寄存器 + BM=128 + split-K。

---
配套阅读:`docs/code_tutorial.md`(逐段代码讲解)、`docs/perf_results.md`(数字)、
`docs/optimization_log.md`(调优与踩坑史)、`docs/ncu_guide.md`(profile 指标)。
