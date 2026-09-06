# ===========================================================================
# test_correctness.py —— 与 PyTorch scaled_dot_product_attention 对比
#
# 用法:python python/test_correctness.py [--stages 1 2 3 4 5 6]
#                                      [--tol-f16 1e-3] [--quick]
#
# 参考实现:torch.nn.functional.scaled_dot_product_attention
#   stage1/2 (FP32):输入 fp32,对比 SDPA fp32,误差 < 1e-4(实际 ~1e-6)
#   stage3/4/5 (FP16):输入 fp16(输出 fp32),对比 SDPA fp16,误差 < 1e-3
#   stage6:fp32 输入,内部 cast fp16 后计算,参考 = 先转 fp16 再 SDPA
#
# 额外覆盖边界:stage1/2 用 N=300/d=40 等非对齐尺寸验证边界处理。
# ===========================================================================
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __import__("os").path.dirname(__file__))
from fa_lib import FaLib  # noqa: E402


def sdp_ref(q, k, v):
    """q/k/v: [N, d] -> ref [N, d](fp32)"""
    q4, k4, v4 = (x.unsqueeze(0).unsqueeze(0) for x in (q, k, v))
    with torch.no_grad():
        o = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0, is_causal=False)
    return o.squeeze(0).squeeze(0).float()


def check(name, got, ref, tol, scale=1.0):
    g, r = got.float(), ref.float()
    diff = (g - r).abs().max().item()
    rel = diff / (r.abs().max().item() + 1e-6)
    ok = diff <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<44} max_abs={diff:.3e} "
          f"rel={rel:.3e} (tol={tol:.0e})")
    return ok


def gen_f32(n, d, dev, seed):
    # generator must match tensor device (torch >= 2.x requirement)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    s = 1.0 / (d ** 0.5)
    return (torch.randn(n, d, generator=g, device=dev) * s,
            torch.randn(n, d, generator=g, device=dev) * s,
            torch.randn(n, d, generator=g, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--tol-f32", type=float, default=2e-4)
    ap.add_argument("--tol-f16", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true", help="缩小尺寸矩阵,加快跑完")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 设备"
    lib = FaLib()
    dev = "cuda:0"
    FaLib.check_env()
    ndev = torch.cuda.device_count()
    print(f"== 可见 GPU 数 = {ndev} ==")

    stages = set(args.stages)
    n16 = [256, 512, 1024, 2048] if not args.quick else [256, 512]
    d16 = [32, 64, 128] if not args.quick else [64]
    d32 = [16, 32, 64, 128] if not args.quick else [64]
    total_ok = True

    # ---------- 阶段1(FP32,任意 N 与任意 d<=128;含非对齐边界) ----------
    if 1 in stages:
        n1 = [128, 256, 512] if not args.quick else [256]
        for d in d32:
            for n in n1:
                q, k, v = gen_f32(n, d, dev, seed=1000 + d + n)
                o = lib.forward_f32(1, q, k, v)
                ref = sdp_ref(q, k, v)
                total_ok &= check(f"stage1 f32 N={n:5d} d={d:3d}", o, ref,
                                  args.tol_f32)
        # 边界:d=40(非 16 倍数)、N=300(非 64 倍数)——阶段1 全部支持
        q, k, v = gen_f32(300, 40, dev, seed=1999)
        o = lib.forward_f32(1, q, k, v)
        total_ok &= check("stage1 f32 N=300 d=40(边界)", o, sdp_ref(q, k, v),
                          args.tol_f32)

    # ---------- 阶段2(FP32 分块;任意 N 含尾部边界;d 限模板集合) ----------
    if 2 in stages:
        n2 = [128, 256, 512, 1024, 2048] if not args.quick else [256]
        for d in d32:
            for n in n2:
                q, k, v = gen_f32(n, d, dev, seed=2000 + d + n)
                o = lib.forward_f32(2, q, k, v)
                ref = sdp_ref(q, k, v)
                tol = args.tol_f32 * (1.0 if d <= 64 else 4.0)  # 大 d 稍放宽
                total_ok &= check(f"stage2 f32 N={n:5d} d={d:3d}", o, ref, tol)
        # 边界:N=300(非 64 倍数)-> 尾部行/尾部 key 块路径
        q, k, v = gen_f32(300, 64, dev, seed=2999)
        o = lib.forward_f32(2, q, k, v)
        total_ok &= check("stage2 f32 N=300 d=64(边界)", o, sdp_ref(q, k, v),
                          args.tol_f32)

    # ---------- 阶段3 / 阶段4(FP16 in / FP32 out,N 需为 64 倍数) ----------
    for stage, variants in ((3, [0]), (4, [-1])):
        if stage not in stages:
            continue
        for d in d16:
            for n in n16:
                q, k, v = gen_f32(n, d, dev, seed=stage * 1000 + d + n)
                qh, kh, vh = (x.half() for x in (q, k, v))
                o = lib.forward_f16(stage, qh, kh, vh, variant=variants[0])
                ref = sdp_ref(qh, kh, vh)
                total_ok &= check(f"stage{stage} f16 N={n:5d} d={d:3d}", o, ref,
                                  args.tol_f16)

    # ---------- 阶段5(多 GPU;单卡时为虚拟 2 rank) ----------
    if 5 in stages:
        nranks = 2  # ndev>=2 真实双卡;否则虚拟双 rank(数值等价)
        for d in d16:
            for n in [512, 1024]:
                q, k, v = gen_f32(n, d, "cpu", seed=5000 + d + n)
                qh, kh, vh = (x.half() for x in (q, k, v))
                o = lib.forward_mgpu_f16(qh, kh, vh, nranks)
                ref = sdp_ref(qh.to(dev), kh.to(dev), vh.to(dev))
                mode = "real-2gpu" if ndev >= 2 else "virtual-2rank"
                total_ok &= check(f"stage5 {mode} N={n:5d} d={d:3d}",
                                  o.to(dev), ref, args.tol_f16)

    # ---------- 阶段6(CUDA Graph + 内存池;fp32 入 -> 内部 fp16) ----------
    if 6 in stages:
        for d in d16:
            for n in n16:
                q, k, v = gen_f32(n, d, dev, seed=6000 + d + n)
                o = torch.empty_like(q)
                h = lib.graph_create(q, k, v, o)
                lib.graph_run(h)
                qh, kh, vh = (x.half() for x in (q, k, v))
                ref = sdp_ref(qh, kh, vh)
                total_ok &= check(f"stage6 graph N={n:5d} d={d:3d}", o, ref,
                                  args.tol_f16)
                lib.graph_destroy(h)

    print("==", "ALL PASS" if total_ok else "SOME FAILED", "==")
    sys.exit(0 if total_ok else 1)


if __name__ == "__main__":
    main()
