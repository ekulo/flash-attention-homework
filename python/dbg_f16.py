# dbg_f16.py - locate the TC-kernel numerical bug elementwise.
# Compares our stage3/4 output against several torch references to isolate
# whether the error is scale-related, row-mapping-related, or something else.
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_lib import FaLib  # noqa: E402


def report(tag, N, d, o):
    torch.manual_seed(0)
    dev = "cuda:0"
    s = d ** -0.5
    q = (torch.randn(N, d, device=dev) * s).half()
    k = (torch.randn(N, d, device=dev) * s).half()
    v = (torch.randn(N, d, device=dev) * 0.5).half()

    qf, kf, vf = q.float(), k.float(), v.float()
    S = qf @ kf.T                       # raw scores (unscaled)
    ref_sdp = F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None])[0, 0].float()
    ref_scaled = torch.softmax(S * s, dim=-1) @ vf          # correct reference
    ref_unscaled = torch.softmax(S, dim=-1) @ vf            # no /sqrt(d)
    ref_inv = torch.softmax(S / s, dim=-1) @ vf             # *sqrt(d)
    ref_scale1 = torch.softmax(S * 1.0, dim=-1) @ vf        # scale=1

    def md(a, b):
        return (a - b).abs().max().item()

    print(f"== {tag} N={N} d={d} ==")
    print(" ours vs SDPA(fp16)  :", md(o, ref_sdp))
    print(" ours vs exact fp32  :", md(o, ref_scaled))
    print(" ours vs NO-scale    :", md(o, ref_unscaled))
    print(" ours vs scale*d     :", md(o, ref_inv))
    print(" ours vs scale=1     :", md(o, ref_scale1))
    diff = (o - ref_scaled).abs()
    flat = diff.argmax().item()
    print(" argmax at (row,col) =", flat // d, flat % d,
          " o=", o.flatten()[flat].item(), " ref=", ref_scaled.flatten()[flat].item())
    print(" row err>1e-2 count  :",
          (diff.max(dim=1).values > 1e-2).sum().item(), "/", N)
    print(" col err>1e-2 count  :",
          (diff.max(dim=0).values > 1e-2).sum().item(), "/", d)
    # show a couple of full rows
    for r in (0, 1, 3, 7):
        print(f" row {r}: o[:6]={o[r, :6].tolist()}")
        print(f"      ref[:6]={ref_scaled[r, :6].tolist()}")


def main():
    lib = FaLib()
    for (st, tag, N, d, variant) in [
            (3, "stage3-sync", 64, 64, 0),
            (3, "stage3-sync", 64, 32, 0),
            (4, "stage4-rec", 64, 64, -1),
            (4, "stage4-v0", 64, 64, 0),
    ]:
        torch.manual_seed(0)
        dev = "cuda:0"
        s = d ** -0.5
        q = (torch.randn(N, d, device=dev) * s).half()
        k = (torch.randn(N, d, device=dev) * s).half()
        v = (torch.randn(N, d, device=dev) * 0.5).half()
        o = lib.forward_f16(st, q, k, v, variant=variant)
        report(tag, N, d, o)


if __name__ == "__main__":
    main()
