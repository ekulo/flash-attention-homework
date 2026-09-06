# dbg_shift.py - per-row cyclic-shift analysis to locate the stride-1 bug.
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_lib import FaLib  # noqa: E402


def analyze(tag, N, d, variant, lib):
    torch.manual_seed(0)
    dev = "cuda:0"
    s = d ** -0.5
    q = (torch.randn(N, d, device=dev) * s).half()
    k = (torch.randn(N, d, device=dev) * s).half()
    v = (torch.randn(N, d, device=dev) * 0.5).half()
    o = lib.forward_f16(4 if tag.startswith("s4") else 3, q, k, v,
                        variant=variant)
    qf, kf, vf = q.float(), k.float(), v.float()
    ref = torch.softmax(qf @ kf.T * s, dim=-1) @ vf

    per_row = (o - ref).abs().max(dim=1).values
    good = (per_row <= 1e-3).nonzero().flatten().tolist()
    print(f"== {tag} N={N} d={d} ==  good rows({len(good)}): {good}")

    # best cyclic shift per row (shift s means ours[c] ~ ref[(c+s)%d])
    shifts = []
    for r in range(min(N, 24)):
        errs = []
        row_ref = ref[r]
        for sh in range(d):
            rolled = torch.roll(row_ref, sh)
            errs.append((o[r] - rolled).abs().mean().item())
        best = min(range(d), key=lambda i: errs[i])
        shifts.append((best, errs[best]))
    print(" row: best cyclic shift ours vs ref (and mean err):")
    print("  ", shifts)
    # is ours[r] a copy of some other ref row? check best row match without shift
    bestmatch = []
    for r in range(min(N, 16)):
        errs = [(o[r] - ref[rr]).abs().mean().item() for rr in range(N)]
        bm = min(range(N), key=lambda i: errs[i])
        bestmatch.append((bm, errs[bm]))
    print("  ours[r] best-matching ref row (no shift):", bestmatch)


def main():
    lib = FaLib()
    analyze("s3", 64, 64, 0, lib)
    analyze("s4-v0", 64, 64, 0, lib)
    analyze("s4-v1-BN32", 64, 64, 1, lib)
    analyze("s4-v2", 64, 64, 2, lib)
    analyze("s3", 64, 32, 0, lib)
    analyze("s3", 64, 16, 0, lib)


if __name__ == "__main__":
    main()
