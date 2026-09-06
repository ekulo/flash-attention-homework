# dbg5b.py - ablation matrix for the stage5 d=128 invalid-argument failure.
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_lib import FaLib  # noqa: E402


def main():
    lib = FaLib()
    torch.manual_seed(0)
    for d in (64, 128):
        for n in (256, 512, 1024):
            for r in (1, 2):
                q = torch.randn(n, d).half() * (d ** -0.5)
                k = torch.randn(n, d).half() * (d ** -0.5)
                v = torch.randn(n, d).half() * 0.5
                try:
                    o = lib.forward_mgpu_f16(q, k, v, r)
                    print(f"OK   d={d:4d} n={n:5d} ranks={r}  max|o|={o.abs().max().item():.4f}")
                except Exception as e:
                    print(f"FAIL d={d:4d} n={n:5d} ranks={r}  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
