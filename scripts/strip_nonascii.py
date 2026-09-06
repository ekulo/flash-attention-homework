# strip_nonascii.py - Remove all non-ASCII bytes from compiled sources.
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def clean(p: pathlib.Path) -> bool:
    data = p.read_bytes()
    out = bytes(b for b in data if b < 0x80)
    if out == data:
        return False
    p.write_bytes(out)
    return True


def main() -> int:
    files = []
    for sub in ("src", "bench"):
        d = ROOT / sub
        files += sorted(d.glob("*.cu"))
        files += sorted(d.glob("*.cuh"))
        files += sorted(d.glob("*.h"))
    changed = 0
    for f in files:
        if clean(f):
            print("cleaned:", f.relative_to(ROOT))
            changed += 1
    print("done,", changed, "file(s) rewritten as pure ASCII")
    return 0


if __name__ == "__main__":
    sys.exit(main())
