# ===========================================================================
# gen_report.py —— 把 fa_bench 输出的 JSONL 汇总为 Markdown 表格
# 用法:python scripts/gen_report.py build/results/bench_results.jsonl
# 输出:docs/perf_results.md(自动生成,可直接放进作业报告)
# ===========================================================================
import json
import os
import sys


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def fmt(rows, stage, variant=None, key="ms", ndigits=4):
    out = []
    for r in rows:
        if r["stage"] != stage:
            continue
        if variant is not None and r["variant"] != variant:
            continue
        out.append(r)
    return out


def table(rows, stage, variant=None, cols=("N", "d", "ms", "tflops")):
    rows = fmt(rows, stage, variant)
    if not rows:
        return ""
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for r in sorted(rows, key=lambda x: (x["d"], x["N"])):
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "bench_results.jsonl"
    rows = load(src)
    dev = next((r for r in rows if r["stage"] == "header"), None)
    out = []
    out.append("# 性能结果(自动生成)\n")
    if dev:
        out.append(f"- 设备:{dev['device']} (CC {dev['cc']},SMs={dev['sms']})")
    out.append(f"- 数据来源:`{src}`(运行 `python scripts/run_all.py` 重新生成)\n")

    for st, name in [("stage1", "阶段1 朴素(FP32)"), ("stage2", "阶段2 分块+在线softmax(FP32)"),
                     ("stage3", "阶段3 TensorCore 同步基线(FP16)")]:
        out.append(f"\n## {name}\n")
        t = table(rows, st, None)
        out.append(t if t else "(无数据)")

    out.append("\n## 阶段4 调优变体对比(FP16)\n")
    t = table(rows, "stage4", None)
    out.append(t if t else "(无数据)")

    # 阶段3 vs 阶段2 vs 阶段4最优(d=64 主表)
    out.append("\n## 汇总:d=64,阶段2/3/4(最优)对比\n")
    best = {}
    for r in rows:
        if r["stage"] == "stage4":
            key = (r["N"], r["d"])
            if key not in best or r["ms"] < best[key][0]:
                best[key] = (r["ms"], r["variant"])
    head = "| N | stage2(ms) | stage3(ms) | stage4-best(ms) | 变体 | 加速 vs stage2 | 加速 vs stage3 |"
    out.append(head)
    out.append("|---|---|---|---|---|---|---|")
    for r in sorted(fmt(rows, "stage2"), key=lambda x: x["N"]):
        if r["d"] != 64:
            continue
        s3 = next((x["ms"] for x in fmt(rows, "stage3") if x["N"] == r["N"] and x["d"] == 64), None)
        b = best.get((r["N"], 64))
        if not b:
            continue
        line = (f"| {r['N']} | {r['ms']:.4f} | {s3:.4f} | {b[0]:.4f} | {b[1]} | "
                f"{r['ms']/b[0]:.1f}x | {s3/b[0]:.1f}x |")
        out.append(line)

    out.append("\n## 阶段5 多 GPU\n")
    out.append(table(rows, "stage5", None, ("variant", "N", "d", "ms")))
    for r in sorted(fmt(rows, "stage5"), key=lambda x: (x["d"], x["N"])):
        if r["variant"] == "baseline-1rank":
            continue
        base = next((x for x in fmt(rows, "stage5", "baseline-1rank")
                     if x["N"] == r["N"] and x["d"] == r["d"]), None)
        if base:
            out.append(f"- N={r['N']} d={r['d']}:2 rank {r['ms']:.3f} ms vs 单 rank "
                       f"{base['ms']:.3f} ms(加速 {base['ms']/r['ms']:.2f}x,"
                       f"{r['variant']})")

    out.append("\n## 阶段6 CUDA Graph + 内存池(单次调用延迟,含同步)\n")
    for r in sorted(fmt(rows, "stage6", None), key=lambda x: (x["d"], x["N"], x["variant"])):
        if r["variant"] == "graph":
            eg = next((x for x in fmt(rows, "stage6", "eager")
                       if x["N"] == r["N"] and x["d"] == r["d"]), None)
            if eg:
                out.append(f"- N={r['N']} d={r['d']}:eager {eg['ms']*1000:.1f} us,"
                           f"graph {r['ms']*1000:.1f} us,省 "
                           f"{(1-r['ms']/eg['ms'])*100:.1f}%")

    text = "\n".join(out) + "\n"
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "perf_results.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[已写入 {dest}]")


if __name__ == "__main__":
    main()
