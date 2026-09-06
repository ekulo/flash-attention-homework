# ===========================================================================
# run_all.py —— 一键:构建 -> 正确性测试 -> 基准 -> 生成性能报告
#
# Windows 示例:
#   python scripts/run_all.py --arch 120
# 也可只做其中一步:
#   python scripts/run_all.py --build-only
#   python scripts/run_all.py --test-only --quick
#   python scripts/run_all.py --bench-only
#
# 依赖:CMake >= 3.20;CUDA Toolkit >= 12.8(sm_120);Visual Studio 2022;
#       Python 3.8+ 与 torch(CUDA 版,用于参考实现对比)。
# ===========================================================================
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
BIN = os.path.join(BUILD, "bin")
RESULTS = os.path.join(BUILD, "results")


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=kw.pop("cwd", None), **kw)


def find_bin(name):
    for cand in (os.path.join(BIN, name), os.path.join(BIN, name + ".exe"),
                 os.path.join(BUILD, name), os.path.join(BUILD, name + ".exe")):
        if os.path.exists(cand):
            return cand
    return None


def _find_ninja():
    exe = shutil.which("ninja")
    if exe:
        return exe
    try:
        import ninja  # pip install ninja 提供的模块
        d = ninja.BIN_DIR
        exe = os.path.join(d, "ninja.exe" if os.name == "nt" else "ninja")
        if os.path.exists(exe):
            return exe
    except Exception:
        pass
    return None


def do_build(arch, generator):
    os.makedirs(BUILD, exist_ok=True)
    cache = os.path.join(BUILD, "CMakeCache.txt")
    # 坏缓存判定:生成器不匹配,或 CMAKE_MAKE_PROGRAM 是无效的
    # "python -m ninja"(CMake 找不到 ninja 时的错误回退,会污染后续配置)
    wipe = False
    if generator and os.path.exists(cache):
        gen_line, make_line = "", ""
        try:
            with open(cache, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                        gen_line = line.strip()
                    elif line.startswith("CMAKE_MAKE_PROGRAM"):
                        make_line = line.strip()
        except OSError:
            pass
        if generator not in gen_line:
            wipe = True
        if "python" in make_line and "ninja" in make_line:
            wipe = True
    if wipe:
        print(f">> 检测到损坏/不匹配的 CMake 缓存({gen_line!r} {make_line!r}),"
              f"删除 {BUILD} 后重新配置")
        shutil.rmtree(BUILD, ignore_errors=True)
        os.makedirs(BUILD, exist_ok=True)

    env = dict(os.environ)
    cmd = ["cmake", "-S", ROOT, "-B", BUILD]
    if generator:
        cmd += ["-G", generator]
        if "Ninja" in generator:
            ninja = _find_ninja()
            if not ninja:
                sys.exit("找不到 ninja。请先:python -m pip install ninja,"
                         "并把 ninja.exe 所在目录加入 PATH(或重开终端)")
            # 显式指定 ninja.exe 绝对路径,避免 CMake 的 python -m ninja 回退
            cmd += ["-DCMAKE_MAKE_PROGRAM=" + ninja]
            env["PATH"] = os.path.dirname(ninja) + os.pathsep + env.get("PATH", "")
    cmd += ["-DCMAKE_CUDA_ARCHITECTURES=" + arch]
    if generator and "Ninja" in generator:
        # 单配置生成器需要显式 Release(否则 Debug 的 /RTC1 与优化冲突)
        cmd += ["-DCMAKE_BUILD_TYPE=Release"]
    cfg = run(cmd, env=env)
    if cfg.returncode != 0:
        sys.exit(
            "CMake 配置失败,请查看上方输出。常见原因:\n"
            "  1) 'No CUDA toolset found'(VS 生成器):CUDA VS 集成未注册到当前 VS,\n"
            "     请改用 --generator Ninja;\n"
            "  2) 'unsupported Microsoft Visual Studio version'(nvcc 的 host_config.h\n"
            "     版本检查):本项目 CMakeLists.txt 已自动加\n"
            "     -allow-unsupported-compiler,请确认当前目录文件与最新版一致;\n"
            "  3) 找不到 cl/nvcc:请在 VS x64 开发环境中运行。")
    b = run(["cmake", "--build", BUILD, "--config", "Release", "-j"], env=env)
    if b.returncode != 0:
        sys.exit("编译失败,请查看上方错误输出")
    return find_bin("fa_bench")


def do_test(quick, stages):
    os.makedirs(BUILD, exist_ok=True)
    env = dict(os.environ)
    lib = find_bin("fa_api")
    if lib:
        env["FA_LIB"] = lib
    py = sys.executable or "python"
    cmd = [py, os.path.join(ROOT, "python", "test_correctness.py")]
    if quick:
        cmd.append("--quick")
    if stages:
        cmd += ["--stages"] + [str(s) for s in stages]
    r = run(cmd, env=env)
    return r.returncode


def do_bench():
    bench = find_bin("fa_bench")
    if not bench:
        sys.exit("找不到 fa_bench,请先构建(--build-only)")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "bench_results.jsonl")
    r = run([bench, out])
    return r.returncode, out


def do_report(jsonl):
    py = sys.executable or "python"
    return run([py, os.path.join(ROOT, "scripts", "gen_report.py"), jsonl]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="120", help="CUDA 架构(默认 120=RTX50 系)")
    ap.add_argument("--generator", default="",
                    help="CMake 生成器,如 Ninja(Windows 无 CUDA VS 集成时推荐)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="+", type=int)
    args = ap.parse_args()

    if not (args.build_only or args.test_only or args.bench_only):
        do_build(args.arch, args.generator)
        rc = do_test(args.quick, args.stages)
        if rc != 0:
            sys.exit("正确性测试未全部通过,已停止(可加 --quick 或 --stages 缩小范围)")
        bench = find_bin("fa_bench")
        if bench:
            _, jsonl = do_bench()
            do_report(jsonl)
        return

    if args.build_only:
        do_build(args.arch, args.generator)
        return
    if args.test_only:
        sys.exit(0 if do_test(args.quick, args.stages) == 0 else 1)
    if args.bench_only:
        _, jsonl = do_bench()
        do_report(jsonl)


if __name__ == "__main__":
    main()
