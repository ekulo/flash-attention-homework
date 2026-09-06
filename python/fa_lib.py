# ===========================================================================
# fa_lib.py —— 通过 ctypes 加载 fa_api 动态库并提供 tensor 友好的封装
#
# 用法:
#   from fa_lib import FaLib
#   lib = FaLib()                       # 自动搜索 build/bin/fa_api.{dll,so}
#   o = lib.forward_f32(stage=1, q=q32, k=k32, v=v32)      # q/k/v: cuda fp32
#   o = lib.forward_f16(stage=3, variant=0, q=q16, ...)    # q/k/v: cuda fp16
#   lib.forward_mgpu_f16(...)  /  lib.graph_create/run/destroy
# ===========================================================================
import ctypes
import os
import sys

import torch

_FA_STATUS = {
    0: "OK",
    1: "CUDA runtime error",
    2: "invalid parameter",
    3: "unsupported config (d/variant)",
    4: "shared memory budget exceeded",
    5: "multi-GPU error",
    6: "NCCL error",
    7: "CUDA graph / mem pool error",
}


def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("FA_LIB")
    candidates = []
    if env:
        candidates.append(env)
    for root in (here, os.path.join(here, "..", "build", "bin"),
                 os.path.join(here, "..", "build")):
        for name in ("fa_api.dll", "libfa_api.so", "libfa_api.dylib", "fa_api.so"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                candidates.append(p)
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到 fa_api 动态库。请先构建项目,或用环境变量 FA_LIB 指定路径。")


class FaLib:
    def __init__(self, path=None):
        self.path = path or _find_lib()
        self.lib = ctypes.CDLL(self.path)
        f = self.lib.fa_forward
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p]
        self._forward = f

        self._fa5 = self.lib.fa5_forward_f16
        self._fa5.restype = ctypes.c_int
        self._fa5.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3

        self._fa6c = self.lib.fa6_create
        self._fa6c.restype = ctypes.c_int
        self._fa6c.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                               ctypes.c_int] + [ctypes.c_void_p] * 4
        self._fa6r = self.lib.fa6_run
        self._fa6r.restype = ctypes.c_int
        self._fa6r.argtypes = [ctypes.c_void_p]
        self._fa6d = self.lib.fa6_destroy
        self._fa6d.restype = ctypes.c_int
        self._fa6d.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _raise(code):
        raise RuntimeError("fa_api 返回错误 %d: %s" % (code, _FA_STATUS.get(code, "?")))

    # ---------- 阶段 1/2:FP32 in / FP32 out ----------
    def forward_f32(self, stage, q, k, v, variant=0, stream=None):
        assert q.dtype == torch.float32 and q.is_cuda
        o = torch.empty_like(q)
        s = stream.cuda_stream if stream is not None else None
        rc = self._forward(stage, variant, q.data_ptr(), k.data_ptr(),
                           v.data_ptr(), o.data_ptr(), q.shape[0], q.shape[1],
                           s if s is not None else None)
        if rc:
            self._raise(rc)
        return o

    # ---------- 阶段 3/4:FP16 in / FP32 out ----------
    def forward_f16(self, stage, q, k, v, variant=0, stream=None):
        assert q.dtype == torch.float16 and q.is_cuda
        o = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        rc = self._forward(stage, variant, q.data_ptr(), k.data_ptr(),
                           v.data_ptr(), o.data_ptr(), q.shape[0], q.shape[1], None)
        if rc:
            self._raise(rc)
        return o

    # ---------- 阶段5:host 侧多 GPU(h_q 等为 CPU 张量)----------
    def forward_mgpu_f16(self, q, k, v, nranks):
        assert q.dtype == torch.float16 and not q.is_cuda
        o = torch.zeros(q.shape, dtype=torch.float32)
        rc = self._fa5(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                       q.shape[0], q.shape[1], nranks)
        if rc:
            self._raise(rc)
        return o

    # ---------- 阶段6:CUDA Graph + 内存池 ----------
    def graph_create(self, q, k, v, o):
        h = ctypes.c_void_p()
        rc = self._fa6c(ctypes.byref(h), q.shape[0], q.shape[1],
                        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr())
        if rc:
            self._raise(rc)
        return h

    def graph_run(self, h):
        rc = self._fa6r(h)
        if rc:
            self._raise(rc)

    def graph_destroy(self, h):
        rc = self._fa6d(h)
        if rc:
            self._raise(rc)

    @staticmethod
    def check_env():
        print("torch", torch.__version__, "| CUDA", torch.version.cuda,
              "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
