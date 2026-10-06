# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Neodragon-X runtime helpers: run the same pipeline on CUDA, Apple MPS or a plain CPU
# (no GPU at all), with the right thread count, dtype and autocast for each device.

import contextlib
import copy
import os
import platform
from typing import Optional, Tuple

import torch

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def usable_cpu_count() -> int:
    """Number of CPUs this process may actually run on.

    `os.cpu_count()` / `nproc` are wrong inside many containers and sandboxes (gVisor reports 1
    while 17 cores are usable). PyTorch then defaults to 1 intra-op thread, which we measured to
    be 12.7x slower for one Neodragon DiT forward pass (27.1 s vs 2.13 s on 17 Zen-4 cores).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS / Windows
        return max(1, os.cpu_count() or 1)


def configure_cpu_threads(num_threads: Optional[int] = None) -> int:
    n = int(num_threads or os.environ.get("NEODRAGON_THREADS", 0) or usable_cpu_count())
    torch.set_num_threads(n)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n


def cpu_has_fast_bf16() -> bool:
    """True if the CPU has native bf16 matmul (AVX512_BF16 or AMX). Without it bf16 is emulated
    and several times slower than fp32 (e.g. AMD Zen 4 without the flag, older Intel)."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except OSError:
        return False
    return ("avx512_bf16" in flags) or ("amx_bf16" in flags)


def auto_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested != "auto":
        if requested not in _DTYPES:
            raise ValueError(f"unknown dtype {requested}; choose from auto/{'/'.join(_DTYPES)}")
        dtype = _DTYPES[requested]
        if device.type == "cpu" and dtype == torch.float16:
            print("[neodragon-x] fp16 is not supported well on CPU -> using fp32")
            return torch.float32
        return dtype
    if device.type == "cuda":
        try:  # native bf16 only (Ampere+); T4/V100 report emulated bf16 as "supported"
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older PyTorch
            native_bf16 = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if native_bf16 else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16 if cpu_has_fast_bf16() else torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Device-correct autocast. The upstream script always used `torch.cuda.amp.autocast`,
    which is a no-op (plus a warning) on CPU/MPS."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """CPU generator: identical noise for a given seed on every device."""
    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(int(seed))


# --------------------------------------------------------------------------------------------- #
# INT8 dynamic quantisation (CPU only)                                                          #
# --------------------------------------------------------------------------------------------- #
# Linear layers that stay fp32: input/output projections and the adaLN modulation (`norm1*`)
# that turns the timestep embedding into scale/shift. The adaLN layers hold ~1/3 of the DiT
# weights but only multiply a single vector, so keeping them fp32 costs no speed. Quantising those caused most of the error in
# our measurements (see docs/NEODRAGON_X.md, "INT8").
_KEEP_FP32_SUBSTRINGS = (
    "proj_out", "norm_out", "norm1", "time_text_embed", "pos_embed", "context_embedder",
)


def quantize_linear_int8(module: torch.nn.Module, keep_fp32: Tuple[str, ...] = _KEEP_FP32_SUBSTRINGS):
    """Dynamic INT8 (VNNI/AVX-512 on x86) for nn.Linear layers, fp32 for sensitive layers.

    Returns a new module; the original is left untouched.
    """
    module = copy.deepcopy(module)
    names = {
        name
        for name, m in module.named_modules()
        if isinstance(m, torch.nn.Linear) and not any(k in name for k in keep_fp32)
    }
    if not names:
        return module
    return torch.ao.quantization.quantize_dynamic(module, names, dtype=torch.qint8)
