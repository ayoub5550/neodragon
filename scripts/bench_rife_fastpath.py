# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Compare the two correct RIFE CPU paths on a real Neodragon clip.

* `native`: stock IFNet (c=90/45); the at-risk convs run on PyTorch's native kernel.
* `padded`: IFNet rebuilt with 96/48 channels (zero-padded, same function), oneDNN everywhere.

Reports the max pixel difference between the two (must be ~0) and the 2x time (49 -> 97 frames).

    python scripts/bench_rife_fastpath.py --video output/bench/baseline-p0.mp4 --out results/rife_fastpath.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon.postprocess import RIFEInterpolator
from neodragon.runtime import configure_cpu_threads


def timed(rife, frames, repeats):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = rife(frames, factor=2)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="results/rife_fastpath.json")
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    threads = configure_cpu_threads()
    frames = [Image.fromarray(f) for f in iio.imread(args.video, plugin="pyav")]

    native = RIFEInterpolator(cache_dir=args.cache, fast_path=False)
    padded = RIFEInterpolator(cache_dir=args.cache, fast_path=True)
    t_native, out_n = timed(native, frames, args.repeats)
    t_padded, out_p = timed(padded, frames, args.repeats)

    a = np.stack([np.asarray(f, np.int16) for f in out_n])
    b = np.stack([np.asarray(f, np.int16) for f in out_p])
    diff = np.abs(a - b)
    # raw float check on one pair (before uint8 rounding)
    x = RIFEInterpolator._to_tensor(frames[:2])
    x = torch.nn.functional.pad(x, (0, (32 - x.shape[3] % 32) % 32, 0, (32 - x.shape[2] % 32) % 32), mode="replicate")
    with torch.no_grad():
        inp = torch.cat([x[:1], x[1:2]], 1)
        f_diff = (native.model(inp) - padded.model(inp)).abs().max().item()

    res = {
        "video": Path(args.video).name, "threads": threads, "frames_in": len(frames),
        "frames_out": len(out_p), "torch": torch.__version__,
        "native": {"seconds_2x": round(t_native, 2), "bypassed_convs": native.num_bypassed_convs},
        "padded": {"seconds_2x": round(t_padded, 2), "bypassed_convs": padded.num_bypassed_convs,
                   "padded_blocks": padded.num_padded_blocks},
        "speedup": round(t_native / t_padded, 2),
        "max_float_diff_one_pair": f_diff,
        "uint8_frames_max_abs_diff": int(diff.max()),
        "uint8_pixels_differing_pct": round(100.0 * float((diff > 0).mean()), 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
