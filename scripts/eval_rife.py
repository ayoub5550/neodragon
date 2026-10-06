# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check RIFE interpolation quality and speed on a generated Neodragon video.

Hold-out test: drop every odd frame, rebuild it from its two neighbours with RIFE, and compare
with the real frame (PSNR). A plain 50/50 blend of the neighbours is the baseline to beat.
Then time a full 2x pass (49 -> 97 frames).

    python scripts/eval_rife.py --video output/bench/baseline-p0.mp4 --out results/rife_eval.json
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


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="results/rife_eval.json")
    ap.add_argument("--cache", default="./models")
    args = ap.parse_args()

    threads = configure_cpu_threads()
    frames = [Image.fromarray(f) for f in iio.imread(args.video, plugin="pyav")]
    rife = RIFEInterpolator(cache_dir=args.cache)

    # hold-out: even frames are inputs, odd frames are ground truth
    inputs = frames[0::2]
    truth = frames[1::2][: len(inputs) - 1]
    rebuilt = rife(inputs, factor=2)[1::2]
    p_rife = [psnr(np.asarray(r), np.asarray(t)) for r, t in zip(rebuilt, truth)]
    p_blend = [
        psnr((np.asarray(a, np.float32) + np.asarray(b, np.float32)) / 2, np.asarray(t))
        for a, b, t in zip(inputs[:-1], inputs[1:], truth)
    ]

    t0 = time.perf_counter()
    out = rife(frames, factor=2)
    t2 = time.perf_counter() - t0

    res = {
        "video": Path(args.video).name,
        "threads": threads,
        "frames_in": len(frames),
        "holdout_psnr_db": {"rife": round(float(np.mean(p_rife)), 2), "blend": round(float(np.mean(p_blend)), 2)},
        "full_2x": {"frames_out": len(out), "seconds": round(t2, 2),
                    "ms_per_new_frame": round(1000 * t2 / (len(out) - len(frames)), 1)},
        "note": "hold-out gaps are 2 frames wide (harder than the real 1-frame 2x case)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
