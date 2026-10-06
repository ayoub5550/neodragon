# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Evaluate the Real-ESRGAN compact upscaler (neodragon/postprocess/upscale.py) on Neodragon videos.

For each model ("general", "fast") and a bicubic baseline:
  * hold-out PSNR / sharpness: every frame (512x320) is downscaled 2x to 256x160, upscaled 2x
    back, and compared with the original frame (the only ground truth we have).
  * speed: ms per frame for the real use case, 512x320 -> 1024x640, on a full 49-frame clip.
  * temporal stability: mean |frame[t+1]-frame[t]| of the upscaled clip divided by the same
    value for bicubic (1.0 = no extra flicker added by the network).
  * numerics: float32 output vs a float64 reference of the same network (guards against the
    oneDNN conv bug documented in rife.onednn_conv_at_risk).

    python scripts/bench_upscale.py --videos output/bench/*.mp4 --out results/upscale_eval.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon.postprocess.upscale import VideoUpscaler  # noqa: E402
from neodragon.runtime import configure_cpu_threads  # noqa: E402


def read_video(path):
    import imageio.v3 as iio

    return [Image.fromarray(f) for f in iio.imread(path, plugin="pyav")]


def psnr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def sharpness(img):
    """Mean absolute Laplacian of the luma (higher = more fine detail)."""
    g = torch.from_numpy(np.asarray(img.convert("L"), np.float32))[None, None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None]
    return float(F.conv2d(g, k).abs().mean())


def bicubic(frames, factor):
    return [f.resize((f.width * factor, f.height * factor), Image.BICUBIC) for f in frames]


def flicker(frames):
    a = np.stack([np.asarray(f, np.float32) for f in frames])
    return float(np.abs(np.diff(a, axis=0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="results/upscale_eval.json")
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--models", default="general,fast")
    ap.add_argument("--shots", default=None, help="write a crop comparison jpg here")
    args = ap.parse_args()

    threads = configure_cpu_threads()
    paths = sorted(p for v in args.videos for p in glob.glob(v))
    clips = {Path(p).stem: read_video(p) for p in paths}
    res = {"threads": threads, "torch": torch.__version__, "videos": list(clips), "models": {}}

    def evaluate(name, up):
        r = {"holdout_psnr_db": [], "holdout_sharpness": [], "ms_per_frame_512x320_to_1024x640": [],
             "flicker_ratio_vs_bicubic": []}
        for clip in clips.values():
            small = [f.resize((f.width // 2, f.height // 2), Image.BICUBIC) for f in clip]
            rec = up(small)
            r["holdout_psnr_db"].append(float(np.mean([psnr(a, b) for a, b in zip(rec, clip)])))
            r["holdout_sharpness"].append(float(np.mean([sharpness(f) for f in rec[::8]])))
            t0 = time.perf_counter()
            big = up(clip)
            r["ms_per_frame_512x320_to_1024x640"].append(1000 * (time.perf_counter() - t0) / len(clip))
            r["flicker_ratio_vs_bicubic"].append(flicker(big) / flicker(bicubic(clip, 2)))
        out = {k: round(float(np.mean(v)), 3) for k, v in r.items()}
        out["holdout_sharpness_ground_truth"] = round(
            float(np.mean([sharpness(f) for c in clips.values() for f in c[::8]])), 3)
        return out

    res["models"]["bicubic"] = evaluate("bicubic", lambda fr: bicubic(fr, 2))
    print("bicubic", res["models"]["bicubic"], flush=True)

    first = next(iter(clips.values()))
    for name in args.models.split(","):
        up = VideoUpscaler(name, cache_dir=args.cache)
        up(first[:4])  # warm-up
        m = evaluate(name, up)
        # numerics vs float64 on one frame
        x = torch.from_numpy(np.asarray(first[0], np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            y32 = up.net(x.contiguous(memory_format=torch.channels_last) if up.channels_last else x)
            ref = up.net.to(memory_format=torch.contiguous_format).double()(x.double())
            up.net.float()
        m["max_abs_err_vs_float64"] = float((y32.double() - ref).abs().max())
        res["models"][name] = m
        print(name, m, flush=True)

        if args.shots:
            # side-by-side crop: bicubic | model, from the middle of the first clip
            f = first[len(first) // 2]
            a = bicubic([f], 2)[0]
            b = up([f])[0]
            box = (a.width // 4, a.height // 4, a.width // 4 + 384, a.height // 4 + 256)
            sheet = Image.new("RGB", (768, 256))
            sheet.paste(a.crop(box), (0, 0))
            sheet.paste(b.crop(box), (384, 0))
            Path(args.shots).mkdir(parents=True, exist_ok=True)
            sheet.save(Path(args.shots) / f"upscale_bicubic_vs_{name}.jpg", quality=92)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["models"], indent=2))


if __name__ == "__main__":
    main()
