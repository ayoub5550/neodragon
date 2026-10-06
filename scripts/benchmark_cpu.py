#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Neodragon-X benchmark: speed + quality of each optimisation, one at a time.

Runs the hybrid pipeline with a fixed seed for every (config, prompt) pair and records
  * wall time per stage (first frame, DiT, VAE decode) and in total,
  * CLIP-score (ViT-B/32 cosine similarity x 100 between prompt and frames, averaged over
    5 evenly spaced frames) as a text-alignment quality proxy,
  * PSNR of every frame against the baseline video of the same prompt/seed
    (only meaningful when the first frame is unchanged).

Usage:
  python scripts/benchmark_cpu.py --out results/cpu_benchmark.json --videos output/bench
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neodragon import NeodragonPipeline  # noqa: E402
from neodragon.runtime import configure_cpu_threads, make_generator  # noqa: E402

HF_MODEL_ID = "Qualcomm-AI-Research/Neodragon"
PROMPTS = [
    "A red fox running through fresh snow in a pine forest at sunrise",
    "Waves crashing on black volcanic rocks, slow motion, golden hour",
    "A busy night market street in an old city, lanterns glowing, people walking",
]
MODIFIER = ", cinematic, realistic textures, high detail, natural colours"


class StageTimer:
    """Wraps pipeline components to time them without touching generation code."""

    def __init__(self, pipe):
        self.t = {}
        self._wrap(pipe.first_frame_gen_pipeline, "__call__", "first_frame", obj_is_callable=True)
        self._wrap(pipe.dit, "forward", "dit")
        self._wrap(pipe.vae, "decode", "vae_decode")
        self._wrap(pipe.vae, "encode", "vae_encode")

    def _wrap(self, obj, attr, key, obj_is_callable=False):
        if obj is None:
            return
        cls_fn = getattr(type(obj), attr) if obj_is_callable else getattr(obj, attr)
        timer = self

        def wrapped(*a, **k):
            t0 = time.perf_counter()
            out = cls_fn(obj, *a, **k) if obj_is_callable else cls_fn(*a, **k)
            timer.t[key] = timer.t.get(key, 0.0) + time.perf_counter() - t0
            return out

        if obj_is_callable:
            # instances can't override __call__ directly; patch via a subclass
            obj.__class__ = type(type(obj).__name__ + "Timed", (type(obj),), {"__call__": lambda self, *a, **k: wrapped(*a, **k)})
        else:
            setattr(obj, attr, wrapped)

    def reset(self):
        self.t = {}


def clip_score(clip, proc, prompt, frames):
    idx = np.linspace(0, len(frames) - 1, 5).round().astype(int)
    inputs = proc(text=[prompt], images=[frames[i] for i in idx], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img @ txt.T).mean() * 100)


def psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def strip(frames, path):
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, 5).round().astype(int)
    w, h = frames[0].size
    canvas = Image.new("RGB", (w * len(idx), h))
    for j, i in enumerate(idx):
        canvas.paste(frames[i], (j * w, 0))
    canvas.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cpu_benchmark.json")
    ap.add_argument("--videos", default="output/bench")
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="baseline,tiny_vae,tiny_vae+int8,i2v_fast")
    ap.add_argument("--num_prompts", type=int, default=len(PROMPTS))
    args = ap.parse_args()

    from diffusers.utils import export_to_video
    from transformers import CLIPModel, CLIPProcessor

    threads = configure_cpu_threads()
    Path(args.videos).mkdir(parents=True, exist_ok=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache)
    prompts = PROMPTS[: args.num_prompts]

    results = {
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "threads": threads,
            "torch": torch.__version__,
            "dtype": "fp32",
        },
        "seed": args.seed,
        "runs": [],
    }
    baseline_frames = {}
    first_frames = {}

    for config in args.configs.split(","):
        pipe = NeodragonPipeline.from_pretrained(
            HF_MODEL_ID, torch_dtype=torch.float32, mode="hybrid", cache_dir=args.cache
        ).to("cpu")
        if "tiny_vae" in config or config == "i2v_fast":
            pipe.enable_tiny_first_frame_vae(cache_dir=args.cache)
        if "int8" in config or config == "i2v_fast":
            pipe.quantize_int8()
        timer = StageTimer(pipe)

        for pi, prompt in enumerate(prompts):
            image = first_frames.get(pi) if config.startswith("i2v") else None
            if config.startswith("i2v") and image is None:
                continue
            timer.reset()
            t0 = time.perf_counter()
            with torch.no_grad():
                frames = pipe(
                    prompt=prompt,
                    prompt_modifier=MODIFIER,
                    image=image,
                    generator=make_generator(args.seed),
                )
            total = time.perf_counter() - t0
            if config == "baseline":
                baseline_frames[pi] = frames
                first_frames[pi] = frames[0]
            tag = f"{config.replace('+', '_')}-p{pi}"
            export_to_video(frames, f"{args.videos}/{tag}.mp4", fps=24)
            strip(frames, f"{args.videos}/{tag}.jpg")
            run = {
                "config": config,
                "prompt": prompt,
                "total_s": round(total, 2),
                "stages_s": {k: round(v, 2) for k, v in timer.t.items()},
                "clip_score": round(clip_score(clip, proc, prompt, frames), 2),
            }
            if pi in baseline_frames and config != "baseline":
                run["psnr_vs_baseline_db"] = round(
                    float(np.mean([psnr(a, b) for a, b in zip(frames, baseline_frames[pi])])), 2
                )
            results["runs"].append(run)
            print(json.dumps(run), flush=True)
        del pipe

    summary = {}
    for r in results["runs"]:
        s = summary.setdefault(r["config"], {"total_s": [], "clip_score": []})
        s["total_s"].append(r["total_s"])
        s["clip_score"].append(r["clip_score"])
    results["summary"] = {
        k: {"mean_total_s": round(float(np.mean(v["total_s"])), 2),
            "mean_clip_score": round(float(np.mean(v["clip_score"])), 2)}
        for k, v in summary.items()
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
