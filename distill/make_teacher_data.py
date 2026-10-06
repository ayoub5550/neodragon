# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Step 1 of Neodragon-X distillation: generate teacher clips with Wan2.2-TI2V-5B (Apache-2.0).

Runs in its OWN environment (needs diffusers >= 0.35 for Wan2.2; Neodragon pins 0.34):
    python -m venv .venv-wan && .venv-wan/bin/pip install -r distill/requirements-wan.txt

Each clip is written as  <out>/<idx:05d>.mp4  plus a line in  <out>/index.jsonl
({"idx", "prompt", "seed", "height", "width", "num_frames", "steps", "seconds"}). The script is
resumable (existing mp4s are skipped) and shardable across GPUs/machines with --shard i/n.

The student renders 49 frames at 512x320, 24 fps, so by default the teacher renders the same
duration (49 frames) at a higher resolution with the same 8:5 aspect (480x768 by default); the
student-side script resizes to 512x320. Typical use:

    # GPU (recommended): ~1-3 min/clip on an RTX 4090 / A100 at 480x768x49, 40 steps
    python distill/make_teacher_data.py --out data/teacher --prompts distill/prompts.txt \
        --clips_per_prompt 4 --height 480 --width 768 --steps 40

    # CPU smoke test (proves the pipeline end to end; numbers in results/distill_smoke.json)
    python distill/make_teacher_data.py --out /tmp/teacher --limit 1 --height 320 --width 512 \
        --num_frames 17 --steps 4 --device cpu --dtype fp32
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

WAN_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# Wan's recommended negative prompt (translated subset of the official Chinese one)
NEGATIVE = ("bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, "
            "still picture, grey overall, worst quality, low quality, JPEG artifacts, ugly, deformed, "
            "extra fingers, poorly drawn hands, poorly drawn face, disfigured, fused fingers, "
            "messy background, three legs, many people in the background, walking backwards")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_prompts(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="distill/prompts.txt")
    ap.add_argument("--clips_per_prompt", type=int, default=1)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num_frames", type=int, default=49, help="must be 4k+1")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: this worker takes every n-th clip")
    ap.add_argument("--limit", type=int, default=None, help="stop after N NEW clips in this run (per worker)")
    ap.add_argument("--cpu_offload", action="store_true", help="model CPU offload for <24 GB GPUs")
    args = ap.parse_args()
    assert (args.num_frames - 1) % 4 == 0, "num_frames must be 4k+1"
    assert args.height % 32 == 0 and args.width % 32 == 0, "height/width must be multiples of 32"

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    if args.device == "cpu":
        # nproc/os.cpu_count() are wrong in many containers; torch would default to 1 thread
        torch.set_num_threads(len(os.sched_getaffinity(0)))
    dtype = DTYPES[args.dtype]
    vae = AutoencoderKLWan.from_pretrained(WAN_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=args.cache)
    pipe = WanPipeline.from_pretrained(WAN_ID, vae=vae, torch_dtype=dtype, cache_dir=args.cache)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = map(int, args.shard.split("/"))
    # round-robin over prompts (k outer), so any prefix of the job list is maximally diverse
    jobs = [(p, k) for k in range(args.clips_per_prompt) for p in load_prompts(args.prompts)]
    done = 0
    for idx, (prompt, k) in enumerate(jobs):
        if idx % shard_n != shard_i:
            continue
        path = out / f"{idx:05d}.mp4"
        if path.exists() and path.stat().st_size > 0:
            continue
        seed = 1000 * k + idx
        t0 = time.perf_counter()
        frames = pipe(
            prompt=prompt, negative_prompt=NEGATIVE, height=args.height, width=args.width,
            num_frames=args.num_frames, guidance_scale=args.guidance, num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).frames[0]
        secs = time.perf_counter() - t0
        export_to_video(frames, str(path), fps=args.fps)
        rec = dict(idx=idx, prompt=prompt, seed=seed, height=args.height, width=args.width,
                   num_frames=args.num_frames, steps=args.steps, guidance=args.guidance,
                   seconds=round(secs, 1), device=args.device, dtype=args.dtype)
        with open(out / "index.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        done += 1
        if args.limit and done >= args.limit:
            break


if __name__ == "__main__":
    main()
