# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Does torch.compile (Inductor, CPU C++ backend) speed up the Neodragon DiT / video-VAE decoder?

Image-to-video mode is used so the SSD-1B first frame is not part of the measurement: the
same input picture and seed go through eager and compiled variants and we record
  * compile (first-call) time, steady-state seconds per video (best of --repeats),
  * PSNR of the compiled output against eager (should be ~lossless).

    python scripts/bench_compile.py --image first_frame.jpg --out results/compile_eval.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon import NeodragonPipeline  # noqa: E402
from neodragon.runtime import configure_cpu_threads, make_generator  # noqa: E402

PROMPT = "A red fox running through fresh snow in a pine forest at sunrise"
HF_MODEL_ID = "Qualcomm-AI-Research/Neodragon"
MODIFIER = ", cinematic, realistic textures, high detail, natural colours"


def psnr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def run(pipe, image, seed):
    t0 = time.perf_counter()
    with torch.no_grad():
        frames = pipe(prompt=PROMPT, prompt_modifier=MODIFIER, image=image, generator=make_generator(seed))
    return frames, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="results/compile_eval.json")
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--variants", default="eager,compile_dit,compile_dit_vae,int8,int8_compile_dit")
    args = ap.parse_args()

    threads = configure_cpu_threads()
    image = Image.open(args.image).convert("RGB")
    res = {"threads": threads, "torch": torch.__version__, "variants": {}}
    ref = None
    for name in args.variants.split(","):
        torch._dynamo.reset()
        pipe = NeodragonPipeline.from_pretrained(
            HF_MODEL_ID, torch_dtype=torch.float32, mode="hybrid", cache_dir=args.cache).to("cpu")
        if "int8" in name:
            pipe.quantize_int8()
        entry = {}
        try:
            if "compile_dit" in name:
                pipe.dit.forward = torch.compile(pipe.dit.forward, dynamic=None)
            if "vae" in name:
                pipe.vae.decoder.forward = torch.compile(pipe.vae.decoder.forward, dynamic=None)
            frames, first = run(pipe, image, args.seed)
            times = [run(pipe, image, args.seed)[1] for _ in range(args.repeats)]
            entry.update(first_call_s=round(first, 2), best_s=round(min(times), 2),
                         times_s=[round(t, 2) for t in times])
            if name == "eager":
                ref = frames
            elif ref is not None:
                entry["psnr_vs_eager_db"] = round(float(np.mean([psnr(a, b) for a, b in zip(frames, ref)])), 2)
        except Exception as e:  # record failures instead of hiding them
            entry["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            entry["trace_tail"] = traceback.format_exc().splitlines()[-6:]
        res["variants"][name] = entry
        print(name, json.dumps(entry), flush=True)
        del pipe

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
