# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Speed vs quality of the SSD-1B first-frame generator (the biggest T2V stage on CPU).

Every variant runs the same prompts and seed with the TAESDXL decoder (as in `--fast`) and
records seconds per image and CLIP-score (ViT-B/32, image vs prompt, x100). A contact sheet per
variant is written to --shots for visual checking. Variants:
  steps   : LCM timesteps (upstream: 999,749,499,249)
  res     : generation size (upstream 1024x640; the video uses 512x320)
  cl      : channels_last memory format for the UNet convs

    python scripts/bench_first_frame.py --out results/first_frame_bench.json --shots results/shots
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline  # noqa: E402
from neodragon.runtime import configure_cpu_threads, make_generator  # noqa: E402

PROMPTS = [
    "A red fox running through fresh snow in a pine forest at sunrise",
    "Waves crashing on black volcanic rocks, slow motion, golden hour",
    "A busy night market street in an old city, lanterns glowing, people walking",
    "An astronaut riding a horse on the moon, earth in the sky",
    "A cup of coffee on a wooden table, steam rising, morning light",
    "A drone shot over a desert canyon with a river, sunset",
]
MODIFIER = ", cinematic, realistic textures, high detail, natural colours"

VARIANTS = {
    "base_4step_1024": dict(steps=[999, 749, 499, 249], h=640, w=1024, cl=False),
    "cl_4step_1024": dict(steps=[999, 749, 499, 249], h=640, w=1024, cl=True),
    "4step_768": dict(steps=[999, 749, 499, 249], h=480, w=768, cl=False),
    "3step_1024": dict(steps=[999, 666, 333], h=640, w=1024, cl=False),
    "2step_1024": dict(steps=[999, 499], h=640, w=1024, cl=False),
    "3step_768": dict(steps=[999, 666, 333], h=480, w=768, cl=False),
    "2step_768": dict(steps=[999, 499], h=480, w=768, cl=False),
}


def clip_score(clip, proc, prompt, img):
    inputs = proc(text=[prompt], images=[img], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip(**inputs)
    i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((i @ t.T).item() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/first_frame_bench.json")
    ap.add_argument("--shots", default="results/shots")
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    from diffusers import AutoencoderTiny
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    threads = configure_cpu_threads()
    root = snapshot_download("Qualcomm-AI-Research/Neodragon", cache_dir=args.cache)
    pipe = SSD1B_FirstFrameGeneratorPipeline.from_pretrained(root)
    pipe.register_modules(vae=AutoencoderTiny.from_pretrained("madebyollin/taesdxl", cache_dir=args.cache))
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache)

    # warm-up (first call allocates / picks kernels)
    with torch.no_grad():
        pipe(prompt="warm up", timesteps=[999, 499], num_inference_steps=2, height=320, width=512)

    res = {"threads": threads, "torch": torch.__version__, "seed": args.seed, "prompts": PROMPTS, "variants": {}}
    Path(args.shots).mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        v = VARIANTS[name]
        pipe.unet.to(memory_format=torch.channels_last if v["cl"] else torch.contiguous_format)
        times, scores, imgs = [], [], []
        for p in PROMPTS:
            t0 = time.perf_counter()
            with torch.no_grad():
                img = pipe(prompt=p + MODIFIER, timesteps=v["steps"], num_inference_steps=len(v["steps"]),
                           height=v["h"], width=v["w"], generator=make_generator(args.seed)).images[0]
            times.append(time.perf_counter() - t0)
            img = img.resize((512, 320), Image.BICUBIC)  # what the video model actually sees
            scores.append(clip_score(clip, proc, p, img))
            imgs.append(img)
        sheet = Image.new("RGB", (512 * 3, 320 * 2))
        for k, im in enumerate(imgs):
            sheet.paste(im, ((k % 3) * 512, (k // 3) * 320))
        sheet.save(f"{args.shots}/first_frame_{name}.jpg", quality=85)
        res["variants"][name] = {**{k: v[k] for k in ("steps", "h", "w", "cl")},
                                 "mean_s": round(float(np.mean(times)), 2),
                                 "mean_clip": round(float(np.mean(scores)), 2),
                                 "clip_per_prompt": [round(s, 2) for s in scores],
                                 "s_per_prompt": [round(t, 2) for t in times]}
        print(name, json.dumps(res["variants"][name]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
