# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Step 2 of Neodragon-X distillation: turn teacher mp4s into Neodragon training samples.

Wan2.2 and Neodragon do NOT share a latent space (Wan2.2 VAE: 48 ch, 4x16x16; Neodragon:
16 ch, 8x8x8), so the teacher is distilled through *pixels*: every teacher clip is resized /
center-cropped to 512x320, cut or resampled to 49 frames, and encoded with Neodragon's own
causal video VAE, using the exact scale/shift the sampler uses (first frame = image factors,
the rest = video factors). Text is encoded with Neodragon's text bundle (CLIP-L + CLIP-G +
DistilT5) with the same cinematic prompt modifier as inference.

Also writes <out>/null_prompt.pt (empty-prompt embedding, used for CFG dropout in training).
Output per clip: <out>/<idx:05d>.pt = {latents [16,7,40,64], prompt_embeds, prompt_attention_mask,
pooled_prompt_embeds, prompt, roundtrip_psnr_db}. `roundtrip_psnr_db` (decode(encode(x)) vs x)
is a built-in sanity check of the encoding; samples below --min_psnr are skipped.

    python distill/encode_for_student.py --teacher data/teacher --out data/student --cache ./models
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon import NeodragonPipeline  # noqa: E402
from neodragon.runtime import auto_device, configure_cpu_threads  # noqa: E402
from neodragon.utils.generation_utils import (  # noqa: E402
    DEFAULT_PROMPT_MODIFIER,
    VAE_SCALE_FACTOR,
    VAE_SHIFT_FACTOR,
    VAE_VIDEO_SCALE_FACTOR,
    VAE_VIDEO_SHIFT_FACTOR,
    _decode_latent,
)

HF_MODEL_ID = "Qualcomm-AI-Research/Neodragon"


def read_clip(path, num_frames, width, height):
    import imageio.v3 as iio

    frames = [Image.fromarray(f) for f in iio.imread(path, plugin="pyav")]
    if len(frames) != num_frames:  # uniform temporal resample to the student length
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    out = []
    for f in frames:
        s = max(width / f.width, height / f.height)
        f = f.resize((round(f.width * s), round(f.height * s)), Image.LANCZOS)
        left, top = (f.width - width) // 2, (f.height - height) // 2
        out.append(f.crop((left, top, left + width, top + height)).convert("RGB"))
    return out


def to_tensor(frames, device, dtype):
    arr = np.stack([np.asarray(f, np.float32) / 127.5 - 1.0 for f in frames])  # T H W C in [-1, 1]
    return rearrange(torch.from_numpy(arr), "t h w c -> 1 c t h w").to(device=device, dtype=dtype)


def scale_latents(z):
    z = z.clone()
    z[:, :, :1] = (z[:, :, :1] - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR
    z[:, :, 1:] = (z[:, :, 1:] - VAE_VIDEO_SHIFT_FACTOR) * VAE_VIDEO_SCALE_FACTOR
    return z


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="folder with <idx>.mp4 + index.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--min_psnr", type=float, default=22.0)
    args = ap.parse_args()

    device = auto_device(args.device)
    if device.type == "cpu":
        configure_cpu_threads()
    dtype = torch.float32
    pipe = NeodragonPipeline.from_pretrained(HF_MODEL_ID, torch_dtype=dtype, mode="hybrid",
                                             cache_dir=args.cache).to(device)
    vae, text = pipe.vae, pipe.text_encoder_bundle
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = [json.loads(ln) for ln in (Path(args.teacher) / "index.jsonl").read_text().splitlines() if ln.strip()]

    null = out / "null_prompt.pt"  # empty-prompt embedding for CFG dropout in train_stage_a.py
    if not null.exists():
        with torch.no_grad():
            emb, mask, pooled = text("", device)
        torch.save({"prompt_embeds": emb[0].cpu(), "prompt_attention_mask": mask[0].cpu(),
                    "pooled_prompt_embeds": pooled[0].cpu(), "prompt": ""}, null)

    for rec in index:
        dst = out / f"{rec['idx']:05d}.pt"
        if dst.exists():
            continue
        frames = read_clip(Path(args.teacher) / f"{rec['idx']:05d}.mp4", args.num_frames, args.width, args.height)
        with torch.no_grad():
            z = vae.encode(to_tensor(frames, device, dtype), temporal_chunk=True).latent_dist.mode()
            latents = scale_latents(z)
            recon = _decode_latent(vae, latents.clone())
            emb, mask, pooled = text(rec["prompt"] + DEFAULT_PROMPT_MODIFIER, device)
        rt = float(np.mean([psnr(a, b) for a, b in zip(recon, frames)]))
        info = {"idx": rec["idx"], "latent_shape": list(latents.shape[1:]), "roundtrip_psnr_db": round(rt, 2)}
        if rt < args.min_psnr:
            info["skipped"] = True
            print(json.dumps(info), flush=True)
            continue
        torch.save({
            "latents": latents[0].cpu(), "prompt_embeds": emb[0].cpu(),
            "prompt_attention_mask": mask[0].cpu(), "pooled_prompt_embeds": pooled[0].cpu(),
            "prompt": rec["prompt"], "teacher": rec, "roundtrip_psnr_db": rt,
        }, dst)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
