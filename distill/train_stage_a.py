# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Stage A of Neodragon-X distillation: LoRA fine-tune of the MULTISTEP DiT on teacher (Wan2.2) clips.

Input: the `.pt` samples written by `distill/encode_for_student.py` (Neodragon latents of teacher
clips + text embeddings). Output: `<out>/lora_step<N>.safetensors` + `<out>/lora_last.safetensors`
+ `<out>/train_log.jsonl`. Load the result with
`scripts/inference_neodragon.py --mode monolithic --lora <file>`.

Objective = the Pyramid-Flow training loss (Jin et al. 2024, arXiv 2410.05954, MIT code),
reproduced to match this repo's sampler exactly:
  * pick a latent unit u >= 1 (unit 0 is the given first frame) and a pyramid stage s;
  * history = clean latents of units < u, laid out per stage by the SAME function inference uses
    (`_prepare_past_condition_latents`), optionally corrupted with a little noise;
  * stage start point = start_sigma * noise_s + (1 - start_sigma) * nearest_up(clean_{s-1})
    (pure noise at s = 0); end point = end_sigma * noise_s + (1 - end_sigma) * clean_s (clean at the
    last stage); x_t = r * start + (1 - r) * end with r ~ U(sigmas_per_stage);
  * target velocity = start - end (the scheduler integrates x += (sigma_next - sigma) * v).
Prompts are dropped with p = --cfg_drop (replaced by the empty-prompt embedding) to keep CFG working.

Why the multistep DiT and not the shipped 1-1-1 DiT: the hybrid DiT is DMD step-distilled. A plain
flow-matching loss would teach it multi-step denoising and break the 1-step speed. Stage B
(see docs/DISTILLATION.md) transfers the gain back to 1-1-1.

The validation loss is DETERMINISTIC (fixed u, s, r and noise per held-out sample), so `val_loss` at
step 0 (the original model) and later steps are directly comparable. That is the stop rule.

    # Kaggle T4 / any CUDA GPU
    python distill/train_stage_a.py --data data/student --out runs/stage_a --cache ./models
    # CPU smoke test (proves the code; far too slow for real training)
    python distill/train_stage_a.py --data /tmp/student --out /tmp/stage_a --steps 2 --device cpu
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neodragon import MULTISTEP_CONTEXT_ADAPTER_ID, MULTISTEP_DIT_ID  # noqa: E402
from neodragon.context_adapter import ContextAdapter  # noqa: E402
from neodragon.lora import apply_lora, load_lora, lora_state_dict, save_lora  # noqa: E402
from neodragon.pyramid_mmdit import PyramidMMDiT  # noqa: E402
from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler  # noqa: E402
from neodragon.runtime import auto_device, configure_cpu_threads  # noqa: E402
from neodragon.utils.generation_utils import (  # noqa: E402
    _get_pyramid_latent,
    _prepare_past_condition_latents,
)

HF_MODEL_ID = "Qualcomm-AI-Research/Neodragon"
NUM_STAGES = 3


# ----------------------------------------------------------------------------------------------- #
# data                                                                                             #
# ----------------------------------------------------------------------------------------------- #
def load_samples(folder: Path):
    files = sorted(p for p in folder.glob("*.pt") if p.name != "null_prompt.pt")
    if not files:
        raise SystemExit(f"no samples in {folder} (run distill/encode_for_student.py first)")
    null = folder / "null_prompt.pt"
    if not null.exists():
        raise SystemExit(f"{null} missing: re-run encode_for_student.py (it writes the empty-prompt embedding)")
    return files, torch.load(null, map_location="cpu")


def noise_pyramid(shape, generator, device, dtype):
    """Full-res N(0, I) noise and its 2x bilinear*2 downsamples (same as Pyramid-Flow training)."""
    noise = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    out, cur = [noise], noise
    t = noise.shape[2]
    for _ in range(NUM_STAGES - 1):
        h, w = cur.shape[-2] // 2, cur.shape[-1] // 2
        cur = rearrange(cur, "b c t h w -> (b t) c h w")
        cur = torch.nn.functional.interpolate(cur, size=(h, w), mode="bilinear") * 2.0
        cur = rearrange(cur, "(b t) c h w -> b c t h w", t=t)
        out.append(cur)
    return list(reversed(out))  # low -> high resolution


def up2_nearest(x):
    t = x.shape[2]
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    return rearrange(x, "(b t) c h w -> b c t h w", t=t)


def build_example(latents, sched, u, stage, r_index, generator, history_noise, device, dtype):
    """Returns (dit sample list, timestep [1], target) for unit u at pyramid stage `stage`."""
    lat = latents.unsqueeze(0).to(device=device, dtype=dtype)  # [1, 16, T, H, W]
    cur = lat[:, :, u:u + 1]
    history = [lat[:, :, i:i + 1] for i in range(u)]
    if history_noise > 0:
        sig = float(torch.rand(1, generator=generator)) * history_noise
        history = [(1 - sig) * h + sig * torch.randn(h.shape, generator=generator).to(h) for h in history]
    past = _prepare_past_condition_latents(history, NUM_STAGES, do_classifier_free_guidance=False)

    clean = _get_pyramid_latent(cur, NUM_STAGES)
    noise = noise_pyramid(cur.shape, generator, device, dtype)
    s_start, s_end = sched.start_sigmas[stage], sched.end_sigmas[stage]
    if stage == 0:
        start = noise[0]
    else:
        start = s_start * noise[stage] + (1 - s_start) * up2_nearest(clean[stage - 1])
    end = clean[stage] if stage == NUM_STAGES - 1 else s_end * noise[stage] + (1 - s_end) * clean[stage]
    r = sched.sigmas_per_stage[stage][r_index].to(device=device, dtype=dtype)
    t = sched.timesteps_per_stage[stage][r_index].to(device=device, dtype=dtype).reshape(1)
    x_t = r * start + (1 - r) * end
    return past[stage] + [x_t], t, (start - end)


def text_inputs(sample, null, drop, context_adapter, device, dtype):
    src = null if drop else sample
    emb = src["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
    mask = src["prompt_attention_mask"].unsqueeze(0).to(device=device)
    pooled = src["pooled_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        emb = context_adapter(emb)
    return emb, mask, pooled


def loss_for(dit, context_adapter, sched, sample, null, u, stage, r_index, gen, drop, args, device, dtype):
    x, t, target = build_example(sample["latents"], sched, u, stage, r_index, gen,
                                 args.history_noise, device, dtype)
    emb, mask, pooled = text_inputs(sample, null, drop, context_adapter, device, dtype)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        pred = dit(sample=[x], encoder_hidden_states=emb, encoder_attention_mask=mask,
                   pooled_projections=pooled, timestep_ratio=t)[0]
    return torch.nn.functional.mse_loss(pred.float(), target.float())


@torch.no_grad()
def val_loss(dit, context_adapter, sched, val_files, null, args, device, dtype):
    """Fixed (u, stage, r, noise) per held-out sample -> comparable across steps."""
    dit.eval()
    tot, n = 0.0, 0
    for i, f in enumerate(val_files):
        s = torch.load(f, map_location="cpu")
        T = s["latents"].shape[1]
        for stage in range(NUM_STAGES):
            g = torch.Generator().manual_seed(1000 * i + stage)
            u = 1 + (i + stage) % (T - 1)
            r_index = int(torch.randint(0, 1000, (1,), generator=g))
            v_args = argparse.Namespace(**{**vars(args), "history_noise": 0.0})
            tot += float(loss_for(dit, context_adapter, sched, s, null, u, stage, r_index, g, False,
                                  v_args, device, dtype))
            n += 1
    dit.train()
    return tot / max(n, 1)


# ----------------------------------------------------------------------------------------------- #
# main                                                                                             #
# ----------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of encode_for_student.py .pt samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="./models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                    help="auto = bf16 if the GPU supports it, fp16 on older GPUs (T4), fp32 on CPU")
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--history_noise", type=float, default=0.1,
                    help="max sigma of the noise mixed into the clean history (robustness to own errors)")
    ap.add_argument("--val_count", type=int, default=8, help="held-out samples for the val loss")
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--resume", default=None, help="LoRA .safetensors to continue from ('auto' = <out>/lora_last)")
    ap.add_argument("--grad_ckpt", type=float, default=0.0,
                    help="fraction of DiT blocks WITHOUT checkpointing (0 = checkpoint all; saves memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_budget_h", type=float, default=11.5,
                    help="stop and save cleanly before this many hours (Kaggle sessions end at 12 h)")
    args = ap.parse_args()

    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = auto_device(args.device)
    if device.type == "cpu":
        configure_cpu_threads()
    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files, null = load_samples(Path(args.data))
    random.Random(args.seed).shuffle(files)
    n_val = min(args.val_count, max(1, len(files) // 10)) if len(files) > 1 else 0
    val_files, train_files = files[:n_val], files[n_val:] or files
    print(f"samples: train {len(train_files)}, val {len(val_files)}; device {device}, dtype {dtype}", flush=True)

    root = snapshot_download(HF_MODEL_ID, cache_dir=args.cache,
                             allow_patterns=[f"{MULTISTEP_DIT_ID}/*", f"{MULTISTEP_CONTEXT_ADAPTER_ID}/*"])
    base_dtype = torch.float32 if dtype == torch.float32 else dtype
    dit = PyramidMMDiT.from_pretrained(f"{root}/{MULTISTEP_DIT_ID}", torch_dtype=base_dtype).to(device)
    context_adapter = ContextAdapter.from_pretrained(f"{root}/{MULTISTEP_CONTEXT_ADAPTER_ID}",
                                                     torch_dtype=base_dtype).to(device).eval()
    dit.requires_grad_(False)
    context_adapter.requires_grad_(False)
    dit.gradient_checkpointing = True
    dit.gradient_checkpointing_ratio = args.grad_ckpt
    dit.train()
    sched = PyramidFlowMatchEulerDiscreteScheduler()

    resume = str(out / "lora_last.safetensors") if args.resume == "auto" else args.resume
    start_step = 0
    if resume and Path(resume).exists():
        load_lora(dit, resume, merge=False)
        from safetensors import safe_open
        with safe_open(resume, framework="pt") as f:
            start_step = int((f.metadata() or {}).get("step", 0))
        params = [p for n, p in dit.named_parameters() if n.endswith(("lora_A", "lora_B"))]
        print(f"resumed from {resume} at step {start_step}", flush=True)
    else:
        params = apply_lora(dit, rank=args.rank, alpha=args.alpha)
    for p in params:
        p.requires_grad_(True)
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    opt_state = out / "optimizer_last.pt"
    if start_step and opt_state.exists():
        opt.load_state_dict(torch.load(opt_state, map_location="cpu"))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    log = open(out / "train_log.jsonl", "a")

    def checkpoint(step, tag):
        meta = {"step": step, "teacher": "Wan2.2-TI2V-5B", "base": MULTISTEP_DIT_ID}
        save_lora(dit, str(out / f"lora_{tag}.safetensors"), meta)
        save_lora(dit, str(out / "lora_last.safetensors"), meta)
        torch.save(opt.state_dict(), opt_state)

    if val_files and start_step == 0:
        v0 = val_loss(dit, context_adapter, sched, val_files, null, args, device, dtype)
        print(json.dumps({"step": 0, "val_loss": round(v0, 5), "note": "original model"}), flush=True)
        log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
        log.flush()

    gen = torch.Generator().manual_seed(args.seed + start_step)
    step, running = start_step, []
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            s = torch.load(random.choice(train_files), map_location="cpu")
            T = s["latents"].shape[1]
            u = random.randint(1, T - 1)
            stage = random.randrange(NUM_STAGES)
            r_index = random.randrange(1000)
            drop = random.random() < args.cfg_drop
            loss = loss_for(dit, context_adapter, sched, s, null, u, stage, r_index, gen, drop, args, device, dtype)
            scaler.scale(loss / args.grad_accum).backward()
            running.append(float(loss))
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 10 == 0 or step == args.steps:
            rec = {"step": step, "loss": round(sum(running) / len(running), 5), "grad_norm": round(gnorm, 4),
                   "elapsed_min": round((time.time() - t_start) / 60, 1)}
            running = []
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if val_files and (step % args.val_every == 0 or step == args.steps):
            v = val_loss(dit, context_adapter, sched, val_files, null, args, device, dtype)
            print(json.dumps({"step": step, "val_loss": round(v, 5)}), flush=True)
            log.write(json.dumps({"step": step, "val_loss": v}) + "\n")
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            checkpoint(step, f"step{step}")
        if (time.time() - t_start) / 3600 > args.time_budget_h:
            checkpoint(step, f"step{step}")
            print(f"time budget reached at step {step}; resume with --resume auto", flush=True)
            break
    log.close()
    print(json.dumps({"done": True, "step": step, "lora": str(out / "lora_last.safetensors"),
                      "lora_MB": round(sum(t.numel() for t in lora_state_dict(dit).values()) * 4 / 1e6, 1)}))


if __name__ == "__main__":
    main()
