# Distilling a big open model (Wan2.2) into Neodragon-X

Status: **paused by the owner (2026-10-06).** Steps 1–2 (teacher data, encoding) and the Stage A trainer are implemented and
smoke-tested on CPU** (`results/distill_smoke.json`). Real training needs a GPU. The free path is
Kaggle's ~30 GPU-hours/week with the ready notebook `notebooks/neodragon_x_distill_kaggle.ipynb`
(§6). Stage B (bringing the gain back to the fast 1-1-1 mode) is designed, not written.

## 1. Why this is not "download Wan and copy it in"

| Fact | Consequence |
|---|---|
| Wan2.2-TI2V-5B is Apache-2.0. Its generated videos may be used as training data | ✅ legal to use as a teacher. Wan2.2-A14B (126 GB) is also Apache-2.0 but costs about 3× more |
| Wan2.2's VAE is 48 ch, 4×16×16. Neodragon's is 16 ch, 8×8×8 | Latent spaces differ, so logit/feature/score distillation is **impossible directly**. Distil through **pixels**: teacher video → Neodragon VAE encoder → student latents |
| The hybrid DiT is **DMD step-distilled** (1 step per pyramid stage, "1-1-1") | A plain flow-matching fine-tune on teacher clips would teach it multi-step denoising and **break the 1-step speed**. The fine-tune has to target the *multistep* DiT, which is then re-distilled. Alternatively use an adversarial objective that keeps 1 step |
| The repo ships the pre-distillation DiT: `diffusion_transformer_320p_multistep_t2v` (+ its context adapter) | ✅ there is a proper flow-matching model to fine-tune. We don't need to recreate Qualcomm's teacher |
| Prior art: **S2DiT** (arXiv 2601.12719) distilled a streaming mobile DiT from Wan2.2-14B | The direction is validated, but at a large GPU budget |

## 2. Recipe

```
[Wan2.2-TI2V-5B]  --make_teacher_data.py-->  teacher mp4s (480x768, 49 f, 24 fps)
                                                  |
                    encode_for_student.py  (resize 512x320, Neodragon VAE + text bundle)
                                                  v
                                   student samples .pt  (latents 16x7x40x64 + text embeds)
                                                  |
      Stage A: LoRA / full fine-tune of the MULTISTEP DiT with the Pyramid-Flow loss
               (temporal pyramid, per-stage noise, history conditioning, as in Pyramid-Flow's
               train script, MIT); evaluate in `--mode monolithic`
                                                  |
      Stage B: bring the gain back to 1-1-1:
        B1. DMD2: real score = fine-tuned multistep DiT, fake score = trainable copy,
            generator = hybrid 1-1-1 DiT.  (Same recipe Qualcomm used, now with a better teacher)
        B2. or adversarial post-training (LADD / APT style): real = teacher clips,
            fake = 1-1-1 student samples, discriminator = frozen DiT features + light heads
                                                  v
                                  new hybrid DiT -> same speed, better motion and detail
```

**Evaluate every stage** with `scripts/benchmark_cpu.py` (CLIP-score, plus PSNR vs baseline where
meaningful) and, ideally, VBench subsets (motion smoothness, subject consistency, dynamic degree).
Keep a held-out prompt list that is not in `distill/prompts.txt`.

## 3. Files

| File | Env | What it does |
|---|---|---|
| `distill/prompts.txt` | – | 60 diverse prompts (subjects × motion × camera × light). Use `--clips_per_prompt` for more data, and add prompts for real runs (2k+ unique) |
| `distill/make_teacher_data.py` | **separate venv** (`distill/requirements-wan.txt`, diffusers ≥ 0.35) | Wan2.2-TI2V-5B text-to-video. Resumable, `--shard i/n` for several GPUs, `--cpu_offload` for <24 GB cards, writes `index.jsonl` |
| `distill/encode_for_student.py` | Neodragon venv | Resize/crop to 512×320 and resample to 49 frames. Encodes with Neodragon's VAE (image factors for frame 0, video factors for the rest, exactly like the sampler) and text bundle. Checks round-trip PSNR and skips bad samples |
| `distill/train_stage_a.py` | Neodragon venv | **Stage A**: LoRA on the multistep DiT with the Pyramid-Flow loss (history laid out by the same `_prepare_past_condition_latents` as inference, per-stage start/end points, velocity target, CFG dropout). Deterministic held-out `val_loss` (step 0 = original model) as the stop rule. fp16 + GradScaler on T4, bf16 on Ampere+, fp32 on CPU. `--time_budget_h` + `--resume auto` for 12 h Kaggle sessions |
| `neodragon/lora.py` | – | Dependency-free LoRA (no `peft`): wrap/save/load/merge. Default targets: q/k/v/out of both attention streams, rank 32 |
| `scripts/inference_neodragon.py --lora F [--lora_strength s]` | Neodragon venv | Merges a trained LoRA into the DiT before generation (zero runtime cost). Use with `--mode monolithic` until Stage B exists |
| `notebooks/neodragon_x_distill_kaggle.ipynb` | Kaggle 2×T4 | Teacher clips (one Wan worker per GPU) → encode → Stage A with a time budget → original-vs-distilled comparison. Everything persists between sessions via *Save Version* |
| `distill/train_stage_b.py` | – | **Not written yet** (DMD2 or adversarial, see §2) |

## 4. Cost (what a run needs)

The CPU numbers are measured (`results/distill_smoke.json`). The GPU numbers are **estimates**
from public Wan2.2 figures ("5 s 720p in under 9 min on one RTX 4090") scaled by token count.

| Item | CPU dev box (17 Zen-4 cores) | 1× RTX 4090 / A100 (estimate) |
|---|---|---|
| Wan2.2-5B teacher, 480×768×49 f, 40 steps + CFG | ~1 h per clip (estimate from the measured 42.4 s/step at 25 f) | ~1–3 min per clip |
| 2 000 teacher clips | months | ~35–100 GPU-h |
| Encoding for the student | ~10 s per clip | <1 s per clip |
| Stage A LoRA (multistep DiT, ~0.75 B) | not feasible | ~20–50 GPU-h |
| Stage B (DMD2 / adversarial) | not feasible | ~50–150 GPU-h |
| **Total first iteration** | – | **≈ 100–300 GPU-h ≈ $50–250 rented** (Kaggle's free 2×T4 quota of ~30 h/week covers teacher data slowly) |

There is no guarantee of a quality gain. Treat it as an experiment with a measurable stop rule:
stop if Stage A's monolithic CLIP / VBench score does not beat the original multistep DiT.

## 5. Smoke test (reproduce)

```bash
# teacher env
python -m venv .venv-wan && .venv-wan/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
  && .venv-wan/bin/pip install -r distill/requirements-wan.txt
.venv-wan/bin/python distill/make_teacher_data.py --out /tmp/teacher --limit 1 \
  --height 320 --width 512 --num_frames 17 --steps 4 --device cpu --dtype fp32
# student env (this repo's)
python distill/encode_for_student.py --teacher /tmp/teacher --out /tmp/student --num_frames 17
```

## 6. Free path: Kaggle (no GPU of your own)

Kaggle gives a phone-verified account ~30 GPU-hours per week (2× T4 16 GB, 29 GB RAM, 12 h per
session). Plan per week with `notebooks/neodragon_x_distill_kaggle.ipynb`:

| Session | Part | What happens |
|---|---|---|
| 1–2 | B + C | Wan2.2-5B teacher clips, two in parallel (one per T4, fp16 + CPU offload). **Estimate**: 5–10 min per clip per T4 → ~100–200 clips per 12 h session |
| 3 | D | Stage A LoRA for ~10 h. Stop rule: `val_loss` must fall below the step-0 value |
| any | E | Same prompt + seed, original vs distilled, side by side |

T4 timings above are **estimates** (T4 ≈ 1/4–1/6 of an RTX 4090 in fp16). The notebook has not yet
been run on Kaggle by the author. The first session will give the real numbers; record them in
`results/gpu_t4.json`. Expect a visible gain in motion/detail in monolithic mode after a few hundred
clips. Matching Wan2.2 quality is not realistic on this budget.
