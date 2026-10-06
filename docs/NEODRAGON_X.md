# Neodragon-X — what this fork adds, with measurements

[العربية في الأسفل](#بالعربية)

Neodragon (Qualcomm, ICLR 2026) was released as CUDA-only research code. Neodragon-X makes it run
**on any machine** (CPU without a GPU, NVIDIA CUDA, Apple MPS), makes it **3× faster on a CPU**
for image-to-video and **~2× faster** for text-to-video at the same or better prompt adherence, fixes a crash,
and adds image-to-video, seeds and smooth 48 fps output. Defaults still reproduce the upstream
pipeline; every lossy speed-up is an opt-in flag.

## Results (17-core AMD Zen 4 CPU, no GPU, fp32, seed 42, 3 prompts)

Source: `results/cpu_benchmark.json` and `results/cpu_benchmark_phase2.json` (made by
`scripts/benchmark_cpu.py`). 49 frames, 512×320.

| config | flags | time / clip | first frame | DiT | VAE decode | CLIP-score |
|---|---|---|---|---|---|---|
| upstream pipeline (with the thread + crash fixes) | — | **50.5 s** | 18.4 s | 22.2 s | 7.2 s | 31.78 |
| tiny first-frame VAE | `--tiny_first_frame_vae` | 42.0 s | 12.2 s | 21.1 s | 6.0 s | 31.38 |
| tiny VAE + INT8 (session 1 `--fast`) | `--tiny_first_frame_vae --int8` | 29.8 s | 10.8 s | 12.1 s | 4.9 s | 31.38 |
| **+ 2-step first frame (new `--fast`)** | `--fast` | **23.2 s** | 5.7 s | 11.4 s | 4.2 s | **32.28** |
| image-to-video | `--image photo.jpg --fast` | **18.1 s** | — | 11.7 s | 4.4 s | 31.53 |
| + RIFE 2× (49 → 97 frames) | `--interpolate 2` | +5.05 s | | | | hold-out 34.0 dB |
| + Real-ESRGAN 2× (→ 1024×640) | `--upscale 2 --upscale_model fast` | +~15 s (49 f) | | | | sharpness 5.9 (GT 7.7, bicubic 4.0) |

Frame strips: `results/shots/`. Without the thread fix the same CPU needs ~13× longer.

## Changes

| # | change | why / number | where |
|---|---|---|---|
| 1 | Run on CPU / MPS / CUDA (`--device auto`) | upstream hard-codes CUDA | `neodragon/runtime.py`, CLI |
| 2 | Real thread count via `sched_getaffinity` | containers report 1 CPU; DiT pass 27.1 s → 2.1 s | `configure_cpu_threads` |
| 3 | `--model_dtype auto` | bf16 is emulated on CPUs without AVX512_BF16/AMX → fp32 there | `auto_dtype` |
| 4 | **Crash fix**: exact block-noise sampler | `MultivariateNormal` with a singular covariance raises on torch ≥ 2.5 | `generation_utils._sample_block_noise` |
| 5 | Selective dynamic INT8 (DiT + text encoder) | DiT 22 s → 12 s, CLIP unchanged; adaLN / IO stay fp32 (error 16 % → 5 %) | `quantize_int8`, `--int8` |
| 6 | TAESDXL for the SSD-1B frame | decode 8.2 s → 1.2 s, 31.4 dB | `--tiny_first_frame_vae` |
| 7 | **Image-to-video** in hybrid mode | animate any picture; skips SSD-1B | `--image` |
| 8 | Reproducible seeds | one generator drives every random draw | `--seed` |
| 9 | **RIFE v4 frame interpolation** | 49 → 97/193 frames; +7.5 dB over blending | `neodragon/postprocess/`, `--interpolate`, `--slow_motion` |
| 10 | Multi-prompt CLI, safe filenames, CPU-friendly DataLoader | quality of life | `scripts/inference_neodragon.py` |
| 11 | Benchmarks + evals | speed *and* quality for every idea | `scripts/benchmark_cpu.py`, `scripts/eval_rife.py` |
| 12 | Free-GPU notebook | Kaggle / Colab T4 (not executed by us) | `notebooks/` |
| 13 | **oneDNN conv bug guard** | torch 2.5.1 / oneDNN 3.5.3 on Zen 4 returns wrong Conv2d results when in_ch > 16 and not a multiple of 16 (45 → NaN, 90 → error 0.5–0.65). Neodragon's 152 convs are safe (checked) | `scripts/check_onednn_conv.py`, `results/onednn_conv_check.json`, `results/pipeline_conv_check.json` |

| 14 | **2-step first frame** | SSD-1B 4 → 2 steps on the distilled grid [999, 499]: 10.4 s → 5.5 s, image CLIP 33.6 → 33.4; part of `--fast` | `--first_frame_steps`, `results/first_frame_steps.json` |
| 15 | **RIFE fast path** | zero-pad IFNet to 48/96 channels so oneDNN's correct kernel applies: 11.3 s → 5.05 s, lossless (8e-6) | `neodragon/postprocess/rife.py`, `results/rife_fastpath.json` |
| 16 | **Real-ESRGAN upscaler** (opt-in) | 512×320 → 1024×640, restores sharpness; 307–565 ms/frame | `neodragon/postprocess/upscale.py`, `--upscale`, `results/upscale_eval.json` |
| 17 | **Distillation pipeline from Wan2.2** (data tools + Stage A LoRA trainer; real training not run yet) | teacher videos → Neodragon latents → LoRA on the multistep DiT; CPU smoke-tested, real run on free Kaggle GPUs | `distill/`, `docs/DISTILLATION.md` |
| 18 | **LoRA support** | dependency-free LoRA (inject / save / exact merge, 0 inference overhead); `--lora file.safetensors --lora_strength 1.0` with `--mode monolithic` | `neodragon/lora.py`, CLI |
| 19 | **Free distillation notebook** | teacher data + training on Kaggle 2×T4, resumes across 12 h sessions | `notebooks/neodragon_x_distill_kaggle.ipynb` |

Tried and rejected (details in `docs/DEVLOG.md`): INT8 for *all* Linear layers (16 % error),
INT8 for the SSD-1B UNet (CLIP 31.4 → 11.5), bf16 on this CPU (slower), KV cache for history
(≤10 %), step caching in hybrid mode (only one step per stage), channels_last for SSD-1B (slower),
768×480 first frame (loses more CLIP than 2 steps), `torch.compile` on CPU (no gain, 16-minute
warm-up: `results/compile_eval.json`).

## Usage

```bash
python scripts/inference_neodragon.py --prompt "A red fox running in snow" --seed 42 --fast \
       --interpolate 2 --disable_safety_checker --output_video_folder output
python scripts/inference_neodragon.py --prompt "A red fox running in snow" --fast --interpolate 2 \
       --upscale 2 --upscale_model fast     # 97 frames at 1024x640
python scripts/inference_neodragon.py --prompt "waves crashing on rocks" --image my_photo.jpg --fast
```

## Honest caveats

- Measured on one CPU class only. The CUDA/MPS paths follow the same code but were **not run**
  (no GPU here). The notebook says so.
- INT8 and the tiny VAE are lossy: CLIP-score is unchanged, but videos differ from fp32 ones.
- CLIP-score (ViT-B/32, mean over frames) measures prompt adherence, not motion quality.
- 512×320 / 49 frames is the trained setting. `--num_frames 97` runs, but colours over-saturate and
  the subject drifts in the second half (`results/shots/e2e_t2v_97f_rife2x.jpg`). For longer smooth
  clips prefer 49 frames + `--interpolate 2 --slow_motion`.

---

## بالعربية

**Neodragon-X** نسخة مطوّرة من Neodragon (كوالكوم، ICLR 2026). الكود الأصلي كان يعمل فقط على بطاقات
NVIDIA. هذه النسخة:

- **تعمل على أي جهاز**: معالج CPU بلا بطاقة رسوميات، أو NVIDIA، أو Apple (MPS).
- **أسرع**: على معالج 17 نواة، النص ← فيديو من **50.5 ثانية إلى 23.2 ثانية** (‎`--fast`‎)، والصورة ←
  فيديو في **18.1 ثانية**، مع التزام أفضل بالوصف (CLIP-score ‏32.3 مقابل 31.8 للأصل).
- **الإطار الأول بخطوتين بدل 4** (‎`--first_frame_steps 2`‎، ضمن ‎`--fast`‎): من 10.4 إلى 5.5 ثانية.
- **RIFE أسرع 2.24 مرة** (11.3 ← 5.05 ثانية) بدون أي فرق في النتيجة، بحشو القنوات إلى مضاعفات 16.
- **تكبير الدقة** إلى 1024×640 بـ Real-ESRGAN (‎`--upscale 2`‎، اختياري): صورة أوضح بكثير.
- **خط تقطير من Wan2.2** (‎`distill/`‎ و ‎`docs/DISTILLATION.md`‎): توليد بيانات المعلّم وترميزها ومدرّب LoRA
  (المرحلة A) جاهزة ومختبرة على المعالج. التدريب الحقيقي يُشغَّل مجانًا على Kaggle عبر
  ‎`notebooks/neodragon_x_distill_kaggle.ipynb`‎، ثم يُحمَّل الناتج بـ ‎`--lora`‎.
- **أصلحنا عطلًا** يمنع الكود الأصلي من العمل على PyTorch 2.5 (مصفوفة تغاير منفردة في توليد الضجيج).
- **اكتشفنا عطلًا في مكتبة oneDNN** (مع PyTorch 2.5.1 على معالجات AMD Zen 4): عمليات الالتفاف تعطي نتائج خاطئة
  عندما لا يكون عدد القنوات من مضاعفات 16. تحققنا أن Neodragon نفسه سليم، وحمينا RIFE منه.
- **ميزات جديدة**: تحريك صورتك الخاصة (‎`--image`‎)، بذرة ثابتة لنتائج قابلة للتكرار (‎`--seed`‎)،
  ومضاعفة الإطارات بـ RIFE (‏49 ← 97 إطارًا، فيديو أنعم 48 إطارًا/ثانية أو تصوير بطيء).
- **كل تقنية مقاسة**: الوقت والجودة في `results/`، وما جُرّب ورُفض ولماذا في `docs/DEVLOG.md`.

أهم ما جاء من مشروع my-gpu: عدد الأنوية الحقيقي (أسرع 12.7 مرة)، fp32 بدل bf16 على المعالجات التي لا
تدعمه، INT8 مع VNNI لطبقات Linear (مع إبقاء الطبقات الحساسة بدقة كاملة)، ومفكك TAESD الصغير.

**تحذيرات**: المسار على GPU لم يُجرَّب (لا توجد بطاقة هنا). INT8 والمفكك الصغير يغيّران الفيديو قليلًا.
INT8 على UNet الخاص بالإطار الأول أفسد النتيجة فرُفض.
