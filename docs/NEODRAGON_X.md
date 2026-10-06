# Neodragon-X — what this fork adds, with measurements

[العربية في الأسفل](#بالعربية)

Neodragon (Qualcomm, ICLR 2026) was released as CUDA-only research code. Neodragon-X makes it run
**on any machine** (CPU without a GPU, NVIDIA CUDA, Apple MPS), makes it **3× faster on a CPU**
for image-to-video and **1.7× faster** for text-to-video at the same prompt adherence, fixes a crash,
and adds image-to-video, seeds and smooth 48 fps output. Defaults still reproduce the upstream
pipeline; every lossy speed-up is an opt-in flag.

## Results (17-core AMD Zen 4 CPU, no GPU, fp32, seed 42, 3 prompts)

Source: `results/cpu_benchmark.json` (made by `scripts/benchmark_cpu.py`). 49 frames, 512×320.

| config | flags | time / clip | first frame | DiT | VAE decode | CLIP-score |
|---|---|---|---|---|---|---|
| upstream pipeline (with the thread + crash fixes) | — | **50.5 s** | 18.4 s | 22.2 s | 7.2 s | 31.78 |
| tiny first-frame VAE | `--tiny_first_frame_vae` | 42.0 s | 12.2 s | 21.1 s | 6.0 s | 31.38 |
| tiny VAE + INT8 | `--fast` | **29.8 s** | 10.8 s | 12.1 s | 4.9 s | 31.38 |
| image-to-video | `--image photo.jpg --fast` | **18.1 s** | — | 11.7 s | 4.4 s | 31.53 |
| + RIFE 2× (49 → 97 frames) | `--interpolate 2` | +11.9 s | | | | hold-out 34.0 dB |

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
| 13 | **oneDNN conv bug guard** | torch 2.5.1 / oneDNN 3.5.3 on Zen 4 returns wrong Conv2d results for channel counts not divisible by 16 (45 → NaN, 90 → error 0.5–0.65). Neodragon's 152 convs are safe (checked); RIFE's 90/45-channel convs run on the native kernel | `scripts/check_onednn_conv.py`, `results/onednn_conv_check.json`, `results/pipeline_conv_check.json` |

Tried and rejected (details in `docs/DEVLOG.md`): INT8 for *all* Linear layers (16 % error),
INT8 for the SSD-1B UNet (CLIP 31.4 → 11.5), bf16 on this CPU (slower), KV cache for history
(≤10 %), step caching in hybrid mode (only one step per stage).

## Usage

```bash
python scripts/inference_neodragon.py --prompt "A red fox running in snow" --seed 42 --fast \
       --interpolate 2 --disable_safety_checker --output_video_folder output
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
- **أسرع**: على معالج 17 نواة، النص ← فيديو من **50.5 ثانية إلى 29.8 ثانية** (‎`--fast`‎)، والصورة ←
  فيديو في **18.1 ثانية**، بنفس دقة الالتزام بالوصف (CLIP-score ‏31.4–31.8).
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
