# Neodragon-X development log

Newest entries at the bottom. Every entry: what was tried, the number, kept or rejected, and why.
Machine: AMD EPYC Zen 4, 17 usable cores, AVX-512 + VNNI, no AVX512_BF16/AMX, no GPU, torch 2.5.1.

## 2026-10-06 — session 1 (fork → Neodragon-X)

### Getting it to run at all on a CPU
- Upstream assumes CUDA (`.to("cuda")`, `torch.cuda.amp.autocast`, DataLoader with pinned memory
  and workers). Added `neodragon/runtime.py` (`auto_device`, `auto_dtype`, `autocast_context`) and
  a `--device` flag. **Kept.**
- **Upstream crash** on torch 2.5.1: `_sample_block_noise` builds
  `MultivariateNormal(cov=(1+γ)I − γJ)`; with γ = 1/3 that matrix is singular and the constructor
  raises `ValueError: ... PositiveDefinite`. Replaced it with the exact closed form
  `x = a·z + b·mean(z)`, `a = √(1+γ)`, `b = −a + √(1−3γ)`; checked the empirical covariance
  matches. Also vectorised (0.093 s for a 1×16×1×400×640 tensor). **Kept.**
- Downloaded only what hybrid mode needs (~12 GB, `ignore_patterns=["*multistep*",
  "*.fp16.safetensors", "assets/*"]`).

### Baseline (fp32, 17 threads, seed 42, 3 prompts) — `results/cpu_benchmark.json`
- 50.5 s per 49-frame clip: SSD-1B first frame ~18 s, DiT ~22 s, video-VAE decode ~7 s,
  VAE encode ~2 s. CLIP-score 31.78. Looked at the frames: good (results/shots/baseline-*.jpg).

### Threads (from my-gpu rule 1)
- `nproc` reports 1 in this container → PyTorch used 1 thread. One DiT forward (largest call):
  1 thread 27.09 s, 4 → 8.76 s, 8 → 3.83 s, 12 → 2.72 s, 17 → 2.13 s (**12.7×**).
  `configure_cpu_threads()` uses `os.sched_getaffinity`. **Kept** (automatic).

### dtype
- bf16 on this CPU is emulated (no AVX512_BF16/AMX) → slower than fp32, as my-gpu measured
  (bf16 0.63 TFLOPS vs INT8 6.4 TOPS). `--model_dtype auto` picks fp32 on such CPUs, bf16 on CUDA
  with native bf16, fp16 otherwise. **Kept.**

### INT8 (dynamic, PyTorch, VNNI)
- All Linear layers of the DiT: 18-call sum 21.90 s → 11.46 s, but output error 16.4 %.
  **Rejected** as default.
- Selective: keep adaLN (`norm1`, `norm1_context`, ≈1/3 of all weights, they only multiply one
  vector), `norm_out`, `proj_out`, `time_text_embed`, `pos_embed`, `context_embedder` in fp32.
  18-call sum 21.17 s → 12.25 s (**1.73×**), error 3.1–8.7 % per call (mean ~5 %). **Kept.**
- Text encoder INT8: no visible quality change (i2v_fast CLIP 31.53 vs baseline 31.78). **Kept.**
- SSD-1B UNet INT8 (keeping time/add embeddings and convs fp32): end-to-end CLIP-score fell
  **31.4 → 11.5**, the first frame became an unrelated image (a man in a cap for a fox prompt).
  Evidence: `results/cpu_benchmark_rejected_unet_int8.json`. **Rejected**; `quantize_int8()` now
  defaults to `("dit", "text")`. With that fix: `tiny_vae+int8` 29.8 s, CLIP 31.38.

### Tiny VAE (my-gpu v0.2 idea)
- SSD-1B first-frame decode with TAESDXL: 8.15 s → 1.15 s, PSNR 31.4 dB at 512×320 vs the SDXL
  VAE. End-to-end CLIP 31.38 vs 31.78. **Kept** as `--tiny_first_frame_vae` (opt-in, lossy).
- PSNR of whole videos vs baseline is low (14–28 dB) because a slightly different first frame
  changes the whole autoregressive trajectory; that measures *difference*, not *quality*.
  CLIP-score + looking at the strips is the quality check.

### Image-to-video
- `generate_hybrid(image=...)` skips SSD-1B (about 40 % of CPU time) and animates the user's
  picture: 18.1 s per clip with `--fast`, CLIP 31.53. **Kept** (`--image`).

### Frame interpolation (RIFE v4, MIT, 12 MB)
- Hold-out test on a generated clip (rebuild every odd frame from its neighbours):
  RIFE **34.03 dB** vs 50/50 blend 26.49 dB. Full 2× (49 → 97 frames): 11.9 s on CPU
  (249 ms per new frame) after the oneDNN guard below (5.6 s before, but partly wrong).
  `results/rife_eval.json`. **Kept** (`--interpolate 2|4`, `--slow_motion`).

### oneDNN convolution bug (found through RIFE)
- The end-to-end CLI test (`--fast --interpolate 2 --num_frames 97`) produced **all 96
  interpolated frames black** (NaN). Traced with forward hooks to `block0.conv0.1.0`
  (Conv2d 45 → 90 channels). Comparing against a float64 `F.conv2d`: with oneDNN on, convs with
  45 or 90 input channels are wrong (NaN, or max error 0.5–0.65); 3/4/16/48/64/96/128 channels are
  exact (1e-6). Not NNPACK (disabling it changed nothing). `torch.backends.mkldnn.flags(enabled=False)`
  gives exact results. Repro: `scripts/check_onednn_conv.py` → `results/onednn_conv_check.json`.
- Checked every conv the Neodragon pipeline runs (152 of 187 hooked; DiT patch embed, both VAEs,
  SSD-1B UNet, TAESDXL): max relative error 1.0e-6 → **pipeline unaffected**
  (`results/pipeline_conv_check.json`: name, weight shape, input shape, rel. error).
- Fix: `rife._bypass_onednn_for_odd_channels` runs the 56 IFNet convs (all have 90/45/4 channels)
  on PyTorch's native kernel. Output equals the all-native run exactly (max diff 0.0). Cost: RIFE 2×
  5.6 s → 11.9 s. Next: pad IFNet to 96/48 channels with zeros to get the fast kernel back
  (ROADMAP).

### End-to-end CLI checks — `results/e2e_cli.json`
- `--fast --interpolate 2 --num_frames 97`: 193 frames at 48 fps, 93 s wall incl. model load.
  **Long video works**, but the second half over-saturates and the subject drifts out of frame
  (model distilled on 49 frames) → long-video quality is a ROADMAP item, not solved.
- `--image waves.png --fast`: 46 s wall incl. load; the picture is kept as frame 0 and animated.

### Things looked at and not done (reasons in RESEARCH.md)
- KV cache for history tokens: only stage-0 history is identical between calls → ≤10 %. Not done.
- TeaCache/FlowCache: hybrid mode has 1 step per stage, nothing to reuse. Candidate for monolithic.
- FreeInit: multiplies cost. Sparse attention: Linear layers dominate on CPU, not attention.
- `torch.compile`: blocked (no setuptools / C++ toolchain in the venv). Not measured.
- Phone (Android CPU) without NPU: not attempted; upstream targets Qualcomm NPUs via QNN.

### Tooling added
- `scripts/benchmark_cpu.py` (per-stage time, CLIP-score, PSNR vs baseline, strips).
- `scripts/eval_rife.py` (RIFE hold-out PSNR + timing).
- `notebooks/neodragon_x_free_gpu.ipynb` (Kaggle/Colab T4, **not executed** by us).
- Scripts insert the repo root in `sys.path`, so `python scripts/...` works from a clone.

## 2026-10-06 — session 2 (phase 2: faster first frame, RIFE fast path, upscaler, distillation pipeline)

### RIFE fast path (ROADMAP #12) — `results/rife_fastpath.json`
- IFNet's first convs of each block take 4/45/90 input channels. Zero-padding the weights and
  activations to 48/96 channels (extra channels are exactly 0, so the math is unchanged) lets
  every IFNet conv run on oneDNN's correct, fast kernel again. No conv is bypassed any more.
- RIFE 2× on 49 frames: **11.31 s → 5.05 s (2.24×)**. Max float difference vs the native-kernel
  path 8.4e-6 for one frame pair; after uint8 conversion 0.0005 % of pixels differ by 1 level.
  **Kept as the default** (it is lossless). `scripts/bench_rife_fastpath.py`.

### oneDNN rule, refined — `results/onednn_conv_check.json`
- A wider sweep of input-channel counts shows the bug is **in_ch > 16 and not a multiple of 16**
  (1–16 are fine, 17/45/90… are wrong, 32/48/96… are fine). `scripts/check_onednn_conv.py`
  prints the predicted vs measured result per channel count.

### First frame with fewer SSD-1B steps (ROADMAP #3) — `results/first_frame_steps.json`
- 6 prompts, TAESDXL decode, seconds per image / CLIP-score of the image:

  | variant | s / image | CLIP |
  |---|---|---|
  | 4 steps, 1024×640 (upstream) | 10.41 | 33.58 |
  | 4 steps, channels_last | 11.87 | 33.84 |
  | 4 steps, 768×480 | 6.11 | 32.01 |
  | 3 steps [999,666,333], 1024 | 7.77 | 33.64 |
  | **2 steps [999,499], 1024** | **5.48** | **33.44** |
  | 3 steps, 768 | 4.72 | 32.66 |
  | 2 steps, 768 | 3.34 | 31.28 |

- 2 steps at 1024 keeps the timesteps on the distilled grid (3 steps is off-grid and diffusers
  warns), halves the time and keeps the CLIP-score. Lower resolution costs more quality than fewer
  steps. channels_last is slower on this CPU → rejected.
- New `--first_frame_steps {2,3,4}`; `--fast` now implies `--first_frame_steps 2`.
- End-to-end, clean machine, 3 prompts (`results/cpu_benchmark_phase2.json`):
  old `--fast` (tiny VAE + INT8) **28.81 s, CLIP 31.38** → new `--fast` (+ 2-step first frame)
  **23.2 s, CLIP 32.28**. First frame 9.6–10.4 s → 5.2–6.7 s. vs the upstream pipeline
  (42.8 s same session, 50.5 s in session 1) that is **1.85–2.2× faster**.

### Real-ESRGAN upscaler (ROADMAP #5) — `results/upscale_eval.json`
- Opt-in `--upscale {2,4}` with `--upscale_model {general,fast}` (realesr-general-x4v3 and
  realesr-animevideov3, both BSD-3), applied after RIFE. 512×320 → 1024×640.

  | model | hold-out PSNR | sharpness (GT 7.67) | ms / frame | flicker ratio |
  |---|---|---|---|---|
  | bicubic | 34.68 dB | 4.00 | 10 | 1.0 |
  | general | 30.11 dB | 7.58 | 565 | 1.09 |
  | fast | 31.73 dB | 5.86 | 307 | 1.05 |

- Lower PSNR than bicubic is normal for GAN super-resolution: it restores sharpness to the level
  of the real frames instead of averaging. Visibly sharper fur/texture
  (`results/shots/upscale_bicubic_vs_*.jpg`). Cost for 49 frames: ~28 s (general) / ~15 s (fast).
  Both models checked against float64 (max error ≤ 7e-5) because of the oneDNN bug.

### torch.compile (ROADMAP #2) — `results/compile_eval.json`
- First run failed: inductor could not find `Python.h` (uv's Python keeps headers outside
  `/usr/include`). Fix: `export CPATH=<uv python>/include/python3.11`.
- Second run (same session, 17 threads): eager best **26.17 s**; `compile_dit` best **26.68 s**
  (no gain, +2%), PSNR vs eager 83.9 dB (identical output), but the **first call took 964 s**.
  The DiT recompiles once per latent unit because the history length changes, and dynamo hits
  `cache_size_limit=8`, then falls back to eager. `compile_dit_vae` not run.
- **Rejected**: no `--compile` flag. Static shapes per (stage, unit) via OpenVINO (#1) remain the
  better path for graph-level speedups.

### Distillation from Wan2.2 — `docs/DISTILLATION.md`, `results/distill_smoke.json`
- Pipeline written: teacher data with Wan2.2-TI2V-5B (separate venv, diffusers ≥ 0.35) and
  encoding into Neodragon's latent space. Both steps smoke-tested on CPU.
- Teacher speed on CPU (fp32, 17 threads): 320×512×17 f×4 steps = 141.9 s; 480×768×25 f at
  **42.4 s/step** (run stopped at step 23/30 when distillation was paused). Estimate for
  480×768×49 f×40 steps: ~1 h per clip, so CPU teacher data is a dead end.
- Stage A trainer (`distill/train_stage_a.py`) written and smoke-tested on CPU: val_loss
  0.10539 (original) → 0.10528 after 3 steps, `--resume auto` works, 14.06 M LoRA params (56 MB).
  `neodragon/lora.py` merge is exact (max error 0.0); `--lora` works end to end in the CLI.
- Real training needs a GPU and was not run; Stage B is design only. **Paused by the owner on
  2026-10-06.** Restart point: `notebooks/neodragon_x_distill_kaggle.ipynb`.
