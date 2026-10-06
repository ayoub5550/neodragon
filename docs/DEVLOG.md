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
