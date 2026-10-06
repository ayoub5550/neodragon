# AGENTS.md — instructions for AI agents working on Neodragon-X

This repository is a fork of [Qualcomm-AI-research/neodragon](https://github.com/Qualcomm-AI-research/neodragon)
(ICLR 2026, BSD-3-Clause-Clear). The fork, **Neodragon-X**, makes the model run anywhere (CPU without a
GPU, CUDA, Apple MPS) and adds measured speed and quality techniques. Read this file first, then
`docs/NEODRAGON_X.md` (what was changed and measured), `docs/RESEARCH.md` (literature survey and
which techniques apply) and `docs/ROADMAP.md` (what to do next).

## 0. Owner rules

1. **Everything goes into the repo.** Code, the sources you relied on, measured JSON, install steps,
   dead ends. Any AI agent must be able to continue from the repo alone.
2. **Measure, don't guess.** Every optimisation gets a number in `results/*.json` (speed *and* a
   quality metric) and a line in `docs/DEVLOG.md`: research done, tools kept/rejected and why.
3. **Additive and opt-in.** Defaults must still reproduce the upstream pipeline. Lossy speed-ups
   (`--int8`, `--tiny_first_frame_vae`) are flags, never silent defaults.
4. **Never use `git`/`gh` from a shell if your platform gives you a git tool**; open a PR per topic.

## 1. Reference machine (all numbers in `results/`)

| item | value |
|---|---|
| CPU | AMD EPYC Zen 4, AVX-512 + VNNI, **17 usable cores**, *no* AVX512_BF16 / AMX |
| RAM | 328 GB (the pipeline needs ~16 GB in fp32) |
| GPU | none |
| Python / torch | 3.11 / 2.5.1, diffusers 0.34.0, transformers 4.46.3 |

## 2. Setup (≈10 min, ≈11 GB of weights)

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate        # or python -m venv
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-x.txt
python scripts/inference_neodragon.py --prompt "A red fox running in snow" --seed 42 --fast \
       --disable_safety_checker --output_video_folder output
```

Weights download to `./models` on first run (`--local_cache_folder`). To fetch only what hybrid mode
needs: `huggingface_hub.snapshot_download("Qualcomm-AI-Research/Neodragon", cache_dir="models",
ignore_patterns=["*multistep*", "*.fp16.safetensors", "assets/*"])`.

## 3. Pitfalls that cost time (all verified here)

1. **Thread count.** Inside containers `nproc` / `os.cpu_count()` may report 1, so PyTorch runs
   single-threaded: one DiT forward is 27.1 s instead of 2.1 s (12.7×). `neodragon.runtime.
   configure_cpu_threads()` uses `os.sched_getaffinity`; the script calls it automatically.
2. **bf16 on CPU** is emulated (slow) unless the CPU has AVX512_BF16 or AMX. `--model_dtype auto`
   picks fp32 then. fp16 on CPU is never used.
3. **Upstream crashed on PyTorch ≥ 2.5** (tested 2.5.1) in `_sample_block_noise`: the
   block-noise covariance is singular for gamma = 1/3. Fixed with an exact closed-form sampler —
   do not reintroduce `MultivariateNormal` there.
4. `torch.cuda.amp.autocast` is a no-op on CPU; use `neodragon.runtime.autocast_context`.
5. INT8: quantising *all* Linear layers gives 16 % output error; keep the adaLN (`norm1*`), `norm_out`,
   `proj_out`, `time_text_embed` in fp32 (5 % error, and faster). `quantize_linear_int8` does that.
   **Never INT8 the SSD-1B UNet**: end-to-end CLIP-score fell 31.4 → 11.5 (the first frame stops
   following the prompt), so `quantize_int8()` defaults to `("dit", "text")` only.
6. RIFE needs H and W padded to multiples of 32 (handled in `neodragon/postprocess/rife.py`).
7. NNPACK warning "Unsupported hardware" on AMD CPUs is harmless.
8. **oneDNN conv bug** (torch 2.5.1 / oneDNN 3.5.3, AMD Zen 4): Conv2d with **in_ch > 16 and not a
   multiple of 16** (e.g. 17, 45, 90) returns wrong values or NaN; 1–16 and multiples of 16 are exact.
   Workaround that keeps the fast kernel: zero-pad channels to the next multiple of 16 (RIFE does this). Run `scripts/check_onednn_conv.py`
   on a new machine/torch version. Any new conv model you add (upscaler, etc.) must be checked
   against a float64 reference; RIFE already uses the guard in `neodragon/postprocess/rife.py`.
9. **Wan2.2 needs its own venv** (diffusers ≥ 0.35, newer torch); Neodragon is pinned to
   diffusers 0.34 / torch 2.5.1. Never install `distill/requirements-wan.txt` into the main venv.
10. **torch.compile on CPU** needs a C++ compiler *and* the Python headers (`Python.h`). A system
    Python without `python3-dev` fails with `CppCompileError`; install the dev package or point
    `CPATH` at a uv-managed CPython's `include/python3.11`.
11. Don't benchmark while anything else uses the CPU (downloads, other jobs): timings drift 20–40 %.

## 4. Map of the code

```
neodragon/runtime.py            device / dtype / threads / autocast / seeds / INT8 helpers   (new)
neodragon/postprocess/rife.py   RIFE v4 frame interpolation (MIT, vendored IFNet, padded fast path) (new)
neodragon/postprocess/upscale.py Real-ESRGAN compact upscaler (BSD-3, opt-in --upscale)      (new)
neodragon/utils/generation_utils.py   sampling loop; exact block noise, seeds, image-to-video
neodragon/__init__.py           NeodragonPipeline (+ enable_tiny_first_frame_vae, quantize_int8)
scripts/inference_neodragon.py  CLI (--prompt --image --seed --fast --int8 --interpolate ...)
scripts/benchmark_cpu.py        speed + CLIP-score + PSNR benchmark -> results/cpu_benchmark*.json
                                (config names: baseline, tiny_vae, int8, +ff2/+ff3 = first-frame steps)
scripts/bench_first_frame.py    SSD-1B steps / resolution sweep -> results/first_frame_steps.json
scripts/bench_rife_fastpath.py  padded vs native RIFE -> results/rife_fastpath.json
scripts/bench_upscale.py        upscaler PSNR / sharpness / flicker / speed -> results/upscale_eval.json
scripts/bench_compile.py        torch.compile variants -> results/compile_eval.json
scripts/check_onednn_conv.py    oneDNN conv correctness sweep -> results/onednn_conv_check.json
distill/                        Wan2.2 teacher data, encoding, Stage A LoRA trainer (docs/DISTILLATION.md)
neodragon/lora.py               LoRA inject/save/load/merge (used by train_stage_a.py and --lora)
notebooks/neodragon_x_free_gpu.ipynb   run on a free Kaggle/Colab T4
notebooks/neodragon_x_distill_kaggle.ipynb   free distillation run (Kaggle 2xT4, resumable)
docs/                           NEODRAGON_X.md, RESEARCH.md, DEVLOG.md, ROADMAP.md, DISTILLATION.md
results/                        measured JSON + frame strips (results/shots/)
```

## 5. How to benchmark a new idea

```bash
python scripts/benchmark_cpu.py --configs baseline,<your_config> --out results/<name>.json
```
Add your config to `benchmark_cpu.py`, keep the fixed seed and the same three prompts, and report
total time, per-stage time (first_frame / dit / vae_encode / vae_decode) and CLIP-score.
Run nothing else on the machine while measuring.
