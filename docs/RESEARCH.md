# Research notes — what can make Neodragon better, and what actually applies

*Neodragon-X, first research pass, 2026-10-06. Survey done with web search over 2024–2026 papers and
repos, plus the owner's [`my-gpu`](https://github.com/ayoub5550/my-gpu) project (CPU-as-GPU techniques).
Every "applies" verdict below was checked against the Neodragon code, not just the paper abstract.*

## 1. What Neodragon is (from the paper and the code)

Neodragon (Qualcomm AI Research, ICLR 2026, arXiv 2511.06055) starts from **Pyramidal Flow**
(arXiv 2410.05954, ICLR 2025) and applies four optimisations to fit a phone NPU:

1. **Text-encoder distillation**: T5-XXL (4.7 B) → DistilT5 (0.2 B) + a "context adapter" MLP.
2. **Asymmetric decoder distillation**: smaller causal video-VAE decoder.
3. **MMDiT block pruning**: 18 joint-attention blocks remain (1.5 B parameters total, 1/3 of which
   are adaLN modulation weights that only multiply one vector).
4. **Step distillation (DMD)**: 1 denoising step per pyramid stage ("1-1-1").

Hybrid mode (default): SSD-1B (distilled SDXL, 4 LCM steps at 1024×640) makes the first frame,
then the DiT generates 6 more latent frames autoregressively, each in 3 pyramid stages
(10×16 → 20×32 → 40×64 latents), conditioned on a compressed history of previous frames.
The causal VAE decodes 7 latent frames into 49 RGB frames (512×320, 24 fps, ~2 s).

## 2. Where the time goes on a CPU (measured, `results/cpu_benchmark.json`)

| stage | fp32, 17 threads | share |
|---|---|---|
| SSD-1B first frame (4 UNet steps ≈ 2.4 s each + SDXL VAE decode 8.2 s) | 15–22 s | ~40 % |
| DiT, 18 forward passes (6 frames × 3 stages) | ~21 s | ~42 % |
| causal video-VAE decode (7 → 49 frames) | ~7 s | ~14 % |
| first-frame VAE encode | ~2 s | ~4 % |

So on CPU there are three comparable bottlenecks; no single trick gives >2× alone. After this
session: `--fast` 29.8 s, image-to-video 18.1 s (table in `docs/NEODRAGON_X.md`).

## 3. Technique catalogue

Verdicts: **✅ implemented** in this fork (measured), **🟡 candidate** (applies, not yet done),
**❌ does not apply** (with the reason, so nobody retries it blindly).

### 3.1 Runtime / systems (from my-gpu)

| technique | source | verdict |
|---|---|---|
| Real thread count via `sched_getaffinity` (containers lie about `nproc`) | my-gpu AGENTS.md rule 1 | ✅ 12.7× on one DiT pass (27.1 s → 2.13 s) |
| fp32 instead of bf16 on CPUs without AVX512_BF16/AMX | my-gpu ROADMAP (bf16 0.63 TFLOPS vs INT8 6.4 TOPS on Zen 4) | ✅ `--model_dtype auto` |
| INT8 with VNNI for matmuls | my-gpu (oneDNN 6.45 TOPS, OpenVINO 3× on ResNet-50) | ✅ PyTorch dynamic INT8, selective, DiT + text encoder: DiT 22 s → 12 s, CLIP unchanged. ❌ on the SSD-1B UNet (CLIP 31.4 → 11.5) |
| Tiny VAE (TAESD family) for decodes | my-gpu v0.2 (SD-Turbo decode 10 s → 1.6 s); madebyollin/taesdxl | ✅ for the SSD-1B frame (8.15 s → 1.15 s, 31.4 dB) |
| OpenVINO graph compile + NNCF INT8 of the DiT | my-gpu v0.4 | 🟡 likely more than PyTorch dynamic INT8 (static activation quant, fused kernels); needs ONNX export of the variable-length pyramid input |
| `torch.compile` (inductor, CPU) | PyTorch | 🟡 blocked here: inductor needs setuptools + C++ toolchain in the venv; try next |
| ISPC / custom AVX-512 kernels | my-gpu v0.2 | ❌ for now: the DiT is GEMM-bound and oneDNN/MKL GEMMs are already near peak |

### 3.2 Diffusion-sampling accelerations

| technique | source | verdict |
|---|---|---|
| Step-redundancy caching: TeaCache, FasterCache, **FlowCache** (2.38× MAGI-1, 6.7× SkyReels-V2) | arXiv 2602.10825 (ICLR 2026) | ❌ for hybrid mode — there is only *one* step per stage, nothing to reuse. 🟡 for **monolithic** mode (20/10 steps + CFG), where caching residuals between adjacent steps should give 1.5–2× |
| KV cache for autoregressive history: Forcing-KV (1.35–1.5×), In-Flight KV with clean anchors, PackCache, FAST-AR, History-Guided Cache | arXiv 2605.09681, 2609.32540, 2601.04359, 2602.01801, CVPR 2026 | ❌ mostly. Analysis of `merge_input`: history frames are re-fed at a *different pyramid resolution* each unit (u-1 at the current stage, u-2 one stage lower, older at stage 0), the timestep differs per stage, and the frame that was "current" last unit was noisy, not clean. Only frames already compressed to stage-0 size are byte-identical between calls: 160 of 1 600 tokens in the largest call → ≤10 % saving for a large code change. Worth it only for much longer videos |
| Sparse / sliding-tile attention | FAST-AR, STA | ❌ sequence ≤ ~2 000 tokens; on CPU the Linear layers dominate, not attention |
| Self Forcing / DMD retraining | Self Forcing, Neodragon §4 | ❌ needs training; Neodragon already uses DMD |

### 3.3 Quality / features without retraining

| technique | source | verdict |
|---|---|---|
| **Image-to-video** with the user's picture as first frame | Pyramidal Flow supports it natively; Neodragon's `generate(image=…)` | ✅ exposed in hybrid mode (`--image`); also skips SSD-1B (~40 % of CPU time) |
| **Frame interpolation** (RIFE v4, 12 MB, MIT) | ECCV2022-RIFE; `TensorForger/RIFE-safetensors` | ✅ `--interpolate 2|4`: 49 → 97/193 frames (48 fps smooth, or slow motion) |
| Reproducible seeds | — | ✅ `--seed`; one CPU generator drives every random draw |
| Longer videos (> 49 frames) | DiT config `max_num_frames: 200`; Pyramidal Flow does 10 s | 🟡 tested in DEVLOG; model was distilled on 49-frame clips |
| FreeInit (noise re-initialisation) | arXiv 2312.07537 | ❌ for hybrid: it re-runs sampling N times, opposite of the mobile goal |
| FreeNoise / Diff-VF (long video by noise rescheduling) | arXiv 2310.15169, 2608.05976 | 🟡 only if long videos drift; the autoregressive design already extends natively |
| Video super-resolution (Real-ESRGAN compact, 2×) | xinntao/Real-ESRGAN | 🟡 post-process to 1024×640; check CPU cost per frame |
| Prompt enrichment with a small local LLM | (common practice; my-gpu runs Qwen/Llama on CPU) | 🟡 the model was trained with a cinematic modifier — a short rewrite step may raise CLIP-score |
| Temporal token merging | On-device Sora (training-free) | 🟡 could shrink history tokens; needs quality check |

### 3.4 Related mobile video models (context for the next big step)

- **MobileVD** (Qualcomm, ICCV 2025): 14×512×256 latents in 1.7 s on a Xiaomi-14 Pro, single step.
- **MobileWan** (arXiv 2607.06173): Wan2.2-5B on phones via recurrent reformulation + compression.
- **S2DiT** (arXiv 2601.12719): sandwich DiT, streaming on mobile, distilled from Wan2.2-14B.
- **On-device Sora**, **MOVD** (WACV 2026, iPhone 15 Pro), **CineMobile** (camera-motion I2V).
These need training or are different models; useful as quality references, not drop-in changes.

## 4. Sources

- Neodragon: https://arxiv.org/abs/2511.06055 · https://qualcomm-ai-research.github.io/neodragon · https://github.com/Qualcomm-AI-research/neodragon · https://huggingface.co/Qualcomm-AI-Research/Neodragon
- Pyramidal Flow: https://arxiv.org/abs/2410.05954 · https://github.com/jy0205/Pyramid-Flow
- FlowCache: https://arxiv.org/abs/2602.10825 · Forcing-KV: https://arxiv.org/abs/2605.09681 · In-Flight KV: https://arxiv.org/abs/2609.32540 · PackCache: https://arxiv.org/abs/2601.04359 · FAST-AR: https://arxiv.org/abs/2602.01801
- FreeInit: https://arxiv.org/abs/2312.07537 · FreeNoise: https://arxiv.org/abs/2310.15169 · Diff-VF: https://arxiv.org/abs/2608.05976
- MobileWan: https://arxiv.org/abs/2607.06173 · S2DiT: https://arxiv.org/abs/2601.12719 · CineMobile: https://arxiv.org/abs/2607.03803 · MobileVD: ICCV 2025 (openaccess.thecvf.com)
- TAESDXL: https://huggingface.co/madebyollin/taesdxl · RIFE: https://github.com/hzwer/ECCV2022-RIFE · https://huggingface.co/TensorForger/RIFE-safetensors
- PyTorch INT8 on x86: https://pytorch.org/blog/int8-quantization/ · torchao in diffusers: https://huggingface.co/docs/diffusers/quantization/torchao
- my-gpu (CPU techniques, measured on the same machine class): https://github.com/ayoub5550/my-gpu (docs/RESEARCH.md, docs/ROADMAP.md)
