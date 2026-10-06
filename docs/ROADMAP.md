# Neodragon-X roadmap

Ordered by expected gain ÷ effort. Each item says how to measure it. Rules: see `AGENTS.md` §0
(measure speed **and** quality, opt-in flags, log everything in `docs/DEVLOG.md`).

Status on 2026-10-06: text-to-video 50.5 s → 29.8 s, image-to-video 18.1 s, both at the same
CLIP-score, on a 17-core CPU (`results/cpu_benchmark.json`).

## Next (high value, no training)

1. **OpenVINO + NNCF static INT8 for the DiT** (my-gpu v0.4 showed OpenVINO ≈3× PyTorch on CNNs).
   PyTorch dynamic INT8 re-quantises activations on every call; static INT8 with fused kernels
   should cut the DiT's 12 s further. Hard part: the pyramid input length changes per stage, so
   export one IR per (stage, unit) shape (18 shapes) or use dynamic shapes. Measure: DiT seconds
   and CLIP-score with `benchmark_cpu.py`.
2. **`torch.compile` (inductor, CPU)** on DiT and video-VAE decoder. Blocked in the first session
   (venv had no setuptools / C++ toolchain). Install `setuptools` + `gcc`, try
   `mode="max-autotune-no-cudagraphs"`; check it works together with dynamic INT8.
3. **Speed up the SSD-1B first frame** (now ~10 s, the biggest remaining stage in T2V):
   - fewer steps: LCM at 2–3 steps instead of 4 (measure CLIP-score and look at the frames);
   - generate at 768×480 instead of 1024×640 (`--first_frame_height/width` already exist);
   - INT8 of the UNet is **rejected** (CLIP 31.4 → 11.5); try *static* INT8 or INT8 only in the
     mid-block, with the quality check.
4. **Video-VAE decoder** (~5–7 s): try INT8 on its convolutions via OpenVINO, or a tiny video
   decoder (TAEHV-style) if one exists for this latent space (16 channels, 8× spatial, causal 4×+1
   temporal).

## Quality

5. **Real-ESRGAN compact 2× upscaler** as an opt-in post-process (`--upscale 2`): 512×320 →
   1024×640. Measure ms/frame on CPU; with RIFE this gives 97 frames at 1024×640.
6. **Prompt enrichment**: rewrite short prompts into the cinematic style the model was trained on,
   with a small local LLM (Qwen2.5-0.5B/1.5B runs on CPU). Measure CLIP-score on 20 prompts.
7. **Long videos**: `--num_frames 97` works (see DEVLOG). Check drift beyond 97 frames; if quality
   decays, try FreeNoise-style noise rescheduling or re-anchoring on the first frame.

## Bigger steps

8. **Step caching for monolithic mode** (FlowCache / TeaCache, arXiv 2602.10825): monolithic mode
   runs 20/10 steps with CFG, so residual caching between steps can give 1.5–2×. Not useful for
   hybrid mode (1 step per stage).
9. **KV cache for history tokens**: ≤10 % for 49 frames (analysis in RESEARCH.md §3.2); revisit only
   for videos of 200+ frames.
10. **Phone runtime**: export the INT8 graphs to ONNX → Qualcomm AI Hub / QNN (the upstream target)
    or to ExecuTorch for Android CPUs.
11. **Verify the GPU path** on a free Kaggle/Colab T4 with `notebooks/neodragon_x_free_gpu.ipynb`
    and record the timings in `results/gpu_t4.json`.
12. **RIFE fast path**: zero-pad IFNet's 90/45-channel convs to 96/48 so oneDNN's correct kernels
    apply again (RIFE 2× 11.9 s → ~5 s); verify bit-for-bit closeness with `scripts/eval_rife.py`.
