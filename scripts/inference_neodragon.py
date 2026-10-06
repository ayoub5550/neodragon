# fmt: off
#!/usr/bin/env python3

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import os
import sys
from concurrent import futures
from pathlib import Path
from typing import Tuple

import torch
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

# allow `python scripts/inference_neodragon.py` from a clone without `pip install .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neodragon import PIPELINE_MODES, NeodragonPipeline
from neodragon.runtime import (
    auto_device,
    auto_dtype,
    autocast_context,
    configure_cpu_threads,
    make_generator,
)
from neodragon.safety_checker import (
    check_generated_video_safety,
    load_stable_diffusion_safety_checker,
)
from neodragon.utils import (
    get_torch_dtype,
    init_distributed_mode,
    is_dist_avail_and_initialized,
)

FPS = 24
MAX_VIDEO_WRITER_WORKERS = 16
HF_MODEL_ID = "Qualcomm-AI-Research/Neodragon"


class _PromptsDataset(Dataset):
    def __init__(
        self,
        prompts_file: str,
        num_samples: int = 5,
        max_num_prompts: int = -1,  # used for debugging
        prompts: list = None,
    ) -> None:
        super().__init__()

        self.prompts_file = prompts_file
        self.num_samples = num_samples
        self.max_num_prompts = max_num_prompts

        if prompts:
            self.all_prompts = list(prompts)
        else:
            # Read the prompts file:
            with open(self.prompts_file, "r") as file:
                lines = file.readlines()
            self.all_prompts = [line.strip() for line in lines if line.strip()]

        # filtering based on max_num_prompts:
        if self.max_num_prompts > 0:
            self.all_prompts = self.all_prompts[: self.max_num_prompts]

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        prompt = self.all_prompts[idx]
        # hacky but allows to couple num_samples with prompt in dataloader
        return (prompt, self.num_samples)

    def __len__(self):
        return len(self.all_prompts)


def _build_data_loader(
    prompts_file: str,
    num_samples: int,
    max_num_prompts: int,
    world_size: int = 1,
    rank: int = 0,
    prompts: list = None,
    use_cuda: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = _PromptsDataset(
        prompts_file=prompts_file,
        num_samples=num_samples,
        max_num_prompts=max_num_prompts,
        prompts=prompts,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    prompts_dataloader = DataLoader(
        dataset,
        batch_size=1,  # one prompt per GPU
        # Prompts are tiny strings: worker processes only cost start-up time (and pinned
        # memory needs CUDA), so they are only used on GPU runs.
        num_workers=6 if use_cuda else 0,
        pin_memory=use_cuda,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        prefetch_factor=2 if use_cuda else None,
    )

    return prompts_dataloader


def get_args():
    parser = argparse.ArgumentParser("Neodragon Video Generation Script", add_help=True)

    parser.add_argument(
        "--model_dtype",
        default="auto",
        type=str,
        help="auto | bf16 | fp16 | fp32. auto = bf16 on CUDA, fp16 on MPS, and on CPU bf16 only "
        "if the CPU has native bf16 (AVX512_BF16/AMX), otherwise fp32",
    )
    # ---------------- Neodragon-X options ---------------- #
    parser.add_argument("--device", default="auto", type=str,
                        help="auto | cuda | mps | cpu (auto picks the best available)")
    parser.add_argument("--prompt", type=str, action="append", default=None,
                        help="A prompt to generate (repeatable). Overrides --prompts_file")
    parser.add_argument("--image", type=str, default=None,
                        help="Image-to-video: animate this picture (used as the first frame; "
                        "skips the SSD-1B first-frame generator)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible videos (sample i uses seed + i)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default: all CPUs this process may use, which is "
                        "often more than `nproc` reports inside containers)")
    parser.add_argument("--tiny_first_frame_vae", action="store_true",
                        help="Decode the SSD-1B first frame with TAESDXL (7x faster decode)")
    parser.add_argument("--int8", action="store_true",
                        help="CPU only: dynamic INT8 for the Linear layers of the DiT and text "
                        "encoder (adaLN/IO kept fp32). ~1.8x faster DiT, CLIP-score unchanged")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA .safetensors from distill/train_stage_a.py, merged into the DiT "
                        "before generation (trained on the multistep DiT: use with --mode monolithic)")
    parser.add_argument("--lora_strength", type=float, default=1.0,
                        help="Scale of the --lora delta (0 = off, 1 = as trained)")
    parser.add_argument("--fast", action="store_true",
                        help="Shortcut: --tiny_first_frame_vae --first_frame_steps 2, plus --int8 on CPU")
    parser.add_argument("--first_frame_steps", type=int, default=None, choices=[2, 3, 4],
                        help="LCM steps for the SSD-1B first frame (upstream 4). 2 steps: 1.9x "
                        "faster first frame, CLIP 33.4 vs 33.6 (results/first_frame_steps.json)")
    parser.add_argument("--interpolate", type=int, default=1, choices=[1, 2, 4],
                        help="RIFE frame interpolation factor: 2 turns 49 frames into 97")
    parser.add_argument("--slow_motion", action="store_true",
                        help="With --interpolate: keep the fps (longer, slow-motion clip) instead "
                        "of multiplying it (same length, smoother)")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4],
                        help="Real-ESRGAN compact super-resolution after generation/RIFE: "
                        "2 -> 1024x640, 4 -> 2048x1280 (results/upscale_eval.json)")
    parser.add_argument("--upscale_model", default="general", choices=["general", "fast"],
                        help="general = realesr-general-x4v3 (photos); fast = realesr-animevideov3")
    parser.add_argument("--first_frame_height", type=int, default=None,
                        help="SSD-1B first-frame height (default 640)")
    parser.add_argument("--first_frame_width", type=int, default=None,
                        help="SSD-1B first-frame width (default 1024)")
    # fmt: off
    parser.add_argument(
        "--mode",
        default="hybrid",
        type=str,
        choices=PIPELINE_MODES,
        help="The Neodragon Pipeline Mode | Choices: [" + ", ".join(PIPELINE_MODES) + "]",
    )
    # fmt: on
    parser.add_argument(
        "--local_cache_folder",
        type=str,
        default="./models",
        help="Path to a local cache folder for storing downloaded models from HuggingFace.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./prompts/showcase_prompts.txt",
        help="Path to a `.txt` file with prompts (one per line)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The number of videos to be generated per prompt",
    )
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=-1,
        help="The number of maximum prompts to use (Helpful for Debugging)",
    )
    parser.add_argument(
        "--use_cinematic_prompt_modifier",
        action="store_true",
        help="Whether to use the cinematic prompt modifier or not",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the generation time or not",
    )

    parser.add_argument("--height", type=int, default=320, help="The video height")
    parser.add_argument("--width", type=int, default=512, help="The video width")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="The number of frames in the generated video",
    )
    parser.add_argument(  # Don't touch this!
        "--fps", type=int, default=FPS, help="fps of the exported videos"
    )

    parser.add_argument(
        "--output_video_folder",
        default="./output/neodragon_videos",
        type=str,
        help="The path where the generated videos should be saved",
    )

    # Safety Checker arguments:
    parser.add_argument(
        "--safety_checker_model_id",
        type=str,
        default="CompVis/stable-diffusion-safety-checker",
        help="Hugging Face model ID for the Stable Diffusion safety checker.",
    )

    parser.add_argument(
        "--safety_check_num_frames",
        type=int,
        default=1,
        help="Number of generated video frames to sample for output safety checking.",
    )

    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Whether to disable the Stable Diffusion safety checker.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    init_distributed_mode(args)

    try:
        world_size = args.world_size
        rank = args.rank
        local_gpu = args.gpu
    except AttributeError:
        world_size = 1
        rank = 0
        local_gpu = 0

    device = auto_device(args.device)
    if device.type == "cuda":
        device = torch.device(f"cuda:{local_gpu}")
    if device.type == "cpu":
        n_threads = configure_cpu_threads(args.threads)
        print(f"[neodragon-x] CPU mode with {n_threads} threads")
    torch_dtype = auto_dtype(device, args.model_dtype)
    print(f"[neodragon-x] device={device} dtype={torch_dtype}")

    if args.fast:
        args.tiny_first_frame_vae = True
        args.first_frame_steps = args.first_frame_steps or 2
        args.int8 = args.int8 or device.type == "cpu"
    if args.int8 and device.type != "cpu":
        print("[neodragon-x] --int8 is a CPU path; ignored on", device)
        args.int8 = False

    # Load the pipeline
    neodragon_pipeline = NeodragonPipeline.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch_dtype,
        mode=args.mode,
        cache_dir=args.local_cache_folder,
    ).to(device)
    if args.lora:
        from neodragon.lora import load_lora

        if args.mode != "monolithic":
            print("[neodragon-x] warning: --lora was trained on the multistep DiT; "
                  "in hybrid mode it is applied to the 1-1-1 DiT, which is untested")
        load_lora(neodragon_pipeline.dit, args.lora, merge=True, strength=args.lora_strength)
        print(f"[neodragon-x] merged LoRA {args.lora} (strength {args.lora_strength})")
    if args.tiny_first_frame_vae:
        neodragon_pipeline.enable_tiny_first_frame_vae(cache_dir=args.local_cache_folder)
    if args.int8:
        neodragon_pipeline.quantize_int8()
    print(f"Loaded Neodragon Pipeline in {args.mode} mode.")

    interpolator = None
    if args.interpolate > 1:
        from neodragon.postprocess import RIFEInterpolator

        interpolator = RIFEInterpolator(device=device, cache_dir=args.local_cache_folder)
    upscaler = None
    if args.upscale > 1:
        from neodragon.postprocess import VideoUpscaler

        upscaler = VideoUpscaler(args.upscale_model, device=device, cache_dir=args.local_cache_folder)
    out_fps = args.fps if (args.interpolate == 1 or args.slow_motion) else args.fps * args.interpolate

    init_image = None
    if args.image is not None:
        from PIL import Image as _PILImage

        init_image = _PILImage.open(args.image).convert("RGB")
        print(f"[neodragon-x] image-to-video from {args.image}")

    # Load the safety checker:
    safety_checker, safety_feature_extractor = None, None
    if not args.disable_safety_checker:
        safety_checker, safety_feature_extractor = load_stable_diffusion_safety_checker(
            model_id=args.safety_checker_model_id,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=args.local_cache_folder,
        )
        print(f"Loaded Stable Diffusion Safety Checker.")

    # Build the distributed data loader:
    dataloader = _build_data_loader(
        prompts_file=args.prompts_file,
        num_samples=args.num_samples,
        max_num_prompts=args.max_num_prompts,
        world_size=world_size,
        rank=rank,
        prompts=args.prompt,
        use_cuda=device.type == "cuda",
    )

    if is_dist_avail_and_initialized():
        # synchronize all processes / GPUs before starting inference
        torch.distributed.barrier()

    prompt_modifier = (
        ", cinematic, realistic textures, high detail, natural colours"
        if args.use_cinematic_prompt_modifier
        else ", hyper quality, Ultra HD, 8K"
    )

    # Video generation loop:
    task_queue = []
    with futures.ThreadPoolExecutor(max_workers=MAX_VIDEO_WRITER_WORKERS) as executor:
        for datum in tqdm(dataloader):
            prompt, num_samples = datum[0][0], datum[1].item()

            with torch.no_grad(), autocast_context(device, torch_dtype):
                for sample_id in range(num_samples):
                    safe_name = "".join(c if c.isalnum() or c in " -_," else "_" for c in prompt)[:150]
                    output_video_path = Path(args.output_video_folder) / f"{safe_name}-{sample_id}.mp4"

                    if output_video_path.exists() and output_video_path.stat().st_size > 0:
                        continue

                    # generate the video:
                    generated_video = neodragon_pipeline(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        prompt_modifier=prompt_modifier,
                        profile=args.profile,
                        image=init_image,
                        generator=make_generator(
                            None if args.seed is None else args.seed + sample_id
                        ),
                        **(
                            {"first_frame_steps": args.first_frame_steps}
                            if args.first_frame_steps
                            else {}
                        ),
                        **(
                            {
                                "first_frame_height": args.first_frame_height,
                                "first_frame_width": args.first_frame_width,
                            }
                            if args.first_frame_height and args.first_frame_width
                            else {}
                        ),
                    )

                    # Safety check the generated video:
                    if not args.disable_safety_checker:
                        is_unsafe, frame_safety_flags = check_generated_video_safety(
                            generated_video=generated_video,
                            safety_checker=safety_checker,
                            feature_extractor=safety_feature_extractor,
                            device=device,
                            torch_dtype=torch_dtype,
                            num_frames=args.safety_check_num_frames,
                        )

                        if is_unsafe:
                            message = (
                                f"Generated video rejected by StableDiffusionSafetyChecker. "
                                f"prompt={prompt!r}, frame_safety_flags={frame_safety_flags}"
                            )
                            raise RuntimeError(message)

                    if interpolator is not None:
                        generated_video = interpolator(generated_video, factor=args.interpolate)
                    if upscaler is not None:
                        generated_video = upscaler(generated_video, factor=args.upscale)

                    # If video passes safety check submit it for exporting
                    task_queue.append(
                        executor.submit(
                            export_to_video,
                            generated_video,
                            output_video_path,
                            fps=out_fps,
                        )
                    )

        for future in futures.as_completed(task_queue):
            # Consume completions while the executor is still in scope
            try:
                future.result()  # raises if export_to_video failed
            except Exception as e:
                # Log and (optionally) decide to abort or continue
                print(f"[export error] {e}")

    # again synchronize all processes / GPUs before exiting
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    print(f"Generation complete! check path: {args.output_video_folder}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    parsed_args = get_args()
    if not Path(parsed_args.output_video_folder).exists():
        Path(parsed_args.output_video_folder).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
