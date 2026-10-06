# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Video super-resolution for Neodragon output with Real-ESRGAN *compact* models (BSD-3-Clause).

Neodragon renders 512x320. `VideoUpscaler` runs a small SRVGGNetCompact network per frame
(x4 internally) and resizes to the requested factor (2 -> 1024x640, 4 -> 2048x1280).

Models (weights downloaded from the official Real-ESRGAN GitHub release, SHA-256 checked):
  * "general"   realesr-general-x4v3   (32 convs, 4.9 MB) - photos / real scenes (default)
  * "fast"      realesr-animevideov3   (16 convs, 2.5 MB) - ~2x faster, flatter textures
Architecture: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
(BSD-3-Clause, (c) 2021 Xintao Wang). All convs have 3 or 64 input channels, so the oneDNN bug
in `rife.onednn_conv_at_risk` does not apply (checked anyway in `scripts/bench_upscale.py`).
"""
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
MODELS = {
    "general": dict(file="realesr-general-x4v3.pth", num_conv=32,
                    sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
    "fast": dict(file="realesr-animevideov3.pth", num_conv=16,
                 sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d"),
}


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN compact network: plain conv+PReLU stack, pixel-shuffle, nearest-upsampled skip."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def _download(name: str, cache_dir: Optional[str]) -> Path:
    spec = MODELS[name]
    folder = Path(cache_dir or os.path.expanduser("~/.cache/neodragon")) / "realesrgan"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec["file"]
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_RELEASE + spec["file"], tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {spec['file']} (got {digest}); deleted, retry")
    return path


class VideoUpscaler:
    def __init__(self, model: str = "general", device: torch.device = torch.device("cpu"),
                 cache_dir: Optional[str] = None, channels_last: bool = True):
        spec = MODELS[model]
        state = torch.load(_download(model, cache_dir), map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        self.net = SRVGGNetCompact(num_conv=spec["num_conv"])
        self.net.load_state_dict(state)
        self.net.eval().to(device)
        if channels_last and device.type == "cpu":
            self.net.to(memory_format=torch.channels_last)
        self.channels_last = channels_last and device.type == "cpu"
        self.device = device

    @torch.no_grad()
    def __call__(self, frames: List[Image.Image], factor: int = 2, batch_size: int = 4) -> List[Image.Image]:
        if factor not in (2, 4):
            raise ValueError("factor must be 2 or 4")
        w, h = frames[0].size
        out = []
        for i in range(0, len(frames), batch_size):
            arr = np.stack([np.asarray(f.convert("RGB"), np.float32) / 255.0 for f in frames[i:i + batch_size]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = self.net(x)
            if factor != 4:
                y = F.interpolate(y, size=(h * factor, w * factor), mode="bicubic", align_corners=False, antialias=True)
            y = (y.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            out.extend(Image.fromarray(a) for a in y)
        return out
