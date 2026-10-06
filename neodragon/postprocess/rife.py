# SPDX-License-Identifier: BSD-3-Clause-Clear
"""2x / 4x frame interpolation for Neodragon videos with RIFE v4 (MIT, 12 MB of weights).

Neodragon generates 49 frames (about 2 s at 24 fps). RIFE inserts optical-flow-warped
in-between frames, turning that into 97 frames: either a smoother 48 fps clip of the same
length, or a 4 s clip at 24 fps (2x slow motion).
"""
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .ifnet import IFNet

RIFE_REPO_ID = "TensorForger/RIFE-safetensors"
RIFE_FILENAME = "flownet.safetensors"


def _bypass_onednn_for_odd_channels(model: torch.nn.Module, multiple: int = 16) -> int:
    """Work around wrong CPU convolutions in oneDNN 3.5.3 (bundled with torch 2.5.1).

    Measured on AMD Zen 4 (AVX-512): Conv2d layers whose channel counts are not multiples of 16
    return wrong values (45 input channels -> NaN, 90 -> max error 0.32 vs a float64 reference).
    IFNet uses c=90 / 45 / 4 channels, so it is hit; the Neodragon pipeline itself is not (all 152
    convs it runs were checked, max relative error 1e-6). Those layers run on PyTorch's native
    kernel instead (exact; RIFE 2x is ~2x slower than the broken fast path).
    """
    count = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and (
            m.in_channels % multiple or m.out_channels % multiple
        ):
            orig = m.forward

            def safe_forward(*a, _orig=orig, **k):
                with torch.backends.mkldnn.flags(enabled=False):
                    return _orig(*a, **k)

            m.forward = safe_forward
            count += 1
    return count


class RIFEInterpolator:
    def __init__(self, device: torch.device = torch.device("cpu"), cache_dir: Optional[str] = None):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(RIFE_REPO_ID, RIFE_FILENAME, cache_dir=cache_dir)
        self.model = IFNet()
        self.model.load_state_dict(load_file(path))
        self.model.to(device).eval()
        self.device = device
        if device.type == "cpu":
            n = _bypass_onednn_for_odd_channels(self.model)
            self.num_bypassed_convs = n

    @staticmethod
    def _to_tensor(frames: List[Image.Image]) -> torch.Tensor:
        arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0 for f in frames])
        return torch.from_numpy(arr).permute(0, 3, 1, 2)  # T, 3, H, W

    @staticmethod
    def _to_pil(t: torch.Tensor) -> List[Image.Image]:
        arr = (t.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    @torch.no_grad()
    def _midpoints(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """x: T,3,H,W in [0,1] -> T-1 middle frames."""
        _, _, h, w = x.shape
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32  # IFNet needs multiples of 32
        xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        mids = []
        for i in range(0, xp.shape[0] - 1, batch_size):
            a = xp[i : min(i + batch_size, xp.shape[0] - 1)]
            b = xp[i + 1 : i + 1 + a.shape[0]]
            mids.append(self.model(torch.cat([a, b], dim=1).to(self.device)).cpu())
        return torch.cat(mids)[:, :, :h, :w]

    def __call__(self, frames: List[Image.Image], factor: int = 2, batch_size: int = 8) -> List[Image.Image]:
        if factor not in (2, 4):
            raise ValueError("factor must be 2 or 4")
        x = self._to_tensor(frames)
        for _ in range(int(np.log2(factor))):
            mids = self._midpoints(x, batch_size)
            out = torch.empty((x.shape[0] * 2 - 1,) + x.shape[1:], dtype=x.dtype)
            out[0::2] = x
            out[1::2] = mids
            x = out
        return self._to_pil(x)


def interpolate_frames(frames, factor=2, device=torch.device("cpu"), cache_dir=None):
    return RIFEInterpolator(device=device, cache_dir=cache_dir)(frames, factor=factor)
