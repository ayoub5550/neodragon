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

from .ifnet import IFBlock, IFNet

RIFE_REPO_ID = "TensorForger/RIFE-safetensors"
RIFE_FILENAME = "flownet.safetensors"


def onednn_conv_at_risk(m: torch.nn.Module, multiple: int = 16) -> bool:
    """True for convs that hit the oneDNN 3.5.3 bug measured on AMD Zen 4 (torch 2.5.1).

    Sweep in `scripts/check_onednn_conv.py` (results/onednn_conv_check.json): a Conv2d is wrong
    (error 0.15-0.65 or NaN vs float64) exactly when its *input* channels are > 16 and not a
    multiple of 16 (17, 20, 24, 30, 40, 45, 50, 60, 90, 100 ...). <= 16 inputs (1/3/4/11/14) and
    multiples of 16 are exact whatever the output channels. Transposed convs are treated the same
    way to be safe.
    """
    if not isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        return False
    return m.in_channels > multiple and m.in_channels % multiple != 0


def _bypass_onednn_for_odd_channels(model: torch.nn.Module, multiple: int = 16) -> int:
    """Run at-risk convs (see `onednn_conv_at_risk`) on PyTorch's native kernel (exact, slower).

    The stock IFNet (c=90/45) is hit; the Neodragon pipeline itself is not (all 152 convs it runs
    were checked, max relative error 1e-6). With `pad_ifnet_channels` no IFNet conv is at risk,
    so this guard is only a safety net. Returns the number of convs routed to the native kernel.
    """
    count = 0
    for m in model.modules():
        if onednn_conv_at_risk(m, multiple):
            orig = m.forward

            def safe_forward(*a, _orig=orig, **k):
                with torch.backends.mkldnn.flags(enabled=False):
                    return _orig(*a, **k)

            m.forward = safe_forward
            count += 1
    return count


def _copy_padded(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy `src` into the leading corner of `dst` (same rank); everything else is zero."""
    dst.zero_()
    dst[tuple(slice(0, n) for n in src.shape)].copy_(src)


@torch.no_grad()
def pad_ifnet_channels(model: IFNet, width: int = 96) -> int:
    """Rebuild every IFBlock with `width` hidden channels (90 -> 96, 45 -> 48), zero-padded.

    The added channels have zero weights and zero bias, so they stay exactly 0 through every
    conv, PReLU (PReLU(0) = 0), residual add and transposed conv: the network computes the same
    function. But 96/48 are multiples of 16, so oneDNN's fast *and correct* kernels apply again
    (see `_bypass_onednn_for_odd_channels`). Measured: max |diff| vs the native-kernel IFNet
    ~1e-6, RIFE 2x about 2x faster (results/rife_fastpath.json). Returns the number of blocks.
    """
    count = 0
    for name, old in list(model.named_children()):
        if not isinstance(old, IFBlock):
            continue
        c_old = old.convblock0[0][0].in_channels
        if c_old >= width:
            continue
        in_planes = old.conv0[0][0].in_channels
        new = IFBlock(in_planes, c=width)
        old_mods = dict(old.named_modules())
        for mname, m in new.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.PReLU)):
                src = old_mods[mname]
                for pname, param in m.named_parameters(recurse=False):
                    _copy_padded(param.data, getattr(src, pname).data)
        setattr(model, name, new)
        count += 1
    return count


class RIFEInterpolator:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        cache_dir: Optional[str] = None,
        fast_path: bool = True,
    ):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(RIFE_REPO_ID, RIFE_FILENAME, cache_dir=cache_dir)
        self.model = IFNet()
        self.model.load_state_dict(load_file(path))
        self.num_padded_blocks = pad_ifnet_channels(self.model) if fast_path else 0
        self.model.to(device).eval()
        self.device = device
        self.num_bypassed_convs = 0
        if device.type == "cpu":
            # After padding no conv is left with 90/45 channels, so this normally guards nothing;
            # it stays as a safety net (and is the whole fix when fast_path=False).
            self.num_bypassed_convs = _bypass_onednn_for_odd_channels(self.model)

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
