# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check whether this machine's PyTorch CPU convolutions (oneDNN) are numerically correct.

Found on 2026-10-06: torch 2.5.1 (oneDNN 3.5.3) on AMD Zen 4 / AVX-512 returns wrong Conv2d
results when channel counts are not multiples of 16 (45 -> NaN, 90 -> error 0.32), while 48/64/128
are exact. Neodragon's own convs are all multiples of 16 (checked); RIFE's are not, so
`neodragon/postprocess/rife.py` routes those layers to PyTorch's native kernel.

    python scripts/check_onednn_conv.py --out results/onednn_conv_check.json
"""
import argparse
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/onednn_conv_check.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    for cin in (3, 4, 16, 45, 48, 64, 90, 96, 128):
        for stride in (1, 2):
            conv = torch.nn.Conv2d(cin, 64 if cin != 45 else 90, 3, stride, 1)
            x = torch.randn(4, cin, 40, 64)
            with torch.no_grad():
                y = conv(x)
                ref = F.conv2d(x.double(), conv.weight.double(), conv.bias.double(), stride=stride, padding=1)
            err = (y.double() - ref).abs().max().item()
            rows.append({"in_ch": cin, "out_ch": conv.out_channels, "stride": stride,
                         "max_abs_err": None if err != err else round(err, 7),
                         "ok": bool(err == err and err < 1e-3)})
    res = {"torch": torch.__version__, "machine": platform.processor() or platform.machine(),
           "onednn": torch.backends.mkldnn.is_available(), "rows": rows,
           "all_ok": all(r["ok"] for r in rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for r in rows:
        print(r)
    print("ALL OK" if res["all_ok"] else "WRONG RESULTS on this machine: see rows with ok=false")


if __name__ == "__main__":
    main()
