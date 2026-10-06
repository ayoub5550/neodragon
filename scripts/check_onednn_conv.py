# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check whether this machine's PyTorch CPU convolutions (oneDNN) are numerically correct.

Found on 2026-10-06: torch 2.5.1 (oneDNN 3.5.3) on AMD Zen 4 / AVX-512 returns wrong Conv2d
results when the *input* channel count is > 16 and not a multiple of 16 (17, 20, 24, 45, 90 ...:
error 0.15-0.65 or NaN), whatever the output channels; <= 16 inputs and multiples of 16 are exact.
The `sweep` block below maps that rule. Neodragon's own convs are all multiples of 16 (checked); RIFE's are not, so
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
    # sweep: which input-channel counts are wrong (output channels 4/45/64/90)
    sweep = []
    for cin in (1, 3, 4, 8, 11, 14, 16, 17, 20, 24, 30, 32, 40, 45, 48, 50, 60, 64, 80, 90, 96, 100, 112):
        errs = []
        for cout in (4, 45, 64, 90):
            conv = torch.nn.Conv2d(cin, cout, 3, 1, 1)
            x = torch.randn(2, cin, 40, 64)
            with torch.no_grad():
                ref = F.conv2d(x.double(), conv.weight.double(), conv.bias.double(), padding=1)
                e = (conv(x).double() - ref).abs().max().item()
            errs.append(float("inf") if e != e else e)
        worst = max(errs)
        sweep.append({"in_ch": cin, "worst_err": None if worst == float("inf") else round(worst, 7),
                      "ok": worst < 1e-3, "predicted_bad": cin > 16 and cin % 16 != 0})
    rule_holds = all(r["ok"] != r["predicted_bad"] for r in sweep)
    res = {"torch": torch.__version__, "sweep": sweep,
           "rule_in_ch_gt16_not_mult16_is_bad_holds": rule_holds, "machine": platform.processor() or platform.machine(),
           "onednn": torch.backends.mkldnn.is_available(), "rows": rows,
           "all_ok": all(r["ok"] for r in rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for r in rows + sweep:
        print(r)
    print("rule (in_ch > 16 and in_ch % 16 != 0 -> wrong) holds:", rule_holds)
    print("ALL OK" if res["all_ok"] else "WRONG RESULTS on this machine: see rows with ok=false")


if __name__ == "__main__":
    main()
