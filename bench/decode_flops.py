"""Per-stage FLOP / activation breakdown of the codec decoder for one latent length (15-decode-ane.md).

    uv run python bench/decode_flops.py [FRAMES]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.flop_counter import FlopCounterMode
from irodori_tts.codec import DACVAECodec
m = DACVAECodec.prepare_cpu().eval(); dec = m.decoder; dec.alpha = 0.0
z = torch.randn(1, 32, int(sys.argv[1]) if len(sys.argv) > 1 else 180)
with torch.no_grad():
    x = m.quantizer.out_proj(z); tot = 0
    for i, l in enumerate(dec.model):
        with FlopCounterMode(display=False) as fc: y = l(x)
        g = fc.get_total_flops()/1e9; tot += g
        snakes = sum(1 for mm in l.modules() if type(mm).__name__ == "Snake1d")
        print(f"stage {i}: {tuple(x.shape)} -> {tuple(y.shape)}  {g:7.1f} GFLOP  snake-layers {snakes}  elems-per-snake {y.numel()/1e6:.1f}M")
        x = y
    with FlopCounterMode(display=False) as fc: y = dec.wm_model.encoder_block.forward_no_conv(x)
    print(f"tail: {tuple(y.shape)} {fc.get_total_flops()/1e9:.1f} GFLOP; total {tot + fc.get_total_flops()/1e9:.0f} GFLOP")
