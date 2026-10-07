"""
Soi cấu trúc key của checkpoint Panderm trước khi áp patch
Chạy: /content/venv_seg/bin/python scripts/inspect_checkpoint.py $ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth
"""

import json
import sys
from collections import Counter
from pathlib import Path

WRAPPER_KEYS = {"model", "state_dict", "module"}

def summarize_state_dict(sd: dict) -> dict:
    # MAE-style checkpoints also carry optimizer/epoch/args next to the weights
    wrapped_in = next((k for k in WRAPPER_KEYS if isinstance(sd.get(k), dict)), None)
    if wrapped_in:
        sd = sd[wrapped_in]
    prefix_counts = dict(Counter(k.split(".")[0] for k in sd).most_common())
    weight = sd.get("encoder.patch_embed.proj.weight", sd.get("patch_embed.proj.weight"))
    shape = list(weight.shape) if weight is not None else None
    has_encoder = "encoder.patch_embed.proj.weight" in sd
    if shape is None:
        verdict = "Không tìm thấy patch_embed.proj.weight - checkpoint lạ, dừng lại hỏi nhóm"
    elif shape[0] == 1024:
        verdict = "ViT-L (1024) — đây là PanDerm Large, không phải base"
    elif shape[0] == 768 and has_encoder:
        verdict = "ViT-B (768) với prefix encoder. — dùng được với patch"
    else:
        verdict = "ViT-B (768) nhưng không có prefix encoder. — sử dụng replace('encoder.','') trong patch"
    return {
        "wrapped_in": wrapped_in,
        "n_key": len(sd),
        "prefix_counts": prefix_counts,
        "first_keys": list(sd)[:10],
        "patch_embed_shape": shape,
        "rel_pos_keys": [k for k in sd if "rel_pos" in k][:3],
        "verdict": verdict,
    }

if __name__ == "__main__":
    import torch
    state = torch.load(Path(sys.argv[1]), map_location="cpu")
    print(json.dumps(summarize_state_dict(state), indent=2, ensure_ascii=False))