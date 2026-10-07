# file: tests/test_inspect_checkpoint.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inspect_checkpoint import summarize_state_dict  # noqa: E402


def test_summarize_counts_prefix_and_patch_shape() -> None:
    sd = {
        "encoder.patch_embed.proj.weight": np.zeros((768, 3, 16, 16)),
        "encoder.blocks.0.norm1.weight": np.zeros(768),
        "decoder.x": np.zeros(1),
    }
    s = summarize_state_dict(sd)
    assert s["prefix_counts"] == {"encoder": 2, "decoder": 1}
    assert s["patch_embed_shape"] == [768, 3, 16, 16]
    assert s["verdict"] == "ViT-B (768) với prefix encoder. — dùng được với patch"


def test_summarize_unwraps_and_flags_large() -> None:
    s = summarize_state_dict({"model": {"encoder.patch_embed.proj.weight": np.zeros((1024, 3, 16, 16))}})
    assert s["wrapped_in"] == "model"
    assert s["verdict"].startswith("ViT-L")


def test_summarize_flags_missing_prefix() -> None:
    s = summarize_state_dict({"patch_embed.proj.weight": np.zeros((768, 3, 16, 16))})
    assert "không có prefix encoder." in s["verdict"]

def test_summarize_unwraps_mae_checkpoint_with_extra_keys() -> None:
    s = summarize_state_dict({"model": {"patch_embed.proj.weight": np.zeros((768, 3, 16, 16))}, "optimizer": {}, "epoch": 499})
    assert s["wrapped_in"] == "model"
    assert s["patch_embed_shape"] == [768, 3, 16, 16]
