# file: tests/test_infer.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nckh.infer import (  # noqa: E402
    ClsPredictor,
    SegPredictor,
    cls_preprocess,
    largest_component,
    map_pretrained_cls_keys,
    seg_logits_to_mask,
    seg_preprocess,
)


def test_seg_preprocess_shape_and_range() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    x = seg_preprocess(rgb)
    assert tuple(x.shape) == (1, 3, 224, 224)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_cls_preprocess_center_crop_and_norm() -> None:
    rgb = np.full((300, 400, 3), 128, dtype=np.uint8)
    x = cls_preprocess(rgb)
    assert tuple(x.shape) == (1, 3, 224, 224)
    # Chuẩn hóa đúng như upstream: std kênh R là 0.228 (không phải 0.229).
    assert x[0, 0, 0, 0].item() == pytest.approx((128 / 255 - 0.485) / 0.228, abs=1e-4)


def test_largest_component_keeps_biggest_and_fills_holes() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[2:12, 2:12] = True
    m[6, 6] = False          # lỗ bên trong
    m[15:17, 15:17] = True   # vùng nhỏ bị bỏ
    out = largest_component(m)
    assert out[6, 6] and not out[15, 15]
    assert out.sum() == 100


def test_largest_component_empty_stays_empty() -> None:
    # Upstream biến mask rỗng thành toàn ảnh; ở đây phải giữ rỗng.
    assert largest_component(np.zeros((8, 8), dtype=bool)).sum() == 0


def test_seg_logits_to_mask_resizes_to_original() -> None:
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, 0:2, 0:2] = 5.0   # vùng lớn 4 px
    logits[0, 1, 3, 3] = 5.0       # vùng nhỏ 1 px
    mask = seg_logits_to_mask(logits, (8, 8))
    assert mask.shape == (8, 8) and mask.dtype == bool
    assert mask[:4, :4].all() and not mask[6:, 6:].any()


class _FakeSeg(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = torch.zeros(x.shape[0], 2, 224, 224)
        out[:, 1, 56:168, 56:168] = 1.0
        return out


class _FakeCls(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.tensor([[2.0, 0.5, -1.0]]).repeat(x.shape[0], 1)


def test_seg_predictor_with_fake_model() -> None:
    mask = SegPredictor(_FakeSeg()).predict(np.zeros((100, 200, 3), dtype=np.uint8))
    assert mask.shape == (100, 200)
    assert mask[50, 100] and not mask[0, 0]
    assert mask.mean() == pytest.approx(0.25, abs=0.02)


def test_cls_predictor_probs_sum_one() -> None:
    probs = ClsPredictor(_FakeCls()).predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert probs.argmax() == 0


def test_map_pretrained_cls_keys() -> None:
    t = torch.zeros(1)
    raw = {
        "encoder.blocks.0.mlp.fc1.weight": t,
        "encoder.norm.weight": t,
        "encoder.rel_pos_bias.relative_position_bias_table": torch.ones(3, 2),
        "encoder.blocks.0.attn.relative_position_index": t,
        "decoder.x": t,
        "teacher.y": t,
    }
    out = map_pretrained_cls_keys(raw, num_layers=2)
    assert "blocks.0.mlp.fc1.weight" in out and "fc_norm.weight" in out
    assert "blocks.1.attn.relative_position_bias_table" in out
    assert not any(k.startswith(("decoder.", "teacher.", "encoder.")) for k in out)
    assert not any("relative_position_index" in k for k in out)