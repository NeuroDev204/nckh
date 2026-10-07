# file: tests/test_bench.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_inference import run_benchmark  # noqa: E402
from nckh.infer import ClsPredictor, SegPredictor  # noqa: E402


class _FakeSeg(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = torch.zeros(x.shape[0], 2, 224, 224)
        out[:, 1, 80:150, 80:150] = 1.0
        return out


class _FakeCls(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.tensor([[1.0, 0.0, -1.0]])


def _images(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"ISIC_{i:07d}.jpg"
        Image.fromarray(np.full((60 + i, 80, 3), 120, np.uint8)).save(p)
        paths.append(p)
    return paths


def test_seg_benchmark_writes_report_and_overlays(tmp_path: Path) -> None:
    report = run_benchmark(SegPredictor(_FakeSeg()), "seg", _images(tmp_path, 5), warmup=1,
                           out_dir=tmp_path / "out", save_overlays=2)
    assert report["n"] == 5 and report["ms_per_image_p95"] >= report["ms_per_image_p50"] > 0
    assert len(list((tmp_path / "out" / "overlays").glob("*.png"))) == 2
    saved = json.loads((tmp_path / "out" / "bench_seg.json").read_text())
    assert saved["checks"]["all_shapes_match"] is True


def test_cls_benchmark_checks_probabilities(tmp_path: Path) -> None:
    report = run_benchmark(ClsPredictor(_FakeCls()), "cls", _images(tmp_path, 3), warmup=0,
                           out_dir=tmp_path / "out", save_overlays=0)
    assert report["checks"]["max_abs_prob_sum_error"] < 1e-5
    assert (tmp_path / "out" / "cls_probs_smoke.csv").exists()


def test_benchmark_rejects_empty_image_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_benchmark(SegPredictor(_FakeSeg()), "seg", [], warmup=0, out_dir=tmp_path, save_overlays=0)