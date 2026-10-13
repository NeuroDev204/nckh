import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("pilot_classification", Path(__file__).resolve().parents[1] / "scripts/pilot_classification.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Linear(4, 8)
        self.head = torch.nn.Linear(8, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.patch_embed.proj(images)))


def test_pilot_restores_logits_and_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    report = pilot.run_pilot(TinyClassifier(), torch.randn(2, 4), torch.tensor([0, 2]), tmp_path)
    assert report["status"] == "PASS"
    assert report["max_abs_logits_error"] == 0
    assert report["restored_optimizer_step"] == 1
    checkpoint = torch.load(tmp_path / "checkpoint-2.pth", weights_only=True)
    assert checkpoint["step"] == 2
    assert all(int(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())


def test_pilot_rejects_frozen_backbone(tmp_path: Path) -> None:
    model = TinyClassifier()
    model.patch_embed.proj.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="Invalid gradient: patch_embed.proj.weight"):
        pilot.run_pilot(model, torch.randn(2, 4), torch.tensor([0, 2]), tmp_path)
