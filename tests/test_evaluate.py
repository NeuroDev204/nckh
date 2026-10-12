import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_inference import evaluate, select_samples, validate_samples  # noqa: E402


class Predictor:
    device = "cpu"

    def __init__(self, task: str) -> None:
        self.task = task

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return np.array([1., 0., 0.]) if self.task == "cls" else np.ones(rgb.shape[:2], dtype=bool)


def sample_images(root: Path, task: str) -> pd.DataFrame:
    rows = []
    for i in range(3):
        relative = f"ISIC2017/ISIC-2017_Validation_Data/ISIC_{i}.jpg" if task == "cls" else f"images/ISIC_{i}.jpg"
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(image)
        if task == "cls":
            rows.append({"image": relative.removeprefix("ISIC2017/"), "label": i, "split": "val"})
        else:
            mask = root / f"ISIC_{i}_segmentation.png"
            Image.fromarray(np.full((4, 4), 255, np.uint8)).save(mask)
            rows.append({"image_id": f"ISIC_{i}", "image_path": relative,
                         "mask_path": mask.name, "split": "val", "exclude_reason": ""})
    return pd.DataFrame(rows)


def test_sampling_is_seeded_and_preserves_classes() -> None:
    df = pd.DataFrame({"image": [f"{i}.jpg" for i in range(12)], "label": [0] * 8 + [1] * 3 + [2], "split": "val"})
    samples = select_samples(df, "cls", "val", 3, 2026)
    assert set(samples.label) == {0, 1, 2}
    pd.testing.assert_frame_equal(samples, select_samples(df, "cls", "val", 3, 2026))
    assert len(select_samples(df, "cls", "val", 0, 2026)) == 12
    with pytest.raises(ValueError):
        select_samples(df, "cls", "test", 3, 2026)
    with pytest.raises(ValueError):
        select_samples(df, "cls", "val", -1, 2026)


@pytest.mark.parametrize("task", ["cls", "seg"])
def test_evaluation_writes_metrics_and_predictions(tmp_path: Path, task: str) -> None:
    pytest.importorskip("torch")
    samples = select_samples(sample_images(tmp_path, task).iloc[::-1], task, "val", 0, 2026)
    out = tmp_path / "run"
    report = evaluate(Predictor(task), task, samples, tmp_path, out, 0, 1)
    assert report["n"] == 3
    saved = json.loads((out / "report.json").read_text())
    predictions = pd.read_csv(out / "predictions.csv")
    assert set(predictions.image_id) == {"ISIC_0", "ISIC_1", "ISIC_2"}
    if task == "cls":
        assert saved["metrics"]["accuracy"] == pytest.approx(1 / 3)
        assert predictions.set_index("image_id").loc["ISIC_2", "label"] == 2
    else:
        assert saved["metrics"]["dice"] == 1
        assert len(list((out / "overlays").glob("*.png"))) == 1
    with pytest.raises(FileExistsError):
        evaluate(Predictor(task), task, samples, tmp_path, out, 0, 0)


def test_missing_ground_truth_is_smoke_only(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    samples = select_samples(sample_images(tmp_path, "cls").drop(columns="label"), "cls", "val", 2, 2026)
    report = evaluate(Predictor("cls"), "cls", samples, tmp_path, tmp_path / "run", 0, 0)
    assert report["metrics"] is None
    assert report["warnings"]


def test_seg_exclusions_are_reported(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    manifest = sample_images(tmp_path, "seg")
    manifest.loc[0, "exclude_reason"] = "missing_mask"
    samples = select_samples(manifest, "seg", "val", 20, 2026)
    report = evaluate(Predictor("seg"), "seg", samples, tmp_path, tmp_path / "run", 0, 0)
    assert report["n"] == 2
    assert report["excluded_counts"] == {"missing_mask": 1}


def test_rejects_duplicates_traversal_and_disguised_trainphase(tmp_path: Path) -> None:
    df = sample_images(tmp_path, "cls")
    with pytest.raises(ValueError, match="trùng"):
        select_samples(pd.concat([df, df.iloc[:1]]), "cls", "val", 20, 2026)
    df.loc[0, "image"] = "../../outside.jpg"
    with pytest.raises(ValueError):
        validate_samples(select_samples(df, "cls", "val", 0, 2026), "cls", tmp_path)
    df = sample_images(tmp_path, "cls").assign(split="test")
    with pytest.raises(ValueError, match="test"):
        select_samples(df, "cls", "test", 20, 2026)


@pytest.mark.parametrize("invalid", ["missing_image", "bad_image", "wrong_size", "bad_mask", "null_mask"])
def test_rejects_bad_seg_data(tmp_path: Path, invalid: str) -> None:
    df = sample_images(tmp_path, "seg")
    if invalid == "missing_image":
        (tmp_path / df.loc[0, "image_path"]).unlink()
    elif invalid == "bad_image":
        (tmp_path / df.loc[0, "image_path"]).write_bytes(b"broken")
    elif invalid == "null_mask":
        df.loc[0, "mask_path"] = None
    else:
        mask = np.full((2, 2) if invalid == "wrong_size" else (4, 4), 128 if invalid == "bad_mask" else 255, np.uint8)
        Image.fromarray(mask).save(tmp_path / df.loc[0, "mask_path"])
    with pytest.raises((ValueError, OSError)):
        validate_samples(select_samples(df, "seg", "val", 0, 2026), "seg", tmp_path)


def test_rejects_invalid_cls_label(tmp_path: Path) -> None:
    df = sample_images(tmp_path, "cls")
    df.loc[0, "label"] = 4
    with pytest.raises(ValueError):
        select_samples(df, "cls", "val", 20, 2026)


def test_cls_smoke_uses_generic_images_without_isic2017_labels(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    manifest = sample_images(tmp_path, "seg").drop(columns=["mask_path", "exclude_reason"])
    samples = select_samples(manifest, "cls", "val", 2, 2026)
    result = evaluate(Predictor("cls"), "cls", samples, tmp_path, tmp_path / "run", 0, 0)
    assert result["metrics"] is None
    assert result["n"] == 2


def test_rejects_two_ids_resolving_to_same_image(tmp_path: Path) -> None:
    df = sample_images(tmp_path, "seg")
    alias = tmp_path / "images/alias.jpg"
    alias.symlink_to(tmp_path / df.loc[0, "image_path"])
    duplicate = df.iloc[:1].copy()
    duplicate["image_path"] = "images/alias.jpg"
    samples = select_samples(pd.concat([df, duplicate]), "seg", "val", 0, 2026)
    with pytest.raises(ValueError, match="trùng"):
        validate_samples(samples, "seg", tmp_path)
