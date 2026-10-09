# file: tests/test_prepare.py
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nckh.isic import count_masks, extract_zip, make_cls_table, make_trainphase


def _gt_csv(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    pd.DataFrame(rows, columns=["image_id", "melanoma", "seborrheic_keratosis"]).to_csv(path, index=False)
    return path


def test_isic2017_labels(tmp_path: Path) -> None:
    gt = {"train": _gt_csv(tmp_path / "tr.csv", [("ISIC_1", 1.0, 0.0), ("ISIC_2", 0.0, 0.0), ("ISIC_3", 0.0, 1.0)])}
    df = make_cls_table(gt, {"train": "ISIC-2017_Training_Data"})
    assert df["label"].tolist() == [0, 1, 2]
    assert df["image"].iloc[0] == "ISIC-2017_Training_Data/ISIC_1.jpg"
    assert (df["split"] == "train").all()


def test_isic2017_rejects_double_positive(tmp_path: Path) -> None:
    gt = {"train": _gt_csv(tmp_path / "tr.csv", [("ISIC_1", 1.0, 1.0)])}
    with pytest.raises(ValueError, match="ISIC_1"):
        make_cls_table(gt, {"train": "X"})


def test_trainphase_has_no_real_test(tmp_path: Path) -> None:
    gt = {
        "train": _gt_csv(tmp_path / "a.csv", [("T1", 0.0, 0.0)]),
        "val": _gt_csv(tmp_path / "b.csv", [("V1", 1.0, 0.0), ("V2", 0.0, 1.0)]),
        "test": _gt_csv(tmp_path / "c.csv", [("X1", 0.0, 0.0)]),
    }
    df = make_cls_table(gt, {"train": "tr", "val": "va", "test": "te"})
    tp = make_trainphase(df)
    assert set(tp.loc[tp.split == "test", "image"]) == set(df.loc[df.split == "val", "image"])
    assert not tp["image"].str.startswith("te/").any()
    assert len(tp[tp.split == "train"]) == 1


def _zip_with_top_dir(path: Path, top: str, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/", "")
        for name in names:
            zf.writestr(f"{top}/{name}", b"x")
    return path


def test_extract_zip_flattens_filters_and_is_idempotent(tmp_path: Path) -> None:
    z = _zip_with_top_dir(tmp_path / "ISIC2018_Task1_Validation_GroundTruth.zip", "ISIC2018_Task1_Validation_GroundTruth",
                          ["ISIC_1_segmentation.png", "ISIC_2_segmentation.png", "LICENSE.txt", "ATTRIBUTION.txt"])
    target = tmp_path / "data" / "ISIC2018" / "Validation_GroundTruth"
    first = extract_zip(z, target, suffixes=(".png",))
    assert first == {"skipped": False, "n_files": 2}
    assert sorted(p.name for p in target.iterdir()) == ["ISIC_1_segmentation.png", "ISIC_2_segmentation.png"]
    assert extract_zip(z, target, suffixes=(".png",)) == {"skipped": True, "n_files": 2}


def test_extract_zip_excludes_superpixels(tmp_path: Path) -> None:
    z = _zip_with_top_dir(tmp_path / "ISIC-2017_Validation_Data.zip", "ISIC-2017_Validation_Data",
                          ["ISIC_1.jpg", "ISIC_1_superpixels.png", "ISIC-2017_Validation_Data_metadata.csv"])
    out = extract_zip(z, tmp_path / "out", suffixes=(".jpg",))
    assert out["n_files"] == 1


def test_count_masks(tmp_path: Path) -> None:
    root = tmp_path / "ISIC2018"
    for folder, ids in {"Training": [1, 2], "Validation": [3], "Test": [4]}.items():
        (root / f"{folder}_Data").mkdir(parents=True)
        (root / f"{folder}_GroundTruth").mkdir(parents=True)
        for i in ids:
            (root / f"{folder}_Data" / f"ISIC_{i}.jpg").write_bytes(b"x")
            (root / f"{folder}_GroundTruth" / f"ISIC_{i}_segmentation.png").write_bytes(b"x")
    counts = count_masks(tmp_path)
    assert counts["train"] == {"images": 2, "masks": 2, "max_masks_per_image": 1}
    assert counts["test"]["images"] == 1
