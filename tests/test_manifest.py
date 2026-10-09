# file: tests/test_manifest.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nckh.manifest import (
    assert_no_group_leakage,
    assign_group_split,
    build_isic_seg_manifest,
    find_cross_split_duplicates,
    load_uq_metadata,
)


def _img(path: Path, size: tuple[int, int] = (32, 24), value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(path)


def _mask(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0]), np.uint8)).save(path)


@pytest.fixture
def isic_tree(tmp_path: Path) -> Path:
    layout = {"Training": [1, 2, 3, 4], "Validation": [5, 6], "Test": [7, 8]}
    for folder, ids in layout.items():
        for i in ids:
            name = f"ISIC_{i:07d}"
            _img(tmp_path / "ISIC2018" / f"{folder}_Data" / f"{name}.jpg", value=10 * i)
            if i != 3:
                _mask(tmp_path / "ISIC2018" / f"{folder}_GroundTruth" / f"{name}_segmentation.png",
                      size=(16, 16) if i == 4 else (32, 24))
    (tmp_path / "ISIC2018" / "Training_Data" / "ISIC_0000002.jpg").write_bytes(b"not an image")
    # Ảnh val 5 trùng byte với ảnh train 1 → phải bị phát hiện là trùng xuyên split.
    (tmp_path / "ISIC2018" / "Validation_Data" / "ISIC_0000005.jpg").write_bytes(
        (tmp_path / "ISIC2018" / "Training_Data" / "ISIC_0000001.jpg").read_bytes())
    return tmp_path


def _all_splits(root: Path) -> pd.DataFrame:
    return pd.concat([build_isic_seg_manifest(root, s) for s in ("train", "val", "test")], ignore_index=True)


def test_manifest_relative_paths(isic_tree: Path) -> None:
    df = _all_splits(isic_tree)
    assert len(df) == 8
    assert not df["image_path"].str.startswith("/").any()
    assert df.loc[df.image_id == "ISIC_0000001", "image_path"].item() == "ISIC2018/Training_Data/ISIC_0000001.jpg"


def test_exclude_reasons(isic_tree: Path) -> None:
    reasons = _all_splits(isic_tree).set_index("image_id")["exclude_reason"]
    assert reasons["ISIC_0000002"] == "unreadable_image"
    assert reasons["ISIC_0000003"] == "missing_mask"
    assert reasons["ISIC_0000004"] == "size_mismatch"
    assert (reasons.drop(["ISIC_0000002", "ISIC_0000003", "ISIC_0000004"]) == "").all()


def test_cross_split_duplicate_found(isic_tree: Path) -> None:
    dup = find_cross_split_duplicates(_all_splits(isic_tree))
    assert set(dup["image_id"]) == {"ISIC_0000001", "ISIC_0000005"}


def _uq_like(n_participants: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "participant_id": [f"P{p:03d}" for p in range(n_participants) for _ in range(3)],
        "image_id": [f"I{i}" for i in range(n_participants * 3)],
    })


def test_group_split_disjoint_and_deterministic() -> None:
    df = _uq_like()
    a = assign_group_split(df, "participant_id", seed=2026)
    b = assign_group_split(df, "participant_id", seed=2026)
    c = assign_group_split(df, "participant_id", seed=7)
    assert a.equals(b) and not a.equals(c)
    df["split"] = a
    assert_no_group_leakage(df, "participant_id")
    counts = df.groupby("split")["participant_id"].nunique().to_dict()
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_group_split_rejects_nan_group() -> None:
    df = _uq_like(5)
    df.loc[0, "participant_id"] = None
    with pytest.raises(ValueError):
        assign_group_split(df, "participant_id")


def test_leakage_detected() -> None:
    df = _uq_like(10)
    df["split"] = assign_group_split(df, "participant_id")
    first = df["participant_id"].iloc[0]
    other = "test" if df["split"].iloc[0] != "test" else "train"
    df.loc[df.index[1], "split"] = other
    with pytest.raises(AssertionError, match=first):
        assert_no_group_leakage(df, "participant_id")


def test_load_uq_metadata_maps_and_validates(tmp_path: Path) -> None:
    csv = tmp_path / "meta.csv"
    pd.DataFrame({
        "pid": [1, 1, 2], "lesion": ["L1", "L1", "L2"], "img": ["a", "b", "c"],
        "file": ["a.jpg", "b.jpg", "c.jpg"],
        "date": ["2020-01-05", "2020-01-05T10:00:00+10:00", "abc"],
        "type": ["Dermoscopy ", "dermoscopy", "Clinical"],
    }).to_csv(csv, index=False)
    mapping = {"participant_id": "pid", "lesion_id": "lesion", "image_id": "img",
               "image_path": "file", "captured_at": "date", "modality": "type"}
    df = load_uq_metadata(csv, mapping)
    assert list(df.columns[:6]) == ["participant_id", "lesion_id", "image_id", "image_path", "captured_at", "modality"]
    assert df["participant_id"].tolist() == ["1", "1", "2"]
    assert df["captured_at"].notna().tolist() == [True, True, False]
    assert df["modality"].tolist() == ["dermoscopy", "dermoscopy", "clinical"]
    with pytest.raises(ValueError, match="missing_col_date"):
        load_uq_metadata(csv, {**mapping, "captured_at": "missing_col_date"})


def test_load_uq_metadata_date_format(tmp_path: Path) -> None:
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"pid": [1, 1], "lesion": ["L", "L"], "img": ["a", "b"], "file": ["a.jpg", "b.jpg"],
                  "date": ["05/01/2020", "13/01/2020"], "type": ["dermoscopy"] * 2}).to_csv(csv, index=False)
    mapping = {"participant_id": "pid", "lesion_id": "lesion", "image_id": "img", "image_path": "file",
               "captured_at": "date", "modality": "type"}
    # Mặc định chỉ nhận ISO-8601: định dạng khác thành NaT (bị loại có lý do) thay vì bị đọc sai ngày/tháng.
    assert load_uq_metadata(csv, mapping)["captured_at"].isna().all()
    parsed = load_uq_metadata(csv, mapping, date_format="%d/%m/%Y")["captured_at"]
    assert parsed.dt.month.tolist() == [1, 1] and parsed.dt.day.tolist() == [5, 13]
