# file: src/nckh/manifest.py
"""Manifest dữ liệu, kiểm tra ảnh, chia tập theo nhóm và kiểm tra rò rỉ.

Mọi đường dẫn trong manifest là tương đối so với data_root: ảnh nằm ở NCKH_LOCAL_DATA (tải lại được, có thể
ở máy khác), manifest nằm ở NCKH_ROOT, nên manifest không được phụ thuộc vào vị trí tuyệt đối của ảnh.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nckh.runcard import sha256_file

UQ_COLUMNS: tuple[str, ...] = ("participant_id", "lesion_id", "image_id", "image_path", "captured_at", "modality")
ISIC2018_FOLDERS = {"train": "Training", "val": "Validation", "test": "Test"}


def check_image(path: Path) -> dict:
    try:
        with Image.open(path) as img:
            img.load()  # open() chỉ đọc header; load() mới phát hiện file bị cắt cụt/hỏng
            return {"readable": True, "width": img.width, "height": img.height, "mode": img.mode, "error": None}
    except (OSError, ValueError) as exc:
        return {"readable": False, "width": None, "height": None, "mode": None, "error": f"{type(exc).__name__}: {exc}"}


def build_isic_seg_manifest(data_root: Path, split: str) -> pd.DataFrame:
    folder = ISIC2018_FOLDERS[split]
    image_dir = data_root / "ISIC2018" / f"{folder}_Data"
    mask_dir = data_root / "ISIC2018" / f"{folder}_GroundTruth"
    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem
        mask_path = mask_dir / f"{image_id}_segmentation.png"  # ghép theo ID, không theo thứ tự tên file
        info = check_image(image_path)
        mask_info = check_image(mask_path) if mask_path.exists() else None
        if not info["readable"]:
            reason = "unreadable_image"
        elif mask_info is None or not mask_info["readable"]:
            reason = "missing_mask"
        elif (mask_info["width"], mask_info["height"]) != (info["width"], info["height"]):
            reason = "size_mismatch"
        else:
            reason = ""
        rows.append({
            "image_id": image_id,
            "image_path": image_path.relative_to(data_root).as_posix(),
            "mask_path": mask_path.relative_to(data_root).as_posix(),
            "split": split,
            "sha256": sha256_file(image_path),
            "width": info["width"],
            "height": info["height"],
            "mask_width": mask_info["width"] if mask_info else None,
            "mask_height": mask_info["height"] if mask_info else None,
            "readable": info["readable"],
            "exclude_reason": reason,
        })
    return pd.DataFrame(rows)


def find_cross_split_duplicates(df: pd.DataFrame, hash_col: str = "sha256", split_col: str = "split") -> pd.DataFrame:
    n_splits = df.groupby(hash_col)[split_col].transform("nunique")
    return df[n_splits > 1].sort_values([hash_col, split_col])


def load_uq_metadata(csv_path: Path, column_map: dict[str, str], date_format: str = "ISO8601") -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    missing = [src for src in column_map.values() if src not in raw.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong {csv_path.name}: {missing}. Cột hiện có: {list(raw.columns)}")
    df = raw.rename(columns={src: std for std, src in column_map.items()})
    for col in ("participant_id", "lesion_id", "image_id", "image_path"):
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    # Định dạng khai báo tường minh (mặc định ISO-8601): "05/01/2020" không bị đoán thành 1/5 hay 5/1 tùy dòng.
    # utc=True gộp được giá trị chỉ có ngày và giá trị có múi giờ; giá trị sai định dạng thành NaT → bị loại có lý do.
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce", format=date_format)
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()
    return df[list(UQ_COLUMNS) + [c for c in df.columns if c not in UQ_COLUMNS]]


def assign_group_split(df: pd.DataFrame, group_col: str, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
                       seed: int = 2026) -> pd.Series:
    if df[group_col].isna().any():
        raise ValueError(f"Có {int(df[group_col].isna().sum())} dòng thiếu {group_col}; không chia tập an toàn được")
    groups = np.array(sorted(df[group_col].astype(str).unique()))
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_val = round(len(groups) * fractions[1])
    n_test = round(len(groups) * fractions[2])
    label = {g: "test" for g in shuffled[:n_test]}
    label.update({g: "val" for g in shuffled[n_test:n_test + n_val]})
    label.update({g: "train" for g in shuffled[n_test + n_val:]})
    return df[group_col].astype(str).map(label).rename("split")


def assert_no_group_leakage(df: pd.DataFrame, group_col: str, split_col: str = "split") -> None:
    per_group = df.groupby(group_col)[split_col].nunique()
    leaked = per_group[per_group > 1].index.tolist()
    if leaked:
        raise AssertionError(f"{len(leaked)} {group_col} nằm ở nhiều split, ví dụ: {leaked[:10]}")
