# file: src/nckh/isic.py
"""Tải, giải nén và chuẩn hóa ISIC 2018 Task 1 (segmentation) và ISIC 2017 Task 3 (classification)."""
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

S3 = "https://isic-archive.s3.amazonaws.com/challenges"

# Tên zip → thư mục mà loader segmentation của PanDerm cần (datasets/dataset_seg.py, _get_paths_official).
ISIC2018_TARGETS = {
    "ISIC2018_Task1-2_Training_Input": ("Training_Data", (".jpg",)),
    "ISIC2018_Task1_Training_GroundTruth": ("Training_GroundTruth", (".png",)),
    "ISIC2018_Task1-2_Validation_Input": ("Validation_Data", (".jpg",)),
    "ISIC2018_Task1_Validation_GroundTruth": ("Validation_GroundTruth", (".png",)),
    "ISIC2018_Task1-2_Test_Input": ("Test_Data", (".jpg",)),
    "ISIC2018_Task1_Test_GroundTruth": ("Test_GroundTruth", (".png",)),
}
ISIC2018_URLS = [f"{S3}/2018/{stem}.zip" for stem in ISIC2018_TARGETS]

ISIC2017_IMAGE_DIRS = {"train": "ISIC-2017_Training_Data", "val": "ISIC-2017_Validation_Data", "test": "ISIC-2017_Test_v2_Data"}
ISIC2017_GT_CSV = {"train": "ISIC-2017_Training_Part3_GroundTruth.csv", "val": "ISIC-2017_Validation_Part3_GroundTruth.csv",
                   "test": "ISIC-2017_Test_v2_Part3_GroundTruth.csv"}
ISIC2017_URLS = [f"{S3}/2017/{d}.zip" for d in ISIC2017_IMAGE_DIRS.values()]
ISIC2017_GT_URLS = [f"{S3}/2017/{f}" for f in ISIC2017_GT_CSV.values()]

CLASS_NAMES = ("melanoma", "nevus", "seborrheic_keratosis")


def download_missing(target_dir: Path, urls: list[str]) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        dest = target_dir / url.rsplit("/", 1)[-1]
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            expected = int(resp.headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == expected:
            logger.info("Đã có %s, bỏ qua", dest.name)
        else:
            logger.info("Tải %s (%.2f GB)", dest.name, expected / 1e9)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
                downloaded = 0
                last_logged = 0
                chunk_size = 2 << 20  # 2 MB chunks
                while chunk := resp.read(chunk_size):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_logged >= 50 * (1 << 20) or downloaded >= expected:
                        logger.info("  %s: %.1f / %.1f MB (%.1f%%)", dest.name, downloaded / 1e6, expected / 1e6, 100 * downloaded / max(expected, 1))
                        last_logged = downloaded
            tmp.rename(dest)
        paths.append(dest)
    return paths


def extract_zip(zip_path: Path, target_dir: Path, suffixes: tuple[str, ...]) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        # Bỏ thư mục cha trong zip, bỏ LICENSE/ATTRIBUTION/metadata và ảnh superpixel của ISIC 2017.
        members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith(suffixes)
                   and "_superpixels" not in m.filename]
        target_dir.mkdir(parents=True, exist_ok=True)
        if sum(1 for p in target_dir.iterdir() if p.suffix.lower() in suffixes) >= len(members):
            return {"skipped": True, "n_files": len(members)}
        for member in members:
            (target_dir / Path(member.filename).name).write_bytes(zf.read(member))
    return {"skipped": False, "n_files": len(members)}


def prepare_isic2018(zips_dir: Path, data_root: Path) -> dict:
    report = {}
    for stem, (folder, suffixes) in ISIC2018_TARGETS.items():
        report[stem] = extract_zip(zips_dir / f"{stem}.zip", data_root / "ISIC2018" / folder, suffixes)
    return report


def count_masks(data_root: Path) -> dict:
    counts = {}
    for split, folder in {"train": "Training", "val": "Validation", "test": "Test"}.items():
        images = sorted((data_root / "ISIC2018" / f"{folder}_Data").glob("*.jpg"))
        mask_dir = data_root / "ISIC2018" / f"{folder}_GroundTruth"
        per_image = [len(list(mask_dir.glob(f"{p.stem}_*.png"))) for p in images]
        counts[split] = {
            "images": len(images),
            "masks": len(list(mask_dir.glob("*.png"))),
            "max_masks_per_image": max(per_image, default=0),
        }
    return counts


def prepare_isic2017_images(zips_dir: Path, data_root: Path) -> dict:
    return {d: extract_zip(zips_dir / f"{d}.zip", data_root / "ISIC2017" / d, (".jpg",)) for d in ISIC2017_IMAGE_DIRS.values()}


def make_cls_table(gt_csvs: dict[str, Path], image_dirs: dict[str, str]) -> pd.DataFrame:
    frames = []
    for split, csv_path in gt_csvs.items():
        gt = pd.read_csv(csv_path)
        both = gt[(gt["melanoma"] == 1) & (gt["seborrheic_keratosis"] == 1)]
        if len(both):
            raise ValueError(f"Ảnh vừa melanoma vừa SK trong {csv_path.name}: {both['image_id'].tolist()[:5]}")
        # Nhãn ISIC 2017 Task 3 chỉ có 2 cột nhị phân; nevus là ảnh không thuộc hai lớp còn lại.
        label = (gt["seborrheic_keratosis"] == 1) * 2 + ((gt["melanoma"] == 0) & (gt["seborrheic_keratosis"] == 0)) * 1
        frames.append(pd.DataFrame({
            "image": image_dirs[split] + "/" + gt["image_id"].astype(str) + ".jpg",
            "label": label.astype(int),
            "split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def make_trainphase(df: pd.DataFrame) -> pd.DataFrame:
    # run_class_finetuning.py tự đánh giá split "test" ở epoch cuối; khi train ta trỏ "test" vào bản sao của val
    # để tập test thật không bị mở trước khi khóa cấu hình.
    val_as_test = df[df["split"] == "val"].assign(split="test")
    return pd.concat([df[df["split"] != "test"], val_as_test], ignore_index=True)
