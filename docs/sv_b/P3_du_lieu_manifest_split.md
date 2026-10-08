# P3 — Tải, kiểm tra và khóa dữ liệu: manifest, split, unit test

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 3):** đưa ISIC 2018 Task 1 và ISIC 2017 Task 3 về đúng layout mà code PanDerm cần; tạo **manifest** (mỗi ảnh một dòng: đường dẫn, hash, kích thước, lý do loại); khóa split; viết các unit test chống rò rỉ. Chuẩn bị sẵn schema và hàm chia tập cho UQ để dùng ngay khi có quyền dữ liệu.

| Đầu vào | Đầu ra | Sinh ở |
|---|---|---|
| 6 zip ISIC 2018 Task 1 (S3 chính thức) | `~/nckh_data/ISIC2018/{Training,Validation,Test}_{Data,GroundTruth}/` | Máy local, một lần mỗi máy |
| 3 zip ảnh + 3 CSV nhãn ISIC 2017 Task 3 | `~/nckh_data/ISIC2017/ISIC-2017_*_Data/*.jpg` | Máy local, một lần mỗi máy |
| — | `~/nckh_root/data/manifests/isic2018_seg.csv` + `.sha256` + `.cross_split_duplicates.csv`, `mask_count.json` | Laptop (`nb_cpu`) |
| — | `~/nckh_root/data/manifests/isic2017_cls.csv`, `isic2017_cls_trainphase.csv` | Laptop (`nb_cpu`) |
| Metadata UQ (khi có quyền) | `<repo>/configs/uq_column_map.yaml` đã điền; manifest UQ có cột `split` (tạo ở P6 bằng `build_pairs.py`) | Máy |

## 2. Chạy ở đâu

| Việc | Chạy ở | Thời gian ước tính |
|---|---|---|
| Tạo `nckh/isic.py`, `nckh/manifest.py`, scripts, test; chạy pytest | Laptop (`nb_cpu`) | 20 phút |
| Tải + giải nén ISIC 2018 (~14 GB zip) vào `~/nckh_data` | Máy local (`nb_cpu`), một lần mỗi máy | tùy mạng |
| Tạo manifest ISIC 2018 (băm 3.694 ảnh) | Laptop (`nb_cpu`) | 3–6 phút |
| Tải + giải nén ISIC 2017 (~13 GB zip), tạo CSV nhãn | Máy local (`nb_cpu`), một lần mỗi máy | 10–25 phút |

Cần ~45 GB trống trong `~/nckh_data` lúc giải nén (zip bị xoá sau khi giải nén nhờ `--delete-zips`; sau đó còn ~30 GB). Máy GPU cũng cần ảnh: chạy lại hai lệnh tải ở mục 4.6, hoặc chép bằng `rsync` (`00` mục 5.2). Manifest và CSV nhãn chỉ tạo một lần ở laptop, chép sang cùng `~/nckh_root`.

## 3. Giải thích

### 3.1. Vì sao ảnh và manifest ở hai thư mục khác nhau

- **Ảnh** (~30 GB) nằm ở `~/nckh_data` (`NCKH_LOCAL_DATA`): tải từ S3 của ISIC, tải lại được bất cứ lúc nào. Script **idempotent** (đã giải nén đủ thì bỏ qua).
- **Manifest, CSV nhãn, `mask_count.json`** nằm ở `~/nckh_root/data/manifests` (`NCKH_ROOT`): nhỏ, là bằng chứng của protocol, cần giữ và sao lưu cùng `runs/`.
- **Manifest lưu đường dẫn tương đối** (`ISIC2018/Training_Data/ISIC_0000000.jpg`) so với `data_root`. Nhờ vậy cùng manifest dùng được trên laptop và máy GPU dù `~/nckh_data` được tải lại, và hash SHA-256 giúp phát hiện nếu file tải về khác lần trước.

### 3.2. Layout mà loader PanDerm cần

`segmentation/datasets/dataset_seg.py` (`_get_paths_official`) ghép đường dẫn bằng chuỗi:

```
{parent_path}ISIC2018/Training_Data/*.jpg      ↔  {parent_path}ISIC2018/Training_GroundTruth/{id}_segmentation.png
{parent_path}ISIC2018/Validation_Data/*.jpg    ↔  .../Validation_GroundTruth/{id}_segmentation.png
{parent_path}ISIC2018/Test_Data/*.jpg          ↔  .../Test_GroundTruth/{id}_segmentation.png
```

Mỗi zip ISIC có một thư mục cha cùng tên zip, kèm `LICENSE.txt` và `ATTRIBUTION.txt`. `nckh.isic.extract_zip` bỏ thư mục cha và chỉ giữ `.jpg`/`.png`. Kết quả là đúng layout trên với `--parent_path $HOME/nckh_data/` (P5a).

### 3.3. "Mỗi ảnh có 5 mask"? Không đúng

Đề cương (mục 6.3) ghi ISIC 2018 Task 1 có "5 mask tham chiếu mỗi ảnh". Khi đọc trực tiếp danh sách file trong các zip chính thức (05/10/2026): train có 2.594 ảnh và 2.594 mask, val 100/100, test 1.000/1.000, tức **1 mask cho mỗi ảnh**. Script `prepare_isic2018.py` ghi `mask_count.json` với `max_masks_per_image`. Khi bạn chạy và thấy giá trị `1`, gửi file này cho SV A để sửa đề cương. Phần "cách xử lý nhiều mask" trong kế hoạch 5.1 vì thế chỉ cần một câu: "mỗi ảnh có một mask tham chiếu".

### 3.4. Split

- **ISIC 2018 Task 1:** dùng split chính thức train 2.594 / val 100 / test 1.000 (loader upstream đọc đúng 3 thư mục này). Checkpoint được chọn trên 100 ảnh val. ISIC 2018 không công bố `lesion_id`, nên để chống rò rỉ ta **kiểm tra trùng hash xuyên split**: ảnh giống hệt nhau từng byte mà nằm ở hai split sẽ bị liệt kê trong `.cross_split_duplicates.csv`. Trùng **gần** (cùng tổn thương, khác ảnh) không phát hiện được bằng hash; ghi điều này vào giới hạn.
- **ISIC 2017 Task 3:** split chính thức train 2.000 / val 150 / test 600.
- **UQ:** tự chia **theo `participant_id`**, tỷ lệ 70/15/15, seed 2026 (khóa trong protocol P0). Tất cả ảnh của một người nằm trong đúng một split. **Chia trước, ghép cặp sau** (P6), để cặp t→t+1 không bao giờ nối hai split.

`assign_group_split` sắp xếp danh sách nhóm trước khi hoán vị bằng `numpy.random.default_rng(seed)`, nên cùng seed luôn cho cùng kết quả, bất kể thứ tự dòng trong CSV.

### 3.5. Nhãn ISIC 2017 Task 3

CSV ground truth có 2 cột nhị phân `melanoma`, `seborrheic_keratosis`. Nhãn đa lớp:

| melanoma | seborrheic_keratosis | label |
|---:|---:|---|
| 1 | 0 | 0 = melanoma |
| 0 | 0 | 1 = nevus |
| 0 | 1 | 2 = seborrheic keratosis |
| 1 | 1 | lỗi dữ liệu → `ValueError` |

`isic2017_cls_trainphase.csv` thay các dòng `test` bằng bản sao của `val`. Lý do nằm ở P5b: script fine-tune của PanDerm tự chạy test ở epoch cuối.

### 3.6. Schema UQ chuẩn

Pipeline P6 chỉ làm việc với 6 cột chuẩn:

| Cột chuẩn | Ý nghĩa | Kiểu sau khi đọc |
|---|---|---|
| `participant_id` | mã người tham gia (đơn vị chia tập, bootstrap) | chuỗi |
| `lesion_id` | mã tổn thương (đơn vị ghép cặp) | chuỗi |
| `image_id` | mã ảnh, duy nhất | chuỗi |
| `image_path` | đường dẫn ảnh, tương đối so với `data_root` | chuỗi |
| `captured_at` | thời điểm chụp | `datetime64[ns, UTC]`, đọc theo định dạng khai báo `captured_at_format` (mặc định ISO-8601); sai định dạng → `NaT` (bị loại có lý do ở P6) |
| `modality` | loại ảnh | chuỗi viết thường, bỏ khoảng trắng |

`load_uq_metadata` đổi tên cột theo `configs/uq_column_map.yaml`. Nếu thiếu cột gốc, hàm dừng và liệt kê đúng tên cột thiếu, không đoán.

## 4. Code

### 4.1. `src/nckh/isic.py`

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/isic.py`

```python
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
            # Tải vào file tạm rồi đổi tên: mạng hoặc máy bị ngắt giữa chừng sẽ không để lại file "đủ tên nhưng thiếu byte".
            logger.info("Tải %s (%.2f GB)", dest.name, expected / 1e9)
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
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
```

### 4.2. `src/nckh/manifest.py`

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/manifest.py`

```python
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
```

### 4.3. Scripts

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/prepare_isic2018.py`

```python
# file: scripts/prepare_isic2018.py
"""Tải + giải nén ISIC 2018 Task 1 về data_root theo đúng layout loader PanDerm, đếm số mask mỗi ảnh.

Ví dụ (.venv):  python scripts/prepare_isic2018.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --delete-zips
"""
import argparse
import json
import logging
from pathlib import Path

from nckh.isic import ISIC2018_URLS, count_masks, download_missing, prepare_isic2018


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--skip-download", action="store_true", help="zip đã có sẵn trong --zips-dir")
    ap.add_argument("--delete-zips", action="store_true", help="xoá zip sau khi giải nén để giải phóng đĩa")
    args = ap.parse_args(argv)

    if not args.skip_download:
        download_missing(args.zips_dir, ISIC2018_URLS)
    report = prepare_isic2018(args.zips_dir, args.data_root)
    counts = count_masks(args.data_root)
    out = args.data_root / "ISIC2018" / "mask_count.json"
    out.write_text(json.dumps({"extract": report, "counts": counts}, indent=2))
    print(json.dumps(counts, indent=2))
    if args.delete_zips:
        for url in ISIC2018_URLS:
            (args.zips_dir / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/prepare_isic2017_cls.py`

```python
# file: scripts/prepare_isic2017_cls.py
"""Tải ảnh + nhãn ISIC 2017 Task 3 và tạo CSV cho run_class_finetuning.py của PanDerm.

Sinh hai file trong --out-dir:
  isic2017_cls.csv            split thật (train/val/test) — chỉ dùng cho lần eval cuối
  isic2017_cls_trainphase.csv dòng test = bản sao val    — dùng khi fine-tune (xem P5b)
Ví dụ (.venv):
  python scripts/prepare_isic2017_cls.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --out-dir ~/nckh_root/data/manifests
"""
import argparse
import logging
from pathlib import Path

from nckh.isic import (
    CLASS_NAMES,
    ISIC2017_GT_CSV,
    ISIC2017_GT_URLS,
    ISIC2017_IMAGE_DIRS,
    ISIC2017_URLS,
    download_missing,
    make_cls_table,
    make_trainphase,
    prepare_isic2017_images,
)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="nơi ghi CSV (nên là NCKH_ROOT/data/manifests)")
    ap.add_argument("--labels-only", action="store_true", help="chỉ tải CSV nhãn, không tải ảnh")
    ap.add_argument("--delete-zips", action="store_true")
    args = ap.parse_args(argv)

    download_missing(args.zips_dir, ISIC2017_GT_URLS)
    if not args.labels_only:
        download_missing(args.zips_dir, ISIC2017_URLS)
        print(prepare_isic2017_images(args.zips_dir, args.data_root))

    df = make_cls_table({s: args.zips_dir / f for s, f in ISIC2017_GT_CSV.items()}, ISIC2017_IMAGE_DIRS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "isic2017_cls.csv", index=False)
    make_trainphase(df).to_csv(args.out_dir / "isic2017_cls_trainphase.csv", index=False)
    table = df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASS_NAMES)))
    print(table)
    if args.delete_zips:
        for url in ISIC2017_URLS:
            (args.zips_dir / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/build_manifest.py`

```python
# file: scripts/build_manifest.py
"""Tạo manifest ISIC 2018 (3 split chính thức), báo ảnh trùng xuyên split, ghi hash của manifest.

Ví dụ: python scripts/build_manifest.py --data-root ~/nckh_data --out ~/nckh_root/data/manifests/isic2018_seg.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from nckh.manifest import build_isic_seg_manifest, find_cross_split_duplicates
from nckh.runcard import sha256_file


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    df = pd.concat([build_isic_seg_manifest(args.data_root, s) for s in ("train", "val", "test")], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    dups = find_cross_split_duplicates(df)
    dups.to_csv(args.out.with_suffix(".cross_split_duplicates.csv"), index=False)
    digest = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(f"{digest}  {args.out.name}\n")

    print(df.groupby("split").size().rename("n_images"))
    print("Lý do loại:", df["exclude_reason"].replace("", "(giữ)").value_counts().to_dict())
    print("Ảnh trùng hash xuyên split:", len(dups))
    print("SHA-256 manifest:", digest)


if __name__ == "__main__":
    main()
```

### 4.4. `configs/uq_column_map.yaml`

Đây là template: bạn điền tên cột gốc sau P1. Giá trị `<ĐIỀN TÊN CỘT GỐC>` cố ý để trống, và P6 sẽ báo lỗi rõ ràng nếu bạn quên điền.

📁 **Tạo trên máy cá nhân:** `<repo>/configs/uq_column_map.yaml`

```yaml
# file: configs/uq_column_map.yaml
# Điền TÊN CỘT GỐC trong metadata UQ cho từng cột chuẩn (làm sau P1, khi đã đọc data dictionary).
participant_id: "<ĐIỀN TÊN CỘT GỐC>"
lesion_id: "<ĐIỀN TÊN CỘT GỐC>"
image_id: "<ĐIỀN TÊN CỘT GỐC>"
image_path: "<ĐIỀN TÊN CỘT GỐC>"
captured_at: "<ĐIỀN TÊN CỘT GỐC>"
modality: "<ĐIỀN TÊN CỘT GỐC>"
# Giá trị (sau khi viết thường, bỏ khoảng trắng) của cột modality được coi là ảnh dermoscopy.
modality_dermoscopy_value: "dermoscopy"
# Định dạng ngày giờ của cột captured_at: "ISO8601" (vd 2020-01-05, 2020-01-05T10:00:00+10:00) hoặc mẫu strftime
# như "%d/%m/%Y". Khai báo rõ để ngày/tháng không bị đọc nhầm; giá trị sai định dạng bị loại (missing_timestamp).
captured_at_format: "ISO8601"
```

### 4.5. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_manifest.py`

```python
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
```

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_prepare.py`

```python
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
```

### 4.6. Cell chạy

```python
# cell: nb_cpu
# ISIC 2018: tải + giải nén về ~/nckh_data (một lần mỗi máy), manifest + mask_count vào ~/nckh_root.
!mkdir -p {ROOT}/data/manifests
!cd {REPO} && {PY} scripts/prepare_isic2018.py --zips-dir {DATA}/zips --data-root {DATA} --delete-zips
!cp {DATA}/ISIC2018/mask_count.json {ROOT}/data/manifests/
!cd {REPO} && {PY} scripts/build_manifest.py --data-root {DATA} --out {ROOT}/data/manifests/isic2018_seg.csv
```

```python
# cell: nb_cpu
# ISIC 2017: ảnh về ~/nckh_data/ISIC2017 (một lần mỗi máy), CSV nhãn vào ~/nckh_root/data/manifests.
!cd {REPO} && {PY} scripts/prepare_isic2017_cls.py --zips-dir {DATA}/zips --data-root {DATA} --out-dir {ROOT}/data/manifests --delete-zips
!ls {DATA}/ISIC2017/*/ | head; ls {DATA}/ISIC2017/ISIC-2017_Training_Data | wc -l
```

Trên máy GPU chỉ cần ảnh (manifest và CSV nhãn chép từ laptop, không tạo lại). Hoặc `rsync` cả `~/nckh_data` (`00` mục 5.2), hoặc tải lại; CSV nhãn ISIC 2017 ghi ra thư mục tạm để không đè bản của laptop:

```bash
# terminal (máy GPU, .venv)
cd ~/Documents/nckh
.venv/bin/python scripts/prepare_isic2018.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --delete-zips
.venv/bin/python scripts/prepare_isic2017_cls.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --out-dir /tmp/isic2017_labels --delete-zips
```

Kỳ vọng:
- `ISIC-2017_Training_Data` có 2.000 file `.jpg` (ảnh superpixel đã bị lọc).
- Bảng phân bố lớp in ra: train 374 / 1.372 / 254, val 30 / 78 / 42, test 117 / 393 / 90 (melanoma / nevus / SK).

Kiểm tra nhanh manifest:

```python
# cell: nb_cpu
import pandas as pd
m = pd.read_csv(f'{ROOT}/data/manifests/isic2018_seg.csv')
print(m.groupby('split').size())                     # train 2594, val 100, test 1000
print(m['exclude_reason'].value_counts())             # kỳ vọng toàn bộ "" (NaN khi đọc lại)
print(m[['width', 'height']].describe().loc[['min', 'max']])
print(open(f'{ROOT}/data/manifests/isic2018_seg.csv.sha256').read())
```

Ghi SHA-256 của manifest vào protocol. Từ thời điểm này **không sửa** manifest test.

### 4.7. Khi có quyền UQ

1. Điền `<repo>/configs/uq_column_map.yaml` theo bảng P1, rồi commit + push ở laptop (00 mục 5.1).
2. Thử đọc:

```python
# cell: nb_cpu
import yaml
from nckh.manifest import load_uq_metadata, assign_group_split, assert_no_group_leakage
cmap = yaml.safe_load(open(f'{REPO}/configs/uq_column_map.yaml'))
dermo = cmap.pop('modality_dermoscopy_value')
uq = load_uq_metadata(Path(f'{ROOT}/data/uq/metadata.csv'), cmap)   # đổi tên file cho đúng gói thật
uq['split'] = assign_group_split(uq, 'participant_id', seed=2026)
assert_no_group_leakage(uq, 'participant_id')
print(uq.groupby('split')['participant_id'].nunique(), (uq['modality'] == dermo).mean())
```

Toàn bộ bước ghép cặp và manifest UQ cuối cùng được làm bằng `scripts/build_pairs.py` ở P6. Cell trên chỉ để kiểm tra map cột đã đúng chưa.

## 5. Test

```python
# cell: nb_cpu
!cd {REPO} && {PY} -m pytest -q tests/test_manifest.py tests/test_prepare.py
```

Kỳ vọng: `14 passed`.

Ánh xạ **8 unit test bắt buộc** của kế hoạch (Mục 3.4) sang test cụ thể:

| # | Yêu cầu của kế hoạch | Test |
|---|---|---|
| 1 | Mỗi ảnh trong manifest mở được | `test_exclude_reasons` (ảnh hỏng → `unreadable_image`); manifest thật: cột `readable` |
| 2 | Mask ISIC khớp ID và kích thước | `test_exclude_reasons` (`missing_mask`, `size_mismatch`), `test_manifest_relative_paths` |
| 3 | Không có participant giao giữa các split UQ | `test_group_split_disjoint_and_deterministic`, `test_leakage_detected` |
| 4 | Không có lesion giao giữa các split ISIC | ISIC không có lesion_id → `test_cross_split_duplicate_found` (trùng hash) |
| 5 | Mỗi cặp UQ cùng lesion, participant, timestamp tăng | P6: `tests/test_pairs.py` |
| 6 | Δt dương, hữu hạn, đúng đơn vị | P6: `tests/test_pairs.py` |
| 7 | Feature chỉ lấy từ thời điểm t | P6: `tests/test_features.py` |
| 8 | Tạo lại manifest với cùng seed cho kết quả giống hệt | `test_group_split_disjoint_and_deterministic` (+ so SHA-256 manifest giữa hai lần chạy) |

## 6. Benchmark / đánh giá

Số lượng **kỳ vọng** lấy theo dữ liệu chính thức. Nếu số thật khác, phải dừng lại tìm nguyên nhân.

| Bộ | Split | Ảnh kỳ vọng | Ảnh thật | Bị loại (lý do) | Trùng hash xuyên split |
|---|---|---:|---:|---|---:|
| ISIC 2018 Task 1 | train | 2.594 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| ISIC 2018 Task 1 | val | 100 | [điền sau khi chạy] | [điền sau khi chạy] | — |
| ISIC 2018 Task 1 | test | 1.000 | [điền sau khi chạy] | [điền sau khi chạy] | — |
| ISIC 2017 Task 3 | train / val / test | 2.000 / 150 / 600 | [điền sau khi chạy] | — | — |

| Thời gian | Giá trị |
|---|---|
| Tải + giải nén ISIC 2018 | [điền sau khi chạy] |
| Tải + giải nén ISIC 2017 | [điền sau khi chạy] |
| `build_manifest.py` | [điền sau khi chạy] |

## 7. Lỗi thường gặp (máy local)

| Triệu chứng | Cách xử lý |
|---|---|
| Mạng ngắt giữa lúc tải | Chạy lại cell. File `.part` dở dang sẽ bị ghi đè; zip đã đủ byte thì bỏ qua |
| `No space left on device` | Thêm `--delete-zips`; xoá `~/nckh_data/smoke` nếu không cần; kiểm tra `df -h ~ /tmp` |
| Số ảnh train ít hơn 2.594 | Giải nén bị ngắt: xoá thư mục `Training_Data` rồi chạy lại (`extract_zip` chỉ bỏ qua khi đã đủ file) |
| Manifest có `missing_mask` hàng loạt | Thiếu zip GroundTruth hoặc sai tên thư mục; kiểm tra `ls ~/nckh_data/ISIC2018` |
| Manifest trên máy GPU là bản cũ | Chép lại `~/nckh_root/data/manifests` từ laptop (`00` mục 5.2) |
| Máy GPU chạy code cũ | Push ở laptop, `git pull` ở máy GPU |
| `ValueError: Thiếu cột trong metadata.csv` | Tên cột trong YAML sai; đối chiếu danh sách "Cột hiện có" trong thông báo lỗi |

## 8. Checklist bàn giao cho SV A

- [ ] `~/nckh_root/data/manifests/isic2018_seg.csv` + `.sha256` + `.cross_split_duplicates.csv`; SHA-256 đã ghi vào protocol.
- [ ] `mask_count.json` gửi SV A (bằng chứng 1 mask/ảnh), đề cương được sửa.
- [ ] `~/nckh_root/data/manifests/isic2017_cls.csv` và `isic2017_cls_trainphase.csv`; bảng phân bố lớp gửi SV A.
- [ ] `pytest`: 14 passed cho phần P3.
- [ ] Bảng mục 6 đã điền số thật; mọi ảnh bị loại đều có lý do.
- [ ] (Khi có UQ) `configs/uq_column_map.yaml` đã điền, commit, và SV A xác nhận đúng nghĩa từng cột.
