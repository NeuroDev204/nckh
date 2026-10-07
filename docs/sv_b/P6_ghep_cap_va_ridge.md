# P6 — Ghép cặp ảnh dọc UQ và dự báo Δa bằng Ridge

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 6):** từ ảnh dermoscopy tại lần khám t (mask PanDerm, hình dạng mask, xác suất 3 nhóm tham khảo) và khoảng thời gian Δt, dự báo **thay đổi tỷ lệ diện tích mask** Δa ở lần chụp kế tiếp. So sánh với baseline "không thay đổi" (Δa = 0) trên participant test.

| Đầu vào | Đầu ra (`runs/<run_id>/`) | Sinh ở |
|---|---|---|
| Metadata + ảnh UQ (chỉ khi qua cổng Go/No-Go), `<repo>/configs/uq_column_map.yaml` | `uq_manifest.csv` (có `split`, `readable`), `pairs.csv`, `pairs_excluded.csv`, `flow.json` | Máy: `~/nckh_drive/runs/<run_id>/` → đẩy lên Drive bằng rclone |
| Checkpoint fine-tune seg (P5a), cls (P5b) | `masks/*.png`, `seg_features.csv`, `cls_probs.csv`, `*_failed.csv` | Colab `NB_seg`/`NB_cls` ghi vào `MyDrive/NCKH_PanDerm/runs/<run_id>/` → kéo về máy |
| — | `features.csv`, `alpha_selection.csv`, `ridge.joblib`, `predictions_val.csv`, `ridge_summary.json`, `run_card.json`; sau khi khóa: `predictions_test.csv` | Máy: `~/nckh_drive/runs/<run_id>/ridge/`, `ridge_test/` |

## 2. Chạy ở đâu

| Việc | Chạy ở | Ghi chú |
|---|---|---|
| Viết module, test, chạy pipeline trên dữ liệu **giả** | Máy (`NB_cpu`) | Làm ngay, không cần UQ; `infer_images.py --fake` không cần GPU |
| `build_pairs.py` trên UQ thật | Máy (`NB_cpu`) | Chỉ sau Go/No-Go |
| Đẩy run lên Drive | Máy (terminal, rclone) | Để Colab đọc `uq_manifest.csv` |
| `infer_images.py --task seg` | Colab `NB_seg` (GPU) | ms/ảnh từ benchmark P2 × số ảnh UQ = thời gian ước tính |
| `infer_images.py --task cls` | Colab `NB_cls` (GPU) | Như trên |
| Kéo kết quả suy luận về máy | Máy (terminal, rclone) | |
| `train_ridge.py` | Máy (`NB_cpu`) | Vài giây; Ridge không cần GPU |

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2. `SegPredictor`/`ClsPredictor.from_checkpoint` đã được thử trên CPU với code PanDerm thật và checkpoint giả. Toàn bộ phần CPU (ghép cặp, đặc trưng, Ridge, các script) đã chạy pytest và chạy đầu-cuối trên dữ liệu giả.

## 3. Giải thích

### 3.1. Cổng Go/No-Go trước khi đưa UQ lên Drive/Colab

Chỉ tạo `data/uq/` trên Drive khi **cả 5** điều kiện của kế hoạch Phase 1 đã được SV A xác nhận bằng văn bản:

- [ ] Quyền sử dụng cho nghiên cứu, **bao gồm xử lý trên dịch vụ đám mây** (Google Drive/Colab).
- [ ] Nối được các ảnh của cùng `lesion_id`.
- [ ] Có thứ tự thời gian đáng tin cậy để lập cặp t → t+1.
- [ ] Có `participant_id` để chia tập chống rò rỉ.
- [ ] Đủ cặp sau lọc cho train/val/test theo participant (đếm bằng `build_pairs.py`).
- [ ] Quyền lưu bản sao trên máy cá nhân của thành viên nhóm (`~/nckh_drive/data/uq`); nếu không có, ghi rõ chỉ xử lý trên Colab/Drive và không chạy rclone thư mục `uq` về máy (khi đó chạy `build_pairs.py`/`train_ridge.py` ở mục 4.7 trong `NB_seg` bằng `{VENV}/bin/python {CODE}/scripts/...`).

Nếu điều kiện đầu tiên chưa rõ: **không** tải UQ lên. Toàn bộ code của phase này vẫn hoàn thành và kiểm thử được bằng dữ liệu giả (mục 4.6).

### 3.2. Định nghĩa

Với ảnh tại lần khám t và ảnh kế tiếp t+1 của cùng lesion:

```
a_t      = (số pixel thuộc mask PanDerm của ảnh t) / (tổng số pixel ảnh t)
Δa       = a_{t+1} − a_t                                  ← target
z_t      = [a_t, circularity_t, eccentricity_t, p_mel_t, p_nev_t, p_sk_t, Δt_ngày]   ← đặc trưng, CHỈ từ thời điểm t
Δâ       = β₀ + βᵀ · chuẩn_hóa(z_t)                        ← Ridge (StandardScaler + Ridge, fit trên train)
â_{t+1}  = clip(a_t + Δâ, 0, 1)
baseline: Δâ = 0
```

- **circularity** = 4π·diện tích / chu vi² của thành phần lớn nhất (hình tròn ≈ 1, hình răng cưa → nhỏ). **eccentricity** của ellipse cùng moment (tròn = 0, dẹt → 1).
- **Δt** là số ngày thực tế, số thực (giây / 86.400), **không** giả định 6 tháng. Mọi cặp có Δt ≥ 1 ngày (ảnh dưới 1 ngày bị loại `same_visit`).
- **Định dạng ngày** được khai báo trong `configs/uq_column_map.yaml` (khóa `captured_at_format`, mặc định ISO-8601). Không để pandas tự đoán: `05/01/2020` có thể bị đọc là 1/5 ở dòng này và 5/1 ở dòng khác.
- Mask của t+1 chỉ dùng để tạo target, **không bao giờ** là đầu vào.

### 3.3. Ghép cặp và lý do loại

`build_consecutive_pairs` lọc theo thứ tự cố định, mỗi ảnh mang **một** lý do loại để bảng flow cộng lại khớp:

| Thứ tự | Lý do | Ý nghĩa |
|---|---|---|
| 1 | `missing_lesion_id` | Không biết thuộc tổn thương nào |
| 2 | `unreadable_image` | Ảnh không mở được |
| 3 | `not_dermoscopy` | Ảnh clinical/tile… |
| 4 | `missing_timestamp` | Không có hoặc không đọc được thời điểm chụp |
| 5 | `participant_mismatch` | Một lesion gắn với hơn một người → lỗi ID; loại cả lesion, không đoán |
| 6 | `same_timestamp` | Hai ảnh cùng thời điểm: giữ ảnh đầu (theo `image_id`), loại ảnh còn lại |
| 7 | `same_visit` | Ảnh cách ảnh **được giữ** trước đó < 1 ngày (nhiều ảnh trong một buổi khám, hoặc ngày-không-giờ lệch múi giờ): giữ ảnh đầu buổi, loại ảnh sau |

Sau khi lọc, sắp xếp theo `(lesion_id, captured_at, image_id)` và chỉ ghép **ảnh kề nhau**: lesion có 3 lần chụp cho 2 cặp (1→2, 2→3), không có cặp 1→3. Split được gán theo participant **trước** khi ghép, nên mỗi cặp tự kế thừa split của người đó.

Cặp có mask rỗng ở t hoặc t+1 (`empty_mask`), thiếu kết quả suy luận (`missing_inference`) hoặc đặc trưng không hữu hạn (`nonfinite_feature`) được giữ trong `features.csv` với `usable = False` và bị đếm, không bị xoá âm thầm.

### 3.4. Hai mức đánh giá (kế hoạch 6.4), không được gộp

- **Mức A (tự động):** target Δa tính từ **mask PanDerm** ở cả t và t+1. Đây là "dự báo tỷ lệ mask PanDerm sẽ tạo ra ở ảnh sau", **không** phải thay đổi sinh học.
- **Mức B (audit thủ công):** trên các cặp đã gán mask tay (P4), target tính từ **mask thủ công**. Mô hình đã khóa ở mức A được **đánh giá** (không fit lại) trên target thủ công (mục 4.8).

### 3.5. Bảy điều chống rò rỉ (kế hoạch 6.5) → cơ chế/test

| Điều | Cơ chế / test |
|---|---|
| Participant test không xuất hiện ở train/val | `assign_group_split` theo participant + `assert_no_group_leakage` (gọi trong `build_pairs.py`); `test_no_participant_in_two_splits` |
| Features chỉ lấy từ visit t | `FEATURE_COLUMNS` không có cột `_t1`, không có target; `test_feature_table_no_future_columns` |
| Không fit scaler/chọn alpha trên test | Scaler nằm trong `Pipeline` fit trên train; `select_alpha` chỉ nhận train + val; `test_scaler_fit_on_train_only`, `test_select_alpha_uses_val` |
| Không dùng t+1 để chọn cặp/feature/ngưỡng/checkpoint | Lọc cặp chỉ dựa trên metadata và chất lượng ảnh; checkpoint chọn ở P5 trên ISIC |
| Không ghép cặp sau khi chia bằng quy tắc làm lẫn participant | Split gán trước, cặp kế thừa; `test_pairs_inherit_split_and_no_leakage` |
| Cặp bị loại được thống kê trước khi mở test | `flow.json`, `pairs_excluded.csv`, `features.csv` (`exclude_reason`) ghi ngay khi chạy, trước `--open-test` |
| Kiểm thử tự động | `pytest` (mục 5) |

`train_ridge.py` mặc định **không** ghi dự báo test, và trong `features.csv` các cột `delta_area`, `area_ratio_t1`, `empty_t1` của cặp test bị để trống (target test không nằm trên Drive trước khi khóa). Chỉ khi chạy với `--open-test` và gõ đúng `mo test` vào câu hỏi xác nhận thì mới ghi `predictions_test.csv`.

## 4. Code

### 4.1. `src/nckh/pairs.py`

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/pairs.py`

```python
# file: src/nckh/pairs.py
"""Ghép cặp ảnh liên tiếp t → t+1 của cùng tổn thương, kèm bảng lý do loại; chọn mẫu audit phân tầng."""
import numpy as np
import pandas as pd

PAIR_EXCLUDE_REASONS = ("missing_lesion_id", "unreadable_image", "not_dermoscopy", "missing_timestamp",
                        "participant_mismatch", "same_timestamp", "same_visit")
MIN_DELTA_DAYS = 1.0  # ảnh cách ảnh trước < 1 ngày coi là cùng buổi khám, không phải "lần khám kế tiếp"
PAIR_COLUMNS = ["participant_id", "lesion_id", "split", "image_id_t", "image_path_t", "captured_at_t",
                "image_id_t1", "image_path_t1", "captured_at_t1", "delta_days"]


def build_consecutive_pairs(df: pd.DataFrame, dermoscopy_value: str = "dermoscopy",
                            min_delta_days: float = MIN_DELTA_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df: manifest UQ đã có cột split (chia theo participant TRƯỚC khi gọi hàm này)."""
    work = df.copy()
    excluded = []

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal work
        excluded.append(work.loc[mask, ["image_id", "lesion_id"]].assign(reason=reason))
        work = work.loc[~mask]

    # Thứ tự lọc cố định để mỗi ảnh chỉ mang một lý do loại, giúp bảng flow cộng lại khớp.
    drop(work["lesion_id"].isna(), "missing_lesion_id")
    if "readable" in work:
        drop(~work["readable"].astype(bool), "unreadable_image")
    drop(work["modality"] != dermoscopy_value, "not_dermoscopy")
    drop(work["captured_at"].isna(), "missing_timestamp")
    n_people = work.groupby("lesion_id")["participant_id"].transform("nunique")
    drop(n_people > 1, "participant_mismatch")  # một lesion thuộc hai người = lỗi ID, không đoán người nào đúng

    work = work.sort_values(["lesion_id", "captured_at", "image_id"])
    # Hai ảnh cùng thời điểm không cho biết thứ tự: giữ ảnh đầu (theo image_id), loại ảnh còn lại.
    drop(work.duplicated(["lesion_id", "captured_at"], keep="first"), "same_timestamp")
    # Nhiều ảnh trong một buổi khám (cách nhau vài phút): chỉ giữ ảnh đầu buổi, so với ảnh GIỮ LẠI gần nhất.
    same_visit = pd.Series(False, index=work.index)
    for _, group in work.groupby("lesion_id", sort=False):
        kept = None
        for idx, ts in group["captured_at"].items():
            if kept is not None and (ts - kept).total_seconds() / 86400 < min_delta_days:
                same_visit[idx] = True
            else:
                kept = ts
    drop(same_visit, "same_visit")

    nxt = work.groupby("lesion_id").shift(-1)
    has_next = nxt["image_id"].notna()
    t, t1 = work[has_next], nxt[has_next]
    pairs = pd.DataFrame({
        "participant_id": t["participant_id"].to_numpy(),
        "lesion_id": t["lesion_id"].to_numpy(),
        "split": t["split"].to_numpy(),
        "image_id_t": t["image_id"].to_numpy(),
        "image_path_t": t["image_path"].to_numpy(),
        "captured_at_t": t["captured_at"].to_numpy(),
        "image_id_t1": t1["image_id"].to_numpy(),
        "image_path_t1": t1["image_path"].to_numpy(),
        "captured_at_t1": t1["captured_at"].to_numpy(),
    })
    # Δt thực tế theo ngày (số thực), không giả định các lần khám cách đều 6 tháng.
    pairs["delta_days"] = (pd.to_datetime(pairs["captured_at_t1"]) - pd.to_datetime(pairs["captured_at_t"])).dt.total_seconds() / 86400
    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame(columns=["image_id", "lesion_id", "reason"])
    return pairs[PAIR_COLUMNS], excluded_df


def stratified_audit_sample(pairs: pd.DataFrame, n_pairs: int, strata_cols: list[str], seed: int = 2026) -> pd.DataFrame:
    if n_pairs <= 0:
        raise ValueError("n_pairs phải > 0")
    # Chia mỗi cột thành 4 nhóm theo tứ phân vị (xếp hạng trước để giá trị trùng không làm hỏng điểm cắt).
    codes = [pd.qcut(pairs[c].rank(method="first"), 4, labels=False, duplicates="drop").astype(int).astype(str)
             for c in strata_cols]
    stratum = codes[0].str.cat(codes[1:], sep="_") if len(codes) > 1 else codes[0]
    rng = np.random.default_rng(seed)
    pools = {s: list(rng.permutation(idx)) for s, idx in sorted(stratum.groupby(stratum).groups.items())}
    per = n_pairs // len(pools)
    chosen = []
    for s in pools:
        chosen += pools[s][:per]
        pools[s] = pools[s][per:]
    # Phần dư chia vòng tròn qua các ô theo thứ tự; ô hết cặp thì bỏ qua.
    while len(chosen) < n_pairs and any(pools.values()):
        for s in pools:
            if pools[s] and len(chosen) < n_pairs:
                chosen.append(pools[s].pop(0))
    return pairs.loc[chosen].assign(stratum=stratum.loc[chosen])
```

### 4.2. `src/nckh/features.py`

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/features.py`

```python
# file: src/nckh/features.py
"""Đặc trưng tại thời điểm t và target Δa cho Ridge.

Chỉ cột trong FEATURE_COLUMNS được đưa vào mô hình. Mọi thứ của ảnh t+1 (area_ratio_t1, empty_t1)
chỉ dùng để tạo target hoặc lọc cặp, không bao giờ là đầu vào.
"""
import math

import numpy as np
import pandas as pd
from skimage.measure import label, perimeter, regionprops

FEATURE_COLUMNS: tuple[str, ...] = ("area_ratio_t", "circularity_t", "eccentricity_t", "p_mel_t", "p_nev_t", "p_sk_t",
                                    "delta_days")
TARGET_COLUMN = "delta_area"
PER_IMAGE_COLUMNS = ("area_ratio", "circularity", "eccentricity", "empty", "p_mel", "p_nev", "p_sk")


def mask_features(mask: np.ndarray) -> dict:
    m = mask.astype(bool)
    if not m.any():
        # Mask rỗng: không bịa ra hình dạng; cặp chứa ảnh này sẽ bị loại với lý do empty_mask.
        return {"area_ratio": 0.0, "circularity": float("nan"), "eccentricity": float("nan"), "empty": True}
    labeled = label(m)
    largest = max(regionprops(labeled), key=lambda r: r.area)
    region = labeled == largest.label
    p = perimeter(region)
    circ = float(min(4 * math.pi * largest.area / p ** 2, 1.0)) if p > 0 else float("nan")
    return {"area_ratio": float(m.mean()), "circularity": circ, "eccentricity": float(largest.eccentricity), "empty": False}


def build_feature_table(pairs: pd.DataFrame, per_image: pd.DataFrame) -> pd.DataFrame:
    per = per_image.set_index("image_id")[list(PER_IMAGE_COLUMNS)]
    at_t = per.add_suffix("_t").reindex(pairs["image_id_t"]).reset_index(drop=True)
    at_t1 = per[["area_ratio", "empty"]].add_suffix("_t1").reindex(pairs["image_id_t1"]).reset_index(drop=True)
    table = pd.concat([pairs.reset_index(drop=True), at_t, at_t1], axis=1)
    table[TARGET_COLUMN] = table["area_ratio_t1"] - table["area_ratio_t"]

    missing = at_t["area_ratio_t"].isna() | at_t1["area_ratio_t1"].isna()
    empty = table["empty_t"].fillna(False).astype(bool) | table["empty_t1"].fillna(False).astype(bool)
    nonfinite = ~np.isfinite(table[list(FEATURE_COLUMNS)].to_numpy(float)).all(axis=1)
    table["exclude_reason"] = np.select([missing, empty, nonfinite], ["missing_inference", "empty_mask", "nonfinite_feature"], "")
    table["usable"] = table["exclude_reason"] == ""
    return table
```

### 4.3. `src/nckh/forecast.py`

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/forecast.py`

```python
# file: src/nckh/forecast.py
"""Ridge dự báo Δa một bước và baseline "không thay đổi"."""
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN

ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)  # khóa trong protocol; không mở rộng sau khi xem test
DELTA_DAYS_ERROR = "Δt phải là số ngày dương, ví dụ 180"


def make_model(alpha: float) -> Pipeline:
    # Scaler nằm TRONG pipeline nên chỉ học mean/std từ dữ liệu được fit (train), không bao giờ từ val/test.
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def fit_forecaster(train: pd.DataFrame, alpha: float) -> Pipeline:
    return make_model(alpha).fit(train[list(FEATURE_COLUMNS)].to_numpy(float), train[TARGET_COLUMN].to_numpy(float))


def predict_delta(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[list(FEATURE_COLUMNS)].to_numpy(float))


def select_alpha(train: pd.DataFrame, val: pd.DataFrame, grid: Sequence[float] = ALPHA_GRID) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in grid:
        pred = predict_delta(fit_forecaster(train, alpha), val)
        rows.append({"alpha": float(alpha), "val_mae": float(np.mean(np.abs(pred - val[TARGET_COLUMN].to_numpy(float))))})
    table = pd.DataFrame(rows)
    best_mae = table["val_mae"].min()
    # Hòa (trong sai số làm tròn) thì chọn alpha lớn hơn: mô hình co mạnh hơn, đơn giản hơn.
    best = float(table.loc[table["val_mae"] <= best_mae + 1e-12, "alpha"].max())
    return best, table


def baseline_delta(n: int) -> np.ndarray:
    return np.zeros(n)


def predict_next_area(area_t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(area_t, float) + np.asarray(delta, float), 0.0, 1.0)


def validate_delta_days(value: object) -> float:
    # bool là lớp con của int trong Python: True sẽ thành 1 ngày nếu không chặn riêng.
    if value is None or isinstance(value, bool):
        raise ValueError(DELTA_DAYS_ERROR)
    try:
        days = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(DELTA_DAYS_ERROR) from exc
    if not math.isfinite(days) or days <= 0:
        raise ValueError(DELTA_DAYS_ERROR)
    return days
```

`validate_delta_days` dùng chung cho demo P9: từ chối `True` (vì trong Python `True == 1`), chuỗi rỗng, `"6 tháng"`, `"nan"`, `"inf"`, số âm, số 0.

### 4.4. Scripts

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/make_fake_uq.py`

```python
# file: scripts/make_fake_uq.py
"""Sinh bộ dữ liệu dọc GIẢ có cấu trúc giống UQ để phát triển/kiểm thử P6–P8 khi chưa có quyền dữ liệu thật.

Ảnh: nền da + một elip tối có bán kính đổi tuyến tính theo thời gian. Metadata cố ý dùng tên cột KHÁC schema
(pid, lesion, img, file, date, type) để luyện bước map cột, và cố ý chèn đúng 3 lỗi: 1 ảnh trùng ngày,
1 ảnh thiếu ngày, 1 ảnh clinical. KHÔNG dùng kết quả trên dữ liệu này cho bài báo.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SKIN, LESION = (214, 170, 150), (92, 58, 42)


def _draw(radius: float, center: tuple[float, float], rng: np.random.Generator) -> Image.Image:
    noise = rng.normal(0, 6, (256, 256, 3))
    img = Image.fromarray(np.clip(np.array(SKIN) + noise, 0, 255).astype(np.uint8))
    cx, cy = center
    ImageDraw.Draw(img).ellipse([cx - radius, cy - 0.8 * radius, cx + radius, cy + 0.8 * radius], fill=LESION)
    return img


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--participants", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    for p in range(args.participants):
        for les in range(int(rng.integers(1, 3))):
            lesion = f"P{p:03d}_L{les}"
            day = date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
            radius, growth = rng.uniform(20, 60), rng.uniform(-0.02, 0.04)  # pixel/ngày
            center = (128 + rng.uniform(-10, 10), 128 + rng.uniform(-10, 10))
            for v in range(int(rng.integers(2, 5))):
                img_id = f"{lesion}_V{v}"
                _draw(max(radius, 5), center, rng).save(args.out_dir / "images" / f"{img_id}.jpg", quality=95)
                rows.append({"pid": f"P{p:03d}", "lesion": lesion, "img": img_id, "file": f"images/{img_id}.jpg",
                             "date": day.isoformat(), "type": "Dermoscopy"})
                gap = int(rng.integers(150, 211))
                day += timedelta(days=gap)
                radius += growth * gap

    meta = pd.DataFrame(rows)
    # Ba lỗi cố ý, mỗi loại đúng một ảnh, rơi vào ba lesion khác nhau.
    first = meta.iloc[0]
    dup_id = f"{first['lesion']}_DUP"
    _draw(30, (128, 128), rng).save(args.out_dir / "images" / f"{dup_id}.jpg")
    meta = pd.concat([meta, pd.DataFrame([{**first.to_dict(), "img": dup_id, "file": f"images/{dup_id}.jpg"}])],
                     ignore_index=True)
    lesions = meta["lesion"].unique()
    meta.loc[meta.index[meta["lesion"] == lesions[1]][0], "date"] = ""
    meta.loc[meta.index[meta["lesion"] == lesions[2]][0], "type"] = "Clinical"
    meta.to_csv(args.out_dir / "metadata.csv", index=False)

    (args.out_dir / "fake_column_map.yaml").write_text(
        'participant_id: "pid"\nlesion_id: "lesion"\nimage_id: "img"\nimage_path: "file"\n'
        'captured_at: "date"\nmodality: "type"\nmodality_dermoscopy_value: "dermoscopy"\n', encoding="utf-8")
    print(f"Đã sinh {len(meta)} ảnh GIẢ, {meta['pid'].nunique()} participant, {meta['lesion'].nunique()} lesion → {args.out_dir}")


if __name__ == "__main__":
    main()
```

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/build_pairs.py`

```python
# file: scripts/build_pairs.py
"""Đọc metadata UQ → kiểm tra ảnh → chia tập theo participant → ghép cặp t→t+1 → ghi manifest, cặp, flow.

Ví dụ: python scripts/build_pairs.py --metadata $ROOT/data/uq/metadata.csv --column-map configs/uq_column_map.yaml \
           --data-root $ROOT/data/uq --out-dir $ROOT/runs/<run_id> --seed 2026
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from nckh.manifest import assert_no_group_leakage, assign_group_split, check_image, load_uq_metadata
from nckh.pairs import build_consecutive_pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--column-map", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    cmap = yaml.safe_load(args.column_map.read_text(encoding="utf-8"))
    unfilled = [k for k, v in cmap.items() if "ĐIỀN" in str(v)]
    if unfilled:
        raise SystemExit(f"Chưa điền tên cột gốc trong {args.column_map}: {unfilled} (<ĐIỀN TÊN CỘT GỐC>)")
    dermo = cmap.pop("modality_dermoscopy_value", "dermoscopy")
    date_format = cmap.pop("captured_at_format", "ISO8601")

    uq = load_uq_metadata(args.metadata, cmap, date_format)
    uq["readable"] = [check_image(args.data_root / p)["readable"] if isinstance(p, str) else False for p in uq["image_path"]]
    # Chia theo participant TRƯỚC khi ghép cặp: mọi ảnh/cặp của một người nằm trong đúng một split.
    uq["split"] = assign_group_split(uq, "participant_id", seed=args.seed)
    assert_no_group_leakage(uq, "participant_id")
    pairs, excluded = build_consecutive_pairs(uq, dermo)
    assert_no_group_leakage(pairs, "participant_id")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    uq.to_csv(args.out_dir / "uq_manifest.csv", index=False)
    pairs.to_csv(args.out_dir / "pairs.csv", index=False)
    excluded.to_csv(args.out_dir / "pairs_excluded.csv", index=False)
    flow = {
        "n_images": int(len(uq)),
        "n_participants": int(uq["participant_id"].nunique()),
        "n_lesions": int(uq["lesion_id"].nunique()),
        "excluded_by_reason": {k: int(v) for k, v in sorted(excluded["reason"].value_counts().items())},
        "n_lesions_with_pair": int(pairs["lesion_id"].nunique()),
        "n_pairs_by_split": {k: int(v) for k, v in pairs["split"].value_counts().sort_index().items()},
        "n_participants_by_split": {k: int(v) for k, v in pairs.groupby("split")["participant_id"].nunique().items()},
        "delta_days_quantiles": {str(q): float(v) for q, v in pairs["delta_days"].quantile([0, 0.25, 0.5, 0.75, 1]).items()},
        "seed": args.seed,
    }
    (args.out_dir / "flow.json").write_text(json.dumps(flow, indent=2, ensure_ascii=False))
    print(json.dumps(flow, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/infer_images.py`

```python
# file: scripts/infer_images.py
"""Suy luận PanDerm cho một danh sách ảnh: seg → masks/*.png + seg_features.csv; cls → cls_probs.csv.

Chạy bằng python của venv tương ứng (venv_seg cho --task seg, venv_cls cho --task cls).
--fake: thay PanDerm bằng ngưỡng độ tối / xác suất cố định để chạy thử pipeline trên CPU với dữ liệu GIẢ.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nckh.features import mask_features
from nckh.infer import ClsPredictor, SegPredictor, largest_component

logger = logging.getLogger("infer_images")


class FakeSeg:
    device = "cpu"

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return largest_component(rgb.mean(axis=2) < 100)


class FakeCls:
    device = "cpu"

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return np.array([0.2, 0.6, 0.2])


def _items(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.images:
        return [(Path(p).stem, Path(p)) for p in args.images]
    df = pd.read_csv(args.manifest)
    return [(str(i), args.data_root / p) for i, p in zip(df[args.id_col], df[args.path_col])]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["seg", "cls"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+")
    src.add_argument("--manifest", type=Path)
    ap.add_argument("--path-col", default="image_path")
    ap.add_argument("--id-col", default="image_id")
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--panderm-dir", type=Path)
    ap.add_argument("--pretrained", type=Path)
    ap.add_argument("--finetuned", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args(argv)

    if args.fake:
        print("CẢNH BÁO: chế độ --fake, không dùng cho kết quả", file=sys.stderr)
        predictor = FakeSeg() if args.task == "seg" else FakeCls()
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.finetuned is None:
            raise SystemExit("Cần --finetuned (checkpoint đã fine-tune ở P5a/P5b) khi không dùng --fake")
        if args.task == "seg":
            predictor = SegPredictor.from_checkpoint(args.panderm_dir, args.pretrained, args.finetuned, device)
        else:
            predictor = ClsPredictor.from_checkpoint(args.panderm_dir, 3, args.finetuned, None, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for image_id, path in _items(args):
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        except OSError as exc:
            # Ảnh lỗi không làm dừng cả lô: ghi lại, cặp chứa ảnh này sẽ bị loại với lý do missing_inference.
            logger.warning("Bỏ qua %s: %s", path, exc)
            failed.append({"image_id": image_id, "path": str(path), "error": str(exc)})
            continue
        out = predictor.predict(rgb)
        if args.task == "seg":
            (args.out_dir / "masks").mkdir(exist_ok=True)
            Image.fromarray(out.astype(np.uint8) * 255).save(args.out_dir / "masks" / f"{image_id}.png")
            rows.append({"image_id": image_id, **mask_features(out)})
        else:
            rows.append({"image_id": image_id, "p_mel": out[0], "p_nev": out[1], "p_sk": out[2]})

    name = "seg_features.csv" if args.task == "seg" else "cls_probs.csv"
    pd.DataFrame(rows).to_csv(args.out_dir / name, index=False)
    pd.DataFrame(failed, columns=["image_id", "path", "error"]).to_csv(args.out_dir / f"{args.task}_failed.csv", index=False)
    print(f"{args.task}: {len(rows)} ảnh OK, {len(failed)} lỗi → {args.out_dir / name}")


if __name__ == "__main__":
    main()
```

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/train_ridge.py`

```python
# file: scripts/train_ridge.py
"""Dựng bảng đặc trưng → chọn alpha trên val → fit Ridge trên train → dự báo val (và test khi được phép).

Ví dụ: python scripts/train_ridge.py --pairs $RUN/pairs.csv --seg-features $RUN/seg_features.csv \
           --cls-probs $RUN/cls_probs.csv --out-dir $RUN/ridge [--open-test]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_table
from nckh.forecast import baseline_delta, fit_forecaster, predict_delta, predict_next_area, select_alpha
from nckh.runcard import sha256_file, write_run_card

PREDICTION_COLUMNS = ["participant_id", "lesion_id", "image_id_t", "split", "delta_days", "area_ratio_t", "delta_area",
                      "pred_ridge", "pred_baseline", "next_area_ridge"]


def _predictions(model: object, df: pd.DataFrame) -> pd.DataFrame:
    out = df[["participant_id", "lesion_id", "image_id_t", "split", "delta_days", "area_ratio_t", TARGET_COLUMN]].copy()
    out["pred_ridge"] = predict_delta(model, df)
    out["pred_baseline"] = baseline_delta(len(df))
    out["next_area_ridge"] = predict_next_area(df["area_ratio_t"], out["pred_ridge"])
    return out[PREDICTION_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--seg-features", type=Path, required=True)
    ap.add_argument("--cls-probs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--open-test", action="store_true", help="ghi predictions_test.csv (chỉ một lần, sau khi khóa)")
    ap.add_argument("--yes", action="store_true", help="bỏ câu hỏi xác nhận khi --open-test")
    args = ap.parse_args(argv)

    per_image = pd.read_csv(args.seg_features).merge(pd.read_csv(args.cls_probs), on="image_id", how="outer")
    table = build_feature_table(pd.read_csv(args.pairs), per_image)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved = table.astype({"empty_t1": "object"})  # cột bool không chứa được NaN
    if not args.open_test:
        # Chưa mở test: không để target/giá trị t+1 của cặp test nằm trên Drive (ai đó có thể vô tình đọc).
        saved.loc[saved["split"] == "test", ["area_ratio_t1", "empty_t1", TARGET_COLUMN]] = np.nan
    saved.to_csv(args.out_dir / "features.csv", index=False)

    usable = table[table["usable"]]
    train, val, test = (usable[usable["split"] == s] for s in ("train", "val", "test"))
    if train.empty or val.empty:
        raise SystemExit(f"Không đủ cặp dùng được: train={len(train)}, val={len(val)}")
    alpha, selection = select_alpha(train, val)
    selection.to_csv(args.out_dir / "alpha_selection.csv", index=False)
    model = fit_forecaster(train, alpha)
    joblib.dump({"model": model, "alpha": alpha, "feature_columns": list(FEATURE_COLUMNS),
                 "pairs_sha256": sha256_file(args.pairs)}, args.out_dir / "ridge.joblib")

    pred_val = _predictions(model, val)
    pred_val.to_csv(args.out_dir / "predictions_val.csv", index=False)
    summary = {
        "alpha": alpha,
        "n_usable_by_split": {s: int(len(d)) for s, d in (("train", train), ("val", val), ("test", test))},
        "excluded_by_reason": {k: int(v) for k, v in table.loc[~table["usable"], "exclude_reason"].value_counts().items()},
        "val_mae_ridge": float(np.abs(pred_val["pred_ridge"] - pred_val[TARGET_COLUMN]).mean()),
        "val_mae_baseline": float(np.abs(pred_val[TARGET_COLUMN]).mean()),
    }
    if args.open_test:
        if not args.yes:
            answer = input("Protocol đã khóa và SV A đã xác nhận mở participant test? Gõ 'mo test' để tiếp tục: ")
            if answer.strip() != "mo test":
                raise SystemExit("Đã hủy: chưa mở test")
        _predictions(model, test).to_csv(args.out_dir / "predictions_test.csv", index=False)
        summary["test_opened"] = True
    (args.out_dir / "ridge_summary.json").write_text(json.dumps(summary, indent=2))
    write_run_card(args.out_dir, seed=0, config={"alpha": alpha, "open_test": args.open_test},
                   inputs={"pairs": args.pairs, "seg_features": args.seg_features, "cls_probs": args.cls_probs})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

### 4.5. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_pairs.py`

```python
# file: tests/test_pairs.py
import pandas as pd
import pytest

from nckh.manifest import assert_no_group_leakage, assign_group_split
from nckh.pairs import build_consecutive_pairs, stratified_audit_sample


def _uq(rows: list[tuple]) -> pd.DataFrame:
    """rows: (participant, lesion, image, date_str_or_None, modality)."""
    df = pd.DataFrame(rows, columns=["participant_id", "lesion_id", "image_id", "captured_at", "modality"])
    df["image_path"] = df["image_id"] + ".jpg"
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce", format="mixed")
    df["split"] = "train"
    return df


def _reasons(excluded: pd.DataFrame) -> dict[str, str]:
    return dict(zip(excluded["image_id"], excluded["reason"]))


def test_pairs_consecutive_only() -> None:
    pairs, _ = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "c", "2021-01-01", "dermoscopy"),
        ("P1", "L1", "b", "2020-07-01", "dermoscopy"),
    ]))
    assert list(zip(pairs.image_id_t, pairs.image_id_t1)) == [("a", "b"), ("b", "c")]


def test_pairs_never_cross_lesion() -> None:
    pairs, _ = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L2", "b", "2020-07-01", "dermoscopy"),
    ]))
    assert pairs.empty


def test_delta_days_positive_float() -> None:
    pairs, _ = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "b", "2020-07-01", "dermoscopy"),
    ]))
    assert pairs["delta_days"].tolist() == [182.0]


def test_same_day_excluded_reason() -> None:
    pairs, excluded = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "b", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "c", "2020-07-19", "dermoscopy"),
    ]))
    assert list(zip(pairs.image_id_t, pairs.image_id_t1)) == [("a", "c")]
    assert _reasons(excluded) == {"b": "same_timestamp"}


def test_missing_timestamp_reason() -> None:
    pairs, excluded = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "b", None, "dermoscopy"),
        ("P1", "L1", "c", "2020-07-01", "dermoscopy"),
    ]))
    assert _reasons(excluded) == {"b": "missing_timestamp"}
    assert len(pairs) == 1


def test_not_dermoscopy_reason() -> None:
    _, excluded = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P1", "L1", "b", "2020-02-01", "clinical"),
    ]))
    assert _reasons(excluded) == {"b": "not_dermoscopy"}


def test_missing_lesion_reason() -> None:
    _, excluded = build_consecutive_pairs(_uq([("P1", None, "a", "2020-01-01", "dermoscopy")]))
    assert _reasons(excluded) == {"a": "missing_lesion_id"}


def test_participant_mismatch_reason() -> None:
    pairs, excluded = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-01", "dermoscopy"),
        ("P2", "L1", "b", "2020-07-01", "dermoscopy"),
    ]))
    assert pairs.empty
    assert _reasons(excluded) == {"a": "participant_mismatch", "b": "participant_mismatch"}


def test_unreadable_reason() -> None:
    df = _uq([("P1", "L1", "a", "2020-01-01", "dermoscopy"), ("P1", "L1", "b", "2020-07-01", "dermoscopy")])
    df["readable"] = [True, False]
    pairs, excluded = build_consecutive_pairs(df)
    assert pairs.empty and _reasons(excluded) == {"b": "unreadable_image"}


def test_timezone_and_date_only_mix() -> None:
    pairs, _ = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-05", "dermoscopy"),
        ("P1", "L1", "b", "2020-01-06T00:00:00+00:00", "dermoscopy"),
    ]))
    assert pairs["delta_days"].tolist() == [1.0]


def test_pairs_inherit_split_and_no_leakage() -> None:
    rows = []
    for p in range(40):
        for lesion in range(2):
            for v, date in enumerate(["2020-01-01", "2020-07-01", "2021-01-01"]):
                rows.append((f"P{p}", f"P{p}_L{lesion}", f"P{p}_L{lesion}_{v}", date, "dermoscopy"))
    df = _uq(rows)
    df["split"] = assign_group_split(df, "participant_id", seed=2026)
    pairs, _ = build_consecutive_pairs(df)
    assert len(pairs) == 40 * 2 * 2
    assert_no_group_leakage(pairs, "participant_id")
    merged = pairs.merge(df[["image_id", "split"]], left_on="image_id_t1", right_on="image_id", suffixes=("", "_t1"))
    assert (merged["split"] == merged["split_t1"]).all()


def test_stratified_audit_sample_size_and_seed() -> None:
    pool = pd.DataFrame({"delta_days": range(200), "area_ratio_t": [i % 17 / 17 for i in range(200)]})
    a = stratified_audit_sample(pool, n_pairs=50, strata_cols=["delta_days", "area_ratio_t"], seed=2026)
    b = stratified_audit_sample(pool, n_pairs=50, strata_cols=["delta_days", "area_ratio_t"], seed=2026)
    assert len(a) == 50 and a.index.equals(b.index)
    assert a["stratum"].nunique() >= 8
    with pytest.raises(ValueError):
        stratified_audit_sample(pool, n_pairs=0, strata_cols=["delta_days"])


def test_same_visit_within_one_day_excluded() -> None:
    # Hai ảnh chụp cách nhau vài phút / lệch múi giờ trong cùng buổi khám không phải "lần khám kế tiếp".
    pairs, excluded = build_consecutive_pairs(_uq([
        ("P1", "L1", "a", "2020-01-05T10:00:00", "dermoscopy"),
        ("P1", "L1", "b", "2020-01-05T10:05:00", "dermoscopy"),
        ("P1", "L1", "c", "2020-07-05T10:00:00", "dermoscopy"),
    ]))
    assert list(zip(pairs.image_id_t, pairs.image_id_t1)) == [("a", "c")]
    assert _reasons(excluded) == {"b": "same_visit"}
```

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_features.py`

```python
# file: tests/test_features.py
import numpy as np
import pandas as pd
import pytest

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_table, mask_features


def test_square_area_ratio() -> None:
    m = np.zeros((100, 100), bool)
    m[10:20, 10:20] = True
    assert mask_features(m)["area_ratio"] == pytest.approx(0.01)


def test_disk_circularity_near_one() -> None:
    yy, xx = np.mgrid[:200, :200]
    disk = (yy - 100) ** 2 + (xx - 100) ** 2 <= 30 ** 2
    f = mask_features(disk)
    assert f["circularity"] > 0.85 and f["eccentricity"] < 0.1 and not f["empty"]


def test_empty_mask_flagged() -> None:
    f = mask_features(np.zeros((10, 10), bool))
    assert f["empty"] and f["area_ratio"] == 0.0 and np.isnan(f["circularity"]) and np.isnan(f["eccentricity"])


def test_single_pixel_mask_no_crash() -> None:
    m = np.zeros((10, 10), bool)
    m[5, 5] = True
    f = mask_features(m)
    assert not f["empty"] and f["area_ratio"] == pytest.approx(0.01)
    assert np.isnan(f["circularity"]) or 0 <= f["circularity"] <= 1


def _per_image(rows: dict[str, tuple]) -> pd.DataFrame:
    cols = ["area_ratio", "circularity", "eccentricity", "empty", "p_mel", "p_nev", "p_sk"]
    return pd.DataFrame([(k, *v) for k, v in rows.items()], columns=["image_id", *cols])


def _pairs() -> pd.DataFrame:
    return pd.DataFrame({"participant_id": ["P1", "P1"], "lesion_id": ["L1", "L1"], "split": ["train", "train"],
                         "image_id_t": ["a", "b"], "image_id_t1": ["b", "c"], "delta_days": [180.0, 190.0]})


def test_feature_table_no_future_columns() -> None:
    assert not any(c.endswith("_t1") for c in FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    per = _per_image({"a": (0.10, 0.8, 0.3, False, 0.2, 0.7, 0.1), "b": (0.12, 0.7, 0.4, False, 0.3, 0.6, 0.1),
                      "c": (0.15, 0.6, 0.5, False, 0.3, 0.6, 0.1)})
    table = build_feature_table(_pairs(), per)
    assert table.loc[0, "area_ratio_t"] == 0.10 and table.loc[0, "p_mel_t"] == 0.2
    assert table.loc[0, TARGET_COLUMN] == pytest.approx(0.02)
    assert table["usable"].all()


def test_feature_table_empty_t1_unusable() -> None:
    per = _per_image({"a": (0.10, 0.8, 0.3, False, 0.2, 0.7, 0.1), "b": (0.0, np.nan, np.nan, True, 0.3, 0.6, 0.1),
                      "c": (0.15, 0.6, 0.5, False, 0.3, 0.6, 0.1)})
    table = build_feature_table(_pairs(), per)
    assert table["exclude_reason"].tolist() == ["empty_mask", "empty_mask"]
    assert not table["usable"].any()


def test_feature_table_missing_inference_reason() -> None:
    per = _per_image({"a": (0.10, 0.8, 0.3, False, 0.2, 0.7, 0.1), "b": (0.12, 0.7, 0.4, False, 0.3, 0.6, 0.1)})
    table = build_feature_table(_pairs(), per)
    assert table["exclude_reason"].tolist() == ["", "missing_inference"]


def test_feature_table_nonfinite_reason() -> None:
    per = _per_image({"a": (0.10, np.nan, 0.3, False, 0.2, 0.7, 0.1), "b": (0.12, 0.7, 0.4, False, 0.3, 0.6, 0.1),
                      "c": (0.15, 0.6, 0.5, False, 0.3, 0.6, 0.1)})
    assert build_feature_table(_pairs(), per)["exclude_reason"].tolist() == ["nonfinite_feature", ""]
```

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_forecast.py`

```python
# file: tests/test_forecast.py
import numpy as np
import pandas as pd
import pytest

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN
from nckh.forecast import (
    baseline_delta,
    fit_forecaster,
    predict_delta,
    predict_next_area,
    select_alpha,
    validate_delta_days,
)


def _frame(n: int, seed: int, shift: float = 0.0, noise: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    df["delta_days"] = rng.uniform(150, 210, n) + shift
    df[TARGET_COLUMN] = 0.0001 * (df["delta_days"] - 180) + 0.01 * df["area_ratio_t"] + rng.normal(0, noise, n)
    return df


def test_scaler_fit_on_train_only() -> None:
    train, val = _frame(200, 0), _frame(200, 1, shift=1000.0)
    model = fit_forecaster(train, alpha=1.0)
    means = dict(zip(FEATURE_COLUMNS, model.named_steps["scale"].mean_))
    assert means["delta_days"] == pytest.approx(train["delta_days"].mean())
    assert abs(means["delta_days"] - val["delta_days"].mean()) > 500


def test_select_alpha_uses_val() -> None:
    train, val = _frame(30, 2, noise=0.05), _frame(300, 3, noise=0.0)
    best, table = select_alpha(train, val)
    assert list(table["alpha"]) == [0.01, 0.1, 1.0, 10.0, 100.0]
    for alpha, val_mae in zip(table["alpha"], table["val_mae"]):
        # Bảng phải là MAE trên VAL của mô hình fit trên TRAIN — không phải MAE train.
        manual = np.abs(predict_delta(fit_forecaster(train, alpha), val) - val[TARGET_COLUMN]).mean()
        assert val_mae == pytest.approx(manual)
    assert best == table.loc[table["val_mae"].idxmin(), "alpha"]


def test_select_alpha_tie_prefers_larger() -> None:
    train = _frame(50, 4)
    train[TARGET_COLUMN] = 0.0   # target hằng số → mọi alpha cho cùng MAE trên val
    val = _frame(50, 5)
    val[TARGET_COLUMN] = 0.0
    best, _ = select_alpha(train, val)
    assert best == 100.0


def test_baseline_zero() -> None:
    assert (baseline_delta(4) == 0).all() and baseline_delta(4).shape == (4,)


def test_next_area_clipped() -> None:
    out = predict_next_area(np.array([0.95, 0.02, 0.3]), np.array([0.2, -0.5, 0.1]))
    assert out.tolist() == pytest.approx([1.0, 0.0, 0.4])


@pytest.mark.parametrize("value, expected", [(30, 30.0), (30.5, 30.5), ("45", 45.0), (" 60 ", 60.0)])
def test_validate_delta_days_accepts(value: object, expected: float) -> None:
    assert validate_delta_days(value) == expected


@pytest.mark.parametrize("value", [-30, 0, "", "6 tháng", "nan", "inf", None, True, float("nan"), float("inf")])
def test_validate_delta_days_rejects(value: object) -> None:
    with pytest.raises(ValueError, match="Δt"):
        validate_delta_days(value)


def test_ridge_beats_baseline_on_learnable_synthetic() -> None:
    train, val = _frame(400, 6), _frame(200, 7)
    model = fit_forecaster(train, alpha=1.0)
    mae_ridge = np.abs(predict_delta(model, val) - val[TARGET_COLUMN]).mean()
    mae_base = np.abs(baseline_delta(len(val)) - val[TARGET_COLUMN]).mean()
    assert mae_ridge < mae_base
```

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_pipeline_fake.py`

```python
# file: tests/test_pipeline_fake.py
"""Chạy toàn bộ pipeline P6 trên dữ liệu UQ GIẢ: sinh dữ liệu → ghép cặp → suy luận giả → Ridge."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_pairs  # noqa: E402
import infer_images  # noqa: E402
import make_fake_uq  # noqa: E402
import train_ridge  # noqa: E402


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("uq")
    data, run = root / "fake_uq", root / "run"
    make_fake_uq.main(["--out-dir", str(data), "--participants", "30", "--seed", "2026"])
    build_pairs.main(["--metadata", str(data / "metadata.csv"), "--column-map", str(data / "fake_column_map.yaml"),
                      "--data-root", str(data), "--out-dir", str(run), "--seed", "2026"])
    for task in ("seg", "cls"):
        infer_images.main(["--task", task, "--manifest", str(run / "uq_manifest.csv"), "--path-col", "image_path",
                           "--id-col", "image_id", "--data-root", str(data), "--out-dir", str(run), "--fake"])
    train_ridge.main(["--pairs", str(run / "pairs.csv"), "--seg-features", str(run / "seg_features.csv"),
                      "--cls-probs", str(run / "cls_probs.csv"), "--out-dir", str(run), "--open-test", "--yes"])
    return run


def test_flow_counts_injected_problems(fake_run: Path) -> None:
    flow = json.loads((fake_run / "flow.json").read_text())
    assert flow["excluded_by_reason"] == {"missing_timestamp": 1, "not_dermoscopy": 1, "same_timestamp": 1}
    assert flow["n_participants"] == 30
    assert sum(flow["n_pairs_by_split"].values()) > 50


def test_predictions_have_required_columns(fake_run: Path) -> None:
    pred = pd.read_csv(fake_run / "predictions_test.csv")
    assert list(pred.columns) == train_ridge.PREDICTION_COLUMNS
    assert (pred["split"] == "test").all()
    assert (pred["pred_baseline"] == 0).all()
    assert pred["next_area_ridge"].between(0, 1).all()


def test_no_participant_in_two_splits(fake_run: Path) -> None:
    feats = pd.read_csv(fake_run / "features.csv")
    assert feats.groupby("participant_id")["split"].nunique().max() == 1


def test_ridge_bundle_and_alpha_table(fake_run: Path) -> None:
    import joblib

    bundle = joblib.load(fake_run / "ridge.joblib")
    assert set(bundle) == {"model", "alpha", "feature_columns", "pairs_sha256"}
    assert bundle["alpha"] in (0.01, 0.1, 1.0, 10.0, 100.0)
    assert len(pd.read_csv(fake_run / "alpha_selection.csv")) == 5


def test_build_pairs_rejects_unfilled_column_map(tmp_path: Path) -> None:
    cmap = tmp_path / "map.yaml"
    cmap.write_text('participant_id: "<ĐIỀN TÊN CỘT GỐC>"\n')
    with pytest.raises(SystemExit, match="ĐIỀN"):
        build_pairs.main(["--metadata", str(tmp_path / "m.csv"), "--column-map", str(cmap), "--data-root", str(tmp_path),
                          "--out-dir", str(tmp_path / "o")])


def test_test_predictions_require_open_test_flag(fake_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "r2"
    train_ridge.main(["--pairs", str(fake_run / "pairs.csv"), "--seg-features", str(fake_run / "seg_features.csv"),
                      "--cls-probs", str(fake_run / "cls_probs.csv"), "--out-dir", str(out)])
    assert (out / "predictions_val.csv").exists() and not (out / "predictions_test.csv").exists()
    # Chưa mở test: target và mọi giá trị của t+1 ở các cặp test phải bị che trong features.csv.
    feats = pd.read_csv(out / "features.csv")
    test_rows = feats[feats["split"] == "test"]
    assert len(test_rows) > 0
    assert test_rows[["delta_area", "area_ratio_t1"]].isna().all().all()
    assert feats.loc[feats["split"] != "test", "delta_area"].notna().any()
```

### 4.6. Chạy thử trên dữ liệu GIẢ (máy, NB_cpu)

```python
# cell: NB_cpu (máy cá nhân)
from nckh.runcard import new_run_id
FAKE = '/tmp/fake_uq'
RUN = f"{ROOT}/runs/{new_run_id('p6_fake')}"
%cd {REPO}
!{PY} scripts/make_fake_uq.py --out-dir {FAKE} --participants 30 --seed 2026
!{PY} scripts/build_pairs.py --metadata {FAKE}/metadata.csv --column-map {FAKE}/fake_column_map.yaml --data-root {FAKE} --out-dir {RUN} --seed 2026
!{PY} scripts/infer_images.py --task seg --manifest {RUN}/uq_manifest.csv --data-root {FAKE} --out-dir {RUN} --fake
!{PY} scripts/infer_images.py --task cls --manifest {RUN}/uq_manifest.csv --data-root {FAKE} --out-dir {RUN} --fake
!{PY} scripts/train_ridge.py --pairs {RUN}/pairs.csv --seg-features {RUN}/seg_features.csv --cls-probs {RUN}/cls_probs.csv --out-dir {RUN}
```

**Kết quả mẫu trên dữ liệu GIẢ** (khi viết hướng dẫn, seed 2026; chỉ để bạn đối chiếu là code chạy đúng, **không** phải kết quả nghiên cứu):
- `make_fake_uq`: 142 ảnh, 30 participant, 48 lesion.
- `flow.json`: `excluded_by_reason = {missing_timestamp: 1, not_dermoscopy: 1, same_timestamp: 1}`; `n_pairs_by_split = {train: 65, val: 13, test: 13}`; `n_participants_by_split = {train: 22, val: 4, test: 4}`; Δt từ 150 đến 210 ngày.
- `infer_images --fake`: 142 ảnh OK, 0 lỗi (mỗi task).
- `train_ridge`: alpha được chọn và `val_mae_ridge` < `val_mae_baseline` (dữ liệu giả có quy luật tăng/giảm tuyến tính nên Ridge học được).

Phiên bản thư viện khác có thể làm các số MAE lệch nhẹ. Số cặp và số ảnh bị loại phải khớp chính xác.

### 4.7. Chạy với UQ thật (sau Go/No-Go)

Lấy metadata + ảnh UQ từ Drive về máy (chỉ khi checklist 3.1 cho phép lưu trên máy):

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/data/uq ~/nckh_drive/data/uq --progress
```

```python
# cell: NB_cpu (máy cá nhân)
UQ = f"{ROOT}/data/uq"
RUN = f"{ROOT}/runs/{new_run_id('p6_uq')}"
!cd {REPO} && {PY} scripts/build_pairs.py --metadata {UQ}/metadata.csv --column-map configs/uq_column_map.yaml --data-root {UQ} --out-dir {RUN} --seed 2026
print(Path(RUN).name)   # run_id: dùng ở lệnh rclone và cell NB_seg/NB_cls
```

Gửi `flow.json` và `pairs_excluded.csv` cho SV A **trước khi** chạy bước tiếp theo (kế hoạch: cặp bị loại được thống kê trước khi mở test).

Đẩy run lên Drive cho Colab:

```bash
# terminal VS Code (máy cá nhân)
rclone copy ~/nckh_drive/runs/<p6_uq_run_id> gdrive:NCKH_PanDerm/runs/<p6_uq_run_id> --progress
```

```python
# cell: NB_seg (Colab GPU)
RUN = f"{ROOT}/runs/<p6_uq_run_id>"
SEG_FT = f"{ROOT}/runs/<seg_main_run_id>/0/model_best_0.ckpt"
# Ảnh UQ nằm trên Drive; nếu nhiều ảnh, copy sang /content trước cho nhanh: !rsync -a {ROOT}/data/uq/ /content/uq/
!cd /content && {VENV}/bin/python {CODE}/scripts/infer_images.py --task seg --manifest {RUN}/uq_manifest.csv --data-root {ROOT}/data/uq --out-dir {RUN} --panderm-dir /content/PanDerm/segmentation --pretrained {CK} --finetuned {SEG_FT}
```

```python
# cell: NB_cls (Colab GPU)
RUN = f"{ROOT}/runs/<p6_uq_run_id>"
CLS_FT = f"{ROOT}/runs/<cls_main_run_id>/checkpoint-best.pth"
!cd /content && {VENV}/bin/python {CODE}/scripts/infer_images.py --task cls --manifest {RUN}/uq_manifest.csv --data-root {ROOT}/data/uq --out-dir {RUN} --panderm-dir /content/PanDerm/classification --finetuned {CLS_FT}
```

Kéo kết quả suy luận về máy:

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<p6_uq_run_id> ~/nckh_drive/runs/<p6_uq_run_id> --progress
```

```python
# cell: NB_cpu (máy cá nhân)
# Lần 1: chỉ train + val (chọn alpha, xem MAE val). Gửi ridge_summary.json cho SV A.
!cd {REPO} && {PY} scripts/train_ridge.py --pairs {RUN}/pairs.csv --seg-features {RUN}/seg_features.csv --cls-probs {RUN}/cls_probs.csv --out-dir {RUN}/ridge
```

```python
# cell: NB_cpu (máy cá nhân)
# Lần 2 — ĐÚNG MỘT LẦN, sau khi khóa protocol và SV A xác nhận. Script sẽ hỏi; gõ: mo test
!cd {REPO} && {PY} scripts/train_ridge.py --pairs {RUN}/pairs.csv --seg-features {RUN}/seg_features.csv --cls-probs {RUN}/cls_probs.csv --out-dir {RUN}/ridge_test --open-test
```

Lần 2 ghi vào thư mục mới (`ridge_test`), nên run lần 1 được giữ nguyên làm bằng chứng. Dữ liệu và seed giống nhau nên alpha và mô hình phải trùng lần 1: so `ridge_summary.json` hai lần.

### 4.8. Mức B: đánh giá trên mask thủ công (khi có audit P4)

```python
# cell: NB_cpu (máy cá nhân)
import joblib, numpy as np, pandas as pd
from PIL import Image
from nckh.features import build_feature_table, mask_features
from nckh.forecast import predict_delta
from nckh.metrics import regression_metrics

AUDIT = f"{ROOT}/data/uq/audit/consensus"                  # mask thống nhất <image_id>.png
manual = pd.DataFrame([{"image_id": p.stem, **mask_features(np.asarray(Image.open(p).convert('L')) > 127)}
                       for p in Path(AUDIT).glob("*.png")])
per_image = manual.merge(pd.read_csv(f"{RUN}/cls_probs.csv"), on="image_id", how="inner")
audit_pairs = pd.read_csv(f"{ROOT}/data/uq/audit/audit_pairs.csv")[pd.read_csv(f"{RUN}/pairs.csv").columns]
table = build_feature_table(audit_pairs, per_image)
ok = table[table.usable & (table.split == 'test')]
bundle = joblib.load(f"{RUN}/ridge/ridge.joblib")           # mô hình ĐÃ KHÓA ở mức A, không fit lại
print("n cặp:", len(ok))
print("Ridge   :", regression_metrics(ok.delta_area, predict_delta(bundle["model"], ok)))
print("Baseline:", regression_metrics(ok.delta_area, np.zeros(len(ok))))
```

Đồng thời so mask PanDerm với mask thủ công trên cùng ảnh audit (Dice/IoU/sai số tỷ lệ diện tích) bằng `annotator_agreement.py --dir-a {RUN}/masks_audit --dir-b {AUDIT}`. Trước đó, copy riêng mask PanDerm của các ảnh audit vào `masks_audit/`. Báo cáo mức B tách riêng, nêu rõ số cặp nhỏ.

### 4.9. Benchmark thời gian suy luận

`scripts/bench_inference.py` (P2) dùng được với checkpoint fine-tune: thêm `--finetuned SEG_FT` (hoặc `CLS_FT`). Chạy trên 50 ảnh UQ để ước lượng thời gian suy luận toàn bộ tập.

## 5. Test

```python
# cell: NB_cpu (máy cá nhân)
!cd {REPO} && {PY} -m pytest -q tests/test_pairs.py tests/test_features.py tests/test_forecast.py tests/test_pipeline_fake.py
```

Kỳ vọng: `47 passed` (13 + 8 + 20 + 6).

| Nhóm | Test tiêu biểu |
|---|---|
| Ghép cặp | chỉ ghép kề nhau; không nối lesion khác; Δt = 182,0 ngày cho 01/01→01/07/2020; trộn ngày thường và có múi giờ; 7 lý do loại (gồm `same_visit` < 1 ngày) |
| Đặc trưng | hình vuông 10×10 trên 100×100 → 0,01; hình tròn có circularity > 0,85; mask rỗng/1 pixel không lỗi; không có cột tương lai |
| Ridge | scaler chỉ học train; bảng alpha là MAE val; hòa thì chọn alpha lớn; clip [0, 1]; Δt sai bị từ chối; Ridge thắng baseline trên dữ liệu có quy luật |
| Đầu-cuối | flow đếm đúng 3 lỗi cố ý; cột dự báo đúng; không ai ở 2 split; `ridge.joblib` đúng cấu trúc; không có `--open-test` thì không ghi test và target test bị che |

## 6. Benchmark / đánh giá

**Bảng flow UQ** (lấy từ `flow.json` và `ridge_summary.json`):

| Bước | Ảnh | Participant | Lesion | Cặp |
|---|---:|---:|---:|---:|
| Metadata ban đầu | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | — |
| Loại `missing_lesion_id` / `unreadable_image` / `not_dermoscopy` / `missing_timestamp` / `participant_mismatch` / `same_timestamp` / `same_visit` | [điền sau khi chạy] | — | — | — |
| Cặp liên tiếp hợp lệ (train / val / test) | — | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| Loại sau suy luận (`empty_mask` / `missing_inference` / `nonfinite_feature`) | — | — | — | [điền sau khi chạy] |
| Cặp dùng cho Ridge (train / val / test) | — | [điền sau khi chạy] | — | [điền sau khi chạy] |

| Hạng mục | Giá trị |
|---|---|
| Alpha được chọn (val) | [điền sau khi chạy] |
| MAE val: Ridge / baseline | [điền sau khi chạy] |
| Thời gian suy luận UQ seg / cls | [điền sau khi chạy] |

Kết quả test (MAE/RMSE/bias + CI) tính ở **P7** bằng `evaluate_forecast.py`.

## 7. Lỗi thường gặp (máy / Colab extension)

| Triệu chứng | Cách xử lý |
|---|---|
| `AssertionError: Chưa mount Drive` | `Ctrl+Shift+P` → *Colab: Mount Google Drive to Server...*, chạy lại cell setup |
| Kernel Colab mất kết nối / server bị thu hồi | *Select Kernel → Colab → New Colab Server*, chạy lại cell setup + venv; dữ liệu trên Drive vẫn còn |
| Colab chạy code cũ | Ở máy `git push`, chạy lại cell setup (có `git pull`) |
| `userdata.get` / `files.upload` lỗi | Chưa hỗ trợ trong extension; không dùng, file đi qua Drive |
| `rclone` báo `couldn't fetch token` | `rclone config reconnect gdrive:` |
| Colab báo không thấy `uq_manifest.csv` / ở máy không thấy `seg_features.csv` | Chưa chạy lệnh rclone đẩy lên / kéo về ở mục 4.7 |
| `SystemExit: Chưa điền tên cột gốc …` | Điền `configs/uq_column_map.yaml` (P3 mục 4.7) |
| `ValueError: Thiếu cột trong metadata.csv` | Tên cột trong YAML sai so với file thật |
| Rất nhiều `missing_timestamp` | Ngày không theo ISO-8601 (ví dụ `05/01/2020`): đặt `captured_at_format: "%d/%m/%Y"` (hoặc mẫu đúng) trong `configs/uq_column_map.yaml` sau khi SV A xác nhận định dạng |
| Nhiều `same_visit` | Bình thường nếu mỗi buổi khám chụp nhiều ảnh; báo số này trong bảng flow |
| `Không đủ cặp dùng được: train=…, val=0` | Quá ít participant có ≥ 2 lần chụp; báo SV A (tiêu chí No-Go) |
| Suy luận UQ rất chậm | Copy ảnh sang `/content` trước; kiểm tra server Colab có GPU |
| Nhiều `empty_mask` | Mask PanDerm kém trên UQ (chuyển miền): báo trong audit P4, không tự hạ ngưỡng |

## 8. Checklist bàn giao cho SV A

- [ ] Văn bản Go/No-Go đã có trước khi UQ lên Drive.
- [ ] `pytest` phần P6: 47 passed; chạy được pipeline giả.
- [ ] `flow.json`, `pairs_excluded.csv`, `features.csv` gửi SV A **trước** khi mở test.
- [ ] `ridge.joblib`, `alpha_selection.csv`, `predictions_val.csv`, `run_card.json` trong `runs/`.
- [ ] Nhật ký quyết định ghi thời điểm chạy `--open-test` (một lần).
- [ ] Mức B (nếu có) báo cáo tách riêng, ghi rõ số cặp.
