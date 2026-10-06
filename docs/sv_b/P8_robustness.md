# P8 — Độ bền trước ảnh suy giảm và tiền xử lý tương phản

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 8):** đo xem segmentation, phân loại tham khảo và dự báo Δa thay đổi bao nhiêu khi ảnh đầu vào bị: lệch sáng, mờ, lệch cân bằng trắng, có lông che. Mỗi phép thử so **cùng ảnh** ở bản sạch và bản suy giảm, **cùng tham chiếu**, bằng paired bootstrap. Ngoài ra thử Ben Graham/Gamma như tiền xử lý phụ, **chỉ chọn trên val**.

| Đầu vào | Đầu ra |
|---|---|
| Checkpoint đã khóa (P5a, P5b), `ridge.joblib` (P6) | `runs/<p8_run>/robustness.json` dạng `{mức: {metric: {estimate, ci_low, ci_high, …}}}` |
| ISIC 2018 test, ISIC 2017 test, cặp UQ test | `seg_<mức>.csv`, `forecast_<mức>.csv` (từng ảnh/cặp), `enhancement_val.json` |

## 2. Chạy ở đâu

| Việc | Notebook | Ghi chú |
|---|---|---|
| Tạo `degrade.py`, `make_degraded.py`, `evaluate_robustness.py`, test | NB_cpu | |
| Sinh ảnh suy giảm | Notebook có GPU đang dùng (CPU làm) | ~0,25 s/ảnh 3000×2000 cho mức lông; các mức khác nhanh hơn |
| Suy luận lại PanDerm trên ảnh suy giảm | NB_seg, NB_cls | 10 mức × số ảnh test × ms/ảnh (P2) |
| `evaluate_robustness.py` | Bất kỳ (CPU) | Vài giây mỗi mức |

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2. Phần CPU (sinh ảnh, đánh giá, ghi JSON) đã chạy pytest và chạy thử đầu-cuối trên dữ liệu giả.

## 3. Giải thích

### 3.1. Các phép suy giảm (đúng bảng 8.1 của kế hoạch)

| Tên mức | Công thức | Ghi chú |
|---|---|---|
| `brightness_0.8`, `brightness_1.2` | I' = clip(α·I, 0, 255) | Thiếu sáng / dư sáng |
| `blur_1.0`, `blur_1.5` | Gaussian σ = 1,0 / 1,5 pixel | Pillow `GaussianBlur(radius=σ)` (tham số radius của Pillow chính là độ lệch chuẩn). σ tính trên ảnh **gốc**, trước khi model resize về 224 |
| `wb_r1.1_b0.9`, `wb_r0.9_b1.1`, `wb_r1.2_b0.8`, `wb_r0.8_b1.2` | R' = g_R·R, B' = g_B·B, G giữ nguyên | Lệch cân bằng trắng ±10%, ±20%, theo cả hai chiều ấm/lạnh |
| `hair_0.01`, `hair_0.03` | Vẽ đường cong Bézier bậc 2 màu nâu/đen, rộng 1–3 px, đến khi che 1% / 3% diện tích (sai số ≤ 0,5 điểm %) | **Vật cản tổng hợp**, không phải lông thật. Độ che thực tế được ghi vào `hair_coverage` |

Seed của từng ảnh = 2026 + CRC32(image_id), nên chạy lại hay chạy song song đều ra đúng ảnh cũ.

### 3.2. Quy trình không rò rỉ (kế hoạch 8.2) → lệnh cụ thể

| Bước | Thực hiện |
|---|---|
| 1. Chia tập ảnh gốc trước | Split đã khóa ở P3/P6 |
| 2. Tạo biến thể **sau** khi chia | `make_degraded.py` đọc manifest có cột `split` |
| 3. Chỉ biến thể của ảnh **test**, không đưa sang train | `make_degraded.py` từ chối mọi split khác `test` (thoát mã 2), trừ khi có `--allow-non-test` cho thí nghiệm tiền xử lý trên val |
| 4. Giữ target và mask tham chiếu gốc | `evaluate_robustness.py`: seg so với GT gốc; forecast chỉ thay đặc trưng của ảnh t, target Δa lấy từ mask **sạch** |
| 5. Đánh giá theo cặp sạch/suy giảm | Paired bootstrap: cùng mẫu ảnh/participant cho cả hai |
| 6. Báo mức giảm tuyệt đối + CI | `delta_dice`, `delta_iou`, `delta_macro_f1`, `delta_mae`, `delta_rmse` = (suy giảm − sạch) |
| 7. Không chạy tích chéo tiền xử lý × suy giảm | Mỗi lần chỉ một mức; tiền xử lý thử riêng trên val (mục 3.4) |

**Dấu của hiệu:** `delta_dice`/`delta_macro_f1` **âm** nghĩa là kém đi; `delta_mae` **dương** nghĩa là kém đi.

### 3.3. Segmentation robustness đo khác P5a một chút

Ở P5a, Dice tính bằng code upstream ở 224×224. Ở P8, Dice cho **cả** ảnh sạch lẫn ảnh suy giảm đều tính bằng `nckh.infer` + `dice_iou` ở **độ phân giải gốc**, so với GT gốc. Vì vậy "Dice sạch" ở P8 có thể lệch nhẹ so với P5a. Điều quan trọng là hai vế của phép so sánh dùng cùng một cách đo, và hiệu được báo theo cặp.

### 3.4. Ben Graham / Gamma (thí nghiệm phụ, kế hoạch 8.3)

- Cấu hình chính vẫn là RGB + chuẩn hóa PanDerm.
- Thử `ben_graham_10` (σ = 10; I' = clip(4I − 4·G_σ*I + 128)), `gamma_0.8`, `gamma_1.2` **trên val** của ISIC 2018 (100 ảnh) với checkpoint đã khóa.
- Chỉ áp dụng lúc suy luận (không fine-tune lại). Đây là giới hạn cần ghi: model được train trên ảnh RGB thường, nên tiền xử lý chỉ ở bước test cũng là một dạng lệch miền.
- Nếu một biến thể làm Dice val tăng (CI của hiệu > 0), khóa **tối đa một** biến thể rồi đánh giá test **một lần**. Nếu không biến thể nào tốt hơn: báo kết quả âm tính, không bỏ khỏi báo cáo.
- Gabor nằm ngoài phạm vi cấu hình chính, không ghép kênh thứ tư.

## 4. Code

### 4.1. `src/nckh/degrade.py`

```python
# file: src/nckh/degrade.py
"""Suy giảm ảnh có kiểm soát (P8) và tiền xử lý tương phản phụ (Ben Graham, Gamma).

Mọi hàm nhận/trả np.uint8 HxWx3 và KHÔNG sửa ảnh đầu vào. Mức tham số lấy đúng bảng 8.1 của kế hoạch.
"""
import logging
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

logger = logging.getLogger(__name__)

HAIR_COLORS = ((25, 18, 12), (45, 30, 20), (70, 48, 30))
HAIR_TOLERANCE = 0.005


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def adjust_brightness(img: np.ndarray, alpha: float) -> np.ndarray:
    return _to_uint8(img.astype(np.float32) * alpha)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # Với Pillow, tham số radius của GaussianBlur chính là độ lệch chuẩn σ (pixel).
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=sigma)))


def white_balance(img: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray:
    out = img.astype(np.float32)
    out[..., 0] *= red_gain
    out[..., 2] *= blue_gain
    return _to_uint8(out)


def _bezier(rng: np.random.Generator, h: int, w: int) -> list[tuple[float, float]]:
    start = rng.uniform([0, 0], [w, h])
    angle = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(0.2, 0.6) * min(h, w)
    end = start + length * np.array([np.cos(angle), np.sin(angle)])
    ctrl = (start + end) / 2 + rng.normal(0, length / 4, 2)  # điểm điều khiển lệch → sợi cong
    t = np.linspace(0, 1, 40)[:, None]
    pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
    return [tuple(p) for p in pts]


def add_hair(img: np.ndarray, coverage: float, seed: int) -> tuple[np.ndarray, float]:
    """Vẽ sợi lông tổng hợp (đường cong bậc 2, rộng 1–3 px) cho tới khi che ≈ coverage diện tích ảnh."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = Image.fromarray(img.copy())
    covered = np.zeros((h, w), bool)
    target, limit, n_covered, attempts = coverage * h * w, (coverage + HAIR_TOLERANCE) * h * w, 0, 0
    while n_covered < target and attempts < 5000:
        attempts += 1
        pts = np.array(_bezier(rng, h, w))
        color = HAIR_COLORS[rng.integers(len(HAIR_COLORS))]
        # Chỉ vẽ trong khung bao của nét: nhanh hơn nhiều lần so với tạo lớp mask cỡ cả ảnh mỗi nét.
        x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int) - 3, 0)
        x1, y1 = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + 4, [w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        local = [tuple(p) for p in pts - [x0, y0]]
        for width in (int(rng.integers(1, 4)), 1):
            stroke = Image.new("L", (int(x1 - x0), int(y1 - y0)), 0)
            ImageDraw.Draw(stroke).line(local, fill=255, width=width)
            region = covered[y0:y1, x0:x1]
            added = int(((np.asarray(stroke) > 0) & ~region).sum())
            # Không vượt quá mục tiêu + dung sai; vượt thì thử lại nét mảnh hơn, vẫn vượt thì bỏ nét này.
            if n_covered + added <= limit:
                out.paste(color, (int(x0), int(y0)), mask=stroke)
                region |= np.asarray(stroke) > 0
                n_covered += added
                break
    if n_covered < target:
        logger.warning("Chỉ che được %.4f < %.4f sau %d lần vẽ", n_covered / (h * w), coverage, attempts)
    return np.asarray(out), float(covered.mean())


def ben_graham(img: np.ndarray, sigma: float) -> np.ndarray:
    # I_BG = clip(4·I − 4·(G_σ * I) + 128): trừ nền mờ để làm nổi chi tiết cục bộ (đề cương 6.4).
    x = img.astype(np.float32)
    blurred = ndimage.gaussian_filter(x, sigma=(sigma, sigma, 0))
    return _to_uint8(4 * x - 4 * blurred + 128)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    return _to_uint8(255.0 * (img.astype(np.float32) / 255.0) ** gamma)


DEGRADATION_LEVELS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "brightness_0.8": lambda im, s: adjust_brightness(im, 0.8),
    "brightness_1.2": lambda im, s: adjust_brightness(im, 1.2),
    "blur_1.0": lambda im, s: gaussian_blur(im, 1.0),
    "blur_1.5": lambda im, s: gaussian_blur(im, 1.5),
    "wb_r1.1_b0.9": lambda im, s: white_balance(im, 1.1, 0.9),
    "wb_r0.9_b1.1": lambda im, s: white_balance(im, 0.9, 1.1),
    "wb_r1.2_b0.8": lambda im, s: white_balance(im, 1.2, 0.8),
    "wb_r0.8_b1.2": lambda im, s: white_balance(im, 0.8, 1.2),
    "hair_0.01": lambda im, s: add_hair(im, 0.01, s)[0],
    "hair_0.03": lambda im, s: add_hair(im, 0.03, s)[0],
}

# Tiền xử lý tương phản: chỉ thử trên val để chọn tối đa MỘT biến thể (kế hoạch 8.3).
ENHANCEMENTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ben_graham_10": lambda im: ben_graham(im, 10.0),
    "gamma_0.8": lambda im: gamma_correct(im, 0.8),
    "gamma_1.2": lambda im: gamma_correct(im, 1.2),
}
```

### 4.2. `scripts/make_degraded.py`

```python
# file: scripts/make_degraded.py
"""Sinh ảnh suy giảm (hoặc ảnh tiền xử lý tương phản) cho MỘT mức, từ ảnh của MỘT split.

Mặc định chỉ cho split test (đánh giá độ bền sau khi khóa cấu hình). Ben Graham/Gamma được chọn trên val
nên phải thêm --allow-non-test. Ảnh ghi dạng PNG (không nén mất dữ liệu) để không cộng thêm nhiễu JPEG.
Ví dụ: python scripts/make_degraded.py --manifest $ROOT/data/manifests/isic2018_seg.csv --data-root /content/data \
           --split test --level blur_1.5 --out-dir /content/degraded/blur_1.5
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nckh.degrade import DEGRADATION_LEVELS, ENHANCEMENTS, add_hair


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--path-col", default="image_path")
    ap.add_argument("--id-col", default="image_id")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--level", required=True, choices=list(DEGRADATION_LEVELS) + list(ENHANCEMENTS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow-non-test", action="store_true")
    args = ap.parse_args(argv)

    if args.split != "test" and not args.allow_non_test:
        print("Chỉ tạo ảnh suy giảm cho split test; dùng --allow-non-test khi chọn tiền xử lý trên val.", file=sys.stderr)
        raise SystemExit(2)
    rows = pd.read_csv(args.manifest)
    rows = rows[rows["split"] == args.split]
    if "exclude_reason" in rows:
        rows = rows[rows["exclude_reason"].fillna("") == ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for image_id, rel in zip(rows[args.id_col].astype(str), rows[args.path_col]):
        img = np.asarray(Image.open(args.data_root / rel).convert("RGB"))
        # Seed theo từng ảnh (không theo thứ tự duyệt) → chạy lại/chạy song song vẫn ra đúng ảnh cũ.
        seed = args.seed + zlib.crc32(image_id.encode()) % 2**31
        coverage = np.nan
        if args.level.startswith("hair_"):
            res, coverage = add_hair(img, float(args.level.split("_")[1]), seed)
        elif args.level in DEGRADATION_LEVELS:
            res = DEGRADATION_LEVELS[args.level](img, seed)
        else:
            res = ENHANCEMENTS[args.level](img)
        Image.fromarray(res).save(args.out_dir / f"{image_id}.png")
        out.append({"image_id": image_id, "image_path": f"{image_id}.png", "source_path": rel, "split": args.split,
                    "level": args.level, "hair_coverage": coverage})
    pd.DataFrame(out).to_csv(args.out_dir / "degraded_manifest.csv", index=False)
    print(f"{args.level}: {len(out)} ảnh ({args.split}) → {args.out_dir}")


if __name__ == "__main__":
    main()
```

### 4.3. `scripts/evaluate_robustness.py`

```python
# file: scripts/evaluate_robustness.py
"""So kết quả trên ảnh sạch với ảnh suy giảm (cùng ảnh, cùng tham chiếu) bằng paired bootstrap; ghi robustness.json.

  seg      : Dice/IoU so với mask GT gốc        → delta_dice, delta_iou (theo ảnh)
  cls      : Macro-F1 so với nhãn gốc           → delta_macro_f1 (theo ảnh)
  forecast : chỉ ảnh t bị suy giảm, target giữ  → delta_mae, delta_rmse (theo participant)
Hiệu luôn là (suy giảm − sạch): Dice/F1 âm = kém đi; MAE dương = kém đi.
"""
import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN
from nckh.forecast import predict_delta
from nckh.metrics import dice_iou, paired_bootstrap_diff

P_COLS = ["p_mel", "p_nev", "p_sk"]


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def seg_delta(clean_dir: Path, degraded_dir: Path, gt_dir: Path, image_ids: list[str], n_boot: int = 2000,
              seed: int = 2026) -> tuple[dict, pd.DataFrame]:
    missing = [i for i in image_ids if not (degraded_dir / f"{i}.png").exists() or not (clean_dir / f"{i}.png").exists()]
    if missing:
        raise ValueError(f"Thiếu mask sạch/suy giảm cho {len(missing)} ảnh, ví dụ {missing[:5]}")
    rows = []
    for i in image_ids:
        gt = _mask(gt_dir / f"{i}_segmentation.png")
        dc, ic = dice_iou(_mask(clean_dir / f"{i}.png"), gt)
        dd, idg = dice_iou(_mask(degraded_dir / f"{i}.png"), gt)
        rows.append({"image_id": i, "dice_clean": dc, "dice_degraded": dd, "iou_clean": ic, "iou_degraded": idg})
    df = pd.DataFrame(rows)
    res = {f"delta_{m}": paired_bootstrap_diff(df, "image_id", lambda d, m=m: d[f"{m}_degraded"].mean(),
                                               lambda d, m=m: d[f"{m}_clean"].mean(), n_boot=n_boot, seed=seed)
           for m in ("dice", "iou")}
    res.update({"dice_clean": float(df.dice_clean.mean()), "dice_degraded": float(df.dice_degraded.mean()), "n": len(df)})
    return res, df


def cls_delta(clean: pd.DataFrame, degraded: pd.DataFrame, labels: pd.DataFrame, n_boot: int = 2000,
              seed: int = 2026) -> dict:
    df = labels.merge(clean, on="image_id").merge(degraded, on="image_id", suffixes=("_clean", "_degraded"))
    if len(df) != len(labels):
        raise ValueError(f"Chỉ ghép được {len(df)}/{len(labels)} ảnh giữa nhãn, sạch, suy giảm")

    def macro_f1(kind: str) -> Callable[[pd.DataFrame], float]:
        cols = [f"{c}_{kind}" for c in P_COLS]
        return lambda d: float(f1_score(d["label"], d[cols].to_numpy().argmax(axis=1), labels=[0, 1, 2],
                                        average="macro", zero_division=0))

    return {"delta_macro_f1": paired_bootstrap_diff(df, "image_id", macro_f1("degraded"), macro_f1("clean"),
                                                    n_boot=n_boot, seed=seed),
            "macro_f1_clean": macro_f1("clean")(df), "macro_f1_degraded": macro_f1("degraded")(df), "n": len(df)}


def forecast_delta(features: pd.DataFrame, seg_degraded: pd.DataFrame, cls_degraded: pd.DataFrame, bundle: dict,
                   split: str = "test", n_boot: int = 2000, seed: int = 2026) -> tuple[dict, pd.DataFrame]:
    clean = features[features["usable"].astype(bool) & (features["split"] == split)].reset_index(drop=True)
    deg_t = seg_degraded.merge(cls_degraded, on="image_id").rename(columns={
        "image_id": "image_id_t", "area_ratio": "area_ratio_t", "circularity": "circularity_t",
        "eccentricity": "eccentricity_t", "p_mel": "p_mel_t", "p_nev": "p_nev_t", "p_sk": "p_sk_t"})
    t_cols = [c for c in FEATURE_COLUMNS if c != "delta_days"]
    # Chỉ thay đặc trưng của ảnh t; Δt và target Δa (từ mask sạch) giữ nguyên → đo đúng tác động của đầu vào.
    degraded = clean.drop(columns=t_cols).merge(deg_t[["image_id_t", *t_cols]], on="image_id_t", how="left")
    ok = np.isfinite(degraded[list(FEATURE_COLUMNS)].to_numpy(float)).all(axis=1)
    per_pair = clean.loc[ok, ["participant_id", "image_id_t", TARGET_COLUMN]].copy()
    per_pair["pred_clean"] = predict_delta(bundle["model"], clean[ok])
    per_pair["pred_degraded"] = predict_delta(bundle["model"], degraded[ok])

    def err(col: str, power: int) -> Callable[[pd.DataFrame], float]:
        return lambda d: float((np.abs(d[col] - d[TARGET_COLUMN]) ** power).mean() ** (1 / power))

    res = {"delta_mae": paired_bootstrap_diff(per_pair, "participant_id", err("pred_degraded", 1), err("pred_clean", 1),
                                              n_boot=n_boot, seed=seed),
           "delta_rmse": paired_bootstrap_diff(per_pair, "participant_id", err("pred_degraded", 2), err("pred_clean", 2),
                                               n_boot=n_boot, seed=seed),
           "mae_clean": err("pred_clean", 1)(per_pair), "mae_degraded": err("pred_degraded", 1)(per_pair),
           "n_pairs": int(ok.sum()),
           # Ảnh t suy giảm làm mask rỗng → không dự báo được; báo riêng thay vì âm thầm bỏ.
           "n_unpredictable_degraded": int((~ok).sum())}
    return res, per_pair


def update_robustness_json(path: Path, level: str, metrics: dict) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(level, {}).update(metrics)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    import joblib

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["seg", "cls", "forecast"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--robustness-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True, help="seg: thư mục masks sạch | cls: cls_probs.csv sạch | forecast: features.csv")
    ap.add_argument("--degraded", type=Path, required=True, help="seg: thư mục masks suy giảm | cls/forecast: thư mục kết quả suy giảm")
    ap.add_argument("--gt-dir", type=Path, help="seg: thư mục *_segmentation.png")
    ap.add_argument("--labels-csv", type=Path, help="cls: isic2017_cls.csv")
    ap.add_argument("--ridge", type=Path, help="forecast: ridge.joblib")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "seg":
        ids = sorted(p.stem for p in args.degraded.glob("*.png"))
        res, per = seg_delta(args.clean, args.degraded, args.gt_dir, ids)
        per.to_csv(args.out_dir / f"seg_{args.level}.csv", index=False)
        keys = ("delta_dice", "delta_iou")
    elif args.task == "cls":
        lab = pd.read_csv(args.labels_csv)
        lab = lab[lab["split"] == "test"].assign(image_id=lambda d: d["image"].map(lambda s: Path(s).stem))[["image_id", "label"]]
        res = cls_delta(pd.read_csv(args.clean), pd.read_csv(args.degraded / "cls_probs.csv"), lab)
        keys = ("delta_macro_f1",)
    else:
        res, per = forecast_delta(pd.read_csv(args.clean), pd.read_csv(args.degraded / "seg_features.csv"),
                                  pd.read_csv(args.degraded / "cls_probs.csv"), joblib.load(args.ridge))
        per.to_csv(args.out_dir / f"forecast_{args.level}.csv", index=False)
        keys = ("delta_mae", "delta_rmse")
    update_robustness_json(args.robustness_json, args.level, {k: res[k] for k in keys})
    print(json.dumps({k: v for k, v in res.items()}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
```

### 4.4. Test

```python
# file: tests/test_degrade.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nckh.degrade import (
    DEGRADATION_LEVELS,
    ENHANCEMENTS,
    add_hair,
    adjust_brightness,
    ben_graham,
    gamma_correct,
    gaussian_blur,
    white_balance,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _img(seed: int = 0, size: int = 256) -> np.ndarray:
    return np.random.default_rng(seed).integers(30, 220, (size, size, 3), dtype=np.uint8)


def test_levels_are_the_protocol_ones() -> None:
    assert list(DEGRADATION_LEVELS) == [
        "brightness_0.8", "brightness_1.2", "blur_1.0", "blur_1.5",
        "wb_r1.1_b0.9", "wb_r0.9_b1.1", "wb_r1.2_b0.8", "wb_r0.8_b1.2", "hair_0.01", "hair_0.03"]
    assert list(ENHANCEMENTS) == ["ben_graham_10", "gamma_0.8", "gamma_1.2"]


@pytest.mark.parametrize("name", list(DEGRADATION_LEVELS) + list(ENHANCEMENTS))
def test_all_levels_dtype_shape_range(name: str) -> None:
    img = _img()
    fn = DEGRADATION_LEVELS.get(name)
    out = fn(img, 0) if fn else ENHANCEMENTS[name](img)
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_input_not_mutated() -> None:
    img = _img()
    before = img.copy()
    for fn in DEGRADATION_LEVELS.values():
        fn(img, 0)
    assert np.array_equal(img, before)


def test_brightness_clips() -> None:
    out = adjust_brightness(np.full((2, 2, 3), 250, np.uint8), 1.2)
    assert (out == 255).all()


def test_blur_reduces_variance() -> None:
    img = _img()
    assert gaussian_blur(img, 1.5).astype(float).var() < img.astype(float).var()


def test_white_balance_green_unchanged() -> None:
    img = np.full((2, 2, 3), 100, np.uint8)
    out = white_balance(img, 1.2, 0.8)
    assert out[0, 0].tolist() == [120, 100, 80]


@pytest.mark.parametrize("coverage", [0.01, 0.03])
@pytest.mark.parametrize("seed", range(5))
def test_hair_coverage_within_tolerance(coverage: float, seed: int) -> None:
    _, actual = add_hair(_img(seed), coverage, seed)
    assert abs(actual - coverage) <= 0.005


def test_hair_deterministic_by_seed() -> None:
    a, _ = add_hair(_img(), 0.03, 7)
    b, _ = add_hair(_img(), 0.03, 7)
    c, _ = add_hair(_img(), 0.03, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_ben_graham_flat_image_goes_gray() -> None:
    out = ben_graham(np.full((64, 64, 3), 77, np.uint8), 10)
    assert np.abs(out.astype(int) - 128).max() <= 1


def test_gamma_identity_at_one() -> None:
    img = _img()
    assert np.array_equal(gamma_correct(img, 1.0), img)


def test_make_degraded_refuses_train_split(tmp_path: Path) -> None:
    import make_degraded

    pd.DataFrame({"image_id": ["a"], "image_path": ["a.jpg"], "split": ["train"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(SystemExit) as exc:
        make_degraded.main(["--manifest", str(tmp_path / "m.csv"), "--data-root", str(tmp_path), "--split", "train",
                            "--level", "blur_1.0", "--out-dir", str(tmp_path / "o")])
    assert exc.value.code == 2


def test_make_degraded_writes_images_and_manifest(tmp_path: Path) -> None:
    import make_degraded

    Image.fromarray(_img()).save(tmp_path / "a.jpg")
    pd.DataFrame({"image_id": ["a"], "image_path": ["a.jpg"], "split": ["test"]}).to_csv(tmp_path / "m.csv", index=False)
    make_degraded.main(["--manifest", str(tmp_path / "m.csv"), "--data-root", str(tmp_path), "--split", "test",
                        "--level", "hair_0.03", "--out-dir", str(tmp_path / "o")])
    dm = pd.read_csv(tmp_path / "o" / "degraded_manifest.csv")
    assert dm.loc[0, "image_path"] == "a.png" and abs(dm.loc[0, "hair_coverage"] - 0.03) <= 0.005
    assert (tmp_path / "o" / "a.png").exists()
```

```python
# file: tests/test_robustness.py
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_robustness import cls_delta, forecast_delta, seg_delta, update_robustness_json  # noqa: E402

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from nckh.forecast import fit_forecaster  # noqa: E402


def _png(path: Path, box: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((20, 20), np.uint8)
    m[: box[0], : box[1]] = 255
    Image.fromarray(m).save(path)


def test_seg_delta_detects_worse_masks(tmp_path: Path) -> None:
    ids = [f"ISIC_{i}" for i in range(12)]
    for i in ids:
        _png(tmp_path / "gt" / f"{i}_segmentation.png", (10, 10))
        _png(tmp_path / "clean" / f"{i}.png", (10, 10))
        _png(tmp_path / "deg" / f"{i}.png", (10, 5))
    res, per_image = seg_delta(tmp_path / "clean", tmp_path / "deg", tmp_path / "gt", ids, n_boot=100)
    assert per_image["dice_clean"].eq(1.0).all()
    assert res["delta_dice"]["estimate"] == pytest.approx(2 * 50 / 150 - 1)
    assert res["delta_dice"]["ci_high"] < 0


def test_cls_delta_zero_when_identical(tmp_path: Path) -> None:
    probs = pd.DataFrame({"image_id": [f"I{i}" for i in range(9)], "p_mel": [0.8, 0.1, 0.1] * 3,
                          "p_nev": [0.1, 0.8, 0.1] * 3, "p_sk": [0.1, 0.1, 0.8] * 3})
    labels = pd.DataFrame({"image_id": probs["image_id"], "label": [0, 1, 2] * 3})
    res = cls_delta(probs, probs, labels, n_boot=50)
    assert res["delta_macro_f1"]["estimate"] == 0.0


def _forecast_fixture(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(0)
    feats = pd.DataFrame({c: rng.uniform(0.1, 0.9, n) for c in FEATURE_COLUMNS})
    feats["delta_days"] = rng.uniform(150, 210, n)
    feats[TARGET_COLUMN] = 0.05 * feats["area_ratio_t"] - 0.02
    feats["participant_id"] = [f"P{i // 2}" for i in range(n)]
    feats["image_id_t"] = [f"T{i}" for i in range(n)]
    feats["split"], feats["usable"] = "test", True
    bundle = {"model": fit_forecaster(feats, 0.01)}
    seg_deg = pd.DataFrame({"image_id": feats["image_id_t"], "area_ratio": feats["area_ratio_t"] * 1.5,
                            "circularity": feats["circularity_t"], "eccentricity": feats["eccentricity_t"], "empty": False})
    cls_deg = pd.DataFrame({"image_id": feats["image_id_t"], "p_mel": feats["p_mel_t"], "p_nev": feats["p_nev_t"],
                            "p_sk": feats["p_sk_t"]})
    return feats, seg_deg, cls_deg, bundle


def test_forecast_delta_keeps_target_and_worsens_mae() -> None:
    feats, seg_deg, cls_deg, bundle = _forecast_fixture()
    res, per_pair = forecast_delta(feats, seg_deg, cls_deg, bundle, n_boot=100)
    assert np.allclose(per_pair[TARGET_COLUMN], feats[TARGET_COLUMN])   # target giữ nguyên bản sạch
    assert res["delta_mae"]["estimate"] > 0


def test_update_robustness_json_merges_levels(tmp_path: Path) -> None:
    path = tmp_path / "robustness.json"
    update_robustness_json(path, "blur_1.0", {"delta_dice": {"estimate": -0.1, "ci_low": -0.2, "ci_high": 0.0}})
    update_robustness_json(path, "blur_1.0", {"delta_mae": {"estimate": 0.01, "ci_low": 0.0, "ci_high": 0.02}})
    update_robustness_json(path, "hair_0.03", {"delta_dice": {"estimate": -0.3, "ci_low": -0.4, "ci_high": -0.2}})
    data = json.loads(path.read_text())
    assert set(data) == {"blur_1.0", "hair_0.03"} and set(data["blur_1.0"]) == {"delta_dice", "delta_mae"}
```

### 4.5. Segmentation: ISIC 2018 test, vòng lặp theo mức (NB_seg)

Chuẩn bị: pull code ở NB_cpu (P2 4.1a), cell mở đầu 4.1b, `venv_seg`, patch, dữ liệu ISIC 2018 (P2/P3).

```python
# cell: NB_seg
from nckh.runcard import new_run_id
P8 = f"{ROOT}/runs/{new_run_id('p8_seg')}"
SEG_FT = f"{ROOT}/runs/<seg_main_run_id>/0/model_best_0.ckpt"
MAN = f"{ROOT}/data/manifests/isic2018_seg.csv"
SEG = f"--task seg --panderm-dir /content/PanDerm/segmentation --pretrained {CK} --finetuned {SEG_FT}"
# 1) Mask trên ảnh SẠCH của test (một lần)
!python -c "import pandas as pd; m=pd.read_csv('{MAN}'); m[(m.split=='test')&m.exclude_reason.isna()].to_csv('/content/isic18_test.csv', index=False)"
!cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {SEG} --manifest /content/isic18_test.csv --data-root /content/data --out-dir /content/clean_seg
```

```python
# cell: NB_seg
LEVELS = ["brightness_0.8", "brightness_1.2", "blur_1.0", "blur_1.5", "wb_r1.1_b0.9", "wb_r0.9_b1.1",
          "wb_r1.2_b0.8", "wb_r0.8_b1.2", "hair_0.01", "hair_0.03"]
for lv in LEVELS:
    D = f"/content/degraded/{lv}"
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/make_degraded.py --manifest /content/isic18_test.csv --data-root /content/data --split test --level {lv} --out-dir {D}
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {SEG} --manifest {D}/degraded_manifest.csv --data-root {D} --out-dir {D}/out
    !{VENV}/bin/python {ROOT}/nckh/scripts/evaluate_robustness.py seg --level {lv} --robustness-json {P8}/robustness.json --out-dir {P8} --clean /content/clean_seg/masks --degraded {D}/out/masks --gt-dir /content/data/ISIC2018/Test_GroundTruth | grep -E '"dice_(clean|degraded)"'
    !rm -rf {D}   # xoá ngay để không đầy đĩa /content
```

Nếu runtime ngắt giữa vòng lặp: `robustness.json` đã lưu các mức xong. Bỏ những mức đó khỏi `LEVELS` rồi chạy tiếp.

### 4.6. Phân loại: ISIC 2017 test (NB_cls)

```python
# cell: NB_cls
from nckh.runcard import new_run_id
from nckh.degrade import DEGRADATION_LEVELS
LEVELS = list(DEGRADATION_LEVELS)
P8C = f"{ROOT}/runs/{new_run_id('p8_cls')}"
CLS_FT = f"{ROOT}/runs/<cls_main_run_id>/checkpoint-best.pth"
LAB = f"{ROOT}/data/manifests/isic2017_cls.csv"
!python -c "import pandas as pd; from pathlib import Path; m=pd.read_csv('{LAB}'); m=m[m.split=='test']; m.assign(image_id=m.image.map(lambda s: Path(s).stem)).to_csv('/content/isic17_test.csv', index=False)"
CLS = f"--task cls --panderm-dir /content/PanDerm/classification --finetuned {CLS_FT}"
!cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {CLS} --manifest /content/isic17_test.csv --path-col image --data-root /content/data/ISIC2017 --out-dir /content/clean_cls
for lv in LEVELS:
    D = f"/content/degraded17/{lv}"
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/make_degraded.py --manifest /content/isic17_test.csv --path-col image --data-root /content/data/ISIC2017 --split test --level {lv} --out-dir {D}
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {CLS} --manifest {D}/degraded_manifest.csv --data-root {D} --out-dir {D}/out
    !{VENV}/bin/python {ROOT}/nckh/scripts/evaluate_robustness.py cls --level {lv} --robustness-json {P8C}/robustness.json --out-dir {P8C} --clean /content/clean_cls/cls_probs.csv --degraded {D}/out --labels-csv {LAB} | grep macro_f1_
    !rm -rf {D}
```

Ở đây xác suất được tính lại bằng `nckh.infer` (không TTA), nên Macro-F1 "sạch" có thể lệch nhẹ so với `evaluate_cls.py` ở P5b nếu P5b có bật TTA.

### 4.7. Dự báo: chỉ ảnh t của cặp UQ test bị suy giảm

```python
# cell: NB_cpu
RUN = "<run P6 trên UQ>"
f = pd.read_csv(f"{RUN}/ridge_test/features.csv")   # chỉ run đã mở test mới có target của cặp test
t = f[f.usable & (f.split == 'test')][['image_id_t', 'image_path_t']].drop_duplicates()
t.rename(columns={'image_id_t': 'image_id', 'image_path_t': 'image_path'}).assign(split='test').to_csv(f"{RUN}/p8_t_images.csv", index=False)
```

Sau đó, với mỗi mức (`D = {RUN}/p8_degraded/<mức>`):
1. Chạy `make_degraded.py --manifest {RUN}/p8_t_images.csv --data-root {ROOT}/data/uq --split test --level <mức> --out-dir {RUN}/p8_degraded/<mức>`.
2. Chạy `infer_images.py` **seg** (NB_seg) **và** **cls** (NB_cls) trên `{D}/degraded_manifest.csv`, cùng `--out-dir {D}/out`. Thư mục `D` nằm trên Drive để hai notebook dùng chung.
3. Chạy (NB_cpu):

```python
# cell: NB_cpu
P8F = f"{RUN}/p8_forecast"          # cố định cho mọi mức
lv, D = "hair_0.03", f"{RUN}/p8_degraded/hair_0.03"   # đổi theo từng mức
!cd {ROOT}/nckh && python scripts/evaluate_robustness.py forecast --level {lv} --robustness-json {P8F}/robustness.json --out-dir {P8F} --clean {RUN}/ridge_test/features.csv --degraded {D}/out --ridge {RUN}/ridge_test/ridge.joblib
```

`n_unpredictable_degraded` cho biết có bao nhiêu ảnh t bị suy giảm đến mức mask rỗng, không dự báo được. Báo con số này cạnh ΔMAE.

### 4.8. Ben Graham / Gamma trên val (NB_seg)

```python
# cell: NB_seg
!python -c "import pandas as pd; m=pd.read_csv('{MAN}'); m[(m.split=='val')&m.exclude_reason.isna()].to_csv('/content/isic18_val.csv', index=False)"
!cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {SEG} --manifest /content/isic18_val.csv --data-root /content/data --out-dir /content/clean_val
for en in ["ben_graham_10", "gamma_0.8", "gamma_1.2"]:
    D = f"/content/enh/{en}"
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/make_degraded.py --manifest /content/isic18_val.csv --data-root /content/data --split val --allow-non-test --level {en} --out-dir {D}
    !cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/infer_images.py {SEG} --manifest {D}/degraded_manifest.csv --data-root {D} --out-dir {D}/out
    !{VENV}/bin/python {ROOT}/nckh/scripts/evaluate_robustness.py seg --level {en} --robustness-json {P8}/enhancement_val.json --out-dir {P8} --clean /content/clean_val/masks --degraded {D}/out/masks --gt-dir /content/data/ISIC2018/Validation_GroundTruth | grep -E '"dice_'
```

Trong `enhancement_val.json`, `delta_dice` > 0 nghĩa là tiền xử lý **tốt hơn** RGB thường trên val. Chọn tối đa một biến thể có CI hoàn toàn > 0, ghi vào nhật ký quyết định, rồi mới chạy biến thể đó trên test (giống vòng lặp 4.5, với `--level <biến thể>` và `--allow-non-test` không cần vì split là test).

## 5. Test

```python
# cell: NB_cpu
!cd {ROOT}/nckh && python -m pytest -q tests/test_degrade.py tests/test_robustness.py
```

Kỳ vọng: `37 passed` (33 + 4).

| Test | Kiểm tra gì |
|---|---|
| `test_levels_are_the_protocol_ones` | Đúng 10 mức suy giảm + 3 tiền xử lý của protocol, không thêm bớt |
| `test_all_levels_dtype_shape_range` | Mọi mức giữ `uint8` và kích thước |
| `test_input_not_mutated` | Không sửa ảnh gốc tại chỗ |
| `test_brightness_clips`, `test_blur_reduces_variance`, `test_white_balance_green_unchanged`, `test_gamma_identity_at_one`, `test_ben_graham_flat_image_goes_gray` | Công thức từng phép |
| `test_hair_coverage_within_tolerance` (10 trường hợp), `test_hair_deterministic_by_seed` | Độ che 1%/3% ± 0,5 điểm %; cùng seed thì cùng ảnh |
| `test_make_degraded_*` | Từ chối split train (mã 2); ghi ảnh + manifest |
| `test_robustness.py` | Dice kém đi thì `delta_dice` < 0 với CI < 0; ảnh giống nhau thì ΔF1 = 0; forecast giữ nguyên target; JSON gộp nhiều mức |

## 6. Benchmark / đánh giá

| Phép suy giảm | ΔDice (CI 95%) — ISIC 2018 | ΔMacro-F1 (CI 95%) — ISIC 2017 | ΔMAE (CI 95%) — UQ test | Ảnh t không dự báo được |
|---|---|---|---|---:|
| brightness 0,8 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| brightness 1,2 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| blur σ 1,0 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| blur σ 1,5 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| WB R×1,1 B×0,9 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| WB R×0,9 B×1,1 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| WB R×1,2 B×0,8 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| WB R×0,8 B×1,2 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| lông 1% | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| lông 3% | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |

| Tiền xử lý (val, ISIC 2018, n = 100) | ΔDice so với RGB (CI 95%) | Chọn? |
|---|---|---|
| Ben Graham σ 10 | [điền sau khi chạy] | [điền sau khi chạy] |
| Gamma 0,8 | [điền sau khi chạy] | [điền sau khi chạy] |
| Gamma 1,2 | [điền sau khi chạy] | [điền sau khi chạy] |

Biểu đồ robustness sinh tự động bởi `make_figures.py` (P10) từ `robustness.json`.

## 7. Lỗi thường gặp trên Colab

| Triệu chứng | Cách xử lý |
|---|---|
| `make_degraded` thoát mã 2 | Đúng thiết kế: chỉ split test; dùng `--allow-non-test` **chỉ** cho tiền xử lý trên val |
| `ValueError: Thiếu mask sạch/suy giảm` | Chạy `infer_images` sạch chưa đủ ảnh hoặc sai thư mục `--clean` |
| `ValueError: Chỉ ghép được …` (cls) | Manifest suy giảm và nhãn lệch tên ảnh; dùng đúng `/content/isic17_test.csv` |
| Đầy đĩa `/content` | Giữ dòng `rm -rf {D}` trong vòng lặp |
| Vòng lặp bị ngắt | Đọc `robustness.json` để biết mức nào đã xong, bỏ khỏi `LEVELS` |

## 8. Checklist bàn giao cho SV A

- [ ] SV A đã duyệt danh sách mức (đúng bảng 8.1) trước khi chạy.
- [ ] `robustness.json` cho seg, cls, forecast (3 run) và `enhancement_val.json` trên Drive.
- [ ] Bảng mục 6 đã điền, kể cả các mức không làm metric thay đổi đáng kể.
- [ ] Quyết định về Ben Graham/Gamma (chọn hoặc không) ghi vào nhật ký **trước** khi đụng test.
- [ ] Ghi rõ trong báo cáo: lông là vật cản tổng hợp; σ blur tính trên ảnh gốc.
