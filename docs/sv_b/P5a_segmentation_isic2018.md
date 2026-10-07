# P5a — Fine-tune PanDerm Base segmentation trên ISIC 2018 Task 1

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 5.1):** fine-tune nhánh segmentation PanDerm Base trên ISIC 2018 Task 1. Chọn checkpoint trên validation, đánh giá **đúng một lần** trên test. Mọi bước phải có cấu hình, seed và log để tái lập. Phase này cũng tạo `src/nckh/metrics.py`, module metric dùng chung cho P5b, P6, P7 và P8.

| Đầu vào | Đầu ra (`MyDrive/NCKH_PanDerm/runs/<run_id>/`) | Sinh ở |
|---|---|---|
| `/content/data/ISIC2018/...` (Colab, P3), manifest + SHA-256 (`{ROOT}/data/manifests/`) | `0/model_best_0.ckpt` (chọn theo `Val/Jac`), `0/model_checkpoint_0.ckpt` (epoch mới nhất, để resume) | Colab `NB_seg` ghi lên Drive |
| Checkpoint pretrain + patch (P2) | `<log_name>/version_*/metrics.csv` (log từng epoch), `config.json` | Colab `NB_seg` ghi lên Drive |
| — | Sau khi khóa: `count_results_ISIC2018_0_<seed>.xlsx`, `results_ISIC2018/…png` (overlay), `eval/seg_metrics.json`, `eval/seg_per_image.csv`, `run_card.json` | Colab `NB_seg` ghi lên Drive → kéo về `~/nckh_drive/runs/<run_id>/` (trừ checkpoint) |

## 2. Chạy ở đâu

| Việc | Chạy ở | Thời gian ước tính trên T4 |
|---|---|---|
| Tạo `metrics.py`, `evaluate_seg.py`, test | Máy (`NB_cpu`) | 15 phút |
| Pilot 1 epoch trên 5% train + thử resume | Colab `NB_seg` (GPU) | 10–15 phút |
| Fine-tune đầy đủ (100 epoch như upstream) | Colab `NB_seg` (GPU) | Nhiều giờ, **vượt một phiên server Colab miễn phí (qua extension)**: chạy theo đợt và resume. Ước lượng đúng = (thời gian/epoch đo ở pilot) × 100 |
| Test một lần + `evaluate_seg.py` | Colab `NB_seg` | 5–10 phút |
| Kéo run về máy để đánh giá/điền bảng | Máy (terminal, rclone) | 1–3 phút |

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2/P5a. Code `run.py` upstream bắt buộc `accelerator='gpu'`, nên không chạy thử trên CPU được khi viết docs. Phần đã kiểm chứng trên CPU: dựng model ViT-B sau patch, nạp checkpoint, forward; `metrics.py` và `evaluate_seg.py` (pytest).

## 3. Giải thích

### 3.1. Upstream làm gì khi train (đọc trước khi chạy)

- **Loss:** CrossEntropy 2 lớp (nền / tổn thương). Optimizer AdamW, cosine LR theo epoch.
- **Augmentation (chỉ train):** ColorJitter, RandomGrayscale, RandomAutocontrast, RandomInvert. Val và test chỉ resize + chuẩn hóa. Đúng yêu cầu "chỉ augment train" của kế hoạch.
- **Chọn checkpoint:** `ModelCheckpoint(monitor="Val/Jac", mode="max")`, tức IoU trung bình trên tập val sau mỗi epoch. Đây là quy tắc chọn đã khóa trong protocol; không chọn theo test.
- **Đánh giá:** ở `test_step`, upstream:
  1. lấy `argmax`;
  2. giữ **thành phần liên thông lớn nhất** và lấp lỗ (`largestConnectComponent`);
  3. so với mask GT ở **224×224** bằng `medpy.dc`/`jc`.
- **Ba điểm phải ghi vào Methods:**
  1. Metric tính ở độ phân giải 224×224, không phải độ phân giải gốc.
  2. Nếu mask GT rỗng, upstream gán 1 pixel ở tâm GT trước khi tính.
  3. Nếu model không dự đoán pixel tổn thương nào, `largestConnectComponent` chọn nhãn 0 (nền) nên mask thành **toàn ảnh**. Dice khi đó thường rất thấp, nghĩa là lỗi "bỏ sót" vẫn bị phạt; nhưng cần báo số ảnh rơi vào trường hợp này. `nckh.infer` (dùng cho UQ) giữ mask rỗng là rỗng, khác upstream.
- **Test sau train:** upstream gọi `test_worker` ngay sau `train_worker`. Patch P2 bỏ lời gọi này, nên test chỉ chạy bằng `--evaluate` (mục 4.7).
- **Logger:** mặc định upstream dùng Weights & Biases (cần tài khoản). Cờ `--smoke_test` chỉ đổi sang `CSVLogger` ghi file `metrics.csv`, **không** làm giảm dữ liệu train. Dùng cờ này để không phụ thuộc wandb.

### 3.2. Vì sao cần resume và patch callback

Một phiên Colab miễn phí có thể bị ngắt sau vài giờ (khi nối qua extension, ngắt kết nối VS Code lâu cũng làm server bị thu hồi; checkpoint epoch nằm trên Drive nên resume vẫn được), trong khi 100 epoch trên 2.594 ảnh dài hơn thế. Upstream có cờ `--resume <fold>` để đọc `model_checkpoint_<fold>.ckpt`, nhưng callback tạo ra file này lại không được truyền vào `Trainer`. Patch P2 sửa thành `callbacks=[checkpoint_best, checkpoint_callback]`. Từ đó mỗi epoch lưu:
- `model_best_0.ckpt`: tốt nhất theo `Val/Jac`, dùng cho test;
- `model_checkpoint_0.ckpt`: epoch mới nhất, dùng để resume, giữ cả optimizer và scheduler.

Cả hai nằm trong `--save_name` trên **Drive**, nên không mất khi runtime ngắt.

### 3.3. Batch size và kích thước val

- Upstream để `--batch_size 1`. Trên T4 16 GB, ViT-B 224×224 thường chạy được batch 8. Bắt đầu từ 8; nếu `CUDA out of memory` thì giảm xuống 4. Ghi giá trị cuối vào run card và protocol.
- `val_loader` của upstream có `drop_last=True`: với `--test_batch_size 8`, chỉ 96/100 ảnh val được dùng để chọn checkpoint. Dùng **`--test_batch_size 4`** (100 chia hết cho 4) để không mất ảnh val nào. Test loader không bỏ ảnh.

### 3.4. Đường dẫn và dấu `/`

`run.py` ghép chuỗi trực tiếp: `args.parent_path + "ISIC2018/..."`, `args.save_name + "config.json"`. Vì vậy `--parent_path` và `--save_name` **bắt buộc kết thúc bằng `/`**.

### 3.5. Bootstrap cho segmentation

`evaluate_seg.py` đọc file Dice/IoU từng ảnh của upstream, tính trung bình **theo ảnh** (macro) và CI 95% bằng bootstrap 2.000 lần theo ảnh (seed 2026). ISIC 2018 không có `lesion_id`, nên đơn vị lấy mẫu là ảnh; ghi rõ điều này trong Methods (kế hoạch 7.5).

## 4. Code

### 4.1. `src/nckh/metrics.py` (dùng chung cho cả dự án)

📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/metrics.py`

```python
# file: src/nckh/metrics.py
"""Chỉ số đánh giá dùng chung cho segmentation, classification, dự báo + bootstrap theo nhóm."""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def dice_iou(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    if pred.shape != ref.shape:
        raise ValueError(f"Khác kích thước: pred {pred.shape} vs ref {ref.shape}")
    p, r = pred.astype(bool), ref.astype(bool)
    inter, total, union = (p & r).sum(), p.sum() + r.sum(), (p | r).sum()
    if total == 0:
        # Cả hai đều rỗng: dự đoán "không có tổn thương" là đúng hoàn toàn.
        return 1.0, 1.0
    return float(2 * inter / total), float(inter / union)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if np.abs(prob.sum(axis=1) - 1).max() > 1e-4:
        raise ValueError("Mỗi hàng xác suất phải có tổng = 1 (đã softmax chưa?)")
    k = len(class_names)
    y_pred = prob.argmax(axis=1)
    onehot = np.eye(k)[y_true]
    auroc = {}
    for i, name in enumerate(class_names):
        # AUROC một-lớp-với-phần-còn-lại không xác định nếu lớp đó vắng mặt hoặc chiếm toàn bộ.
        auroc[name] = float(roc_auc_score(onehot[:, i], prob[:, i])) if 0 < onehot[:, i].sum() < len(y_true) else float("nan")
    recalls = recall_score(y_true, y_pred, labels=list(range(k)), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(k)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc_macro": float(np.nanmean(list(auroc.values()))),
        "auroc_per_class": auroc,
        "recall_per_class": {name: float(r) for name, r in zip(class_names, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
        "brier": float(np.mean(np.sum((prob - onehot) ** 2, axis=1))),
        "n_per_class": {name: int((y_true == i).sum()) for i, name in enumerate(class_names)},
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),  # dương = dự báo diện tích tăng nhiều hơn thực tế
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "n": int(len(y_true)),
    }


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, stable_eps: float) -> float:
    def direction(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return np.where(np.abs(x) <= stable_eps, 0, np.sign(x))

    return float(np.mean(direction(y_true) == direction(y_pred)))


def _bootstrap(df: pd.DataFrame, group_col: str, stat: Callable[[pd.DataFrame], float], n_boot: int, seed: int,
               alpha: float) -> dict:
    # Lấy mẫu lại NHÓM (participant/lesion/ảnh) có hoàn lại rồi ghép mọi dòng của nhóm được chọn:
    # các cặp của cùng một người tương quan nhau, lấy mẫu từng dòng sẽ cho CI hẹp giả tạo.
    groups = df[group_col].to_numpy()
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    values, n_failed = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[g] for g in pick])
        try:
            values.append(stat(df.iloc[idx]))
        except ValueError as exc:
            # Ví dụ: một lần lặp thiếu hẳn một lớp nên AUROC không tính được. Bỏ lần đó, có đếm và log.
            n_failed += 1
            logger.debug("Bỏ một lần bootstrap: %s", exc)
    values = np.asarray(values, float)
    if n_failed:
        logger.warning("%d/%d lần bootstrap không tính được thống kê", n_failed, n_boot)
    return {
        "estimate": float(stat(df)),
        "ci_low": float(np.nanpercentile(values, 100 * alpha / 2)),
        "ci_high": float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
        "n_groups": int(len(uniq)),
        "n_boot": int(n_boot),
        "n_failed": int(n_failed),
    }


def group_bootstrap(df: pd.DataFrame, group_col: str, stat_fn: Callable[[pd.DataFrame], float], n_boot: int = 2000,
                    seed: int = 2026, alpha: float = 0.05) -> dict:
    return _bootstrap(df, group_col, stat_fn, n_boot, seed, alpha)


def paired_bootstrap_diff(df: pd.DataFrame, group_col: str, stat_fn_a: Callable[[pd.DataFrame], float],
                          stat_fn_b: Callable[[pd.DataFrame], float], n_boot: int = 2000, seed: int = 2026,
                          alpha: float = 0.05) -> dict:
    # Cùng một mẫu nhóm cho cả hai phương pháp ở mỗi lần lặp → CI của hiệu a − b hẹp và đúng hơn so với hai CI rời.
    return _bootstrap(df, group_col, lambda d: stat_fn_a(d) - stat_fn_b(d), n_boot, seed, alpha)
```

Điểm cần hiểu:
- `dice_iou`: cả hai mask rỗng → `(1, 1)`, vì dự đoán "không có tổn thương" khi thật sự không có là đúng hoàn toàn.
- `classification_metrics` từ chối xác suất chưa softmax (tổng ≠ 1). AUROC của một lớp vắng mặt trả `NaN` thay vì làm sập chương trình.
- `group_bootstrap`/`paired_bootstrap_diff`: lấy mẫu lại **nhóm** chứ không lấy từng dòng (xem `test_group_bootstrap_resamples_groups`). Kết quả có `n_failed`, là số lần lặp không tính được thống kê; nếu khác 0 phải báo cáo.

### 4.2. `scripts/evaluate_seg.py`

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/evaluate_seg.py`

```python
# file: scripts/evaluate_seg.py
"""Tổng hợp Dice/IoU từ file kết quả từng ảnh của PanDerm segmentation + CI bootstrap theo ảnh.

Ví dụ: python scripts/evaluate_seg.py --results-xlsx $RUN/count_results_ISIC2018_0_0.xlsx --out-dir $RUN/eval
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from nckh.metrics import group_bootstrap


def evaluate_seg_results(xlsx: Path, out_dir: Path, n_boot: int = 2000, seed: int = 2026) -> dict:
    per_image = pd.read_excel(xlsx, index_col=0)[["name", "dice", "jac"]].rename(columns={"jac": "iou"})
    if per_image["name"].duplicated().any():
        raise ValueError("Trùng tên ảnh trong file kết quả — có thể đã chạy test nhiều lần vào cùng thư mục")
    # Trung bình theo ảnh (macro), không gộp mọi pixel thành một mask lớn (kế hoạch 7.2).
    report = {
        metric: group_bootstrap(per_image, "name", lambda d, m=metric: d[m].mean(), n_boot=n_boot, seed=seed)
        for metric in ("dice", "iou")
    }
    report["n_images"] = int(len(per_image))
    report["source"] = str(xlsx)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / "seg_per_image.csv", index=False)
    (out_dir / "seg_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-xlsx", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    print(json.dumps(evaluate_seg_results(args.results_xlsx, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
```

### 4.3. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_metrics.py`

```python
# file: tests/test_metrics.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nckh.metrics import (
    classification_metrics,
    dice_iou,
    direction_accuracy,
    group_bootstrap,
    paired_bootstrap_diff,
    regression_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_dice_iou_hand() -> None:
    pred = np.zeros((1, 6), bool)
    ref = np.zeros((1, 6), bool)
    pred[0, 0:4] = True
    ref[0, 2:6] = True
    dice, iou = dice_iou(pred, ref)
    assert dice == pytest.approx(0.5) and iou == pytest.approx(1 / 3)


def test_dice_both_empty() -> None:
    assert dice_iou(np.zeros((3, 3)), np.zeros((3, 3))) == (1.0, 1.0)


def test_dice_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dice_iou(np.zeros((3, 3)), np.zeros((3, 4)))


def test_classification_perfect() -> None:
    y = np.array([0, 1, 2, 1])
    prob = np.eye(3)[y]
    m = classification_metrics(y, prob, ["mel", "nev", "sk"])
    assert m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0 and m["auroc_macro"] == 1.0
    assert m["brier"] == 0.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert m["n_per_class"] == {"mel": 1, "nev": 2, "sk": 1}


def test_classification_rejects_bad_prob() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), np.array([[0.5, 0.2, 0.2], [0.3, 0.3, 0.4]]), ["a", "b", "c"])


def test_regression_hand() -> None:
    m = regression_metrics(np.array([0.0, 0.0]), np.array([1.0, -1.0]))
    assert (m["mae"], m["rmse"], m["bias"], m["n"]) == (1.0, 1.0, 0.0, 2)


def test_direction_accuracy() -> None:
    acc = direction_accuracy(np.array([0.1, -0.1, 0.0]), np.array([0.2, 0.05, 0.001]), stable_eps=0.01)
    assert acc == pytest.approx(2 / 3)


def test_group_bootstrap_resamples_groups() -> None:
    # Nhóm A: 100 dòng = 0; nhóm B: 1 dòng = 1. Lấy mẫu theo NHÓM thì có lần chỉ toàn A (mean 0),
    # có lần chỉ toàn B (mean 1) → CI trải [0, 1]. Lấy mẫu theo dòng sẽ cho CI hẹp quanh 0,01.
    df = pd.DataFrame({"g": ["A"] * 100 + ["B"], "v": [0.0] * 100 + [1.0]})
    res = group_bootstrap(df, "g", lambda d: d["v"].mean(), n_boot=500, seed=1)
    assert res["n_groups"] == 2
    assert res["ci_low"] == 0.0 and res["ci_high"] == 1.0


def test_paired_diff_sign() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.arange(50), "y": rng.normal(size=50)})
    df["pred_a"] = df["y"] + 0.01
    df["pred_b"] = df["y"] + 0.05
    res = paired_bootstrap_diff(df, "g", lambda d: (d.pred_a - d.y).abs().mean(),
                                lambda d: (d.pred_b - d.y).abs().mean(), n_boot=300)
    assert res["estimate"] == pytest.approx(-0.04)
    assert res["ci_high"] < 0


def test_bootstrap_deterministic() -> None:
    df = pd.DataFrame({"g": np.arange(30) % 10, "v": np.arange(30, dtype=float)})
    a = group_bootstrap(df, "g", lambda d: d["v"].mean(), n_boot=200, seed=5)
    b = group_bootstrap(df, "g", lambda d: d["v"].mean(), n_boot=200, seed=5)
    assert a == b


def test_evaluate_seg_xlsx(tmp_path: Path) -> None:
    from evaluate_seg import evaluate_seg_results

    xlsx = tmp_path / "count_results_ISIC2018_0_0.xlsx"
    pd.DataFrame({"name": ["a", "b", "c"], "dice": [0.9, 0.8, 0.7], "jac": [0.8, 0.7, 0.6]}).to_excel(xlsx)
    res = evaluate_seg_results(xlsx, tmp_path / "out", n_boot=200)
    assert res["dice"]["estimate"] == pytest.approx(0.8)
    assert res["n_images"] == 3
    assert json.loads((tmp_path / "out" / "seg_metrics.json").read_text())["iou"]["estimate"] == pytest.approx(0.7)
    assert (tmp_path / "out" / "seg_per_image.csv").exists()
```

```python
# cell: NB_cpu (máy cá nhân)
!cd {REPO} && {PY} -m pytest -q tests/test_metrics.py
```

Kỳ vọng: `11 passed`.

### 4.4. Chuẩn bị mỗi phiên NB_seg

Đầu mỗi phiên Colab: push code ở máy, rồi chạy lần lượt trong `NB_seg`: cell setup 4.1b của P2 (có `git pull`) → cell dựng `venv_seg` (P2 mục 4.4) → cell áp patch (P2 mục 4.6) → tải ISIC 2018 vào `/content/data` bằng Python của `venv_seg` (đã cài `-e nckh`):

```python
# cell: NB_seg (Colab GPU)
!cd {CODE} && {VENV}/bin/python scripts/prepare_isic2018.py --zips-dir /content/zips --data-root /content/data --delete-zips
```

### 4.5. Pilot: 1 epoch trên 5% train + thử resume

```python
# cell: NB_seg (Colab GPU)
from nckh.runcard import new_run_id
PILOT = f"{ROOT}/runs/{new_run_id('seg_pilot')}/"   # dấu / cuối là bắt buộc
os.environ['PANDERM_CKPT'] = CK
os.environ['WANDB_MODE'] = 'disabled'
!cd /content/PanDerm/segmentation && {VENV}/bin/python run.py --workers 2 --gpu "0," --batch_size 8 --test_batch_size 4 --epoch 2 --lr 1e-4 --weight_decay 0.05 --model cae_seg --size 224 --dataset ISIC2018 --parent_path /content/data/ --save_name {PILOT} --seed 0 --smoke_test --percent 5 2>&1 | tee /content/pilot.log | grep --line-buffered -E "Matched|Val/|Error"
```

Khi thấy epoch 0 đã xong (log có `Val/Dice`), dừng cell bằng nút ■, hoặc restart kernel Colab (nút *Restart* trên thanh notebook) để giả lập bị ngắt. Sau đó chạy lại cell mở đầu (4.1b của P2) và `venv_seg`, rồi chạy:

```python
# cell: NB_seg (Colab GPU)
!ls -la {PILOT}0/
!cd /content/PanDerm/segmentation && {VENV}/bin/python run.py --workers 2 --gpu "0," --batch_size 8 --test_batch_size 4 --epoch 2 --lr 1e-4 --weight_decay 0.05 --model cae_seg --size 224 --dataset ISIC2018 --parent_path /content/data/ --save_name {PILOT} --seed 0 --smoke_test --percent 5 --resume 0 2>&1 | grep -E "loading checkpoint|Restoring|Epoch 1|Error" | head
```

Kỳ vọng:
- `ls` thấy `model_checkpoint_0.ckpt` và `model_best_0.ckpt`;
- lần chạy lại in `=> loading checkpoint '...model_checkpoint_0.ckpt'` và train tiếp từ epoch 1, **không** bắt đầu lại từ epoch 0.

Ghi lại các số sau vào bảng benchmark (mục 6):
- **thời gian 1 epoch:** trên 5% train, nhân 20 để ra ước lượng cho 100% train;
- **VRAM:** đọc bằng `!nvidia-smi` trong lúc chạy.

Pilot **không** dùng để báo cáo hiệu năng. Nhờ patch P2, train xong sẽ in `=> Test skipped. Lock the config, then rerun with --evaluate` và **không** chạy test. Upstream gốc thì tự chạy test ngay sau train, làm lộ kết quả test trước khi khóa cấu hình.

### 4.6. Fine-tune chính (nhiều phiên, có resume)

```python
# cell: NB_seg (Colab GPU)
SEED = 0                                              # 0, rồi 1, 2 nếu chạy 3 seed
BATCH = 8                                             # giảm 4 nếu OOM — dùng cùng giá trị ở mục 4.7
RUN = f"{ROOT}/runs/20261020-090000_seg_main_s{SEED}/"  # ĐẶT MỘT LẦN cho mỗi seed, giữ nguyên qua mọi phiên resume
os.makedirs(RUN, exist_ok=True)
# run.py lưu checkpoint vào {RUN}{SEED}/ — kiểm tra đúng thư mục của seed đang chạy.
RESUME = "--resume 0" if os.path.exists(f"{RUN}{SEED}/model_checkpoint_0.ckpt") else ""
print("RESUME =", RESUME or "(train từ đầu)")
os.environ['PANDERM_CKPT'] = CK
os.environ['WANDB_MODE'] = 'disabled'
!cd /content/PanDerm/segmentation && {VENV}/bin/python run.py --workers 2 --gpu "0," --batch_size {BATCH} --test_batch_size 4 --epoch 100 --lr 1e-4 --weight_decay 0.05 --model cae_seg --size 224 --dataset ISIC2018 --parent_path /content/data/ --save_name {RUN} --seed {SEED} --smoke_test {RESUME} 2>&1 | tee -a {RUN}train_stdout.log | grep --line-buffered -E "Val/|finished|skipped|Error"
```

- Lấy `run_id` bằng `new_run_id('seg_main')` ở phiên **đầu tiên**, rồi chép cứng vào dòng `RUN` để các phiên sau dùng đúng thư mục đó.
- Mỗi lần mở lại phiên: chạy mục 4.4 rồi chạy lại cell này. `RESUME` tự bật khi đã có checkpoint.
- Các tham số `lr`, `weight_decay`, `epoch` giữ đúng `run.sh` upstream. Không chỉnh theo kết quả test.
- **3 seed (nếu đủ tài nguyên và dung lượng Drive, mục 4.8):** đổi `SEED = 1` rồi `SEED = 2`. Mỗi seed có `RUN` riêng (đuôi `_s{SEED}`), và mọi đường dẫn bên dưới đều dùng `{SEED}`. Báo mean ± SD qua 3 seed, tách khỏi CI bootstrap của từng seed.
- Patch P2 tắt bộ đếm phiên bản của Lightning (`enable_version_counter=False`), nên khi resume file checkpoint được ghi đè đúng tên, không sinh `model_*-v1.ckpt`. Nếu vẫn thấy file `-v1`, nghĩa là patch chưa được áp: dừng lại và kiểm tra.

Theo dõi đường cong học (chạy trong `NB_seg`, vì log đang được ghi trên Drive trong lúc train và `RUN` đã có trong phiên):

```python
# cell: NB_seg (Colab GPU)
import pandas as pd, glob
log = sorted(glob.glob(f"{RUN}**/metrics.csv", recursive=True))[-1]
m = pd.read_csv(log)
print(m.groupby('epoch')[['Train/CE_Loss', 'Val/Dice', 'Val/Jac']].mean().tail(10))
```

### 4.7. Khóa cấu hình → test đúng một lần

Trước khi chạy test:
1. Ghi vào nhật ký quyết định: `RUN`, seed, epoch tốt nhất (theo `Val/Jac`), batch size, ngày giờ.
2. Ghi run card.
3. SV A xác nhận đã khóa protocol.

```python
# cell: NB_seg (Colab GPU)
!ls -la {RUN}{SEED}/                                   # kỳ vọng: model_best_0.ckpt, model_checkpoint_0.ckpt, không có file -v1
!cp {RUN}config.json {RUN}config_train.json            # --evaluate ghi đè config.json bằng tham số mặc định
!{VENV}/bin/python -m nckh.runcard {RUN}eval --seed {SEED} --input best={RUN}{SEED}/model_best_0.ckpt --input manifest={ROOT}/data/manifests/isic2018_seg.csv --config batch_size={BATCH} --config epochs=100 --config select=Val/Jac
!cd /content/PanDerm/segmentation && {VENV}/bin/python run.py --workers 2 --gpu "0," --test_batch_size 4 --model cae_seg --size 224 --dataset ISIC2018 --parent_path /content/data/ --save_name {RUN} --seed {SEED} --smoke_test --evaluate --save_results 2>&1 | grep -E "Writing|saved|Error"
!cd {CODE} && {VENV}/bin/python scripts/evaluate_seg.py --results-xlsx {RUN}count_results_ISIC2018_0_{SEED}.xlsx --out-dir {RUN}eval
```

`--save_results` ghi ảnh overlay (viền xanh = dự đoán, viền đỏ = GT) vào `{RUN}results_ISIC2018_{SEED}/`. Dùng các ảnh này cho phân tích lỗi ở P10.

Kéo kết quả đánh giá về máy (bỏ checkpoint để không tải ~1,8 GB/file):

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id> --exclude "*.ckpt" --progress
```

### 4.8. Dung lượng Drive cho checkpoint (đọc trước khi train)

Model segmentation có **160,7 triệu tham số** (đếm trên model ViT-B sau patch khi viết hướng dẫn). Checkpoint Lightning lưu cả trạng thái AdamW nên khoảng **1,8 GB/file**. Classification (85,9 triệu tham số) khoảng **1 GB/file**.

| Thứ trên Drive | Ước lượng |
|---|---:|
| Checkpoint pretrain PanDerm Base | xem `ls -lh` ở P2 |
| Pilot seg (`model_best_0` + `model_checkpoint_0`) | ~3,7 GB → **xoá ngay sau pilot** |
| Mỗi seed seg đang train (best + last) | ~3,7 GB |
| Mỗi seed seg sau khi thu gọn (chỉ trọng số best) | ~0,6 GB |
| `checkpoint-best.pth` classification | ~1 GB |

Drive miễn phí 15 GB: chỉ train **một seed tại một thời điểm**, và thu gọn ngay khi seed đó xong. **Sau khi đã chạy `--evaluate`** (mục 4.7) và ghi run card:

```python
# cell: NB_seg (Colab GPU)
!rm -rf {PILOT}                                          # nếu còn thư mục pilot
!rm -f {RUN}{SEED}/model_checkpoint_0.ckpt               # checkpoint để resume, không còn cần
# Bỏ trạng thái optimizer khỏi checkpoint best (~1,8 GB → ~0,6 GB); test_worker và nckh.infer chỉ đọc "state_dict".
!{VENV}/bin/python -c "import torch; p='{RUN}{SEED}/model_best_0.ckpt'; c=torch.load(p, map_location='cpu'); torch.save({{'state_dict': c['state_dict'], 'epoch': c.get('epoch')}}, p)"
!du -sh {ROOT}/runs/* | sort -h | tail
```

SHA-256 của file best **trước** khi thu gọn đã nằm trong `eval/run_card.json`. Sau khi thu gọn, file có hash mới; ghi thêm một dòng vào nhật ký quyết định. Nếu nhóm có Google One 100 GB thì có thể bỏ bước thu gọn.

## 5. Test

| Test | Kiểm tra gì |
|---|---|
| `test_dice_iou_hand`, `test_dice_both_empty`, `test_dice_shape_mismatch` | Công thức Dice/IoU trên ví dụ tính tay, mask rỗng, sai kích thước |
| `test_classification_perfect`, `test_classification_rejects_bad_prob` | Metric phân loại (dùng ở P5b) |
| `test_regression_hand`, `test_direction_accuracy` | Metric dự báo (dùng ở P6/P7) |
| `test_group_bootstrap_resamples_groups` | Lấy mẫu theo nhóm, không theo dòng |
| `test_paired_diff_sign`, `test_bootstrap_deterministic` | Paired bootstrap đúng chiều; cùng seed thì cùng kết quả |
| `test_evaluate_seg_xlsx` | Đọc đúng file kết quả của upstream |

**Sanity check trên dữ liệu thật:**
- `seg_metrics.json` có `n_images = 1000`, `n_failed = 0`.
- Không ảnh nào trong `seg_per_image.csv` có Dice `NaN`.
- Kiểm tra bằng mắt 10 overlay bất kỳ.
- Đếm số ảnh có Dice < 0,1 và mở xem: đây có thể là ảnh mà model "không thấy gì" (xem mục 3.1).

## 6. Benchmark / đánh giá

| Hạng mục | Giá trị |
|---|---|
| GPU / VRAM đỉnh (batch 8) | [điền sau khi chạy] |
| Thời gian 1 epoch (100% train) | [điền sau khi chạy] |
| Epoch tốt nhất theo `Val/Jac` / giá trị `Val/Jac`, `Val/Dice` | [điền sau khi chạy] |
| Số phiên Colab đã dùng / số lần resume | [điền sau khi chạy] |

| Thử nghiệm | N test | Dice (CI 95%) | IoU (CI 95%) | Ghi chú |
|---|---:|---|---|---|
| PanDerm Base seg — ISIC 2018 test, seed 0 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | 1 mask/ảnh; metric ở 224×224, giữ thành phần lớn nhất |
| 3 seed (mean ± SD) | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | Nếu chạy được |

Không so trực tiếp với bảng xếp hạng ISIC 2018 hay bài PanDerm (kế hoạch 5.3): split, độ phân giải metric và hậu xử lý đều khác.

## 7. Lỗi thường gặp (máy / Colab extension)

| Triệu chứng | Cách xử lý |
|---|---|
| `AssertionError: Chưa mount Drive` | `Ctrl+Shift+P` → *Colab: Mount Google Drive to Server...*, chạy lại cell setup |
| Kernel Colab mất kết nối / server bị thu hồi | *Select Kernel → Colab → New Colab Server*, chạy lại mục 4.4 rồi cell train (tự resume); dữ liệu trên Drive vẫn còn |
| Colab chạy code cũ | Ở máy `git push`, chạy lại cell setup (có `git pull`) |
| `userdata.get` / `files.upload` lỗi | Chưa hỗ trợ trong extension; không dùng, file đi qua Drive |
| `rclone` báo `couldn't fetch token` | `rclone config reconnect gdrive:` |
| Ở máy không thấy file Colab vừa ghi | Chạy lệnh `rclone copy gdrive:NCKH_PanDerm/... ~/nckh_drive/...` |
| `wandb: ERROR api_key not configured` | Thiếu `--smoke_test`, hoặc chưa đặt `WANDB_MODE=disabled` |
| `FileNotFoundError: ...ISIC2018/...` hoặc `=> Loading train dataset with 0 images` | `--parent_path` thiếu `/` cuối, hoặc chưa chạy P3 trong phiên này |
| `CUDA out of memory` | Giảm `--batch_size 4`; ghi vào run card |
| Resume nhưng train lại từ epoch 0 | Chưa áp patch có hunk `train.py`, hoặc `RUN` khác thư mục lần trước |
| `MisconfigurationException` về `devices` | Dùng đúng `--gpu "0,"` (có dấu phẩy) |
| `SyncBatchNorm` lỗi khi gọi model ngoài `run.py` | Trong `run.py`, DDP khởi tạo process group; ngoài đó hãy dùng `nckh.infer` (model ở chế độ `eval` không cần đồng bộ) |
| Runtime bị ngắt liên tục | Chạy vào giờ thấp điểm; lưu checkpoint mỗi epoch (đã có); cân nhắc GPU của trường (kế hoạch Mục 6) |

## 8. Checklist bàn giao cho SV A

- [ ] Log pilot chứng minh resume được (dòng `loading checkpoint` + epoch tiếp theo).
- [ ] `runs/<seg_main_s{SEED}>/` có `{SEED}/model_best_0.ckpt`, `metrics.csv`, `train_stdout.log`, `config_train.json`; pilot đã xoá; dung lượng Drive còn đủ.
- [ ] Nhật ký quyết định ghi cấu hình khóa **trước** khi chạy `--evaluate`.
- [ ] `eval/seg_metrics.json`, `eval/seg_per_image.csv`, `eval/run_card.json`, thư mục overlay.
- [ ] Bảng mục 6 đã điền. SV A tự tính lại mean Dice từ `seg_per_image.csv` và khớp với JSON.
- [ ] Báo số ảnh Dice < 0,1 và số ảnh mà model không dự đoán pixel nào.
