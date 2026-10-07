# P5b — Fine-tune PanDerm Base classification trên ISIC 2017 Task 3

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 5.2):** fine-tune nhánh phân loại PanDerm Base cho 3 nhóm tham khảo (melanoma, nevus, seborrheic keratosis). Chọn checkpoint trên validation, đánh giá test **một lần**, lưu xác suất từng ảnh để tính lại mọi metric. Đầu ra là **điểm/xác suất theo nhãn dữ liệu**, không phải chẩn đoán.

| Đầu vào | Đầu ra (`MyDrive/NCKH_PanDerm/runs/<run_id>/`) | Sinh ở |
|---|---|---|
| `/content/data/ISIC2017/ISIC-2017_*_Data/` (Colab, P3) | `checkpoint-best.pth` (chọn theo val) | Colab `NB_cls` ghi lên Drive |
| `{ROOT}/data/manifests/isic2017_cls_trainphase.csv` (khi train), `isic2017_cls.csv` (khi test) | `val.csv` (xác suất val của epoch gần nhất), `log.txt`, `train_stdout.log` | Colab `NB_cls` ghi lên Drive |
| Checkpoint pretrain (P2) | Sau khi khóa: `eval_test/test.csv`, `eval_test/cls_metrics.json`, `run_card.json` | Colab `NB_cls` ghi lên Drive → kéo về `~/nckh_drive/runs/<run_id>/` |
| — | `eval_test/calibration.png` | Máy, trong `~/nckh_drive/runs/<run_id>/eval_test/` |

## 2. Chạy ở đâu

| Việc | Chạy ở | Thời gian ước tính |
|---|---|---|
| Tạo `evaluate_cls.py`, `annotator_agreement.py`, test | Máy (`NB_cpu`) | 10 phút |
| Tải ISIC 2017 (P3) | Colab `NB_cls` | 10–25 phút mỗi phiên |
| Pilot 1 epoch | Colab `NB_cls` (GPU) | 5–10 phút |
| Fine-tune 50 epoch | Colab `NB_cls` (GPU) | Ước lượng = thời gian/epoch đo ở pilot × 50 (thường vừa trong một phiên) |
| Eval test một lần + metric | Colab `NB_cls` | 5 phút |
| Kéo run về máy, vẽ calibration | Máy (terminal rclone + `NB_cpu`) | 1–3 phút |

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2/P5b. Phần đã kiểm chứng trên CPU khi viết docs: dựng `PanDerm_Base_FT` và nạp pretrain bằng code `classification/` thật; tạo CSV nhãn từ ground truth thật của ISIC 2017; `evaluate_cls.py` (pytest).

## 3. Giải thích

### 3.1. Bẫy lớn nhất: script upstream tự mở test

Ở epoch cuối, `run_class_finetuning.py` (khoảng dòng 720) nạp `checkpoint-best.pth` rồi **tự chạy trên split `test`** của CSV. Nếu dùng CSV có test thật, bạn sẽ thấy kết quả test trước khi khóa cấu hình. Đó đúng là điều kế hoạch cấm: không chọn checkpoint, ngưỡng, TTA hay augmentation dựa trên test.

**Cách xử lý (không sửa upstream):** khi train dùng `isic2017_cls_trainphase.csv`, trong đó các dòng `test` là **bản sao của val** (tạo ở P3). Lần chạy "test" tự động lúc cuối khi đó thực chất chạy trên val. Kiểm tra bằng cách xem `test.csv` sinh ra khi train: phải có **150 dòng** (bằng val), không phải 600. Test thật chỉ chạy một lần với `--eval` và `isic2017_cls.csv` (mục 4.5).

### 3.2. Cấu hình fine-tune

Giữ cấu hình khuyến nghị của README upstream: LR 5e-4, 50 epoch, warmup 10, layer decay 0,65, drop path 0,2, weight decay 0,05, mixup 0,8, cutmix 1,0, weighted sampler. Chỉ đổi những gì bắt buộc vì phần cứng:

| Tham số | Giá trị | Lý do |
|---|---|---|
| `--model` | `PanDerm_Base_FT` | Đề cương chốt Base |
| `--nb_classes` | `3` | mel / nev / sk |
| `--batch_size 32 --update_freq 4` | batch hiệu dụng 128 = khuyến nghị upstream | T4 không chứa nổi batch 128 một lần |
| `--weights` | bật | Mất cân bằng lớp (nevus ≈ 69% train). Sampler chỉ áp dụng cho train (dòng ~323 upstream) |
| `--monitor` | `recall` | Recall macro = balanced accuracy = metric chính. **Phải ghi trong protocol P0 trước khi chạy** |
| `--TTA` | **không dùng** (khuyến nghị) | `TTAHandler` của upstream dùng augmentation ngẫu nhiên không seed, nên kết quả test thay đổi giữa các lần chạy; ngoài ra transform test khi bật TTA khác val. Nếu protocol vẫn chọn TTA thì dùng TTA cho cả val lẫn test và ghi rõ |
| `--no_auto_resume` | giữ như upstream | Upstream chỉ lưu `checkpoint-best.pth`, không có checkpoint theo epoch, nên không resume được. 50 epoch thường vừa một phiên; nếu bị ngắt thì chạy lại từ đầu |
| `WANDB_MODE=disabled` | — | Script luôn gọi `wandb.init` |

### 3.3. Tiền xử lý lúc đánh giá

Val/test (không TTA): `Resize(256)` (bilinear) → `CenterCrop(224)` → chuẩn hóa mean `(0,485; 0,456; 0,406)`, std `(0,228; 0,224; 0,225)` (lấy đúng từ dòng 268–271 của upstream). `nckh.infer.cls_preprocess` (P2) lặp lại y hệt, nên xác suất trên ảnh UQ (P6) và trong demo (P9) cùng phân phối với lúc đánh giá.

### 3.4. Metric và cách đọc

- **Chính:** Macro-F1, balanced accuracy, AUROC một-lớp-với-phần-còn-lại cho từng lớp và trung bình.
- **Bắt buộc kèm:** confusion matrix (số lượng), recall từng lớp, số ảnh mỗi lớp. Có Brier score và đường calibration vì demo hiển thị xác suất.
- `evaluate_cls.py` tính lại tất cả từ `test.csv`, không dùng con số in ra trong log upstream (log upstream báo *weighted* F1, không phải macro). CI 95% của Macro-F1 bằng bootstrap 2.000 lần theo ảnh.

## 4. Code

### 4.1. `scripts/evaluate_cls.py`

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/evaluate_cls.py`

```python
# file: scripts/evaluate_cls.py
"""Tính lại metric phân loại từ test.csv do run_class_finetuning.py ghi (xác suất softmax từng ảnh).

Ví dụ: python scripts/evaluate_cls.py --pred-csv $RUN/test.csv --labels-csv $ROOT/data/manifests/isic2017_cls.csv --out-dir $RUN/eval
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from nckh.metrics import classification_metrics, group_bootstrap

CLASS_NAMES = ("melanoma", "nevus", "seborrheic_keratosis")


def evaluate_cls_csv(pred_csv: Path, out_dir: Path, labels_csv: Path | None = None,
                     class_names: tuple[str, ...] = CLASS_NAMES, n_boot: int = 2000, seed: int = 2026) -> dict:
    pred = pd.read_csv(pred_csv)
    if labels_csv is not None:
        n_test = int((pd.read_csv(labels_csv)["split"] == "test").sum())
        if n_test != len(pred):
            # Lệch số dòng = đang đọc nhầm file test của pha train (bản sao val) hoặc thiếu ảnh.
            raise ValueError(f"{pred_csv.name} có {len(pred)} dòng nhưng split test có {n_test} ảnh")
    prob_cols = [f"probability_class_{i}" for i in range(len(class_names))]
    prob = pred[prob_cols].to_numpy(float)
    prob = prob / prob.sum(axis=1, keepdims=True)  # CSV upstream làm tròn 8 chữ số; chuẩn hóa lại cho tổng đúng bằng 1
    y = pred["true_label"].to_numpy(int)
    report = classification_metrics(y, prob, list(class_names))

    def macro_f1(d: pd.DataFrame) -> float:
        p = d[prob_cols].to_numpy(float).argmax(axis=1)
        return float(f1_score(d["true_label"], p, labels=list(range(len(class_names))), average="macro", zero_division=0))

    # ISIC 2017 không có lesion_id công khai: đơn vị bootstrap là ảnh (ghi rõ trong Methods).
    report["macro_f1_ci"] = group_bootstrap(pred, "filename", macro_f1, n_boot=n_boot, seed=seed)
    report["n_images"] = int(len(pred))
    report["source"] = str(pred_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cls_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    res = evaluate_cls_csv(args.pred_csv, args.out_dir, args.labels_csv)
    print(json.dumps({k: res[k] for k in ("macro_f1", "balanced_accuracy", "auroc_macro", "n_per_class", "macro_f1_ci")},
                     indent=2, ensure_ascii=False))
    print(np.array(res["confusion_matrix"]))


if __name__ == "__main__":
    main()
```

### 4.2. `scripts/annotator_agreement.py` (dùng ở P4)

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/annotator_agreement.py`

```python
# file: scripts/annotator_agreement.py
"""Độ đồng thuận giữa hai người gán mask (P4): Dice, IoU, tỷ lệ diện tích từng người và chênh lệch.

Mask: PNG đen trắng, tên <image_id>.png, pixel > 127 là tổn thương; mỗi người một thư mục.
Ví dụ: python scripts/annotator_agreement.py --dir-a $ROOT/data/uq/audit/annotator_A --dir-b $ROOT/data/uq/audit/annotator_B --out $ROOT/runs/<run_id>/agreement.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nckh.metrics import dice_iou


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def annotator_agreement(dir_a: Path, dir_b: Path) -> pd.DataFrame:
    names_a = {p.name for p in dir_a.glob("*.png")}
    names_b = {p.name for p in dir_b.glob("*.png")}
    if names_a != names_b:
        # Không âm thầm bỏ ảnh: audit phải so trên đúng cùng một tập ảnh.
        raise ValueError(f"Chỉ có ở A: {sorted(names_a - names_b)[:10]}; chỉ có ở B: {sorted(names_b - names_a)[:10]}")
    rows = []
    for name in sorted(names_a):
        a, b = _load(dir_a / name), _load(dir_b / name)
        dice, iou = dice_iou(a, b)
        rows.append({"image_id": Path(name).stem, "dice": dice, "iou": iou, "area_a": float(a.mean()),
                     "area_b": float(b.mean()), "area_diff": float(abs(a.mean() - b.mean()))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir-a", type=Path, required=True)
    ap.add_argument("--dir-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    df = annotator_agreement(args.dir_a, args.dir_b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df[["dice", "iou", "area_diff"]].describe().loc[["count", "mean", "50%", "min", "max"]])


if __name__ == "__main__":
    main()
```

### 4.3. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_eval_cls_agreement.py`

```python
# file: tests/test_eval_cls_agreement.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from annotator_agreement import annotator_agreement  # noqa: E402
from evaluate_cls import evaluate_cls_csv  # noqa: E402


def _pred_csv(path: Path) -> Path:
    # Định dạng test.csv mà run_class_finetuning.py ghi ra.
    rows = [
        ("ISIC-2017_Test_v2_Data/ISIC_1.jpg", 0, 0, 0.8, 0.1, 0.1),
        ("ISIC-2017_Test_v2_Data/ISIC_2.jpg", 1, 1, 0.1, 0.8, 0.1),
        ("ISIC-2017_Test_v2_Data/ISIC_3.jpg", 2, 2, 0.1, 0.1, 0.8),
        ("ISIC-2017_Test_v2_Data/ISIC_4.jpg", 1, 0, 0.6, 0.3, 0.1),
        ("ISIC-2017_Test_v2_Data/ISIC_5.jpg", 0, 0, 0.7, 0.2, 0.1),
        ("ISIC-2017_Test_v2_Data/ISIC_6.jpg", 2, 2, 0.2, 0.2, 0.6),
    ]
    cols = ["filename", "true_label", "predicted_label", "probability_class_0", "probability_class_1", "probability_class_2"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def test_evaluate_cls_csv(tmp_path: Path) -> None:
    res = evaluate_cls_csv(_pred_csv(tmp_path / "test.csv"), tmp_path / "out", n_boot=100)
    assert res["n_per_class"] == {"melanoma": 2, "nevus": 2, "seborrheic_keratosis": 2}
    assert 0 < res["macro_f1"] < 1
    assert res["macro_f1_ci"]["ci_low"] <= res["macro_f1"] <= res["macro_f1_ci"]["ci_high"]
    assert json.loads((tmp_path / "out" / "cls_metrics.json").read_text())["confusion_matrix"][1] == [1, 1, 0]


def test_evaluate_cls_label_count_mismatch(tmp_path: Path) -> None:
    labels = tmp_path / "isic2017_cls.csv"
    pd.DataFrame({"image": ["a.jpg"] * 5, "label": [0] * 5, "split": ["test"] * 5}).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="6"):
        evaluate_cls_csv(_pred_csv(tmp_path / "test.csv"), tmp_path / "out", labels_csv=labels, n_boot=10)


def _mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((20, 20), np.uint8)
    m[box[0]:box[1], box[2]:box[3]] = 255
    Image.fromarray(m).save(path)


def test_agreement_identical(tmp_path: Path) -> None:
    _mask(tmp_path / "a" / "img1.png", (0, 10, 0, 10))
    _mask(tmp_path / "b" / "img1.png", (0, 10, 0, 10))
    df = annotator_agreement(tmp_path / "a", tmp_path / "b")
    assert df.loc[0, "dice"] == 1.0 and df.loc[0, "area_diff"] == 0.0


def test_agreement_area_diff(tmp_path: Path) -> None:
    _mask(tmp_path / "a" / "img1.png", (0, 10, 0, 10))   # 100 px / 400 = 0.25
    _mask(tmp_path / "b" / "img1.png", (0, 10, 0, 5))    # 50 px / 400 = 0.125
    row = annotator_agreement(tmp_path / "a", tmp_path / "b").iloc[0]
    assert row["area_a"] == 0.25 and row["area_b"] == 0.125 and row["area_diff"] == pytest.approx(0.125)
    assert row["dice"] == pytest.approx(2 * 50 / 150)


def test_agreement_missing_file_raises(tmp_path: Path) -> None:
    _mask(tmp_path / "a" / "img1.png", (0, 5, 0, 5))
    _mask(tmp_path / "a" / "img2.png", (0, 5, 0, 5))
    _mask(tmp_path / "b" / "img1.png", (0, 5, 0, 5))
    with pytest.raises(ValueError, match="img2"):
        annotator_agreement(tmp_path / "a", tmp_path / "b")
```

```python
# cell: NB_cpu (máy cá nhân)
!cd {REPO} && {PY} -m pytest -q tests/test_eval_cls_agreement.py
```

Kỳ vọng: `5 passed`.

### 4.4. Pilot rồi fine-tune (NB_cls)

Đầu mỗi phiên Colab: push code ở máy, rồi chạy trong `NB_cls`: cell setup 4.1b của P2 (có `git pull`) → `venv_cls` (P2 4.10, các dòng cài đặt) → tải ISIC 2017 vào `/content/data` (P3 4.6, cell `NB_cls`).

```python
# cell: NB_cls (Colab GPU)
from nckh.runcard import new_run_id
os.environ['WANDB_MODE'] = 'disabled'
TRAIN_CSV = f'{ROOT}/data/manifests/isic2017_cls_trainphase.csv'
COMMON = (f"--model PanDerm_Base_FT --pretrained_checkpoint {CK} --nb_classes 3 --batch_size 32 --update_freq 4 "
          f"--lr 5e-4 --warmup_epochs 10 --layer_decay 0.65 --drop_path 0.2 --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 "
          f"--weights --monitor recall --sin_pos_emb --no_auto_resume --imagenet_default_mean_and_std "
          f"--root_path /content/data/ISIC2017/ --seed 0")

# Pilot: 1 epoch, 10% train. Chỉ để đo thời gian/VRAM và chắc chắn mọi thứ chạy.
PILOT = f"{ROOT}/runs/{new_run_id('cls_pilot')}/"
!cd /content/PanDerm/classification && {VENV}/bin/python run_class_finetuning.py {COMMON} --epochs 1 --warmup_epochs 0 --percent_data 0.1 --exp_name pilot --wandb_name pilot --output_dir {PILOT} --csv_path {TRAIN_CSV} 2>&1 | tail -5
!wc -l {PILOT}test.csv   # kỳ vọng 151 dòng (150 ảnh val + header): đúng là bản sao val, không phải test thật
```

```python
# cell: NB_cls (Colab GPU)
RUN = f"{ROOT}/runs/{new_run_id('cls_main')}/"
os.makedirs(RUN, exist_ok=True)   # tee cần thư mục tồn tại trước
!cd /content/PanDerm/classification && {VENV}/bin/python run_class_finetuning.py {COMMON} --epochs 50 --exp_name isic2017_ft --wandb_name isic2017_ft_s0 --output_dir {RUN} --csv_path {TRAIN_CSV} 2>&1 | tee {RUN}train_stdout.log | grep -E "Max val|Epoch: \[[0-9]+\] Total|Error"
```

Lưu ý `--warmup_epochs 0` ở pilot: chạy 1 epoch thì không thể có 10 epoch warmup. Trong lúc train, theo dõi `Max val mean recall` mỗi epoch.

**Dung lượng:** `checkpoint-best.pth` khoảng 1 GB (model 85,9 triệu tham số + trạng thái optimizer). Xoá thư mục pilot ngay sau khi đo xong (`!rm -rf {PILOT}`); xem bảng dung lượng tổng ở P5a mục 4.8.

### 4.5. Khóa cấu hình → test thật đúng một lần

Trước khi chạy: ghi vào nhật ký quyết định `RUN`, epoch tốt nhất, `--monitor`, có TTA hay không, ngày giờ. SV A xác nhận đã khóa.

```python
# cell: NB_cls (Colab GPU)
EVAL = f"{RUN}eval_test/"
!{VENV}/bin/python -m nckh.runcard {EVAL} --seed 0 --input best={RUN}checkpoint-best.pth --input labels={ROOT}/data/manifests/isic2017_cls.csv --config monitor=recall --config tta=false
!cd /content/PanDerm/classification && {VENV}/bin/python run_class_finetuning.py {COMMON} --epochs 50 --exp_name isic2017_test --wandb_name isic2017_test --output_dir {EVAL} --csv_path {ROOT}/data/manifests/isic2017_cls.csv --resume {RUN}checkpoint-best.pth --eval 2>&1 | tail -3
!cd {CODE} && {VENV}/bin/python scripts/evaluate_cls.py --pred-csv {EVAL}test.csv --labels-csv {ROOT}/data/manifests/isic2017_cls.csv --out-dir {EVAL}
```

`evaluate_cls.py --labels-csv` dừng với lỗi nếu `test.csv` không có đúng 600 dòng. Đây là hàng rào chặn việc lỡ đọc nhầm file test của pha train.

### 4.6. Đường calibration (máy)

Kéo kết quả eval về máy (bỏ checkpoint ~1 GB):

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id> --exclude "*.pth" --progress
```

```python
# cell: NB_cpu (máy cá nhân)
import pandas as pd, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
EVAL = f"{ROOT}/runs/<cls_main_run_id>/eval_test/"      # thay đúng run_id
t = pd.read_csv(f"{EVAL}test.csv")
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for k, name in enumerate(["melanoma", "nevus", "seborrheic_keratosis"]):
    frac, mean_pred = calibration_curve((t.true_label == k).astype(int), t[f"probability_class_{k}"], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Xác suất dự đoán"); ax.set_ylabel("Tỷ lệ thực tế"); ax.legend(); fig.tight_layout()
fig.savefig(f"{EVAL}calibration.png", dpi=200)
```

## 5. Test

| Test | Kiểm tra gì |
|---|---|
| `test_evaluate_cls_csv` | Đọc đúng định dạng `test.csv` upstream; confusion matrix; CI bao quanh ước lượng |
| `test_evaluate_cls_label_count_mismatch` | Số dòng sai thì dừng (chống đọc nhầm file) |
| `test_agreement_*` (3 test) | Dùng ở P4: Dice/diện tích giữa hai người gán, thiếu file thì báo lỗi |

**Sanity check trên dữ liệu thật:**
- `test.csv` khi train có 150 dòng; khi eval có 600 dòng.
- `n_per_class` khi eval là `{melanoma: 117, nevus: 393, seborrheic_keratosis: 90}`.
- `macro_f1_ci.n_failed` = 0.
- Tổng mỗi hàng của confusion matrix khớp `n_per_class`.

## 6. Benchmark / đánh giá

| Hạng mục | Giá trị |
|---|---|
| GPU / VRAM đỉnh (batch 32) | [điền sau khi chạy] |
| Thời gian 1 epoch / tổng 50 epoch | [điền sau khi chạy] |
| Epoch tốt nhất (`Max val mean recall`) | [điền sau khi chạy] |

| Thử nghiệm | N test (mel/nev/sk) | Macro-F1 (CI 95%) | Balanced acc. | AUROC macro | Brier |
|---|---|---|---:|---:|---:|
| PanDerm Base cls — ISIC 2017 test | 600 (117/393/90) | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |

| Lớp | n | Recall | AUROC |
|---|---:|---:|---:|
| melanoma | 117 | [điền sau khi chạy] | [điền sau khi chạy] |
| nevus | 393 | [điền sau khi chạy] | [điền sau khi chạy] |
| seborrheic keratosis | 90 | [điền sau khi chạy] | [điền sau khi chạy] |

Không gọi kết quả là "độ chính xác chẩn đoán". Khi viết, dùng cách diễn đạt: "điểm nhóm tham khảo theo nhãn ISIC 2017".

## 7. Lỗi thường gặp (máy / Colab extension)

| Triệu chứng | Cách xử lý |
|---|---|
| `AssertionError: Chưa mount Drive` | `Ctrl+Shift+P` → *Colab: Mount Google Drive to Server...*, chạy lại cell setup |
| Kernel Colab mất kết nối / server bị thu hồi | *Select Kernel → Colab → New Colab Server*, chạy lại cell setup + venv; dữ liệu trên Drive vẫn còn |
| Colab chạy code cũ | Ở máy `git push`, chạy lại cell setup (có `git pull`) |
| `userdata.get` / `files.upload` lỗi | Chưa hỗ trợ trong extension; không dùng, file đi qua Drive |
| `rclone` báo `couldn't fetch token` | `rclone config reconnect gdrive:` |
| Ở máy không thấy file Colab vừa ghi | Chạy lệnh `rclone copy gdrive:NCKH_PanDerm/... ~/nckh_drive/...` |
| Hỏi đăng nhập wandb | `os.environ['WANDB_MODE'] = 'disabled'` trước khi chạy |
| `Error opening file: /content/data/ISIC2017/...` rồi lỗi `NoneType` | Chưa tải ảnh ISIC 2017 trong phiên này, hoặc `--root_path` thiếu `/` cuối |
| `test.csv` khi train có 600 dòng | Đang dùng nhầm `isic2017_cls.csv`. **Dừng lại**: dừng, ghi vào nhật ký quyết định rằng test đã bị lộ, báo SV A |
| `CUDA out of memory` | `--batch_size 16 --update_freq 8` (vẫn giữ batch hiệu dụng 128) |
| `ModuleNotFoundError: open_clip` | Cài lại `classification/requirements.txt` trong `venv_cls` |
| Server Colab bị ngắt giữa chừng | Chạy lại toàn bộ với một `RUN` mới; ghi chú run cũ là "không hoàn tất" |

## 8. Checklist bàn giao cho SV A

- [ ] Protocol ghi `--monitor recall` và quyết định về TTA **trước** khi train.
- [ ] `test.csv` của pha train có 150 dòng (bằng chứng test thật chưa bị mở).
- [ ] `runs/<cls_main>/checkpoint-best.pth`, `train_stdout.log`, `log.txt`.
- [ ] `eval_test/test.csv` (600 dòng), `cls_metrics.json`, `run_card.json` trên Drive; `calibration.png` ở máy (`~/nckh_drive/runs/<cls_main>/eval_test/`), đẩy lên Drive nếu SV A cần: `rclone copyto ~/nckh_drive/runs/<cls_main>/eval_test/calibration.png gdrive:NCKH_PanDerm/runs/<cls_main>/eval_test/calibration.png`.
- [ ] Bảng mục 6 đã điền. SV A tự tính lại confusion matrix từ `test.csv` và khớp với JSON.
