# P7 — Bộ kiểm thử, chỉ số và bootstrap

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 7):** gom mọi phép kiểm tra dữ liệu/hệ thống thành bộ test chạy được bằng một lệnh; chốt cách tính metric chính/phụ cho 3 nhánh; tính khoảng tin cậy bootstrap đúng đơn vị lấy mẫu; so Ridge với baseline bằng **paired bootstrap** theo participant.

| Đầu vào | Đầu ra | Sinh ở |
|---|---|---|
| `predictions_val.csv`, `predictions_test.csv` (P6) | `eval_forecast/forecast_metrics.json`, `eval_forecast/forecast_by_dt_bin.csv` | Máy: `~/nckh_drive/runs/<p6_uq_run_id>/eval_forecast/` |
| `seg_metrics.json` (P5a), `cls_metrics.json` (P5b) | Bảng kết quả tổng (điền ở mục 6, sinh tự động ở P10) | Máy |
| Toàn bộ `<repo>/tests/` | Log `pytest` lưu vào `runs/<run_id>/pytest.log` | Máy: `~/nckh_drive/runs/<run_id>/pytest.log` |

## 2. Chạy ở đâu

Tất cả chạy **trên máy** (`NB_cpu`, chạy cell setup P2 4.1a trước). Một lần `evaluate_forecast.py` mất vài giây đến vài chục giây (2.000 lần bootstrap). Biến `RUN` là run P6 trên UQ ở máy: `RUN = f"{ROOT}/runs/<p6_uq_run_id>"`.

Run P6 (`ridge/`, `ridge_test/`) đã nằm sẵn ở máy. Các run P5a/P5b (`seg_metrics.json`, `cls_metrics.json`) do Colab ghi, nên kéo về trước nếu chưa làm:

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id> --exclude "*.ckpt" --exclude "*.pth" --progress
```

## 3. Giải thích

### 3.1. Metric chính và phụ (chốt trong protocol P0, không đổi sau khi xem test)

| Nhánh | Chính | Phụ / bắt buộc kèm | Tính ở |
|---|---|---|---|
| Segmentation | Dice, IoU: trung bình **theo ảnh** + CI 95% | Tỷ lệ mask rỗng/lỗi, ví dụ thất bại; UQ audit: sai số tuyệt đối tỷ lệ diện tích | `evaluate_seg.py` (P5a), `annotator_agreement.py` (P4/P6) |
| Phân loại tham khảo | Macro-F1, balanced accuracy, AUROC OvR từng lớp + macro | Confusion matrix (số lượng), recall từng lớp, n từng lớp, Brier, calibration | `evaluate_cls.py` (P5b) |
| Dự báo Δa | **MAE(Δa) trên participant test**, đây là kết quả chính được chỉ định trước | RMSE, bias (sai lệch trung bình có dấu), R², độ đúng hướng tăng/giảm/ổn định, sai số theo nhóm Δt | `evaluate_forecast.py` (P7) |

**Đọc MAE theo điểm phần trăm diện tích ảnh:** MAE = 0,01 nghĩa là trung bình dự báo lệch 1 điểm phần trăm diện tích ảnh (ví dụ thật là 12%, đoán 11% hoặc 13%). **Không** gọi đó là sai số tăng trưởng vật lý.

**Bias:** `mean(dự báo − thật)`. Dương nghĩa là mô hình có xu hướng dự báo mask lớn lên nhiều hơn thực tế.

### 3.2. Vì sao bootstrap theo participant, không theo cặp

Một người có nhiều lesion, một lesion có nhiều cặp; các cặp của cùng người tương quan với nhau (cùng da, cùng máy chụp, cùng người chụp). Nếu lấy mẫu lại **từng cặp**, ta coi chúng là độc lập, nên CI hẹp hơn thực tế. `group_bootstrap` lấy mẫu lại **cả người**: chọn ngẫu nhiên có hoàn lại N participant, rồi ghép **mọi** cặp của những người được chọn.

Ví dụ trong `test_group_bootstrap_resamples_groups`: nhóm A có 100 dòng giá trị 0, nhóm B có 1 dòng giá trị 1.
- Lấy mẫu theo **dòng**: hầu như lần nào cũng ~1% số 1, CI hẹp quanh 0,01, trông rất "chắc chắn".
- Lấy mẫu theo **nhóm**: có lần chỉ chọn A (mean 0), có lần chỉ chọn B (mean 1), nên CI là [0; 1]. Kết quả này phản ánh đúng việc ta chỉ có **2 người**.

Với ISIC không có `participant_id`/`lesion_id` công khai, nên đơn vị bootstrap là **ảnh**; ghi rõ trong Methods.

### 3.3. Paired bootstrap: cách đọc hiệu Ridge − baseline

Ở mỗi lần lặp, **cùng một** mẫu participant được dùng để tính MAE của Ridge và của baseline, rồi lấy hiệu. Cách này loại bỏ phần biến thiên do "lần này lấy phải nhóm người khó dự báo", nên chính xác hơn việc so hai CI rời.

| CI 95% của `MAE_Ridge − MAE_baseline` | Kết luận được phép viết |
|---|---|
| Toàn bộ < 0 | Ridge có MAE thấp hơn baseline trên participant test, trong phạm vi dữ liệu này |
| Chứa 0 | Chưa đủ bằng chứng Ridge khác baseline. **Vẫn là kết quả khoa học**, báo trung thực và phân tích lý do |
| Toàn bộ > 0 | Ridge kém hơn baseline |

Ví dụ thật trên **dữ liệu giả** (P6 mục 4.6, chỉ có 4 participant test): `Ridge − baseline MAE +0,0014 [−0,0014; +0,0056]`. CI chứa 0 vì quá ít người. Đây đúng là tình huống phải viết "chưa đủ bằng chứng", dù MAE val của Ridge thấp hơn.

### 3.4. Ngưỡng "ổn định" `stable_eps`

Độ đúng hướng (tăng / giảm / ổn định) cần một ngưỡng: |Δa| ≤ `stable_eps` được coi là "ổn định". Ngưỡng này phải chốt **trên val** hoặc từ độ bất đồng giữa người gán, **trước** khi mở test (kế hoạch 7.4). Hai cách, chọn một và ghi vào protocol:

1. **Trung vị |Δa| trên val** (cell dưới).
2. **Độ bất đồng diện tích giữa hai người gán** ở P4: trung vị `area_diff` trong `agreement.csv`. Ý nghĩa: thay đổi nhỏ hơn mức hai người gán đã lệch nhau thì không phân biệt được với nhiễu đo.

```python
# cell: NB_cpu (máy cá nhân)
import pandas as pd
val = pd.read_csv(f"{RUN}/ridge/predictions_val.csv")
print("Phương án 1 — trung vị |Δa| val:", round(val.delta_area.abs().median(), 4))
# Phương án 2 (nếu đã có P4):
# print("Phương án 2 — trung vị area_diff:", round(pd.read_csv(f"{ROOT}/runs/<audit_run>/agreement.csv").area_diff.median(), 4))
```

### 3.5. Seed và CI là hai loại biến thiên khác nhau

Nếu chạy 3 seed cho segmentation/classification (P5a/P5b):
- Báo **mean ± SD qua 3 seed**: đây là độ dao động do huấn luyện.
- Báo **CI bootstrap của từng seed**: đây là độ bất định do mẫu test.
- Không gộp hai loại này thành một con số (kế hoạch 7.5).

## 4. Code

### 4.1. Metric dùng chung

`src/nckh/metrics.py` đã tạo ở P5a (mục 4.1). Phase này chỉ thêm script đánh giá dự báo.

### 4.2. `scripts/evaluate_forecast.py`

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/evaluate_forecast.py`

```python
# file: scripts/evaluate_forecast.py
"""Đánh giá dự báo Δa: Ridge vs baseline Δa = 0, CI bootstrap theo participant, hiệu ghép cặp, theo nhóm Δt.

Ví dụ: python scripts/evaluate_forecast.py --predictions $RUN/ridge_test/predictions_test.csv \
           --out-dir $RUN/eval_forecast --stable-eps 0.005
"""
import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from nckh.metrics import direction_accuracy, group_bootstrap, paired_bootstrap_diff, regression_metrics

MIN_BIN_PAIRS = 20


def _err(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda d: d[col] - d["delta_area"]


def evaluate_forecast(pred_csv: Path, out_dir: Path, stable_eps: float, n_boot: int = 2000, seed: int = 2026) -> dict:
    df = pd.read_csv(pred_csv)
    if df["split"].nunique() != 1:
        # Trộn val và test trong cùng một bảng sẽ làm số liệu test bị "pha loãng".
        raise ValueError(f"File có nhiều split: {sorted(df['split'].unique())}; chỉ đánh giá một split mỗi lần")
    stats = {
        "mae": lambda c: (lambda d: float(np.abs(_err(c)(d)).mean())),
        "rmse": lambda c: (lambda d: float(np.sqrt((_err(c)(d) ** 2).mean()))),
        "bias": lambda c: (lambda d: float(_err(c)(d).mean())),
    }
    report: dict = {"split": str(df["split"].iloc[0]), "n_pairs": int(len(df)),
                    "n_participants": int(df["participant_id"].nunique()), "stable_eps": stable_eps,
                    "stable_eps_source": "chọn trên val (ghi trong protocol)"}
    for name, col in (("ridge", "pred_ridge"), ("baseline", "pred_baseline")):
        point = regression_metrics(df["delta_area"], df[col])
        report[name] = {m: group_bootstrap(df, "participant_id", f(col), n_boot=n_boot, seed=seed) for m, f in stats.items()}
        report[name]["r2"] = point["r2"]
        report[name]["direction_accuracy"] = direction_accuracy(df["delta_area"], df[col], stable_eps)
    # Cùng tập participant được lấy mẫu cho cả hai phương pháp → CI của hiệu, không phải hai CI rời.
    report["diff_ridge_minus_baseline"] = {
        m: paired_bootstrap_diff(df, "participant_id", stats[m]("pred_ridge"), stats[m]("pred_baseline"),
                                 n_boot=n_boot, seed=seed)
        for m in ("mae", "rmse")
    }

    df["dt_bin"] = pd.qcut(df["delta_days"], 3, labels=["Δt thấp", "Δt giữa", "Δt cao"], duplicates="drop")
    bins = []
    for label, d in df.groupby("dt_bin", observed=True):
        enough = len(d) >= MIN_BIN_PAIRS
        bins.append({"bin": label, "dt_min": d["delta_days"].min(), "dt_max": d["delta_days"].max(), "n_pairs": len(d),
                     "mae_ridge": float(np.abs(d.pred_ridge - d.delta_area).mean()) if enough else np.nan,
                     "mae_baseline": float(np.abs(d.pred_baseline - d.delta_area).mean()) if enough else np.nan,
                     "note": "" if enough else f"<{MIN_BIN_PAIRS} cặp"})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bins).to_csv(out_dir / "forecast_by_dt_bin.csv", index=False)
    (out_dir / "forecast_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stable-eps", type=float, required=True, help="ngưỡng |Δa| coi là ổn định, chọn trên val")
    args = ap.parse_args(argv)
    res = evaluate_forecast(args.predictions, args.out_dir, args.stable_eps)
    for name in ("ridge", "baseline"):
        mae = res[name]["mae"]
        print(f"{name:9s} MAE {mae['estimate']:.4f} [{mae['ci_low']:.4f}, {mae['ci_high']:.4f}]")
    d = res["diff_ridge_minus_baseline"]["mae"]
    print(f"Ridge − baseline MAE {d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
```

### 4.3. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_evaluate_forecast.py`

```python
# file: tests/test_evaluate_forecast.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_forecast import evaluate_forecast  # noqa: E402


def _pred(tmp_path: Path, n_people: int = 40, ridge_err: float = 0.002, split: str = "test") -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_people):
        for k in range(2):
            target = rng.normal(0.01, 0.02)
            rows.append({"participant_id": f"P{p}", "lesion_id": f"P{p}_L{k}", "image_id_t": f"P{p}_{k}", "split": split,
                         "delta_days": rng.uniform(150, 210), "area_ratio_t": 0.2, "delta_area": target,
                         "pred_ridge": target + rng.choice([-1, 1]) * ridge_err, "pred_baseline": 0.0,
                         "next_area_ridge": 0.2 + target})
    path = tmp_path / "predictions_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_evaluate_forecast_keys(tmp_path: Path) -> None:
    res = evaluate_forecast(_pred(tmp_path), tmp_path / "out", stable_eps=0.005, n_boot=200)
    assert res["n_pairs"] == 80 and res["n_participants"] == 40 and res["stable_eps"] == 0.005
    for method in ("ridge", "baseline"):
        assert set(res[method]["mae"]) >= {"estimate", "ci_low", "ci_high"}
        assert "direction_accuracy" in res[method] and "r2" in res[method]
    assert json.loads((tmp_path / "out" / "forecast_metrics.json").read_text())["split"] == "test"


def test_diff_negative_when_ridge_better(tmp_path: Path) -> None:
    res = evaluate_forecast(_pred(tmp_path), tmp_path / "out", stable_eps=0.005, n_boot=300)
    assert res["ridge"]["mae"]["estimate"] == pytest.approx(0.002)
    assert res["diff_ridge_minus_baseline"]["mae"]["ci_high"] < 0


def test_requires_single_split(tmp_path: Path) -> None:
    path = _pred(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, "split"] = "val"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="split"):
        evaluate_forecast(path, tmp_path / "out", stable_eps=0.005, n_boot=10)


def test_bin_min_size(tmp_path: Path) -> None:
    evaluate_forecast(_pred(tmp_path, n_people=20), tmp_path / "out", stable_eps=0.005, n_boot=10)
    bins = pd.read_csv(tmp_path / "out" / "forecast_by_dt_bin.csv")
    assert len(bins) == 3 and bins["n_pairs"].sum() == 40
    small = bins[bins["n_pairs"] < 20]
    assert small["mae_ridge"].isna().all() and (small["note"] == "<20 cặp").all()
```

### 4.4. Chạy

```python
# cell: NB_cpu (máy cá nhân)
EPS = 0.005   # ← giá trị đã chốt trong protocol (mục 3.4), KHÔNG chọn lại sau khi xem test
!cd {REPO} && {PY} scripts/evaluate_forecast.py --predictions {RUN}/ridge_test/predictions_test.csv --out-dir {RUN}/eval_forecast --stable-eps {EPS}
!cat {RUN}/eval_forecast/forecast_by_dt_bin.csv
```

Chạy thêm trên `predictions_val.csv` (thư mục `eval_forecast_val`) để có số val đặt cạnh số test.

### 4.5. Chạy toàn bộ test và lưu log

```python
# cell: NB_cpu (máy cá nhân)
from nckh.runcard import new_run_id
LOG = f"{ROOT}/runs/{new_run_id('pytest')}"
!mkdir -p {LOG} && cd {REPO} && {PY} -m pytest -q 2>&1 | tee {LOG}/pytest.log | tail -3
```

## 5. Test

### 5.1. Bảng kiểm tra của kế hoạch (Mục 7.1) → test/script cụ thể

| Nhóm | Phép kiểm | Test / script | Kết quả lưu |
|---|---|---|---|
| Tệp | Ảnh/mask mở được, kích thước đúng, hash trùng được xử lý | `test_manifest.py::test_exclude_reasons`, `::test_cross_split_duplicate_found`; `build_manifest.py` | `isic2018_seg.csv` (`exclude_reason`), `.cross_split_duplicates.csv` |
| Metadata | ID, ngày chụp, nhãn, thiếu | `test_manifest.py::test_load_uq_metadata_maps_and_validates`; `test_prepare.py::test_isic2017_labels` | `flow.json`, P1 data dictionary |
| Split | Không trùng participant/lesion giữa tập | `test_manifest.py::test_group_split_*`, `::test_leakage_detected`; `test_pipeline_fake.py::test_no_participant_in_two_splits` | `assert_no_group_leakage` trong `build_pairs.py` |
| Thời gian | Cặp cùng lesion, timestamp tăng, Δt hợp lệ | `test_pairs.py` (13 test) | `pairs.csv`, `pairs_excluded.csv` |
| Pipeline | Input/output đúng shape, output hữu hạn | `test_infer.py`, `test_bench.py`; `bench_inference.py` trên dữ liệu thật | `bench_*.json`, run card |
| Tái lập | Chạy lại cùng cấu hình cho cùng kết quả | `test_bootstrap_deterministic`, `test_group_split_disjoint_and_deterministic`; so SHA-256 manifest, `ridge_summary.json` giữa 2 lần chạy | run card |
| Demo | Ảnh hỏng, mask rỗng, Δt sai | `test_forecast.py::test_validate_delta_days_rejects`; P9 `tests/test_demo_pipeline.py` | log kiểm thử P9 |

### 5.2. Danh sách test (đến hết P7)

| File | Số test | Phase |
|---|---:|---|
| `tests/test_paths_runcard.py` | 5 | P2 |
| `tests/test_inspect_checkpoint.py` | 3 | P2 |
| `tests/test_infer.py` | 8 | P2 |
| `tests/test_bench.py` | 3 | P2 |
| `tests/test_manifest.py` | 8 | P3 |
| `tests/test_prepare.py` | 6 | P3 |
| `tests/test_metrics.py` | 11 | P5a |
| `tests/test_eval_cls_agreement.py` | 5 | P5b/P4 |
| `tests/test_pairs.py` | 13 | P6 |
| `tests/test_features.py` | 8 | P6 |
| `tests/test_forecast.py` | 20 | P6 |
| `tests/test_pipeline_fake.py` | 6 | P6 |
| `tests/test_evaluate_forecast.py` | 4 | P7 |
| **Tổng** | **100** | |

P8, P9, P10 thêm test của riêng các phase đó. Nếu môi trường không có torch, `test_infer.py` và `test_bench.py` báo `skipped`; ghi rõ điều đó trong log.

## 6. Benchmark / đánh giá

Bảng kết quả theo kế hoạch Mục 10. **Không điền trước số kỳ vọng.** Nếu Ridge không vượt baseline hoặc CI rộng, giữ nguyên số thật và điều chỉnh kết luận.

| Thử nghiệm | N test | Metric | Kết quả | CI 95% | Ghi chú |
|---|---:|---|---:|---|---|
| PanDerm segmentation — ISIC 2018 | [điền sau khi chạy] | Dice / IoU | [điền sau khi chạy] | [điền sau khi chạy] | 1 mask/ảnh; metric 224×224 |
| PanDerm segmentation — UQ audit | [điền sau khi chạy] | Dice / IoU / sai số diện tích | [điền sau khi chạy] | [điền sau khi chạy] | Số ảnh gán tay: [điền sau khi chạy] |
| Nhãn tham khảo — ISIC 2017 | 600 | Macro-F1 / BAcc / AUROC | [điền sau khi chạy] | [điền sau khi chạy] | 117 / 393 / 90 |
| Ridge dự báo Δa — UQ, mức A | [điền sau khi chạy] cặp, [điền sau khi chạy] người | MAE / RMSE / bias | [điền sau khi chạy] | [điền sau khi chạy] | Target = mask PanDerm |
| Baseline Δa = 0 — UQ, mức A | như trên | MAE / RMSE / bias | [điền sau khi chạy] | [điền sau khi chạy] | Cùng participant test |
| Ridge − baseline (paired) | như trên | ΔMAE / ΔRMSE | [điền sau khi chạy] | [điền sau khi chạy] | Kết quả chính |
| Ridge — mức B (thủ công) | [điền sau khi chạy] | MAE / RMSE / bias | [điền sau khi chạy] | [điền sau khi chạy] | Tách riêng, mẫu nhỏ |
| Độ bền ảnh | [điền sau khi chạy] | ΔDice / ΔMacro-F1 / ΔMAE | [điền sau khi chạy] | [điền sau khi chạy] | Từng phép suy giảm (P8) |

## 7. Lỗi thường gặp (máy / Colab extension)

| Triệu chứng | Cách xử lý |
|---|---|
| `rclone` báo `couldn't fetch token` | `rclone config reconnect gdrive:` |
| Ở máy không thấy file Colab vừa ghi (`seg_metrics.json`, `cls_metrics.json`) | Chạy lệnh `rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id>` |
| `ValueError: File có nhiều split` | Đang đưa file trộn val+test; dùng đúng `predictions_test.csv` |
| CI rất rộng | Ít participant test; báo cỡ mẫu hiệu dụng, gọi kết quả là thăm dò (kế hoạch Phase 1) |
| Cảnh báo `n lần bootstrap không tính được thống kê` | Thường do AUROC khi một lần lặp thiếu lớp; báo `n_failed` trong bảng |
| Số trong bảng khác lần chạy trước | Kiểm tra seed (2026), phiên bản thư viện trong run card, SHA-256 của file dự báo |

## 8. Checklist bàn giao cho SV A

- [ ] `stable_eps` và cách chọn đã ghi trong protocol **trước** khi chạy test.
- [ ] `eval_forecast/forecast_metrics.json` + `forecast_by_dt_bin.csv` cho test; bản tương ứng cho val.
- [ ] `runs/<pytest>/pytest.log`: toàn bộ test pass (ghi rõ nếu có `skipped`).
- [ ] Bảng mục 6 đã điền các dòng có dữ liệu; SV A tự tính lại MAE từ `predictions_test.csv` và khớp.
