# P10 — Tái lập, phân tích lỗi và sinh bảng/hình

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 10 + Mục 7):** mọi bảng và hình trong bài báo được **sinh bằng script** từ file kết quả, truy ngược được tới một `run_id`. Phân tích lỗi theo quy tắc chọn ví dụ định trước. Một người **chưa** chạy thí nghiệm chính (SV A) phải tái lập được kết quả từ môi trường sạch.

| Đầu vào | Đầu ra (`runs/<p10_run>/`) | Sinh ở |
|---|---|---|
| `seg_metrics.json`, `cls_metrics.json`, `forecast_metrics.json`, `flow.json`, `robustness.json`, `predictions_test.csv` | `tables/table_{seg,cls,forecast,flow,robustness}.{csv,md}` | Laptop (`.venv`): `~/nckh_root/runs/<p10_run>/tables/` |
| `seg_per_image.csv`, overlay P5a, `predictions_test.csv` | `figures/{pred_vs_obs,residual_vs_dt,robustness}.png`, `examples_seg.csv`, `examples_forecast.csv` | Laptop (`.venv`): `~/nckh_root/runs/<p10_run>/` |

## 2. Chạy ở đâu

Toàn bộ trên laptop bằng `.venv` (`nb_cpu`, chạy cell setup P2 4.1a trước). Mỗi lệnh chạy vài giây.

Các run P5a/P5b/P8 được tạo trên máy GPU: chép về laptop theo `00` mục 5.2 (lệnh có `--exclude '*.ckpt' --exclude '*.pth'`). Run có dữ liệu UQ (P6, P7) chỉ chép khi laptop được phép giữ UQ (P6 mục 3.1); nếu không, chạy phần bảng/hình dự báo trên máy giữ UQ.

## 3. Giải thích

### 3.1. Quy tắc lưu kết quả

- Mỗi lần chạy có **một thư mục riêng** `runs/<YYYYmmdd-HHMMSS>_<tag>/` (tạo bằng `new_run_id`) và **một `run_card.json`**. Run card ghi: commit git, phiên bản gói, GPU, seed, cấu hình, SHA-256 của mọi file đầu vào.
- **Không ghi đè.** Muốn chạy lại thì tạo run mới. Run cũ giữ làm bằng chứng.
- Bảng/hình **không** nhập số bằng tay: luôn chạy `make_tables.py` / `make_figures.py` trên file JSON/CSV của run đã khóa.

Cây thư mục `runs/` điển hình:

```
runs/
├── 20261006-101500_smoke_seg/        bench_seg.json, overlays/, run_card.json          (P2)
├── 20261020-090000_seg_main/         0/model_best_0.ckpt, metrics.csv, eval/…           (P5a)
├── 20261021-140000_cls_main/         checkpoint-best.pth, eval_test/…                   (P5b)
├── 20261101-090000_p6_uq/            pairs.csv, flow.json, ridge/, ridge_test/, eval_forecast/  (P6/P7)
├── 20261110-090000_p8_seg/ …         robustness.json                                    (P8)
└── 20261120-090000_p10/              tables/, figures/, examples_*.csv, run_card.json   (P10)
```

### 3.2. Bảng truy vết (8 hình/bảng tối thiểu của kế hoạch Phase 10)

| # | Hình/bảng | Script | Đầu vào | run_id |
|---|---|---|---|---|
| 1 | Sơ đồ dòng dữ liệu | Vẽ tay theo `00_tong_quan…` mục 3; số liệu lấy từ `table_flow` | `flow.json`, `ridge_summary.json` | [điền sau khi chạy] |
| 2 | Đặc điểm dữ liệu theo split | `make_tables.py` → `table_flow` | `flow.json`, manifest P3 | [điền sau khi chạy] |
| 3 | Segmentation ISIC + audit UQ | `make_tables.py` → `table_seg`; audit: `agreement.csv` | `seg_metrics.json` | [điền sau khi chạy] |
| 4 | Classification | `make_tables.py` → `table_cls`; confusion matrix có trong `cls_metrics.json` | `cls_metrics.json` | [điền sau khi chạy] |
| 5 | Forecast mức A (và mức B riêng) | `make_tables.py` → `table_forecast` | `forecast_metrics.json` | [điền sau khi chạy] |
| 6 | Biểu đồ robustness | `make_figures.py` → `robustness.png`; `make_tables.py` → `table_robustness` | `robustness.json` (seg/cls/forecast) | [điền sau khi chạy] |
| 7 | Predicted-vs-observed, residual-vs-Δt | `make_figures.py` | `predictions_test.csv` | [điền sau khi chạy] |
| 8 | Hình định tính: thành công và thất bại | `select_examples` (mục 4.4) + overlay P5a | `seg_per_image.csv`, `results_ISIC2018_0/` | [điền sau khi chạy] |

### 3.3. Chọn ví dụ phân tích lỗi theo quy tắc, không chọn ảnh "đẹp"

`select_examples` lấy **3 ví dụ gần nhất** với phân vị 90 (tốt), 50 (trung bình), 10 (kém) của một điểm số. Khi hai ví dụ cùng điểm, thứ tự do seed 2026 quyết định, không do người chọn.
- **Segmentation:** điểm = Dice từng ảnh.
- **Dự báo:** điểm = −|phần dư|, vì phần dư nhỏ là tốt.

Với mỗi ví dụ, ghi nguyên nhân **quan sát được**: blur, lông, tương phản thấp, tổn thương nhỏ, biên mờ, màu lệch, mask cắt biên. Với dự báo, tách hai nguồn lỗi: mask sai ở t/t+1 (lỗi segmentation làm sai target) và Ridge dự báo sai (kế hoạch Phase 10).

## 4. Code

### 4.1. `scripts/make_tables.py`

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/make_tables.py`

```python
# file: scripts/make_tables.py
"""Sinh bảng kết quả (CSV + Markdown) từ các file JSON đánh giá — không gõ số bằng tay.

Ví dụ: python scripts/make_tables.py --seg $R/seg_metrics.json --cls $R/cls_metrics.json --forecast $R/forecast_metrics.json \
           --flow $R/flow.json --robustness $R/robustness.json --out-dir $ROOT/runs/<run_id>/tables
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("make_tables")
MISSING = pd.DataFrame({"ghi chú": ["chưa có dữ liệu"]})


def fmt(x: float | None) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.3f}"


def fmt_ci(d: dict | None) -> str:
    if not d:
        return "—"
    return f"{fmt(d['estimate'])} [{fmt(d['ci_low'])}, {fmt(d['ci_high'])}]"


def _to_markdown(df: pd.DataFrame) -> str:
    # Tự viết thay cho DataFrame.to_markdown để không phải thêm gói tabulate.
    cells = [[str(c) for c in df.columns]] + [[str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines) + "\n"


def _load(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        logger.warning("Thiếu file %s → bảng ghi 'chưa có dữ liệu'", path)
        return None
    return json.loads(Path(path).read_text())


def _seg(d: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Chỉ số": "Dice", "Ước lượng [CI 95%]": fmt_ci(d["dice"]), "N ảnh": d["n_images"]},
                         {"Chỉ số": "IoU", "Ước lượng [CI 95%]": fmt_ci(d["iou"]), "N ảnh": d["n_images"]}])


def _cls(d: dict) -> pd.DataFrame:
    rows = [{"Mục": "Macro-F1", "Giá trị": fmt_ci(d.get("macro_f1_ci")) if d.get("macro_f1_ci") else fmt(d["macro_f1"])},
            {"Mục": "Balanced accuracy", "Giá trị": fmt(d["balanced_accuracy"])},
            {"Mục": "AUROC macro", "Giá trị": fmt(d["auroc_macro"])},
            {"Mục": "Brier", "Giá trị": fmt(d.get("brier"))}]
    for name, n in d["n_per_class"].items():
        rows.append({"Mục": f"{name} (n={n})",
                     "Giá trị": f"recall {fmt(d['recall_per_class'][name])}, AUROC {fmt(d['auroc_per_class'][name])}"})
    return pd.DataFrame(rows)


def _forecast(d: dict) -> pd.DataFrame:
    rows = []
    for key, label in (("ridge", "Ridge"), ("baseline", "Baseline Δa = 0")):
        m = d[key]
        rows.append({"Phương pháp": label, "MAE [CI]": fmt_ci(m["mae"]), "RMSE [CI]": fmt_ci(m["rmse"]),
                     "Bias [CI]": fmt_ci(m["bias"]), "R²": fmt(m.get("r2")), "Đúng hướng": fmt(m.get("direction_accuracy"))})
    diff = d["diff_ridge_minus_baseline"]
    rows.append({"Phương pháp": "Ridge − baseline (paired)", "MAE [CI]": fmt_ci(diff["mae"]), "RMSE [CI]": fmt_ci(diff["rmse"]),
                 "Bias [CI]": "—", "R²": "—", "Đúng hướng": "—"})
    out = pd.DataFrame(rows)
    out["N cặp / người"] = f"{d['n_pairs']} / {d['n_participants']}"
    return out


def _flatten(d: dict, prefix: str = "") -> list[dict]:
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        rows += _flatten(v, key + ".") if isinstance(v, dict) else [{"Mục": key, "Giá trị": v}]
    return rows


def _robustness(d: dict) -> pd.DataFrame:
    metrics = sorted({m for level in d.values() for m in level})
    return pd.DataFrame([{"Mức suy giảm": lv, **{m: fmt_ci(v.get(m)) for m in metrics}} for lv, v in d.items()])


BUILDERS = {"seg": _seg, "cls": _cls, "forecast": _forecast, "flow": lambda d: pd.DataFrame(_flatten(d)),
            "robustness": _robustness}


def make_tables(inputs: dict[str, Path | None], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in BUILDERS.items():
        data = _load(inputs.get(name))
        table = MISSING if data is None else build(data)
        csv, md = out_dir / f"table_{name}.csv", out_dir / f"table_{name}.md"
        table.to_csv(csv, index=False)
        md.write_text(_to_markdown(table), encoding="utf-8")
        written += [csv, md]
    return written


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    for name in BUILDERS:
        ap.add_argument(f"--{name}", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    for path in make_tables({n: getattr(args, n) for n in BUILDERS}, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
```

### 4.2. `scripts/make_figures.py`

📁 **Tạo trên máy cá nhân:** `<repo>/scripts/make_figures.py`

```python
# file: scripts/make_figures.py
"""Sinh hình cho bài báo từ file dự báo và robustness.json; chọn ví dụ phân tích lỗi theo quy tắc định trước.

Ví dụ: python scripts/make_figures.py --predictions $R/predictions_test.csv --robustness $R/robustness.json --out-dir $R/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chạy từ terminal không có màn hình
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GROUPS = (("tốt (p90)", 0.9), ("trung bình (p50)", 0.5), ("kém (p10)", 0.1))


def select_examples(df: pd.DataFrame, score_col: str, n: int = 3, seed: int = 2026) -> pd.DataFrame:
    """Chọn n ví dụ gần phân vị 90/50/10 của score_col (cao = tốt) — quy tắc cố định, không chọn ảnh "đẹp"."""
    shuffled = df.sample(frac=1, random_state=seed)  # hòa điểm thì thứ tự do seed quyết định, không do tay chọn
    picks = []
    for label, q in GROUPS:
        target = df[score_col].quantile(q)
        nearest = (shuffled[score_col] - target).abs().sort_values(kind="stable").index[:n]
        picks.append(df.loc[nearest].assign(group=label))
    return pd.concat(picks)


def _scatter(pred: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(pred["delta_area"], pred["pred_ridge"], s=12, alpha=0.7)
    lim = [min(pred["delta_area"].min(), pred["pred_ridge"].min()), max(pred["delta_area"].max(), pred["pred_ridge"].max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("Δa quan sát (tỷ lệ diện tích mask)")
    ax.set_ylabel("Δa dự báo (Ridge)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _residual(pred: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(pred["delta_days"], pred["pred_ridge"] - pred["delta_area"], s=12, alpha=0.7)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Δt (ngày)")
    ax.set_ylabel("Phần dư (dự báo − quan sát)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _robustness(rob: dict, out: Path) -> Path:
    metrics = sorted({m for level in rob.values() for m in level})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 0.45 * len(rob) + 1.5), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        levels = [lv for lv in rob if m in rob[lv]]
        est = np.array([rob[lv][m]["estimate"] for lv in levels])
        err = np.array([[est[i] - rob[lv][m]["ci_low"], rob[lv][m]["ci_high"] - est[i]] for i, lv in enumerate(levels)]).T
        ax.errorbar(est, range(len(levels)), xerr=err, fmt="o", capsize=3)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(levels)), levels)
        ax.set_title(f"{m} (suy giảm − sạch)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_figures(pred_csv: Path, robustness_json: Path | None, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(pred_csv)
    paths = [_scatter(pred, out_dir / "pred_vs_obs.png"), _residual(pred, out_dir / "residual_vs_dt.png")]
    if robustness_json is not None and Path(robustness_json).exists():
        paths.append(_robustness(json.loads(Path(robustness_json).read_text()), out_dir / "robustness.png"))
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--robustness", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    for path in make_figures(args.predictions, args.robustness, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
```

### 4.3. Test

📁 **Tạo trên máy cá nhân:** `<repo>/tests/test_tables_figures.py`

```python
# file: tests/test_tables_figures.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_figures import make_figures, select_examples  # noqa: E402
from make_tables import make_tables  # noqa: E402


def _ci(e: float, lo: float, hi: float) -> dict:
    return {"estimate": e, "ci_low": lo, "ci_high": hi}


def _write(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj))
    return path


def test_tables_from_fake_jsons(tmp_path: Path) -> None:
    inputs = {
        "seg": _write(tmp_path / "seg.json", {"dice": _ci(0.85, 0.84, 0.86), "iou": _ci(0.76, 0.75, 0.78), "n_images": 1000}),
        "cls": _write(tmp_path / "cls.json", {
            "macro_f1": 0.7, "macro_f1_ci": _ci(0.7, 0.65, 0.75), "balanced_accuracy": 0.72, "auroc_macro": 0.88,
            "brier": 0.3, "n_per_class": {"melanoma": 117, "nevus": 393, "seborrheic_keratosis": 90},
            "recall_per_class": {"melanoma": 0.6, "nevus": 0.8, "seborrheic_keratosis": 0.75},
            "auroc_per_class": {"melanoma": 0.8, "nevus": 0.85, "seborrheic_keratosis": 0.95}}),
        "forecast": _write(tmp_path / "fc.json", {
            "n_pairs": 120, "n_participants": 30,
            "ridge": {"mae": _ci(0.015, 0.010, 0.020), "rmse": _ci(0.02, 0.015, 0.025), "bias": _ci(0.001, -0.002, 0.004),
                      "r2": 0.1, "direction_accuracy": 0.6},
            "baseline": {"mae": _ci(0.018, 0.012, 0.024), "rmse": _ci(0.025, 0.02, 0.03), "bias": _ci(-0.01, -0.012, -0.008),
                         "r2": -0.2, "direction_accuracy": 0.4},
            "diff_ridge_minus_baseline": {"mae": _ci(-0.003, -0.006, 0.0), "rmse": _ci(-0.005, -0.008, -0.001)}}),
        "flow": _write(tmp_path / "flow.json", {"n_images": 142, "excluded_by_reason": {"same_timestamp": 1},
                                                "n_pairs_by_split": {"train": 65, "val": 13, "test": 13}}),
        "robustness": _write(tmp_path / "rob.json", {"blur_1.0": {"delta_dice": _ci(-0.01, -0.02, 0.0)}}),
    }
    paths = make_tables(inputs, tmp_path / "out")
    assert {p.name for p in paths} >= {"table_seg.md", "table_cls.csv", "table_forecast.md", "table_flow.csv",
                                       "table_robustness.md"}
    assert "0.015 [0.010, 0.020]" in (tmp_path / "out" / "table_forecast.md").read_text()
    assert "excluded_by_reason.same_timestamp" in (tmp_path / "out" / "table_flow.md").read_text()


def test_tables_missing_json_marked(tmp_path: Path) -> None:
    make_tables({"seg": None, "cls": tmp_path / "nope.json", "forecast": None, "flow": None, "robustness": None},
                tmp_path / "out")
    assert "chưa có dữ liệu" in (tmp_path / "out" / "table_cls.md").read_text()


def test_figures_created(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pred = pd.DataFrame({"delta_area": rng.normal(0, 0.02, 50), "delta_days": rng.uniform(150, 210, 50)})
    pred["pred_ridge"] = pred["delta_area"] + rng.normal(0, 0.005, 50)
    pred.to_csv(tmp_path / "pred.csv", index=False)
    rob = _write(tmp_path / "rob.json", {"blur_1.0": {"delta_dice": _ci(-0.01, -0.02, 0.0)},
                                         "hair_0.03": {"delta_dice": _ci(-0.05, -0.07, -0.03)}})
    paths = make_figures(tmp_path / "pred.csv", rob, tmp_path / "fig")
    assert [p.name for p in paths] == ["pred_vs_obs.png", "residual_vs_dt.png", "robustness.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_select_examples_groups_and_seed() -> None:
    df = pd.DataFrame({"image_id": [f"I{i}" for i in range(100)], "dice": np.linspace(0, 1, 100)})
    a = select_examples(df, "dice", n=3, seed=2026)
    b = select_examples(df, "dice", n=3, seed=2026)
    assert a.equals(b) and len(a) == 9
    assert set(a["group"]) == {"tốt (p90)", "trung bình (p50)", "kém (p10)"}
    assert a.loc[a.group == "kém (p10)", "dice"].max() < a.loc[a.group == "tốt (p90)", "dice"].min()
```

### 4.4. Chạy

```python
# cell: nb_cpu
from nckh.runcard import new_run_id
P10 = f"{ROOT}/runs/{new_run_id('p10')}"
SEG = f"{ROOT}/runs/<seg_main_run_id>/eval"
CLS = f"{ROOT}/runs/<cls_main_run_id>/eval_test"
UQ = f"{ROOT}/runs/<p6_uq_run_id>"
ROB = f"{ROOT}/runs/<p8_seg_run_id>/robustness.json"      # lặp lại make_tables/figures cho p8_cls, p8_forecast nếu muốn tách
!cd {REPO} && {PY} scripts/make_tables.py --seg {SEG}/seg_metrics.json --cls {CLS}/cls_metrics.json --forecast {UQ}/eval_forecast/forecast_metrics.json --flow {UQ}/flow.json --robustness {ROB} --out-dir {P10}/tables
!cd {REPO} && {PY} scripts/make_figures.py --predictions {UQ}/ridge_test/predictions_test.csv --robustness {ROB} --out-dir {P10}/figures
!{PY} -m nckh.runcard {P10} --seed 2026 --input seg={SEG}/seg_metrics.json --input cls={CLS}/cls_metrics.json --input forecast={UQ}/eval_forecast/forecast_metrics.json --input predictions={UQ}/ridge_test/predictions_test.csv --input robustness={ROB}
```

```python
# cell: nb_cpu
import sys, pandas as pd
sys.path.insert(0, f"{REPO}/scripts")
from make_figures import select_examples
seg = pd.read_csv(f"{SEG}/seg_per_image.csv")
select_examples(seg, "dice", n=3, seed=2026).to_csv(f"{P10}/examples_seg.csv", index=False)
pred = pd.read_csv(f"{UQ}/ridge_test/predictions_test.csv")
pred["neg_abs_residual"] = -(pred.pred_ridge - pred.delta_area).abs()
select_examples(pred, "neg_abs_residual", n=3, seed=2026).to_csv(f"{P10}/examples_forecast.csv", index=False)
print(pd.read_csv(f"{P10}/examples_seg.csv")[["group", "name", "dice"]])
```

Overlay của từng ảnh ví dụ segmentation nằm ở `runs/<seg_main>/results_ISIC2018_0/<name>.png` (P5a mục 4.7). Ví dụ dự báo trên UQ: chỉ xem mask trong `masks/`; **không** đưa ảnh UQ vào bài nếu điều khoản không cho phép.

### 4.5. Kiểm tra tái lập cuối (kế hoạch Mục 7), do SV A làm

Từ một máy **mới** có GPU (clone repo, làm `00` mục 6b), chỉ làm theo docs:

1. **Đọc manifest (`nb_cpu`):** chạy P3 mục 4.6, rồi so SHA-256 `isic2018_seg.csv` với giá trị ghi trong protocol.
2. **Nạp checkpoint (máy GPU, `venv_seg`):** dựng `venv_seg` (P2 4.4), áp patch (P2 4.6), chạy `bench_inference.py --finetuned <model_best_0.ckpt>` trên 20 ảnh val. Kỳ vọng `SMOKE TEST OK`.
3. **Chạy một inference mẫu (máy GPU):** demo P9 mục 4.5 với một ảnh ISIC val, ghi lại tỷ lệ mask và xác suất.
4. **Tạo một metric nhỏ (`.venv`, sau khi chép run về):** `evaluate_seg.py` trên file xlsx của run chính. Kỳ vọng Dice/IoU **trùng** `seg_metrics.json` (cùng seed bootstrap). Chạy `evaluate_forecast.py` trên `predictions_test.csv`: kỳ vọng MAE trùng.
5. **Xác nhận định dạng bảng cuối:** chạy mục 4.4 vào một `run_id` mới, rồi `diff` các file `table_*.md` với bản của SV B. Kỳ vọng giống hệt.

Nếu bước nào không tái lập được: sửa docs, hoặc ghi giới hạn vào bài **trước** khi nộp.

## 5. Test

```python
# cell: nb_cpu
!cd {REPO} && {PY} -m pytest -q
```

Kỳ vọng cho **toàn bộ dự án**: `158 passed` (hoặc một phần `skipped` nếu `.venv` không có torch).

| File | Số test |
|---|---:|
| Đến hết P7 (xem P7 mục 5.2) | 102 |
| `tests/test_degrade.py` + `tests/test_robustness.py` (P8) | 37 |
| `tests/test_demo_pipeline.py` (P9) | 15 |
| `tests/test_tables_figures.py` (P10) | 4 |
| **Tổng** | **158** |

Test P10 kiểm: bảng sinh đúng từ JSON (định dạng `0.015 [0.010, 0.020]`); file thiếu thì ghi "chưa có dữ liệu" thay vì crash; 3 hình được tạo; chọn ví dụ đúng 3 nhóm, cùng seed thì cùng kết quả.

## 6. Benchmark / đánh giá

| Hạng mục kiểm tra tái lập | Kết quả |
|---|---|
| SHA-256 manifest khớp protocol | [điền sau khi chạy] |
| Smoke test checkpoint fine-tune (SV A chạy) | [điền sau khi chạy] |
| Dice/IoU tính lại khớp `seg_metrics.json` | [điền sau khi chạy] |
| MAE tính lại khớp `forecast_metrics.json` | [điền sau khi chạy] |
| `table_*.md` của SV A và SV B giống hệt | [điền sau khi chạy] |
| Thời gian SV A dựng lại môi trường | [điền sau khi chạy] |

## 7. Lỗi thường gặp (máy local)

| Triệu chứng | Cách xử lý |
|---|---|
| Bảng ghi "chưa có dữ liệu" | Sai đường dẫn JSON; kiểm tra `run_id` |
| Số trong bảng khác lần trước | So run card hai lần: phiên bản gói, seed, SHA-256 đầu vào |
| `ModuleNotFoundError: matplotlib` | `%pip install -e "{REPO}[demo]"` trong `nb_cpu` (extras `demo` có matplotlib) |
| Không thấy run của máy GPU ở laptop | Chép về theo `00` mục 5.2 |
| Chữ tiếng Việt trong hình lỗi font | Font mặc định DejaVu Sans của matplotlib hỗ trợ tiếng Việt; nếu đổi font, chọn font có dấu |

## 8. Checklist bàn giao cho SV A (trước khi nộp)

- [ ] Mọi bảng/hình trong bài có dòng tương ứng trong bảng truy vết mục 3.2, có `run_id`.
- [ ] `runs/<p10>/run_card.json` liệt kê SHA-256 mọi file đầu vào.
- [ ] `examples_seg.csv`, `examples_forecast.csv` được chọn bằng `select_examples` (seed 2026), không chọn tay.
- [ ] Kết quả bất lợi (Ridge không hơn baseline, robustness kém, CI rộng) vẫn nằm trong bảng.
- [ ] SV A đã làm xong 5 bước tái lập ở mục 4.5 và điền bảng mục 6.
- [ ] `pytest` toàn dự án pass; log lưu ở `runs/<pytest>/pytest.log`.
- [ ] Phụ lục kỹ thuật P11 (file `P0_P1_P4_P11_vai_tro_ho_tro.md`) dùng đúng số liệu từ run card.
