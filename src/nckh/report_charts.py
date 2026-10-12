"""Biểu đồ từ báo cáo đã lưu; không suy luận thêm hoặc thay thế số liệu thiếu."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from nckh.manifest import find_cross_split_duplicates
from nckh.runcard import sha256_file

BRANCH_NAMES = {"cls": "Phân loại", "seg": "Phân đoạn"}
PILOT_KEYS = {
    "cls": ("patch_embed.proj.weight", "head.weight"),
    "seg": ("model.segmentor.backbone.patch_embed.proj.weight", "model.segmentor.decode_head.conv_seg.weight"),
}
CHART_COLOR = "#181818"
P3_SPLITS = ("train", "val", "test")
P3_CLASSES = {0: "Melanoma", 1: "Nevus", 2: "Seborrheic keratosis"}


def _read_p3_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu artifact P3: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu cột: {', '.join(sorted(missing))}")
    return frame


def p3_tables(manifest_dir: Path) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    seg = _read_p3_csv(manifest_dir / "isic2018_seg.csv", {
        "image_id", "image_path", "mask_path", "split", "sha256", "width", "height",
        "mask_width", "mask_height", "readable", "exclude_reason",
    })
    cls = _read_p3_csv(manifest_dir / "isic2017_cls.csv", {"image", "label", "split"})
    trainphase = _read_p3_csv(manifest_dir / "isic2017_cls_trainphase.csv", {"image", "label", "split"})
    duplicates = _read_p3_csv(manifest_dir / "isic2018_seg.cross_split_duplicates.csv", {"sha256", "split"})

    seg_counts = seg["split"].value_counts().reindex(P3_SPLITS, fill_value=0).astype(int)
    mask_counts = seg.groupby("split")["mask_path"].count().reindex(P3_SPLITS, fill_value=0).astype(int)
    seg_overview = pd.DataFrame({"split": P3_SPLITS, "images": seg_counts.to_numpy(), "masks": mask_counts.to_numpy()})
    valid_cls = cls[cls["split"].isin(P3_SPLITS) & cls["label"].isin(P3_CLASSES)]
    distribution = pd.crosstab(valid_cls["label"], valid_cls["split"]).reindex(
        index=P3_CLASSES, columns=P3_SPLITS, fill_value=0
    )
    distribution.index = [P3_CLASSES[index] for index in distribution.index]

    checksum_path = manifest_dir / "isic2018_seg.csv.sha256"
    checksum_parts = checksum_path.read_text(encoding="utf-8").split()
    if len(checksum_parts) < 2:
        raise ValueError(f"Sai định dạng checksum: {checksum_path}")
    actual_sha256 = sha256_file(manifest_dir / "isic2018_seg.csv")
    checksum_matches = checksum_parts[0] == actual_sha256 and checksum_parts[1] == "isic2018_seg.csv"

    mask_count_path = manifest_dir / "mask_count.json"
    mask_report = json.loads(mask_count_path.read_text(encoding="utf-8"))
    reported_counts = mask_report["counts"]
    mask_report_matches = all(
        int(reported_counts[split]["images"]) == int(seg_counts[split])
        and int(reported_counts[split]["masks"]) == int(mask_counts[split])
        for split in P3_SPLITS
    )
    max_masks_per_image = max(int(reported_counts[split]["max_masks_per_image"]) for split in P3_SPLITS)

    actual_duplicates = find_cross_split_duplicates(seg)
    listed_duplicates = set(zip(duplicates["sha256"], duplicates["split"]))
    actual_duplicate_rows = set(zip(actual_duplicates["sha256"], actual_duplicates["split"]))
    duplicate_report_matches = len(duplicates) == len(actual_duplicates) and listed_duplicates == actual_duplicate_rows
    readable = seg["readable"].isin([True, 1, "True", "true"])
    size_mismatch = (seg["width"] != seg["mask_width"]) | (seg["height"] != seg["mask_height"])
    excluded = seg["exclude_reason"].fillna("").astype(str).str.strip().ne("")
    invalid_labels = ~cls["label"].isin(P3_CLASSES)
    duplicate_cls_images = int(cls["image"].duplicated().sum())
    val_rows = cls.loc[cls["split"] == "val", ["image", "label"]].sort_values("image").reset_index(drop=True)
    trainphase_test = trainphase.loc[trainphase["split"] == "test", ["image", "label"]].sort_values("image").reset_index(drop=True)
    official_train_val = cls.loc[cls["split"].isin(("train", "val")), ["image", "label", "split"]]
    trainphase_train_val = trainphase.loc[trainphase["split"].isin(("train", "val")), ["image", "label", "split"]]
    trainphase_test_is_val = val_rows.equals(trainphase_test) and official_train_val.sort_values(
        ["split", "image"]
    ).reset_index(drop=True).equals(trainphase_train_val.sort_values(["split", "image"]).reset_index(drop=True))
    known_splits = set(seg["split"].dropna()) == set(P3_SPLITS) and set(cls["split"].dropna()) == set(P3_SPLITS)

    checks = [
        ("SHA-256 manifest ISIC 2018", checksum_matches, actual_sha256),
        ("Đủ split chính thức", known_splits, "ISIC 2018 và ISIC 2017 có train/val/test"),
        ("Ảnh ISIC 2018 đọc được", bool(len(seg) and readable.all()), f"{int(readable.sum())}/{len(seg)} ảnh"),
        ("Kích thước ảnh và mask khớp", not bool(size_mismatch.any()), f"{int(size_mismatch.sum())} cặp lệch kích thước"),
        ("Không có ảnh bị loại", not bool(excluded.any()), f"{int(excluded.sum())} ảnh bị loại"),
        ("Mask count khớp manifest", mask_report_matches and max_masks_per_image == 1,
         f"tối đa {max_masks_per_image} mask/ảnh"),
        ("Bảng trùng khớp manifest", duplicate_report_matches, f"{len(duplicates)} dòng artifact · {len(actual_duplicates)} dòng tính lại"),
        ("Không trùng hash xuyên split", actual_duplicates.empty, f"{len(actual_duplicates)} dòng trùng chính xác"),
        ("Nhãn ISIC 2017 hợp lệ và ID duy nhất", not bool(invalid_labels.any()) and duplicate_cls_images == 0,
         f"{int(invalid_labels.sum())} nhãn lỗi · {duplicate_cls_images} ID trùng"),
        ("Trainphase dùng validation thay test", trainphase_test_is_val,
         f"{len(trainphase_test)} dòng test mô phỏng · test thật giữ riêng {int((cls['split'] == 'test').sum())} ảnh"),
    ]
    quality = pd.DataFrame(checks, columns=["Kiểm tra", "Đạt", "Chi tiết"])
    summary: dict[str, object] = {
        "seg_images": len(seg),
        "seg_masks": int(mask_counts.sum()),
        "cls_images": len(cls),
        "seg_excluded": int(excluded.sum()),
        "seg_unreadable": int((~readable).sum()),
        "seg_size_mismatch": int(size_mismatch.sum()),
        "seg_duplicates": len(actual_duplicates),
        "seg_sha256": actual_sha256,
        "checksum_matches": checksum_matches,
        "max_masks_per_image": max_masks_per_image,
        "trainphase_test_is_val": trainphase_test_is_val,
        "quality_ok": bool(quality["Đạt"].all()),
    }
    return summary, {"seg_split": seg_overview, "class_distribution": distribution, "quality": quality}


def p3_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Figure]:
    seg = tables["seg_split"]
    figure = Figure(figsize=(6.5, 3.2), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(seg))
    for offset, column, label, alpha in ((-.18, "images", "Ảnh", 1.), (.18, "masks", "Mask", .5)):
        bars = axis.bar(positions + offset, seg[column], width=.36, color=CHART_COLOR, alpha=alpha, label=label)
        axis.bar_label(bars, padding=3)
    axis.set_xticks(positions, seg["split"])
    axis.set(title="ISIC 2018 · ảnh và mask theo split", ylabel="Số lượng")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    distribution = tables["class_distribution"]
    figure_cls = Figure(figsize=(6.5, 3.2), layout="constrained")
    axis_cls = figure_cls.subplots()
    positions = np.arange(len(distribution))
    width = .24
    for index, split in enumerate(P3_SPLITS):
        bars = axis_cls.bar(positions + (index - 1) * width, distribution[split], width=width,
                            label=split, color=CHART_COLOR, alpha=1. - index * .2)
        axis_cls.bar_label(bars, padding=2, fontsize=8)
    axis_cls.set_xticks(positions, distribution.index)
    axis_cls.set(title="ISIC 2017 · phân bố lớp theo split", ylabel="Số ảnh")
    axis_cls.spines[["top", "right"]].set_visible(False)
    axis_cls.legend(frameon=False)
    return {"seg_split": figure, "class_distribution": figure_cls}


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Số liệu phải hữu hạn và không âm")
    return float(value)


def pilot_tables(summary: dict) -> dict[str, pd.DataFrame]:
    status, losses, memory, updates = {}, [], [], []
    try:
        for task, name in BRANCH_NAMES.items():
            branch = summary["branches"][task]
            train, load, resume = (branch[stage] for stage in ("train", "load", "resume"))
            gradients = []
            changes = []
            for stage, report in (("Train", train), ("Resume", resume)):
                losses.append({"branch": name, "step": _number(report["optimizer_step"]),
                               "stage": stage, "loss": _number(report["loss"])})
                grad_values = [report["grad_norm"]] if task == "cls" else list(report["grad_norms"].values())
                gradients.append(bool(grad_values) and all(_number(value) > 0 for value in grad_values))
                for key, component in zip(PILOT_KEYS[task], ("Backbone", "Head")):
                    delta = _number(report["updates"][key])
                    changes.append(delta > 0)
                    updates.append({"branch": name, "stage": stage, "component": component, "max_abs_delta": delta})
            memory.append({"branch": name, "peak_vram_mb": max(_number(train["peak_vram_mb"]), _number(resume["peak_vram_mb"]))})
            step, restored, next_step = (_number(report[field]) for report, field in (
                (train, "optimizer_step"), (resume, "restored_optimizer_step"), (resume, "optimizer_step"),
            ))
            _number(load["max_abs_logits_error"])
            status[name] = {
                "Backward": train["status"] == resume["status"] == "PASS" and all(gradients),
                "Cập nhật backbone/head": all(changes),
                "Lưu & nạp checkpoint": bool(train["checkpoint"]) and bool(resume["checkpoint"])
                                       and load["status"] == "PASS",
                "Resume optimizer": resume["status"] == "PASS" and step == restored == 1 and next_step == 2,
            }
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Báo cáo pilot thiếu nhánh, giai đoạn hoặc trường số liệu bắt buộc") from exc
    return {"status": pd.DataFrame(status), "loss": pd.DataFrame(losses),
            "vram": pd.DataFrame(memory), "updates": pd.DataFrame(updates)}


def _bars(labels: list[str], values: list[float], title: str, unit: str,
          upper: float | None = None) -> Figure:
    figure = Figure(figsize=(7, 3.4), layout="constrained")
    axis = figure.subplots()
    bars = axis.bar(labels, values, color=CHART_COLOR, width=.55)
    axis.bar_label(bars, labels=[f"{value:.3f}" if upper else f"{value:.1f}" for value in values], padding=5)
    axis.set(title=title, ylabel=unit, ylim=(0, upper or max(max(values, default=0) * 1.2, 1)))
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def pilot_figures(summary: dict) -> dict[str, Figure]:
    tables = pilot_tables(summary)
    figures = {}
    figure = Figure(figsize=(7, 3), layout="constrained")
    axis = figure.subplots()
    states = tables["status"]
    axis.imshow(states.to_numpy(dtype=int), cmap=ListedColormap(["#c70036", "#007a55"]), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(2), states.columns)
    axis.set_yticks(range(len(states)), states.index)
    axis.tick_params(length=0)
    for row in range(len(states)):
        for col in range(2):
            axis.text(col, row, "PASS" if states.iloc[row, col] else "FAIL", ha="center", va="center", color="white", weight="bold")
    axis.set_title("Kiểm tra kỹ thuật P2")
    figures["status"] = figure
    for task, name in BRANCH_NAMES.items():
        frame = tables["loss"][tables["loss"]["branch"] == name]
        figure = Figure(figsize=(6, 3.4), layout="constrained")
        axis = figure.subplots()
        axis.plot(frame["step"], frame["loss"], marker="o", color=CHART_COLOR)
        axis.set_xticks(frame["step"], [f"{stage} · bước {step:g}" for stage, step in zip(frame["stage"], frame["step"])])
        axis.set(title=f"Loss pilot · {name}", ylabel="Cross-entropy loss", ylim=(0, max(frame["loss"].max() * 1.2, .1)))
        axis.spines[["top", "right"]].set_visible(False)
        for step, loss in zip(frame["step"], frame["loss"]):
            axis.annotate(f"{loss:.4f}", (step, loss), xytext=(0, 10), textcoords="offset points", ha="center")
        figures[f"loss_{task}"] = figure
    figures["vram"] = _bars(tables["vram"]["branch"].tolist(), tables["vram"]["peak_vram_mb"].tolist(),
                            "VRAM cao nhất trong pilot", "MiB")
    figure = Figure(figsize=(10, 3.6), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for axis, name in zip(axes, BRANCH_NAMES.values()):
        frame = tables["updates"][tables["updates"]["branch"] == name]
        for offset, stage, alpha in ((-.18, "Train", 1.), (.18, "Resume", .45)):
            values = frame[frame["stage"] == stage]["max_abs_delta"].tolist()
            bars = axis.bar(np.arange(2) + offset, values, width=.36, color=CHART_COLOR, alpha=alpha, label=stage)
            axis.bar_label(bars, labels=[f"{value:.2e}" for value in values], padding=4, fontsize=8)
        axis.set_xticks([0, 1], ["Backbone", "Head"])
        axis.set(title=name, ylim=(0, max(tables["updates"]["max_abs_delta"].max() * 1.3, 1e-8)))
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Max |Δ trọng số|")
    figure.suptitle("Cập nhật trọng số sau mỗi bước")
    figures["updates"] = figure
    return figures


def evaluation_figures(report: dict) -> dict[str, Figure]:
    figures = {}
    metrics = report.get("metrics") or {}
    names = (["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f2", "macro_auroc"]
             if report["task"] == "cls" else ["dice", "iou", "precision", "recall", "f2", "pixel_accuracy"])
    scores = {name: _number(metrics[name]) for name in names if metrics.get(name) is not None}
    if any(value > 1 for value in scores.values()):
        raise ValueError("Chỉ số accuracy/F-score/AUROC/Dice/IoU phải trong [0,1]")
    if scores:
        figures["scores"] = _bars([name.replace("_", "\n") for name in scores],
                                  list(scores.values()), "Chỉ số đối chiếu ground truth", "Giá trị (0–1)", upper=1.15)
    if metrics.get("per_class"):
        classes = metrics["per_class"]
        figure = Figure(figsize=(7, 3.4), layout="constrained")
        axis = figure.subplots()
        for offset, field, alpha in ((-.18, "recall", 1.), (.18, "f2", .45)):
            values = [_number(row[field]) for row in classes]
            if any(value > 1 for value in values):
                raise ValueError("Recall và F2 từng lớp phải trong [0,1]")
            bars = axis.bar(np.arange(len(classes)) + offset, values, width=.36, label=field, color=CHART_COLOR, alpha=alpha)
            axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_xticks(range(len(classes)), [row["name"].replace("_", "\n") for row in classes])
        axis.set(title="Recall và F2 từng lớp", ylim=(0, 1.15), ylabel="Giá trị (0–1)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        figures["per_class"] = figure
    if "confusion_matrix" in metrics:
        counts = np.asarray(metrics["confusion_matrix"])
        if counts.shape != (3, 3) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Confusion matrix phải là số đếm không âm 3×3")
        figure = Figure(figsize=(5, 3.8), layout="constrained")
        axis = figure.subplots()
        axis.imshow(counts, cmap="Greys", vmin=0)
        axis.set_xticks(range(3), ["Melanoma", "Nevus", "SK"])
        axis.set_yticks(range(3), ["Melanoma", "Nevus", "SK"])
        axis.set(title="Confusion matrix", xlabel="Nhãn dự đoán", ylabel="Nhãn thật")
        for row in range(3):
            for col in range(3):
                axis.text(col, row, f"{counts[row, col]:g}", ha="center", va="center",
                          color="white" if counts[row, col] > counts.max() / 2 else CHART_COLOR)
        figures["confusion"] = figure
    times = {label: _number(report[key]) for label, key in (("Mean", "ms_per_image_mean"), ("P50", "ms_per_image_p50"),
                                                          ("P95", "ms_per_image_p95")) if report.get(key) is not None}
    if times:
        figures["latency"] = _bars(list(times), list(times.values()), "Thời gian suy luận", "ms/ảnh")
    return figures
