"""Biểu đồ từ báo cáo đã lưu; không suy luận thêm hoặc thay thế số liệu thiếu."""
import math

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

BRANCH_NAMES = {"cls": "Phân loại", "seg": "Phân đoạn"}
PILOT_KEYS = {
    "cls": ("patch_embed.proj.weight", "head.weight"),
    "seg": ("model.segmentor.backbone.patch_embed.proj.weight", "model.segmentor.decode_head.conv_seg.weight"),
}
CHART_COLOR = "#181818"


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
