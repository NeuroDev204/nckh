import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
from nckh.report_charts import evaluation_figures, p3_figures, p3_tables, pilot_figures, pilot_tables  # noqa: E402


def write_p3_artifacts(manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    seg_rows = []
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for index in range(count):
            image_id = f"{split}_{index}"
            seg_rows.append({"image_id": image_id, "image_path": f"{image_id}.jpg", "mask_path": f"{image_id}.png",
                             "split": split, "sha256": f"hash-{image_id}", "width": 10, "height": 10,
                             "mask_width": 10, "mask_height": 10, "readable": True, "exclude_reason": ""})
    seg_path = manifest_dir / "isic2018_seg.csv"
    pd.DataFrame(seg_rows).to_csv(seg_path, index=False)
    (manifest_dir / "isic2018_seg.csv.sha256").write_text(
        f"{hashlib.sha256(seg_path.read_bytes()).hexdigest()}  isic2018_seg.csv\n"
    )
    pd.DataFrame(columns=["sha256", "split"]).to_csv(manifest_dir / "isic2018_seg.cross_split_duplicates.csv", index=False)
    counts = {split: {"images": count, "masks": count, "max_masks_per_image": 1}
              for split, count in (("train", 2), ("val", 1), ("test", 1))}
    (manifest_dir / "mask_count.json").write_text(json.dumps({"counts": counts}))
    cls = pd.DataFrame([
        {"image": "train_0.jpg", "label": 0, "split": "train"},
        {"image": "train_1.jpg", "label": 1, "split": "train"},
        {"image": "val_0.jpg", "label": 2, "split": "val"},
        {"image": "test_0.jpg", "label": 1, "split": "test"},
    ])
    cls.to_csv(manifest_dir / "isic2017_cls.csv", index=False)
    pd.concat([cls[cls["split"] != "test"], cls[cls["split"] == "val"].assign(split="test")]).to_csv(
        manifest_dir / "isic2017_cls_trainphase.csv", index=False
    )


@pytest.fixture
def pilot_summary() -> dict:
    branches = {}
    for task, keys, losses, vram in (
        ("cls", ["patch_embed.proj.weight", "head.weight"], [1.1, 1.0], 1600),
        ("seg", ["model.segmentor.backbone.patch_embed.proj.weight", "model.segmentor.decode_head.conv_seg.weight"], [.6, .4], 3000),
    ):
        stages = {}
        for i, stage in enumerate(("train", "resume")):
            stages[stage] = {"status": "PASS", "loss": losses[i], "peak_vram_mb": vram + i,
                             "updates": {key: .0001 for key in keys}, "optimizer_step": i + 1,
                             "checkpoint": "pilot.ckpt", "images": ["a.jpg", "b.jpg"], "seed": 2026}
            if task == "cls":
                stages[stage]["grad_norm"] = 2.0
            else:
                stages[stage]["grad_norms"] = {key: 2.0 for key in keys}
        stages["resume"]["restored_optimizer_step"] = 1
        stages["load"] = {"status": "PASS", "max_abs_logits_error": 0.0}
        branches[task] = stages
    return {"synthetic_targets": True, "branches": branches}


def test_pilot_tables_preserve_real_values(pilot_summary: dict) -> None:
    tables = pilot_tables(pilot_summary)
    assert tables["status"].shape == (4, 2)
    assert tables["status"].to_numpy().all()
    assert tables["loss"]["loss"].tolist() == [1.1, 1., .6, .4]
    assert tables["loss"]["step"].tolist() == [1, 2, 1, 2]
    assert tables["vram"]["peak_vram_mb"].tolist() == [1601, 3001]
    assert len(tables["updates"]) == 8


def test_failed_resume_never_becomes_pass(pilot_summary: dict) -> None:
    pilot_summary["branches"]["cls"]["resume"]["optimizer_step"] = 1
    tables = pilot_tables(pilot_summary)
    assert not tables["status"].loc["Resume optimizer", "Phân loại"]


def test_checkpoint_status_uses_native_tolerance_result(pilot_summary: dict) -> None:
    pilot_summary["branches"]["cls"]["load"]["max_abs_logits_error"] = .00005
    assert pilot_tables(pilot_summary)["status"].loc["Lưu & nạp checkpoint", "Phân loại"]


@pytest.mark.parametrize("invalid", ["missing_branch", "nan_loss", "missing_stage", "missing_updates"])
def test_rejects_incomplete_pilot_reports(pilot_summary: dict, invalid: str) -> None:
    if invalid == "missing_branch":
        del pilot_summary["branches"]["seg"]
    elif invalid == "nan_loss":
        pilot_summary["branches"]["cls"]["train"]["loss"] = float("nan")
    elif invalid == "missing_stage":
        del pilot_summary["branches"]["seg"]["load"]
    else:
        del pilot_summary["branches"]["seg"]["train"]["updates"]
    with pytest.raises(ValueError):
        pilot_tables(pilot_summary)


def test_pilot_figures_have_two_observed_loss_points(pilot_summary: dict, tmp_path: Path) -> None:
    before = copy.deepcopy(pilot_summary)
    figures = pilot_figures(pilot_summary)
    assert len(figures) == 5
    assert figures["loss_cls"].axes[0].lines[0].get_ydata().tolist() == [1.1, 1.]
    assert figures["loss_seg"].axes[0].lines[0].get_xdata().tolist() == [1, 2]
    for name, figure in figures.items():
        figure.savefig(tmp_path / f"{name}.png")
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert pilot_summary == before


def test_evaluation_omits_null_metrics_and_draws_confusion_counts() -> None:
    report = {"task": "cls", "metrics": {"accuracy": .5, "macro_f2": .4, "macro_auroc": None,
                                         "confusion_matrix": [[1, 2, 0], [0, 2, 0], [1, 0, 1]]},
              "ms_per_image_mean": 10., "ms_per_image_p50": 9., "ms_per_image_p95": 15.}
    figures = evaluation_figures(report)
    heights = [patch.get_height() for patch in figures["scores"].axes[0].patches]
    assert heights == [.5, .4]
    np.testing.assert_array_equal(figures["confusion"].axes[0].images[0].get_array(), [[1, 2, 0], [0, 2, 0], [1, 0, 1]])
    assert [patch.get_height() for patch in figures["latency"].axes[0].patches] == [10., 9., 15.]


def test_smoke_only_does_not_invent_accuracy() -> None:
    figures = evaluation_figures({"task": "seg", "metrics": None, "ms_per_image_mean": 5.})
    assert "scores" not in figures
    assert "confusion" not in figures
    assert "latency" in figures


def test_p3_report_checks_manifests_and_builds_dataset_figures(tmp_path: Path) -> None:
    write_p3_artifacts(tmp_path)
    summary, tables = p3_tables(tmp_path)
    assert summary["quality_ok"]
    assert summary["seg_images"] == summary["seg_masks"] == 4
    assert tables["class_distribution"].loc["Melanoma"].to_dict() == {"train": 1, "val": 0, "test": 0}
    figures = p3_figures(tables)
    assert set(figures) == {"seg_split", "class_distribution"}
    assert [bar.get_height() for bar in figures["seg_split"].axes[0].patches[:3]] == [2, 1, 1]
    assert [bar.get_height() for bar in figures["class_distribution"].axes[0].patches[:3]] == [1, 1, 0]


def test_p3_report_flags_changed_manifest_checksum(tmp_path: Path) -> None:
    write_p3_artifacts(tmp_path)
    (tmp_path / "isic2018_seg.csv.sha256").write_text(f"{'0' * 64}  isic2018_seg.csv\n")
    summary, tables = p3_tables(tmp_path)
    assert not summary["quality_ok"]
    assert not tables["quality"].iloc[0]["Đạt"]


def test_p3_report_recomputes_cross_split_duplicates(tmp_path: Path) -> None:
    write_p3_artifacts(tmp_path)
    seg_path = tmp_path / "isic2018_seg.csv"
    seg = pd.read_csv(seg_path)
    seg.loc[1, "sha256"] = seg.loc[2, "sha256"]
    seg.to_csv(seg_path, index=False)
    (tmp_path / "isic2018_seg.csv.sha256").write_text(
        f"{hashlib.sha256(seg_path.read_bytes()).hexdigest()}  isic2018_seg.csv\n"
    )
    summary, tables = p3_tables(tmp_path)
    assert summary["seg_duplicates"] == 2
    assert not tables["quality"].set_index("Kiểm tra").loc["Không trùng hash xuyên split", "Đạt"]


def test_pilot_dashboard_renders_saved_figures(pilot_summary: dict, tmp_path: Path) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(pilot_summary))
    script = Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py"
    app = AppTest.from_file(str(script)).run()
    app.text_input(key="pilot_summary").set_value(str(path)).run()
    assert not app.exception
    assert len(app.get("image")) == 7
    assert any("P2 đạt" in message.value for message in app.success)
    assert any("tổng hợp" in message.value for message in app.warning)
    assert {tab.label for tab in app.tabs} == {"Báo cáo P2", "Báo cáo P3", "Đánh giá mô hình"}
    app.text_input(key="pilot_summary").set_value(str(tmp_path / "missing.json")).run()
    assert not app.exception
    assert app.info
    assert len(app.get("image")) == 2


def test_pilot_dashboard_rejects_bad_json(tmp_path: Path) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    path = tmp_path / "summary.json"
    path.write_text("not json")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py")).run()
    app.text_input(key="pilot_summary").set_value(str(path)).run()
    assert not app.exception
    assert app.error
    assert len(app.get("image")) == 2


def test_p3_dashboard_renders_charts_and_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    write_p3_artifacts(tmp_path / "data/manifests")
    monkeypatch.setenv("NCKH_ROOT", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py")).run()
    assert not app.exception
    assert len(app.get("image")) == 2
    assert any("Các kiểm tra dữ liệu P3 đều đạt" in message.value for message in app.success)
    assert any("Báo cáo P3" == tab.label for tab in app.tabs)


def test_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        evaluation_figures({"task": "cls", "metrics": {"accuracy": 2.}})


def test_reopens_saved_evaluation_without_running_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("NCKH_ROOT", str(tmp_path))
    run = tmp_path / "runs/saved_run"
    output = run / "cls"
    output.mkdir(parents=True)
    report = {"task": "cls", "n": 1, "checks": {}, "warnings": [], "metrics": {
        "accuracy": .5, "macro_f2": .4, "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    }, "ms_per_image_mean": 10., "ms_per_image_p50": 9., "ms_per_image_p95": 15.}
    (output / "report.json").write_text(json.dumps(report))
    (output / "predictions.csv").write_text("image_id,image_path\nISIC_0,/missing.jpg\n")
    script = Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py"
    app = AppTest.from_file(str(script)).run()
    app.selectbox(key="saved_run").select(str(run)).run()
    assert not app.exception
    assert len(app.get("image")) == 3
    assert any("Smoke test OK" in message.value for message in app.success)
    assert list((tmp_path / "runs").iterdir()) == [run]
