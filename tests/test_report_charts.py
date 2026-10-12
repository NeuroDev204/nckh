import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from nckh.report_charts import evaluation_figures, pilot_figures, pilot_tables  # noqa: E402


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


def test_pilot_dashboard_renders_saved_figures(pilot_summary: dict, tmp_path: Path) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(pilot_summary))
    script = Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py"
    app = AppTest.from_file(str(script)).run()
    app.text_input(key="pilot_summary").set_value(str(path)).run()
    assert not app.exception
    assert len(app.get("image")) == 5
    assert any("P2 đạt" in message.value for message in app.success)
    assert any("tổng hợp" in message.value for message in app.warning)
    assert {tab.label for tab in app.tabs} == {"Báo cáo P2", "Đánh giá mô hình"}
    app.text_input(key="pilot_summary").set_value(str(tmp_path / "missing.json")).run()
    assert not app.exception
    assert app.info
    assert not app.get("image")


def test_pilot_dashboard_rejects_bad_json(tmp_path: Path) -> None:
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    path = tmp_path / "summary.json"
    path.write_text("not json")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "scripts/streamlit_smoke.py")).run()
    app.text_input(key="pilot_summary").set_value(str(path)).run()
    assert not app.exception
    assert app.error
    assert not app.get("image")


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
