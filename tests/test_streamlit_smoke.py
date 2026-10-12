import json
import sys
from pathlib import Path

import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from streamlit_smoke import load_result, run_tasks, run_worker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_worker_logs_stdout_stderr_and_failure(tmp_path: Path) -> None:
    code = run_worker([sys.executable, "-c", "import sys; print('output'); print('error', file=sys.stderr); sys.exit(3)"],
                      ROOT, tmp_path / "worker.log", 5)
    assert code == 3
    assert "output" in (tmp_path / "worker.log").read_text()
    assert "error" in (tmp_path / "worker.log").read_text()


def test_worker_timeout_and_missing_executable(tmp_path: Path) -> None:
    assert run_worker([sys.executable, "-c", "import time; time.sleep(20)"], ROOT, tmp_path / "timeout.log", 1) == 124
    assert "Timeout" in (tmp_path / "timeout.log").read_text()
    assert run_worker([str(tmp_path / "missing")], ROOT, tmp_path / "missing.log", 1) == 127
    assert "FileNotFoundError" in (tmp_path / "missing.log").read_text()


def test_load_result_rejects_missing_or_malformed_artifacts(tmp_path: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        load_result(tmp_path)
    (tmp_path / "report.json").write_text('{"n": 1}')
    with pytest.raises(ValueError):
        load_result(tmp_path)
    (tmp_path / "report.json").write_text("not json")
    with pytest.raises(ValueError):
        load_result(tmp_path)


def test_both_tasks_keep_second_result_after_first_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def worker(command: list[str], cwd: Path, log_path: Path, timeout_seconds: int) -> int:
        task = command[command.index("--task") + 1]
        log_path.write_text("worker log")
        if task == "cls":
            return 1
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir()
        (out / "report.json").write_text(json.dumps({"task": task, "n": 1, "checks": {}, "metrics": None, "warnings": []}))
        pd.DataFrame([{"image_id": "ISIC_0", "image_path": "/tmp/image.jpg"}]).to_csv(out / "predictions.csv", index=False)
        return 0

    monkeypatch.setattr("streamlit_smoke.run_worker", worker)
    config = {"python": sys.executable, "panderm_dir": str(ROOT), "manifest": str(ROOT / "pyproject.toml"),
              "pretrained": str(ROOT / "pyproject.toml"), "finetuned": ""}
    common = {"data_root": str(ROOT), "split": "val", "n": 20, "seed": 2026}
    result = run_tasks({"cls": config, "seg": config}, common, tmp_path / "run", 5)
    assert result["cls"]["error"]
    assert result["seg"]["error"] is None
    assert load_result(Path(result["seg"]["report_dir"]))[0]["n"] == 1
    with pytest.raises(FileExistsError):
        run_tasks({"cls": config}, common, tmp_path / "run", 5)


def test_success_exit_without_artifacts_is_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("streamlit_smoke.run_worker", lambda *args: 0)
    config = {"python": sys.executable, "panderm_dir": str(ROOT), "manifest": str(ROOT / "pyproject.toml"),
              "pretrained": str(ROOT / "pyproject.toml"), "finetuned": ""}
    result = run_tasks({"cls": config}, {"data_root": str(ROOT), "split": "val", "n": 20, "seed": 2026}, tmp_path / "run", 5)
    assert result["cls"]["error"]


def test_app_startup_and_results_survive_rerun(tmp_path: Path) -> None:
    app = AppTest.from_file(str(ROOT / "scripts/streamlit_smoke.py")).run()
    assert not app.exception
    assert app.selectbox(key="split").value == "val"
    assert app.number_input(key="n").value == 20
    (tmp_path / "report.json").write_text(json.dumps({"task": "cls", "n": 1, "checks": {}, "metrics": {"accuracy": .5},
                                                     "warnings": ["Head chưa fine-tune"]}))
    pd.DataFrame([{"image_id": "ISIC_0", "image_path": "/missing.jpg", "p_mel": .2, "p_nev": .3, "p_sk": .5}]).to_csv(tmp_path / "predictions.csv", index=False)
    app.session_state["results"] = {"cls": {"report_dir": str(tmp_path), "error": None, "log": str(tmp_path / "worker.log")}}
    app.run()
    assert not app.exception
    assert app.warning
    assert len(app.dataframe) >= 2
    app.run()
    assert not app.exception
    assert app.session_state["results"]["cls"]["report_dir"] == str(tmp_path)
