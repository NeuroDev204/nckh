# file: tests/test_paths_runcard.py
import json
from datetime import datetime
from pathlib import Path

from nckh.paths import project_root, runs_dir
from nckh.runcard import new_run_id, sha256_file, write_run_card


def test_project_root_reads_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NCKH_ROOT", str(tmp_path))
    assert project_root() == tmp_path
    assert runs_dir() == tmp_path / "runs"


def test_sha256_known(tmp_path: Path) -> None:
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_new_run_id_format() -> None:
    assert new_run_id("seg", datetime(2026, 10, 5, 9, 7, 3)) == "20261005-090703_seg"


def test_write_run_card_keys(tmp_path: Path) -> None:
    x = tmp_path / "x.csv"
    x.write_text("a,b\n1,2\n")
    card_path = write_run_card(tmp_path / "20261005-090703_seg", seed=0, config={"lr": 1e-4}, inputs={"x": x})
    card = json.loads(card_path.read_text())
    assert set(card) == {"run_id", "created_at", "git_commit", "python", "packages", "gpu", "seed", "config", "inputs"}
    assert card["run_id"] == "20261005-090703_seg"
    assert card["inputs"]["x"]["sha256"] == sha256_file(x)


def test_runcard_cli(tmp_path: Path) -> None:
    from nckh.runcard import main

    x = tmp_path / "ckpt.pth"
    x.write_bytes(b"weights")
    main([str(tmp_path / "run1"), "--seed", "3", "--input", f"pretrained={x}", "--config", "task=smoke_seg", "--config", "n=20"])
    card = json.loads((tmp_path / "run1" / "run_card.json").read_text())
    assert card["seed"] == 3
    assert card["config"] == {"task": "smoke_seg", "n": "20"}
    assert card["inputs"]["pretrained"]["sha256"] == sha256_file(x)