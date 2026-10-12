"""Streamlit local: smoke test và metric PanDerm trong hai venv riêng."""
import json
import logging
import os
import signal
import subprocess
import traceback
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from nckh.paths import checkpoints_dir, local_data_root, manifests_dir, runs_dir
from nckh.runcard import new_run_id

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def run_worker(command: list[str], cwd: Path, log_path: Path, timeout_seconds: int) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        try:
            with subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    return process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    # Dừng cả process group để worker timeout không để lại tiến trình con.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    log.write(f"\nTimeout sau {timeout_seconds}s; worker đã dừng\n")
                    return 124
        except OSError:
            log.write(traceback.format_exc())
            logger.exception("Không chạy được worker")
            return 127


def load_result(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or not {"task", "n", "checks", "metrics", "warnings"} <= report.keys()
            or report["task"] not in ("cls", "seg") or not isinstance(report["n"], int) or report["n"] < 1
            or not isinstance(report["checks"], dict) or not isinstance(report["warnings"], list)
            or (report["metrics"] is not None and not isinstance(report["metrics"], dict))):
        raise ValueError("Báo cáo worker thiếu trường hoặc sai schema")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    if not {"image_id", "image_path"} <= set(predictions) or len(predictions) != report["n"]:
        raise ValueError("CSV prediction thiếu trường hoặc số dòng không khớp báo cáo")
    return report, predictions


def run_tasks(configs: dict[str, dict[str, str]], common: dict[str, object], run_dir: Path,
              timeout_seconds: int) -> dict[str, dict]:
    run_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for task, config in configs.items():
        output = run_dir / task
        log_path = run_dir / f"{task}_subprocess.log"
        result = {"report_dir": str(output), "log": str(log_path), "error": None}
        try:
            python = Path(config["python"]).expanduser().absolute()
            upstream = Path(config["panderm_dir"]).expanduser().resolve()
            manifest = Path(config["manifest"]).expanduser().resolve()
            data_root = Path(str(common["data_root"])).expanduser().resolve()
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError(f"Python executable không tồn tại/không chạy được: {python}")
            if not upstream.is_dir() or not data_root.is_dir() or not manifest.is_file():
                raise ValueError("Thư mục PanDerm, data root hoặc manifest không tồn tại")
            command = [str(python), str(ROOT / "scripts/evaluate_inference.py"), "--task", task,
                       "--panderm-dir", str(upstream), "--manifest", str(manifest),
                       "--data-root", str(data_root), "--out-dir", str(output)]
            for key in ("split", "n", "seed"):
                command.extend([f"--{key}", str(common[key])])
            for key in ("pretrained", "finetuned"):
                if config[key].strip():
                    checkpoint = Path(config[key]).expanduser().resolve()
                    if not checkpoint.is_file():
                        raise ValueError(f"Checkpoint không tồn tại: {checkpoint}")
                    command.extend([f"--{key}", str(checkpoint)])
            code = run_worker(command, ROOT, log_path, timeout_seconds)
            if code:
                raise ValueError(f"Worker {task} thất bại (exit code {code}); xem log")
            report, _ = load_result(output)
            if report["task"] != task:
                raise ValueError("Task trong báo cáo không khớp worker")
        except (ValueError, OSError) as exc:
            logger.exception("Nhánh %s thất bại", task)
            result["error"] = str(exc)
            if not log_path.exists():
                with log_path.open("w", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
        results[task] = result
    return results


def _render_result(task: str, result: dict) -> None:
    st.subheader("Phân loại" if task == "cls" else "Phân đoạn")
    if result["error"]:
        st.error(result["error"])
    else:
        try:
            run_dir = Path(result["report_dir"])
            report, predictions = load_result(run_dir)
            st.success(f"Smoke test OK · {report['n']} ảnh · split {report.get('split', '—')}")
            for warning in report["warnings"]:
                st.warning(warning)
            metrics = report["metrics"]
            if metrics:
                scalar = {key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}
                st.dataframe(pd.DataFrame(scalar.items(), columns=["Chỉ số", "Giá trị"]), hide_index=True)
                if "per_class" in metrics:
                    st.dataframe(pd.DataFrame(metrics["per_class"]), hide_index=True)
                    st.caption("Confusion matrix: dòng = nhãn thật, cột = dự đoán; melanoma / nevus / seborrheic keratosis")
                    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["mel", "nev", "sk"], columns=["mel", "nev", "sk"]))
            benchmark = {key: report[key] for key in ("device", "ms_per_image_mean", "ms_per_image_p50", "ms_per_image_p95", "peak_vram_mb") if key in report}
            st.json(benchmark)
            st.caption(report.get("timing_scope", ""))
            st.dataframe(predictions, hide_index=True)
            for name in ("report.json", "predictions.csv"):
                st.download_button(f"Tải {name}", (run_dir / name).read_bytes(), file_name=f"{task}_{name}",
                                   mime="application/json" if name.endswith("json") else "text/csv", key=f"download_{task}_{name}")
            with st.expander("Ảnh mẫu và đầu ra"):
                for row in predictions.head(3).to_dict("records"):
                    image = Path(row["image_path"])
                    if image.is_file():
                        st.image(str(image), caption=row["image_id"], width=300)
                    overlay = run_dir / "overlays" / f"{row['image_id']}.png"
                    if task == "seg" and overlay.is_file():
                        st.image(str(overlay), caption="Mask overlay", width=300)
                    elif task == "cls":
                        st.write({name: row[name] for name in ("p_mel", "p_nev", "p_sk") if name in row})
            st.caption(f"Kết quả: {run_dir}")
        except (ValueError, OSError) as exc:
            logger.exception("Không đọc được kết quả nhánh %s", task)
            st.error(f"Không đọc được artifacts: {exc}")
    log_path = Path(result["log"])
    if log_path.is_file():
        with st.expander("Log worker"):
            with log_path.open(encoding="utf-8", errors="replace") as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 16000))
                st.code(log.read())
            st.caption(f"Log đầy đủ: {log_path}")


def main() -> None:
    st.set_page_config(page_title="PanDerm smoke test", layout="wide")
    st.title("PanDerm · Smoke test và metric")
    st.caption("Demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu, không phải chẩn đoán hay khuyến nghị điều trị.")
    home = Path.home()
    with st.form("configuration"):
        branch = st.selectbox("Nhánh", ["Cả hai", "Phân loại", "Phân đoạn"])
        mode = st.selectbox("Chế độ", ["Smoke test", "Đánh giá toàn split"])
        split = st.selectbox("Split", ["val", "test", "train"], key="split")
        n = st.number_input("Số ảnh mỗi nhánh (smoke test)", min_value=1, value=20, key="n")
        seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=2026)
        data_root = st.text_input("Data root", str(local_data_root()))
        output_root = st.text_input("Thư mục lưu runs", str(runs_dir()))
        timeout = st.number_input("Timeout mỗi nhánh (giây)", min_value=1, value=600)
        configs = {}
        for task, label, upstream, manifest in (
            ("cls", "Phân loại", "classification", "isic2017_cls.csv"),
            ("seg", "Phân đoạn", "segmentation", "isic2018_seg.csv"),
        ):
            default_manifest = manifests_dir() / manifest
            smoke_manifest = manifests_dir() / "smoke_images.csv"
            if not default_manifest.is_file() and smoke_manifest.is_file():
                default_manifest = smoke_manifest
            with st.expander(f"Cấu hình {label}"):
                configs[task] = {
                    "python": st.text_input("Python venv", str(home / "venvs" / f"venv_{task}" / "bin/python"), key=f"{task}_python"),
                    "panderm_dir": st.text_input("Thư mục PanDerm", str(home / "PanDerm" / upstream), key=f"{task}_upstream"),
                    "manifest": st.text_input("Manifest CSV", str(default_manifest), key=f"{task}_manifest"),
                    "pretrained": st.text_input("Checkpoint pretrain", str(checkpoints_dir() / "panderm_bb_data6_checkpoint-499.pth"), key=f"{task}_pretrained"),
                    "finetuned": st.text_input("Checkpoint fine-tune (nếu có)", "", key=f"{task}_finetuned"),
                }
        submitted = st.form_submit_button("Chạy", disabled=st.session_state.get("running", False))
    if submitted:
        tasks = ["cls", "seg"] if branch == "Cả hai" else ["cls" if branch == "Phân loại" else "seg"]
        common = {"data_root": data_root, "split": split, "n": 0 if mode == "Đánh giá toàn split" else int(n), "seed": int(seed)}
        run_dir = Path(output_root).expanduser().resolve() / f"{new_run_id('streamlit')}_{uuid.uuid4().hex[:8]}"
        st.session_state["running"] = True
        try:
            with st.spinner("Đang chạy worker theo từng nhánh…"):
                st.session_state["results"] = run_tasks({task: configs[task] for task in tasks}, common, run_dir, int(timeout))
        except OSError as exc:
            logger.exception("Không tạo được thư mục run")
            st.error(f"Không tạo được thư mục run: {exc}")
        finally:
            st.session_state["running"] = False
    for task, result in st.session_state.get("results", {}).items():
        _render_result(task, result)


if __name__ == "__main__":
    main()
