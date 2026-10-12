"""Streamlit local: smoke test và metric PanDerm trong hai venv riêng."""
import json
import io
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
from nckh.report_charts import evaluation_figures, pilot_figures, pilot_tables
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _show_figure(figure: Figure, name: str, key: str) -> None:
    st.pyplot(figure, clear_figure=False)
    with io.BytesIO() as output:
        figure.savefig(output, format="png", dpi=160, bbox_inches="tight")
        st.download_button("Tải biểu đồ PNG", output.getvalue(), file_name=f"{name}.png",
                           mime="image/png", key=key)


def _render_pilot() -> None:
    candidates = sorted(runs_dir().glob("*p2*pilot*/summary.json"), reverse=True)
    default = candidates[0] if candidates else runs_dir() / "p2-training-pilot/summary.json"
    path = Path(st.text_input("Báo cáo pilot P2 (summary.json)", str(default), key="pilot_summary")).expanduser()
    if not path.is_file():
        st.info("Chọn file summary.json của pilot P2 để xem biểu đồ đã lưu.")
        return
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        tables = pilot_tables(summary)
        figures = pilot_figures(summary)
    except (ValueError, OSError) as exc:
        logger.exception("Không đọc được báo cáo pilot P2")
        st.error(f"Không đọc được báo cáo pilot P2: {exc}")
        return
    st.warning("Pilot kỹ thuật dùng nhãn/mask tổng hợp. Loss chỉ có hai điểm quan sát; không phải kết quả đánh giá hiệu năng.")
    count = int(tables["status"].to_numpy().sum())
    cards = st.columns(3)
    cards[0].metric("Kiểm tra đạt", f"{count}/8")
    cards[1].metric("Nhánh mô hình", "2")
    cards[2].metric("Điểm loss mỗi nhánh", str(len(tables["loss"]) // 2))
    _show_figure(figures["status"], "p2_status", "p2_download_status")
    loss_columns = st.columns(2)
    for task, column in zip(("cls", "seg"), loss_columns):
        with column:
            _show_figure(figures[f"loss_{task}"], f"p2_loss_{task}", f"p2_download_loss_{task}")
    _show_figure(figures["vram"], "p2_vram", "p2_download_vram")
    _show_figure(figures["updates"], "p2_weight_updates", "p2_download_updates")
    if tables["status"].to_numpy().all():
        st.success("Pilot kỹ thuật P2 đạt · Chuyển sang kiểm chứng loader và dữ liệu thật ở P3.")
    else:
        st.error("Pilot P2 chưa đạt · Xử lý các kiểm tra FAIL trước khi chuyển sang P3.")
    with st.expander("Số liệu chi tiết và nguồn báo cáo"):
        for name, table in tables.items():
            st.caption(name)
            st.dataframe(table)
        st.caption(str(path))
        st.download_button("Tải báo cáo P2 JSON", path.read_bytes(), file_name="p2_summary.json", mime="application/json")


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
            for name, figure in evaluation_figures(report).items():
                _show_figure(figure, f"{task}_{name}", f"download_{task}_{name}_png")
            if metrics and metrics.get("macro_auroc", "defined") is None:
                st.caption("AUROC chưa xác định do thiếu lớp trong mẫu; không biểu diễn giá trị thiếu thành 0.")
            benchmark = {key: report[key] for key in ("device", "ms_per_image_mean", "ms_per_image_p50", "ms_per_image_p95", "peak_vram_mb") if key in report}
            st.caption(report.get("timing_scope", ""))
            with st.expander("Bảng số liệu và prediction chi tiết"):
                if metrics:
                    scalar = {key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}
                    st.dataframe(pd.DataFrame(scalar.items(), columns=["Chỉ số", "Giá trị"]), hide_index=True)
                    if "per_class" in metrics:
                        st.dataframe(pd.DataFrame(metrics["per_class"]), hide_index=True)
                st.json(benchmark)
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


def _render_evaluation() -> None:
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
    saved = sorted({path.parent.parent if path.parent.name in ("cls", "seg") else path.parent
                    for pattern in ("*/report.json", "*/*/report.json") for path in runs_dir().glob(pattern)}, reverse=True)
    if submitted:
        st.session_state["saved_run"] = ""
    selected = st.selectbox("Xem báo cáo đã lưu", [""] + [str(path) for path in saved],
                            format_func=lambda value: Path(value).name if value else "Chọn báo cáo đã lưu", key="saved_run")
    display_results = st.session_state.get("results", {})
    if selected:
        root = Path(selected)
        directories = [path.parent for path in root.glob("*/report.json")] if not (root / "report.json").is_file() else [root]
        display_results = {}
        for directory in directories:
            try:
                report, _ = load_result(directory)
                task = report["task"]
                display_results[task] = {"report_dir": str(directory), "error": None,
                                         "log": str(root / f"{task}_subprocess.log" if (root / f"{task}_subprocess.log").is_file() else directory / "worker.log")}
            except (ValueError, OSError) as exc:
                logger.exception("Không đọc được báo cáo đã lưu")
                st.error(f"Không đọc được báo cáo đã lưu: {exc}")
    for task, result in display_results.items():
        _render_result(task, result)


def main() -> None:
    st.set_page_config(page_title="PanDerm · Báo cáo nghiên cứu", layout="wide")
    st.title("PanDerm · Báo cáo nghiên cứu")
    st.caption("Demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu, không phải chẩn đoán hay khuyến nghị điều trị.")
    pilot, evaluation = st.tabs(["Báo cáo P2", "Đánh giá mô hình"])
    with pilot:
        _render_pilot()
    with evaluation:
        _render_evaluation()


if __name__ == "__main__":
    main()
