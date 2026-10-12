"""Worker smoke test và metric; chạy bằng Python riêng của venv_cls hoặc venv_seg."""
import argparse
import json
import logging
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bench_inference import run_benchmark
from nckh.isic import ISIC2017_IMAGE_DIRS
from nckh.metrics import classification_metrics, segmentation_metrics, validate_probabilities
from nckh.runcard import write_run_card

logger = logging.getLogger(__name__)
SEG_METRICS = ("dice", "iou", "precision", "recall", "f2", "pixel_accuracy")


def select_samples(manifest: pd.DataFrame, task: str, split: str, n: int, seed: int) -> pd.DataFrame:
    path_col = "image" if task == "cls" and "image" in manifest else "image_path"
    if task not in ("cls", "seg") or n < 0 or not {path_col, "split"} <= set(manifest.columns):
        raise ValueError("Task, số ảnh hoặc schema manifest không hợp lệ")
    df = manifest[manifest["split"] == split].copy()
    if df.empty:
        raise ValueError(f"Không có ảnh trong split {split}")
    if df[path_col].isna().any():
        raise ValueError("Manifest chứa đường dẫn ảnh thiếu")
    df["image_id"] = df[path_col].map(lambda value: Path(str(value)).stem)
    if df["image_id"].duplicated().any():
        raise ValueError("Manifest có image ID trùng trong split")
    excluded_counts = {}
    if task == "seg" and "exclude_reason" in df:
        reasons = df["exclude_reason"].fillna("").astype(str)
        excluded_counts = {key: int(count) for key, count in reasons[reasons != ""].value_counts().items()}
        df = df[reasons == ""].copy()
    if df.empty:
        raise ValueError("Không còn ảnh hợp lệ sau khi loại")
    if task == "cls":
        if "label" in df and not df["label"].isin([0, 1, 2]).all():
            raise ValueError("Nhãn classification phải trong {0,1,2}, không được thiếu")
        if split == "test" and any(ISIC2017_IMAGE_DIRS["val"] in Path(str(p)).parts for p in df[path_col]):
            raise ValueError("CSV trainphase chứa ảnh validation gán split test; cần manifest test thật")
    df = df.sort_values("image_id").reset_index(drop=True)
    if n and n < len(df):
        if task == "cls" and "label" in df and n >= df["label"].nunique():
            selected = [group.sample(n=1, random_state=seed).index[0] for _, group in df.groupby("label")]
            remaining = df.drop(index=selected).sample(n=n - len(selected), random_state=seed).index
            df = df.loc[selected + remaining.tolist()]
        else:
            df = df.sample(n=n, random_state=seed)
    df = df.sort_values("image_id").reset_index(drop=True)
    df.attrs.update(excluded_counts=excluded_counts, split=split, seed=seed)
    return df


def _resolve_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Đường dẫn ảnh/mask thiếu hoặc không phải chuỗi")
    relative = Path(value)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Đường dẫn phải tương đối và nằm trong data root: {value}")
    return resolved


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Mask phải grayscale nhị phân 0/1 hoặc 0/255: {path}")
    return mask > 0


def validate_samples(samples: pd.DataFrame, task: str, data_root: Path) -> list[Path]:
    path_col = "image" if task == "cls" and "image" in samples else "image_path"
    root = data_root / "ISIC2017" if path_col == "image" else data_root
    images = []
    for _, row in samples.iterrows():
        path = _resolve_path(root, row[path_col])
        if path in images:
            raise ValueError(f"Đường dẫn ảnh trùng sau khi resolve: {path}")
        with Image.open(path) as image:
            image.load()
            shape = (image.height, image.width)
        if task == "seg" and "mask_path" in samples:
            mask = _read_mask(_resolve_path(data_root, row["mask_path"]))
            if mask.shape != shape:
                raise ValueError(f"Kích thước mask không khớp ảnh: {path}")
        images.append(path)
    return images


def evaluate(predictor: object, task: str, samples: pd.DataFrame, data_root: Path, out_dir: Path,
             warmup: int, save_overlays: int) -> dict:
    if samples.empty or warmup < 0 or save_overlays < 0:
        raise ValueError("Mẫu rỗng hoặc warmup/save_overlays âm")
    images = validate_samples(samples, task, data_root)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows, probabilities = [], []
    ground_truth = ("label" if task == "cls" else "mask_path") in samples
    indexed = dict(zip(images, samples.to_dict("records")))

    def record(path: Path, output: np.ndarray) -> None:
        source = indexed[path]
        row = {"image_id": source["image_id"], "image_path": str(path)}
        if task == "cls":
            validate_probabilities(output[np.newaxis, :])
            probabilities.append(output)
            row.update(predicted_label=int(output.argmax()), p_mel=float(output[0]),
                       p_nev=float(output[1]), p_sk=float(output[2]))
            if ground_truth:
                row["label"] = int(source["label"])
        else:
            with Image.open(path) as image:
                shape = (image.height, image.width)
            # Cùng validation dùng cho cả smoke-only và evaluation.
            reference = _read_mask(_resolve_path(data_root, source["mask_path"])) if ground_truth else np.zeros(shape, bool)
            measures = segmentation_metrics(reference, output)
            row.update(area_ratio=float(output.mean()), prediction_empty=measures["prediction_empty"])
            if ground_truth:
                row.update(measures)
        rows.append(row)

    report = run_benchmark(predictor, task, images, warmup, out_dir, save_overlays, on_prediction=record)
    table = pd.DataFrame(rows)
    metrics = None
    if ground_truth:
        if task == "cls":
            metrics = classification_metrics(table["label"].to_numpy(), np.stack(probabilities))
        else:
            metrics = {name: float(table[name].mean()) for name in SEG_METRICS}
            metrics.update(n_reference_empty=int(table["reference_empty"].sum()),
                           n_prediction_empty=int(table["prediction_empty"].sum()))
    warnings = [] if ground_truth else ["Thiếu ground truth: chỉ chạy smoke test, không tính metric"]
    if task == "cls" and metrics and metrics["missing_classes"]:
        warnings.append("Thiếu lớp trong mẫu; xem support và AUROC null trong báo cáo")
    report.update(metrics=metrics, warnings=warnings, split=samples.attrs.get("split"),
                  seed=samples.attrs.get("seed"), excluded_counts=samples.attrs.get("excluded_counts", {}),
                  timing_scope="Inference gồm preprocessing/postprocessing; không gồm model load, đọc ảnh, khởi chạy worker hoặc tính metric")
    table.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["cls", "seg"], required=True)
    ap.add_argument("--panderm-dir", type=Path, required=True)
    ap.add_argument("--pretrained", type=Path)
    ap.add_argument("--finetuned", type=Path)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--n", type=int, default=20, help="0 = toàn split")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save-overlays", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        ap.error("out-dir đã tồn tại; cần thư mục run mới để không ghi đè")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        handler = logging.StreamHandler(log)
        logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()], force=True)
        try:
            if args.warmup < 0 or args.save_overlays < 0:
                raise ValueError("warmup và save-overlays không được âm")
            if not args.panderm_dir.is_dir() or not args.data_root.is_dir():
                raise ValueError("Thư mục PanDerm hoặc data root không tồn tại")
            if args.task == "seg" and args.pretrained is None:
                raise ValueError("Segmentation cần --pretrained để dựng backbone")
            if args.pretrained is None and args.finetuned is None:
                raise ValueError("Cần checkpoint pretrain hoặc fine-tune")
            for path in (args.pretrained, args.finetuned):
                if path is not None and not path.is_file():
                    raise ValueError(f"Checkpoint không tồn tại: {path}")
            if "trainphase" in args.manifest.name.lower():
                raise ValueError("Không dùng manifest trainphase để đánh giá; dùng CSV split thật")
            samples = select_samples(pd.read_csv(args.manifest), args.task, args.split, args.n, args.seed)
            validate_samples(samples, args.task, args.data_root)
            import torch
            from nckh.infer import ClsPredictor, SegPredictor

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if args.task == "cls":
                predictor = ClsPredictor.from_checkpoint(args.panderm_dir, 3, args.finetuned, args.pretrained, device)
            else:
                predictor = SegPredictor.from_checkpoint(args.panderm_dir, args.pretrained, args.finetuned, device)
            report = evaluate(predictor, args.task, samples, args.data_root, args.out_dir, args.warmup, args.save_overlays)
            report["checkpoint_mode"] = "finetuned" if args.finetuned else "pretrained_only"
            if not args.finetuned:
                report["warnings"].append("Head chưa fine-tune: kết quả chỉ kiểm tra pipeline, không đánh giá chất lượng model đã huấn luyện")
            (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            inputs = {"manifest": args.manifest}
            inputs.update({name: path for name, path in (("pretrained", args.pretrained), ("finetuned", args.finetuned)) if path is not None})
            write_run_card(args.out_dir, seed=args.seed, config={key: str(value) for key, value in vars(args).items()}, inputs=inputs)
            logger.info("SMOKE TEST OK: %s, %s ảnh", args.task, report["n"])
        except Exception:
            logger.exception("Worker thất bại")
            raise
        finally:
            handler.flush()
            log.seek(0)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "worker.log").write_text(log.read(), encoding="utf-8")


if __name__ == "__main__":
    main()
