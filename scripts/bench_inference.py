# file: scripts/bench_inference.py
"""Smoke test + benchmark suy luận PanDerm (seg hoặc cls) trên một nhóm ảnh nhỏ.

Chạy bằng Python của venv tương ứng (venv_seg / venv_cls), không phải Python của .venv.
Ví dụ:
  ~/venvs/venv_seg/bin/python scripts/bench_inference.py --task seg \
      --panderm-dir ~/PanDerm/segmentation --pretrained ~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth \
      --images "$HOME/nckh_data/ISIC2018/Validation_Data/*.jpg" --n 20 --out-dir ~/nckh_root/runs/<run_id>
"""
import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_benchmark(predictor: object, task: str, images: list[Path], warmup: int, out_dir: Path,
                  save_overlays: int) -> dict:
    import torch

    if not images:
        raise ValueError("Không có ảnh nào để chạy; kiểm tra lại --images")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = predictor.device
    cuda = device.startswith("cuda")
    rgbs = [np.asarray(Image.open(p).convert("RGB")) for p in images]

    # Lượt warmup không tính giờ: lần chạy đầu của CUDA tốn thêm thời gian khởi tạo kernel.
    for rgb in rgbs[:warmup]:
        predictor.predict(rgb)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    times_ms, outputs = [], []
    for rgb in rgbs:
        _sync(device)
        t0 = time.perf_counter()
        outputs.append(predictor.predict(rgb))
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000)

    if task == "seg":
        checks = {
            "all_shapes_match": all(o.shape == r.shape[:2] for o, r in zip(outputs, rgbs)),
            "area_ratio": [round(float(o.mean()), 4) for o in outputs],
            "n_empty_masks": int(sum(not o.any() for o in outputs)),
        }
        if save_overlays:
            from skimage.segmentation import mark_boundaries

            (out_dir / "overlays").mkdir(exist_ok=True)
            for path, rgb, mask in list(zip(images, rgbs, outputs))[:save_overlays]:
                vis = (mark_boundaries(rgb, mask.astype(int), color=(0, 1, 1)) * 255).astype(np.uint8)
                Image.fromarray(vis).save(out_dir / "overlays" / f"{path.stem}.png")
    else:
        probs = np.stack(outputs)
        checks = {
            "all_finite": bool(np.isfinite(probs).all()),
            "max_abs_prob_sum_error": float(np.abs(probs.sum(axis=1) - 1).max()),
        }
        with open(out_dir / "cls_probs_smoke.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_id", "p_mel", "p_nev", "p_sk"])
            for path, p in zip(images, probs):
                writer.writerow([path.stem, *[f"{v:.6f}" for v in p]])

    report = {
        "task": task,
        "n": len(images),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "ms_per_image_mean": float(np.mean(times_ms)),
        "ms_per_image_p50": float(np.percentile(times_ms, 50)),
        "ms_per_image_p95": float(np.percentile(times_ms, 95)),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "checks": checks,
    }
    (out_dir / f"bench_{task}.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    import torch

    from nckh.infer import ClsPredictor, SegPredictor

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["seg", "cls"], required=True)
    ap.add_argument("--panderm-dir", type=Path, required=True, help="PanDerm/segmentation hoặc PanDerm/classification")
    ap.add_argument("--pretrained", type=Path, help="checkpoint pretrain PanDerm Base")
    ap.add_argument("--finetuned", type=Path, help="checkpoint fine-tune (bỏ trống khi smoke test P2)")
    ap.add_argument("--images", required=True, help='glob, ví dụ "$HOME/nckh_data/ISIC2018/Validation_Data/*.jpg"')
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save-overlays", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task == "seg":
        predictor = SegPredictor.from_checkpoint(args.panderm_dir, args.pretrained, args.finetuned, device)
    else:
        predictor = ClsPredictor.from_checkpoint(args.panderm_dir, 3, args.finetuned, args.pretrained, device)
    images = [Path(p) for p in sorted(glob.glob(args.images))[: args.n]]
    report = run_benchmark(predictor, args.task, images, args.warmup, args.out_dir, args.save_overlays)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    failed = (args.task == "seg" and not report["checks"]["all_shapes_match"]) or (
        args.task == "cls" and (not report["checks"]["all_finite"] or report["checks"]["max_abs_prob_sum_error"] > 1e-4))
    if failed:
        raise SystemExit(f"SMOKE TEST FAILED: {report['checks']}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()