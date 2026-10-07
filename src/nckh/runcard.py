# file: src/nckh/runcard.py
"""Run card: bản ghi đủ thông tin để truy ngược một kết quả về code, dữ liệu và môi trường."""
import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scikit-image",
    "torch", "torchvision", "mmsegmentation", "timm",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    # Đọc theo khối để băm được checkpoint vài GB mà không tràn RAM.
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def new_run_id(tag: str, now: datetime | None = None) -> str:
    # Tiền tố thời gian giúp sắp xếp các run và không bao giờ ghi đè run cũ.
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}_{tag}"


def _git_commit() -> str:
    repo_dir = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Không lấy được git commit (%s); ghi 'unknown'", exc)
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            # Mỗi runtime chỉ cài một phần các gói; thiếu gói là bình thường.
            continue
    return versions


def _gpu_info() -> dict | None:
    try:
        import torch
    except ImportError:
        logger.warning("Không có torch trong môi trường này; bỏ qua thông tin GPU")
        return None
    if not torch.cuda.is_available():
        logger.warning("torch không thấy CUDA; run này chạy trên CPU")
        return None
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "total_mb": round(props.total_memory / 2**20)}


def write_run_card(run_dir: Path, *, seed: int, config: dict, inputs: dict[str, Path]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "gpu": _gpu_info(),
        "seed": seed,
        "config": config,
        "inputs": {name: {"path": str(p), "sha256": sha256_file(Path(p))} for name, p in inputs.items()},
    }
    out = run_dir / "run_card.json"
    out.write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _key_values(items: list[str]) -> dict[str, str]:
    pairs = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Sai định dạng '{item}', cần dạng tên=giá_trị")
        key, value = item.split("=", 1)
        pairs[key] = value
    return pairs


def main(argv: list[str] | None = None) -> None:
    # CLI để gọi từ cell Colab bằng python của venv, tránh phải escape dấu ngoặc nhọn trong lệnh "!".
    ap = argparse.ArgumentParser(description="Ghi run_card.json cho một thư mục run")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--input", action="append", default=[], help="tên=đường_dẫn (lặp lại được)")
    ap.add_argument("--config", action="append", default=[], help="khóa=giá_trị (lặp lại được)")
    args = ap.parse_args(argv)
    inputs = {k: Path(v) for k, v in _key_values(args.input).items()}
    out = write_run_card(args.run_dir, seed=args.seed, config=_key_values(args.config), inputs=inputs)
    print("Đã ghi", out)


if __name__ == "__main__":
    main()