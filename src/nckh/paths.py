# file: src/nckh/paths.py
"""Đường dẫn dự án.

Đọc biến môi trường mỗi lần gọi (không cache ở mức module) để test và notebook
có thể đổi NCKH_ROOT mà không phải import lại.
"""
import os
from pathlib import Path

DEFAULT_ROOT = str(Path.home() / "nckh_root")
DEFAULT_LOCAL_DATA = str(Path.home() / "nckh_data")


def project_root() -> Path:
    return Path(os.environ.get("NCKH_ROOT", DEFAULT_ROOT))


def data_dir() -> Path:
    return project_root() / "data"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def runs_dir() -> Path:
    return project_root() / "runs"


def local_data_root() -> Path:
    # Ảnh giải nén để riêng khỏi NCKH_ROOT: nặng hàng chục GB, tải lại được, không cần sao lưu cùng runs/.
    return Path(os.environ.get("NCKH_LOCAL_DATA", DEFAULT_LOCAL_DATA))