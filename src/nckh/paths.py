"""
Đường dẫn dữ án.
đọc biến môi trường mỗi lần gọi để test và notebook có thể đổi NCKH_ROOT mà không phải import lại
"""

import os
from pathlib import Path

DEFAULT_ROOT = '/content/drive/Mydrive/NCKH_PanDerm'
DEFAULT_LOCAL_DATA = '/content/data'

def project_root() -> Path:
    return Path(os.environ.get("NCKH_ROOT"), DEFAULT_ROOT)
def data_dir() -> Path:
    return project_root() / "data"
def manifests_dir() -> Path:
    return data_dir() / "manifests"
def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"
def runs_dir() -> Path:
    return project_root() / "runs"
def local_data_root() -> Path:
    # ảnh giải nén nằm trên đĩa /content của colab 
    return Path(os.environ.get("NCKH_LOCAL_DATA", DEFAULT_LOCAL_DATA))