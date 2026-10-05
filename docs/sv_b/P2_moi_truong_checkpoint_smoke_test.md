# P2 — Môi trường, checkpoint PanDerm Base và smoke test thật

> Thay thế toàn bộ hướng dẫn Tuần 1 cũ (`docs/Huong_dan_SV_B_Tuan_1_*.md`). Bản cũ có 3 lỗi: smoke test dùng `torch.rand` thay cho model (VRAM 1,34 MB chứng tỏ chưa nạp model nào), link checkpoint `example.com` trả 404, và cài `mmcv>=2.0.0` sẽ kéo về mmcv 2.2.0, khiến `import mmseg` lỗi.

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 2):** dựng môi trường sạch cho hai nhánh PanDerm, tải checkpoint Base chính thức, sửa code segmentation để chạy được ViT-B, rồi chạy **smoke test thật** trên 20 ảnh. Smoke test phải kiểm: nạp được checkpoint, đầu ra đúng kích thước, xác suất hữu hạn và có tổng bằng 1, không lỗi khi ảnh có tỷ lệ khác nhau, đo thời gian và VRAM.

| Đầu vào | Đầu ra (trên Drive) |
|---|---|
| Repo `nckh` (GitHub), PanDerm upstream `fd7a807` | `nckh/` có package `src/nckh/`, `tests/`, `scripts/`, `patches/`, `.gitignore`, 3 notebook |
| Checkpoint PanDerm_Base (Google Drive ID `17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB`) | `checkpoints/panderm_bb_data6_checkpoint-499.pth` + `checkpoints/SHA256SUMS` |
| 20 ảnh ISIC 2018 Validation | `runs/<run_id>/bench_seg.json`, `bench_cls.json`, `overlays/*.png`, `cls_probs_smoke.csv`, `run_card.json` |

## 2. Chạy ở đâu

| Việc | Notebook | Phần cứng | Thời gian ước tính |
|---|---|---|---|
| Tạo file code, chạy pytest, tải checkpoint | `NB_cpu` | CPU | 15–30 phút lần đầu |
| Dựng `venv_seg`, soi checkpoint, áp patch, smoke test seg | `NB_seg` | GPU (T4 trở lên) | 10–15 phút cài + 2 phút chạy |
| Dựng `venv_cls`, smoke test cls | `NB_cls` | GPU | 8–12 phút cài + 1 phút chạy |

**Tạo 3 notebook một lần:** trên Colab chọn *File → New notebook*, đặt tên `NB_seg`, `NB_cls`, `NB_cpu`. Sau khi có repo trên Drive (mục 4.1), dùng *File → Save a copy in Drive* rồi chuyển file `.ipynb` vào `MyDrive/NCKH_PanDerm/nckh/notebooks/`. Từ đó mở notebook trực tiếp từ thư mục này để mọi thay đổi nằm trong repo.
Với `NB_seg` và `NB_cls`: *Runtime → Change runtime type → T4 GPU*.

## 3. Giải thích

### 3.1. Vì sao code GPU phải chạy bằng `!{VENV}/bin/python script.py`

Kernel của notebook Colab luôn là Python mặc định của Colab (3.11/3.12). Venv Python 3.10 tạo bằng `uv` **không** phải kernel. Vì vậy:
- Cell Python thường (mount Drive, git, tạo `run_id`) chạy trong kernel.
- Mọi việc cần torch/mmseg/timm đúng phiên bản phải chạy dưới dạng **script** gọi bằng python của venv, ví dụ `!{VENV}/bin/python scripts/bench_inference.py ...`.

Đó là lý do smoke test được viết thành `scripts/bench_inference.py` thay vì dán code vào cell.

### 3.2. Vì sao venv segmentation dùng torch 2.1.2 chứ không phải 2.2.1 như README upstream

`Segmentation.md` hướng dẫn `torch==2.2.1` và `mim install mmcv==2.1.0`. Kiểm tra kho wheel của OpenMMLab (05/10/2026):

| torch | CUDA | wheel mmcv dựng sẵn |
|---|---|---|
| 2.1.x | cu118 | **mmcv 2.1.0** (cp310, cp311) |
| 2.2.x | cu118 | chỉ **mmcv 2.2.0** |

Trong khi đó `mmsegmentation==1.2.2` có dòng `assert mmcv_min_version <= mmcv_version < '2.2.0'`, nghĩa là bắt buộc dùng mmcv < 2.2.0. Với torch 2.2.1 thì không có wheel mmcv 2.1.0, nên `mim` sẽ phải build mmcv từ mã nguồn. Bước build này cần nvcc cùng phiên bản CUDA với torch (11.8), mà Colab chỉ có CUDA 12, nên việc build thực tế sẽ hỏng.

**Quyết định:** `venv_seg` dùng `torch==2.1.2`, `torchvision==0.16.2` (cu118) cùng wheel `mmcv==2.1.0` dựng sẵn. Tổ hợp torch 2.1 + mmcv 2.1.0 + mmseg 1.2.2 + code PanDerm đã áp patch đã được chạy thử khi viết hướng dẫn: dựng model ViT-B, forward ra `(1, 2, 224, 224)`. **Ghi sai lệch này vào nhật ký quyết định** (kế hoạch Mục 10): ngày, lý do như trên, ảnh hưởng là "môi trường khác README upstream một bản phụ của torch".

Nhánh classification giữ đúng README: `torch==2.4.1`. Riêng `timm` được ghim ở `0.9.16`, vì `modeling_finetune.py` import `timm.models.layers` và `timm.models.registry`, trong khi `classification/requirements.txt` không ghim phiên bản.

### 3.3. Vì sao phải sửa (patch) code segmentation upstream

Code segmentation của PanDerm hiện chỉ chạy được bản **Large**:
- `segmentation/models/cae_config.py` cố định `embed_dim=1024, depth=24, num_heads=16`.
- `segmentation/models/cae_seg.py` đọc cứng `model_weights/panderm_ll_data6_checkpoint-499.pth` (file của bản Large).
- Vòng nạp trọng số có lỗi: nhánh `else` gọi `model_dict[k]` cả khi `k` **không có** trong `model_dict`, nên văng `KeyError` khi gặp key lạ.

Patch `patches/panderm_base_seg.patch` sửa 3 chỗ:
1. **Config ViT-B/16:** `embed_dim=768, depth=12, num_heads=12`. Đầu ra lấy ở các block `[3, 5, 7, 11]`, tương đương các block `[7, 11, 15, 23]` của bản 24 tầng. Đầu UPerHead đổi `in_channels`/`channels` thành 768.
2. **Đường dẫn checkpoint** lấy từ biến môi trường `PANDERM_CKPT`, không còn đọc cứng.
3. **Nạp trọng số an toàn:**
   - không còn `KeyError`;
   - in ra số key đã nạp và độ phủ của các trọng số ViT (`blocks.*`, `patch_embed.*`, `cls_token`);
   - **dừng với `RuntimeError` nếu độ phủ < 90%**. Nếu không có bước này, model sẽ âm thầm train từ trọng số ngẫu nhiên và bạn chỉ phát hiện khi kết quả kém bất thường.

Các lớp `fpn1..4` và `norm` là lớp mới của đầu segmentation, không có trong checkpoint pretrain, nên không được tính vào độ phủ.

### 3.4. Smoke test kiểm tra gì

`scripts/bench_inference.py` dựng predictor thật (`nckh.infer.SegPredictor` / `ClsPredictor`), chạy lần lượt từng ảnh và ghi `bench_<task>.json`:

- **seg:** mask có cùng kích thước ảnh gốc (ảnh ISIC có nhiều tỷ lệ khác nhau, nên đây cũng là phép thử "ảnh khác tỷ lệ"), tỷ lệ diện tích từng ảnh, số mask rỗng. Ghi 10 ảnh overlay để kiểm bằng mắt.
- **cls:** xác suất hữu hạn, |tổng − 1| < 1e-4, ghi `cls_probs_smoke.csv`.
- **Cả hai:** ms/ảnh (trung bình, p50, p95, đã warmup và `cuda.synchronize`), VRAM đỉnh, tên GPU.

Ở P2 **chưa có checkpoint fine-tune**, nên đầu segmentation còn khởi tạo ngẫu nhiên và mask trông vô nghĩa. Đó là bình thường: smoke test chỉ chứng minh đường ống chạy đúng (nạp checkpoint, shape, tốc độ). Không báo cáo hiệu năng từ bước này.

### 3.5. Tiền xử lý phải khớp upstream

`src/nckh/infer.py` sao chép đúng phần tiền xử lý của upstream để xác suất và mask khi suy luận (P6, P8, P9) khớp với lúc đánh giá:

| Nhánh | Upstream | `nckh.infer` |
|---|---|---|
| seg | `dataset_seg.py`: `cv2.resize(224×224, INTER_CUBIC)` → `Normalize(0.5, 0.5)` | PIL `BICUBIC` 224×224 → chuẩn hóa 0,5 (lệch rất nhỏ so với OpenCV; đo trong pilot P5a) |
| seg hậu xử lý | `largestConnectComponent`: giữ vùng lớn nhất + lấp lỗ | Giống hệt, **trừ** mask rỗng: upstream biến mask rỗng thành toàn ảnh (nhãn 0 = nền bị chọn), còn `nckh` giữ rỗng |
| cls | `run_class_finetuning.py` dòng 268–271: `Resize(256)` (bilinear) → `CenterCrop(224)` → `Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))` | Giống hệt (giữ cả `0.228` của upstream) |

## 4. Code

### 4.1. Cell đồng bộ Drive ↔ GitHub (đầu **cả 3** notebook)

Làm theo `00_tong_quan_va_lo_trinh.md` mục 5 (token `GH_TOKEN` trong Colab Secrets), rồi dán cell sau vào đầu mỗi notebook:

```python
# cell: NB_cpu
import os, sys
from pathlib import Path
from google.colab import drive, userdata
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/NCKH_PanDerm'
os.environ['NCKH_ROOT'] = ROOT
REPO_URL = f"https://{userdata.get('GH_TOKEN')}@github.com/NeuroDev204/nckh.git"
!mkdir -p {ROOT}
!test -d {ROOT}/nckh/.git || git clone -q {REPO_URL} {ROOT}/nckh
%cd {ROOT}/nckh
!git remote set-url origin {REPO_URL}
!git config user.name "NeuroDev204" && git config user.email "eduteam.hutech@gmail.com"
!git pull --rebase -q && git log --oneline -1
# Package nckh chỉ dùng thư viện thuần Python, nên kernel đọc được trực tiếp từ src/ mà không cần cài.
sys.path.insert(0, f'{ROOT}/nckh/src')
```

> Lần đầu repo chưa có `src/`: cell vẫn chạy được, dòng `sys.path` không gây lỗi.

### 4.2. Tạo package `nckh` (trong `NB_cpu`)

Tạo các file dưới đây trong `MyDrive/NCKH_PanDerm/nckh/`. Có thể dùng trình sửa file của Colab (biểu tượng thư mục → chuột phải → *New file*), hoặc tạo trên máy rồi push lên. Khối nào bắt đầu bằng `# file:` thì chép **nguyên văn**, kể cả dòng `# file:` đầu tiên (dòng này là comment, vô hại).

**`pyproject.toml`**: khai báo package. Phụ thuộc lõi **không có torch**, để cài được vào cả 3 môi trường mà không làm lệch phiên bản torch.

```toml
# file: pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nckh"
version = "0.1.0"
description = "Pipeline nghiên cứu PanDerm: manifest, ghép cặp, Ridge, metric, robustness"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "scipy",
    "pillow",
    "joblib",
    "pyyaml",
]

[project.optional-dependencies]
demo = ["streamlit", "matplotlib"]
test = ["pytest", "openpyxl"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**`src/nckh/__init__.py`**

```python
# file: src/nckh/__init__.py
"""Package nckh: code nghiên cứu PanDerm của nhóm."""
```

**`src/nckh/paths.py`**: mọi đường dẫn lấy từ biến môi trường, nên cùng một code chạy được trên Colab (`/content/drive/...`) và trong test (`tmp_path`).

```python
# file: src/nckh/paths.py
"""Đường dẫn dự án.

Đọc biến môi trường mỗi lần gọi (không cache ở mức module) để test và notebook
có thể đổi NCKH_ROOT mà không phải import lại.
"""
import os
from pathlib import Path

DEFAULT_ROOT = "/content/drive/MyDrive/NCKH_PanDerm"
DEFAULT_LOCAL_DATA = "/content/data"


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
    # Ảnh giải nén nằm trên đĩa /content của Colab (nhanh) chứ không trên Drive.
    return Path(os.environ.get("NCKH_LOCAL_DATA", DEFAULT_LOCAL_DATA))
```

**`src/nckh/runcard.py`**: mỗi lần chạy ghi một `run_card.json` (kế hoạch Mục 7). Có CLI `python -m nckh.runcard` để gọi từ cell bằng python của venv; venv thì biết đúng phiên bản torch và GPU.

```python
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
```

**`.gitignore`** (thay toàn bộ file cũ): chặn ảnh, trọng số, dữ liệu, `runs/`; mở `docs/sv_b/` để commit được hướng dẫn.

```gitignore
# file: .gitignore
# Chỉ commit docs của SV B và spec/plan; các docs cũ khác giữ ngoài git như trước.
/docs/*
!/docs/sv_b/
!/docs/superpowers/
# Môi trường, cache
.venv*/
__pycache__/
*.egg-info/
.ipynb_checkpoints/
.pytest_cache/
.idea/
# Dữ liệu và trọng số: không bao giờ lên GitHub
*.pth
*.ckpt
*.joblib
data/
runs/
```

**`tests/test_paths_runcard.py`**

```python
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
```

Cài và chạy test trong `NB_cpu`:

```python
# cell: NB_cpu
!pip install -q -e "{ROOT}/nckh[demo,test]"
!cd {ROOT}/nckh && python -m pytest -q tests/test_paths_runcard.py
```

Kỳ vọng: `5 passed`.

### 4.3. Tải checkpoint PanDerm_Base (trong `NB_cpu`)

Đọc điều khoản trước: PanDerm phát hành theo CC BY-NC-ND 4.0, chỉ dùng cho nghiên cứu phi thương mại và phải ghi nguồn.

```python
# cell: NB_cpu
CK = f'{ROOT}/checkpoints/panderm_bb_data6_checkpoint-499.pth'
!mkdir -p {ROOT}/checkpoints
!test -f {CK} || gdown 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB -O {CK}
!ls -lh {CK}
from nckh.runcard import sha256_file
digest = sha256_file(Path(CK))
print(digest)
with open(f'{ROOT}/checkpoints/SHA256SUMS', 'a') as fh:
    fh.write(f'{digest}  panderm_bb_data6_checkpoint-499.pth  # tải {__import__("datetime").date.today()}\n')
```

Nếu `gdown` báo vượt quota: mở `https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view` trên trình duyệt, chọn *Add shortcut to Drive* (hoặc *Make a copy*), rồi `cp` từ `MyDrive` sang `{ROOT}/checkpoints/`.

### 4.4. Dựng `venv_seg` (trong `NB_seg`)

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2. Các lệnh dưới đây đã được thử với bản CPU tương đương (torch 2.1.0 CPU + wheel mmcv 2.1.0 CPU); trên Colab dùng bản cu118.

```python
# cell: NB_seg
VENV = '/content/venv_seg'
CK = f'{ROOT}/checkpoints/panderm_bb_data6_checkpoint-499.pth'
!test -d /content/PanDerm || git clone -q https://github.com/SiyuanYan1/PanDerm.git /content/PanDerm
!cd /content/PanDerm && git checkout -q fd7a80748ba7fc3e203fed88f909f4689d0d6f24 && git log --oneline -1
!pip install -q uv
!uv python install 3.10
!test -x {VENV}/bin/python || uv venv -q --python 3.10 --seed {VENV}
# setuptools<81: mmengine/mmseg còn dùng pkg_resources đã bị bỏ ở setuptools 81.
!uv pip install -q --python {VENV}/bin/python "setuptools<81" wheel
!uv pip install -q --python {VENV}/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
!uv pip install -q --python {VENV}/bin/python mmengine==0.10.4 mmcv==2.1.0 mmsegmentation==1.2.2 --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
!uv pip install -q --python {VENV}/bin/python -r /content/PanDerm/segmentation/requirements.txt openpyxl
!uv pip install -q --python {VENV}/bin/python -e {ROOT}/nckh
!{VENV}/bin/python -c "import torch, mmcv, mmseg; print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')"
```

Kỳ vọng dòng cuối: `torch 2.1.2+cu118 | mmcv 2.1.0 | mmseg 1.2.2 | GPU Tesla T4` (hoặc tên GPU khác).

Venv nằm trên `/content` nên mất khi runtime bị ngắt; chạy lại cell này (khoảng 5–10 phút). **Không** đặt venv trên Drive: chậm và dễ hỏng. Xoá thư mục `MyDrive/NCKH_PanDerm/.venv-panderm` cũ để giải phóng dung lượng.

### 4.5. Soi key checkpoint (trong `NB_seg`)

**`scripts/inspect_checkpoint.py`**

```python
# file: scripts/inspect_checkpoint.py
"""Soi cấu trúc key của checkpoint PanDerm trước khi áp patch.

Chạy: /content/venv_seg/bin/python scripts/inspect_checkpoint.py $ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth
"""
import json
import sys
from collections import Counter
from pathlib import Path

WRAPPER_KEYS = {"model", "state_dict", "module"}


def summarize_state_dict(sd: dict) -> dict:
    wrapped_in = None
    if set(sd) <= WRAPPER_KEYS:
        wrapped_in = next(iter(sd))
        sd = sd[wrapped_in]
    prefix_counts = dict(Counter(k.split(".")[0] for k in sd).most_common())
    weight = sd.get("encoder.patch_embed.proj.weight", sd.get("patch_embed.proj.weight"))
    shape = list(weight.shape) if weight is not None else None
    has_encoder = "encoder.patch_embed.proj.weight" in sd
    if shape is None:
        verdict = "Không tìm thấy patch_embed.proj.weight — checkpoint lạ, dừng lại hỏi nhóm"
    elif shape[0] == 1024:
        verdict = "ViT-L (1024) — đây là PanDerm Large, không phải Base"
    elif shape[0] == 768 and has_encoder:
        verdict = "ViT-B (768) với prefix encoder. — dùng được với patch"
    else:
        verdict = "ViT-B (768) nhưng không có prefix encoder. — sửa dòng replace('encoder.', '') trong patch"
    return {
        "wrapped_in": wrapped_in,
        "n_keys": len(sd),
        "prefix_counts": prefix_counts,
        "first_keys": list(sd)[:10],
        "patch_embed_shape": shape,
        "rel_pos_keys": [k for k in sd if "rel_pos" in k][:3],
        "verdict": verdict,
    }


if __name__ == "__main__":
    import torch

    state = torch.load(Path(sys.argv[1]), map_location="cpu")
    print(json.dumps(summarize_state_dict(state), indent=2, ensure_ascii=False))
```

**`tests/test_inspect_checkpoint.py`**

```python
# file: tests/test_inspect_checkpoint.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inspect_checkpoint import summarize_state_dict  # noqa: E402


def test_summarize_counts_prefix_and_patch_shape() -> None:
    sd = {
        "encoder.patch_embed.proj.weight": np.zeros((768, 3, 16, 16)),
        "encoder.blocks.0.norm1.weight": np.zeros(768),
        "decoder.x": np.zeros(1),
    }
    s = summarize_state_dict(sd)
    assert s["prefix_counts"] == {"encoder": 2, "decoder": 1}
    assert s["patch_embed_shape"] == [768, 3, 16, 16]
    assert s["verdict"] == "ViT-B (768) với prefix encoder. — dùng được với patch"


def test_summarize_unwraps_and_flags_large() -> None:
    s = summarize_state_dict({"model": {"encoder.patch_embed.proj.weight": np.zeros((1024, 3, 16, 16))}})
    assert s["wrapped_in"] == "model"
    assert s["verdict"].startswith("ViT-L")


def test_summarize_flags_missing_prefix() -> None:
    s = summarize_state_dict({"patch_embed.proj.weight": np.zeros((768, 3, 16, 16))})
    assert "không có prefix encoder." in s["verdict"]
```

```python
# cell: NB_seg
!{VENV}/bin/python {ROOT}/nckh/scripts/inspect_checkpoint.py {CK}
```

| Bạn thấy `verdict` | Làm gì |
|---|---|
| `ViT-B (768) với prefix encoder. — dùng được với patch` | Đi tiếp 4.6 |
| `ViT-L (1024) …` | Tải nhầm bản Large. Kiểm tra lại ID Google Drive ở 4.3 |
| `ViT-B (768) nhưng không có prefix encoder.` | Trong patch, ở dòng `new_state_dict = {k.replace('encoder.', ''): v ... if 'encoder' in k}`, đổi `'encoder.'`/`'encoder'` thành prefix thật (xem `prefix_counts`). Ghi vào nhật ký quyết định |
| `Không tìm thấy patch_embed.proj.weight` | Dừng lại, gửi `first_keys` cho nhóm/giảng viên |

### 4.6. Áp patch segmentation (trong `NB_seg`)

Tạo file `patches/panderm_base_seg.patch` với **đúng** nội dung dưới đây. Khối này không có dòng `# file:` vì file patch không được có dòng thừa. File phải kết thúc bằng một dòng trống.

```diff
diff --git a/segmentation/models/cae_config.py b/segmentation/models/cae_config.py
index 487d5d4..cd2a09b 100644
--- a/segmentation/models/cae_config.py
+++ b/segmentation/models/cae_config.py
@@ -18,24 +18,24 @@ model = dict(
     backbone=dict(
         type='CAE',
         patch_size=16,
-        embed_dim=1024,
-        depth=24,
-        num_heads=16,
+        embed_dim=768,
+        depth=12,
+        num_heads=12,
         mlp_ratio=4,
         qkv_bias=True,
         use_abs_pos_emb=False,
         use_rel_pos_bias=False,
         init_values=1e-5,
         drop_path_rate=0.15,
-        out_indices=[7, 11, 15, 23],
+        out_indices=[3, 5, 7, 11],
         out_with_norm=True,
     ),
     decode_head=dict(
         type='UPerHead',
-        in_channels=[1024, 1024, 1024, 1024],
+        in_channels=[768, 768, 768, 768],
         in_index=[0, 1, 2, 3],
         pool_scales=(1, 2, 3, 6),
-        channels=1024,
+        channels=768,
         dropout_ratio=0.0,
         num_classes=2,
         norm_cfg=norm_cfg,
diff --git a/segmentation/models/cae_seg.py b/segmentation/models/cae_seg.py
index 0f5f5e7..d01c3f0 100644
--- a/segmentation/models/cae_seg.py
+++ b/segmentation/models/cae_seg.py
@@ -1,3 +1,5 @@
+import os
+
 import torch
 import torch.nn as nn
 from mmengine.config import Config
@@ -13,18 +15,32 @@ class CAEv2_seg(nn.Module):
         self.segmentor = build_segmentor(config.model)
         self.segmentor.init_weights()
 
-        print('=> Loading CAE weights from xxx')
-        cae_weight = torch.load('model_weights/panderm_ll_data6_checkpoint-499.pth', map_location='cpu')
+        # PanDerm Base (ViT-B/16) thay cho ban Large ma upstream doc cung; duong dan lay tu bien moi truong.
+        ckpt_path = os.environ.get('PANDERM_CKPT', 'model_weights/panderm_bb_data6_checkpoint-499.pth')
+        print('=> Loading CAE weights from', ckpt_path)
+        cae_weight = torch.load(ckpt_path, map_location='cpu')
         new_state_dict = {k.replace('encoder.', ''): v for k, v in cae_weight.items() if 'encoder' in k}
 
         model_dict = self.segmentor.backbone.state_dict()
         matched_dict = {}
+        unexpected = []
 
         for k, v in new_state_dict.items():
             if k in model_dict and model_dict[k].size() == v.size():
                 matched_dict[k] = v
-            else:
+            elif k in model_dict:
                 print(f'Skipping {k} due to size mismatch: {v.size()} vs {model_dict[k].size()}')
+            else:
+                unexpected.append(k)
+
+        # fpn*/norm la lop moi cua dau segmentation, khong co trong checkpoint pretrain: chi tinh tren ViT.
+        vit_keys = [k for k in model_dict if k.startswith(('blocks.', 'patch_embed.', 'cls_token'))]
+        ratio = sum(k in matched_dict for k in vit_keys) / max(len(vit_keys), 1)
+        print(f'=> Matched {len(matched_dict)}/{len(model_dict)} backbone keys; ViT coverage {ratio:.1%}; '
+              f'{len(unexpected)} unexpected, e.g. {unexpected[:5]}')
+        if ratio < 0.9:
+            # Nap qua it trong so nghia la dang train tu khoi tao ngau nhien: dung ngay thay vi im lang.
+            raise RuntimeError(f'Only {ratio:.1%} of backbone weights matched {ckpt_path}; check key prefix/architecture')
 
         model_dict.update(matched_dict)
         self.segmentor.backbone.load_state_dict(model_dict, strict=False)
```

```python
# cell: NB_seg
PATCH = f'{ROOT}/nckh/patches/panderm_base_seg.patch'
# Kiểm tra theo chiều ngược trước, để chạy lại cell này nhiều lần cũng không áp patch hai lần.
!cd /content/PanDerm && (git apply --reverse --check {PATCH} 2>/dev/null && echo "Patch đã áp từ trước") || (git apply {PATCH} && echo "Đã áp patch")
!cd /content/PanDerm && git diff --stat
```

Kỳ vọng: `segmentation/models/cae_config.py` và `segmentation/models/cae_seg.py` (cùng `segmentation/workers/train.py` sau khi bổ sung ở P5a) có thay đổi.

### 4.7. Module suy luận `nckh.infer`

**`src/nckh/infer.py`**: dùng chung cho smoke test (P2), suy luận UQ (P6), robustness (P8) và demo (P9).

```python
# file: src/nckh/infer.py
"""Suy luận PanDerm Base cho segmentation và classification.

torch chỉ được import bên trong hàm: package nckh phải cài được ở runtime CPU không có torch.
Tiền xử lý bám đúng code upstream để xác suất/mask khi suy luận khớp với lúc đánh giá:
- seg: datasets/dataset_seg.py → resize 224x224 bicubic, Normalize(0.5, 0.5).
- cls: run_class_finetuning.py (val_trans) → Resize(256) bilinear, CenterCrop(224),
  Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)).
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label

logger = logging.getLogger(__name__)

SEG_SIZE = 224
CLS_MEAN = (0.485, 0.456, 0.406)
CLS_STD = (0.228, 0.224, 0.225)  # upstream ghi 0.228, giữ nguyên để khớp lúc fine-tune
CLS_LABELS = ("melanoma", "nevus", "seborrheic_keratosis")


def seg_preprocess(rgb: np.ndarray) -> "torch.Tensor":
    import torch

    # Upstream dùng cv2.INTER_CUBIC; PIL BICUBIC lệch rất nhỏ (đo lại Dice trong pilot P2).
    img = Image.fromarray(rgb).resize((SEG_SIZE, SEG_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - 0.5) / 0.5).unsqueeze(0)


def cls_preprocess(rgb: np.ndarray) -> "torch.Tensor":
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLS_MEAN, CLS_STD),
    ])
    return tf(Image.fromarray(rgb)).unsqueeze(0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    # Giống largestConnectComponent của upstream (8-liên thông + lấp lỗ),
    # nhưng mask rỗng giữ nguyên rỗng thay vì biến thành toàn ảnh.
    labeled, num = label(mask.astype(bool), background=0, return_num=True)
    if num == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(labeled == sizes.argmax())


def seg_logits_to_mask(logits: "torch.Tensor", out_hw: tuple[int, int]) -> np.ndarray:
    pred = logits[0].argmax(dim=0).cpu().numpy().astype(bool)
    pred = largest_component(pred)
    resized = Image.fromarray(pred.astype(np.uint8) * 255).resize((out_hw[1], out_hw[0]), Image.NEAREST)
    return np.asarray(resized) > 127


def map_pretrained_cls_keys(state_dict: dict, num_layers: int) -> dict:
    """Đổi tên key checkpoint pretrain PanDerm theo đúng run_class_finetuning.py (dòng ~455–530)."""
    if set(state_dict) <= {"model", "state_dict", "module"}:
        state_dict = next(iter(state_dict.values()))
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith(("decoder.", "teacher.")) or "relative_position_index" in key:
            continue
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        if key.startswith("norm."):
            # Model fine-tune dùng mean pooling nên lớp norm cuối tên là fc_norm.
            key = "fc_norm." + key[len("norm."):]
        mapped[key] = value
    shared = mapped.pop("rel_pos_bias.relative_position_bias_table", None)
    if shared is not None:
        for i in range(num_layers):
            mapped[f"blocks.{i}.attn.relative_position_bias_table"] = shared.clone()
    return mapped


class _InDir:
    """chdir tạm thời (contextlib.chdir chỉ có từ Python 3.11, venv PanDerm là 3.10)."""

    def __init__(self, path: Path) -> None:
        self.path, self.old = path, None

    def __enter__(self) -> None:
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc: object) -> None:
        os.chdir(self.old)


def _add_to_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SegPredictor:
    def __init__(self, model: "torch.nn.Module", device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, panderm_seg_dir: Path, pretrained_path: Path,
                        finetuned_ckpt: Path | None = None, device: str = "cuda") -> "SegPredictor":
        import torch

        seg_dir = Path(panderm_seg_dir).resolve()
        # CAEv2_seg đọc pretrained qua biến môi trường (patch P2) và đọc config bằng đường dẫn tương đối.
        os.environ["PANDERM_CKPT"] = str(Path(pretrained_path).resolve())
        _add_to_sys_path(seg_dir)
        with _InDir(seg_dir):
            import models.cae_backbone  # noqa: F401  (đăng ký backbone "CAE" vào registry của mmseg, như run.py)
            from models.cae_seg import CAEv2_seg
            model = CAEv2_seg()
        if finetuned_ckpt is not None:
            state = torch.load(finetuned_ckpt, map_location="cpu")["state_dict"]
            # Checkpoint Lightning bọc CAEv2_seg trong thuộc tính "model".
            state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            model.load_state_dict(state, strict=True)
        else:
            logger.warning("Không có checkpoint fine-tune: decode head còn khởi tạo ngẫu nhiên (chỉ dùng cho smoke test)")
        return cls(model, device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model(seg_preprocess(rgb).to(self.device))
        return seg_logits_to_mask(logits, rgb.shape[:2])


class ClsPredictor:
    def __init__(self, model: "torch.nn.Module", device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, panderm_cls_dir: Path, nb_classes: int = 3, finetuned_ckpt: Path | None = None,
                        pretrained_path: Path | None = None, device: str = "cuda") -> "ClsPredictor":
        import torch

        _add_to_sys_path(Path(panderm_cls_dir).resolve())
        from models.modeling_finetune import panderm_base_patch16_224_finetune

        # Tham số kiến trúc = giá trị mặc định trong run_class_finetuning.py (layer scale 0.1, rel pos bias, mean pooling).
        model = panderm_base_patch16_224_finetune(
            pretrained=False, num_classes=nb_classes, drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=True,
            init_values=0.1, lin_probe=False,
        )
        if finetuned_ckpt is not None:
            model.load_state_dict(torch.load(finetuned_ckpt, map_location="cpu")["model"], strict=True)
        elif pretrained_path is not None:
            mapped = map_pretrained_cls_keys(torch.load(pretrained_path, map_location="cpu"), model.get_num_layers())
            own = model.state_dict()
            mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
            block_keys = [k for k in own if k.startswith("blocks.") and "relative_position_index" not in k]
            coverage = sum(k in mapped for k in block_keys) / max(len(block_keys), 1)
            logger.warning("Pretrained → %d/%d key khớp; độ phủ blocks %.1f%%", len(mapped), len(own), 100 * coverage)
            if coverage < 0.9:
                raise RuntimeError(f"Chỉ {coverage:.1%} trọng số blocks khớp với {pretrained_path}")
            model.load_state_dict(mapped, strict=False)
        else:
            raise ValueError("Cần finetuned_ckpt hoặc pretrained_path")
        return cls(model, device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model(cls_preprocess(rgb).to(self.device))
        return torch.softmax(logits, dim=1)[0].cpu().numpy()
```

Điểm cần hiểu:
- **`from_checkpoint` của seg:** `chdir` vào `PanDerm/segmentation`, vì `CAEv2_seg` đọc config bằng đường dẫn tương đối `models/cae_config.py`. Hàm cũng import `models.cae_backbone` để đăng ký backbone `CAE` vào registry của mmseg; thiếu dòng này sẽ lỗi `CAE is not in the mmseg::model registry`.
- **Checkpoint fine-tune seg** (Lightning) có key dạng `model.segmentor...`, nên phải bỏ tiền tố `model.` trước khi nạp với `strict=True`.
- **`from_checkpoint` của cls** dựng `panderm_base_patch16_224_finetune` với đúng tham số mặc định của `run_class_finetuning.py`: layer scale `0.1`, relative position bias, mean pooling. Sai một tham số là `strict=True` báo lỗi ngay.
- **`map_pretrained_cls_keys`** lặp lại logic đổi tên key pretrain của upstream (dòng ~455–530): bỏ `encoder.`, `decoder.`, `teacher.`; đổi `norm.` thành `fc_norm.`; nhân bản bảng relative position bias dùng chung cho từng block.

**`tests/test_infer.py`**: chạy được trên CPU với model giả, không cần PanDerm.

```python
# file: tests/test_infer.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nckh.infer import (  # noqa: E402
    ClsPredictor,
    SegPredictor,
    cls_preprocess,
    largest_component,
    map_pretrained_cls_keys,
    seg_logits_to_mask,
    seg_preprocess,
)


def test_seg_preprocess_shape_and_range() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    x = seg_preprocess(rgb)
    assert tuple(x.shape) == (1, 3, 224, 224)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_cls_preprocess_center_crop_and_norm() -> None:
    rgb = np.full((300, 400, 3), 128, dtype=np.uint8)
    x = cls_preprocess(rgb)
    assert tuple(x.shape) == (1, 3, 224, 224)
    # Chuẩn hóa đúng như upstream: std kênh R là 0.228 (không phải 0.229).
    assert x[0, 0, 0, 0].item() == pytest.approx((128 / 255 - 0.485) / 0.228, abs=1e-4)


def test_largest_component_keeps_biggest_and_fills_holes() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[2:12, 2:12] = True
    m[6, 6] = False          # lỗ bên trong
    m[15:17, 15:17] = True   # vùng nhỏ bị bỏ
    out = largest_component(m)
    assert out[6, 6] and not out[15, 15]
    assert out.sum() == 100


def test_largest_component_empty_stays_empty() -> None:
    # Upstream biến mask rỗng thành toàn ảnh; ở đây phải giữ rỗng.
    assert largest_component(np.zeros((8, 8), dtype=bool)).sum() == 0


def test_seg_logits_to_mask_resizes_to_original() -> None:
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, 0:2, 0:2] = 5.0   # vùng lớn 4 px
    logits[0, 1, 3, 3] = 5.0       # vùng nhỏ 1 px
    mask = seg_logits_to_mask(logits, (8, 8))
    assert mask.shape == (8, 8) and mask.dtype == bool
    assert mask[:4, :4].all() and not mask[6:, 6:].any()


class _FakeSeg(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = torch.zeros(x.shape[0], 2, 224, 224)
        out[:, 1, 56:168, 56:168] = 1.0
        return out


class _FakeCls(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.tensor([[2.0, 0.5, -1.0]]).repeat(x.shape[0], 1)


def test_seg_predictor_with_fake_model() -> None:
    mask = SegPredictor(_FakeSeg()).predict(np.zeros((100, 200, 3), dtype=np.uint8))
    assert mask.shape == (100, 200)
    assert mask[50, 100] and not mask[0, 0]
    assert mask.mean() == pytest.approx(0.25, abs=0.02)


def test_cls_predictor_probs_sum_one() -> None:
    probs = ClsPredictor(_FakeCls()).predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert probs.argmax() == 0


def test_map_pretrained_cls_keys() -> None:
    t = torch.zeros(1)
    raw = {
        "encoder.blocks.0.mlp.fc1.weight": t,
        "encoder.norm.weight": t,
        "encoder.rel_pos_bias.relative_position_bias_table": torch.ones(3, 2),
        "encoder.blocks.0.attn.relative_position_index": t,
        "decoder.x": t,
        "teacher.y": t,
    }
    out = map_pretrained_cls_keys(raw, num_layers=2)
    assert "blocks.0.mlp.fc1.weight" in out and "fc_norm.weight" in out
    assert "blocks.1.attn.relative_position_bias_table" in out
    assert not any(k.startswith(("decoder.", "teacher.", "encoder.")) for k in out)
    assert not any("relative_position_index" in k for k in out)
```

### 4.8. Script smoke test + benchmark

**`scripts/bench_inference.py`**

```python
# file: scripts/bench_inference.py
"""Smoke test + benchmark suy luận PanDerm (seg hoặc cls) trên một nhóm ảnh nhỏ.

Chạy bằng Python của venv tương ứng (venv_seg / venv_cls), không phải kernel Colab.
Ví dụ:
  /content/venv_seg/bin/python scripts/bench_inference.py --task seg \
      --panderm-dir /content/PanDerm/segmentation --pretrained $ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth \
      --images "/content/data/ISIC2018/Validation_Data/*.jpg" --n 20 --out-dir $ROOT/runs/<run_id>
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
    ap.add_argument("--images", required=True, help='glob, ví dụ "/content/data/ISIC2018/Validation_Data/*.jpg"')
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
```

**`tests/test_bench.py`**

```python
# file: tests/test_bench.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_inference import run_benchmark  # noqa: E402
from nckh.infer import ClsPredictor, SegPredictor  # noqa: E402


class _FakeSeg(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = torch.zeros(x.shape[0], 2, 224, 224)
        out[:, 1, 80:150, 80:150] = 1.0
        return out


class _FakeCls(torch.nn.Module):
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.tensor([[1.0, 0.0, -1.0]])


def _images(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"ISIC_{i:07d}.jpg"
        Image.fromarray(np.full((60 + i, 80, 3), 120, np.uint8)).save(p)
        paths.append(p)
    return paths


def test_seg_benchmark_writes_report_and_overlays(tmp_path: Path) -> None:
    report = run_benchmark(SegPredictor(_FakeSeg()), "seg", _images(tmp_path, 5), warmup=1,
                           out_dir=tmp_path / "out", save_overlays=2)
    assert report["n"] == 5 and report["ms_per_image_p95"] >= report["ms_per_image_p50"] > 0
    assert len(list((tmp_path / "out" / "overlays").glob("*.png"))) == 2
    saved = json.loads((tmp_path / "out" / "bench_seg.json").read_text())
    assert saved["checks"]["all_shapes_match"] is True


def test_cls_benchmark_checks_probabilities(tmp_path: Path) -> None:
    report = run_benchmark(ClsPredictor(_FakeCls()), "cls", _images(tmp_path, 3), warmup=0,
                           out_dir=tmp_path / "out", save_overlays=0)
    assert report["checks"]["max_abs_prob_sum_error"] < 1e-5
    assert (tmp_path / "out" / "cls_probs_smoke.csv").exists()


def test_benchmark_rejects_empty_image_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_benchmark(SegPredictor(_FakeSeg()), "seg", [], warmup=0, out_dir=tmp_path, save_overlays=0)
```

### 4.9. Chạy smoke test segmentation (trong `NB_seg`)

```python
# cell: NB_seg
# 20 ảnh ISIC 2018 Validation (0,24 GB) chỉ để smoke test; dữ liệu đầy đủ chuẩn bị ở P3.
!mkdir -p /content/smoke && cd /content/smoke && test -d ISIC2018_Task1-2_Validation_Input || (wget -q https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip && unzip -q ISIC2018_Task1-2_Validation_Input.zip && rm ISIC2018_Task1-2_Validation_Input.zip)
from nckh.runcard import new_run_id
RUN = f"{ROOT}/runs/{new_run_id('smoke_seg')}"
!cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/bench_inference.py --task seg --panderm-dir /content/PanDerm/segmentation --pretrained {CK} --images "/content/smoke/ISIC2018_Task1-2_Validation_Input/*.jpg" --n 20 --save-overlays 10 --out-dir {RUN}
!{VENV}/bin/python -m nckh.runcard {RUN} --seed 0 --input pretrained={CK} --input patch={ROOT}/nckh/patches/panderm_base_seg.patch --config task=smoke_seg --config n=20
```

Kỳ vọng trong log:
- `=> Matched …/… backbone keys; ViT coverage 100.0%` (hoặc ≥ 90%);
- cảnh báo `Không có checkpoint fine-tune …` (bình thường ở P2);
- JSON có `ms_per_image_mean`, `peak_vram_mb`;
- dòng cuối `SMOKE TEST OK`.

Mở vài file trong `{RUN}/overlays/` để chắc ảnh đọc đúng màu và đúng chiều.

### 4.10. Dựng `venv_cls` và smoke test classification (trong `NB_cls`)

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2. `ClsPredictor.from_checkpoint` đã được thử trên CPU với code `classification/` thật và một checkpoint pretrain giả có prefix `encoder.`: độ phủ blocks 100%, xác suất cộng lại bằng 1.

```python
# cell: NB_cls
VENV = '/content/venv_cls'
CK = f'{ROOT}/checkpoints/panderm_bb_data6_checkpoint-499.pth'
!test -d /content/PanDerm || git clone -q https://github.com/SiyuanYan1/PanDerm.git /content/PanDerm
!cd /content/PanDerm && git checkout -q fd7a80748ba7fc3e203fed88f909f4689d0d6f24
!pip install -q uv
!uv python install 3.10
!test -x {VENV}/bin/python || uv venv -q --python 3.10 --seed {VENV}
!uv pip install -q --python {VENV}/bin/python torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# timm ghim 0.9.16: modeling_finetune.py dùng timm.models.layers / timm.models.registry.
!uv pip install -q --python {VENV}/bin/python -r /content/PanDerm/classification/requirements.txt timm==0.9.16
!uv pip install -q --python {VENV}/bin/python -e {ROOT}/nckh
!{VENV}/bin/python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')"

!mkdir -p /content/smoke && cd /content/smoke && test -d ISIC2018_Task1-2_Validation_Input || (wget -q https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip && unzip -q ISIC2018_Task1-2_Validation_Input.zip && rm ISIC2018_Task1-2_Validation_Input.zip)
from nckh.runcard import new_run_id
RUN = f"{ROOT}/runs/{new_run_id('smoke_cls')}"
!cd /content && {VENV}/bin/python {ROOT}/nckh/scripts/bench_inference.py --task cls --panderm-dir /content/PanDerm/classification --pretrained {CK} --images "/content/smoke/ISIC2018_Task1-2_Validation_Input/*.jpg" --n 20 --out-dir {RUN}
!{VENV}/bin/python -m nckh.runcard {RUN} --seed 0 --input pretrained={CK} --config task=smoke_cls --config n=20
```

Kỳ vọng:
- log có `Pretrained → …/… key khớp; độ phủ blocks 100.0%`;
- `cls_probs_smoke.csv` có 20 dòng;
- dòng cuối `SMOKE TEST OK`.

Ở bước này đầu phân loại còn ngẫu nhiên, nên xác suất xấp xỉ 1/3 mỗi lớp là bình thường.

### 4.11. Pilot 1 epoch

Pilot train 1 epoch cần dữ liệu ISIC 2018 đúng layout của loader (P3), nên được làm ở **P5a mục 4.2**, cùng với phép thử ngắt runtime rồi resume.

### 4.12. Commit cuối phiên

```python
# cell: NB_cpu
%cd {ROOT}/nckh
!git add pyproject.toml .gitignore src tests scripts patches notebooks
!git status --short
!git commit -q -m "P2: package nckh, infer, smoke test, patch PanDerm Base" && git push -q && git log --oneline -1
```

## 5. Test

| Test | Kiểm tra gì | Chạy ở |
|---|---|---|
| `tests/test_paths_runcard.py` (5) | Đọc `NCKH_ROOT`; SHA-256 đúng giá trị chuẩn của "abc"; định dạng `run_id`; run card đủ 9 khóa; CLI | NB_cpu |
| `tests/test_inspect_checkpoint.py` (3) | Nhận ra ViT-B/ViT-L, checkpoint bị bọc, thiếu prefix | NB_cpu |
| `tests/test_infer.py` (8) | Shape/dải giá trị tiền xử lý; std 0,228; giữ vùng lớn nhất + lấp lỗ; mask rỗng giữ rỗng; resize về kích thước gốc; xác suất tổng 1; đổi tên key pretrain | NB_cpu (cần torch; Colab có sẵn) |
| `tests/test_bench.py` (3) | Báo cáo JSON, overlay, kiểm tra xác suất, từ chối danh sách ảnh rỗng | NB_cpu |

```python
# cell: NB_cpu
!cd {ROOT}/nckh && python -m pytest -q
```

Kỳ vọng: `19 passed`. Nếu kernel không có torch, `test_infer.py` và `test_bench.py` sẽ `skipped`. Đó không phải lỗi.

**Sanity check trên dữ liệu thật** (đã có trong 4.9/4.10): độ phủ trọng số ≥ 90%, `SMOKE TEST OK`, overlay đúng màu và đúng chiều, `cls_probs_smoke.csv` có 20 dòng.

## 6. Benchmark / đánh giá

Điền từ `runs/<run_id>/bench_seg.json` và `bench_cls.json`. **Không điền số ước lượng.**

| Ngày | GPU (`gpu_name`) | Nhánh | n | ms/ảnh mean | ms/ảnh p95 | VRAM đỉnh (MB) | run_id |
|---|---|---|---:|---:|---:|---:|---|
| [điền sau khi chạy] | [điền sau khi chạy] | seg | 20 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |
| [điền sau khi chạy] | [điền sau khi chạy] | cls | 20 | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] | [điền sau khi chạy] |

Số ms/ảnh ở đây tính cho batch 1, bao gồm cả tiền xử lý. Dùng số này để ước lượng thời gian suy luận toàn bộ ảnh UQ ở P6 (số ảnh × ms/ảnh).

## 7. Lỗi thường gặp trên Colab

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `AssertionError: MMCV==2.2.0 is used but incompatible` | Cài `mmcv` không ghim phiên bản (notebook cũ) | Dùng đúng cell 4.4 (`mmcv==2.1.0` + `--find-links … torch2.1.0`) |
| `mim`/`pip` build mmcv rất lâu rồi lỗi CUDA | Không có wheel cho tổ hợp torch/CUDA đó | Như trên; không build mmcv trên Colab |
| `ModuleNotFoundError: pkg_resources` | setuptools ≥ 81 | `uv pip install "setuptools<81"` (đã có trong 4.4) |
| `CAE is not in the mmseg::model registry` | Gọi `CAEv2_seg` mà chưa import `models.cae_backbone` | Dùng `SegPredictor.from_checkpoint` (đã import sẵn) |
| `RuntimeError: Only x% of backbone weights matched` | Checkpoint sai bản, hoặc prefix khác `encoder.` | Chạy lại 4.5 và làm theo bảng |
| `torch.cuda.is_available()` = False | Runtime chưa bật GPU hoặc hết quota GPU | *Runtime → Change runtime type → T4 GPU*; nếu hết quota thì làm phần CPU (P3, P6 với dữ liệu giả) trong lúc chờ |
| Mất venv sau khi ngắt kết nối | `/content` bị xoá khi runtime reset | Chạy lại cell setup; dữ liệu trên Drive vẫn còn |
| `gdown` báo quota | File Drive công khai bị giới hạn lượt tải | Xem cách xử lý ở 4.3 |
| `git push` hỏi mật khẩu | Token hết hạn hoặc chưa bật *Notebook access* | Tạo token mới, cập nhật Colab Secrets |
| `ModuleNotFoundError: nckh` trong kernel | Chưa chạy cell 4.1 (dòng `sys.path`) | Chạy lại cell 4.1 |
| `ModuleNotFoundError: open_clip` khi chạy cls | `classification/models/__init__.py` import `open_clip` | Đã có trong `classification/requirements.txt`; kiểm tra lại cell 4.10 |

## 8. Checklist bàn giao cho SV A

- [ ] `checkpoints/SHA256SUMS` có dòng của `panderm_bb_data6_checkpoint-499.pth` (ngày tải, link nguồn).
- [ ] `python -m pytest -q` trong `NB_cpu`: 19 passed (hoặc skipped vì thiếu torch, ghi rõ).
- [ ] `inspect_checkpoint.py` cho verdict ViT-B; độ phủ ViT ≥ 90% ở cả seg lẫn cls.
- [ ] `runs/<run_id>/bench_seg.json`, `bench_cls.json`, `overlays/`, `run_card.json` có trên Drive; bảng mục 6 đã điền.
- [ ] Nhật ký quyết định có dòng về torch 2.1.2 cho segmentation (mục 3.2) và timm 0.9.16 cho classification.
- [ ] Code đã push lên GitHub; SV A clone về, chạy `pytest` và chạy lại được một smoke test theo hướng dẫn này (kế hoạch Phase 2, điều kiện đạt cuối cùng).
