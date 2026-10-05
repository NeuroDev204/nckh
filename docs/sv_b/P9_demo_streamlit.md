# P9 — Demo Streamlit và kiểm thử end-to-end

## 1. Mục tiêu và đầu vào/đầu ra

**Mục tiêu (kế hoạch Phase 9):** demo cục bộ nhận **một ảnh dermoscopy** và **khoảng thời gian đến lần chụp kế tiếp**, hiển thị: ảnh gốc, mask overlay, 3 điểm nhóm tham khảo, tỷ lệ diện tích mask hiện tại, tỷ lệ dự báo và Δa, cảnh báo khi ảnh/mask không đạt. Demo luôn có thông báo giới hạn cố định.

| Đầu vào | Đầu ra |
|---|---|
| Checkpoint seg (P5a), cls (P5b), `ridge.joblib` (P6), `stable_eps` (P7) | Demo chạy được; ảnh chụp màn hình (chỉ dùng ảnh ISIC); log kiểm thử |

## 2. Chạy ở đâu

- **Khuyến nghị:** chạy trong `NB_seg` (runtime GPU). Dựng thêm `venv_cls` trong cùng runtime, vì demo cần cả hai venv.
- **Thử giao diện trước khi có model:** `NB_cpu` với chế độ `--fake` (mục 4.4). Không cần GPU.
- **Laptop:** chỉ khi máy dựng được cả hai venv (Linux, có Python 3.10). Nếu không, dùng Colab.

> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2. Khi viết docs đã kiểm tra: toàn bộ logic `demo/pipeline.py` (pytest, gồm chạy thật `infer_images.py --fake` qua tiến trình con); `demo/app.py` chạy không lỗi bằng `streamlit.testing.AppTest`; server Streamlit trả HTTP 200.

## 3. Giải thích

### 3.1. Kiến trúc

```
Trình duyệt ──(URL tunnel)──► Streamlit (Python của kernel Colab, demo/app.py)
                                   │  cache theo SHA-256 của ảnh
                                   ├─► /content/venv_seg/bin/python scripts/infer_images.py --task seg …  → masks/<id>.png
                                   ├─► /content/venv_cls/bin/python scripts/infer_images.py --task cls …  → cls_probs.csv
                                   └─► ridge.joblib + nckh.forecast  → a_t, Δâ, â_{t+1}, xu hướng
```

- **Vì sao gọi qua tiến trình con:** segmentation cần torch 2.1 + mmcv, classification cần torch 2.4 + timm. Hai bộ này không nạp chung một process được.
- **Cái giá:** mỗi ảnh mới phải nạp lại model (vài giây trên GPU). Vì vậy kết quả được cache theo SHA-256 của ảnh: đổi Δt với cùng ảnh thì chỉ chạy lại Ridge, gần như tức thì.

### 3.2. Xử lý lỗi (kế hoạch Phase 9, mục kiểm thử)

| Tình huống | Demo làm gì |
|---|---|
| File không phải ảnh / ảnh hỏng | Báo "Không đọc được ảnh", **không** chạy model |
| Δt rỗng, âm, 0, "6 tháng", "nan", "inf" | Báo "Δt phải là số ngày dương, ví dụ 180", **không** dự báo |
| Mask rỗng | Báo "Không tìm thấy vùng tổn thương…", **không** đưa ra số dự báo |
| Tỷ lệ mask < 0,1% hoặc > 90% | Vẫn hiển thị nhưng cảnh báo chất lượng mask |
| Lệnh suy luận lỗi | Hiện thông báo kèm 2.000 ký tự cuối của stderr, không crash âm thầm |
| Kết quả dự báo | Luôn được clip về [0, 1] |

**Xu hướng mask:** |Δâ| ≤ `stable_eps` (giá trị đã chốt ở P7) là "ổn định"; lớn hơn thì "tăng" hoặc "giảm". Đây là xu hướng của **mask trên ảnh**, không phải của bệnh.

### 3.3. An toàn dữ liệu

URL `trycloudflare.com` là **công khai**: ai có link đều mở được. Vì vậy:
- chỉ upload ảnh ISIC (giấy phép cho phép), **tuyệt đối không** dùng ảnh UQ;
- tắt tunnel ngay khi xong (dừng runtime hoặc `pkill cloudflared`);
- không chia sẻ link công khai; ảnh chụp màn hình chỉ chứa ảnh ISIC.

## 4. Code

### 4.1. `demo/pipeline.py`

```python
# file: demo/pipeline.py
"""Logic của demo, tách khỏi giao diện Streamlit để kiểm thử được.

Hai nhánh PanDerm cần hai phiên bản torch khác nhau nên không nạp chung một process: demo gọi
scripts/infer_images.py bằng python của venv_seg và venv_cls như hai tiến trình con.
"""
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nckh.features import FEATURE_COLUMNS, mask_features
from nckh.forecast import predict_delta, predict_next_area, validate_delta_days

DISCLAIMER = ("Đây là demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu; "
              "dự báo là thay đổi tỷ lệ diện tích mask trên ảnh, không phải chẩn đoán, tiên lượng bệnh "
              "hoặc khuyến nghị điều trị.")
MIN_AREA, MAX_AREA = 0.001, 0.9


def _run(cmd: list[str], image_path: Path, work_dir: Path, timeout_s: int) -> None:
    full = [*cmd, "--images", str(image_path), "--out-dir", str(work_dir)]
    proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Lệnh suy luận lỗi ({proc.returncode}): {proc.stderr[-2000:]}")


def run_inference(image_path: Path, work_dir: Path, seg_cmd: list[str], cls_cmd: list[str],
                  timeout_s: int = 600) -> tuple[np.ndarray, dict[str, float]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    _run(seg_cmd, image_path, work_dir, timeout_s)
    failed = pd.read_csv(work_dir / "seg_failed.csv")
    if len(failed):
        raise ValueError(f"Không đọc được ảnh: {failed['error'].iloc[0]}")
    _run(cls_cmd, image_path, work_dir, timeout_s)
    mask = np.asarray(Image.open(work_dir / "masks" / f"{image_path.stem}.png")) > 127
    probs = pd.read_csv(work_dir / "cls_probs.csv").iloc[0]
    return mask, {k: float(probs[k]) for k in ("p_mel", "p_nev", "p_sk")}


def forecast_from_outputs(mask: np.ndarray, probs: dict[str, float], delta_days_raw: object, ridge_bundle: dict,
                          stable_eps: float) -> dict:
    days = validate_delta_days(delta_days_raw)  # kiểm tra đầu vào trước mọi thứ khác
    feats = mask_features(mask)
    out = {"area_ratio_t": feats["area_ratio"], "delta_area_pred": None, "next_area_pred": None, "direction": None,
           "warnings": []}
    if feats["empty"]:
        out["warnings"].append("Không tìm thấy vùng tổn thương trong ảnh; không đưa ra dự báo.")
        return out
    if not MIN_AREA <= feats["area_ratio"] <= MAX_AREA:
        out["warnings"].append(f"Tỷ lệ mask {feats['area_ratio']:.4f} nằm ngoài [{MIN_AREA}, {MAX_AREA}]: "
                               "chất lượng mask có thể thấp, hãy xem lại ảnh overlay.")
    z = pd.DataFrame([{"area_ratio_t": feats["area_ratio"], "circularity_t": feats["circularity"],
                       "eccentricity_t": feats["eccentricity"], "p_mel_t": probs["p_mel"], "p_nev_t": probs["p_nev"],
                       "p_sk_t": probs["p_sk"], "delta_days": days}])[list(FEATURE_COLUMNS)]
    if not all(math.isfinite(v) for v in z.iloc[0]):
        out["warnings"].append("Không tính được đặc trưng hình dạng của mask; không đưa ra dự báo.")
        return out
    delta = float(predict_delta(ridge_bundle["model"], z)[0])
    out["delta_area_pred"] = delta
    out["next_area_pred"] = float(predict_next_area(np.array([feats["area_ratio"]]), np.array([delta]))[0])
    out["direction"] = "ổn định" if abs(delta) <= stable_eps else ("tăng" if delta > 0 else "giảm")
    return out
```

### 4.2. `demo/app.py`

```python
# file: demo/app.py
"""Demo nghiên cứu: ảnh dermoscopy + Δt → mask, xác suất 3 nhóm tham khảo, tỷ lệ mask dự báo lần sau.

Chạy: NCKH_SEG_CMD=... NCKH_CLS_CMD=... NCKH_RIDGE=.../ridge.joblib NCKH_STABLE_EPS=0.005 streamlit run demo/app.py
"""
import hashlib
import os
import shlex
import tempfile
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.segmentation import mark_boundaries

from pipeline import DISCLAIMER, forecast_from_outputs, run_inference

LABELS = {"p_mel": "Melanoma", "p_nev": "Nevus", "p_sk": "Seborrheic keratosis"}


@st.cache_data(show_spinner="Đang chạy PanDerm (lần đầu có thể mất 1–2 phút)…")
def infer_cached(digest: str, data: bytes, suffix: str) -> tuple[np.ndarray, dict[str, float]]:
    # Khóa cache là SHA-256 của ảnh: tải lại cùng ảnh, chỉ đổi Δt, thì không chạy lại PanDerm.
    work = Path(tempfile.gettempdir()) / "nckh_demo" / digest
    work.mkdir(parents=True, exist_ok=True)
    image_path = work / f"{digest}{suffix}"
    image_path.write_bytes(data)
    return run_inference(image_path, work, shlex.split(os.environ["NCKH_SEG_CMD"]), shlex.split(os.environ["NCKH_CLS_CMD"]))


@st.cache_resource
def load_ridge(path: str) -> dict:
    return joblib.load(path)


st.set_page_config(page_title="PanDerm demo nghiên cứu", layout="wide")
st.warning(DISCLAIMER)
st.title("Phân đoạn, nhóm tham khảo và dự báo tỷ lệ mask ở lần chụp kế tiếp")

uploaded = st.file_uploader("Ảnh dermoscopy (JPG/PNG)", type=["jpg", "jpeg", "png"])
delta_raw = st.text_input("Khoảng thời gian đến lần chụp kế tiếp (số ngày)", value="180")

if uploaded is not None:
    data = uploaded.getvalue()
    try:
        rgb = np.asarray(Image.open(uploaded).convert("RGB"))
    except OSError:
        st.error("Không đọc được ảnh. Hãy tải lên một file JPG/PNG hợp lệ.")
        st.stop()
    try:
        mask, probs = infer_cached(hashlib.sha256(data).hexdigest(), data, Path(uploaded.name).suffix.lower())
        result = forecast_from_outputs(mask, probs, delta_raw, load_ridge(os.environ["NCKH_RIDGE"]),
                                       float(os.environ.get("NCKH_STABLE_EPS", "0.005")))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error("Lỗi khi chạy mô hình. Chi tiết kỹ thuật bên dưới.")
        st.code(str(exc))
        st.stop()

    left, right = st.columns(2)
    left.image(rgb, caption="Ảnh gốc", width="stretch")
    right.image(mark_boundaries(rgb, mask.astype(int), color=(0, 1, 1)), caption="Mask PanDerm (viền xanh)",
                width="stretch", clamp=True)

    st.subheader("Điểm nhóm tham khảo (theo nhãn ISIC 2017, không phải chẩn đoán)")
    st.table({LABELS[k]: [f"{v:.1%}"] for k, v in probs.items()})

    st.subheader("Tỷ lệ diện tích mask trên ảnh")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hiện tại (a_t)", f"{result['area_ratio_t']:.2%}")
    if result["next_area_pred"] is not None:
        c2.metric("Dự báo lần sau", f"{result['next_area_pred']:.2%}", f"{result['delta_area_pred'] * 100:+.2f} điểm %")
        c3.metric("Xu hướng mask", result["direction"])
    for w in result["warnings"]:
        st.warning(w)

st.caption(DISCLAIMER)
```

### 4.3. Test

```python
# file: tests/test_demo_pipeline.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "demo"))

from pipeline import DISCLAIMER, forecast_from_outputs, run_inference  # noqa: E402

from nckh.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from nckh.forecast import fit_forecaster  # noqa: E402


@pytest.fixture(scope="module")
def bundle() -> dict:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.uniform(0, 1, 60) for c in FEATURE_COLUMNS})
    df["delta_days"] = rng.uniform(150, 210, 60)
    df[TARGET_COLUMN] = 0.0002 * (df["delta_days"] - 150)
    return {"model": fit_forecaster(df, 1.0), "alpha": 1.0, "feature_columns": list(FEATURE_COLUMNS)}


def _disk(r: int = 20) -> np.ndarray:
    yy, xx = np.mgrid[:100, :100]
    return (yy - 50) ** 2 + (xx - 50) ** 2 <= r ** 2


PROBS = {"p_mel": 0.2, "p_nev": 0.7, "p_sk": 0.1}


def test_forecast_valid(bundle: dict) -> None:
    out = forecast_from_outputs(_disk(), PROBS, "180", bundle, stable_eps=0.005)
    assert 0 <= out["next_area_pred"] <= 1
    assert out["area_ratio_t"] == pytest.approx(_disk().mean())
    assert out["direction"] in {"tăng", "giảm", "ổn định"} and out["warnings"] == []


def test_forecast_empty_mask_no_prediction(bundle: dict) -> None:
    out = forecast_from_outputs(np.zeros((50, 50), bool), PROBS, 180, bundle, stable_eps=0.005)
    assert out["delta_area_pred"] is None and out["next_area_pred"] is None and out["direction"] is None
    assert "Không tìm thấy vùng tổn thương" in out["warnings"][0]


@pytest.mark.parametrize("value", [-30, 0, "", "6 tháng", "nan", "inf", None, True])
def test_forecast_rejects_bad_delta(bundle: dict, value: object) -> None:
    with pytest.raises(ValueError, match="Δt"):
        forecast_from_outputs(_disk(), PROBS, value, bundle, stable_eps=0.005)


def test_forecast_tiny_mask_warns(bundle: dict) -> None:
    m = np.zeros((200, 200), bool)
    m[100:102, 100:103] = True
    out = forecast_from_outputs(m, PROBS, 180, bundle, stable_eps=0.005)
    assert any("chất lượng mask" in w for w in out["warnings"])


def test_run_inference_raises_on_failure(tmp_path: Path) -> None:
    fail = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"]
    with pytest.raises(RuntimeError, match="boom"):
        run_inference(tmp_path / "x.jpg", tmp_path / "w", fail, fail)


def test_run_inference_reads_outputs_with_fake_models(tmp_path: Path) -> None:
    img = np.full((80, 120, 3), 210, np.uint8)
    img[20:60, 30:90] = 40
    Image.fromarray(img).save(tmp_path / "lesion.jpg")
    script = str(REPO / "scripts" / "infer_images.py")
    mask, probs = run_inference(tmp_path / "lesion.jpg", tmp_path / "w",
                                [sys.executable, script, "--task", "seg", "--fake"],
                                [sys.executable, script, "--task", "cls", "--fake"])
    assert mask.shape == (80, 120) and mask[40, 60] and not mask[0, 0]
    assert sum(probs.values()) == pytest.approx(1.0)


def test_run_inference_unreadable_image(tmp_path: Path) -> None:
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    script = str(REPO / "scripts" / "infer_images.py")
    with pytest.raises(ValueError, match="Không đọc được ảnh"):
        run_inference(tmp_path / "bad.jpg", tmp_path / "w", [sys.executable, script, "--task", "seg", "--fake"],
                      [sys.executable, script, "--task", "cls", "--fake"])


def test_disclaimer_exact() -> None:
    assert DISCLAIMER == ("Đây là demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu; "
                          "dự báo là thay đổi tỷ lệ diện tích mask trên ảnh, không phải chẩn đoán, tiên lượng bệnh "
                          "hoặc khuyến nghị điều trị.")
```

### 4.4. Thử giao diện bằng chế độ giả (NB_cpu, không cần model)

Cần một `ridge.joblib`: dùng file từ lần chạy dữ liệu giả ở P6 mục 4.6.

```python
# cell: NB_cpu
import os, subprocess, time
os.environ['NCKH_SEG_CMD'] = f"python {ROOT}/nckh/scripts/infer_images.py --task seg --fake"
os.environ['NCKH_CLS_CMD'] = f"python {ROOT}/nckh/scripts/infer_images.py --task cls --fake"
os.environ['NCKH_RIDGE'] = f"{ROOT}/runs/<p6_fake_run_id>/ridge.joblib"
os.environ['NCKH_STABLE_EPS'] = "0.005"
!pip install -q -e "{ROOT}/nckh[demo]"
subprocess.Popen(f"streamlit run {ROOT}/nckh/demo/app.py --server.port 8501 --server.headless true > /content/streamlit.log 2>&1", shell=True)
time.sleep(8)
!curl -s localhost:8501/_stcore/health && echo " ← Streamlit OK"
```

Rồi mở tunnel (mục 4.6). Kết quả trong chế độ giả là vô nghĩa về mặt nghiên cứu; bước này chỉ để kiểm tra giao diện và luồng xử lý lỗi.

### 4.5. Chạy với model thật (NB_seg, GPU)

Chuẩn bị trong cùng runtime:
1. (NB_cpu) cell 4.1a của P2 để pull code, rồi (NB_seg) cell mở đầu 4.1b;
2. `venv_seg` (P2 4.4) và patch (P2 4.6);
3. **cả** phần cài `venv_cls` (P2 4.10, chỉ các dòng cài đặt).

```python
# cell: NB_seg
import os, subprocess, time
SEG_FT = f"{ROOT}/runs/<seg_main_run_id>/0/model_best_0.ckpt"
CLS_FT = f"{ROOT}/runs/<cls_main_run_id>/checkpoint-best.pth"
SCRIPT = f"{ROOT}/nckh/scripts/infer_images.py"
os.environ['NCKH_SEG_CMD'] = (f"/content/venv_seg/bin/python {SCRIPT} --task seg --panderm-dir /content/PanDerm/segmentation "
                              f"--pretrained {CK} --finetuned {SEG_FT}")
os.environ['NCKH_CLS_CMD'] = (f"/content/venv_cls/bin/python {SCRIPT} --task cls --panderm-dir /content/PanDerm/classification "
                              f"--finetuned {CLS_FT}")
os.environ['NCKH_RIDGE'] = f"{ROOT}/runs/<p6_uq_run_id>/ridge/ridge.joblib"
os.environ['NCKH_STABLE_EPS'] = "<giá trị đã chốt ở P7>"
!pip install -q -e "{ROOT}/nckh[demo]"
subprocess.Popen(f"streamlit run {ROOT}/nckh/demo/app.py --server.port 8501 --server.headless true > /content/streamlit.log 2>&1", shell=True)
time.sleep(8)
!curl -s localhost:8501/_stcore/health && echo " ← Streamlit OK"
```

Nếu chưa có `ridge.joblib` từ UQ thật (chưa qua Go/No-Go), demo vẫn chạy được phần mask + xác suất với ridge của dữ liệu giả. Khi đó **phải** ghi rõ trên ảnh chụp màn hình rằng con số dự báo chỉ là minh họa từ dữ liệu giả.

### 4.6. Mở tunnel để xem từ trình duyệt

```python
# cell: NB_seg
!test -x /content/cloudflared || (wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared)
subprocess.Popen("/content/cloudflared tunnel --url http://localhost:8501 > /content/cloudflared.log 2>&1", shell=True)
time.sleep(10)
!grep -o "https://[a-z0-9-]*\.trycloudflare\.com" /content/cloudflared.log | head -1
```

Mở link in ra. Khi xong: `!pkill cloudflared; pkill -f "streamlit run"`.

## 5. Test

```python
# cell: NB_cpu
!cd {ROOT}/nckh && python -m pytest -q tests/test_demo_pipeline.py
```

Kỳ vọng: `15 passed`.

### 5.1. Kiểm thử end-to-end (kế hoạch Phase 9, 10 bước)

| # | Yêu cầu | Cách kiểm | Kết quả lưu |
|---|---|---|---|
| 1 | Dựng môi trường sạch theo hướng dẫn | Runtime mới → chạy lần lượt mục 4.5 | Ghi thời gian dựng |
| 2 | Nạp checkpoint seg, cls, Ridge, cấu hình | Lần upload đầu không lỗi; log `/content/streamlit.log` sạch | Ảnh chụp |
| 3 | Một ảnh hợp lệ chạy từ đầu đến cuối | Upload 1 ảnh ISIC 2018 **val** | Ảnh chụp toàn trang |
| 4 | Overlay đúng vị trí, kích thước, màu | So overlay với ảnh gốc; mở thêm mask ở `/tmp/nckh_demo/<sha>/masks/` | Ghi nhận |
| 5 | Xác suất hữu hạn, tổng đúng quy ước | Cộng 3 xác suất trên bảng ≈ 100%; `test_run_inference_reads_outputs_with_fake_models` | Ảnh chụp |
| 6 | Diện tích dự báo trong [0, 1] | `test_forecast_valid`; xem trên giao diện | — |
| 7 | Δt âm, 0, thiếu, không phải số bị từ chối | Nhập lần lượt `-30`, `0`, (rỗng), `6 tháng`; `test_forecast_rejects_bad_delta` | Ảnh chụp thông báo |
| 8 | Ảnh hỏng, mask rỗng có thông báo | Upload file `.jpg` rỗng/hỏng; ảnh da không có tổn thương; `test_run_inference_unreadable_image`, `test_forecast_empty_mask_no_prediction` | Ảnh chụp |
| 9 | Cùng ảnh + checkpoint + Δt cho cùng đầu ra | Upload lại cùng ảnh → số giống hệt (cache), restart server rồi thử lại → số giống trong sai số số học | Ghi nhận |
| 10 | Chỉ dùng dữ liệu công khai được phép | Chỉ ảnh ISIC; tunnel tắt sau khi dùng | Ghi vào checklist |

## 6. Benchmark / đánh giá

| Hạng mục | Giá trị |
|---|---|
| GPU | [điền sau khi chạy] |
| Thời gian từ upload đến kết quả, ảnh mới (lần đầu) | [điền sau khi chạy] |
| Như trên, cùng ảnh đổi Δt (cache) | [điền sau khi chạy] |
| Thời gian dựng môi trường từ runtime mới | [điền sau khi chạy] |

## 7. Lỗi thường gặp trên Colab

| Triệu chứng | Cách xử lý |
|---|---|
| `KeyError: 'NCKH_SEG_CMD'` | Chưa đặt biến môi trường **trước** khi `Popen` streamlit |
| `ModuleNotFoundError: pipeline` | Chạy `streamlit run` với đường dẫn tới `demo/app.py` (Streamlit tự thêm thư mục `demo/` vào `sys.path`) |
| Báo lỗi mô hình, stderr có `CUDA out of memory` | Đang có process khác giữ GPU (notebook train); restart runtime |
| Link tunnel không mở | Đọc `/content/cloudflared.log`; chờ thêm 10 giây; chạy lại cell |
| Mỗi ảnh chậm 30–60 giây | Đang chạy trên CPU; dùng runtime GPU |

## 8. Checklist bàn giao cho SV A

- [ ] SV A duyệt câu chữ của thông báo giới hạn và nhãn hiển thị (không có từ "chẩn đoán").
- [ ] `pytest` phần demo: 15 passed.
- [ ] Bảng 10 bước mục 5.1 đã đánh dấu, có ảnh chụp kèm theo (chỉ ảnh ISIC).
- [ ] Bảng thời gian mục 6 đã điền.
- [ ] Ghi rõ ridge dùng trong demo là từ UQ thật hay dữ liệu giả.
