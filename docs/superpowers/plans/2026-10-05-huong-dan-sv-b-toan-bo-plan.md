# Bộ hướng dẫn SV B toàn phase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Viết 11 file hướng dẫn `docs/sv_b/*.md` cho SV B. Mỗi file có giải thích, code đầy đủ, test, benchmark và checklist bàn giao. SV B tự tạo code trong repo bằng cách làm theo các file này.

**Architecture:** Code trong docs được viết và chạy pytest trước trong một sandbox nằm ngoài repo (`$SANDBOX`). Docs chép nguyên văn code từ sandbox, mỗi khối code mở đầu bằng dòng `# file: <đường dẫn>`. Script `check_docs.py` (cũng nằm trong sandbox) đối chiếu từng khối code trong docs với file trong sandbox. Repo chỉ nhận các file `.md`.

**Tech Stack:** Markdown; Python ≥3.10 với numpy, pandas, scikit-learn, scikit-image, pillow, joblib, pyyaml, pytest, streamlit, matplotlib, openpyxl; torch (chỉ cho `infer.py`); PanDerm upstream commit `fd7a807`; Google Colab + Drive + GitHub.

**Spec:** `docs/superpowers/specs/2026-10-05-huong-dan-sv-b-toan-bo-design.md` (đọc cả mục 7b).

## Global Constraints

- **Chỉ ghi vào repo:** `docs/sv_b/*.md` và một dòng ghi chú ở đầu 3 file `docs/Huong_dan_SV_B_Tuan_1_*.md`. Không tạo hoặc sửa bất kỳ file code, notebook hay config nào trong repo.
- **Cách commit:** `git add -f docs/sv_b/<file>.md` (vì `.gitignore` đang chặn `/docs/`). Không add `nckh.ipynb`, `.idea/`, `requirements_colab.txt`; đây là thay đổi dở của người dùng.
- **Ngôn ngữ:** docs viết bằng tiếng Việt có dấu. Comment trong code bằng tiếng Việt và giải thích **vì sao**, không mô tả lại code làm gì.
- **Type hints** bắt buộc cho mọi hàm trong code của docs. Không có `except: pass`; mọi lỗi phải được log hoặc raise kèm thông báo rõ.
- **Đường dẫn:** lấy từ biến môi trường `NCKH_ROOT`, mặc định `/content/drive/MyDrive/NCKH_PanDerm`. Manifest lưu đường dẫn tương đối so với `data_root`.
- **Checkpoint PanDerm_Base:** Google Drive ID `17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB`, tên file `panderm_bb_data6_checkpoint-499.pth`.
- **Ba runtime:**
  - `NB_seg`: venv 3.10; torch 2.2.1/torchvision 0.17.1 cu118; mmengine 0.10.4; mmcv 2.1.0; mmsegmentation 1.2.2.
  - `NB_cls`: venv 3.10; torch 2.4.1/torchvision 0.19.1/torchaudio 2.4.1 cu118.
  - `NB_cpu`: Python mặc định của Colab.
- **Package `nckh`:** phụ thuộc lõi gồm `numpy, pandas, scikit-learn, scikit-image, pillow, joblib, pyyaml`; extras `demo = [streamlit, matplotlib]`, `test = [pytest, openpyxl]`. Không phụ thuộc torch.
- **Hằng số khóa:**
  - split UQ 70/15/15 theo `participant_id`, seed 2026;
  - bootstrap 2000 lần, seed 2026;
  - `ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)`;
  - nhãn mel=0, nev=1, sk=2.
- **Mức suy giảm:**
  - brightness α ∈ {0.8, 1.2}; blur σ ∈ {1.0, 1.5};
  - gain (R, B) ∈ {(1.1, 0.9), (0.9, 1.1), (1.2, 0.8), (0.8, 1.2)};
  - lông che {0.01, 0.03} với sai số ≤ 0.005;
  - gamma γ ∈ {0.8, 1.2}.
- **Phần GPU** (patch chạy thật, fine-tune, benchmark, `infer.py` với PanDerm): luôn có hộp `> ⚠️ Chưa kiểm chứng trên GPU — xác minh trong pilot P2`.
- **Disclaimer demo** lấy nguyên văn từ kế hoạch: "Đây là demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu; dự báo là thay đổi tỷ lệ diện tích mask trên ảnh, không phải chẩn đoán, tiên lượng bệnh hoặc khuyến nghị điều trị."
- **Không điền số kết quả giả** vào bất kỳ bảng nào; dùng `[điền sau khi chạy]`.

## Review Focus

1. **Drive hết dung lượng:** tổng zip ISIC khoảng 27 GB, vượt 15 GB. Docs P3 không được bảo lưu zip ảnh lên Drive; script tải phải idempotent vào `/content`. Test (Task 4): `extract_all` chạy lần hai thì bỏ qua và không lỗi; manifest chỉ chứa đường dẫn tương đối.
2. **Timestamp UQ "bẩn":** chỉ có ngày, có múi giờ, hai ảnh cùng ngày, hoặc thiếu ngày. Người dùng mong cặp Δt=0 bị loại với lý do `same_timestamp`, còn thiếu ngày thì bị loại với lý do `missing_timestamp`, không crash. Test thuộc Task 7 (`test_pairs`).
3. **Mask rỗng hoặc tổn thương rất nhỏ (1–2 pixel):** `mask_features` không được trả NaN âm thầm. Mask rỗng phải có `empty=True` và các đặc trưng hình dạng = NaN; cặp bị loại khỏi Ridge với lý do `empty_mask`. Test thuộc Task 7 (`test_features`).
4. **Δt nhập sai trong demo** ("6 tháng", "-30", "", "nan", "inf"): phải hiện lỗi dễ hiểu và không chạy dự báo. Test thuộc Task 7 (`test_forecast::test_validate_delta_days_rejects`) và Task 10 (`test_forecast_rejects_bad_delta`).
5. **Colab ngắt giữa lúc train:** người dùng mong train tiếp từ checkpoint trên Drive. Không test tự động được; docs P5a/P5b có bước "ngắt runtime cố ý rồi resume" trong pilot, và checklist yêu cầu log bằng chứng resume.

---

## Quy ước chung cho mọi task

**Biến môi trường** (đặt ở đầu mọi lệnh):

```bash
export S=/tmp/claude-1000/-home-neuro-Documents-nckh/53bd151a-08ce-48ab-b3ae-bfc09d196ad5/scratchpad
export SANDBOX=$S/sandbox          # mirror cấu trúc repo mà SV B sẽ tạo
export PANDERM=$S/PanDerm           # clone upstream fd7a807 (đã có)
export REPO=/home/neuro/Documents/nckh
```

**Khung bắt buộc cho mỗi file phase** (các heading cấp 2, đúng thứ tự):
`## 1. Mục tiêu và đầu vào/đầu ra` · `## 2. Chạy ở đâu` · `## 3. Giải thích` · `## 4. Code` · `## 5. Test` · `## 6. Benchmark / đánh giá` · `## 7. Lỗi thường gặp trên Colab` · `## 8. Checklist bàn giao cho SV A`.

**Quy ước khối code trong docs:**
- Khối tạo file: dòng đầu là `# file: src/nckh/manifest.py` (hoặc đường dẫn tương ứng), và toàn bộ nội dung phải trùng khớp với `$SANDBOX/<đường dẫn>`. File trong sandbox cũng bắt đầu bằng đúng dòng đó.
- Khối là cell Colab: dòng đầu là `# cell: NB_cpu`, `# cell: NB_seg` hoặc `# cell: NB_cls`, không đối chiếu với sandbox.
- Khối patch dùng ` ```diff ` và không có dòng `# file:` (dòng đó sẽ làm hỏng patch). Đối chiếu bằng `diff` tay ở Task 3 và Task 5.

**Vòng làm việc của mỗi task có code:**
1. Viết test trong sandbox.
2. Chạy để thấy test fail.
3. Viết code.
4. Chạy để thấy test pass.
5. Viết file `.md`.
6. Chạy `check_docs.py`.
7. Commit file `.md`.

---

### Task 1: Dựng sandbox kiểm chứng và `check_docs.py`

**Files:**
- Create (ngoài repo): `$S/check_docs.py`, `$SANDBOX/` (thư mục rỗng), `$S/venv/`

**Interfaces:**
- Produces: lệnh `$S/venv/bin/python $S/check_docs.py <file.md>...`. Exit 0 và in `OK` nếu mọi khối `# file:` trùng khớp với `$SANDBOX`; exit 1 và in diff nếu lệch hoặc thiếu file.

- [ ] **Step 1: Tạo venv và cài deps**

```bash
python3 -m venv $S/venv
$S/venv/bin/pip install -q numpy pandas scikit-learn scikit-image pillow joblib pyyaml pytest streamlit matplotlib openpyxl
$S/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p $SANDBOX
$S/venv/bin/python -c "import torch, sklearn, skimage, openpyxl; print('deps ok')"
```

Expected: `deps ok`.

- [ ] **Step 2: Viết `check_docs.py`**

```python
# $S/check_docs.py
"""Đối chiếu khối code '# file: <path>' trong docs với file đã test trong sandbox."""
import difflib
import os
import re
import sys
from pathlib import Path

SANDBOX = Path(os.environ["SANDBOX"])
FENCE = re.compile(r"```[a-z]*\n(# file: (\S+)\n.*?)```", re.S)


def check(md_path: Path) -> int:
    errors = 0
    for block, rel in FENCE.findall(md_path.read_text(encoding="utf-8")):
        target = SANDBOX / rel
        if not target.exists():
            print(f"MISSING in sandbox: {rel}")
            errors += 1
            continue
        if block != target.read_text(encoding="utf-8"):
            errors += 1
            print(f"DIFF: {rel}")
            sys.stdout.writelines(difflib.unified_diff(
                target.read_text(encoding="utf-8").splitlines(True), block.splitlines(True),
                "sandbox", "docs"))
    return errors


if __name__ == "__main__":
    total = sum(check(Path(p)) for p in sys.argv[1:])
    print("OK" if total == 0 else f"{total} lỗi")
    sys.exit(1 if total else 0)
```

- [ ] **Step 3: Tự kiểm `check_docs.py`**

```bash
printf '# file: x.py\nprint(1)\n' > $SANDBOX/x.py
printf '```python\n# file: x.py\nprint(1)\n```\n' > $S/t.md
SANDBOX=$SANDBOX $S/venv/bin/python $S/check_docs.py $S/t.md          # Expected: OK, exit 0
sed -i 's/print(1)/print(2)/' $S/t.md
SANDBOX=$SANDBOX $S/venv/bin/python $S/check_docs.py $S/t.md; echo $?  # Expected: DIFF: x.py ... 1
rm $SANDBOX/x.py $S/t.md
```

---

### Task 2: `00_tong_quan_va_lo_trinh.md` + `P0_P1_P4_P11_vai_tro_ho_tro.md` (phần P0/P1/P11)

**Files:**
- Create: `docs/sv_b/00_tong_quan_va_lo_trinh.md`
- Create: `docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md` (mục P4 tạm giữ chỗ, hoàn thiện ở Task 6)

**Interfaces:**
- Produces: tên 11 file docs và thứ tự đọc. Các task sau link tới mục "Bố cục Drive" của file 00 và mục "Cell setup" của P2.

- [ ] **Step 1: Viết `00_tong_quan_va_lo_trinh.md`**

Nội dung bắt buộc (mỗi mục là một heading `##`):

1. **Vai trò SV B:** tóm tắt bảng Mục 2 của kế hoạch, nêu rõ SV B làm cả classification (ghi chú: đề cương mục 7 dòng 4 cần SV A sửa).
2. **Bảng lộ trình 14 tuần:** cột Tuần · Phase · Việc SV B · File docs · Runtime · Mốc bàn giao, lấy từ Mục 3 của kế hoạch.
3. **Sơ đồ dòng dữ liệu (mermaid):**
   - ISIC 2018 → P5a seg checkpoint;
   - ISIC 2017 → P5b cls checkpoint;
   - UQ → manifest → split participant → pairs → `infer_images` (seg + cls) → features → Ridge vs baseline → evaluate;
   - nhánh P8 degrade → infer → so sánh;
   - P9 demo.
4. **Bố cục Drive:** cây thư mục theo spec 3.1, kèm bảng dung lượng ISIC (số liệu trong spec 7b) và quy tắc "ảnh tải về `/content` mỗi phiên".
5. **Đồng bộ Drive ↔ GitHub:** cell đồng bộ dùng `userdata.get('GH_TOKEN')`, hướng dẫn tạo fine-grained token chỉ cho repo `NeuroDev204/nckh` với quyền Contents: Read and write, cách thêm token vào Colab Secrets, và 3 quy tắc tránh xung đột.
6. **Ba runtime:** bảng theo spec 3.3.
7. **Nguyên tắc dữ liệu và đạo đức:**
   - UQ chỉ được đưa lên Drive khi có văn bản cho phép;
   - không commit ảnh, checkpoint hoặc token;
   - test chỉ mở một lần;
   - không gọi output là chẩn đoán.
8. **Bản đồ file code:** bảng "file → phase tạo ra nó" theo spec 3.4 cộng mục 7b.
9. **Thứ tự đọc docs:** 00 → P2 → P3 → P5a → P5b → P6 → P7 → P8 → P9 → P10, và đọc P0_P1_P4_P11 song song.

- [ ] **Step 2: Viết phần P0, P1, P11 của `P0_P1_P4_P11_vai_tro_ho_tro.md`**

Mỗi phase gồm: SV A chủ trì việc gì, SV B làm gì, sản phẩm SV B nộp, checklist.

- **P0:** SV B phản biện protocol bằng checklist kỹ thuật:
  - đơn vị phân tích;
  - target Δa;
  - baseline;
  - metric chính MAE(Δa);
  - lưới alpha;
  - seed;
  - `--monitor` của cls;
  - TTA có/không;
  - quy tắc test chỉ mở một lần.
- **P1:** SV B kiểm tra cấu trúc và dung lượng gói UQ khi có quyền:
  - cell đếm file, tổng dung lượng, đuôi file;
  - đọc metadata bằng pandas (`df.dtypes`, `df.isna().mean()`, `df.nunique()`);
  - đề xuất bản đồ cột sang schema `participant_id, lesion_id, image_id, image_path, captured_at, modality`.

  Cell này chỉ chạy sau khi có quyền.
- **P11:** SV B viết phụ lục kỹ thuật. Dàn ý gồm 6 mục, liệt kê sẵn trong docs:
  - môi trường;
  - phiên bản;
  - checkpoint và hash;
  - run card;
  - quy trình tái lập;
  - giới hạn kỹ thuật.
- **P4:** chỉ để heading `## P4 — Audit mask UQ` với dòng `Nội dung mục này được bổ sung cùng P5b.`. Task 6 sẽ thay dòng này.

- [ ] **Step 3: Kiểm tra heading**

Run: `grep -c '^## ' $REPO/docs/sv_b/00_tong_quan_va_lo_trinh.md`
Expected: ≥ 9

- [ ] **Step 4: Commit**

```bash
cd $REPO && git add -f docs/sv_b/00_tong_quan_va_lo_trinh.md docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md
git commit -m "docs(sv_b): add overview and supporting-phase guide"
```

---

### Task 3: `P2_moi_truong_checkpoint_smoke_test.md`

**Files:**
- Sandbox create: `pyproject.toml`, `src/nckh/__init__.py`, `src/nckh/paths.py`, `src/nckh/runcard.py`, `tests/test_paths_runcard.py`, `patches/panderm_base_seg.patch`, `.gitignore`
- Create: `docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md`

**Interfaces:**
- Produces:
  - `nckh.paths.project_root() -> Path`: đọc `NCKH_ROOT` mỗi lần gọi.
  - `nckh.paths.data_dir() -> Path`, `manifests_dir() -> Path` (= `data_dir()/"manifests"`), `checkpoints_dir() -> Path`, `runs_dir() -> Path`.
  - `nckh.paths.local_data_root() -> Path`: đọc `NCKH_LOCAL_DATA`, mặc định `/content/data`.
  - `nckh.runcard.sha256_file(path: Path, chunk: int = 1 << 20) -> str`.
  - `nckh.runcard.new_run_id(tag: str, now: datetime | None = None) -> str`: định dạng `YYYYmmdd-HHMMSS_<tag>`.
  - `nckh.runcard.write_run_card(run_dir: Path, *, seed: int, config: dict, inputs: dict[str, Path]) -> Path`: ghi `run_dir/run_card.json` với 9 khóa `run_id, created_at, git_commit, python, packages, gpu, seed, config, inputs`. `inputs` = {tên: {"path": str, "sha256": str}}; `run_id` = `run_dir.name`; `git_commit` = "unknown" nếu không phải repo git; `gpu` = null nếu không có torch hoặc CUDA.

- [ ] **Step 1: Viết test (fail)**

`tests/test_paths_runcard.py`:
- `test_project_root_reads_env(monkeypatch, tmp_path)`: đặt `NCKH_ROOT=tmp_path` → `project_root()==tmp_path` và `runs_dir()==tmp_path/'runs'`.
- `test_sha256_known(tmp_path)`: file `b"abc"` → `"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"`.
- `test_new_run_id_format`: `new_run_id('seg', datetime(2026,10,5,9,7,3)) == '20261005-090703_seg'`.
- `test_write_run_card_keys(tmp_path)`: tạo file input `x`, ghi card, đọc JSON; đủ 9 khóa, `inputs['x']['sha256'] == sha256_file(x)`.

Run: `cd $SANDBOX && $S/venv/bin/pip install -q -e . && $S/venv/bin/pytest tests/test_paths_runcard.py -q`. Lần chạy đầu `pip install` sẽ lỗi vì chưa có `pyproject.toml`. Bước fail mong đợi là: tạo `pyproject.toml` và `src/nckh/__init__.py` rỗng trước, rồi pytest báo `ImportError`.

- [ ] **Step 2: Viết `paths.py`, `runcard.py`; chạy test pass**

Yêu cầu:
- `pyproject.toml`: setuptools, `[tool.setuptools.packages.find] where = ["src"]`, `requires-python = ">=3.10"`, deps và extras theo Global Constraints, `[tool.pytest.ini_options] testpaths = ["tests"]`.
- `runcard`:
  - git commit lấy bằng `subprocess.run(["git","rev-parse","HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)`;
  - version gói lấy bằng `importlib.metadata.version` cho `numpy, pandas, scikit-learn, scikit-image, torch, torchvision, mmsegmentation, timm`; gói không cài thì bỏ qua (`PackageNotFoundError`);
  - GPU: thử `import torch`, nếu `torch.cuda.is_available()` thì lấy `{"name": get_device_name(0), "total_mb": ...}`;
  - mọi fallback đều `logging.warning` kèm lý do.

Expected: `4 passed`.

- [ ] **Step 3: Tạo và kiểm patch PanDerm Base**

Trong `$PANDERM`, sửa 2 file:
- `segmentation/models/cae_config.py`: `embed_dim=768, depth=12, num_heads=12, out_indices=[3, 5, 7, 11]`, `in_channels=[768, 768, 768, 768]`, `channels=768`.
- `segmentation/models/cae_seg.py`:
  - thêm `import os`;
  - đường dẫn = `os.environ.get('PANDERM_CKPT', 'model_weights/panderm_bb_data6_checkpoint-499.pth')`;
  - vòng nạp chỉ so `size` khi `k in model_dict`, ngược lại đưa vào danh sách `unexpected`;
  - in `matched/len(model_dict)` cùng 5 key `unexpected` đầu;
  - `raise RuntimeError` nếu tỷ lệ < 0.9.

Rồi chạy:

```bash
cd $PANDERM && git diff > $SANDBOX/patches/panderm_base_seg.patch && git checkout -- . \
 && git apply --check $SANDBOX/patches/panderm_base_seg.patch && echo PATCH_OK
```

Expected: `PATCH_OK`. Ngoài ra, sau khi áp patch tạm, kiểm cú pháp: `git apply $SANDBOX/patches/panderm_base_seg.patch && $S/venv/bin/python -m py_compile segmentation/models/cae_seg.py segmentation/models/cae_config.py && git checkout -- .`

- [ ] **Step 4: Viết `.gitignore` cho sandbox**

Nội dung:

```
/docs/*
!/docs/sv_b/
!/docs/superpowers/
.venv*/
__pycache__/
.ipynb_checkpoints/
*.pth
*.ckpt
data/
runs/
.idea/
```

- [ ] **Step 5: Viết `P2_moi_truong_checkpoint_smoke_test.md`**

Theo khung 8 mục, gồm:
- **(a) Cell setup cho `NB_cpu`, `NB_seg`, `NB_cls`:**
  - mount Drive và đồng bộ git theo spec 3.2;
  - `pip install -q uv`; `uv python install 3.10`; `uv venv --python 3.10 --seed /content/venv_seg`;
  - cài theo đúng Segmentation.md (`torch==2.2.1 torchvision==0.17.1 --index-url .../cu118`, `-r requirements.txt`, `mim install mmengine==0.10.4 mmcv==2.1.0 mmsegmentation==1.2.2`), cộng `openpyxl gdown`, cộng `-e $ROOT/nckh`;
  - tương tự cho `venv_cls` (torch 2.4.1, classification/requirements.txt);
  - mỗi cell setup in `torch.__version__` và `torch.cuda.get_device_name(0)`.
- **(b) Tạo repo:** `pyproject.toml`, `paths.py`, `runcard.py`, test, `.gitignore` (nguyên văn từ sandbox), rồi chạy `!pytest -q` trong NB_cpu, commit và push.
- **(c) Tải checkpoint:** `gdown --id 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB -O $ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth`, ghi sha256 vào `checkpoints/SHA256SUMS`. Ghi chú: nếu gdown báo quota thì mở link trên trình duyệt, chọn "Add shortcut to Drive" rồi copy.
- **(d) Cell soi key checkpoint:** `torch.load(map_location='cpu')`, đếm key theo prefix (`encoder.`, `decoder.`, `teacher.`, khác), in 10 key đầu và shape `encoder.patch_embed.proj.weight` (kỳ vọng `[768, 3, 16, 16]`). Kèm bảng "thấy X → làm Y" (ví dụ không có prefix `encoder.` thì sửa dòng `replace` trong patch).
- **(e) Áp patch** (khối diff nguyên văn), giải thích từng hunk; `cp` checkpoint vào `segmentation/model_weights/` hoặc đặt `PANDERM_CKPT`.
- **(f) Smoke test thật cho seg:** chạy trong `segmentation/`; `CAEv2_seg()`; forward 20 ảnh ISIC 2018 Validation (transform giống `dataset_seg.py`: cv2 resize 224 bicubic, ToTensor, Normalize 0.5); assert shape `[1, 2, 224, 224]` và `torch.isfinite`; lưu 10 overlay PNG vào `runs/<run_id>/smoke/`.
- **(g) Smoke test thật cho cls:**
  - đọc default bằng `grep -n "use_mean_pooling\|init_scale\|layer_scale_init_value\|rel_pos_bias\|sin_pos_emb" run_class_finetuning.py`;
  - dựng `panderm_base_patch16_224_finetune(num_classes=3, ...)`;
  - nạp pretrained theo đúng logic dòng 440–570 của upstream: bỏ prefix `encoder.`, bỏ `decoder.` và `teacher.`, đổi `norm.` thành `fc_norm.`;
  - in số key khớp;
  - forward 20 ảnh, assert tổng `softmax` = 1 ± 1e-5.
- **(h) Đo VRAM/thời gian inline:** `torch.cuda.reset_peak_memory_stats`, `torch.cuda.synchronize`, warmup 5 ảnh.
- **(i) Pilot 1 epoch seg** (`--epoch 1 --percent 5`): ngắt runtime cố ý, chạy lại với `--resume 0`, ghi log bằng chứng. Lệnh đầy đủ ở P5a; P2 chỉ đổi `--epoch 1 --percent 5`.
- **(j) Run card** cho lần smoke test.
- **(k) Bảng benchmark trống:** GPU · task · ms/ảnh mean/p95 · VRAM đỉnh.
- **(l) Lỗi Colab thường gặp:**
  - mmcv build lâu: dùng `mim` để lấy wheel cu118;
  - không được cấp GPU;
  - `devices` của Lightning: dùng `"0,"`;
  - wandb hỏi đăng nhập: đặt `WANDB_MODE=disabled`;
  - Drive hết dung lượng;
  - token GitHub hết hạn.
- **(m) Ghi chú:** 3 file Tuần 1 cũ đã lỗi thời, và lý do.

Đánh dấu ⚠️ GPU ở (c)–(i).

- [ ] **Step 6: Kiểm docs và commit**

Run: `SANDBOX=$SANDBOX $S/venv/bin/python $S/check_docs.py $REPO/docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md` → `OK`. Patch: trích khối diff trong docs ra file tạm và `diff` với `$SANDBOX/patches/panderm_base_seg.patch`; kỳ vọng không có khác biệt.

```bash
cd $REPO && git add -f docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md && git commit -m "docs(sv_b): add P2 environment, checkpoint and smoke test guide"
```

---

### Task 4: `P3_du_lieu_manifest_split.md`

**Files:**
- Sandbox create: `src/nckh/manifest.py`, `scripts/prepare_isic2018.py`, `scripts/prepare_isic2017_cls.py`, `scripts/build_manifest.py`, `configs/uq_column_map.yaml`, `tests/test_manifest.py`, `tests/test_prepare.py`
- Create: `docs/sv_b/P3_du_lieu_manifest_split.md`

**Interfaces:**
- Consumes: `nckh.runcard.sha256_file`, `nckh.paths.*`
- Produces:
  - `UQ_COLUMNS: tuple[str, ...] = ("participant_id","lesion_id","image_id","image_path","captured_at","modality")`.
  - `check_image(path: Path) -> dict`: các khóa `readable: bool, width: int | None, height: int | None, mode: str | None, error: str | None`. Mở bằng PIL, gọi `.load()` để phát hiện file hỏng.
  - `build_isic_seg_manifest(data_root: Path, split: str) -> pd.DataFrame`:
    - split ∈ {"train","val","test"} ánh xạ sang thư mục `Training`, `Validation`, `Test`;
    - quét `data_root/ISIC2018/<X>_Data/*.jpg`, mask tại `<X>_GroundTruth/<id>_segmentation.png`;
    - các cột `image_id, image_path, mask_path, split, sha256, width, height, mask_width, mask_height, readable, exclude_reason`;
    - đường dẫn là tương đối so với `data_root` (dạng POSIX);
    - `exclude_reason` ∈ {"", "unreadable_image", "missing_mask", "size_mismatch"};
    - sắp xếp theo `image_id`.
  - `find_cross_split_duplicates(df: pd.DataFrame, hash_col: str = "sha256", split_col: str = "split") -> pd.DataFrame`.
  - `load_uq_metadata(csv_path: Path, column_map: dict[str, str]) -> pd.DataFrame`:
    - key là tên chuẩn, value là tên cột gốc;
    - thiếu cột gốc → `ValueError` liệt kê đúng tên;
    - `captured_at` parse bằng `pd.to_datetime(..., utc=True, errors="coerce", format="mixed")`;
    - `modality` chuyển chữ thường và bỏ khoảng trắng;
    - các id chuyển sang `str` (NaN giữ nguyên NaN).
  - `assign_group_split(df: pd.DataFrame, group_col: str, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 2026) -> pd.Series`:
    - NaN trong `group_col` → `ValueError`;
    - nhóm duy nhất sắp xếp tăng dần, hoán vị bằng `np.random.default_rng(seed).permutation`;
    - `n_val = round(n*0.15)`, `n_test = round(n*0.15)`, train là phần còn lại;
    - thứ tự cắt: test trước, rồi val, rồi train;
    - trả Series tên `"split"`, cùng index với df.
  - `assert_no_group_leakage(df: pd.DataFrame, group_col: str, split_col: str = "split") -> None`: raise `AssertionError` liệt kê tối đa 10 nhóm bị lộ.
- `scripts/prepare_isic2018.py`:
  - hàm `download_missing(zips_dir: Path, urls: list[str]) -> list[Path]` (dùng `urllib.request.urlretrieve`, bỏ qua file đã tồn tại với kích thước khớp header `Content-Length`);
  - hàm `extract_all(zips_dir: Path, data_root: Path) -> dict` (giải nén từng zip vào `data_root/ISIC2018/<tên chuẩn>`, bỏ qua khi thư mục đích đã có số file ≥ số entry trong zip, trả `{zip_name: {"skipped": bool, "n_files": int}}`);
  - hàm `count_masks(data_root: Path) -> dict` (cho mỗi split: `images`, `masks`, `max_masks_per_image`);
  - CLI `--zips-dir --data-root [--skip-download]`, ghi `mask_count.json`.
  - Bảng ánh xạ zip → thư mục: `ISIC2018_Task1-2_Training_Input` → `Training_Data`, `ISIC2018_Task1_Training_GroundTruth` → `Training_GroundTruth`, tương tự cho Validation và Test. Trong zip có thư mục cha cùng tên, nên khi giải nén phải bỏ một cấp thư mục và bỏ các file không phải ảnh (ví dụ `LICENSE.txt`, `ATTRIBUTION.txt`).
- `scripts/prepare_isic2017_cls.py`:
  - hàm `make_cls_table(gt_csvs: dict[str, Path], image_dirs: dict[str, str]) -> pd.DataFrame` với các cột `image, label, split`;
  - nhãn: 0 nếu melanoma==1, 2 nếu seborrheic_keratosis==1, còn lại 1; cả hai bằng 1 → `ValueError`;
  - `image` = `f"{image_dirs[split]}/{image_id}.jpg"`;
  - hàm `make_trainphase(df) -> pd.DataFrame`: bỏ các hàng test, thêm bản sao các hàng val với `split="test"`;
  - CLI `--gt-dir --out-dir`, ghi `isic2017_cls.csv` và `isic2017_cls_trainphase.csv`;
  - thư mục ảnh mặc định: `ISIC-2017_Training_Data`, `ISIC-2017_Validation_Data`, `ISIC-2017_Test_v2_Data`.
- `scripts/build_manifest.py --data-root DIR --out CSV`: ghép 3 split, in số dòng mỗi `exclude_reason`, ghi `<out>`, `<out>.sha256`, `<out>.cross_split_duplicates.csv`.

URL tải (chính xác, đã kiểm tra trả về 200 ngày 05/10/2026):

```
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip      (11,2 GB)
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip          (2,4 GB)
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip               (6,2 GB)
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part3_GroundTruth.csv
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Data.zip
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part3_GroundTruth.csv
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Data.zip                (5,8 GB)
https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv
```

- [ ] **Step 1: Viết test (fail)**

`tests/test_manifest.py`. Fixture `isic_tree(tmp_path)` tạo bằng PIL ảnh 32×24:
- train `ISIC_0000001…4`, val `ISIC_0000005…6`, test `ISIC_0000007…8`;
- `ISIC_0000002.jpg` là bytes rác;
- `ISIC_0000003` không có mask;
- mask của `ISIC_0000004` có kích thước 16×16;
- `ISIC_0000005.jpg` có bytes giống hệt `ISIC_0000001.jpg`.

Các test:
- `test_manifest_relative_paths`.
- `test_exclude_reasons`: 0000002 có lý do `unreadable_image`, 0000003 có `missing_mask`, 0000004 có `size_mismatch`, các ảnh còn lại có "".
- `test_cross_split_duplicate_found`: tìm thấy 0000001 và 0000005.
- `test_group_split_disjoint_and_deterministic`: 100 participant × 3 ảnh; hai lần gọi cùng seed cho Series bằng nhau; seed khác cho Series khác; `assert_no_group_leakage` không raise; số participant (70, 15, 15).
- `test_group_split_rejects_nan_group`.
- `test_leakage_detected`.
- `test_load_uq_metadata_maps_and_validates`: map đúng cột; thiếu cột thì `ValueError` chứa tên cột; "2020-01-05" và "2020-01-05T10:00:00+10:00" đều parse được; "abc" thành NaT; "Dermoscopy " thành "dermoscopy".

`tests/test_prepare.py`:
- `test_isic2017_labels`: 3 dòng mel, nev, sk cho nhãn [0, 1, 2]; dòng mel=1 và sk=1 → `ValueError`.
- `test_trainphase_has_no_real_test`.
- `test_extract_all_idempotent`: tạo zip giả có thư mục cha `ISIC2018_Task1_Validation_GroundTruth/` chứa 2 PNG và 1 `LICENSE.txt`; lần 1 `skipped=False, n_files=2`; lần 2 `skipped=True`.
- `test_count_masks`.

- [ ] **Step 2: Viết code; pytest pass**

Run: `cd $SANDBOX && $S/venv/bin/pytest tests/test_manifest.py tests/test_prepare.py -q`
Expected: tất cả pass.

- [ ] **Step 3: Viết `configs/uq_column_map.yaml`**

Dạng mapping 6 dòng `participant_id: "<ĐIỀN TÊN CỘT GỐC>"`… Đây là template có chủ đích cho SV B điền sau P1. Thêm khóa `modality_dermoscopy_value: "dermoscopy"` để khai báo giá trị nào trong cột modality được coi là dermoscopy. Khóa này được dùng ở Task 7.

- [ ] **Step 4: Viết `P3_du_lieu_manifest_split.md`**

Bao gồm:
- Bảng dung lượng và quy tắc Drive/`/content`.
- Cell NB_cpu tải và giải nén vào `/content/zips` và `/content/data`; tùy chọn chỉ lưu các file GT nhỏ lên Drive. Cell xoá zip sau khi giải nén để giải phóng ổ `/content`.
- Giải thích layout mà loader upstream yêu cầu (trích `_get_paths_official`).
- Kiểm chứng câu "5 mask/ảnh" bằng `mask_count.json`, kèm ghi chú gửi SV A.
- CSV ISIC 2017 và lý do có file trainphase (link sang P5b).
- Manifest và hash; cách đọc bảng `exclude_reason`.
- Schema UQ, cách điền YAML; chia split theo participant **trước** khi ghép cặp.
- Bảng 8 unit test bắt buộc trong Mục 3.4 của kế hoạch, ánh xạ sang test cụ thể; test số 5–7 trỏ sang P6, test số 8 là `test_group_split_disjoint_and_deterministic`.
- Benchmark: thời gian tải/giải nén, số ảnh mỗi split, bảng flow lọc (để trống).

- [ ] **Step 5: `check_docs.py` OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P3_du_lieu_manifest_split.md && git commit -m "docs(sv_b): add P3 data, manifest and split guide"
```

---

### Task 5: `P5a_segmentation_isic2018.md` (kèm `metrics.py`)

**Files:**
- Sandbox create: `src/nckh/metrics.py`, `scripts/evaluate_seg.py`, `tests/test_metrics.py`
- Sandbox modify: `patches/panderm_base_seg.patch` (thêm hunk callback resume)
- Create: `docs/sv_b/P5a_segmentation_isic2018.md`
- Modify: `docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md` (khối diff patch được cập nhật)

**Interfaces:**
- Produces (`nckh.metrics`):
  - `dice_iou(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]`: cả hai rỗng → (1.0, 1.0); shape khác → `ValueError`.
  - `classification_metrics(y_true: np.ndarray, prob: np.ndarray, class_names: list[str]) -> dict`: các khóa `macro_f1, balanced_accuracy, auroc_macro, auroc_per_class (dict), recall_per_class (dict), confusion_matrix (list[list[int]]), brier, n_per_class (dict)`. Tổng prob mỗi hàng lệch 1 quá 1e-4 → `ValueError`. Brier = mean over samples của Σ_k (p_k − y_k)².
  - `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict`: các khóa `mae, rmse, bias, r2, n`; bias = mean(pred − true); r2 = NaN nếu n < 2.
  - `direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, stable_eps: float) -> float`.
  - `group_bootstrap(df: pd.DataFrame, group_col: str, stat_fn: Callable[[pd.DataFrame], float], n_boot: int = 2000, seed: int = 2026, alpha: float = 0.05) -> dict`: các khóa `estimate, ci_low, ci_high, n_groups, n_boot`.
  - `paired_bootstrap_diff(df: pd.DataFrame, group_col: str, stat_fn_a: Callable[[pd.DataFrame], float], stat_fn_b: Callable[[pd.DataFrame], float], n_boot: int = 2000, seed: int = 2026, alpha: float = 0.05) -> dict`: cùng khóa, cho a − b.
- `scripts/evaluate_seg.py`: hàm `evaluate_seg_results(xlsx: Path, out_dir: Path, n_boot: int = 2000, seed: int = 2026) -> dict` (cột upstream `name, dice, jac`); ghi `seg_metrics.json` (`{"dice": {...}, "iou": {...}, "n_images": n}`) và `seg_per_image.csv`; CLI `--results-xlsx --out-dir`.

- [ ] **Step 1: Viết test (fail)**

`tests/test_metrics.py`:
- `test_dice_iou_hand`: pred = cột 0–3 hàng 0, ref = cột 2–5 hàng 0 → dice 0.5, iou 1/3.
- `test_dice_both_empty`.
- `test_dice_shape_mismatch`.
- `test_classification_perfect`.
- `test_classification_rejects_bad_prob`.
- `test_regression_hand`: true [0, 0], pred [1, −1] → mae 1, rmse 1, bias 0.
- `test_direction_accuracy`: true [0.1, −0.1, 0.0], pred [0.2, 0.05, 0.001], eps 0.01 → 2/3.
- `test_group_bootstrap_resamples_groups`: df có nhóm A gồm 100 hàng giá trị 0 và nhóm B gồm 1 hàng giá trị 1; stat là mean; `n_groups == 2`; `ci_low == 0.0` và `ci_high == 1.0`.
- `test_paired_diff_sign`: 50 nhóm, |err_a| = 0.01, |err_b| = 0.05 → `ci_high < 0`.
- `test_bootstrap_deterministic`.
- `test_evaluate_seg_xlsx(tmp_path)`.

- [ ] **Step 2: Code; pytest pass**

- [ ] **Step 3: Thêm hunk resume vào patch**

Trong `$PANDERM/segmentation/workers/train.py`, đổi `callbacks=[checkpoint_best]` thành `callbacks=[checkpoint_best, checkpoint_callback]`. Lý do: upstream khai báo `checkpoint_callback` nhưng không truyền vào Trainer, nên file `model_checkpoint_0.ckpt` mà `--resume` cần không bao giờ được tạo. Tạo lại patch gồm cả 3 file theo cách của Task 3 Step 3, chạy `git apply --check` (kỳ vọng `PATCH_OK`), rồi cập nhật khối diff trong P2.md và chạy lại check của Task 3 Step 6.

- [ ] **Step 4: Viết `P5a_segmentation_isic2018.md`**

Bao gồm:
- Cell NB_seg (lệnh đầy đủ):

  ```
  cd /content/PanDerm/segmentation && PANDERM_CKPT=$ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth python run.py --workers 2 --gpu "0," --batch_size 8 --test_batch_size 8 --epoch 100 --lr 1e-4 --weight_decay 0.05 --model cae_seg --size 224 --dataset ISIC2018 --parent_path /content/data/ --save_name $ROOT/runs/<run_id>/ --seed 0 --smoke_test
  ```

  Giải thích từng cờ: `--smoke_test` chỉ để dùng CSVLogger; dấu `/` cuối của `parent_path` và `save_name` là bắt buộc vì upstream ghép chuỗi trực tiếp.
- Batch size: bắt đầu từ 8, giảm nếu OOM, ghi vào run card.
- Hành vi upstream cần biết: chọn checkpoint theo `Val/Jac`; augmentation chỉ dùng cho train; test tính ở 224×224 và giữ thành phần liên thông lớn nhất; GT rỗng được gán 1 pixel ở tâm. Viết sẵn đoạn mô tả cho phần Methods.
- Resume bằng `--resume 0` (cần hunk callback trong patch).
- Khóa cấu hình trước khi mở test: run card và dòng nhật ký quyết định.
- Lệnh test một lần: thêm `--evaluate --save_results` vào lệnh trên, sau đó chạy `evaluate_seg.py` trên `count_results_ISIC2018_0_0.xlsx`.
- Nếu tài nguyên cho phép, chạy 3 seed và báo mean ± sd.
- Toàn bộ code `metrics.py`, `evaluate_seg.py`, `test_metrics.py`.
- Benchmark: thời gian mỗi epoch, VRAM, bảng Dice/IoU trống.
- Đánh dấu ⚠️ GPU.

- [ ] **Step 5: check_docs OK cho P5a và P2; commit**

```bash
cd $REPO && git add -f docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md && git commit -m "docs(sv_b): add P5a segmentation guide and shared metrics"
```

---

### Task 6: `P5b_classification_isic2017.md` + hoàn thiện mục P4

**Files:**
- Sandbox create: `scripts/evaluate_cls.py`, `scripts/annotator_agreement.py`, `tests/test_eval_cls_agreement.py`
- Create: `docs/sv_b/P5b_classification_isic2017.md`
- Modify: `docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md` (thay dòng giữ chỗ P4)

**Interfaces:**
- Consumes: `nckh.metrics.classification_metrics`, `group_bootstrap`, `dice_iou`; `nckh.pairs.stratified_audit_sample` (định nghĩa ở Task 7; docs P4 chỉ gọi lại hàm này).
- Produces:
  - `evaluate_cls_csv(pred_csv: Path, out_dir: Path, labels_csv: Path | None = None, class_names: tuple[str, ...] = ("melanoma","nevus","seborrheic_keratosis"), n_boot: int = 2000, seed: int = 2026) -> dict`:
    - đọc `test.csv` của upstream (cột `filename, true_label, predicted_label, probability_class_0..2`);
    - nếu có `labels_csv`, kiểm số dòng bằng số hàng `split=="test"`, ngược lại `ValueError`;
    - ghi `cls_metrics.json` = `classification_metrics(...)` cộng `macro_f1_ci` (bootstrap theo `filename`).
  - `annotator_agreement(dir_a: Path, dir_b: Path) -> pd.DataFrame`: các cột `image_id, dice, iou, area_a, area_b, area_diff`; file chỉ có một phía → `ValueError`; mask đọc bằng PIL, pixel > 127 là tổn thương.

- [ ] **Step 1: Test (fail)**

- `test_evaluate_cls_csv`: CSV giả 6 dòng có đủ 3 lớp → JSON có `macro_f1` và `macro_f1_ci`.
- `test_evaluate_cls_label_count_mismatch`.
- `test_agreement_identical`: dice = 1.
- `test_agreement_area_diff`.
- `test_agreement_missing_file_raises`.

- [ ] **Step 2: Code; pytest pass**

- [ ] **Step 3: Viết `P5b_classification_isic2017.md`**

Bao gồm:
- Cell NB_cls: tải ảnh ISIC 2017 về `/content/data/ISIC2017/`, chạy `prepare_isic2017_cls.py`, `%env WANDB_MODE=disabled`.
- Lệnh fine-tune đầy đủ:

  ```
  python3 run_class_finetuning.py --model PanDerm_Base_FT --pretrained_checkpoint $ROOT/checkpoints/panderm_bb_data6_checkpoint-499.pth --nb_classes 3 --batch_size 32 --update_freq 4 --lr 5e-4 --warmup_epochs 10 --epochs 50 --layer_decay 0.65 --drop_path 0.2 --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --weights --monitor recall --sin_pos_emb --auto_resume --exp_name isic2017_ft --imagenet_default_mean_and_std --wandb_name isic2017_ft_s0 --output_dir $ROOT/runs/<run_id>/ --csv_path $ROOT/data/manifests/isic2017_cls_trainphase.csv --root_path /content/data/ISIC2017/ --seed 0
  ```

- Giải thích `--monitor recall`: macro recall bằng balanced accuracy, khớp với metric chính. Phải chốt trong protocol P0.
- Giải thích `--update_freq 4`: batch hiệu dụng 128 như khuyến nghị của upstream.
- Cảnh báo dòng ~720 của upstream (tự chạy test ở epoch cuối), giải thích file trainphase vô hiệu hóa việc này, và cách kiểm: `test.csv` sinh ra khi train có số dòng bằng số ảnh val.
- Lệnh eval một lần: `--eval --resume $ROOT/runs/<run_id>/checkpoint-best.pth --csv_path .../isic2017_cls.csv`; dùng `--TTA` hay không phải theo protocol. Sau đó chạy `evaluate_cls.py --labels-csv`.
- Calibration plot (cell matplotlib dùng `sklearn.calibration.calibration_curve`, mỗi lớp OvR).
- Toàn bộ code `evaluate_cls.py` và test.
- Benchmark và bảng metric trống (theo lớp: n, recall, AUROC).
- Đánh dấu ⚠️ GPU.

- [ ] **Step 4: Thay mục P4 trong file P0_P1_P4_P11**

Nội dung P4:
- SV B gán độc lập một phần ngẫu nhiên đã định trước (chọn bằng `stratified_audit_sample` với seed 2026, `strata_cols=["delta_days","area_ratio_t"]`).
- Mask lưu dạng PNG 0/255, tên file `<image_id>.png`, mỗi người một thư mục `audit/annotator_A/` và `audit/annotator_B/` trên Drive (chỉ khi đã có quyền UQ).
- Gợi ý công cụ gán: CVAT hoặc labelme; xuất mask nhị phân.
- Toàn bộ code `annotator_agreement.py`, lệnh chạy và cách đọc kết quả.
- Không xoá mask độc lập sau khi đã thống nhất.

- [ ] **Step 5: check_docs OK cả 2 file; commit**

```bash
cd $REPO && git add -f docs/sv_b/P5b_classification_isic2017.md docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md && git commit -m "docs(sv_b): add P5b classification guide and P4 audit support"
```

---

### Task 7: `P6_ghep_cap_va_ridge.md`

**Files:**
- Sandbox create: `src/nckh/pairs.py`, `src/nckh/features.py`, `src/nckh/forecast.py`, `src/nckh/infer.py`, `scripts/make_fake_uq.py`, `scripts/build_pairs.py`, `scripts/infer_images.py`, `scripts/train_ridge.py`, `scripts/bench_inference.py`, `tests/test_pairs.py`, `tests/test_features.py`, `tests/test_forecast.py`, `tests/test_infer.py`
- Create: `docs/sv_b/P6_ghep_cap_va_ridge.md`

**Interfaces:**
- Consumes: `nckh.manifest.UQ_COLUMNS`, `load_uq_metadata`, `assign_group_split`, `assert_no_group_leakage`, `check_image`; `nckh.metrics.regression_metrics`; `nckh.runcard.write_run_card`.
- Produces:
  - **`nckh.pairs`**
    - `PAIR_EXCLUDE_REASONS = ("not_dermoscopy","missing_timestamp","missing_lesion_id","same_timestamp","participant_mismatch","unreadable_image")`.
    - `build_consecutive_pairs(df: pd.DataFrame, dermoscopy_value: str = "dermoscopy") -> tuple[pd.DataFrame, pd.DataFrame]`:
      - đầu vào có `UQ_COLUMNS`, `split`, và tùy chọn `readable`;
      - lọc theo thứ tự: lesion thiếu id, ảnh không đọc được, sai modality, thiếu thời gian, lesion có nhiều participant;
      - sắp xếp theo `(lesion_id, captured_at, image_id)`;
      - với mỗi cặp kề nhau: Δt = 0 thì đưa ảnh sau vào `excluded` với lý do `same_timestamp` và ghép ảnh trước với ảnh kế tiếp có thời gian lớn hơn;
      - `pairs` có các cột `participant_id, lesion_id, split, image_id_t, image_path_t, captured_at_t, image_id_t1, image_path_t1, captured_at_t1, delta_days`;
      - `excluded` có các cột `image_id, lesion_id, reason`.
    - `stratified_audit_sample(pairs: pd.DataFrame, n_pairs: int, strata_cols: list[str], seed: int = 2026) -> pd.DataFrame`: strata = tổ hợp quartile (`pd.qcut(..., 4, duplicates="drop")`) của từng cột; lấy `floor(n_pairs/k)` cặp mỗi stratum, phần dư lấy thêm từ các stratum theo thứ tự; stratum thiếu cặp thì lấy hết. Trả về ≤ n_pairs hàng và có thêm cột `stratum`.
  - **`nckh.features`**
    - `mask_features(mask: np.ndarray) -> dict`: các khóa `area_ratio, circularity, eccentricity, empty`.
      - `area_ratio` = tổng pixel tổn thương / (H·W).
      - Circularity = 4πA/P², với P = `skimage.measure.perimeter` của thành phần lớn nhất; P = 0 → NaN; cắt về [0, 1].
      - Eccentricity lấy từ `regionprops` của thành phần lớn nhất.
      - Mask rỗng → `area_ratio=0.0`, circularity và eccentricity = NaN, `empty=True`.
    - `FEATURE_COLUMNS: tuple[str, ...] = ("area_ratio_t","circularity_t","eccentricity_t","p_mel_t","p_nev_t","p_sk_t","delta_days")`.
    - `TARGET_COLUMN = "delta_area"`.
    - `build_feature_table(pairs: pd.DataFrame, per_image: pd.DataFrame) -> pd.DataFrame`:
      - `per_image` có các cột `image_id, area_ratio, circularity, eccentricity, empty, p_mel, p_nev, p_sk`;
      - đầu ra = pairs cộng `FEATURE_COLUMNS` cộng `area_ratio_t1, empty_t1, delta_area, usable, exclude_reason`;
      - `exclude_reason` ưu tiên theo thứ tự `missing_inference` > `empty_mask` > `nonfinite_feature`; `usable = exclude_reason == ""`.
  - **`nckh.forecast`**
    - `ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)`.
    - `make_model(alpha: float) -> Pipeline`: `[("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]`.
    - `select_alpha(train: pd.DataFrame, val: pd.DataFrame, grid: Sequence[float] = ALPHA_GRID) -> tuple[float, pd.DataFrame]`: MAE trên val; chênh nhau trong 1e-12 coi là hòa và chọn alpha lớn hơn; trả bảng cột `alpha, val_mae`.
    - `fit_forecaster(train: pd.DataFrame, alpha: float) -> Pipeline`.
    - `predict_delta(model: Pipeline, df: pd.DataFrame) -> np.ndarray`.
    - `predict_next_area(area_t: np.ndarray, delta: np.ndarray) -> np.ndarray`.
    - `baseline_delta(n: int) -> np.ndarray`.
    - `validate_delta_days(value: object) -> float`: từ chối bool, None, chuỗi rỗng, chuỗi không parse được bằng `float()`, NaN, ±inf, ≤ 0 bằng `ValueError` với thông báo `"Δt phải là số ngày dương, ví dụ 180"`.
  - **`nckh.infer`** (torch và torchvision import lazily trong hàm)
    - `seg_preprocess(rgb: np.ndarray) -> "torch.Tensor"`: PIL resize (224, 224) BICUBIC, chuẩn hóa 0.5/0.5, shape `[1, 3, 224, 224]`. Docs ghi chú upstream dùng `cv2.INTER_CUBIC` nên có thể lệch nhỏ, và pilot P2 sẽ so Dice trên 20 ảnh val giữa hai cách.
    - `cls_preprocess(rgb: np.ndarray) -> "torch.Tensor"`: `Resize(256, BICUBIC)` → `CenterCrop(224)` → `ToTensor` → `Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))`.
    - `largest_component(mask: np.ndarray) -> np.ndarray`.
    - `seg_logits_to_mask(logits: "torch.Tensor", out_hw: tuple[int, int]) -> np.ndarray`.
    - `class SegPredictor`: `__init__(self, model: "torch.nn.Module", device: str = "cpu")`; `from_checkpoint(cls, ckpt_path: Path, panderm_seg_dir: Path, pretrained_path: Path, device: str = "cuda") -> "SegPredictor"` (chdir tạm vào `panderm_seg_dir`, thêm vào `sys.path`, đặt `PANDERM_CKPT`, dựng `models.Segmentation_Module(argparse.Namespace(model="cae_seg"))`, nạp `state_dict` của Lightning ckpt); `predict(self, rgb: np.ndarray) -> np.ndarray`.
    - `class ClsPredictor`: `__init__(self, model, device="cpu")`; `from_checkpoint(cls, ckpt_path: Path, panderm_cls_dir: Path, device: str = "cuda", nb_classes: int = 3, model_kwargs: dict | None = None)` (dựng `panderm_base_patch16_224_finetune`, nạp `torch.load(ckpt)["model"]`); `predict(self, rgb) -> np.ndarray` (softmax, shape (3,)).
  - **Scripts**
    - `make_fake_uq.py --out-dir DIR --participants 30 --seed 2026`:
      - mỗi participant có 1–2 lesion, mỗi lesion 2–4 lần chụp cách nhau 150–210 ngày;
      - ảnh 256×256 nền da, elip tối có bán kính tăng hoặc giảm tuyến tính theo thời gian;
      - `metadata.csv` dùng tên cột `pid, lesion, img, file, date, type`, kèm `fake_column_map.yaml`;
      - cố ý chèn đúng 1 ảnh trùng ngày, 1 ảnh thiếu ngày, 1 ảnh `type=clinical`.
    - `build_pairs.py --metadata CSV --column-map YAML --data-root DIR --out-dir DIR --seed 2026`: đọc metadata, chạy `check_image`, gán split theo participant, assert không leakage, ghép cặp; ghi `uq_manifest.csv`, `pairs.csv`, `pairs_excluded.csv`, `flow.json` (`n_images, n_participants, n_lesions, excluded_by_reason, n_pairs_by_split`).
    - `infer_images.py --task {seg,cls} (--images P [P ...] | --manifest CSV --path-col COL --id-col COL --data-root DIR) --out-dir DIR [--seg-ckpt --pretrained --panderm-dir --cls-ckpt --device] [--fake]`:
      - seg ghi `masks/<image_id>.png` và `seg_features.csv`;
      - cls ghi `cls_probs.csv` (`image_id, p_mel, p_nev, p_sk`);
      - `--fake`: seg dùng ngưỡng xám < 100 rồi giữ thành phần lớn nhất; cls trả (0.2, 0.6, 0.2); in `CẢNH BÁO: chế độ --fake, không dùng cho kết quả` ra stderr.
    - `train_ridge.py --pairs CSV --seg-features CSV --cls-probs CSV --out-dir DIR [--open-test] [--yes]`:
      - ghi `features.csv`, `alpha_selection.csv`, `ridge.joblib` (dict `{"model", "alpha", "feature_columns", "pairs_sha256"}`), `predictions_val.csv`;
      - `--open-test` kèm prompt `input()`, hoặc `--yes` để bỏ qua prompt, thì ghi `predictions_test.csv`;
      - các cột prediction: `participant_id, lesion_id, image_id_t, split, delta_days, area_ratio_t, delta_area, pred_ridge, pred_baseline, next_area_ridge`;
      - ghi run card.
    - `bench_inference.py --task {seg,cls} --images GLOB --n 50 --warmup 5 --out JSON [checkpoint args]`: dùng `SegPredictor`/`ClsPredictor`; ghi `ms_per_image_mean, ms_per_image_p50, ms_per_image_p95, peak_vram_mb, gpu_name, n`.

- [ ] **Step 1: Test (fail)**

`tests/test_pairs.py`:
- `test_pairs_consecutive_only`: 3 lần chụp → 2 cặp (1→2, 2→3).
- `test_pairs_never_cross_lesion`.
- `test_delta_days_positive_float`: 2020-01-01 → 2020-07-01 cho 182.0.
- `test_same_day_excluded_reason`: 3 ảnh (ngày 1, ngày 1, ngày 200) → 1 cặp và 1 dòng excluded với lý do `same_timestamp`.
- `test_missing_timestamp_reason`.
- `test_not_dermoscopy_reason`.
- `test_participant_mismatch_reason`.
- `test_unreadable_reason`.
- `test_timezone_and_date_only_mix`: "2020-01-05" và "2020-01-06T00:00:00+00:00" cho Δt = 1.0.
- `test_pairs_inherit_split_and_no_leakage`.
- `test_stratified_audit_sample_size_and_seed`.

`tests/test_features.py`:
- `test_square_area_ratio`: 10×10 trong 100×100 → 0.01.
- `test_disk_circularity_near_one`: r=30 → > 0.85.
- `test_empty_mask_flagged`.
- `test_single_pixel_mask_no_crash`.
- `test_feature_table_no_future_columns`.
- `test_feature_table_empty_t1_unusable`.
- `test_feature_table_missing_inference_reason`.

`tests/test_forecast.py`:
- `test_scaler_fit_on_train_only`.
- `test_select_alpha_uses_val`.
- `test_select_alpha_tie_prefers_larger`.
- `test_baseline_zero`.
- `test_next_area_clipped`.
- `test_validate_delta_days_accepts` (30, 30.5, "45").
- `test_validate_delta_days_rejects` (parametrize `[-30, 0, "", "6 tháng", "nan", "inf", None, True, float("nan")]`).
- `test_ridge_beats_baseline_on_learnable_synthetic`.

`tests/test_infer.py` (đầu file có `pytest.importorskip("torch")`):
- `test_seg_preprocess_shape_and_range`.
- `test_cls_preprocess_center_crop` (300×400 → `[1, 3, 224, 224]`).
- `test_seg_logits_to_mask_largest_component`.
- `test_seg_predictor_with_fake_model`.
- `test_cls_predictor_probs_sum_one`.

- [ ] **Step 2: Code; pytest pass toàn bộ**

Run: `cd $SANDBOX && $S/venv/bin/pytest -q && $S/venv/bin/pytest tests/test_features.py -q -W error::RuntimeWarning`
Expected: tất cả pass.

- [ ] **Step 3: Chạy end-to-end bằng dữ liệu giả**

```bash
cd $SANDBOX && P=$S/venv/bin/python
$P scripts/make_fake_uq.py --out-dir $S/fake_uq --participants 30 --seed 2026
$P scripts/build_pairs.py --metadata $S/fake_uq/metadata.csv --column-map $S/fake_uq/fake_column_map.yaml --data-root $S/fake_uq --out-dir $S/fake_run --seed 2026
$P scripts/infer_images.py --task seg --manifest $S/fake_run/uq_manifest.csv --path-col image_path --id-col image_id --data-root $S/fake_uq --out-dir $S/fake_run --fake
$P scripts/infer_images.py --task cls --manifest $S/fake_run/uq_manifest.csv --path-col image_path --id-col image_id --data-root $S/fake_uq --out-dir $S/fake_run --fake
$P scripts/train_ridge.py --pairs $S/fake_run/pairs.csv --seg-features $S/fake_run/seg_features.csv --cls-probs $S/fake_run/cls_probs.csv --out-dir $S/fake_run --open-test --yes
```

Expected:
- `flow.json` có `excluded_by_reason` chứa `same_timestamp: 1`, `missing_timestamp: 1`, `not_dermoscopy: 1`;
- các file `ridge.joblib` và `predictions_test.csv` tồn tại;
- không có traceback.

Ghi lại output thật để đưa vào docs dưới nhãn "kết quả mẫu trên dữ liệu GIẢ".

- [ ] **Step 4: Viết `P6_ghep_cap_va_ridge.md`**

Bao gồm:
- **Cổng Go/No-Go UQ:** checklist 5 điều kiện của Phase 1; chỉ khi đủ mới tạo `data/uq/` trên Drive.
- **Công thức** a_t, Δa, z_t, clip; phân biệt mức A và mức B.
- **Bảng 7 điều chống leakage** (kế hoạch Mục 6.5), mỗi điều ánh xạ sang test hoặc cơ chế cụ thể.
- **Toàn bộ code** của các module, script và test trong task này.
- **Chạy trên dữ liệu giả** (cell NB_cpu), kèm output mẫu.
- **Chạy với UQ thật:** `infer_images` seg trên NB_seg, cls trên NB_cls, Ridge trên NB_cpu.
- **`bench_inference.py`** và cách chạy trên NB_seg/NB_cls để hoàn thiện benchmark P2.
- **Mức B:** cell tính `mask_features` cho thư mục mask thủ công thành `seg_features_manual.csv`, rồi chạy `train_ridge` để đánh giá (không fit lại); ghi rõ chỉ đánh giá model đã khóa.
- **Bảng trống:** flow, MAE.
- Đánh dấu ⚠️ GPU cho `from_checkpoint` và `bench_inference`.

- [ ] **Step 5: check_docs OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P6_ghep_cap_va_ridge.md && git commit -m "docs(sv_b): add P6 pairing and Ridge forecast guide"
```

---

### Task 8: `P7_kiem_thu_metrics_bootstrap.md`

**Files:**
- Sandbox create: `scripts/evaluate_forecast.py`, `tests/test_evaluate_forecast.py`
- Create: `docs/sv_b/P7_kiem_thu_metrics_bootstrap.md`

**Interfaces:**
- Consumes: `nckh.metrics.regression_metrics`, `direction_accuracy`, `group_bootstrap`, `paired_bootstrap_diff`; CSV prediction của Task 7.
- Produces: `evaluate_forecast(pred_csv: Path, out_dir: Path, stable_eps: float, n_boot: int = 2000, seed: int = 2026) -> dict`.
  - CSV có > 1 giá trị `split` → `ValueError`.
  - Ghi `forecast_metrics.json` với các khóa `split, n_pairs, n_participants, stable_eps, ridge, baseline, diff_ridge_minus_baseline`. `ridge` và `baseline` chứa `mae, rmse, bias` (mỗi cái gồm estimate/ci_low/ci_high) cộng `r2, direction_accuracy`; `diff` chứa `mae, rmse`.
  - Ghi `forecast_by_dt_bin.csv`: tertile của `delta_days`, các cột `bin, n_pairs, mae_ridge, mae_baseline`; bin < 20 cặp ghi `n_pairs` kèm metric NaN và cột `note="<20 cặp"`.
  - CLI `--predictions --out-dir --stable-eps`.

- [ ] **Step 1: Test (fail)**

- `test_evaluate_forecast_keys`.
- `test_diff_negative_when_ridge_better`.
- `test_requires_single_split`.
- `test_bin_min_size`.

- [ ] **Step 2: Code; pass; chạy trên `$S/fake_run/predictions_test.csv`**

- [ ] **Step 3: Viết `P7_kiem_thu_metrics_bootstrap.md`**

Bao gồm:
- **Bảng 7.1 của kế hoạch:** mỗi dòng ánh xạ sang test hoặc script kèm đường dẫn.
- **Metric chính và phụ của từng nhánh,** công thức và cách đọc (ví dụ MAE = 0.01 nghĩa là sai 1 điểm phần trăm diện tích ảnh).
- **Vì sao bootstrap theo participant,** minh họa bằng `test_group_bootstrap_resamples_groups`.
- **Cách đọc paired bootstrap:** CI của Ridge − baseline chứa 0 nghĩa là chưa đủ bằng chứng Ridge tốt hơn.
- **Hai cách chọn `stable_eps` trên val:**
  - trung vị |Δa| trên val;
  - độ bất đồng diện tích giữa người gán ở P4.

  Phải ghi vào protocol trước khi chạy.
- **3 seed và CI:** không trộn hai loại biến thiên (kế hoạch 7.5).
- **Lệnh chạy toàn bộ pytest** và danh sách file test kèm số test kỳ vọng (lấy số thật từ sandbox).
- **Code `evaluate_forecast.py` và test.**
- **Bảng kết quả trống** theo Mục 10 của kế hoạch.

- [ ] **Step 4: check_docs OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P7_kiem_thu_metrics_bootstrap.md && git commit -m "docs(sv_b): add P7 testing, metrics and bootstrap guide"
```

---

### Task 9: `P8_robustness.md`

**Files:**
- Sandbox create: `src/nckh/degrade.py`, `scripts/make_degraded.py`, `tests/test_degrade.py`
- Create: `docs/sv_b/P8_robustness.md`

**Interfaces:**
- Produces (`nckh.degrade`, mọi hàm nhận `np.uint8` HxWx3, trả mảng mới cùng shape/dtype, không sửa đầu vào):
  - `adjust_brightness(img: np.ndarray, alpha: float) -> np.ndarray`.
  - `gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray` (`PIL.ImageFilter.GaussianBlur(radius=sigma)`; docs ghi rõ radius của Pillow là σ).
  - `white_balance(img: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray`.
  - `add_hair(img: np.ndarray, coverage: float, seed: int) -> tuple[np.ndarray, float]`: vẽ dần các đường Bézier bậc 2 màu nâu/đen, độ rộng 1–3 px, bằng `ImageDraw.line` trên một layer mask, cho đến khi độ che ≥ coverage; nếu nét cuối làm vượt quá coverage + 0.005 thì thử lại nét đó với độ rộng 1. Trả về độ che thực tế.
  - `ben_graham(img: np.ndarray, sigma: float) -> np.ndarray`.
  - `gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray`.
  - `DEGRADATION_LEVELS: dict[str, Callable[[np.ndarray, int], np.ndarray]]`: đúng 10 mục `brightness_0.8, brightness_1.2, blur_1.0, blur_1.5, wb_r1.1_b0.9, wb_r0.9_b1.1, wb_r1.2_b0.8, wb_r0.8_b1.2, hair_0.01, hair_0.03`. Ben Graham và gamma nằm riêng trong `ENHANCEMENTS: dict[str, Callable[[np.ndarray], np.ndarray]]` với `ben_graham_10, gamma_0.8, gamma_1.2`; docs ghi rõ σ = 10 là mặc định và có thể chọn trên val.
- `make_degraded.py --manifest CSV --path-col COL --id-col COL --data-root DIR --split test --level NAME --out-dir DIR --seed 2026 [--allow-non-test]`:
  - `--split` khác `test` mà không có `--allow-non-test` → exit 2 kèm thông báo;
  - seed của từng ảnh = `seed + crc32(image_id)`, để kết quả tái lập được bất kể thứ tự xử lý;
  - ghi ảnh PNG và `degraded_manifest.csv` (`image_id, source_path, level, hair_coverage`).
- **Định dạng `robustness.json`** (do cell P8 ghi, P10 đọc): `{level: {metric_name: {"estimate","ci_low","ci_high"}}}`, trong đó metric_name ∈ {`delta_dice`, `delta_macro_f1`, `delta_mae`}.

- [ ] **Step 1: Test (fail)**

- `test_all_levels_dtype_shape_range` (parametrize theo `DEGRADATION_LEVELS` ∪ `ENHANCEMENTS`).
- `test_input_not_mutated`.
- `test_brightness_clips`.
- `test_blur_reduces_variance`.
- `test_white_balance_green_unchanged`.
- `test_hair_coverage_within_tolerance` (0.01 và 0.03, ảnh 256×256, seed 0–4).
- `test_hair_deterministic_by_seed`.
- `test_ben_graham_range`.
- `test_gamma_identity_at_one`.
- `test_make_degraded_refuses_train_split` (gọi `main([...])`, kỳ vọng `SystemExit` với code 2).

- [ ] **Step 2: Code; pass; chạy `make_degraded.py --level hair_0.03` trên `$S/fake_run/uq_manifest.csv` với split test**

- [ ] **Step 3: Viết `P8_robustness.md`**

Bao gồm:
- Công thức của từng phép biến đổi và lý do chọn mức.
- 7 bước quy trình không rò rỉ của kế hoạch Mục 8.2, ánh xạ sang lệnh cụ thể.
- **Vòng lặp batch trên Colab:** với mỗi level, chạy `make_degraded` vào `/content/degraded/<level>`, rồi `infer_images`, rồi đánh giá, rồi `rm -rf`.
- **Robustness segmentation trên ISIC 2018 test:** cell đầy đủ tính `dice_iou` so với GT gốc cho cả ảnh sạch và ảnh suy giảm, rồi `paired_bootstrap_diff` theo `image_id`; ghi vào `robustness.json`.
- **Robustness classification trên ISIC 2017 test:** đo ΔMacro-F1 bằng paired bootstrap.
- **Robustness forecast:** chỉ suy giảm ảnh t, giữ target, đo ΔMAE.
- **Ben Graham và gamma:** chỉ thử trên val (`--allow-non-test --split val`); chọn tối đa 1 biến thể; Gabor ngoài phạm vi.
- **Bảng và biểu đồ trống.**
- Đánh dấu ⚠️ GPU cho phần suy luận.

- [ ] **Step 4: check_docs OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P8_robustness.md && git commit -m "docs(sv_b): add P8 robustness guide"
```

---

### Task 10: `P9_demo_streamlit.md`

**Files:**
- Sandbox create: `demo/pipeline.py`, `demo/app.py`, `tests/test_demo_pipeline.py`
- Create: `docs/sv_b/P9_demo_streamlit.md`

**Interfaces:**
- Consumes: CLI `scripts/infer_images.py` (Task 7); `nckh.forecast.validate_delta_days`, `predict_next_area`; `nckh.features.mask_features`, `FEATURE_COLUMNS`; `ridge.joblib` (dict theo Task 7).
- Produces (`demo/pipeline.py`):
  - `DISCLAIMER: str`: nguyên văn từ Global Constraints.
  - `run_inference(image_path: Path, work_dir: Path, seg_cmd: list[str], cls_cmd: list[str], timeout_s: int = 600) -> tuple[np.ndarray, dict[str, float]]`:
    - `seg_cmd`/`cls_cmd` là tiền tố lệnh, ví dụ `["/content/venv_seg/bin/python", "scripts/infer_images.py", "--task", "seg", "--seg-ckpt", ...]`; hàm nối thêm `--images <image_path> --out-dir <work_dir>`;
    - returncode ≠ 0 → `RuntimeError` chứa 2000 ký tự cuối của stderr;
    - đọc `masks/<stem>.png` và `cls_probs.csv`.
  - `forecast_from_outputs(mask: np.ndarray, probs: dict[str, float], delta_days_raw: object, ridge_bundle: dict, stable_eps: float) -> dict`:
    - các khóa `area_ratio_t, delta_area_pred, next_area_pred, direction ("tăng"/"giảm"/"ổn định"), warnings: list[str]`;
    - gọi `validate_delta_days` trước tiên (raise `ValueError`);
    - mask rỗng → các trường dự báo = None và cảnh báo `"Không tìm thấy vùng tổn thương trong ảnh; không đưa ra dự báo."`;
    - `area_ratio_t > 0.9` hoặc `< 0.001` → thêm cảnh báo chất lượng mask.
  - `demo/app.py`:
    - đọc cấu hình từ biến môi trường `NCKH_SEG_CMD`, `NCKH_CLS_CMD` (chuỗi được tách bằng `shlex.split`), `NCKH_RIDGE`, `NCKH_STABLE_EPS`;
    - `st.file_uploader` cho jpg/png; PIL không đọc được → `st.error`;
    - `st.number_input` cho Δt và `st.text_input` để thử nhập tự do; giá trị nào cũng đi qua `validate_delta_days`;
    - cache bằng `st.cache_data` với key là sha256 bytes ảnh;
    - hiển thị ảnh gốc, overlay (`mark_boundaries` của skimage), bảng 3 xác suất, a_t, â_{t+1}, Δa, hướng thay đổi; disclaimer ở đầu và cuối trang.

- [ ] **Step 1: Test (fail)**

Fixture tạo `ridge_bundle` bằng `fit_forecaster` trên dữ liệu ngẫu nhiên.
- `test_forecast_valid`.
- `test_forecast_empty_mask_no_prediction`.
- `test_forecast_rejects_bad_delta` (parametrize như Task 7).
- `test_forecast_tiny_mask_warns`.
- `test_run_inference_raises_on_failure` (`seg_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"]`).
- `test_run_inference_reads_outputs`: script giả ghi mask và CSV.
- `test_disclaimer_exact`.

- [ ] **Step 2: Code; pass; smoke Streamlit**

```bash
cd $SANDBOX && (NCKH_SEG_CMD=x NCKH_CLS_CMD=x NCKH_RIDGE=$S/fake_run/ridge.joblib NCKH_STABLE_EPS=0.005 timeout 15 $S/venv/bin/streamlit run demo/app.py --server.headless true --server.port 8599 &) ; sleep 8; curl -s -o /dev/null -w "%{http_code}\n" localhost:8599
```

Expected: `200`.

- [ ] **Step 3: Viết `P9_demo_streamlit.md`**

Bao gồm:
- Sơ đồ kiến trúc: Streamlit (Python của Colab) gọi subprocess `venv_seg` và `venv_cls`; lý do là xung đột phiên bản torch.
- Cell NB_cpu:
  - dựng 2 venv (link cell setup P2);
  - `pip install -e nckh[demo]`;
  - export các biến `NCKH_*`;
  - `nohup streamlit run demo/app.py --server.port 8501 &`;
  - tải `cloudflared-linux-amd64` từ `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64`;
  - chạy `./cloudflared tunnel --url http://localhost:8501` và đọc URL `trycloudflare.com` trong log.
- **Cảnh báo:** URL công khai, chỉ dùng ảnh ISIC, tắt tunnel sau khi dùng.
- **Chạy trên laptop:** chỉ dùng được khi laptop có thể dựng 2 venv; nếu không thì dùng Colab.
- **Bảng 10 bước kiểm thử end-to-end** của kế hoạch Phase 9, mỗi bước ánh xạ sang test hoặc thao tác tay.
- **Hướng dẫn chụp ảnh màn hình** không chứa dữ liệu hạn chế.
- **Benchmark:** thời gian từ upload đến kết quả, lần đầu và khi cache hit.
- Đánh dấu ⚠️ GPU.

- [ ] **Step 4: check_docs OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P9_demo_streamlit.md && git commit -m "docs(sv_b): add P9 Streamlit demo guide"
```

---

### Task 11: `P10_tai_lap_hinh_bang.md`

**Files:**
- Sandbox create: `scripts/make_tables.py`, `scripts/make_figures.py`, `tests/test_tables_figures.py`
- Create: `docs/sv_b/P10_tai_lap_hinh_bang.md`

**Interfaces:**
- Consumes: `seg_metrics.json` (Task 5), `cls_metrics.json` (Task 6), `forecast_metrics.json` (Task 8), `predictions_test.csv` và `flow.json` (Task 7), `robustness.json` (định dạng ở Task 9).
- Produces:
  - `make_tables(inputs: dict[str, Path | None], out_dir: Path) -> list[Path]`:
    - khóa của `inputs`: `seg, cls, forecast, flow, robustness`;
    - ghi `table_seg`, `table_cls`, `table_forecast`, `table_flow`, `table_robustness`, mỗi bảng có cả `.csv` và `.md`;
    - số định dạng 3 chữ số thập phân với CI dạng `[lo, hi]`;
    - file thiếu hoặc `None` → bảng ghi một dòng "chưa có dữ liệu" kèm `logging.warning`.
  - `make_figures(pred_csv: Path, robustness_json: Path | None, out_dir: Path) -> list[Path]`: `pred_vs_obs.png` (có đường y = x), `residual_vs_dt.png`, `robustness.png` (nếu có); `matplotlib.use("Agg")`; dpi 200.

- [ ] **Step 1: Test (fail)**

- `test_tables_from_fake_jsons`: dùng JSON nhỏ viết tay; `table_forecast.md` chứa chuỗi `"[0.010, 0.020]"`.
- `test_tables_missing_json_marked`.
- `test_figures_created`.

- [ ] **Step 2: Code; pass; chạy trên `$S/fake_run`**

- [ ] **Step 3: Viết `P10_tai_lap_hinh_bang.md`**

Bao gồm:
- Quy tắc run card và không ghi đè; cây `runs/`.
- Bảng truy vết: mỗi bảng hoặc hình → script → file đầu vào → run_id (8 hình/bảng của kế hoạch Phase 10).
- Cell chọn ví dụ lỗi định trước: tốt, trung bình, kém theo các phân vị 90, 50, 10 của Dice (seg) và |residual| (forecast), lấy 3 ví dụ mỗi nhóm với seed 2026.
- Quy trình tái lập cuối (5 bước của kế hoạch Mục 7) cho SV A chạy từ notebook sạch.
- Checklist trước khi nộp phần của SV B.
- Code và test.

- [ ] **Step 4: check_docs OK; commit**

```bash
cd $REPO && git add -f docs/sv_b/P10_tai_lap_hinh_bang.md && git commit -m "docs(sv_b): add P10 reproducibility, tables and figures guide"
```

---

### Task 12: Rà soát chéo toàn bộ docs và ghi chú docs cũ

**Files:**
- Modify: `docs/Huong_dan_SV_B_Tuan_1_Colab.md`, `docs/Huong_dan_SV_B_Tuan_1_Colab_vs.md`, `docs/Huong_dan_SV_B_Tuan_1_PyCharm_Colab.md` (chỉ chèn 1 dòng ở đầu file)
- Modify (nếu lệch): bất kỳ `docs/sv_b/*.md` nào

- [ ] **Step 1: Chạy toàn bộ pytest và check_docs**

```bash
cd $SANDBOX && $S/venv/bin/pytest -q
SANDBOX=$SANDBOX $S/venv/bin/python $S/check_docs.py $REPO/docs/sv_b/*.md
```

Expected: tất cả test pass; `OK`.

- [ ] **Step 2: Rà nhất quán**

```bash
cd $REPO/docs/sv_b
grep -n "TODO\|TBD\|XXX" *.md                       # Expected: rỗng
ls | wc -l                                           # Expected: 11
grep -L "## 8. Checklist bàn giao cho SV A" P*.md   # Expected: chỉ P0_P1_P4_P11_vai_tro_ho_tro.md
```

Grep từng tên hàm trong khối Interfaces của Task 3–11 (`select_alpha`, `build_consecutive_pairs`, `build_feature_table`, `evaluate_forecast`, `run_inference`…). Mọi chỗ gọi trong cell `# cell:` phải khớp tên và tham số với code trong sandbox. Mọi khối `# cell:` phải dùng đúng runtime theo bảng spec 3.3.

- [ ] **Step 3: Chèn ghi chú vào 3 docs cũ**

Dòng đầu mỗi file (nếu là `Huong_dan_SV_B_Tuan_1_Colab_vs.md` thì đường dẫn tương đối giữ nguyên):

```
> ⚠️ Tài liệu này đã lỗi thời (smoke test giả bằng torch.rand, dùng ISIC 2017 thay vì ISIC 2018, link checkpoint sai). Dùng [sv_b/P2_moi_truong_checkpoint_smoke_test.md](sv_b/P2_moi_truong_checkpoint_smoke_test.md).
```

- [ ] **Step 4: Commit và kiểm workspace**

```bash
cd $REPO && git add -f docs/sv_b docs/Huong_dan_SV_B_Tuan_1_Colab.md docs/Huong_dan_SV_B_Tuan_1_Colab_vs.md docs/Huong_dan_SV_B_Tuan_1_PyCharm_Colab.md
git commit -m "docs(sv_b): cross-check guides and mark week-1 guides obsolete"
git status --short
```

Expected: chỉ còn 3 thay đổi dở của người dùng (`.idea/nckh.iml`, `nckh.ipynb`, `requirements_colab.txt`).
