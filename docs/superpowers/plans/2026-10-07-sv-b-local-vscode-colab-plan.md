# SV B guides → Local VS Code + Colab extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sửa 11 file `docs/sv_b/*.md` để mọi việc không cần GPU chạy trên máy cá nhân trong VS Code, việc train/chạy model chạy trên kernel Colab nối qua extension VS Code, và mỗi file được ghi rõ tạo ở đâu.

**Architecture:** Chỉ sửa Markdown. Định nghĩa một bộ "khối chuẩn" (cell setup máy, cell setup Colab, lệnh rclone, dòng 📁) ở mục *Shared Blocks* bên dưới; mỗi task chép nguyên văn các khối đó vào đúng chỗ và áp quy tắc thay thế đường dẫn. Kiểm tra bằng `grep` (đóng vai trò test).

**Tech Stack:** Markdown, VS Code + extension `google.colab`, `uv`, `rclone`, git.

**Spec:** `docs/superpowers/specs/2026-10-07-sv-b-local-vscode-colab-design.md` (đọc trước khi làm).

## Global Constraints

- Chỉ sửa `docs/sv_b/*.md` (+ một chỗ trong spec ở Task 8). Không sửa `src/`, `tests/`, `notebooks/`, `pyproject.toml`, `docs/Huong_dan_SV_B_Tuan_1_*.md`, `Ke_hoach_*.md`, `De_cuong_*.docx`.
- **Không đổi nội dung bên trong khối `# file:`** (50 khối). Chỉ thêm dòng 📁 phía trên và sửa chữ mô tả quanh khối.
- Không đổi nội dung kỹ thuật: phiên bản torch/mmcv/timm, metric, split, seed, protocol, số test kỳ vọng.
- Tiếng Việt, giọng văn và định dạng giữ như file hiện có. Đường dẫn máy dùng `~/nckh_drive`, `~/nckh_data`, `<repo>` (= thư mục clone repo, ví dụ `~/Documents/nckh`). Đường dẫn Colab: `/content/nckh`, `/content/data`, `/content/drive/MyDrive/NCKH_PanDerm`.
- Remote rclone tên `gdrive`; thư mục Drive `NCKH_PanDerm`.
- Nhãn cell: `# cell: NB_cpu (máy cá nhân)`, `# cell: NB_seg (Colab GPU)`, `# cell: NB_cls (Colab GPU)`. Lệnh terminal ở máy: khối ```bash dòng đầu `# terminal VS Code (máy cá nhân)`.
- Mục 7 mọi file phase: tiêu đề `## 7. Lỗi thường gặp (máy / Colab extension)`.
- Chỗ nào mô tả menu/hành vi extension chưa chạy thử: thêm `> ⚠️ Chưa kiểm chứng trên extension — xác minh khi chạy P2.`
- Không còn hướng dẫn mở colab.research.google.com, Colab Secrets, `GH_TOKEN`, `AUTH_URL`, `userdata`, repo trên Drive (trừ trong bảng lỗi để giải thích vì sao không dùng).

## Review Focus

1. Cell Colab còn sót `{ROOT}/nckh` → trên Colab code ở `/content/nckh`, sót là `No such file`. Pin: grep ở bước kiểm tra mỗi task.
2. Cell máy dùng `!python`/`!pip` → có thể gọi nhầm Python hệ thống thay vì `.venv`. Pin: cell máy dùng `{PY}` / `%pip`; Task 9 Step 4.
3. Kết quả sinh trên Colab nhưng bước kế tiếp chạy ở máy mà thiếu `rclone copy` kéo về → file không có ở `~/nckh_drive`. Pin: mỗi chuyển Colab→máy trong P2/P5a/P5b/P6/P8 có khối rclone ngay trước cell máy.
4. Manifest/CSV tạo ở máy nhưng cell Colab đọc từ Drive mà thiếu `rclone copy` đẩy lên → Colab đọc file cũ/không có. Pin: P3, P6, P8 có khối đẩy lên ngay sau khi tạo.
5. Khối `# file:` thiếu dòng 📁 hoặc nội dung khối bị sửa. Pin: đếm 📁 = 50; Task 9 Step 5 so khối `# file:` với commit `d77cd5f` (trước khi sửa doc).

---

## Shared Blocks (chép nguyên văn khi task yêu cầu)

### SB1 — Cell setup máy (đầu `NB_cpu`, định nghĩa ở P2 4.1a)

```python
# cell: NB_cpu (máy cá nhân)
# Kernel: chọn .venv của repo (Select Kernel → Python Environments → .venv).
import os, sys
from pathlib import Path
REPO = Path.home() / 'Documents' / 'nckh'      # sửa nếu bạn clone repo ở chỗ khác
ROOT = str(Path.home() / 'nckh_drive')          # bản sao MyDrive/NCKH_PanDerm, đồng bộ bằng rclone
DATA = str(Path.home() / 'nckh_data')           # ảnh ISIC giải nén trên máy
PY = sys.executable                             # Python của .venv, dùng trong lệnh "!"
os.environ['NCKH_ROOT'] = ROOT
os.environ['NCKH_LOCAL_DATA'] = DATA
%cd {REPO}
!git log --oneline -1
```

### SB2 — Cell setup Colab (đầu `NB_seg`/`NB_cls`, định nghĩa ở P2 4.1b)

Trước khi chạy: *Select Kernel → Colab → New Colab Server* → chọn GPU (T4). Sau đó `Ctrl+Shift+P` → **Colab: Mount Google Drive to Server...** (extension chèn một cell mount; chạy cell đó và làm theo hướng dẫn đăng nhập).

```python
# cell: NB_seg (Colab GPU)
import os, sys
from pathlib import Path
ROOT = '/content/drive/MyDrive/NCKH_PanDerm'   # Drive đã mount bằng lệnh extension
CODE = '/content/nckh'                          # bản clone chỉ đọc; sửa code ở máy rồi push
assert os.path.isdir(ROOT), 'Chưa mount Drive: Ctrl+Shift+P → Colab: Mount Google Drive to Server...'
os.environ['NCKH_ROOT'] = ROOT
os.environ['NCKH_LOCAL_DATA'] = '/content/data'
!test -d {CODE}/.git || git clone -q https://github.com/NeuroDev204/nckh.git {CODE}
!git -C {CODE} pull -q --ff-only && git -C {CODE} log --oneline -1
sys.path.insert(0, f'{CODE}/src')
```

Bản `NB_cls` giống hệt, chỉ đổi nhãn dòng đầu thành `# cell: NB_cls (Colab GPU)`.

### SB3 — Kéo kết quả từ Drive về máy

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id> --progress
```

### SB4 — Đẩy file tạo ở máy lên Drive

```bash
# terminal VS Code (máy cá nhân)
rclone copy ~/nckh_drive/data/manifests gdrive:NCKH_PanDerm/data/manifests --progress
```

### SB5 — Dòng vị trí trên mỗi khối `# file:`

Ngay trên khối code (sau đoạn mô tả, ngay trước dòng mở ```` ``` ````), thêm một dòng; đường dẫn lấy từ dòng `# file:` của khối:

```markdown
📁 **Tạo trên máy cá nhân:** `<repo>/src/nckh/paths.py`
```

### SB6 — Quy tắc thay thế trong cell

| Trong cell… | Thay | Bằng |
|---|---|---|
| Colab (`NB_seg`/`NB_cls`) | `{ROOT}/nckh` | `{CODE}` |
| Máy (`NB_cpu`) | `{ROOT}/nckh` | `{REPO}` |
| Máy | `python ` sau `!` hoặc `&&` | `{PY} ` |
| Máy | `!pip install` | `%pip install` |
| Máy | `/content/data` | `{DATA}` |
| Máy | `/content/<tạm>` (log, csv tạm) | `/tmp/<tạm>` |
| Mọi cell | `# cell: NB_x` | nhãn đầy đủ theo Global Constraints |

`{ROOT}/runs/...`, `{ROOT}/data/...`, `{ROOT}/checkpoints/...` **giữ nguyên** ở cả hai nơi (máy: `~/nckh_drive`, Colab: Drive).

### SB7 — Hàng lỗi cho mục 7

Hàng chung (mọi file phase có cell Colab):

```markdown
| `AssertionError: Chưa mount Drive` | Server Colab mới chưa mount | `Ctrl+Shift+P` → *Colab: Mount Google Drive to Server...*, chạy lại cell setup |
| Kernel Colab mất kết nối / server bị thu hồi | Hết thời gian phiên miễn phí hoặc mạng | *Select Kernel → Colab → New Colab Server*, chạy lại cell setup + venv; dữ liệu trên Drive vẫn còn |
| Colab chạy code cũ | Quên `git push` ở máy trước khi chạy cell setup | Ở máy `git push`, chạy lại cell setup (có `git pull`) |
| `userdata.get` / `files.upload` lỗi | Chưa hỗ trợ trong extension | Không dùng; repo public nên không cần token, file đi qua Drive |
```

Hàng rclone (mọi file phase có lệnh rclone):

```markdown
| `rclone` báo `couldn't fetch token` | Token OAuth hết hạn | `rclone config reconnect gdrive:` |
| Ở máy không thấy file Colab vừa ghi | Chưa kéo về | Chạy lệnh `rclone copy gdrive:NCKH_PanDerm/... ~/nckh_drive/...` |
```

---

### Task 1: `00_tong_quan_va_lo_trinh.md`

**Files:**
- Modify: `docs/sv_b/00_tong_quan_va_lo_trinh.md` (dòng 5, bảng 25–39, mục 4 dòng 68–95, mục 5 dòng 97–129, mục 6 dòng 131–141, mục 7 dòng 145, mục 8 dòng 152–174, mục 9 dòng 181–186)

**Interfaces:**
- Produces: tên biến `REPO`, `ROOT`, `DATA`, `PY`, `CODE`; remote `gdrive`; tên mục 7 mới; bảng phân phase Máy/Colab — mọi task sau dựa vào.

- [ ] **Step 1: Chạy kiểm tra trước (phải thấy vi phạm)**

Run: `grep -nE "GH_TOKEN|Secrets|Push/pull ngay trong Colab|Ba runtime Colab|Lỗi thường gặp trên Colab" docs/sv_b/00_tong_quan_va_lo_trinh.md`
Expected: ≥ 4 dòng.

- [ ] **Step 2: Dòng 5** thay bằng:
`> Code, git, dữ liệu, test, đánh giá và demo chạy **trên máy cá nhân trong VS Code**. Việc train/chạy model chạy trên **kernel Colab nối từ VS Code bằng extension Google Colab** (không dùng bản web). **Google Drive** là nơi trung chuyển file bền giữa hai bên, đồng bộ bằng `rclone`.`

- [ ] **Step 3: Bảng lộ trình:** đổi tiêu đề cột `Runtime` → `Chạy ở`. Giá trị: P0 `—`; P1 `Máy`; P2 `Máy + Colab (NB_seg, NB_cls)`; P3 `Máy (+ Colab NB_cls tải ISIC 2017)`; P4 `Máy`; P5a `Colab NB_seg`; P5b `Colab NB_cls`; P6 `Colab (suy luận) + Máy (ghép cặp, Ridge)`; P7 `Máy`; P8 `Colab (suy giảm + suy luận) + Máy (forecast)`; P9 `Máy (chế độ giả) + Colab NB_seg (model thật)`; P10 `Máy`; P11 `—`.

- [ ] **Step 4: Mục 4** đổi tiêu đề thành `## 4. Ba nơi lưu file` và thay cây Drive bằng:

````markdown
```
<repo>/                         ← MÁY: git clone repo (CODE). Sửa, commit, push tại đây
~/nckh_data/                    ← MÁY: ảnh ISIC giải nén (NCKH_LOCAL_DATA), tải một lần
~/nckh_drive/                   ← MÁY: bản sao một phần của Drive (NCKH_ROOT), đồng bộ bằng rclone
MyDrive/NCKH_PanDerm/           ← DRIVE: cầu nối máy ↔ Colab
├── data/
│   ├── zips/                   ← CHỈ file GT nhỏ (mask zip, CSV nhãn). KHÔNG để zip ảnh
│   ├── manifests/              ← manifest, split, hash (tạo ở máy, đẩy lên bằng rclone)
│   └── uq/                     ← chỉ khi đã có văn bản cho phép
├── checkpoints/                ← panderm_bb_data6_checkpoint-499.pth + SHA256SUMS (tải trên Colab)
└── runs/<run_id>/              ← log, run card, checkpoint fine-tune, prediction (Colab ghi, máy kéo về)
/content/  (COLAB, mất khi server bị thu hồi)
├── nckh/                       ← git clone chỉ đọc, pull mỗi phiên
├── data/                       ← ảnh ISIC tải từ S3 mỗi phiên
├── PanDerm/                    ← upstream fd7a807
└── venv_seg/, venv_cls/        ← venv GPU, dựng lại mỗi phiên
```
````

Giữ bảng dung lượng ISIC. Thay đoạn "**Quy tắc:**" bằng: `→ Tổng ≈ 27 GB, vượt 15 GB Drive miễn phí. **Quy tắc:** ảnh không bao giờ lên Drive. Máy tải một lần vào ~/nckh_data (dùng cho P3: manifest, kiểm tra ảnh). Colab tải trực tiếp từ S3 về /content/data mỗi phiên (~1–3 phút/GB) để train/suy luận. Hai nơi tải cùng zip nên manifest (lưu **đường dẫn tương đối** so với data_root + SHA-256) dùng được ở cả hai.` Giữ câu về FUSE.

- [ ] **Step 5: Mục 5** thay toàn bộ (5.1–5.3) bằng:

````markdown
## 5. Đồng bộ: code bằng git, file bằng rclone

### 5.1. Code (git chỉ chạy ở máy)

Repo `NeuroDev204/nckh` để public, nên Colab clone/pull không cần token. Mọi `git add/commit/push` làm trong VS Code ở máy. Trên Colab, `/content/nckh` chỉ đọc: muốn sửa code thì sửa ở máy, push, rồi chạy lại cell setup Colab (có `git pull`).

```bash
# terminal VS Code (máy cá nhân)
cd <repo>
git pull --rebase
git add src scripts tests configs demo notebooks patches pyproject.toml .gitignore
git commit -m "mô tả ngắn thay đổi" && git push
```

### 5.2. File (rclone giữa ~/nckh_drive và Drive)

(chép khối SB3 nguyên văn)

(chép khối SB4 nguyên văn)

`rclone copy` chỉ thêm/ghi đè, không xoá ở đích. Kéo theo từng `runs/<run_id>` thay vì cả `runs/` để không tải checkpoint không cần.

### 5.3. Ba quy tắc tránh xung đột

1. **Push trước khi chạy Colab.** Colab chỉ thấy code đã push.
2. **Colab không chạy `git commit/push`** và không sửa file trong `/content/nckh`.
3. **Không `git add -A`/`git add .`**: luôn chỉ rõ thư mục code như trên, để không bao giờ lỡ đưa ảnh hay checkpoint lên.
````

- [ ] **Step 6: Mục 6** đổi tiêu đề `## 6. Môi trường: máy và Colab` và thay bảng bằng (cột "Cài gì" của NB_seg/NB_cls chép nguyên từ bảng hiện tại):

```markdown
| Notebook | Chạy ở | Python | Dùng cho | Cài gì |
|---|---|---|---|---|
| `notebooks/NB_cpu.ipynb` | Máy, kernel `<repo>/.venv` | 3.10 (`uv`) | P1, P3, P4, P6 (ghép cặp, Ridge), P7, P8 (forecast), P9 (chế độ giả), P10, pytest | `-e .[demo,test]`, `ipykernel`, torch 2.4.1 **CPU** (cho `test_infer.py`, `test_bench.py`) |
| `notebooks/NB_seg.ipynb` | Colab GPU qua extension | venv 3.10 `/content/venv_seg` | P2, P5a, P6 (mask), P8, P9 (model thật) | torch 2.1.2 + torchvision 0.16.2 (cu118), mmengine 0.10.4, wheel mmcv 2.1.0, mmsegmentation 1.2.2, `segmentation/requirements.txt`, `openpyxl`, `-e nckh`. Lý do dùng torch 2.1.2 thay vì 2.2.1 của README: P2 mục 3.2 |
| `notebooks/NB_cls.ipynb` | Colab GPU qua extension | venv 3.10 `/content/venv_cls` | P2, P5b, P6 (xác suất), P8 | torch 2.4.1 + torchvision 0.19.1 + torchaudio 2.4.1 (cu118), `classification/requirements.txt`, `timm==0.9.16`, `-e nckh` |
```

Giữ câu "Package `nckh` … không phụ thuộc torch …"; sửa câu cuối thành: `Venv GPU tạo lại mỗi phiên trên /content của Colab; .venv ở máy tạo một lần.`

- [ ] **Step 7: Thêm mục mới ngay sau mục 6:** `## 6b. Cài đặt máy cá nhân (một lần)`:

````markdown
1. Cài VS Code và 3 extension: **Python** (`ms-python.python`), **Jupyter** (`ms-toolsai.jupyter`), **Colab** (`google.colab`). Lần đầu chọn kernel Colab sẽ yêu cầu đăng nhập Google (dùng tài khoản có Drive `NCKH_PanDerm`).
2. Cài công cụ và tạo môi trường:

```bash
# terminal VS Code (máy cá nhân)
sudo apt install -y git rclone
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/NeuroDev204/nckh.git ~/Documents/nckh   # = <repo>
cd ~/Documents/nckh
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[demo,test]" ipykernel
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
mkdir -p ~/nckh_drive ~/nckh_data
```

3. Nối rclone với Drive: `rclone config` → `n` (new remote) → name `gdrive` → storage `drive` → để trống client_id/secret → scope `1` (full access) → `y` mở trình duyệt đăng nhập → không cấu hình Shared Drive. Kiểm tra: `rclone lsd gdrive:NCKH_PanDerm`.
4. Trong VS Code: *File → Open Folder* → `<repo>`. Mở `notebooks/NB_cpu.ipynb` → *Select Kernel → Python Environments → .venv*.
5. Biến môi trường cho lệnh chạy trong terminal (notebook đã tự đặt trong cell setup P2 4.1a). 📁 **Tạo trên máy cá nhân:** `<repo>/.env` (đã có trong `.gitignore`, không commit):

```bash
# terminal VS Code (máy cá nhân)
cd <repo>
printf 'NCKH_ROOT=%s\nNCKH_LOCAL_DATA=%s\n' "$HOME/nckh_drive" "$HOME/nckh_data" > .env
set -a && source .env && set +a     # chạy mỗi lần mở terminal mới
```

> Lần đầu, `pyproject.toml` chưa có (tạo ở P2 mục 4.2): chỉ chạy `uv pip install ipykernel`, chạy lại dòng `-e ".[demo,test]"` sau khi tạo file.
````

- [ ] **Step 8: Mục 7 dòng 145** (UQ): giữ nguyên ý, thêm câu cuối: `Bản sao trên máy (~/nckh_drive/data/uq) chỉ được giữ khi điều khoản cho phép lưu trên máy cá nhân.`

- [ ] **Step 9: Mục 8** thêm cột `Vị trí tạo` vào bảng: mỗi dòng ghi `Máy: <repo>/<đường dẫn>` cho từng file của dòng đó (ví dụ `Máy: <repo>/src/nckh/paths.py, <repo>/src/nckh/runcard.py`). Dòng notebooks: `Máy: <repo>/notebooks/ (NB_seg/NB_cls mở ở máy, kernel chạy trên Colab)`. Ngay dưới bảng thêm cây:

````markdown
```
<repo>/
├── pyproject.toml  .gitignore  .env (không commit)
├── src/nckh/   __init__.py paths.py runcard.py infer.py isic.py manifest.py metrics.py
│               pairs.py features.py forecast.py degrade.py
├── scripts/    inspect_checkpoint.py bench_inference.py prepare_isic2018.py prepare_isic2017_cls.py
│               build_manifest.py evaluate_seg.py evaluate_cls.py annotator_agreement.py make_fake_uq.py
│               build_pairs.py infer_images.py train_ridge.py evaluate_forecast.py make_degraded.py
│               evaluate_robustness.py make_tables.py make_figures.py
├── configs/    uq_column_map.yaml
├── patches/    panderm_base_seg.patch
├── demo/       pipeline.py app.py
├── tests/      test_*.py
└── notebooks/  NB_cpu.ipynb NB_seg.ipynb NB_cls.ipynb
```
````

Đối chiếu cây với bảng mục 8 và với `grep -h "^# file:" docs/sv_b/*.md | sort -u`: mọi đường dẫn phải có trong cây; thêm vào cây nếu thiếu.

- [ ] **Step 10: Mục 9:** khung phase đổi `7. Lỗi thường gặp trên Colab` → `7. Lỗi thường gặp (máy / Colab extension)`. Danh sách ký hiệu: thay dòng `# cell: NB_seg / NB_cls / NB_cpu` bằng `# cell: NB_cpu (máy cá nhân)` / `# cell: NB_seg (Colab GPU)` / `# cell: NB_cls (Colab GPU)` — cell dán vào notebook tương ứng; thêm `📁 **Tạo trên máy cá nhân:** <repo>/…` — vị trí file bạn tạo; `# terminal VS Code (máy cá nhân)` — lệnh chạy trong terminal ở máy; `> ⚠️ Chưa kiểm chứng trên extension` — mô tả menu extension chưa chạy thử.

- [ ] **Step 11: Kiểm tra**

Run: `grep -nE "GH_TOKEN|Secrets|AUTH_URL|Push/pull ngay trong Colab|Ba runtime Colab|Lỗi thường gặp trên Colab|Mọi thứ chạy trên" docs/sv_b/00_tong_quan_va_lo_trinh.md`
Expected: không có dòng nào.
Run: `grep -c "Vị trí tạo\|nckh_drive\|google.colab\|rclone" docs/sv_b/00_tong_quan_va_lo_trinh.md`
Expected: ≥ 6.

- [ ] **Step 12: Commit**

```bash
git add docs/sv_b/00_tong_quan_va_lo_trinh.md
git commit -m "docs(sv_b): 00 overview for local VS Code + Colab extension"
```

---

### Task 2: `P2_moi_truong_checkpoint_smoke_test.md`

**Files:**
- Modify: `docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md` (mục 1 dòng 9–13, mục 2 dòng 15–24, 3.1 dòng 28–34, 4.1 dòng 93–134, 4.2 dòng 136–138 + 12 khối `# file:`, cell 418–420, 4.3 dòng 424–442, cell `NB_seg`/`NB_cls` 449–1240, 4.12 dòng 1265–1274, mục 5 dòng 1276–1293, mục 7 dòng 1306–1320)

**Interfaces:**
- Consumes: SB1–SB7; tên biến từ Task 1.
- Produces: P2 4.1a = SB1, P2 4.1b = SB2 — các phase sau viết "chạy cell setup P2 4.1a/4.1b".

- [ ] **Step 1: Kiểm tra trước**

Run: `grep -nE "userdata|AUTH_URL|GH_TOKEN|Save a copy in Drive|\{ROOT\}/nckh" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md | wc -l`
Expected: ≥ 15.

- [ ] **Step 2: Mục 1** đổi tiêu đề cột `Đầu ra (trên Drive)` → `Đầu ra (vị trí)`. Dòng 1: `Máy: <repo>/ có src/nckh/, tests/, scripts/, patches/, .gitignore, notebooks/NB_*.ipynb`. Dòng 2: `Drive (tải trên Colab): MyDrive/NCKH_PanDerm/checkpoints/panderm_bb_data6_checkpoint-499.pth + SHA256SUMS`. Dòng 3: `Drive (Colab ghi): MyDrive/NCKH_PanDerm/runs/<run_id>/bench_seg.json, bench_cls.json, overlays/*.png, cls_probs_smoke.csv, run_card.json → kéo về ~/nckh_drive/runs/<run_id>/ bằng rclone`.

- [ ] **Step 3: Mục 2** thay bảng bằng:

```markdown
| Việc | Chạy ở | Phần cứng | Thời gian ước tính |
|---|---|---|---|
| Tạo file code, chạy pytest | Máy (`NB_cpu` / terminal VS Code) | CPU | 15–30 phút lần đầu |
| Tải checkpoint lên Drive | Colab `NB_seg` | — | 2–5 phút |
| Dựng `venv_seg`, soi checkpoint, áp patch, smoke test seg | Colab `NB_seg` | GPU (T4 trở lên) | 10–15 phút cài + 2 phút chạy |
| Dựng `venv_cls`, smoke test cls | Colab `NB_cls` | GPU | 8–12 phút cài + 1 phút chạy |
| Kéo `runs/<run_id>` về máy, điền bảng mục 6 | Máy (terminal) | — | 1 phút |
```

Thay đoạn "**Tạo 3 notebook một lần:**…" + dòng "Với `NB_seg`…" bằng:

```markdown
**Tạo 3 notebook một lần (ở máy):** trong VS Code, `Ctrl+Shift+P` → *Create: New Jupyter Notebook*, lưu thành `<repo>/notebooks/NB_cpu.ipynb`, `<repo>/notebooks/NB_seg.ipynb`, `<repo>/notebooks/NB_cls.ipynb`. Notebook nằm trong repo nên commit như code.
- `NB_cpu`: *Select Kernel → Python Environments → .venv*.
- `NB_seg`, `NB_cls`: *Select Kernel → Colab → New Colab Server* → chọn GPU (T4). Mỗi notebook một server riêng để hai venv không đè nhau.

> ⚠️ Chưa kiểm chứng trên extension — xác minh khi chạy P2 (tên menu chọn GPU có thể khác theo phiên bản extension).
```

- [ ] **Step 4: 3.1** câu đầu đổi `Kernel của notebook Colab luôn là…` → `Kernel Colab (kể cả khi nối qua extension VS Code) luôn là Python mặc định của Colab (3.11/3.12).` Gạch đầu dòng 1 đổi thành `Cell Python thường (đặt biến, git pull, tạo run_id) chạy trong kernel.`

- [ ] **Step 5: 4.1** giữ tiêu đề `### 4.1. Cell mở đầu của 3 notebook`. Xoá 2 đoạn dòng 95–97 (quy tắc git trên Drive, token). Viết:
  - `**4.1a. NB_cpu (máy)** — chạy đầu mỗi phiên:` + SB1 nguyên văn.
  - `**4.1b. NB_seg và NB_cls (Colab)** — chạy đầu mỗi phiên, sau khi push code ở máy:` + đoạn hướng dẫn chọn kernel/mount của SB2 + khối SB2 nguyên văn + câu "Bản NB_cls…".
  - Thay ghi chú dòng 134 bằng: `> Lần đầu repo chưa có src/: cell vẫn chạy, dòng sys.path không lỗi. Nếu trước đây đã clone repo vào MyDrive/NCKH_PanDerm/nckh, có thể xoá thư mục đó để giải phóng Drive; nếu .git/config cũ từng chứa token, thu hồi token đó trên GitHub.`

- [ ] **Step 6: 4.2** tiêu đề `### 4.2. Tạo package nckh (ở máy)`. Thay câu dòng 138 bằng: `Tạo các file dưới đây **ở máy**, trong VS Code (*Explorer → New File*), dưới <repo>/. Khối nào bắt đầu bằng # file: thì chép nguyên văn, kể cả dòng # file: đầu tiên (dòng này là comment, vô hại).` Áp SB5 cho **mọi** khối `# file:` trong P2 (12 khối), kể cả các khối ở 4.5–4.10.

- [ ] **Step 7: Cell dòng 418–420** (cài + pytest) thay bằng:

```python
# cell: NB_cpu (máy cá nhân)
%pip install -q -e "{REPO}[demo,test]"
!cd {REPO} && {PY} -m pytest -q tests/test_paths_runcard.py
```

Câu trên đổi `Cài và chạy test trong NB_cpu:` → `Cài và chạy test ở máy (NB_cpu):`.

- [ ] **Step 8: 4.3** tiêu đề `### 4.3. Tải checkpoint PanDerm_Base (Colab NB_seg, lưu lên Drive)`. Cell đổi nhãn `# cell: NB_seg (Colab GPU)`; thêm dòng chú thích thứ hai `# Chạy sau cell setup 4.1b. Tải trên Colab vì mạng Colab ↔ Drive nhanh, không tốn băng thông máy.`; phần còn lại giữ nguyên. Đoạn "Nếu gdown báo vượt quota…" giữ, đổi "`cp` từ `MyDrive`" → "`!cp` trong cell Colab từ `/content/drive/MyDrive/<tên file>`".

- [ ] **Step 9: Mọi cell `NB_seg`/`NB_cls` từ 4.4 đến 4.10** (dòng 449–1240): đổi nhãn đầy đủ, áp SB6 (`{ROOT}/nckh` → `{CODE}`, ví dụ `-e {ROOT}/nckh` → `-e {CODE}`). Câu "Venv nằm trên /content…" dòng 468: giữ, thêm "(server Colab bị thu hồi khi ngắt kết nối lâu)".

- [ ] **Step 10: Cuối 4.10** thêm:

````markdown
Kéo kết quả về máy để điền mục 6:

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<run_id> ~/nckh_drive/runs/<run_id> --progress
```
````

- [ ] **Step 11: 4.12** thay cell commit bằng:

````markdown
```bash
# terminal VS Code (máy cá nhân)
cd <repo>
git add pyproject.toml .gitignore src tests scripts patches notebooks
git status --short
git commit -m "P2: package nckh, infer, smoke test, patch PanDerm Base" && git push && git log --oneline -1
```
````

- [ ] **Step 12: Mục 5** cột `Chạy ở`: cả 4 dòng = `Máy (NB_cpu)`; dòng `test_infer.py` đổi `(cần torch; Colab có sẵn)` → `(cần torch CPU trong .venv — 00 mục 6b)`. Cell pytest:

```python
# cell: NB_cpu (máy cá nhân)
!cd {REPO} && {PY} -m pytest -q
```

Câu "Nếu kernel không có torch…" đổi "kernel" → ".venv".

- [ ] **Step 13: Mục 7** đổi tiêu đề theo Global Constraints. Xoá hàng `git push/pull hỏi mật khẩu …Colab Secrets…`. Thêm 4 hàng SB7 chung + 2 hàng rclone. Các hàng còn lại giữ.

- [ ] **Step 14: Kiểm tra**

Run: `grep -nE "userdata|AUTH_URL|GH_TOKEN|Secrets|Save a copy in Drive|File → New notebook|\{ROOT\}/nckh" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md`
Expected: chỉ hàng `userdata.get` trong bảng mục 7.
Run: `echo $(grep -c "^# file:" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md)`
Expected: `12 12`.
Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md`
Expected: không có.

- [ ] **Step 15: Commit**

```bash
git add docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md
git commit -m "docs(sv_b): P2 local setup + Colab extension kernels"
```

---

### Task 3: `P3_du_lieu_manifest_split.md`

**Files:**
- Modify: `docs/sv_b/P3_du_lieu_manifest_split.md` (mục 1 dòng 7–12, mục 2 dòng 14–23, 3.1 dòng 27–32, 8 khối `# file:`, 4.6 dòng 686–719, 4.7 dòng 721–738, mục 5 dòng 740–760, mục 7 dòng 779–787)

**Interfaces:**
- Consumes: SB1 (`REPO`, `ROOT`, `DATA`, `PY`), SB2 (`CODE`), SB4–SB7.
- Produces: manifest ở `~/nckh_drive/data/manifests/` và trên Drive — P5a/P6/P8 đọc `{ROOT}/data/manifests`.

- [ ] **Step 1: Kiểm tra trước**

Run: `grep -nE "^# cell: NB_cpu$|/content/data|\{ROOT\}/nckh" docs/sv_b/P3_du_lieu_manifest_split.md | wc -l`
Expected: ≥ 8.

- [ ] **Step 2: Mục 1** bảng đổi cột thành `Đầu vào | Đầu ra | Sinh ở`:
  - ISIC 2018: `~/nckh_data/ISIC2018/{Training,Validation,Test}_{Data,GroundTruth}/` | `Máy (một lần); Colab tải lại vào /content/data/ISIC2018/ mỗi phiên ở P5a`.
  - ISIC 2017: `/content/data/ISIC2017/ISIC-2017_*_Data/*.jpg` | `Colab NB_cls (mỗi phiên)`.
  - Manifest: `~/nckh_drive/data/manifests/isic2018_seg.csv` + `.sha256` + `.cross_split_duplicates.csv`, `mask_count.json` | `Máy → đẩy lên Drive bằng rclone`; `isic2017_cls.csv`, `isic2017_cls_trainphase.csv` | `Colab NB_cls ghi lên Drive → kéo về máy`.
  - UQ: giữ, cột Sinh ở `Máy`.

- [ ] **Step 3: Mục 2** thay bảng:

```markdown
| Việc | Chạy ở | Thời gian ước tính |
|---|---|---|
| Tạo `nckh/isic.py`, `nckh/manifest.py`, scripts, test; chạy pytest | Máy (`NB_cpu`) | 20 phút |
| Tải + giải nén ISIC 2018 (~14 GB zip) vào `~/nckh_data` | Máy (`NB_cpu`), một lần | tùy mạng |
| Tạo manifest ISIC 2018 (băm 3.694 ảnh), đẩy lên Drive | Máy (`NB_cpu` + terminal) | 3–6 phút |
| Tải + giải nén ISIC 2017 (~13 GB zip), CSV nhãn lên Drive | Colab `NB_cls` (vì P5b chạy ở đó) | 10–25 phút mỗi phiên |
```

Câu dưới bảng: `Máy cần ~30 GB trống cho ~/nckh_data (zip bị xoá sau khi giải nén nhờ --delete-zips). Đĩa /content của Colab ~80–100 GB.`

- [ ] **Step 4: 3.1** gạch "**Ảnh**…" thay: `**Ảnh**: máy tải một lần từ S3 của ISIC về ~/nckh_data (giữ lại). Colab tải lại về /content/data mỗi phiên khi cần train/suy luận. Script **idempotent** (đã giải nén đủ thì bỏ qua).` Gạch "**Manifest lưu đường dẫn tương đối**…": thay `Phiên sau ảnh vẫn nằm ở /content/data, manifest vẫn dùng được` → `Cùng manifest dùng được với ~/nckh_data ở máy và /content/data trên Colab`.

- [ ] **Step 5:** Áp SB5 cho 8 khối `# file:`.

- [ ] **Step 6: 4.6** thay cell ISIC 2018 (dòng 688–694) bằng:

```python
# cell: NB_cpu (máy cá nhân)
# ISIC 2018: tải + giải nén về ~/nckh_data (một lần), manifest + mask_count vào ~/nckh_drive.
!mkdir -p {ROOT}/data/manifests
!cd {REPO} && {PY} scripts/prepare_isic2018.py --zips-dir /tmp/nckh_zips --data-root {DATA} --delete-zips
!cp {DATA}/ISIC2018/mask_count.json {ROOT}/data/manifests/
!cd {REPO} && {PY} scripts/build_manifest.py --data-root {DATA} --out {ROOT}/data/manifests/isic2018_seg.csv
```

Ngay sau: `Đẩy manifest lên Drive để Colab dùng:` + SB4 nguyên văn.
Cell `NB_cls` (dòng 696–701): nhãn `# cell: NB_cls (Colab GPU)`, `{ROOT}/nckh` → `{CODE}`. Sau cell thêm `Kéo CSV nhãn ISIC 2017 về máy:` +

````markdown
```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/data/manifests ~/nckh_drive/data/manifests --progress
```
````

Cell "Kiểm tra nhanh manifest" (709–717): nhãn máy; nội dung giữ.

- [ ] **Step 7: 4.7** cell UQ: nhãn máy; `{ROOT}/nckh/configs` → `{REPO}/configs`. Bước 1 thêm: `rồi commit + push ở máy (00 mục 5.1).`

- [ ] **Step 8: Mục 5** cell pytest: nhãn máy, SB6.

- [ ] **Step 9: Mục 7** tiêu đề mới; thêm SB7 chung + 2 hàng rclone + hàng `| Manifest trên Colab là bản cũ | Quên đẩy sau khi tạo lại ở máy | Chạy lại rclone copy ~/nckh_drive/data/manifests gdrive:NCKH_PanDerm/data/manifests |`.

- [ ] **Step 10: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab" docs/sv_b/P3_du_lieu_manifest_split.md`
Expected: không có.
Run: `echo $(grep -c "^# file:" docs/sv_b/P3_du_lieu_manifest_split.md) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/P3_du_lieu_manifest_split.md)`
Expected: `8 8`.

- [ ] **Step 11: Commit**

```bash
git add docs/sv_b/P3_du_lieu_manifest_split.md
git commit -m "docs(sv_b): P3 data prep on local machine, rclone manifests"
```

---

### Task 4: `P5a_segmentation_isic2018.md` + `P5b_classification_isic2017.md`

**Files:**
- Modify: `docs/sv_b/P5a_segmentation_isic2018.md` (mục 1, mục 2 dòng 13–20, dòng 44, 3 khối `# file:`, cell dòng 344–470, mục 7 dòng 491+)
- Modify: `docs/sv_b/P5b_classification_isic2017.md` (mục 1, mục 2 dòng 13+, 3 khối `# file:`, cell dòng 258–320, mục 7 dòng 360+)

**Interfaces:**
- Consumes: SB1–SB7; manifest từ Task 3.
- Produces: `MyDrive/NCKH_PanDerm/runs/<seg_main_run_id>/0/model_best_0.ckpt`, `runs/<cls_main_run_id>/checkpoint-best.pth` — P6/P8/P9 dùng.

- [ ] **Step 1: Đọc** cell `NB_cpu` trong hai file (P5a dòng 344, 408; P5b dòng 258, 312). Với **từng** cell: nếu chỉ đọc file đã có trong `{ROOT}/runs`/`{ROOT}/data` và gọi script không cần torch/GPU (đánh giá trên mask/xác suất đã lưu, tạo run_id, pytest) → **máy**, chèn SB3 ngay trước cell nếu cell đọc `{ROOT}/runs/...`. Nếu cell gọi model hoặc venv GPU → đổi sang `NB_seg`/`NB_cls (Colab GPU)`. Câu dẫn trước cell ghi rõ: `Chạy ở máy sau khi kéo run về:` hoặc `Chạy trên Colab vì cần GPU:`.

- [ ] **Step 2: Mục 1** mỗi file: thêm cột `Sinh ở` (Colab ghi lên Drive / Máy) cho mọi đầu ra; đường dẫn đầy đủ dạng `MyDrive/NCKH_PanDerm/runs/<run_id>/...`.

- [ ] **Step 3: Mục 2** mỗi file: cột `Notebook` → `Chạy ở` (`Máy (NB_cpu)` / `Colab NB_seg` / `Colab NB_cls`). P5a dòng 19 "phiên Colab miễn phí" → "phiên server Colab miễn phí (qua extension)". Thêm dòng `| Kéo run về máy để đánh giá/điền bảng | Máy (terminal, rclone) | 1–3 phút |` (khớp số cột của bảng).

- [ ] **Step 4:** Mọi cell `NB_seg`/`NB_cls`: nhãn đầy đủ, `{ROOT}/nckh` → `{CODE}`. Câu chuẩn bị đầu phần cell (thay câu "pull code ở NB_cpu (P2 4.1a)" nếu có): `Đầu mỗi phiên Colab: push code ở máy, rồi chạy cell setup P2 4.1b, venv (P2 4.4 cho seg / 4.10 cho cls), patch (P2 4.6, chỉ P5a), tải ISIC vào /content/data (P3 4.6).`

- [ ] **Step 5:** Mọi cell `NB_cpu`: nhãn máy + SB6. Áp SB5 cho 3 + 3 khối `# file:`.

- [ ] **Step 6: P5a dòng 44** sau câu "Một phiên Colab miễn phí có thể bị ngắt…" thêm: `Khi nối qua extension, ngắt kết nối VS Code lâu cũng làm server bị thu hồi; checkpoint epoch nằm trên Drive nên resume vẫn được.`

- [ ] **Step 7: Mục 7** hai file: tiêu đề mới + SB7 chung + 2 hàng rclone.

- [ ] **Step 8: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab|P2 4\.1a\)" docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P5b_classification_isic2017.md`
Expected: không có.
Run: `for f in docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P5b_classification_isic2017.md; do echo $(grep -c "^# file:" $f) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" $f); done`
Expected: `3 3` hai lần.
Run: `grep -B4 "^# cell: NB_cpu (máy cá nhân)" docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P5b_classification_isic2017.md` — đọc: cell máy nào đọc `{ROOT}/runs/...` có khối rclone phía trước.

- [ ] **Step 9: Commit**

```bash
git add docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P5b_classification_isic2017.md
git commit -m "docs(sv_b): P5a/P5b train on Colab extension, evaluate locally"
```

---

### Task 5: `P6_ghep_cap_va_ridge.md`

**Files:**
- Modify: `docs/sv_b/P6_ghep_cap_va_ridge.md` (mục 1, mục 2 dòng 13+, 3.1 dòng 27–31, 11 khối `# file:`, cell dòng 1019–1140, mục 7 dòng 1145+)

**Interfaces:**
- Consumes: SB1–SB7; checkpoint từ Task 4.
- Produces: `{ROOT}/runs/<p6 run>/ridge/ridge.joblib`, `ridge_test/features.csv` — P7/P8/P9 dùng.

- [ ] **Step 1: Đọc** cell 1019–1140 và ghi luồng: dữ liệu giả + `build_pairs.py` (máy) → manifest cặp (máy, đẩy lên Drive) → `infer_images.py` seg (Colab `NB_seg`, cell 1052) + cls (Colab `NB_cls`, cell 1060) ghi `{ROOT}/runs/...` → kéo về → `train_ridge.py`/đánh giá (máy).

- [ ] **Step 2:** Cell `NB_cpu` → nhãn máy + SB6; cell `NB_seg`/`NB_cls` → nhãn Colab + `{CODE}`. Ngay sau cell máy cuối cùng trước cell 1052, chèn câu `Đẩy manifest cặp lên Drive cho Colab:` + khối rclone theo mẫu SB4 nhưng đường dẫn là thư mục chứa manifest cặp mà cell đó ghi ra (`rclone copy ~/nckh_drive/<đường dẫn tương đối> gdrive:NCKH_PanDerm/<cùng đường dẫn tương đối> --progress`). Ngay trước cell máy đầu tiên sau cell 1060, chèn `Kéo kết quả suy luận về máy:` + SB3 (với `<run_id>` của run P6).

- [ ] **Step 3: 3.1 Go/No-Go:** giữ các mục; thêm ô `- [ ] Quyền lưu bản sao trên máy cá nhân của thành viên nhóm (~/nckh_drive/data/uq); nếu không có, ghi rõ chỉ xử lý trên Colab/Drive và không chạy rclone thư mục uq về máy.`

- [ ] **Step 4:** Mục 1 thêm cột `Sinh ở`; mục 2 cột `Chạy ở`, thêm dòng rclone hai chiều. Áp SB5 cho 11 khối `# file:`.

- [ ] **Step 5: Mục 7** tiêu đề mới + SB7 chung + 2 hàng rclone.

- [ ] **Step 6: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab" docs/sv_b/P6_ghep_cap_va_ridge.md`
Expected: không có.
Run: `echo $(grep -c "^# file:" docs/sv_b/P6_ghep_cap_va_ridge.md) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/P6_ghep_cap_va_ridge.md)`
Expected: `11 11`.
Run: `grep -c "rclone copy" docs/sv_b/P6_ghep_cap_va_ridge.md`
Expected: ≥ 2.

- [ ] **Step 7: Commit**

```bash
git add docs/sv_b/P6_ghep_cap_va_ridge.md
git commit -m "docs(sv_b): P6 split Colab inference and local Ridge"
```

---

### Task 6: `P7`, `P10`, `P0_P1_P4_P11`

**Files:**
- Modify: `docs/sv_b/P7_kiem_thu_metrics_bootstrap.md` (mục 2 dòng 13+, 2 khối `# file:`, cell 61, 232, 243, mục 7 dòng 299+)
- Modify: `docs/sv_b/P10_tai_lap_hinh_bang.md` (mục 2 dòng 12+, 3 khối `# file:` — dòng 193 nằm trong khối file, **không sửa**, cell 360, 373, 402, dòng 389, mục 7 dòng 429+)
- Modify: `docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md` (dòng 40, cell 49, 92, 112, dòng 136)

**Interfaces:**
- Consumes: SB1, SB3, SB5, SB6; run P6 từ Task 5.

- [ ] **Step 1:** Ba file: mọi cell `NB_cpu` → nhãn máy + SB6. Mục 2 cột `Chạy ở` = `Máy`. Mục 1 (nếu có bảng đầu ra) thêm cột `Sinh ở` = `Máy (~/nckh_drive/...)`. Áp SB5 cho 2 + 3 khối `# file:`.

- [ ] **Step 2: P7, P10:** trước cell đầu tiên đọc `{ROOT}/runs/...` chèn `Kéo các run cần dùng về máy:` + SB3. Mục 7 tiêu đề mới + 2 hàng rclone (không thêm hàng Colab vì không có cell Colab).

- [ ] **Step 3: P10 dòng 389** "Từ một runtime Colab **mới**, chỉ làm theo docs:" → "Từ một máy **mới** (clone repo, làm 00 mục 6b) và một server Colab **mới**, chỉ làm theo docs:". Đọc các bước ngay sau, bước nào làm phần CPU trên Colab thì đổi thành máy.

- [ ] **Step 4: P0_P1_P4_P11:** dòng 40 sau "trên Colab/Drive" thêm " (và lưu bản sao trên máy cá nhân nếu điều khoản cho phép)". Dòng 136 thay `Colab, loại GPU thật đã dùng (từ run card), Python, ba môi trường (seg/cls/cpu) và lý do tách.` → `máy cá nhân (VS Code, .venv Python 3.10, CPU) cho xử lý dữ liệu/đánh giá; Colab nối qua extension VS Code (loại GPU thật từ run card) cho train/suy luận; ba môi trường (seg/cls trên Colab, cpu trên máy) và lý do tách.`

- [ ] **Step 5: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab|runtime Colab \*\*mới\*\*" docs/sv_b/P7_kiem_thu_metrics_bootstrap.md docs/sv_b/P10_tai_lap_hinh_bang.md docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md`
Expected: không có.
Run: `for f in docs/sv_b/P7_kiem_thu_metrics_bootstrap.md docs/sv_b/P10_tai_lap_hinh_bang.md; do echo $(grep -c "^# file:" $f) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" $f); done`
Expected: `2 2`, `3 3`.

- [ ] **Step 6: Commit**

```bash
git add docs/sv_b/P7_kiem_thu_metrics_bootstrap.md docs/sv_b/P10_tai_lap_hinh_bang.md docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md
git commit -m "docs(sv_b): P7/P10/P0-P11 run on local machine"
```

---

### Task 7: `P8_robustness.md`

**Files:**
- Modify: `docs/sv_b/P8_robustness.md` (mục 2 dòng 12–20, dòng 586, 5 khối `# file:`, cell 588–675, 4.7 dòng 637–658, mục 5 dòng 678+, mục 7 dòng 717+)

**Interfaces:**
- Consumes: SB1–SB7; checkpoint Task 4, run P6 Task 5.

- [ ] **Step 1: Mục 2** thay bảng:

```markdown
| Việc | Chạy ở | Ghi chú |
|---|---|---|
| Tạo `degrade.py`, `make_degraded.py`, `evaluate_robustness.py`, test | Máy (`NB_cpu`) | |
| Sinh ảnh suy giảm + suy luận lại PanDerm (ISIC 2018/2017) | Colab `NB_seg`, `NB_cls` | Ảnh suy giảm sinh ngay trên `/content` (CPU của server), không upload từ máy. 10 mức × số ảnh test × ms/ảnh (P2) |
| Đánh giá seg/cls từng mức | Colab, trong cùng vòng lặp | Vài giây mỗi mức; `robustness.json` ghi lên Drive |
| Dự báo UQ: ảnh t suy giảm + suy luận | Colab `NB_seg`, `NB_cls` | Thư mục `{RUN}/p8_degraded` trên Drive |
| `evaluate_robustness.py forecast` | Máy (`NB_cpu`) sau khi kéo về | Vài giây mỗi mức |
```

- [ ] **Step 2: Dòng 586** → "Chuẩn bị: push code ở máy, cell setup Colab (P2 4.1b), `venv_seg`, patch, dữ liệu ISIC 2018 trên `/content/data` (P3 4.6)."

- [ ] **Step 3:** Cell `NB_seg`/`NB_cls` (4.5, 4.6, 4.8): nhãn Colab + `{ROOT}/nckh` → `{CODE}`. `/content/...` giữ.

- [ ] **Step 4: 4.7:** cell đầu (tạo `p8_t_images.csv`) đổi thành **máy** (nó đọc `{RUN}/ridge_test/features.csv` đã có ở máy từ P6); đặt `RUN = f"{ROOT}/runs/<run P6 trên UQ>"` nếu dòng gốc chưa có `{ROOT}`. Sau cell chèn `Đẩy danh sách ảnh t lên Drive:` +

````markdown
```bash
# terminal VS Code (máy cá nhân)
rclone copyto ~/nckh_drive/runs/<p6_run>/p8_t_images.csv gdrive:NCKH_PanDerm/runs/<p6_run>/p8_t_images.csv
```
````

Bước 1–2 của danh sách ghi rõ `(Colab NB_seg)` / `(Colab NB_cls)`; đường dẫn script dùng `{CODE}/scripts/...`. Trước bước 3 chèn `Kéo kết quả về máy:` +

````markdown
```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/runs/<p6_run>/p8_degraded ~/nckh_drive/runs/<p6_run>/p8_degraded --progress
```
````

Cell bước 3: nhãn máy + SB6.

- [ ] **Step 5:** Mục 5 cell pytest: nhãn máy + SB6. Áp SB5 cho 5 khối `# file:`. Mục 7 tiêu đề mới + SB7 chung + 2 hàng rclone.

- [ ] **Step 6: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab|pull code ở NB_cpu" docs/sv_b/P8_robustness.md`
Expected: không có.
Run: `echo $(grep -c "^# file:" docs/sv_b/P8_robustness.md) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/P8_robustness.md)`
Expected: `5 5`.

- [ ] **Step 7: Commit**

```bash
git add docs/sv_b/P8_robustness.md
git commit -m "docs(sv_b): P8 degrade+infer on Colab, forecast eval locally"
```

---

### Task 8: `P9_demo_streamlit.md` + chỉnh spec

**Files:**
- Modify: `docs/sv_b/P9_demo_streamlit.md` (mục 2 dòng 11–17, 3.1 dòng 21–30, 3 khối `# file:`, 4.4 dòng 308–323, 4.5 dòng 325–352, 4.6 dòng 354–364, mục 5 dòng 366–372, bảng 5.1, mục 7 dòng 400+)
- Modify: `docs/superpowers/specs/2026-10-07-sv-b-local-vscode-colab-design.md` (hàng `P9` bảng mục 5, dòng *Phân phase* mục 3)

**Lý do lệch spec:** spec ghi "demo chạy trên máy (CPU)". Demo với model thật là chạy model → theo yêu cầu người dùng phải ở Colab; dựng venv mmcv ở máy cũng ngoài phạm vi. Do đó: chế độ giả ở máy, model thật trên Colab `NB_seg` + tunnel (như doc hiện tại). Đã báo người dùng ở bước duyệt plan.

- [ ] **Step 1: Mục 2** thay 3 gạch đầu dòng bằng:
  - `**Thử giao diện (chế độ giả):** máy, NB_cpu, mở http://localhost:8501 trong trình duyệt. Không cần GPU.`
  - `**Model thật:** Colab NB_seg (GPU), dựng thêm venv_cls trong cùng server vì demo cần cả hai venv. Xem qua tunnel cloudflared (mục 4.6), vì cổng 8501 của server Colab không mở thẳng về máy.`

  Thêm `> ⚠️ Chưa kiểm chứng trên extension — nếu extension hỗ trợ chuyển tiếp cổng thì có thể bỏ tunnel.`

- [ ] **Step 2: 3.1** sơ đồ dòng 1 → `Trình duyệt (máy) ──(URL tunnel)──► Streamlit trên server Colab (demo/app.py)`.

- [ ] **Step 3: 4.4** tiêu đề `### 4.4. Thử giao diện bằng chế độ giả (máy, không cần model)`. Câu "Cần một ridge.joblib…" thêm: `; nếu run đó nằm trên Drive, kéo về máy bằng rclone (00 mục 5.2)`. Cell thay bằng:

```python
# cell: NB_cpu (máy cá nhân)
import os, subprocess, time
os.environ['NCKH_SEG_CMD'] = f"{PY} {REPO}/scripts/infer_images.py --task seg --fake"
os.environ['NCKH_CLS_CMD'] = f"{PY} {REPO}/scripts/infer_images.py --task cls --fake"
os.environ['NCKH_RIDGE'] = f"{ROOT}/runs/<p6_fake_run_id>/ridge.joblib"
os.environ['NCKH_STABLE_EPS'] = "0.005"
%pip install -q -e "{REPO}[demo]"
subprocess.Popen(f"{PY} -m streamlit run {REPO}/demo/app.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1", shell=True)
time.sleep(8)
!curl -s localhost:8501/_stcore/health && echo " ← Streamlit OK, mở http://localhost:8501"
```

Câu sau cell: "Rồi mở tunnel (mục 4.6)." → "Mở http://localhost:8501 trong trình duyệt của máy (không cần tunnel). Tắt: `!pkill -f \"streamlit run\"`."

- [ ] **Step 4: 4.5** tiêu đề `### 4.5. Chạy với model thật (Colab NB_seg, GPU)`. Bước chuẩn bị 1 → `push code ở máy, rồi cell setup Colab P2 4.1b trong NB_seg`. Cell: nhãn `# cell: NB_seg (Colab GPU)`, `{ROOT}/nckh` → `{CODE}` (3 chỗ).

- [ ] **Step 5: 4.6** cell: nhãn `# cell: NB_seg (Colab GPU)`; nội dung giữ. Câu sau: "Mở link in ra trong trình duyệt của máy."

- [ ] **Step 6: Mục 5** cell pytest: nhãn máy, `!cd {REPO} && {PY} -m pytest -q tests/test_demo_pipeline.py`. Bảng 5.1 hàng 1: "Runtime mới" → "Server Colab mới".

- [ ] **Step 7:** Áp SB5 cho 3 khối `# file:`. Mục 7 tiêu đề mới + SB7 chung + hàng `| localhost:8501 không mở được ở máy khi chạy trên Colab | Cổng của server Colab không chuyển về máy | Dùng tunnel mục 4.6 |`.

- [ ] **Step 8: Sửa spec:** bảng mục 5 hàng `P9` → `Chế độ giả chạy trên máy (localhost:8501, không tunnel); demo với model thật chạy trên Colab NB_seg + tunnel cloudflared (chạy model ⇒ Colab).` Mục 3 dòng *Phân phase*: ở **Máy** đổi `P9` → `P9 (chế độ giả)`; ở **Colab** thêm `P9 (model thật)`.

- [ ] **Step 9: Kiểm tra**

Run: `grep -nE "^# cell: NB_(cpu|seg|cls)\s*$|\{ROOT\}/nckh|Lỗi thường gặp trên Colab|Laptop:" docs/sv_b/P9_demo_streamlit.md`
Expected: không có.
Run: `echo $(grep -c "^# file:" docs/sv_b/P9_demo_streamlit.md) $(grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/P9_demo_streamlit.md)`
Expected: `3 3`.

- [ ] **Step 10: Commit**

```bash
git add docs/sv_b/P9_demo_streamlit.md docs/superpowers/specs/2026-10-07-sv-b-local-vscode-colab-design.md
git commit -m "docs(sv_b): P9 fake demo local, real demo on Colab; align spec"
```

---

### Task 9: Kiểm tra chéo toàn bộ

**Files:**
- Modify: chỉ file mà kiểm tra phát hiện lỗi.

- [ ] **Step 1: Từ khoá cấm**

Run: `grep -nE "GH_TOKEN|AUTH_URL|Secrets|userdata|Save a copy in Drive|colab\.research\.google\.com|\{ROOT\}/nckh" docs/sv_b/*.md`
Expected: chỉ các hàng `userdata.get` trong bảng mục 7.

- [ ] **Step 2: Nhãn cell**

Run: `grep -nE "^# cell: " docs/sv_b/*.md | grep -vE "NB_cpu \(máy cá nhân\)|NB_(seg|cls) \(Colab GPU\)"`
Expected: không có.

- [ ] **Step 3: Dòng 📁 = 50 và nằm ngay trên khối**

Run: `echo $(cat docs/sv_b/*.md | grep -c "^# file:") $(cat docs/sv_b/*.md | grep -c "📁 \*\*Tạo trên máy cá nhân:\*\*")`
Expected: `50 50`.
Run: `grep -hA3 "📁 \*\*Tạo trên máy cá nhân:\*\*" docs/sv_b/*.md | grep -c "^# file:"`
Expected: `50`.

- [ ] **Step 4: Cell máy không dùng /content, python trần, !pip**

```bash
for f in docs/sv_b/*.md; do awk -v F="$f" '/^# cell: NB_cpu/{c=1} c&&/^```$/{c=0} c&&(/\/content/||/(!|&& )python /||/!pip/){print F": "$0}' "$f"; done
```

Expected: không in gì.

- [ ] **Step 5: Khối `# file:` không bị sửa**

```bash
for f in docs/sv_b/*.md; do diff <(git show d77cd5f:$f | awk '/^# file:/,/^```$/') <(awk '/^# file:/,/^```$/' $f) >/dev/null || echo "CHANGED: $f"; done
```

Expected: không in `CHANGED`.

- [ ] **Step 6: Đọc chéo 00 ↔ phase:** với mỗi hàng bảng lộ trình 00, mở mục 2 của file phase tương ứng, xác nhận cột `Chạy ở` khớp. Sửa nếu lệch.

- [ ] **Step 7: Commit (nếu có sửa)**

```bash
git add docs/sv_b/
git commit -m "docs(sv_b): cross-check fixes for local + Colab extension"
```
