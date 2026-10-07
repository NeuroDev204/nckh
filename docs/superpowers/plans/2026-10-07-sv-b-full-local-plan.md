# Hướng dẫn SV B chạy hoàn toàn local — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bỏ Colab/Drive/rclone khỏi toàn bộ hướng dẫn SV B, code và notebook; mọi bước chạy trên laptop hoặc máy GPU local.

**Architecture:** Code chỉ đổi giá trị mặc định đường dẫn (`~/nckh_root`, `~/nckh_data`) và chú thích. Docs đổi theo một bảng thay thế chung + một bộ khối lệnh chuẩn (cài venv, smoke test, train trong tmux). Notebook là lớp mỏng, kernel trỏ thẳng vào venv.

**Tech Stack:** Python 3.12 `.venv` (CPU), `uv`, venv Python 3.10 cho PanDerm seg/cls, pytest, Jupyter trong VS Code, tmux.

**Spec:** `docs/superpowers/specs/2026-10-07-sv-b-full-local-design.md`

## Global Constraints

- Bố cục mỗi máy (spec §3): `~/Documents/nckh` (`<repo>`, `.venv`), `~/nckh_root` (`checkpoints/`, `data/manifests/`, `data/uq/`, `runs/<run_id>/`), `~/nckh_data` (ảnh ISIC, `smoke/`, `degraded/`), `~/PanDerm` (commit `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` + patch), `~/venvs/venv_seg`, `~/venvs/venv_cls`.
- Không đổi phiên bản thư viện, commit PanDerm, patch, metric, split, protocol, số epoch, batch size gốc (spec L7).
- Không sửa: `docs/Huong_dan_SV_B_Tuan_1_*.md`, `De_cuong_*.docx`, `docs/hop/*`, spec/plan cũ.
- Nhãn khối (spec §5.1): `# terminal (laptop, .venv)`, `# terminal (máy GPU, venv_seg)`, `# terminal (máy GPU, venv_cls)`, `# terminal (máy GPU, .venv)`, `# terminal (máy local)` cho lệnh chạy trên cả hai máy, `# cell: nb_cpu`, `# cell: nb_seg`, `# cell: nb_cls`.
- Trao đổi file SV A ↔ SV B (spec L5), câu chuẩn: *"SV A gửi thư mục cho SV B qua kênh được điều khoản UQ cho phép (USB, ổ mạng của trường…); SV B chép vào `<đường dẫn>`."*
- Cổng UQ (spec L6): điều kiện "xử lý trên dịch vụ đám mây" → "lưu và xử lý trên máy của thành viên nhóm".
- Tiếng Việt, giọng văn như các file hiện có. Không reformat đoạn không liên quan.
- Nếu một khối `# file:` bị sửa mà file thật tương ứng đã có trong repo, sửa cả file thật trong cùng commit và chạy `.venv/bin/python -m pytest -q`.

### Bảng thay thế chung (áp dụng ở mọi task docs)

| Cũ | Mới |
|---|---|
| `MyDrive/NCKH_PanDerm`, `/content/drive/MyDrive/NCKH_PanDerm`, `{ROOT}` (trong lệnh bash), `~/nckh_drive`, `$ROOT` | `~/nckh_root` (trong cell Python dùng `{ROOT}` với `ROOT = Path.home()/'nckh_root'`) |
| `/content/data` | `~/nckh_data` |
| `/content/smoke` | `~/nckh_data/smoke` |
| `/content/degraded` | `~/nckh_data/degraded` |
| `/content/zips` | `~/nckh_data/zips` |
| `/content/PanDerm` | `~/PanDerm` |
| `/content/nckh`, `{CODE}` | `<repo>` (trong lệnh bash: `~/Documents/nckh`) |
| `/content/venv_seg`, `/content/venv_cls`, `{VENV}/bin/python` | `~/venvs/venv_seg/bin/python`, `~/venvs/venv_cls/bin/python` |
| `/content/<file tạm>.csv` | `~/nckh_root/runs/<run_id>/<file>.csv` (nằm cạnh run) |
| `# cell: NB_seg (Colab GPU)` có lệnh `!{VENV}/bin/python ...` | khối `bash` nhãn `# terminal (máy GPU, venv_seg)` |
| `# cell: NB_cpu (máy cá nhân)` | `# cell: nb_cpu` |
| `# terminal VS Code (máy cá nhân)` | `# terminal (laptop, .venv)` hoặc `# terminal (máy local)` |
| Mọi lệnh `rclone ...`, `drive.mount`, cell setup Colab 4.1b, câu "push code ở máy rồi pull trên Colab" | Xóa |
| "Colab `NB_seg`/`NB_cls`" ở cột *Nơi chạy* | `Máy GPU (venv_seg)` / `Máy GPU (venv_cls)` |
| "Máy" ở cột *Nơi chạy* | `Laptop hoặc máy GPU (.venv)` khi không cần GPU |
| Mục 7 "Lỗi thường gặp (máy / Colab extension)" | "Lỗi thường gặp (máy local)" |
| Trong tham số dạng `--x=~/...` hoặc trong chuỗi `"..."` | `$HOME/...` (shell không mở rộng `~` ở đó) |

### Khối lỗi chung cho mục 7 (thêm vào file có bước GPU: P2, P5a, P5b, P6, P8, P9)

```markdown
| `CUDA out of memory` | Hạ batch size (hoặc `--accum_iter` nếu script hỗ trợ), ghi giá trị mới vào run card; không đổi gì khác |
| `CUDA driver version is insufficient` / `no kernel image` | Driver quá cũ cho wheel cu118: cập nhật driver NVIDIA (≥ 520) |
| Đóng VS Code/terminal làm dừng train | Chạy trong `tmux new -s <tên>`; mở lại bằng `tmux attach -t <tên>` |
```

### Tiêu chí grep (spec §1)

```bash
grep -rniE "colab|/content|rclone|gdrive|MyDrive|drive\.mount|cloudflared|nckh_drive" docs/sv_b notebooks src scripts tests
```
Cuối cùng chỉ được còn đúng 1 dòng: câu lịch sử trong `00` (Task 3). Mỗi task docs chạy lệnh này trên file của mình (thêm `\{VENV\}|\{CODE\}`) → rỗng.

## Review Focus

- **Clone repo ở chỗ khác `~/Documents/nckh`:** `00` mục 6b có đúng một ghi chú "nếu clone chỗ khác, thay đường dẫn này"; cell notebook dùng biến `REPO` có comment "sửa nếu…". Kiểm ở Task 3 step 5, Task 2 step 2.
- **`NCKH_ROOT` còn đặt sẵn từ shell cũ (ví dụ trỏ Drive):** biến môi trường vẫn thắng default — `test_project_root_reads_env` giữ nguyên; Task 1 thêm test default khi không có biến.
- **Chạy lại lệnh cài đặt lần hai:** clone/venv/patch idempotent (`test -d … ||`, `test -x … ||`, `git apply --reverse --check`). Kiểm ở Task 4 step 5, 7.
- **`~` trong chuỗi Python hoặc sau `=`:** không tự mở rộng. Cell Python dùng `Path.home()`; bash dùng `$HOME` trong chuỗi/tham số `=`. Kiểm ở Task 11 step 2.
- **Train dài bị ngắt khi đóng terminal:** mọi pilot/train/suy luận toàn tập ở P5a, P5b, P6, P8 nằm trong `tmux` và `tee` log vào run. Kiểm ở Task 6 step 5, Task 7 step 3, Task 9 step 1.

---

### Task 1: Default đường dẫn local trong code

**Files:**
- Modify: `src/nckh/paths.py:10-11,35`
- Modify: `src/nckh/runcard.py:101`
- Modify: `scripts/inspect_checkpoint.py:3`
- Modify: `scripts/bench_inference.py:4-8,103`
- Test: `tests/test_paths_runcard.py`
- Modify (khối `# file:` tương ứng): `docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md` (khối `src/nckh/paths.py` ~dòng 198, `src/nckh/runcard.py` ~241, `tests/test_paths_runcard.py` ~394, `scripts/inspect_checkpoint.py` ~506, `scripts/bench_inference.py` ~1063)

**Interfaces:**
- Produces: `nckh.paths.DEFAULT_ROOT: str` = `str(Path.home() / "nckh_root")`, `nckh.paths.DEFAULT_LOCAL_DATA: str` = `str(Path.home() / "nckh_data")`. Chữ ký `project_root()`, `local_data_root()`… giữ nguyên.

- [ ] **Step 1: Viết test hỏng**

Sửa import trong `tests/test_paths_runcard.py`: `from nckh.paths import project_root, runs_dir` → `from nckh.paths import local_data_root, project_root, runs_dir`. Thêm ngay sau `test_project_root_reads_env`:

```python
def test_defaults_are_local_home(monkeypatch) -> None:
    monkeypatch.delenv("NCKH_ROOT", raising=False)
    monkeypatch.delenv("NCKH_LOCAL_DATA", raising=False)
    assert project_root() == Path.home() / "nckh_root"
    assert local_data_root() == Path.home() / "nckh_data"
```

- [ ] **Step 2: Chạy test, xác nhận hỏng**

Run: `cd ~/Documents/nckh && .venv/bin/python -m pytest -q tests/test_paths_runcard.py::test_defaults_are_local_home`
Expected: FAIL, `AssertionError` so `/content/drive/MyDrive/NCKH_PanDerm` với `…/nckh_root`.

- [ ] **Step 3: Sửa `src/nckh/paths.py`**

```python
DEFAULT_ROOT = str(Path.home() / "nckh_root")
DEFAULT_LOCAL_DATA = str(Path.home() / "nckh_data")
```
Comment trong `local_data_root()`:
```python
    # Ảnh giải nén để riêng khỏi NCKH_ROOT: nặng hàng chục GB, tải lại được, không cần sao lưu cùng runs/.
```

- [ ] **Step 4: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest -q`
Expected: `21 passed`.

- [ ] **Step 5: Sửa chú thích (không đổi logic)**

`src/nckh/runcard.py:101`:
```python
    # CLI để gọi bằng python của venv seg/cls (khác kernel), tránh phải escape dấu ngoặc nhọn trong lệnh "!".
```
`scripts/inspect_checkpoint.py:3`:
```
Chạy: ~/venvs/venv_seg/bin/python scripts/inspect_checkpoint.py ~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth
```
`scripts/bench_inference.py:4-8`:
```
Chạy bằng Python của venv tương ứng (venv_seg / venv_cls), không phải Python của .venv.
Ví dụ:
  ~/venvs/venv_seg/bin/python scripts/bench_inference.py --task seg \
      --panderm-dir ~/PanDerm/segmentation --pretrained ~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth \
      --images "$HOME/nckh_data/ISIC2018/Validation_Data/*.jpg" --n 20 --out-dir ~/nckh_root/runs/<run_id>
```
`scripts/bench_inference.py:103` help: `'glob, ví dụ "$HOME/nckh_data/ISIC2018/Validation_Data/*.jpg"'`.

- [ ] **Step 6: Đồng bộ khối `# file:` trong P2**

Sửa đúng những dòng tương ứng trong các khối `src/nckh/paths.py`, `src/nckh/runcard.py`, `tests/test_paths_runcard.py`, `scripts/bench_inference.py`, và dòng "Chạy:" của khối `scripts/inspect_checkpoint.py`. Không sửa phần khác: khối `inspect_checkpoint.py`, `test_inspect_checkpoint.py`, `.gitignore` đã lệch với repo từ trước — ngoài phạm vi.

Kiểm tra (không commit script):
```bash
python3 - <<'EOF'
import re, pathlib
repo = pathlib.Path.home()/'Documents/nckh'
md = (repo/'docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md').read_text()
for f in ['src/nckh/paths.py','src/nckh/runcard.py','tests/test_paths_runcard.py','scripts/bench_inference.py']:
    m = re.search(r"```\w*\n# file: "+re.escape(f)+r"\n(.*?)```", md, re.S)
    real = (repo/f).read_text().strip()
    body = m.group(1).strip() if m else None
    ok = body is not None and (body == real or ('# file: '+f+'\n'+body) == real)
    print(f, 'OK' if ok else 'DIFF')
EOF
```
Expected: 4 dòng `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/nckh/paths.py src/nckh/runcard.py scripts/inspect_checkpoint.py scripts/bench_inference.py tests/test_paths_runcard.py docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md
git commit -m "feat(paths): default NCKH_ROOT/NCKH_LOCAL_DATA về ~/nckh_root, ~/nckh_data"
```

---

### Task 2: Notebook local

**Files:**
- Modify: `notebooks/nb_cpu.ipynb` (cell 0)
- Rewrite: `notebooks/nb_seg.ipynb`

**Interfaces:**
- Consumes: default đường dẫn (Task 1); CLI `scripts/inspect_checkpoint.py <ck>`, `scripts/bench_inference.py --task --panderm-dir --pretrained --images --n --save-overlays --out-dir` (ghi overlay vào `<out-dir>/overlays/<stem>.png`); `nckh.runcard.sha256_file(Path) -> str`, `nckh.runcard.new_run_id(str) -> str`, CLI `python -m nckh.runcard <run_dir> --seed --input k=v --config k=v`.
- Produces: biến notebook `REPO`, `ROOT`, `DATA`, `PANDERM`, `CK`, `PY` (P2 4.1 dán nguyên cell 0).

- [ ] **Step 0: Xác nhận ghi đè thay đổi chưa commit**

`git diff notebooks/nb_seg.ipynb` hiện chỉ khác output/execution_count từ Colab. Người dùng đã được báo ở bước handoff; nếu chưa có xác nhận rõ, hỏi trước khi làm step 2.

- [ ] **Step 1: `nb_cpu.ipynb` cell 0**

Thay dòng `ROOT = ...`:
```python
ROOT = str(Path.home() / 'nckh_root')           # NCKH_ROOT: checkpoints, manifests, runs
```
Giữ các dòng khác. Sửa bằng `NotebookEdit` hoặc script `json` (giữ `metadata`, outputs rỗng).

- [ ] **Step 2: Viết lại `nb_seg.ipynb`** (5 cell code, outputs rỗng, `execution_count: null`, giữ `metadata` cấp notebook)

Cell 0:
```python
# cell: nb_seg
# Kernel: Select Kernel → Python Environments → ~/venvs/venv_seg (dựng ở P2 mục 4.4).
import sys
from pathlib import Path
REPO = Path.home() / 'Documents' / 'nckh'       # sửa nếu bạn clone repo ở chỗ khác
ROOT = Path.home() / 'nckh_root'                # NCKH_ROOT (default của nckh.paths)
DATA = Path.home() / 'nckh_data'                # NCKH_LOCAL_DATA
PANDERM = Path.home() / 'PanDerm'
CK = ROOT / 'checkpoints' / 'panderm_bb_data6_checkpoint-499.pth'
PY = sys.executable                             # chính là venv_seg, dùng trong lệnh "!"
%cd {REPO}
!git log --oneline -1
```
Cell 1:
```python
# cell: nb_seg — ghi SHA-256 checkpoint (tải ở P2 mục 4.3)
from datetime import date
from nckh.runcard import sha256_file
digest = sha256_file(CK)
print(digest)
with open(ROOT / 'checkpoints' / 'SHA256SUMS', 'a') as fh:
    fh.write(f'{digest} {CK.name} #tai {date.today()}\n')
```
Cell 2:
```python
# cell: nb_seg
!{PY} scripts/inspect_checkpoint.py {CK}
```
Cell 3:
```python
# cell: nb_seg
PATCH = REPO / 'patches' / 'panderm_base_seg.patch'
# Kiểm tra theo chiều ngược trước, để chạy lại cell này nhiều lần cũng không áp patch hai lần.
!cd {PANDERM} && (git apply --reverse --check {PATCH} 2>/dev/null && echo "Patch đã áp từ trước") || (git apply {PATCH} && echo "Đã áp patch")
!cd {PANDERM} && git diff --stat
```
Cell 4:
```python
# cell: nb_seg — smoke test 20 ảnh rồi xem overlay
from IPython.display import Image, display
from nckh.runcard import new_run_id
RUN = ROOT / 'runs' / new_run_id('smoke_seg')
!{PY} scripts/bench_inference.py --task seg --panderm-dir {PANDERM}/segmentation --pretrained {CK} --images "{DATA}/smoke/ISIC2018_Task1-2_Validation_Input/*.jpg" --n 20 --save-overlays 10 --out-dir {RUN}
!{PY} -m nckh.runcard {RUN} --seed 0 --input pretrained={CK} --input patch={PATCH} --config task=smoke_seg --config n=20
for p in sorted((RUN / 'overlays').glob('*.png'))[:4]:
    display(Image(filename=str(p), width=320))
```

- [ ] **Step 3: Kiểm tra JSON và grep**

```bash
python3 -c "import json; [json.load(open(f'notebooks/{n}.ipynb')) for n in ('nb_cpu','nb_seg')]; print('OK')"
grep -niE "colab|/content|rclone|drive|nckh_drive" notebooks/*.ipynb
```
Expected: `OK`; grep rỗng.

- [ ] **Step 4: Commit**

```bash
git add notebooks/nb_cpu.ipynb notebooks/nb_seg.ipynb
git commit -m "feat(notebooks): nb_cpu/nb_seg chạy local với kernel venv"
```

---

### Task 3: `00_tong_quan_va_lo_trinh.md`

**Files:** Modify `docs/sv_b/00_tong_quan_va_lo_trinh.md`

**Interfaces:**
- Produces: tham chiếu các phase dùng: "`00` mục 4 (bố cục)", "`00` mục 5.2 (chuyển sang máy GPU)", "`00` mục 6 (kernel)", "`00` mục 6b (cài đặt một lần)". Giữ số mục.

- [ ] **Step 1: Giới thiệu + bảng lộ trình (mục 2)**

Cột *Runtime*: P0, P1, P3, P4, P7, P10, P11 → `Laptop`; P2 → `Laptop (smoke test) + máy GPU`; P5a, P5b, P8 → `Máy GPU`; P6 → `Máy GPU (suy luận) + .venv (Ridge)`; P9 → `Laptop (chế độ giả) / máy GPU (model thật)`. Thêm một câu sau đoạn giới thiệu: *"Bản trước của bộ hướng dẫn chạy GPU trên Colab; từ 07/10/2026 mọi bước chạy local."* — dòng duy nhất được chứa chữ "Colab".

- [ ] **Step 2: Mục 4 → "Bố cục thư mục trên mỗi máy"**

Thay mục 4 bằng cây thư mục như spec §3 (khối ```), mỗi dòng một chú thích ngắn, và câu: *"Laptop và máy GPU có cùng bố cục; `nckh.paths` mặc định trỏ `~/nckh_root` và `~/nckh_data`, chỉ đặt `NCKH_ROOT`/`NCKH_LOCAL_DATA` khi muốn để chỗ khác."* Giữ bảng dung lượng ISIC nếu mục 4 cũ có (P3 tham chiếu).

- [ ] **Step 3: Mục 5 → "Đồng bộ giữa hai máy"**

5.1 giữ phần git (sửa nhãn khối). 5.2 thay toàn bộ rclone bằng:

````markdown
### 5.2. Chuyển sang máy GPU

Code: `git clone`/`git pull` repo trên máy GPU. Dữ liệu và kết quả: chép nguyên hai thư mục, hoặc tải lại bằng lệnh ở P2 mục 4.3 và P3.

```bash
# terminal (laptop, .venv)
rsync -a --info=progress2 ~/nckh_root ~/nckh_data <user>@<máy-gpu>:~/
```

Sau đó làm mục 6b trên máy GPU. Chép kết quả về laptop theo chiều ngược lại cho từng `~/nckh_root/runs/<run_id>`; thêm `--exclude '*.ckpt' --exclude '*.pth'` nếu không cần checkpoint.
````
5.3: giữ quy tắc về git, bỏ quy tắc về Drive/rclone; thêm *"Không train cùng một `run_id` trên hai máy."*

- [ ] **Step 4: Mục 6 → bảng kernel**

Bảng `nb_cpu` / `nb_seg` / `nb_cls` (cột: Notebook, Kernel/Python, Phase, Gói chính — giữ cột *Gói chính* cũ, chỉ đổi `/content/venv_*` → `~/venvs/venv_*`). Thêm: *"Logic nằm trong `src/` và `scripts/`; notebook chỉ để xem kết quả. Lệnh train/suy luận dài chạy trong `tmux`."*

- [ ] **Step 5: Mục 6b → cài đặt một lần**

Giữ VS Code/`uv`/`.venv`; bỏ extension Google Colab, `rclone config`, `.env`; thêm:
```bash
# terminal (máy local)
sudo apt install -y tmux git
mkdir -p ~/nckh_root/{checkpoints,data/manifests,runs} ~/nckh_data ~/venvs
```
Thêm: *"Nếu clone repo chỗ khác `~/Documents/nckh`, thay đường dẫn này trong mọi lệnh và biến `REPO` của notebook."* và *"Máy GPU: `nvidia-smi` phải chạy được trước P2 mục 4.4."*

- [ ] **Step 6: Mục 8 (bản đồ file)**

Dòng notebook: *"Máy: `<repo>/notebooks/`, kernel venv tương ứng"*. Thêm dòng `~/venvs/venv_seg`, `~/venvs/venv_cls`, `~/PanDerm` (ngoài repo). Bỏ mọi "kernel chạy trên Colab".

- [ ] **Step 7: Mục 9 (quy ước nhãn)**

Thay các dòng nhãn bằng danh sách nhãn trong Global Constraints.

- [ ] **Step 8: Kiểm tra + commit**

```bash
grep -niE "colab|/content|rclone|gdrive|MyDrive|drive\.mount|cloudflared|nckh_drive" docs/sv_b/00_tong_quan_va_lo_trinh.md
```
Expected: đúng 1 dòng (câu lịch sử).
```bash
git add docs/sv_b/00_tong_quan_va_lo_trinh.md && git commit -m "docs(sv_b): 00 chuyển sang bố cục local hai máy"
```

---

### Task 4: `P2_moi_truong_checkpoint_smoke_test.md`

**Files:** Modify `docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md`

**Interfaces:**
- Consumes: cell 0–4 của `nb_seg` (Task 2), dán nguyên văn.
- Produces: khối "P2 4.4" (venv_seg) và "P2 4.10, các dòng cài đặt" (venv_cls) mà P5a/P5b/P8/P9/P10 tham chiếu.

- [ ] **Step 1: Mục 1–2**

Bảng đầu vào/đầu ra theo bảng thay thế. Mục 2: *Nơi chạy* theo bảng thay thế; "dựng venv" → *"một lần mỗi máy"*. Đoạn "Tạo 3 notebook": kernel `nb_seg` = `~/venvs/venv_seg`, `nb_cls` = `~/venvs/venv_cls` (chọn sau khi làm 4.4/4.10).

- [ ] **Step 2: Mục 3.1 viết lại**

Tiêu đề `### 3.1. Vì sao mỗi venv GPU là một kernel riêng`. 3–5 câu: seg cần torch 2.1.2 + mmcv, cls cần torch 2.4.1 + timm 0.9.16 — không chung một môi trường được; `.venv` Python 3.12 cho phần CPU; mỗi venv cài `ipykernel` nên chọn thẳng làm kernel; lệnh dài vẫn chạy ở terminal để không mất khi đóng notebook.

- [ ] **Step 3: Mục 4.1**

Mở đầu: thay câu "Git chỉ chạy ở máy… Colab clone/pull…" bằng *"Mỗi notebook có một cell mở đầu, chạy đầu mỗi phiên."* 4.1a (`nb_cpu`) với `ROOT` mới (trùng Task 2 step 1), nhãn `# cell: nb_cpu`. 4.1b = cell 0 của `nb_seg` + câu *"Bản `nb_cls` giống hệt, đổi kernel thành `~/venvs/venv_cls` và nhãn thành `# cell: nb_cls`."* Xóa cảnh báo extension và ghi chú token/Drive.

- [ ] **Step 4: Mục 4.3 — tải checkpoint**

```bash
# terminal (máy local)
CK=~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth
test -f $CK || uvx gdown 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB -O $CK
ls -la $CK
```
Kèm: *"Ghi SHA-256: chạy cell 1 của `nb_seg` (sau 4.4). Laptop và máy GPU đều cần file này (tải lại hoặc `rsync` theo `00` mục 5.2)."* Tiêu đề mục bỏ "(Colab `NB_seg`, lưu lên Drive)".

- [ ] **Step 5: Mục 4.4 — dựng `venv_seg`**

````markdown
```bash
# terminal (máy local) — một lần mỗi máy; laptop cũng dựng được để smoke test
test -d ~/PanDerm || git clone -q https://github.com/SiyuanYan1/PanDerm.git ~/PanDerm
git -C ~/PanDerm checkout -q fd7a80748ba7fc3e203fed88f909f4689d0d6f24 && git -C ~/PanDerm log --oneline -1
uv python install 3.10
test -x ~/venvs/venv_seg/bin/python || uv venv -q --python 3.10 --seed ~/venvs/venv_seg
PY=~/venvs/venv_seg/bin/python
# setuptools<81: mmengine/mmseg còn dùng pkg_resources đã bị bỏ ở setuptools 81.
uv pip install -q --python $PY "setuptools<81" wheel
uv pip install -q --python $PY torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install -q --python $PY mmengine==0.10.4 mmcv==2.1.0 mmsegmentation==1.2.2 --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
uv pip install -q --python $PY -r ~/PanDerm/segmentation/requirements.txt openpyxl ipykernel
uv pip install -q --python $PY -e ~/Documents/nckh
$PY -c "import torch, mmcv, mmseg; print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')"
```
````
Giữ các đoạn giải thích phiên bản hiện có của 4.4.

- [ ] **Step 6: Mục 4.5, 4.6, 4.9**

4.5: cell chạy = cell 2 của `nb_seg`. 4.6: cell = cell 3 của `nb_seg`. 4.9: tải ảnh smoke rồi chạy cell 4:
```bash
# terminal (máy local)
mkdir -p ~/nckh_data/smoke && cd ~/nckh_data/smoke && test -d ISIC2018_Task1-2_Validation_Input || (wget -q https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip && unzip -q ISIC2018_Task1-2_Validation_Input.zip && rm ISIC2018_Task1-2_Validation_Input.zip)
```
Thêm: *"Laptop 4 GB VRAM đủ cho smoke test 20 ảnh."* Các tiêu đề bỏ "(trong `NB_seg`)" → "(`nb_seg`)".

- [ ] **Step 7: Mục 4.10 — `venv_cls` + smoke cls**

Cài đặt (giữ nguyên pin):
```bash
# terminal (máy local) — một lần mỗi máy, sau 4.4 (dùng chung ~/PanDerm)
test -x ~/venvs/venv_cls/bin/python || uv venv -q --python 3.10 --seed ~/venvs/venv_cls
PY=~/venvs/venv_cls/bin/python
uv pip install -q --python $PY torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# timm ghim 0.9.16: modeling_finetune.py dùng timm.models.layers / timm.models.registry.
uv pip install -q --python $PY -r ~/PanDerm/classification/requirements.txt timm==0.9.16 ipykernel
uv pip install -q --python $PY -e ~/Documents/nckh
$PY -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')"
```
Smoke test:
```bash
# terminal (máy local)
cd ~/Documents/nckh
PY=~/venvs/venv_cls/bin/python
CK=~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth
RUN=~/nckh_root/runs/$($PY -c "from nckh.runcard import new_run_id; print(new_run_id('smoke_cls'))")
$PY scripts/bench_inference.py --task cls --panderm-dir ~/PanDerm/classification --pretrained $CK --images "$HOME/nckh_data/smoke/ISIC2018_Task1-2_Validation_Input/*.jpg" --n 20 --out-dir $RUN
$PY -m nckh.runcard $RUN --seed 0 --input pretrained=$CK --config task=smoke_cls --config n=20
```
Giữ đoạn "Kỳ vọng". Câu "Phần cài đặt của cell dưới là từ dòng `VENV = …`" → *"Các phase khác ghi 'P2 4.10, các dòng cài đặt' nghĩa là khối cài đặt đầu tiên ở trên."* Xóa khối rclone cuối mục. Cảnh báo "Chưa kiểm chứng trên GPU" giữ nguyên (nói về GPU, không về Colab).

- [ ] **Step 8: Mục 4.11–4.12, 5, 6, 7, 8**

Áp bảng thay thế. Mục 7: bỏ mọi dòng Colab/extension/rclone/Drive; thêm *Khối lỗi chung* và `| Kernel venv_seg/venv_cls không hiện trong VS Code | Chưa cài ipykernel vào venv: chạy lại dòng cài requirements ở 4.4/4.10 |`. Mục 8: môi trường ghi "laptop/máy GPU, kết quả `nvidia-smi` trong run card".

- [ ] **Step 9: Kiểm tra + commit**

```bash
grep -niE "colab|/content|rclone|gdrive|MyDrive|drive\.mount|cloudflared|nckh_drive|\{VENV\}|\{CODE\}" docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md
```
Expected: rỗng. Chạy lại script so khối Task 1 step 6: 4 dòng `OK`.
```bash
git add docs/sv_b/P2_moi_truong_checkpoint_smoke_test.md && git commit -m "docs(sv_b): P2 cài môi trường và smoke test local"
```

---

### Task 5: `P3_du_lieu_manifest_split.md`

**Files:** Modify `docs/sv_b/P3_du_lieu_manifest_split.md` (và file thật trong repo nếu khối `# file:` bị sửa đã tồn tại)

- [ ] **Step 1: Mục 1–2**

ISIC 2018 và 2017 cùng ở `~/nckh_data/ISIC2018/…`, `~/nckh_data/ISIC2017/…`, sinh ở *"Máy local (một lần)"*. Manifest: `~/nckh_root/data/manifests/…`; ISIC 2018 sinh bằng `.venv`, ISIC 2017 bằng `venv_cls` (giữ Python mà lệnh hiện tại dùng). Mục 2: bỏ "mỗi phiên", "đẩy lên Drive", "Colab `NB_cls` (vì P5b chạy ở đó)". Dung lượng: *"Cần ~45 GB trống trong `~/nckh_data` lúc giải nén (zip bị xoá sau khi giải nén nhờ `--delete-zips`)."*

- [ ] **Step 2: Mục 3.1**

Tiêu đề `### 3.1. Vì sao ảnh và manifest ở hai thư mục khác nhau`. Giữ ý "manifest lưu đường dẫn tương đối so với `data_root`", bỏ phần Drive 15 GB/Colab; thêm *"Nhờ vậy cùng manifest dùng được trên laptop và máy GPU dù `~/nckh_data` được tải lại."*

- [ ] **Step 3: Các khối lệnh**

Áp bảng thay thế. Lệnh ISIC 2017 (~dòng 726):
```bash
# terminal (máy local)
cd ~/Documents/nckh
~/venvs/venv_cls/bin/python scripts/prepare_isic2017_cls.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --out-dir ~/nckh_root/data/manifests --delete-zips
```
Xóa đoạn "Chuẩn bị phiên Colab `NB_cls`: push code ở máy…" (~723), thay *"Cần sẵn `venv_cls` (P2 4.10, các dòng cài đặt)."* Comment ~dòng 216 trong khối `# file:` ("ảnh nằm ở /content (mất mỗi phiên)") → "ảnh nằm ở `NCKH_LOCAL_DATA`, có thể tải lại ở máy khác". "Ví dụ (NB_cls hoặc NB_cpu)" → "Ví dụ (`nb_cls` hoặc `nb_cpu`)".

- [ ] **Step 4: Mục 7, kiểm tra, commit**

Mục 7 theo quy ước (không cần khối lỗi GPU). Grep (Task 4 step 9) trên P3 → rỗng. Nếu sửa file thật: `.venv/bin/python -m pytest -q` pass.
```bash
git add docs/sv_b/P3_du_lieu_manifest_split.md && git commit -m "docs(sv_b): P3 tải ISIC và tạo manifest local"
```

---

### Task 6: `P5a_segmentation_isic2018.md`, `P5b_classification_isic2017.md`

**Files:** Modify cả hai.

**Interfaces:**
- Consumes: "P2 4.4" (venv_seg), "P2 4.10, các dòng cài đặt" (venv_cls), "P2 4.6" (patch), ISIC trong `~/nckh_data` (P3).

- [ ] **Step 1: Mục 1–2**

*Đầu ra*: `~/nckh_root/runs/<run_id>/`; *Sinh ở*: `Máy GPU (venv_seg)` / `Máy GPU (venv_cls)`. Mục 2: xóa dòng "Kéo run về máy"; tải ISIC → *"đã có từ P3"*; fine-tune: bỏ "vượt một phiên server Colab", thay *"Chạy trong `tmux`; ước lượng = thời gian/epoch ở pilot × số epoch."*

- [ ] **Step 2: Đoạn resume (P5a ~dòng 45–49)**

Giữ cơ chế `model_checkpoint_0.ckpt` + resume; lý do mới: *"Nếu máy tắt hoặc train bị dừng giữa chừng, chạy lại đúng lệnh với resume để tiếp từ checkpoint cuối."* "nằm trong `--save_name` trên **Drive**" → "nằm trong `--save_name` dưới `~/nckh_root/runs/<run_id>/`".

- [ ] **Step 3: Cell pilot/train/test → bash trong tmux**

Mẫu cho mọi lệnh pilot/train/test (giữ nguyên toàn bộ tham số, chỉ đổi đường dẫn theo bảng thay thế):

````markdown
```bash
# terminal (máy GPU, venv_seg)
tmux new -s p5a          # đã có phiên: tmux attach -t p5a
cd ~/PanDerm/segmentation
PY=~/venvs/venv_seg/bin/python
RUN=~/nckh_root/runs/<run_id>
mkdir -p $RUN
$PY <script và tham số như hiện tại> 2>&1 | tee $RUN/train.log
```
````
P5b: `tmux new -s p5b`, `cd ~/PanDerm/classification`, `PY=~/venvs/venv_cls/bin/python`. Biến Python `COMMON = (...)` trong P5b → biến shell `COMMON="..."` cùng nội dung; `--root_path /content/data/ISIC2017/` → `--root_path $HOME/nckh_data/ISIC2017/`. Lệnh `prepare_isic2018.py` ở P5a (~dòng 363): ISIC 2018 đã tải ở P3 thì xóa, thay *"Dữ liệu đã có ở `~/nckh_data` từ P3."*; nếu P3 không tải phần P5a cần thì giữ lệnh với đường dẫn mới.

Xóa đoạn "Đầu mỗi phiên Colab: push code…" (P5a ~359, P5b ~274). Thay: P5a *"Cần sẵn: `venv_seg` (P2 4.4), patch (P2 4.6), ISIC 2018 (P3)."*; P5b *"Cần sẵn: `venv_cls` (P2 4.10), ISIC 2017 (P3)."*

- [ ] **Step 4: Evaluate và calibration**

`evaluate_seg.py`/`evaluate_cls.py`: nếu lệnh suy luận lại thì `# terminal (máy GPU, venv_*)` với venv đang dùng; nếu chỉ đọc file prediction đã lưu thì `# terminal (máy local)` với `.venv/bin/python`. Calibration P5b: `# cell: nb_cpu`, đường dẫn `~/nckh_root/runs/<run_id>/eval_test/`.

- [ ] **Step 5: Mục 7, kiểm tra, commit**

Mục 7 theo quy ước + *Khối lỗi chung*. Grep trên 2 file → rỗng. Kiểm thủ công: mọi lệnh có `--epochs`/train nằm trong khối có `tmux` và `tee`.
```bash
git add docs/sv_b/P5a_segmentation_isic2018.md docs/sv_b/P5b_classification_isic2017.md
git commit -m "docs(sv_b): P5a/P5b fine-tune trên máy GPU local trong tmux"
```

---

### Task 7: `P6_ghep_cap_va_ridge.md`, `P0_P1_P4_P11_vai_tro_ho_tro.md`

**Files:** Modify cả hai (và file thật nếu khối `# file:` bị sửa đã tồn tại).

- [ ] **Step 1: P6 mục 1–2**

Đầu ra suy luận: `~/nckh_root/runs/<run_id>/masks/*.png`, `seg_features.csv`, `cls_probs.csv`, sinh ở `Máy GPU (venv_seg/venv_cls)`. Xóa dòng "Đẩy run lên Drive" và "Kéo kết quả suy luận về máy". Thêm *"Nếu Ridge chạy trên laptop: chép `runs/<run_id>` theo `00` mục 5.2."*

- [ ] **Step 2: P6 mục 3.1 — cổng Go/No-Go**

Tiêu đề `### 3.1. Cổng Go/No-Go trước khi đưa UQ vào máy`. Câu mở "Chỉ tạo `data/uq/` trên Drive khi…" → "Chỉ tạo `~/nckh_root/data/uq/` khi…". Điều kiện cloud → *"Quyền lưu và xử lý trên máy của thành viên nhóm."* Điều kiện "lưu bản sao trên máy cá nhân (`~/nckh_drive/data/uq`)…" → *"Ghi rõ máy nào được giữ dữ liệu UQ (laptop, máy GPU hay cả hai); chỉ đặt `~/nckh_root/data/uq` trên những máy đó."*

- [ ] **Step 3: P6 khối lệnh**

`infer_images.py --task seg|cls` → bash trong tmux theo mẫu Task 6 step 3 (`tmux new -s p6`, `cd ~/Documents/nckh`, venv tương ứng, `tee` vào `$RUN/infer_seg.log` / `infer_cls.log`). `make_fake_uq.py`, `build_pairs.py`, `train_ridge.py` → `# terminal (laptop, .venv)` với `.venv/bin/python`. Comment ~dòng 611 "…nằm trên Drive (ai đó có thể vô tình đọc)" → "…nằm trong `runs/` (ai đó có thể vô tình đọc)". Đoạn ~1066 "Lấy metadata + ảnh UQ từ Drive về máy…" → câu chuẩn L5 với đích `~/nckh_root/data/uq/`.

- [ ] **Step 4: P0_P1_P4_P11**

~40–53: bỏ điều kiện Colab/Drive và rclone; câu chuẩn L5 với đích `~/nckh_root/data/uq/`; xóa "Nếu điều khoản không cho lưu trên máy: chạy cell này trong `NB_seg` (Colab)…". ~95: xóa ghi chú tương tự. ~113–129: audit ở `~/nckh_root/data/uq/audit/annotator_A/`, `annotator_B/`; rclone → câu chuẩn L5. ~148: *"Laptop (VS Code, `.venv` Python 3.12, CPU) cho xử lý dữ liệu/đánh giá; máy GPU local (`venv_seg`, `venv_cls`) cho fine-tune/suy luận. Ghi `nvidia-smi` vào run card."*

- [ ] **Step 5: Mục 7, kiểm tra, commit**

P6 mục 7 + *Khối lỗi chung*. Grep trên 2 file → rỗng. Sửa file thật thì `pytest` pass.
```bash
git add docs/sv_b/P6_ghep_cap_va_ridge.md docs/sv_b/P0_P1_P4_P11_vai_tro_ho_tro.md
git commit -m "docs(sv_b): P6 và P0/P1/P4/P11 xử lý UQ local, bỏ Drive"
```

---

### Task 8: `P7_kiem_thu_metrics_bootstrap.md`, `P10_tai_lap_hinh_bang.md`

**Files:** Modify cả hai (và file thật nếu khối `# file:` bị sửa đã tồn tại).

- [ ] **Step 1: P7**

Bảng mục 1 theo bảng thay thế. Xóa ~17–24 (rclone + "nếu điều khoản UQ không cho lưu trên máy… chạy trong NB_seg"), thay *"Các run P5a/P5b/P6 nằm trong `~/nckh_root/runs/`; nếu tạo trên máy GPU, chép về theo `00` mục 5.2."* Mục 7: xóa hai dòng rclone.

- [ ] **Step 2: P10**

Xóa ~16–20 (kéo run), thay câu như P7. Comment `matplotlib.use("Agg")  # Colab/CLI không có màn hình` → `# chạy từ terminal không có màn hình`. Mục tái lập ~402–406: *"Từ một máy **mới** (clone repo, làm `00` mục 6b), chỉ làm theo docs:"*; bước 2 *"Nạp checkpoint (máy GPU, `venv_seg`): dựng `venv_seg` (P2 4.4), áp patch (P2 4.6), chạy `bench_inference.py --finetuned …`"*; bước 3 *"demo P9 mục 4.5 (local)"*. Mục 7: xóa hai dòng rclone.

- [ ] **Step 3: Kiểm tra + commit**

Grep trên 2 file → rỗng.
```bash
git add docs/sv_b/P7_kiem_thu_metrics_bootstrap.md docs/sv_b/P10_tai_lap_hinh_bang.md
git commit -m "docs(sv_b): P7/P10 bỏ rclone, tái lập trên máy mới"
```

---

### Task 9: `P8_robustness.md`, `P9_demo_streamlit.md`

**Files:** Modify cả hai (và `demo/app.py` hoặc script thật nếu đã có trong repo và chứa đường dẫn Colab).

- [ ] **Step 1: P8**

Mục 1–2: ảnh suy giảm ở `~/nckh_data/degraded/<mức>`, suy luận bằng venv tương ứng, `robustness.json` ở `~/nckh_root/runs/<p8_run>/`; `evaluate_robustness.py` → `.venv`. Docstring ví dụ ~194–195 → `--data-root ~/nckh_data … --out-dir ~/nckh_data/degraded/blur_1.5`. Cell ~596–616 → bash trong tmux:

````markdown
```bash
# terminal (máy GPU, venv_seg)
tmux new -s p8
cd ~/Documents/nckh
PY=~/venvs/venv_seg/bin/python
RUN=~/nckh_root/runs/<p8_run_id>; mkdir -p $RUN
.venv/bin/python -c "import pandas as pd; m=pd.read_csv('$HOME/nckh_root/data/manifests/isic2018_seg.csv'); m[(m.split=='test')&m.exclude_reason.isna()].to_csv('$RUN/isic18_test.csv', index=False)"
$PY scripts/infer_images.py <tham số như hiện tại> --manifest $RUN/isic18_test.csv --data-root ~/nckh_data --out-dir $RUN/clean_seg 2>&1 | tee $RUN/p8.log
for lv in <danh sách mức như hiện tại>; do
  D=~/nckh_data/degraded/$lv
  $PY scripts/make_degraded.py --manifest $RUN/isic18_test.csv --data-root ~/nckh_data --split test --level $lv --out-dir $D
  $PY scripts/infer_images.py <tham số như hiện tại, đổi đường dẫn> 2>&1 | tee -a $RUN/p8.log
done
```
````
Giữ nguyên danh sách mức, tham số và thứ tự bước của cell gốc (kể cả phần cls nếu có, với `venv_cls`). Xóa "Chuẩn bị: push code ở máy, cell setup Colab…" (~596), thay *"Cần sẵn: `venv_seg`/`venv_cls` (P2), patch, checkpoint đã khóa (P5a/P5b), ISIC trong `~/nckh_data` (P3)."* Thêm *"Có thể xóa `~/nckh_data/degraded/` sau khi có `robustness.json`; ảnh tái tạo được bằng cùng lệnh."*

- [ ] **Step 2: P9**

Kiến trúc (~14–28): *"Trình duyệt ──► Streamlit local (`demo/app.py`, http://localhost:8501) ──► `~/venvs/venv_seg/bin/python scripts/infer_images.py --task seg …` / `~/venvs/venv_cls/bin/python … --task cls …`"*. Xóa cảnh báo extension và mọi nội dung tunnel/`cloudflared` (gồm mục 4.6 nếu chỉ nói tunnel; khi xóa, sửa mọi tham chiếu "mục 4.6" trong file). Mục 4.5 → `### 4.5. Chạy với model thật (máy GPU)`: xóa bước rclone `ridge.joblib` và "cùng server Colab"; *"Cần sẵn `venv_seg` (P2 4.4) và `venv_cls` (P2 4.10) trên cùng máy, cùng `ridge.joblib` từ P6 (`~/nckh_root/runs/<p6_uq_run_id>/ridge/`)."* Lệnh:
```bash
# terminal (máy GPU, .venv)
cd ~/Documents/nckh && .venv/bin/streamlit run demo/app.py
```
(giữ tham số/biến môi trường mà lệnh streamlit hiện có). Tắt: `Ctrl+C` thay `!pkill -f "streamlit run"`. ~318 `~/nckh_drive/runs/…` → `~/nckh_root/runs/…`. Nếu `demo/app.py` có trong repo và chứa `/content`, đổi default trong file thật và chạy `pytest`.

- [ ] **Step 3: Mục 7, kiểm tra, commit**

Mục 7 + *Khối lỗi chung*. Grep trên 2 file → rỗng. `grep -n "4\.6" docs/sv_b/P9_demo_streamlit.md` → không còn tham chiếu tới mục đã xóa.
```bash
git add docs/sv_b/P8_robustness.md docs/sv_b/P9_demo_streamlit.md
git commit -m "docs(sv_b): P8 robustness và P9 demo chạy local, bỏ tunnel"
```

---

### Task 10: `Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md`

**Files:** Modify `Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md` (chỉ các dòng dưới)

- [ ] **Step 1: Sửa từng dòng**

Danh sách: `grep -niE "colab|cloud|đám mây|drive" Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md`. Quy tắc:
- Ô *Nơi chạy* "Google Colab có GPU…" (~62–63) → "Máy GPU local (venv riêng cho mỗi nhánh)".
- ~64 "Colab chỉ khi điều khoản UQ cho phép" → xóa vế đó.
- ~69 "**Cách dùng Colab:** …" → "**Cách dùng máy GPU:** fine-tune và suy luận cần GPU chạy trên máy GPU local; mỗi nhánh PanDerm một venv riêng theo hướng dẫn của repo; chạy trong `tmux`, lưu cấu hình, log và checkpoint định kỳ vào `runs/` để khôi phục khi bị dừng. Nếu máy GPU không đủ tài nguyên, tiếp tục xử lý dữ liệu/Ridge trên CPU và chuyển huấn luyện sang GPU của trường. Không đổi mô hình hay dùng tập test để bù cho giới hạn phần cứng."
- ~73 "Quyền dữ liệu": "chỉ dùng ảnh ISIC trên Colab sau khi kiểm tra điều khoản" → "kiểm tra điều khoản ISIC trước khi dùng"; câu về dịch vụ đám mây cho UQ → "xử lý UQ chỉ trên máy được đơn vị dữ liệu cho phép".
- Bảng tuần ~107, 110, 111, 113: "Colab GPU"/"pilot GPU trên Colab" → "máy GPU local".
- ~169 "dùng dịch vụ đám mây như Colab;" → "lưu và xử lý trên máy của thành viên nhóm;".
- ~259 "tương thích với Colab" → "tương thích với driver máy GPU"; ~263 "Trên Colab, chạy pilot ngắn…" → "Trên máy GPU, chạy pilot ngắn…"; ~264 xóa dòng về FAQ Colab.
- ~363, 368, 388, 546: "Google Colab GPU"/"notebook Colab" → "máy GPU local"/"venv riêng"; "Không tải ảnh UQ lên Colab nếu…" → xóa câu.
- ~765 "GPU/Colab không ổn định" → "GPU không đủ hoặc không ổn định".
- ~842 tài liệu tham khảo FAQ Colab: thay nội dung bằng "(đã bỏ — không dùng Colab)" để giữ số thứ tự các mục sau.

- [ ] **Step 2: Kiểm tra + commit**

`grep -niE "colab" Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md` → chỉ còn dòng "(đã bỏ — không dùng Colab)".
```bash
git add Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md && git commit -m "docs(ke_hoach): nơi chạy GPU là máy GPU local"
```

---

### Task 11: Kiểm tra cuối

- [ ] **Step 1: Grep tiêu chí spec §1**

```bash
cd ~/Documents/nckh
grep -rniE "colab|/content|rclone|gdrive|MyDrive|drive\.mount|cloudflared|nckh_drive" docs/sv_b notebooks src scripts tests
```
Expected: đúng 1 dòng (câu lịch sử ở `00`).

- [ ] **Step 2: `~` không mở rộng**

```bash
grep -nE "['\"]~/|=~/" docs/sv_b/*.md -r
```
Expected: không có kết quả trong khối ```python; trong bash chỉ chấp nhận dạng gán biến `X=~/...` ở đầu dòng (shell có mở rộng), không chấp nhận `--opt=~/` hay `"~/..."`.

- [ ] **Step 3: Nhãn khối**

```bash
grep -nE "^# (cell|terminal)" docs/sv_b/*.md | grep -vE "# cell: nb_(cpu|seg|cls)|# terminal \((laptop, \.venv|máy GPU, (venv_seg|venv_cls|\.venv)|máy local)\)"
```
Expected: rỗng.

- [ ] **Step 4: Test, notebook, khối file**

```bash
.venv/bin/python -m pytest -q
python3 -c "import json; [json.load(open(f'notebooks/{n}.ipynb')) for n in ('nb_cpu','nb_seg')]; print('OK')"
```
Expected: tất cả pass (≥ 21); `OK`. Script so khối Task 1 step 6: 4 dòng `OK`.

- [ ] **Step 5: Đọc chéo**

Cột *Runtime* ở `00` mục 2 khớp mục 2 *Nơi chạy* của từng phase; mọi tham chiếu "P2 4.4", "P2 4.10", "P2 4.6", "`00` mục 5.2/6b" trỏ tới mục có thật (`grep -n "mục 5.2\|mục 6b\|P2 4\.\(4\|6\|10\)" docs/sv_b/*.md`).

- [ ] **Step 6: `git status`**

Expected: sạch.
