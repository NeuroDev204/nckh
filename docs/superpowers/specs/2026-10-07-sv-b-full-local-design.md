# Spec — Chuyển bộ hướng dẫn SV B sang chạy hoàn toàn local

**Ngày:** 07/10/2026 · **Thay thế:** `2026-10-07-sv-b-local-vscode-colab-design.md` (D1–D3: Drive, rclone, Colab extension không còn dùng).

## 1. Mục tiêu

- Mọi bước của SV B chạy trên máy local, không dùng Colab, Google Drive hay rclone.
- Hai máy cùng bố cục: **laptop** (RTX 3050 4GB, chạy P0–P4 và mọi việc CPU) và **máy GPU** (khoảng 16GB, chạy fine-tune/suy luận P5a, P5b, P6 suy luận, P8, P9 model thật). Hướng dẫn giả định máy GPU đủ cho cấu hình gốc của PanDerm; không viết riêng cấu hình cho 4GB.
- Logic nằm trong `.py` (`src/nckh/`, `scripts/`) có pytest. Notebook chỉ để xem kết quả. Train dài chạy ở terminal trong `tmux`.

**Tiêu chí thành công:** `grep -niE "colab|/content|rclone|gdrive|MyDrive|drive.mount|cloudflared|nckh_drive" docs/sv_b/*.md notebooks/*.ipynb src scripts tests` không còn kết quả, trừ đúng một câu lịch sử trong `00` nói rằng bộ hướng dẫn trước đây dùng Colab. Đọc bất kỳ file phase nào cũng trả lời được: lệnh chạy trên máy nào, bằng Python nào, file nằm ở đâu.

## 2. Quyết định đã chốt

| # | Quyết định |
|---|---|
| L1 | Bỏ Drive và rclone. Mọi file nằm trên ổ local theo bố cục §3. |
| L2 | Code đi giữa hai máy bằng `git push/pull`. Dữ liệu/kết quả: `rsync` hoặc tải lại bằng lệnh có sẵn trong P2/P3 (một mục ngắn trong `00`). |
| L3 | Python thuần là nguồn chính; notebook mỏng, chọn kernel thẳng theo venv. Bỏ mẹo `!{VENV}/bin/python`. |
| L4 | Sửa cả code cho khớp docs: default `paths.py`, docstring scripts, `nb_cpu`, `nb_seg`. `Ke_hoach...md` chỉ sửa dòng "nơi chạy" và câu về cloud. 3 file `docs/Huong_dan_SV_B_Tuan_1_*.md` để nguyên. |
| L5 | Trao đổi file giữa SV A và SV B (mask audit): hướng dẫn chỉ ghi "SV A gửi thư mục cho SV B bằng kênh được điều khoản UQ cho phép (USB, ổ mạng trường…)", SV B chép vào đúng thư mục. Không gắn với công cụ cụ thể. |
| L6 | Giữ cổng Go/No-Go UQ; điều kiện "xử lý trên cloud" đổi thành "lưu và xử lý trên máy của thành viên nhóm". |
| L7 | Không đổi nội dung kỹ thuật: phiên bản torch/mmcv/timm, commit PanDerm `fd7a807`, patch, metric, split, protocol, số epoch. |

## 3. Bố cục trên mỗi máy

```
~/Documents/nckh/      <repo>; .venv (Python 3.12, CPU)
~/nckh_root/           NCKH_ROOT: checkpoints/, data/manifests/, data/uq/, runs/<run_id>/
~/nckh_data/           NCKH_LOCAL_DATA: ảnh ISIC 2017/2018 giải nén; degraded/ cho P8
~/PanDerm/             clone upstream, khóa fd7a807 + patches/panderm_base_seg.patch
~/venvs/venv_seg/      Python 3.10, torch 2.1.2 cu118, mmengine/mmcv/mmseg (P2, P5a, P6, P8, P9)
~/venvs/venv_cls/      Python 3.10, torch 2.4.1 cu118, timm 0.9.16 (P2, P5b, P6, P8, P9)
```

| Kernel / Python | Dùng cho |
|---|---|
| `<repo>/.venv` (kernel `nb_cpu`) | P3 manifest/split, P6 ghép cặp + Ridge, P7, P9 chế độ giả, P10, pytest |
| `~/venvs/venv_seg` (kernel `nb_seg`) | smoke test seg, P5a, suy luận mask P6/P8/P9 |
| `~/venvs/venv_cls` (kernel `nb_cls`) | smoke test cls, P5b, xác suất lớp P6/P8/P9 |

Môi trường và dữ liệu cài/tải một lần cho mỗi máy. `tmux` cho mọi lệnh train hoặc suy luận dài, log ghi `~/nckh_root/runs/<run_id>/train.log`.

## 4. Thay đổi code

| File | Thay đổi |
|---|---|
| `src/nckh/paths.py` | `DEFAULT_ROOT = str(Path.home() / "nckh_root")`, `DEFAULT_LOCAL_DATA = str(Path.home() / "nckh_data")`; comment dòng 35 bỏ chữ Colab. Biến môi trường vẫn ghi đè. |
| `tests/test_paths_runcard.py` | Thêm test: không có `NCKH_ROOT`/`NCKH_LOCAL_DATA` thì trả về `~/nckh_root`, `~/nckh_data`. |
| `scripts/inspect_checkpoint.py`, `scripts/bench_inference.py` | Docstring/ví dụ/help sang `~/venvs/venv_seg/bin/python`, `~/PanDerm/...`, `~/nckh_data/...`, `~/nckh_root/...`. Không đổi logic. |
| `src/nckh/runcard.py:101` | Comment bỏ chữ Colab. |
| `notebooks/nb_cpu.ipynb` | `ROOT` → `~/nckh_root`, bỏ chú thích rclone. |
| `notebooks/nb_seg.ipynb` | Viết lại cho kernel `venv_seg` local: cell setup (biến đường dẫn, `sys.path`), sha256 checkpoint, inspect, smoke test, xem overlay. Bỏ `drive.mount`, `git clone/pull`, cài đặt venv (chuyển sang bash trong P2). |

Không thêm: file `.env`, `nb_cls.ipynb` trong repo (SV B tự tạo theo P2), tunnel.

## 5. Thay đổi docs

**Quy ước chung (mọi file `docs/sv_b/`):**
1. Nhãn khối: `# terminal (laptop, .venv)`, `# terminal (máy GPU, venv_seg)`, `# terminal (máy GPU, venv_cls)`, `# cell: nb_cpu`, `# cell: nb_seg`, `# cell: nb_cls`. Bước chạy được trên cả hai máy ghi `máy local`.
2. Cell `!{VENV}/bin/python ...` → lệnh bash ở terminal với đường dẫn Python đầy đủ của venv. Notebook chỉ giữ cell xem kết quả.
3. Đổi đường dẫn: `{ROOT}`/`MyDrive/NCKH_PanDerm`/`~/nckh_drive` → `~/nckh_root`; `/content/data` → `~/nckh_data`; `/content/PanDerm` → `~/PanDerm`; `/content/nckh`/`{CODE}` → `<repo>`; `/content/venv_*` → `~/venvs/venv_*`.
4. Xóa: lệnh `rclone`, `drive.mount`, `git pull` đầu phiên, mục "kéo kết quả về máy", cảnh báo runtime/server bị ngắt.
5. Mục 7 → **"Lỗi thường gặp (máy local)"**: bỏ lỗi Colab/rclone; thêm CUDA OOM (hạ batch size, ghi vào run card), driver/CUDA không khớp wheel cu118, mất phiên `tmux`.
6. Khối `# file:` phải trùng nội dung với file thật trong repo sau khi sửa §4.

**Theo file:**

| File | Thay đổi |
|---|---|
| `00_tong_quan_va_lo_trinh.md` | Cột Runtime → `Laptop`/`Máy GPU`. Mục 4 = bố cục §3. Mục 5 = git + mục "Chuyển sang máy GPU" (`rsync -a ~/nckh_root ~/nckh_data <máy-gpu>:~/` hoặc tải lại). Mục 6 = bảng kernel §3. Mục 6b = cài một lần: VS Code + extension Python/Jupyter, `uv`, `.venv`, `~/venvs/`, `tmux`. Mục 8 bỏ cột Colab. |
| `P2` | 3.1 viết lại (vì sao mỗi venv là một kernel riêng). 4.1 cell setup local. 4.3–4.10: tải checkpoint vào `~/nckh_root/checkpoints`, clone `~/PanDerm` + checkout, dựng `venv_seg`/`venv_cls`, inspect, patch, smoke test → bash một lần. Khối `paths.py`/test khớp §4. |
| `P3` | ISIC 2018 và 2017 tải một lần vào `~/nckh_data` (cả CSV nhãn 2017 tạo local). Mục 3.1 thay bằng ghi chú dung lượng đĩa. |
| `P5a`, `P5b` | Pilot, train, test, `evaluate_*` chạy terminal + `tmux` trên máy GPU; output `~/nckh_root/runs/<run_id>/`. Resume giữ như tính năng upstream, bỏ lý do "runtime ngắt". |
| `P6` | Suy luận UQ và Ridge cùng máy, bỏ bước đẩy/kéo. Cổng 3.1 theo L6. |
| `P7`, `P10` | Bỏ `rclone`, bỏ ghi chú "chạy trong NB_seg nếu không được lưu trên máy". P10 mục tái lập: "máy mới, làm theo `00` mục 6b". |
| `P8` | Ảnh suy giảm sinh vào `~/nckh_data/degraded/<mức>`; suy luận bằng venv tương ứng; `evaluate_robustness.py` bằng `.venv`. |
| `P9` | Chế độ giả và model thật đều chạy `streamlit` local, `localhost:8501`; bỏ `cloudflared`. Model thật cần cả `venv_seg` và `venv_cls` trên máy GPU. |
| `P0_P1_P4_P11` | Bỏ phương án xử lý UQ trên Colab; trao đổi mask audit theo L5; môi trường ghi lại là laptop/máy GPU. |
| `Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md` | Chỉ sửa ô/câu "nơi chạy" (Colab GPU → máy GPU local) và các câu về quyền cloud. |

## 6. Ngoài phạm vi

- 3 file `docs/Huong_dan_SV_B_Tuan_1_*.md`, `De_cuong_...docx`, `docs/hop/*`, các spec/plan cũ.
- Cấu hình riêng cho GPU 4GB.
- Thay đổi logic code ngoài §4.

## 7. Kiểm tra

- Lệnh grep ở §1 sạch.
- `.venv/bin/python -m pytest -q` pass (gồm test default mới).
- Mỗi khối `# file:` trong docs trùng file thật (so bằng script trích khối + `diff`).
- Mọi khối lệnh/cell có nhãn nơi chạy theo §5.1.
- Đọc chéo bảng phân phase ở `00` với mục 2 của từng phase.
- Notebook mở được, JSON hợp lệ (`python -c "import json; json.load(open(...))"`).
