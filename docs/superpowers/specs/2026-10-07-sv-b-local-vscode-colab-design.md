# Spec — Chuyển bộ hướng dẫn SV B sang "Local VS Code + Colab qua extension"

**Ngày:** 07/10/2026 · **Phạm vi:** chỉ sửa tài liệu trong `docs/sv_b/` (11 file). **Không sửa code** (`src/`, `tests/`, `notebooks/`, `pyproject.toml`).

## 1. Mục tiêu

- Mọi việc soạn code, git, xử lý dữ liệu không cần GPU, test, đánh giá, demo: chạy **trên máy cá nhân, trong VS Code**.
- Việc train hoặc chạy model trên GPU: chạy trên **kernel Colab nối từ VS Code bằng extension "Google Colab"** (`googlecolab/colab-vscode`). Không dùng colab.research.google.com.
- Mỗi file mà SV B tạo hoặc sinh ra phải được ghi rõ **đường dẫn đầy đủ** và **nơi tạo** (máy cá nhân hay Colab/Drive).

**Tiêu chí thành công:** đọc bất kỳ file phase nào cũng trả lời được ngay ba câu: cell/lệnh này chạy ở đâu, file này nằm ở đâu, kết quả đi về máy bằng cách nào. Trong `docs/sv_b/` không còn hướng dẫn nào bảo mở Colab web, dùng Colab Secrets, hay đặt repo trên Drive.

## 2. Quyết định đã chốt với người dùng

| # | Quyết định |
|---|---|
| D1 | Google Drive (`MyDrive/NCKH_PanDerm`) là cầu nối file bền giữa máy và Colab. Máy đồng bộ bằng `rclone` (remote tên `gdrive`), thư mục bản sao `~/nckh_drive`. |
| D2 | Ảnh ISIC có ở cả hai nơi: máy tải một lần vào `~/nckh_data`; Colab tải từ S3 về `/content/data` mỗi phiên. P8 tạo ảnh suy giảm ngay trên Colab. |
| D3 | Repo `NeuroDev204/nckh` để public. Git chỉ chạy ở máy, Colab chỉ `git clone`/`git pull` về `/content/nckh` (chỉ đọc). Bỏ `GH_TOKEN`, Colab Secrets và quy tắc "chỉ NB_cpu chạy git". |
| D4 | Giữ cổng Go/No-Go UQ: ảnh UQ chỉ lên Drive/Colab khi có văn bản cho phép xử lý trên cloud. |
| D5 | Ba file cũ `docs/Huong_dan_SV_B_Tuan_1_*.md` đã gắn nhãn lỗi thời, để nguyên. |
| D6 | Doc chỉ mô tả code dùng biến môi trường `NCKH_ROOT`, `NCKH_LOCAL_DATA` (đã có trong `paths.py`). Không đổi nội dung các khối `# file:`, chỉ đổi phần chú thích vị trí và các cell chạy. |

## 3. Kiến trúc hai nơi chạy (nội dung đưa vào `00` mục 4–6)

| | Máy cá nhân | Colab (qua extension) |
|---|---|---|
| Kernel | `NB_cpu` dùng `.venv` local (`uv`, `-e .[demo,test]` + torch CPU) | `NB_seg`, `NB_cls`: *Select Kernel → Colab → New Colab Server*, chọn GPU |
| Code | `<repo>` = thư mục SV B clone repo (ví dụ `~/Documents/nckh`), sửa bằng VS Code | `/content/nckh`, clone/pull mỗi phiên |
| `NCKH_ROOT` | `~/nckh_drive` | `/content/drive/MyDrive/NCKH_PanDerm` (mount bằng lệnh *Colab: Mount Google Drive to Server*) |
| `NCKH_LOCAL_DATA` | `~/nckh_data` | `/content/data` |
| Venv GPU | — | `/content/venv_seg`, `/content/venv_cls` (giữ như cũ) |

Vòng làm việc: sửa code ở máy → `git push` → trên Colab `git pull` + chạy → kết quả ghi lên Drive → ở máy `rclone copy gdrive:NCKH_PanDerm/runs ~/nckh_drive/runs` → đánh giá ở máy. Chiều ngược lại: manifest/config tạo ở máy → `rclone copy ~/nckh_drive/data/manifests gdrive:NCKH_PanDerm/data/manifests`.

Phân phase: **Máy** = P0, P1, P3, P4, P6 (ghép cặp, Ridge), P7, P9 (chế độ giả), P10, P11, pytest. **Colab** = P2 (smoke test, benchmark, cài venv GPU), P5a, P5b, P6 (suy luận mask/xác suất), P8 (tạo ảnh suy giảm + suy luận), P9 (model thật).

## 4. Quy ước ghi vị trí file (áp dụng mọi file phase)

1. **Khối code tạo file:** giữ dòng `# file: <đường dẫn tương đối từ gốc repo>`. Ngay trên khối, thêm một dòng:
   `📁 **Tạo trên máy cá nhân:** <repo>/src/nckh/paths.py`
   Mọi file code đều tạo trên máy. Không có file code nào tạo trên Colab.
2. **Khối cell:** đổi nhãn thành `# cell: NB_cpu (máy cá nhân)`, `# cell: NB_seg (Colab GPU)`, `# cell: NB_cls (Colab GPU)`. Lệnh shell chạy ở máy dùng khối ```bash có dòng đầu `# terminal VS Code (máy cá nhân)`.
3. **Bảng "Đầu vào / Đầu ra" mục 1 của mỗi phase:** mỗi file đầu ra ghi đường dẫn đầy đủ và cột *Sinh ở*. Ví dụ: `~/nckh_drive/data/manifests/isic2018_seg.csv` (máy, đẩy lên Drive) hoặc `MyDrive/NCKH_PanDerm/runs/<run_id>/...` (Colab, kéo về máy bằng rclone).
4. **Mục 2 "Chạy ở đâu":** cột *Notebook* đổi thành *Nơi chạy* (`Máy` / `Colab NB_seg` / `Colab NB_cls`). Thêm dòng *Đồng bộ trước/sau* nếu phase cần rclone.
5. **`00` mục 8 (bản đồ file):** thêm cột *Vị trí tạo* với đường dẫn đầy đủ từ `<repo>/`, kèm cây thư mục repo hoàn chỉnh (`src/nckh/`, `scripts/`, `tests/`, `configs/`, `patches/`, `demo/`, `notebooks/`).

## 5. Thay đổi theo file

| File | Thay đổi |
|---|---|
| `00_tong_quan_va_lo_trinh.md` | Dòng giới thiệu; cột Runtime của lộ trình → `Máy`/`Colab`; mục 4 viết lại thành bố cục 3 nơi (repo trên máy, `~/nckh_drive` ↔ Drive, `/content` trên Colab); mục 5 thay "Drive ↔ GitHub" bằng "Đồng bộ code (git ở máy) và dữ liệu (rclone)"; mục 6 thành bảng §3; thêm mục *Cài đặt máy cá nhân* (VS Code, extension Python/Jupyter/Google Colab, `uv`, `.venv`, `rclone config`, file `.env` với `NCKH_ROOT`/`NCKH_LOCAL_DATA`); mục 8 theo §4.5; mục 9 đổi tên mục 7 của khung phase. |
| `P2` | Mục 2/4.1: bỏ cell mount + token + Secrets; thêm (a) kết nối kernel Colab qua extension, (b) cell Colab: mount Drive bằng lệnh extension, `git clone`/`pull` vào `/content/nckh`, đặt biến môi trường, (c) tạo `NB_*.ipynb` ngay trong `<repo>/notebooks/` trên máy. Các file `pyproject.toml`, `paths.py`, …: tạo trên máy. Tải checkpoint: trên Colab về Drive. Pytest: chạy trên máy. Mục 7 thêm lỗi extension. |
| `P3` | Tải ISIC 2018 + manifest + pytest chạy trên máy với `~/nckh_data`, sau đó rclone manifest lên Drive. ISIC 2017 cho P5b vẫn tải trên Colab (`NB_cls`); CSV nhãn ISIC 2017 tạo ở đó rồi rclone về máy. Mục 3.1 cập nhật lý do. |
| `P5a`, `P5b` | Cell Colab dùng `/content/nckh`; thêm bước `git pull` đầu phiên; đánh giá cuối (`evaluate_seg.py`/`evaluate_cls.py`) chạy ở máy nếu chỉ cần prediction đã lưu, hoặc trên Colab nếu cần suy luận — ghi rõ từng trường hợp. Ghi rõ vị trí từng output. |
| `P6` | Tách phần Colab (suy luận mask/xác suất) và phần máy (`make_fake_uq.py`, `build_pairs.py`, `train_ridge.py`); thêm lệnh rclone giữa các bước. |
| `P7`, `P10`, `P0_P1_P4_P11` | `NB_cpu` → máy; thêm lệnh rclone kéo kết quả. |
| `P8` | Tạo ảnh suy giảm + suy luận trên Colab; `evaluate_robustness.py` trên máy. |
| `P9` | Chế độ giả chạy trên máy (localhost:8501, không tunnel); demo với model thật chạy trên Colab NB_seg + tunnel cloudflared (chạy model ⇒ Colab). |

Tất cả file: mục 7 đổi tên thành **"Lỗi thường gặp (máy / Colab extension)"**. Các lỗi mới: server Colab bị ngắt/thu hồi, Drive chưa mount (`/content/drive` trống), `userdata.get` không dùng được trong extension, rclone token hết hạn, quên `git push` trước khi `pull` trên Colab.

## 6. Ngoài phạm vi

- Không sửa code, notebook hay test hiện có.
- Không sửa 3 file `docs/Huong_dan_SV_B_Tuan_1_*.md`, `Ke_hoach_...md`, `De_cuong_...docx`.
- Không đổi nội dung kỹ thuật (phiên bản torch/mmcv, metric, split, protocol).

## 7. Kiểm tra sau khi sửa

- `grep -n "Secrets\|GH_TOKEN\|userdata\|AUTH_URL\|{ROOT}/nckh" docs/sv_b/*.md` chỉ còn trong mục lỗi thường gặp (giải thích vì sao không dùng).
- Mọi khối `# file:` (50 khối) đều có dòng `📁 **Tạo trên máy cá nhân:**` ngay phía trên.
- Mọi khối `# cell:` đều có nhãn nơi chạy.
- Đọc chéo `00` ↔ từng phase: bảng phân phase khớp mục 2 của từng file.
- Hành vi extension (mount Drive, terminal, nhiều server) theo User Guide https://github.com/googlecolab/colab-vscode/wiki/User-Guide. Chỗ nào chưa chạy thử thì gắn `> ⚠️ Chưa kiểm chứng trên extension`.
