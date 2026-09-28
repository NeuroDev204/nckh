# Hướng dẫn chi tiết: SV B Triển khai Tuần 1 trên Google Colab

Tài liệu này cung cấp toàn bộ mã nguồn từ số 0 để bạn (SV B) copy-paste vào các cell của Google Colab nhằm hoàn thành Phase 1 & 2 của dự án.

## 1. Mục tiêu Tuần 1
- **Thiết lập môi trường:** Clone mã nguồn PanDerm và cài đặt đầy đủ các thư viện.
- **Tải Checkpoint:** Nạp thành công PanDerm Base.
- **Pilot Inference (Smoke Test):** Chạy thuật toán mô phỏng trên 20 ảnh và xuất kết quả mask overlay thành công mà không lỗi bộ nhớ (OOM).

## 2. Chuẩn bị Môi trường Google Colab & Google Drive

**Bước 2.1: Mở Notebook và bật GPU**
1. Vào [Google Colab](https://colab.research.google.com/), chọn **New Notebook**.
2. Trên thanh menu, chọn **Runtime** -> **Change runtime type**.
3. Ở mục **Hardware accelerator**, chọn **T4 GPU** (hoặc A100/V100 nếu có) và nhấn **Save**.

**Bước 2.2: Kết nối với Google Drive**
Viết đoạn code sau vào Cell đầu tiên để lưu file không bị mất sau khi đóng Colab:
```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Tạo thư mục làm việc cho dự án
WORK_DIR = '/content/drive/MyDrive/NCKH_PanDerm'
os.makedirs(WORK_DIR, exist_ok=True)
print(f"Thư mục làm việc: {WORK_DIR}")
```

*Giải thích chi tiết:*
- `drive.mount('/content/drive')`: Hàm gắn kết (mount) Google Drive cá nhân vào môi trường Colab tại thư mục `/content/drive`.
- `WORK_DIR`: Biến chuỗi xác định đường dẫn thư mục làm việc cho dự án trên Google Drive (`/content/drive/MyDrive/NCKH_PanDerm`), đảm bảo mã nguồn và checkpoint không bị xóa sau khi phiên Colab kết thúc.
- `os.makedirs(WORK_DIR, exist_ok=True)`: Hàm tạo thư mục lưu trữ; tham số `exist_ok=True` giúp tránh phát sinh lỗi nếu thư mục đã tồn tại từ trước.

## 3. Clone mã nguồn & Cài đặt thư viện

Chạy cell dưới đây để tải mã nguồn chính thức và cài đặt các thư viện lõi (PyTorch, MMSegmentation, torchvision). 
> **Lưu ý:** Chạy `pip freeze` ở cuối cell để ghi lại log các phiên bản đang dùng.

```bash
# Di chuyển vào thư mục dự án
%cd /content/drive/MyDrive/NCKH_PanDerm

# Clone repo PanDerm (nếu chưa có)
!if [ ! -d "PanDerm" ]; then git clone https://github.com/YingWen-wang/PanDerm.git; fi

# Cài đặt PyTorch và Torchvision phù hợp với CUDA của Colab
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt openmim và MMSegmentation theo hướng dẫn nhánh Segmentation
!pip install -U openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
!pip install "mmsegmentation>=1.0.0"

# Ghi lại cấu hình môi trường
!pip freeze > requirements_colab.txt
!echo "Đã lưu requirements_colab.txt"
```

*Giải thích chi tiết:*
- `%cd /content/drive/MyDrive/NCKH_PanDerm`: Lệnh magic của Colab/IPython để chuyển thư mục làm việc hiện tại sang thư mục dự án trên Drive.
- `git clone https://github.com/YingWen-wang/PanDerm.git`: Tải toàn bộ mã nguồn mô hình PanDerm từ kho lưu trữ GitHub chính thức; cấu trúc `if [ ! -d "PanDerm" ]` giúp kiểm tra và chỉ clone nếu thư mục chưa tồn tại nhằm tránh ghi đè hoặc trùng lặp.
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`: Cài đặt bộ ba PyTorch, Torchvision, Torchaudio phiên bản tương thích với CUDA 12.1 phục vụ GPU của Colab.
- `openmim`, `mmengine`, `mmcv`, `mmsegmentation`: Các thư viện trong hệ sinh thái OpenMMLab dành cho bài toán phân vùng tổn thương da (Semantic Segmentation). `openmim` hỗ trợ cài đặt các phiên bản MMCV (>=2.0.0) và MMSegmentation (>=1.0.0) tương thích với phiên bản PyTorch đã nạp.
- `pip freeze > requirements_colab.txt`: Xuất danh sách toàn bộ thư viện cùng số hiệu phiên bản ra file `requirements_colab.txt` để làm minh chứng và kiểm soát tính tái lập (reproducibility).

## 4. Chuẩn bị Dữ liệu mẫu (Pilot) & Checkpoint

**Bước 4.1: Tải Checkpoint PanDerm Base**
```bash
%cd /content/drive/MyDrive/NCKH_PanDerm/PanDerm
!mkdir -p checkpoints
# Tải checkpoint (URL minh họa - thay bằng URL thực tế nếu cần)
!wget -O checkpoints/panderm_base.pth "https://example.com/panderm_base_link_thuc_te.pth"
```

*Giải thích chi tiết:*
- `%cd /content/drive/MyDrive/NCKH_PanDerm/PanDerm`: Di chuyển vị trí làm việc vào thư mục mã nguồn PanDerm.
- `!mkdir -p checkpoints`: Tạo thư mục con `checkpoints` để chứa các file trọng số mô hình đã huấn luyện trước; cờ `-p` giúp bỏ qua lệnh nếu thư mục đã có.
- `!wget -O checkpoints/panderm_base.pth ...`: Lệnh tải file trọng số từ đường dẫn trực tiếp và lưu với tên `panderm_base.pth` vào thư mục `checkpoints`.

**Bước 4.2: Khởi tạo ảnh Toy Data (Ví dụ 20 ảnh)**
Để test mà chưa cần xin quyền tải toàn bộ bộ ảnh ISIC lớn, đoạn code sau sẽ lấy 20 ảnh giả lập (hoặc ảnh test mẫu) để kiểm tra pipeline.
```python
import os
import shutil
import urllib.request

img_dir = '/content/drive/MyDrive/NCKH_PanDerm/toy_data'
os.makedirs(img_dir, exist_ok=True)

# Tạo 20 ảnh test (Tải một ảnh mẫu nguồn mở rồi copy ra 20 bản để thử nghiệm memory)
sample_url = "https://upload.wikimedia.org/wikipedia/commons/6/6c/Melanoma.jpg"
sample_path = os.path.join(img_dir, "sample_0.jpg")
if not os.path.exists(sample_path):
    # Thiết lập User-Agent trình duyệt để tránh lỗi HTTP 403 Forbidden từ Wikimedia
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(sample_url, sample_path)

for i in range(1, 21):
    shutil.copy(sample_path, os.path.join(img_dir, f"test_img_{i}.jpg"))
print(f"Đã chuẩn bị 20 ảnh toy data tại {img_dir}")
```

*Giải thích chi tiết:*
- `img_dir`: Biến chuỗi định nghĩa đường dẫn lưu trữ tập dữ liệu thử nghiệm (toy dataset) tại `/content/drive/MyDrive/NCKH_PanDerm/toy_data`.
- `os.makedirs(img_dir, exist_ok=True)`: Hàm tạo thư mục lưu trữ ảnh; `exist_ok=True` bảo đảm không phát sinh lỗi nếu thư mục đã tồn tại.
- `sample_url` & `sample_path`: Địa chỉ URL của ảnh mẫu tổn thương da (Melanoma) trên Wikimedia Commons và đường dẫn lưu trữ file ảnh gốc `sample_0.jpg`.
- `opener.addheaders = [('User-Agent', 'Mozilla/5.0')]` & `urllib.request.install_opener(opener)`: Bổ sung header User-Agent mô phỏng trình duyệt cho `urllib` để vượt qua bộ lọc chống bot của Wikimedia (ngăn chặn lỗi HTTP 403 Forbidden khi tải ảnh).
- `urllib.request.urlretrieve(sample_url, sample_path)`: Tải ảnh từ URL về lưu trực tiếp thành file tại `sample_path`.
- `shutil.copy(sample_path, ...)`: Nhân bản file ảnh mẫu gốc thành 20 bản sao từ `test_img_1.jpg` đến `test_img_20.jpg` để tạo tập dữ liệu mô phỏng 20 ảnh phục vụ kiểm tra vòng lặp suy luận hàng loạt (inference loop) và đo lường VRAM tiêu thụ.
