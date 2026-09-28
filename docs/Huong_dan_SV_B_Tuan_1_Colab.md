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
