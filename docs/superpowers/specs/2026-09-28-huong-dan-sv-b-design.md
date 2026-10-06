# Design Spec: Hướng dẫn triển khai Tuần 1 cho SV B (Google Colab)

## 1. Context and Goals
- **Context:** Dự án nghiên cứu "Ứng dụng PanDerm để phân đoạn tổn thương và dự báo thay đổi". SV B chịu trách nhiệm về kỹ thuật, pipeline và mô hình. Tuần 1 cần hoàn thành Phase 1 & 2: Cài đặt môi trường, tải checkpoint, và chạy thử nghiệm (smoke test).
- **Goal:** Tạo một file Markdown (`Huong_dan_SV_B_Tuan_1_Colab.md`) hoàn chỉnh, chi tiết, cung cấp code từ số 0 để SV B copy-paste vào Google Colab và hoàn thành công việc tuần 1.

## 2. Architecture & Components
File hướng dẫn sẽ được chia thành các phần sau:

### 2.1. Mục tiêu Tuần 1 & Nguyên tắc
- Liệt kê các tác vụ: Clone repo, Setup môi trường, Load checkpoint, Smoke test.
- Nhấn mạnh nguyên tắc: Ghi lại môi trường, không dùng test data để thử, không đẩy dữ liệu mật lên Colab công khai.

### 2.2. Khởi tạo môi trường Google Colab
- Hướng dẫn step-by-step: Mở Colab -> Tạo sổ tay mới -> Đổi runtime sang GPU.
- Đoạn code Python để mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### 2.3. Clone mã nguồn & Cài đặt thư viện
- Code bash trên Colab (`!git clone ...`, `!pip install ...`).
- Yêu cầu cài đặt các thư viện lõi của PanDerm (PyTorch, torchvision, MMSegmentation).
- Lệnh ghi log môi trường: `!pip freeze > /content/drive/MyDrive/NCKH/requirements_colab.txt`.

### 2.4. Chuẩn bị Dữ liệu mẫu (Pilot) & Checkpoint
- Viết code bash/python tải một số ảnh ISIC mẫu (nếu được phép) hoặc tự tạo dummy images (ảnh nhiễu/ảnh test) để test pipeline nếu chưa có dữ liệu thật.
- Code tải checkpoint PanDerm Base về Google Drive bằng `wget` hoặc `gdown`.

### 2.5. Thực thi Smoke Test (Inference Code)
- Đây là phần lõi. Cần cung cấp mã nguồn Python hoàn chỉnh cho Colab:
  - Khai báo các module.
  - Xây dựng hàm tiền xử lý (Transforms: Resize, Normalize).
  - Khởi tạo mô hình PanDerm, nạp trọng số (weights).
  - Vòng lặp inference cho 20 ảnh. Đo VRAM (`torch.cuda.max_memory_allocated`) và thời gian.
  - Hàm vẽ đồ thị bằng Matplotlib để tạo ra grid gồm: Ảnh gốc - Ảnh kèm Mask.

### 2.6. Checklist bàn giao
- SV B cần gửi cho SV A: File `requirements_colab.txt`, 10 ảnh kết quả dạng PNG/JPG, và log file về tốc độ xử lý/VRAM để chốt phương án Phase 1 (Go/No-Go).

## 3. Ambiguity Check & Resolution
- *Ambiguity:* Dữ liệu mẫu lấy từ đâu?
  *Resolution:* Do dự án liên quan đến ISIC, code sẽ cung cấp một đoạn script nhỏ dùng `urllib` để tải 2-3 ảnh công khai trên mạng (ví dụ ảnh da liễu Creative Commons) làm mồi (toy data) để smoke test không bị vướng mắc vấn đề tải dữ liệu thật.

## 4. Next Steps
- User duyệt file spec này.
- Invocation tool `writing-plans` để lập kế hoạch chi tiết việc tạo file.
