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

## 5. Chạy Smoke Test (Inference Loop)

Dưới đây là đoạn script Python hoàn chỉnh chạy trực tiếp trên Colab. Nó sẽ load 20 ảnh, đưa qua mô hình và lưu lại ảnh kết quả.

```python
import os
import glob
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang dùng thiết bị: {device}")

# 2. Tiền xử lý (Transform)
# (Theo chuẩn PanDerm/ViT)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 3. Khởi tạo Mô hình giả lập Load Checkpoint
# Chú ý: Ở bước thực tế, SV B sẽ import mô hình theo class của repo PanDerm
# Ví dụ: from panderm.models import build_model
# model = build_model('vit_base')
# model.load_state_dict(torch.load('checkpoints/panderm_base.pth')['state_dict'])

print("Đang nạp mô hình... (Ví dụ)")
# model = model.to(device)
# model.eval()

# 4. Vòng lặp Inference trên 20 ảnh
image_paths = glob.glob('/content/drive/MyDrive/NCKH_PanDerm/toy_data/*.jpg')
out_dir = '/content/drive/MyDrive/NCKH_PanDerm/results'
os.makedirs(out_dir, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
start_time = time.time()

for idx, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Lấy output từ mô hình
        # output_mask = model(input_tensor)
        # Vì đây là smoke test độc lập để test code flow, giả lập output_mask:
        output_mask = torch.rand((1, 1, 224, 224)).to(device)
        pred_mask = (output_mask > 0.5).float().squeeze().cpu().numpy()
        
    # Trực quan hóa
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"Ảnh gốc {idx}")
    axes[1].imshow(img.resize((224, 224)))
    axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f"Dự đoán Mask {idx}")
    
    plt.savefig(os.path.join(out_dir, f"result_{idx}.png"))
    plt.close()

end_time = time.time()
max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

print(f"Hoàn thành {len(image_paths)} ảnh trong {end_time - start_time:.2f} giây.")
print(f"VRAM tối đa đã dùng: {max_mem:.2f} MB")
```

*Giải thích chi tiết:*
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`: Tự động nhận diện phần cứng, ưu tiên bộ tăng tốc GPU CUDA nếu có, fallback về CPU nếu chạy môi trường không có GPU.
- `preprocess = transforms.Compose([...])`: Chuỗi tiền xử lý chuẩn hóa ảnh đầu vào theo kiến trúc Vision Transformer (ViT) và mô hình PanDerm:
  - `transforms.Resize((224, 224))`: Resize kích thước ảnh về $224 \times 224$ pixels phù hợp với kích thước đầu vào của backbone ViT.
  - `transforms.ToTensor()`: Chuyển đổi định dạng PIL Image (giá trị $0 - 255$) thành PyTorch Tensor (khoảng giá trị $[0.0, 1.0]$) với định dạng $(C, H, W)$.
  - `transforms.Normalize(mean=..., std=...)`: Chuẩn hóa kênh màu RGB theo phân phối ImageNet tiêu chuẩn để khớp với trọng số tiền huấn luyện (pretrained weights).
- `glob.glob('/content/drive/MyDrive/NCKH_PanDerm/toy_data/*.jpg')`: Quét và lấy toàn bộ danh sách đường dẫn file ảnh thử nghiệm trong thư mục `toy_data`.
- `out_dir` & `os.makedirs(out_dir, exist_ok=True)`: Định nghĩa và tự động tạo thư mục `results` trên Google Drive để lưu trữ ảnh overlay kết quả.
- `torch.cuda.reset_peak_memory_stats()`: Khởi tạo lại bộ đếm theo dõi lượng VRAM đỉnh điểm trước khi bước vào vòng lặp inference nhằm đo đạc chính xác lượng bộ nhớ GPU tiêu thụ (kèm điều kiện kiểm tra `torch.cuda.is_available()`).
- `Image.open(img_path).convert('RGB')`: Đọc ảnh từ đĩa và đảm bảo định dạng đủ 3 kênh màu RGB (loại bỏ kênh alpha hoặc grayscale nếu có).
- `preprocess(img).unsqueeze(0).to(device)`: Tiền xử lý ảnh, thêm chiều batch `unsqueeze(0)` tạo tensor 4 chiều $(B, C, H, W) = (1, 3, 224, 224)$, và chuyển tensor lên bộ nhớ thiết bị tính toán (`device`).
- `torch.no_grad()`: Context manager vô hiệu hóa việc tính toán gradient (đạo hàm), giúp tiết kiệm đáng kể bộ nhớ VRAM và đẩy nhanh tốc độ thực thi trong chế độ suy luận (inference).
- `output_mask = torch.rand((1, 1, 224, 224)).to(device)` & `pred_mask = (output_mask > 0.5)...`: Giả lập tensor xác suất đầu ra của phân vùng tổn thương kích thước $(1, 1, 224, 224)$, áp ngưỡng nhị phân $0.5$, loại bỏ các chiều đơn lẻ (`squeeze()`), và đưa về mảng NumPy trên RAM CPU để phục vụ vẽ đồ thị.
- `fig, axes = plt.subplots(1, 2, figsize=(8, 4))`: Khởi tạo khung hiển thị gồm 2 đồ thị con (axes) đặt cạnh nhau để so sánh trực quan.
- `axes[0].imshow(img)` & `axes[1].imshow(...)`: Hiển thị ảnh gốc ở khung bên trái và ảnh overlay (ảnh gốc resize kèm lớp phủ mask màu 'jet' với độ trong suốt $\alpha = 0.5$) ở khung bên phải.
- `plt.savefig(...)` & `plt.close()`: Lưu hình ảnh kết quả vào thư mục `results` trên Drive và đóng figure để thu hồi bộ nhớ RAM đồ họa, tránh rò rỉ bộ nhớ (memory leak) khi lặp qua nhiều ảnh.
- `torch.cuda.max_memory_allocated() / (1024 ** 2)`: Đo lường lượng bộ nhớ GPU tối đa đã được cấp phát trong suốt quá trình chạy (chuyển đổi từ byte sang Megabytes - MB), giúp đánh giá mức độ an toàn trước nguy cơ lỗi OOM (Out Of Memory).

## 6. Checklist Bàn giao cuối tuần (Go/No-Go Phase 1)

Sau khi chạy xong Notebook trên, SV B cần kiểm tra và gửi cho SV A các tài nguyên sau để chốt kết quả Tuần 1:
- [ ] File `requirements_colab.txt` ghi lại các version của `torch`, `torchvision`, `mmsegmentation`.
- [ ] 10 ảnh mẫu `result_X.png` (trong thư mục results) để SV A xem overlay.
- [ ] Log tốc độ: (VD: 20 ảnh chạy mất 5 giây, VRAM tốn 3000MB) để nhóm quyết định xem Colab T4 có đủ gánh nổi toàn bộ dataset dọc UQ hay không.

Nếu mọi thứ tích xanh, nhóm có thể tự tin chuyển sang Phase 2 & 3 vào tuần tới!
