> ⚠️ Tài liệu này đã lỗi thời (smoke test giả bằng torch.rand, dùng ISIC 2017 thay vì ISIC 2018, link checkpoint sai, cài mmcv không ghim phiên bản). Dùng [sv_b/P2_moi_truong_checkpoint_smoke_test.md](sv_b/P2_moi_truong_checkpoint_smoke_test.md).

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

Chạy cell dưới đây để tải mã nguồn chính thức và cài đặt các thư viện lõi (PyTorch, MMSegmentation, torchvision) vào một môi trường ảo riêng biệt nhằm tránh xung đột với Python mặc định của Colab.

```bash
# Di chuyển vào thư mục dự án
%cd /content/drive/MyDrive/NCKH_PanDerm

# Clone repo PanDerm (nếu chưa có)
!if [ ! -d "PanDerm" ]; then git clone https://github.com/SiyuanYan1/PanDerm.git; fi

# Cài uv để tạo môi trường Python 3.10 riêng (không phụ thuộc Python 3.13 mặc định của Colab)
!pip install -q uv
!uv python install 3.10

# Tạo virtual env
VENV = "/content/drive/MyDrive/NCKH_PanDerm/.venv-panderm"
!uv venv --python 3.10 --seed {VENV}

# Kiểm tra phiên bản Python trong venv
!{VENV}/bin/python --version

# Di chuyển vào thư mục segmentation
%cd /content/drive/MyDrive/NCKH_PanDerm/PanDerm/segmentation

# Cập nhật công cụ build
!{VENV}/bin/python -m pip install -U pip wheel "setuptools<81"

# Cài đặt PyTorch và Torchvision phù hợp với PanDerm
!{VENV}/bin/pip install \
  torch==2.2.1 \
  torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/cu118

# Cài đặt openmim và MMSegmentation theo hướng dẫn nhánh Segmentation
!{VENV}/bin/pip install -U openmim
!{VENV}/bin/mim install mmengine
!{VENV}/bin/mim install "mmcv>=2.0.0"
!{VENV}/bin/pip install "mmsegmentation>=1.0.0"

# Ghi lại cấu hình môi trường
!{VENV}/bin/pip freeze > /content/drive/MyDrive/NCKH_PanDerm/requirements_colab.txt
!echo "Đã lưu requirements_colab.txt"
```

*Giải thích chi tiết:*
- `git clone https://github.com/SiyuanYan1/PanDerm.git`: Tải toàn bộ mã nguồn mô hình PanDerm từ nhánh chính của tác giả SiyuanYan1.
- `uv venv --python 3.10 --seed {VENV}`: Sử dụng công cụ `uv` (siêu tốc) để khởi tạo một môi trường ảo (virtual environment) sử dụng đích danh phiên bản Python 3.10. Điều này cực kỳ quan trọng vì PanDerm yêu cầu nghiêm ngặt Python 3.10, trong khi môi trường Google Colab mặc định thường xuyên được nâng cấp (ví dụ Python 3.11/3.13) gây rủi ro xung đột thư viện.
- `%cd .../PanDerm/segmentation`: Phải đứng ở thư mục gốc của phân hệ segmentation trước khi cài đặt các phụ thuộc liên quan.
- `setuptools<81`: Ép hạ cấp setuptools vì các bản mới của setuptools có thể làm gãy quá trình build của một số thư viện cũ (MMSegmentation/PyTorch cũ).
- `torch==2.2.1 torchvision==0.17.1 --index-url .../cu118`: Cài đặt đích danh bộ đôi PyTorch và Torchvision mà mô hình yêu cầu (tương thích CUDA 11.8). Lệnh gọi bắt buộc phải qua `!{VENV}/bin/pip` để cài thẳng vào môi trường ảo, không làm ảnh hưởng Colab.
- `{VENV}/bin/mim ...`: Cài đặt các thành phần của OpenMMLab. `mim` đóng vai trò quản lý gói chuyên biệt cho MMEngine, MMCV, MMSegmentation.
- `{VENV}/bin/pip freeze`: Trích xuất danh sách mọi thư viện có trong môi trường ảo của dự án.

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

**Bước 4.2: Tải Dữ liệu ảnh thực tế (ISIC 2017 Task 1)**
Theo `Readme.md` của dự án, bộ dữ liệu ISIC 2017 Task 1 (gồm 2.000 ảnh) được sử dụng để phân đoạn tổn thương. Đoạn lệnh sau sẽ tải bộ dữ liệu này và giải nén (lưu ý quá trình giải nén trên Drive có thể mất thời gian).

```bash
%cd /content/drive/MyDrive/NCKH_PanDerm
!mkdir -p data/isic2017
%cd data/isic2017

# Tải file ảnh huấn luyện và mask phân đoạn từ nguồn ISIC chính thức
!wget -c https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip
!wget -c https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip

# Giải nén dữ liệu (Dùng cờ -q để ẩn bớt log hiển thị trên Colab)
!unzip -n -q ISIC-2017_Training_Data.zip
!unzip -n -q ISIC-2017_Training_Part1_GroundTruth.zip

!echo "Đã chuẩn bị xong dữ liệu ISIC 2017 tại data/isic2017"
```

*Giải thích chi tiết:*
- `%cd .../data/isic2017`: Di chuyển vào thư mục dành riêng cho dữ liệu ISIC 2017.
- `wget -c ...`: Tiện ích tải file từ internet. Cờ `-c` (continue) cho phép tải tiếp nếu bị gián đoạn, tiết kiệm thời gian với các file dữ liệu lớn như ảnh y khoa.
- `unzip -n -q ...`: Giải nén file `.zip` vừa tải. Cờ `-q` (quiet) giúp giảm thiểu log hiển thị trên Colab (tránh làm tràn màn hình với 2.000 dòng log), và `-n` (never overwrite) giúp không giải nén lại nếu thư mục đã tồn tại.

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

# 4. Vòng lặp Inference trên 20 ảnh đầu tiên
image_paths = sorted(glob.glob('/content/drive/MyDrive/NCKH_PanDerm/data/isic2017/ISIC-2017_Training_Data/*.jpg'))[:20]
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
- `sorted(glob.glob(...))[:20]`: Quét và lấy danh sách đường dẫn file ảnh thực tế từ tập ISIC 2017, sắp xếp theo tên và cắt lấy 20 ảnh đầu tiên (`[:20]`) để chạy thử nghiệm, tránh việc phải chạy qua toàn bộ 2.000 ảnh gây mất thời gian.
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

## 6. Đồng bộ Notebook lên GitHub

Để không bị mất mã nguồn đang viết trên Colab, bạn có thể đồng bộ các file Notebook từ Drive sang Github:

```python
import os
import shutil
import glob

# Đường dẫn đích trong thư mục dự án của bạn
DEST_DIR = "/content/drive/MyDrive/NCKH_PanDerm"

# Tìm kiếm tất cả các file notebook (.ipynb) trong thư mục mặc định "Colab Notebooks"
colab_notebooks_path = "/content/drive/MyDrive/Colab Notebooks/*.ipynb"
notebooks = glob.glob(colab_notebooks_path)

print("--- ĐANG TÌM KIẾM NOTEBOOK ĐỂ SAO CHÉP ---")
if notebooks:
    for nb in notebooks:
        nb_name = os.path.basename(nb)
        destination_path = os.path.join(DEST_DIR, nb_name)
        shutil.copy(nb, destination_path)
        print(f"Đã sao chép thành công notebook: {nb_name} vào thư mục dự án.")
else:
    print("Không tìm thấy file notebook nào trong thư mục mặc định 'Colab Notebooks'.")
    print("Hãy bấm 'File -> Save' (Ctrl+S) trên Colab trước để Drive cập nhật tệp tin mới nhất.")
```

Đẩy code lên nhánh chính của dự án:
```bash
%cd /content/drive/MyDrive/NCKH_PanDerm

# Cập nhật và đẩy file Notebook lên GitHub
!git add *.ipynb 2>/dev/null || git add .
!git commit -m "Add/Update Colab notebooks to repository"
!git push -u origin main
```

*Giải thích chi tiết:*
- Lệnh Python đầu tiên tự động tìm các notebook Colab lưu mặc định tại thư mục `Colab Notebooks` trên Drive và copy chúng về thư mục dự án `NCKH_PanDerm` để quản lý tập trung.
- Lệnh bash tiếp theo dùng `git add` và `git push` đẩy các file `.ipynb` vừa sao chép lên kho lưu trữ GitHub của nhóm để SV A có thể dễ dàng theo dõi.

## 7. Checklist Bàn giao cuối tuần (Go/No-Go Phase 1)

Sau khi chạy xong Notebook trên, SV B cần kiểm tra và gửi cho SV A các tài nguyên sau để chốt kết quả Tuần 1:
- [ ] File `requirements_colab.txt` ghi lại các version của `torch`, `torchvision`, `mmsegmentation`.
- [ ] 10 ảnh mẫu `result_X.png` (trong thư mục results) để SV A xem overlay.
- [ ] Log tốc độ: (VD: 20 ảnh chạy mất 5 giây, VRAM tốn 3000MB) để nhóm quyết định xem Colab T4 có đủ gánh nổi toàn bộ dataset dọc UQ hay không.

Nếu mọi thứ tích xanh, nhóm có thể tự tin chuyển sang Phase 2 & 3 vào tuần tới!
