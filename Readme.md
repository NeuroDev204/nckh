# Bộ dữ liệu nghiên cứu ảnh tổn thương da

Tài liệu này tổng hợp các bộ dữ liệu có thể sử dụng cho nghiên cứu ảnh da, gồm dữ liệu theo dõi tổn thương qua thời gian và dữ liệu phân đoạn tổn thương.

## 1. UQ Longitudinal Skin Image Dataset

Bộ dữ liệu này phù hợp với nghiên cứu sự thay đổi hoặc tiến triển của tổn thương da theo thời gian. Dữ liệu gồm **35.909 ảnh dermoscopy**, theo dõi **7.038 tổn thương**, kèm metadata.

- [Trang dữ liệu UQ](https://espace.library.uq.edu.au/view/UQ:a13deaf)
- [Trang mô tả dữ liệu](https://researchdata.edu.au/a-longitudinal-dataset-skin-cancers/3750284)

**Lưu ý về quyền sử dụng:** Trang dữ liệu ghi quyền truy cập là “Open”, trong khi mục bản quyền ghi “Other”. Cần đọc và xác nhận điều khoản sử dụng trước khi tải, xử lý hoặc đưa dữ liệu vào nghiên cứu và công bố kết quả.

## 2. ISIC 2017 – Task 1: Lesion Segmentation

Bộ dữ liệu này phù hợp để huấn luyện hoặc thử nghiệm mô hình phân đoạn tổn thương, chẳng hạn U-Net. Có thể dùng để khảo sát độ bền của mô hình trước các biến đổi ảnh như mờ, thiếu sáng, lệch màu hoặc bị lông che.

Tập huấn luyện Task 1 có **2.000 ảnh** và mask phân đoạn tương ứng.

- [Trang ISIC 2017 – Task 1](https://challenge.isic-archive.com/landing/2017/42/)
- [Ảnh huấn luyện](https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip)
- [Mask phân đoạn huấn luyện](https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip)

**Giới hạn khi sử dụng:** ISIC 2017 không theo dõi cùng một tổn thương qua nhiều lần khám. Vì vậy, bộ dữ liệu phù hợp cho bài toán phân đoạn và đánh giá độ bền trên ảnh, nhưng không đủ để kiểm chứng khả năng dự báo tiến triển tổn thương theo thời gian.

## Gợi ý sử dụng trong nghiên cứu

- Dùng **UQ Longitudinal Skin Image Dataset** làm nguồn dữ liệu chính nếu mục tiêu là nghiên cứu sự thay đổi hoặc dự báo tiến triển tổn thương theo thời gian.
- Dùng **ISIC 2017 Task 1** cho bài toán phân đoạn hoặc thử nghiệm bổ trợ về chất lượng ảnh; không xem đây là dữ liệu kiểm chứng dự báo theo thời gian.
- Trang mô tả UQ hiện không liệt kê sẵn mask phân đoạn. Hãy kiểm tra gói dữ liệu tải xuống để xác nhận có nhãn hay không. Nếu không có, nhóm cần tự gán nhãn một tập ảnh phù hợp để đánh giá U-Net.

## Mã nguồn mô hình tham khảo

- **PanDerm**: [Repository trên GitHub](https://github.com/SiyuanYan1/PanDerm) — nguồn mã để tham khảo và triển khai mô hình PanDerm trong nghiên cứu.

## Hướng dẫn cài đặt môi trường và chạy kiểm thử (Unit Test)

Môi trường kiểm thử có thể chạy trực tiếp trên máy CPU thông thường (không bắt buộc GPU).

### 1. Đồng bộ mã nguồn
```bash
git clone https://github.com/NeuroDev204/nckh.git ~/Documents/nckh
cd ~/Documents/nckh
# hoặc nếu đã clone: git pull origin main
```

### 2. Thiết lập môi trường ảo (.venv với Python 3.10)
```bash
# Cài đặt uv (nếu máy chưa có)
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env)

# Tạo và kích hoạt môi trường ảo
test -x .venv/bin/python || uv venv .venv --python 3.10
source .venv/bin/activate

# Cài đặt package nckh cùng công cụ kiểm thử
uv pip install -e ".[demo,test]" ipykernel

# Cài đặt PyTorch CPU (để chạy đầy đủ các test suy luận)
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Thực thi kiểm thử
```bash
python -m pytest -q
```

### 4. Kết quả kỳ vọng
```text
21 passed in ~2.0s
```

Bộ kiểm thử bao gồm:
- `tests/test_paths_runcard.py` (6 passed): Đường dẫn hệ thống, mã băm SHA-256, sinh `run_id` và thẻ truy vết `run_card.json`.
- `tests/test_inspect_checkpoint.py` (4 passed): Nhận diện kiến trúc ViT-B (768) / ViT-L (1024), xử lý checkpoint bọc ngoài.
- `tests/test_infer.py` (8 passed): Tiền xử lý chuẩn PanDerm, hậu xử lý vùng liên thông lớn nhất của mask, ánh xạ trọng số pretrain sang classification.
- `tests/test_bench.py` (3 passed): Đo độ trễ suy luận, kiểm tra tính hữu hạn và tổng xác suất bằng 1.0.