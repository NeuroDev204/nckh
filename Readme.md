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

## Smoke test và đo metric trên Streamlit

### Pilot kỹ thuật P2: backward và checkpoint resume

Pilot P2 dùng ảnh smoke cùng nhãn/mask tổng hợp để kiểm tra gradient, cập nhật backbone/head, lưu checkpoint, nạp lại logits và resume optimizer. Checkpoint pilot chỉ dùng kiểm tra kỹ thuật, không dùng đánh giá chất lượng hoặc thay cho checkpoint fine-tune trên nhãn thật.

Utility resume classification của PanDerm xóa `head`/`fc_norm` trước khi nạp strict, gây lỗi missing keys với checkpoint vừa train. Sau khi clone PanDerm, áp dụng bản sửa tối thiểu trước khi kiểm tra resume (bỏ qua nếu đã áp dụng):

```bash
git -C ~/PanDerm apply --unidiff-zero ~/Documents/nckh/patches/panderm_cls_resume.patch
```

Probe, self-check, log, báo cáo và checkpoint của lần chạy local được lưu trong `~/nckh_root/runs/20261012-p2-training-pilot/`. Đây là probe tạm cho P2; loader dataset chính thức và lệnh fine-tune đầy đủ sẽ được kiểm tra ở P3/P5.

Trong Streamlit, tab **Báo cáo P2** tự chọn summary pilot gần nhất dưới `runs/`, hoặc cho nhập đường dẫn `summary.json`. Dashboard có ma trận PASS/FAIL, hai biểu đồ loss riêng cho từng nhánh, VRAM cao nhất (MiB), biểu đồ cập nhật backbone/head và kết luận kỹ thuật. Mỗi biểu đồ có nút tải PNG. Loss chỉ gồm hai điểm train/resume với nhãn tổng hợp, không phải đường cong huấn luyện đầy đủ hay kết quả hiệu năng.

Tab **Báo cáo P3** đọc manifest từ `NCKH_ROOT/data/manifests`, hiển thị biểu đồ ảnh/mask theo split ISIC 2018 và phân bố lớp ISIC 2017, cùng các kiểm tra checksum, ảnh/mask, nhãn, split và trùng hash. Có thể đổi thư mục manifest trong giao diện và tải CSV/JSON hoặc PNG. Báo cáo này kiểm tra dữ liệu; không đưa ra metric hiệu năng mô hình.

### Giao diện đánh giá

Chạy từ thư mục repository bằng môi trường demo:

```bash
.venv/bin/python -m streamlit run scripts/streamlit_smoke.py
```

Chọn **Phân loại**, **Phân đoạn** hoặc **Cả hai**. Mỗi nhánh chạy tuần tự trong Python riêng (`~/venvs/venv_cls/bin/python`, `~/venvs/venv_seg/bin/python`); không cần cài hai bộ dependency PanDerm vào venv giao diện. Cấu hình đường dẫn PanDerm, checkpoint, data root và manifest trong app. Mặc định smoke test dùng tối đa 20 ảnh validation có seed 2026. Chọn **Đánh giá toàn split** để chạy toàn bộ split; không dùng test để chọn checkpoint/ngưỡng.

Tab **Đánh giá mô hình** có lựa chọn **Xem báo cáo đã lưu** để mở kết quả mà không chạy model lại. Metric và recall/F2 từng lớp được vẽ bằng cột, confusion matrix bằng heatmap, mean/p50/p95 latency bằng cột. Metric null được bỏ khỏi biểu đồ, không thay bằng 0. Bảng số liệu, prediction và log nằm trong phần thu gọn; JSON/CSV và biểu đồ PNG đều tải được.

Manifest phân loại chính thức là `isic2017_cls.csv` do `scripts/prepare_isic2017_cls.py` tạo, có các cột `image,label,split`; `image` tương đối so với `<data-root>/ISIC2017`, nhãn 0/1/2 lần lượt melanoma/nevus/seborrheic keratosis. Không dùng `isic2017_cls_trainphase.csv`: split test của file đó chứa bản sao validation.

Manifest phân đoạn có các cột `image_path,mask_path,split,exclude_reason`, đường dẫn tương đối so với data root, được tạo bằng `scripts/build_manifest.py`. Mask phải grayscale nhị phân 0/1 hoặc 0/255 và có cùng kích thước ảnh. Dòng có `exclude_reason` được loại và đếm trong báo cáo. Manifest thiếu ground truth column chỉ chạy smoke test; đường dẫn ground truth đã khai báo nhưng thiếu/hỏng sẽ bị báo lỗi.

Với ảnh smoke chưa có nhãn, cả hai nhánh nhận manifest chung `image_path,split` (đường dẫn tương đối data root). Ví dụ theo ảnh mà notebook hiện dùng:

```csv
image_path,split
smoke/ISIC2018_Task1-2_Validation_Input/ISIC_0017460.jpg,val
```

Nếu chưa có manifest chính thức nhưng có `~/nckh_root/data/manifests/smoke_images.csv`, app dùng file smoke đó làm mặc định. Chạy worker trực tiếp để tái lập:

```bash
~/venvs/venv_cls/bin/python scripts/evaluate_inference.py \
  --task cls --panderm-dir ~/PanDerm/classification \
  --pretrained ~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth \
  --data-root ~/nckh_data --manifest ~/nckh_root/data/manifests/isic2017_cls.csv \
  --split val --n 20 --seed 2026 --out-dir ~/nckh_root/runs/eval_cls_example

~/venvs/venv_seg/bin/python scripts/evaluate_inference.py \
  --task seg --panderm-dir ~/PanDerm/segmentation \
  --pretrained ~/nckh_root/checkpoints/panderm_bb_data6_checkpoint-499.pth \
  --data-root ~/nckh_data --manifest ~/nckh_root/data/manifests/isic2018_seg.csv \
  --split val --n 20 --seed 2026 --out-dir ~/nckh_root/runs/eval_seg_example
```

Thêm `--finetuned <checkpoint>` để dùng head đã fine-tune. Segmentation vẫn cần checkpoint pretrain để dựng backbone. `--n 0` chạy toàn split; output directory phải chưa tồn tại. CLI ghi traceback vào `worker.log`; giao diện lưu cả stdout/stderr trong log subprocess và có timeout mỗi nhánh (mặc định 600 giây).

Phân loại báo accuracy, balanced accuracy, macro precision/recall/F1/F2, chỉ số từng lớp, AUROC one-vs-rest, Brier score đa lớp và confusion matrix. Dự đoán dùng argmax; F2 có beta=2. Macro precision/recall/F1/F2 tính trên cả ba lớp với zero division bằng 0; balanced accuracy lấy mean recall trên lớp có nhãn thật. AUROC thiếu positive/negative là null kèm lý do, macro AUROC chỉ tính khi cả ba lớp xác định. Brier score là mean tổng bình phương sai số của ba xác suất so với one-hot.

Phân đoạn báo Dice, IoU, precision/recall/F2 và pixel accuracy: tính từng ảnh rồi lấy mean. Cả hai mask rỗng cho Dice/IoU/F2=1; precision khi prediction rỗng bằng 1 nếu reference cũng rỗng, ngược lại bằng 0; recall khi reference rỗng bằng 1. Báo số mask rỗng riêng. Pixel accuracy có thể cao do nhiều pixel nền; đọc cùng Dice/IoU.

Mỗi run lưu `report.json`, `predictions.csv`, overlay mẫu, log và `run_card.json` có hash manifest/checkpoint. App cho tải JSON/CSV. Benchmark báo mean/p50/p95 và peak VRAM; thời gian gồm preprocessing/inference/postprocessing, không gồm đọc ảnh, model load, tính metric hay khởi chạy worker.

Checkpoint chỉ pretrain có head chưa fine-tune: mọi kết quả được gắn cảnh báo **chỉ kiểm tra pipeline**. Không có nhãn/mask thật thì metric là null; smoke test không thay thế đánh giá hiệu năng. Bản này chưa tính CI 95%, robustness hoặc huấn luyện model.
