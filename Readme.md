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

## Báo cáo tiến độ và trạng thái các giai đoạn nghiên cứu

Căn cứ theo [Kế hoạch triển khai NCKH PanDerm](Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md) và [Báo cáo tiến độ giai đoạn P2](Bao_cao_tien_do_NCKH_PanDerm_Giai_doan_P2.docx), hiện trạng triển khai các nhiệm vụ như sau:

### Bảng tổng hợp tiến độ

| Giai đoạn | Nhiệm vụ | Trạng thái | Minh chứng thực tế |
| :--- | :--- | :---: | :--- |
| **Phase 0** | Khóa Protocol, mục tiêu và định nghĩa endpoint nghiên cứu | **Đạt (100%)** | Quy chuẩn đề cương, protocol và kế hoạch phân công 2 người. |
| **Phase 2** | Hạ tầng môi trường, Checkpoint PanDerm Base, Vá mã nguồn & Smoke test | **Đạt (100%)** | Checkpoint ViT-B 328 MB (SHA-256 `be1e0fb1...`); 2 venv GPU (`venv_seg`, `venv_cls`); bản vá `patches/panderm_base_seg.patch` (độ phủ ViT 100%); smoke test 20 ảnh (RTX 3050 & Tesla T4); pilot backward pass & resume optimizer đạt (`status: PASS`). |
| **Phase 3** | Tải dữ liệu ISIC, Kiểm tra nhãn/mask & Lập manifest chia tập (Split) | **Đạt (100%)** | 3.694 ảnh & mask ISIC 2018 (1 mask/ảnh, 0 lỗi, SHA-256 manifest `8cf5e733...`); 2.750 ảnh ISIC 2017 (3 lớp); tách `isic2017_cls_trainphase.csv` chống leakage; tab Báo cáo P3 trên Streamlit; 97/97 unit tests passed. Toàn bộ dữ liệu lưu trữ trên Google Drive. |
| **Phase 5a** | Tinh chỉnh nhánh phân đoạn (ISIC 2018 Task 1) | **Chưa chạy (0%)** | **Đã sẵn sàng pipeline & dữ liệu**: Đã vá code, kiểm tra loader, chuẩn bị `run.sh`; chờ kích hoạt huấn luyện 100 epoch trên GPU. |
| **Phase 5b** | Tinh chỉnh nhánh phân loại tham khảo (ISIC 2017 Task 3) | **Chưa chạy (0%)** | Đã sẵn sàng manifest và script; chờ huấn luyện trên GPU. |
| **Phase 6** | Ghép cặp ảnh theo dõi và huấn luyện mô hình hồi quy Ridge | **Chưa chạy (0%)** | Đã xây dựng sẵn module suy luận `nckh.infer` và cấu trúc trích xuất đặc trưng; chờ dữ liệu dọc UQ. |
| **Phase 7** | Đánh giá kiểm định mô hình, Bootstrap CI 95% & Robustness | **Chưa chạy (0%)** | Đã xây dựng module `nckh.metrics`; chờ kết quả checkpoint sau fine-tuning. |

### Chi tiết các phần đã hoàn thành

#### 1. Hạ tầng kỹ thuật & Pilot PanDerm Base (P2)
* **Xử lý xung đột phụ thuộc**: Tách lập 2 môi trường GPU độc lập gồm `venv_seg` (PyTorch 2.1.2 + MMCV 2.1.0 + MMSegmentation 1.2.2) và `venv_cls` (PyTorch 2.4.1 + timm 0.9.16).
* **Vá mã nguồn PanDerm**: Áp dụng `patches/panderm_base_seg.patch` để tương thích ViT-B/16 (768 dim), hỗ trợ checkpoint resume (`model_checkpoint_0.ckpt`) và ngắt chế độ tự động mở tập test sau train.
* **Đo kiểm Smoke test (20 ảnh ISIC 2018 Validation)**:
  * Phân đoạn: 177.5 ms/ảnh, VRAM đỉnh 812 MB.
  * Phân loại: 85.6 ms/ảnh, VRAM đỉnh 351 MB.
  * Sinh 10 ảnh mask overlay trực quan và xuất thẻ truy vết `run_card.json`.
* **Pilot Backward pass & Resume Optimizer**: Đã kiểm chứng trên GPU Tesla T4 (loss giảm 1.0985 → 1.0981, khôi phục nguyên vẹn 100% logits và bước optimizer).

#### 2. Dữ liệu, Kiểm tra nhãn/mask & Manifest split (P3)
* **Phân đoạn ISIC 2018 Task 1 (`isic2018_seg.csv`)**:
  * Tải và giải nén đủ 3.694 ảnh và 3.694 mask chuẩn hóa theo cấu trúc của PanDerm (Train: 2.594, Val: 100, Test: 1.000).
  * Kiểm tra toàn vẹn: 100% ảnh đọc được, 100% khớp kích thước ảnh-mask, 0 ảnh bị loại (`exclude_reason: ""`).
  * Xác thực mỗi ảnh có đúng 1 mask tham chiếu (`max_masks_per_image: 1`).
  * 0 ảnh trùng lặp hash xuyên split (`seg_duplicates: 0`). Khóa SHA-256: `8cf5e733a18dfbbe8798f03b4f84354594ec0c10e7f8b5072db2dd65733d09fb`.
* **Phân loại ISIC 2017 Task 3 (`isic2017_cls.csv`)**:
  * Tải đủ 2.750 ảnh (Train: 2.000, Val: 150, Test: 600) với 3 nhóm nhãn chuẩn.
  * Sinh thêm file `isic2017_cls_trainphase.csv` (dùng bản sao val thay cho test) nhằm bảo vệ tập test thật khi fine-tuning.
* **Lưu trữ & Báo cáo trực quan**:
  * Dữ liệu và manifest được lưu trữ an toàn trên Google Drive (`/MyDrive/NCKH_PanDerm/`).
  * Tích hợp tab **Báo cáo P3** trên dashboard Streamlit (`scripts/streamlit_smoke.py`).


## Hướng dẫn cài đặt môi trường và chạy kiểm thử (Unit Test)

Môi trường kiểm thử có thể chạy trực tiếp trên máy CPU thông thường (không bắt buộc GPU).

### 1. Đồng bộ mã nguồn
```bash
git clone --branch main https://github.com/NeuroDev204/nckh.git ~/Documents/nckh
cd ~/Documents/nckh
# Nếu đã clone repository:
git switch main
git pull --ff-only origin main
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
97 passed
```

Bộ kiểm thử bao gồm:
- `tests/test_paths_runcard.py` (6 passed): Đường dẫn hệ thống, mã băm SHA-256, sinh `run_id` và thẻ truy vết `run_card.json`.
- `tests/test_inspect_checkpoint.py` (4 passed): Nhận diện kiến trúc ViT-B (768) / ViT-L (1024), xử lý checkpoint bọc ngoài.
- `tests/test_infer.py` (8 passed): Tiền xử lý chuẩn PanDerm, hậu xử lý vùng liên thông lớn nhất của mask, ánh xạ trọng số pretrain sang classification.
- `tests/test_bench.py` (5 passed): Đo độ trễ suy luận, kiểm tra tính hữu hạn và tổng xác suất bằng 1.0.
- `tests/test_prepare.py` (9 passed): Tải dữ liệu, trích xuất zip và kiểm tra tính toàn vẹn ISIC 2017/2018.
- `tests/test_manifest.py` (8 passed): Kiểm tra ảnh/mask, chia nhóm không rò rỉ và phát hiện trùng hash xuyên split.
- `tests/test_pilot_classification.py` (2 passed): Kiểm tra backward pass, cập nhật trọng số và resume checkpoint.
- `tests/test_metrics.py` (17 passed): Kiểm thử các chỉ số phân loại, phân đoạn và xử lý các trường hợp biên.
- `tests/test_evaluate.py` (14 passed): Đánh giá suy luận theo split, xuất báo cáo và thẻ truy vết.
- `tests/test_report_charts.py` (18 passed): Báo cáo biểu đồ pilot P2, bảng chất lượng manifest P3 và dashboard.
- `tests/test_streamlit_smoke.py` (6 passed): Smoke test các thành phần và tương tác giao diện Streamlit.


## Smoke test và đo metric trên Streamlit

### Chạy Colab trong VS Code

Cài extension **Google Colab** (`google.colab`) và **Jupyter** (`ms-toolsai.jupyter`).
Mở `notebooks/nb_colab.ipynb`, chọn **Select Kernel → Colab → Auto Connect**, rồi đăng nhập Google nếu được yêu cầu.
Chạy lần lượt các cell để kiểm tra runtime, clone source vào `/content/nckh`, cài dependency và chạy test.
Cell Drive chỉ kiểm tra đường dẫn dữ liệu/checkpoint; notebook chưa huấn luyện mô hình.
Repo local và repo trong runtime Colab là hai bản riêng; `/content` là vùng lưu trữ tạm.
Các notebook `nb_cpu`, `nb_cls`, `nb_seg` hiện dùng đường dẫn local và kernel tương ứng trên laptop.

### Pilot kỹ thuật P2: backward và checkpoint resume

Pilot P2 dùng ảnh smoke cùng nhãn/mask tổng hợp để kiểm tra gradient, cập nhật backbone/head, lưu checkpoint, nạp lại logits và resume optimizer. Checkpoint pilot chỉ dùng kiểm tra kỹ thuật, không dùng đánh giá chất lượng hoặc thay cho checkpoint fine-tune trên nhãn thật.

Utility resume classification của PanDerm xóa `head`/`fc_norm` trước khi nạp strict, gây lỗi missing keys với checkpoint vừa train. Sau khi clone PanDerm, áp dụng bản sửa tối thiểu trước khi kiểm tra resume (bỏ qua nếu đã áp dụng):

```bash
git -C ~/PanDerm apply --unidiff-zero ~/Documents/nckh/patches/panderm_cls_resume.patch
```

Probe, self-check, log, báo cáo và checkpoint của lần chạy local được lưu trong `~/nckh_root/runs/20261012-p2-training-pilot/`. Đây là probe tạm cho P2; loader dataset chính thức và lệnh fine-tune đầy đủ sẽ được kiểm tra ở P3/P5.

Trong Streamlit, tab **Báo cáo P2** tự chọn summary pilot gần nhất dưới `runs/`, hoặc cho nhập đường dẫn `summary.json`. Dashboard có ma trận PASS/FAIL, hai biểu đồ loss riêng cho từng nhánh, VRAM cao nhất (MiB), biểu đồ cập nhật backbone/head và kết luận kỹ thuật. Mỗi biểu đồ có nút tải PNG. Loss chỉ gồm hai điểm train/resume với nhãn tổng hợp, không phải đường cong huấn luyện đầy đủ hay kết quả hiệu năng.

Tab **Báo cáo P3** đọc manifest từ `NCKH_ROOT/data/manifests`, hiển thị biểu đồ ảnh/mask theo split ISIC 2018 và phân bố lớp ISIC 2017, cùng các kiểm tra checksum, ảnh/mask, nhãn, split và trùng hash. Có thể đổi thư mục manifest trong giao diện và tải CSV/JSON hoặc PNG. Báo cáo này kiểm tra dữ liệu; không đưa ra metric hiệu năng mô hình.

### Chạy Streamlit sau khi clone

Tạo môi trường và cài các dependency giao diện (nếu chưa cài ở bước trên):

```bash
cd ~/Documents/nckh
test -x .venv/bin/python || uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[demo]"
```

Khởi chạy ứng dụng:

```bash
.venv/bin/python -m streamlit run scripts/streamlit_smoke.py
```

Để tab **Báo cáo P3** hiện số liệu, chép các artifact nhẹ từ máy tạo manifest vào `~/nckh_root/data/manifests/` (hoặc chọn thư mục khác trong tab): `isic2018_seg.csv`, `isic2018_seg.csv.sha256`, `isic2018_seg.cross_split_duplicates.csv`, `isic2017_cls.csv`, `isic2017_cls_trainphase.csv` và `mask_count.json`. Có thể đổi thư mục gốc bằng biến môi trường `NCKH_ROOT`. Các artifact này không nằm trong Git; riêng biểu đồ P3 không cần chép ảnh ISIC. Tab **Báo cáo P2** cần file pilot `summary.json` trong `~/nckh_root/runs/`. Muốn chạy đánh giá mô hình thì cần chép thêm ảnh dữ liệu và checkpoint, đồng thời cài mã PanDerm cùng các môi trường Python tương ứng.

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
