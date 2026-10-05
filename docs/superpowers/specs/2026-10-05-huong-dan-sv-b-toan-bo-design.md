# Design Spec: Bộ hướng dẫn và code SV B cho toàn bộ các phase (Google Colab)

**Ngày:** 05/10/2026
**Căn cứ:** `Ke_hoach_trien_khai_NCKH_PanDerm_2_nguoi.md`, `De_cuong_PanDerm_du_bao_thay_doi_ton_thuong_da.docx`, PanDerm upstream commit `fd7a807` (17/02/2026).
**Vai:** SV B (Người 2): mô hình, pipeline, demo.

## 1. Mục tiêu

Tạo bộ tài liệu hướng dẫn theo từng phase cho SV B. Mỗi phase có giải thích, code, test, benchmark/đánh giá và checklist bàn giao. Mọi thứ chạy trên Google Colab. Code được đồng bộ giữa Google Drive và GitHub, còn dữ liệu chỉ nằm trên Drive.

**Sản phẩm của việc này CHỈ là các file `.md` trong `docs/sv_b/`.** SV B tự tạo toàn bộ code trong repo (package, script, test, notebook, patch, `.gitignore`) bằng cách làm theo hướng dẫn. Người viết hướng dẫn không tạo hay sửa file code nào trong repo.

### Quyết định đã chốt với người dùng

| Câu hỏi | Quyết định |
|---|---|
| Phạm vi | Chỉ viết file hướng dẫn; SV B tự code |
| Mức code trong docs | Code đầy đủ cho từng file (`.py`, test, cell Colab, patch) kèm giải thích từng đoạn. Code CPU được chạy thử trong thư mục tạm ngoài repo trước khi đưa vào docs |
| Tổ chức code SV B sẽ tạo | Package `src/nckh/` có pytest + script CLI mỏng + 3 notebook Colab mỏng |
| Dataset segmentation | ISIC 2018 Task 1 (không dùng ISIC 2017 Task 1 như notebook cũ) |
| Fine-tune classification | SV B làm, theo kế hoạch. Đề cương mục 7 dòng 4 cần SV A sửa; spec này không sửa file docx |
| Dữ liệu UQ | Chưa có hoặc chưa rõ quyền → schema chuẩn và dữ liệu giả; có cổng Go/No-Go trước khi đưa UQ lên Drive/Colab |
| Docs Tuần 1 cũ | Giữ nguyên file, thêm ghi chú "đã lỗi thời". Bộ mới nằm ở `docs/sv_b/` và P2 thay thế hướng dẫn Tuần 1 |
| Nền tảng | Chỉ Google Colab. Dữ liệu lưu trên Drive. Code clone vào Drive và push/pull với GitHub |

## 2. Hiện trạng và lỗi đã phát hiện

1. Smoke test trong `nckh.ipynb` (cell 6–7) dùng `torch.rand`, không nạp model nào (VRAM 1.34 MB). Link checkpoint `example.com` trả về 404.
2. Segmentation upstream cố định ViT-Large: `segmentation/models/cae_config.py` (`embed_dim=1024, depth=24`) và `cae_seg.py` đọc cứng `model_weights/panderm_ll_data6_checkpoint-499.pth`. Nhánh skip trong vòng nạp trọng số gọi `model_dict[k]` khi `k` không có trong dict, nên gây `KeyError`.
3. Notebook và Readme dùng ISIC 2017 Task 1, trong khi loader upstream cố định layout `ISIC2018/{Training,Validation,Test}_Data` và `*_GroundTruth/{id}_segmentation.png`.
4. `classification/run_class_finetuning.py` tự đánh giá tập test ở epoch cuối và luôn gọi `wandb.init`.
5. Cell 9 của notebook (`git push`) lỗi vì thư mục Drive không phải git repo.
6. `.gitignore` chặn toàn bộ `/docs/`.
7. Đề cương ghi "ISIC 2018 Task 1 mỗi ảnh có 5 mask". Script P3 sẽ đếm số mask thực tế để xác nhận.

## 3. Kiến trúc

### 3.1. Bố cục Google Drive

```
MyDrive/NCKH_PanDerm/
├── nckh/            git clone repo GitHub (code); push/pull từ Colab
├── data/
│   ├── zips/        zip gốc ISIC 2018 Task 1, ISIC 2017 Task 3
│   ├── manifests/   manifest, split, hash
│   └── uq/          chỉ khi đã có văn bản cho phép
├── checkpoints/     PanDerm Base pretrained + checkpoint fine-tune
├── runs/<run_id>/   log, run card, prediction, metrics
└── PanDerm/         clone upstream
```

- `data/`, `checkpoints/` và `runs/` nằm ngoài repo, nên git không thể vô tình commit chúng.
- Code tìm đường dẫn qua biến môi trường `NCKH_ROOT`, mặc định là `/content/drive/MyDrive/NCKH_PanDerm`.
- Ảnh để train được copy dưới dạng zip sang `/content` rồi giải nén ở đó. File nhỏ (manifest, CSV, checkpoint) đọc trực tiếp từ Drive.
- Venv tạo trên `/content` mỗi phiên bằng `uv`. Venv cũ trên Drive được xoá.

### 3.2. Đồng bộ code giữa Drive và GitHub

- Mỗi notebook có cell setup chạy lại được nhiều lần: mount Drive → đọc token `GH_TOKEN` từ Colab Secrets → clone nếu chưa có → `git pull --rebase`.
- Cuối phiên làm việc chạy `git add/commit/push`.
- Quy tắc:
  - luôn pull trước khi sửa, push ngay sau khi sửa;
  - chỉ một runtime commit tại một thời điểm;
  - không bao giờ dán token vào cell.

### 3.3. Ba runtime Colab

| Notebook | Python | Dùng cho | Cài đặt |
|---|---|---|---|
| `notebooks/NB_seg.ipynb` (GPU) | venv 3.10 tạo bằng `uv` | P2, P5a, P6 (mask), P8 | torch 2.2.1/cu118, mmengine 0.10.4, mmcv 2.1.0, mmseg 1.2.2, requirements seg, `openpyxl`, `pip install -e nckh` |
| `notebooks/NB_cls.ipynb` (GPU) | venv 3.10 | P2, P5b, P6 (xác suất), P8 | torch 2.4.1/cu118, requirements cls, `pip install -e nckh` |
| `notebooks/NB_cpu.ipynb` | Python mặc định của Colab | P3, P4, P6 (Ridge), P7, P10, pytest, P9 demo | `pip install -e nckh[demo,test]` |

Package `nckh` chỉ phụ thuộc `numpy, pandas, scikit-learn, scikit-image, pillow, joblib, pyyaml`, không phụ thuộc torch. Phần dùng torch nằm trong `infer.py`, import torch lazily bên trong hàm.

### 3.4. Cấu trúc repo mà hướng dẫn dạy SV B tự tạo

Mỗi file dưới đây xuất hiện trong docs của phase tương ứng: đường dẫn, mã nguồn đầy đủ, giải thích và lệnh kiểm tra. Thứ tự tạo file đi theo thứ tự các phase.

```
pyproject.toml                    package nckh, extras: demo (streamlit, matplotlib), test (pytest)
src/nckh/
  paths.py        đường dẫn từ NCKH_ROOT
  manifest.py     sha256, kiểm tra ảnh, ghép ảnh–mask theo ID, ảnh trùng hash, split theo nhóm có seed, bảng lý do loại
  pairs.py        ghép cặp t→t+1, Δt theo ngày, lý do loại; chọn mẫu audit phân tầng
  features.py     a_t, circularity, eccentricity (regionprops), ghép xác suất, cờ mask rỗng
  forecast.py     Pipeline(StandardScaler, Ridge), chọn alpha trên val, baseline, predict + clip
  metrics.py      Dice/IoU, phân loại, hồi quy, bootstrap theo nhóm và paired bootstrap
  degrade.py      brightness, blur, white balance, hair, Ben Graham, gamma
  infer.py        PanDerm seg/cls → mask + xác suất (torch import lazily)
  runcard.py      run card JSON
scripts/          prepare_isic2018.py, prepare_isic2017_cls.py, build_manifest.py, build_pairs.py,
                  train_ridge.py, evaluate_seg.py, evaluate_cls.py, evaluate_forecast.py,
                  make_degraded.py, bench_inference.py, infer_one.py, annotator_agreement.py,
                  make_tables.py, make_figures.py
patches/panderm_base_seg.patch
demo/app.py
tests/            pytest, dữ liệu giả
notebooks/        NB_seg, NB_cls, NB_cpu
docs/sv_b/        11 file hướng dẫn (mục 4)
```

Hướng dẫn P2 dạy SV B sửa `.gitignore`: đổi `/docs/` thành `/docs/*` cộng `!/docs/sv_b/` và `!/docs/superpowers/`, thêm chặn `*.pth, *.ckpt, *.parquet, data/, runs/`. Người viết docs chỉ commit các file `docs/sv_b/*.md` (dùng `git add -f` vì `.gitignore` hiện tại vẫn chặn `docs/`).

**Phân bổ file theo phase:**
- **P2:** `pyproject.toml`, `paths.py`, `runcard.py`, `bench_inference.py`, patch, `NB_seg`/`NB_cls`/`NB_cpu`
- **P3:** `manifest.py`, `prepare_isic2018.py`, `prepare_isic2017_cls.py`, `build_manifest.py`
- **P4:** `annotator_agreement.py`
- **P5a/P5b:** `evaluate_seg.py`, `evaluate_cls.py`
- **P6:** `pairs.py`, `features.py`, `forecast.py`, `infer.py`, `build_pairs.py`, `train_ridge.py`
- **P7:** `metrics.py`, `evaluate_forecast.py`
- **P8:** `degrade.py`, `make_degraded.py`
- **P9:** `infer_one.py`, `demo/app.py`
- **P10:** `make_tables.py`, `make_figures.py`

Test đi kèm nằm ngay trong phase tạo ra module đó.

## 4. Bộ tài liệu `docs/sv_b/`

Mỗi file phase theo cùng một khung:

1. Mục tiêu, đầu vào/đầu ra (đường dẫn Drive)
2. Runtime và thời gian ước tính
3. Giải thích (công thức, bẫy leakage)
4. Code (cell Colab và trích đoạn code chính)
5. Test (pytest và sanity check trên dữ liệu thật)
6. Benchmark/đánh giá (file sinh ra, bảng mẫu để trống)
7. Lỗi thường gặp trên Colab
8. Checklist bàn giao cho SV A

| File | Nội dung chính |
|---|---|
| `00_tong_quan_va_lo_trinh.md` | Lộ trình 14 tuần của SV B, sơ đồ dòng dữ liệu, bố cục Drive, quy tắc đồng bộ, 3 runtime, nguyên tắc dữ liệu |
| `P0_P1_P4_P11_vai_tro_ho_tro.md` | Việc SV B làm ở các phase SV A chủ trì; P4: chọn mẫu audit và `annotator_agreement.py` |
| `P2_moi_truong_checkpoint_smoke_test.md` | Cell setup, `gdown` checkpoint Base (ID `17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB`) + sha256, kiểm tra key, áp patch, smoke test thật, benchmark, pilot 1 epoch + resume |
| `P3_du_lieu_manifest_split.md` | Tải và đổi layout ISIC 2018, CSV ISIC 2017 Task 3 (mel=0, nev=1, sk=2), manifest, đếm số mask, schema UQ + `uq_column_map.yaml`, split theo participant 70/15/15 |
| `P5a_segmentation_isic2018.md` | `run.py` sau patch, `--smoke_test` (CSVLogger), checkpoint trên Drive, `--resume 0`, chọn theo `Val/Jac`, test một lần `--evaluate --save_results`, quy ước metric 224×224 + giữ thành phần liên thông lớn nhất |
| `P5b_classification_isic2017.md` | Thay dòng test bằng bản sao val khi train, `WANDB_MODE=disabled`, `--update_freq` để giữ batch hiệu dụng 128, chạy `--eval` một lần với CSV thật, tính lại metric từ `test.csv` |
| `P6_ghep_cap_va_ridge.md` | Cổng Go/No-Go UQ, `infer.py`, ghép cặp, features, Ridge + alpha, baseline, file prediction, mức A/B |
| `P7_kiem_thu_metrics_bootstrap.md` | Bảng test, metric từng nhánh, bootstrap theo participant/lesion 2000 lần, paired bootstrap Ridge − baseline |
| `P8_robustness.md` | Các mức suy giảm, sinh ảnh sau khi split, chỉ dùng ảnh test, tham chiếu giữ nguyên, paired bootstrap; Ben Graham và gamma chọn trên val |
| `P9_demo_streamlit.md` | App gọi subprocess sang 2 venv, cache theo hash ảnh, kiểm tra đầu vào, disclaimer, tunnel `cloudflared` chỉ với ảnh ISIC, cách chạy trên laptop |
| `P10_tai_lap_hinh_bang.md` | Run card, `make_tables.py` và `make_figures.py`, kiểm tra tái lập từ môi trường sạch |

Ba file `docs/Huong_dan_SV_B_Tuan_1_*.md` được thêm một dòng ghi chú ở đầu, trỏ sang `docs/sv_b/P2_...`.

## 5. Chi tiết kỹ thuật

### 5.1. Patch PanDerm Base cho segmentation
- `cae_config.py`: `embed_dim=768, depth=12, num_heads=12, out_indices=[3,5,7,11]`, `decode_head.in_channels=[768]*4`, `channels=768`.
- `cae_seg.py`: đường dẫn lấy từ `PANDERM_CKPT`. Vòng nạp không còn `KeyError`. In số key đã khớp so với tổng số key backbone, và `raise RuntimeError` nếu tỷ lệ khớp dưới 90%.
- Prefix key của checkpoint Base (`encoder.`) chưa xác minh được ở đây. P2 có cell kiểm tra key trước khi áp patch. Nếu prefix khác thì sửa một dòng `replace` trong patch.

### 5.2. Classification không mở test sớm
- `prepare_isic2017_cls.py` sinh hai file: `isic2017_cls.csv` (split thật) và `isic2017_cls_trainphase.csv`, trong đó các dòng `test` được thay bằng bản sao các dòng `val` gắn nhãn split `test`.
- Lúc train dùng file trainphase. Sau khi khóa cấu hình, chạy `--eval --resume checkpoint-best.pth` với CSV thật, đúng một lần.

### 5.3. Dự báo
- `a_t = |M_t| / (H·W)`, `Δa = a_{t+1} − a_t`, `â_{t+1} = clip(a_t + Δâ, 0, 1)`.
- `z_t = [a_t, circularity, eccentricity, p_mel, p_nev, p_sk, Δt_days]`.
- Lưới alpha cố định `[0.01, 0.1, 1, 10, 100]`, chọn theo MAE trên val, fit lại trên train với alpha đã chọn.
- **Metric chính:** MAE(Δa) trên participant test. Các metric phụ: RMSE, bias, R², độ đúng hướng; ngưỡng "ổn định" lấy từ val.
- Cặp có mask rỗng ở t hoặc t+1 bị loại khỏi train/eval và được đếm riêng.

### 5.4. Bootstrap
- Lấy mẫu lại theo nhóm (participant cho UQ, image/lesion cho ISIC), 2000 lần, seed cố định. Trả về điểm ước lượng và CI 95% theo phương pháp percentile.
- Paired bootstrap: cùng một tập nhóm được lấy mẫu cho cả hai phương pháp ở mỗi lần lặp.

### 5.5. Suy giảm ảnh
- Brightness α ∈ {0.8, 1.2}; Gaussian σ ∈ {1.0, 1.5}; gain R/B ∈ {±10%, ±20%}; lông che {1%, 3%}, sai số độ che ≤ 0.5 điểm phần trăm, có seed.
- Ben Graham: `clip(4I − 4·G_σ*I + 128)`. Gamma: γ ∈ {0.8, 1.2}.
- Hàm nhận và trả `np.uint8 HxWx3`.

### 5.6. Demo
- `infer_one.py --task seg|cls` chạy trong venv tương ứng, ghi `mask.png` hoặc `probs.json`.
- `app.py` gọi hai subprocess, đọc Ridge từ `joblib`, kiểm tra đầu vào bằng `nckh.forecast.validate_delta_days` (số hữu hạn, > 0).
- Ảnh không đọc được hoặc mask rỗng thì hiện thông báo và không xuất dự báo.

## 6. Xử lý lỗi

- Script dừng với thông báo rõ ràng khi thiếu file, sai schema, cặp không hợp lệ hoặc nạp được quá ít trọng số.
- Không có `except: pass`. Ảnh lỗi được ghi vào bảng lý do loại, không bị xoá âm thầm.

## 7. Kiểm thử

Pytest trên CPU với dữ liệu giả, toàn bộ chạy dưới 1 phút:

- `test_manifest`: hash ổn định; phát hiện ảnh hỏng; ghép ảnh–mask theo ID; split theo nhóm không giao nhau; cùng seed cho kết quả giống hệt; báo trùng hash giữa các split.
- `test_pairs`: cùng lesion và participant; timestamp tăng; Δt > 0; không nối hai lesion; lý do loại được ghi đúng.
- `test_features`: diện tích đúng trên hình vuông/hình tròn; circularity của hình tròn ≈ 1; mask rỗng được gắn cờ; không có cột nào của t+1.
- `test_forecast`: scaler chỉ thấy dữ liệu train; alpha được chọn từ val; baseline bằng 0; kết quả clip nằm trong [0, 1]; `validate_delta_days` từ chối giá trị âm, 0, NaN, chuỗi.
- `test_metrics`: Dice/IoU khớp ví dụ tính tay; MAE/RMSE/bias khớp; bootstrap lấy mẫu theo nhóm; paired bootstrap trả CI đúng chiều.
- `test_degrade`: dải giá trị, shape, dtype; seed tái lập được; độ che lông nằm trong dung sai.
- `test_infer`: chạy với model giả (bỏ qua nếu không có torch).

Đây là các test mà docs hướng dẫn SV B viết và chạy (trong `NB_cpu`: `!pytest -q`).

**Kiểm chứng docs:** trước khi đưa code CPU vào docs, người viết docs dựng lại package trong thư mục scratchpad ngoài repo và chạy toàn bộ pytest. Docs chỉ chứa code đã qua bước này. Phần GPU (patch, fine-tune, benchmark, `infer.py` với PanDerm thật) không chạy được ở máy viết docs; docs đánh dấu "chưa kiểm chứng, xác minh trong pilot P2". Riêng patch được kiểm tra bằng `git apply --check` trên bản clone PanDerm `fd7a807`.

## 7b. Điều chỉnh khi lập plan (05/10/2026)

- **Dung lượng Drive:** zip ảnh ISIC 2018 Training nặng 11,2 GB, Test 2,4 GB; ISIC 2017 Train 6,2 GB, Val 0,9 GB, Test 5,8 GB. Tổng vượt 15 GB miễn phí của Drive. Vì vậy mỗi phiên tải ảnh từ S3 về `/content` (script idempotent). Drive chỉ giữ ground truth, CSV nhãn, manifest và kết quả. Mọi manifest lưu đường dẫn **tương đối** so với một `data_root`, không lưu đường dẫn tuyệt đối `/content/...`.
- **Định dạng bảng:** dùng CSV thay cho parquet, để không phải thêm `pyarrow`.
- **Script suy luận:** `scripts/infer_images.py` (nhận `--images` hoặc `--manifest`) thay cho `infer_one.py`. Demo, P6 và P8 dùng chung script này.
- **`metrics.py` được tạo ở P5a** vì P5a/P5b cần dùng trước. P7 dùng lại module này, bổ sung `evaluate_forecast.py` và ma trận test.
- **Thêm `scripts/make_fake_uq.py`:** sinh ảnh giả và metadata UQ giả, để chạy được toàn bộ P6–P8 trước khi có dữ liệu thật.
- **Transform eval của classification phải khớp upstream:** Resize(256, bicubic) → CenterCrop(224) → mean/std ImageNet. Segmentation: resize 224×224 bicubic → mean/std 0.5.

## 8. Ngoài phạm vi

- Không tạo hay sửa file code, notebook, config nào trong repo; chỉ có `docs/sv_b/*.md` và ghi chú lỗi thời ở 3 file docs Tuần 1.
- Không sửa đề cương docx hay kế hoạch; chỉ ghi chú cho SV A.
- Không viết code cho mô hình khác ngoài PanDerm Base và Ridge.
- Không điền số kết quả.
- Không triển khai web công khai.
