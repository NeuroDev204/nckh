# P0 · P1 · P4 · P11 — Việc của SV B ở các phase SV A chủ trì

Bốn phase này do SV A dẫn dắt. SV B không chạy huấn luyện ở đây, nhưng có trách nhiệm **phản biện kỹ thuật**, **kiểm tra cấu trúc dữ liệu** và **gán mask độc lập** để kết quả có kiểm tra chéo. Mỗi mục có: SV A làm gì · SV B làm gì · sản phẩm SV B nộp · checklist.

## P0 — Khóa protocol (tuần 1)

**SV A chủ trì:** viết `protocol.md` (câu hỏi, endpoint, split, metrics, baseline, phép suy giảm, tiêu chí dừng). Giảng viên duyệt.

**SV B làm:** đọc bản nháp và phản biện bằng checklist kỹ thuật dưới đây. Mỗi dòng phải có **một giá trị cụ thể** trong protocol, không để "sẽ chọn sau". Lý do: mọi lựa chọn chưa khóa trước khi mở test đều có nguy cơ bị chọn theo kết quả test.

| Hạng mục | Giá trị đề xuất (SV B mang vào buổi chốt) | Vì sao phải chốt trước |
|---|---|---|
| Đơn vị phân tích seg/cls | 1 ảnh ISIC | Quyết định đơn vị bootstrap |
| Đơn vị dự báo | 1 cặp ảnh dermoscopy liên tiếp của cùng `lesion_id` | Không nối lesion khác nhau |
| Đơn vị split/bootstrap UQ | `participant_id` | Ảnh của một người tương quan nhau |
| Target | Δa = a(t+1) − a(t), a = pixel mask / tổng pixel ảnh | Một định nghĩa, không đổi giữa chừng |
| Baseline | Δa dự báo = 0 | Đối chứng bắt buộc |
| Metric chính dự báo | MAE(Δa) trên participant test | Tránh chọn metric đẹp sau khi xem test |
| Lưới alpha Ridge | (0.01, 0.1, 1, 10, 100), chọn theo MAE val, hòa → alpha lớn hơn | Không mở rộng lưới sau khi xem test |
| Seed | split 2026, bootstrap 2026 (2000 lần), train seg/cls 0 | Tái lập |
| `--monitor` classification | `recall` (= balanced accuracy) | Tiêu chí chọn checkpoint |
| TTA classification | Có hoặc không — chốt một | TTA có thể đổi kết quả test |
| Checkpoint seg | Tốt nhất theo `Val/Jac` (mặc định upstream) | Không chọn theo test |
| Ngưỡng "ổn định" (`stable_eps`) | Trung vị \|Δa\| trên val, **hoặc** bất đồng diện tích giữa người gán (P4) — chốt một | Dùng cho độ đúng hướng tăng/giảm |
| Phép suy giảm | Đúng bảng Mục 8.1 kế hoạch | Không thêm/bớt sau khi xem |
| Tiền xử lý tương phản | Chính: RGB + chuẩn hóa PanDerm; phụ: Ben Graham hoặc Gamma chọn trên val | Không chọn trên test |
| Quy tắc mở test | Mỗi nhánh mở test **một lần**; ghi ngày/giờ vào nhật ký quyết định | Chống "chọn lại" |

**Sản phẩm SV B nộp:** bảng trên đã điền giá trị cuối + danh sách góp ý gửi SV A (ghi vào nhật ký quyết định của kế hoạch, Mục 10).

**Checklist**
- [ ] Mọi dòng trong bảng có giá trị cụ thể trong protocol.
- [ ] SV B tự giải thích được một câu: *mô hình dự báo cái gì, nhãn tạo từ đâu, điều gì tuyệt đối không được kết luận*.
- [ ] Protocol ghi commit hash của repo `nckh` tại thời điểm khóa.

## P1 — Go/No-Go quyền dữ liệu UQ (tuần 1–2)

**SV A chủ trì:** đọc điều khoản UQ, hỏi đơn vị lưu trữ/giảng viên, lập data dictionary, biên bản Go/No-Go.

**SV B làm:** khi (và chỉ khi) SV A xác nhận có quyền tải và xử lý trên Colab/Drive (và lưu bản sao trên máy cá nhân nếu điều khoản cho phép), kiểm tra **cấu trúc kỹ thuật** của gói dữ liệu và đề xuất bản đồ cột sang schema chuẩn của pipeline:

`participant_id, lesion_id, image_id, image_path, captured_at, modality`

> Nếu chưa có văn bản cho phép xử lý trên đám mây: **không** tải UQ lên Drive. Dùng dữ liệu giả ở P6 để phát triển trước.

Cell kiểm tra cấu trúc chạy **trên máy** (`NB_cpu`, sau cell setup P2 4.1a), sau khi gói UQ đã nằm ở `data/uq/` trên Drive và được kéo về máy:

```bash
# terminal VS Code (máy cá nhân)
rclone copy gdrive:NCKH_PanDerm/data/uq ~/nckh_drive/data/uq --progress
```

Nếu điều khoản không cho lưu trên máy: chạy cell này trong `NB_seg` (Colab) thay vì `NB_cpu`; nội dung giữ nguyên vì `ROOT` trỏ tới Drive.

```python
# cell: NB_cpu (máy cá nhân)
from pathlib import Path
from collections import Counter
import pandas as pd

UQ = Path(ROOT) / 'data' / 'uq'
files = [p for p in UQ.rglob('*') if p.is_file()]
print('Số file:', len(files))
print('Tổng dung lượng (GB):', round(sum(p.stat().st_size for p in files) / 1e9, 2))
print('Đuôi file:', Counter(p.suffix.lower() for p in files).most_common(10))

# Đổi tên file metadata cho đúng gói thật sau khi giải nén
meta = pd.read_csv(UQ / 'metadata.csv')
print(meta.shape)
print(meta.dtypes)
print('Tỷ lệ thiếu theo cột:'); print(meta.isna().mean().sort_values(ascending=False).head(20))
print('Số giá trị khác nhau:'); print(meta.nunique().sort_values())
meta.head(3).T
```

Đọc kết quả để trả lời (ghi vào biên bản gửi SV A):

1. Cột nào là mã người tham gia, mã tổn thương, mã ảnh, đường dẫn ảnh, thời điểm chụp, loại ảnh?
2. Thời điểm chụp là ngày đầy đủ, chỉ tháng/năm, hay chỉ "visit order"? Có múi giờ không?
3. Loại ảnh có những giá trị nào (dermoscopy / clinical / tile…)? Giá trị nào là dermoscopy?
4. Có lesion nào gắn với hơn một participant không? (`meta.groupby(<cột lesion>)[<cột participant>].nunique().gt(1).sum()`)
5. Bao nhiêu lesion có ≥ 2 ảnh dermoscopy có thời điểm? (đây là trần trên của số cặp, chưa phải số cặp thật)

**Sản phẩm SV B nộp:** bảng "cột gốc → cột chuẩn" (sẽ điền vào `configs/uq_column_map.yaml` ở P3), câu trả lời 5 câu hỏi trên.

**Checklist**
- [ ] Không tải UQ khi chưa có văn bản cho phép.
- [ ] Không in/chụp màn hình thông tin định danh.
- [ ] Không suy ra số cặp bằng phép chia "35.909 ảnh / 7.038 lesion".

## P4 — Audit mask UQ (tuần 3–7)

**SV A chủ trì:** viết hướng dẫn gán mask một trang (ranh giới, vùng mờ, lông, bọt khí, vật đánh dấu); gán mask cho tập audit (tối đa ~100 ảnh UQ ≈ 50 cặp); dẫn dắt buổi thống nhất.

**SV B làm:**
1. **Chọn mẫu audit** có phân tầng theo Δt và diện tích mask sơ bộ, có seed, để audit không chỉ gồm ảnh dễ. Hàm `stratified_audit_sample` (viết ở P6, `src/nckh/pairs.py`) chia cặp thành các ô theo tứ phân vị của từng cột rồi lấy đều mỗi ô:

```python
# cell: NB_cpu (máy cá nhân)
import pandas as pd
from nckh.pairs import stratified_audit_sample
feat = pd.read_csv(f'{ROOT}/runs/<run_id_P6>/ridge/features.csv')    # bảng cặp + area_ratio_t sơ bộ từ P6 (target test đã bị che)
pool = feat[(feat.split == 'test') & feat.usable]                    # audit lấy từ participant test (kế hoạch P4 bước 3)
audit = stratified_audit_sample(pool, n_pairs=50, strata_cols=['delta_days', 'area_ratio_t'], seed=2026)
audit.to_csv(f'{ROOT}/data/uq/audit/audit_pairs.csv', index=False)
print(audit['stratum'].value_counts())
```

   Lưu `audit_pairs.csv` và seed. **Không** dùng các ảnh này để fine-tune hay chọn checkpoint. File được ghi ở máy (`~/nckh_drive/data/uq/audit/`); đẩy lên Drive cho SV A bằng `rclone copy ~/nckh_drive/data/uq/audit gdrive:NCKH_PanDerm/data/uq/audit`.

2. **Gán độc lập** một phần ngẫu nhiên đã định trước của tập audit (ví dụ 20 ảnh, chọn bằng `audit.sample(n=20, random_state=2026)`), **không xem** mask của SV A.
   - Công cụ gợi ý: CVAT hoặc labelme. Xuất **mask nhị phân PNG** (0 = nền, 255 = tổn thương), cùng kích thước ảnh gốc, tên `<image_id>.png`.
   - Mỗi người một thư mục trên Drive: `data/uq/audit/annotator_A/`, `data/uq/audit/annotator_B/` (chỉ khi đã có quyền UQ).
   - Trước đó, hai người cùng gán thử 10 ảnh, so sánh, rồi cập nhật hướng dẫn (kế hoạch P4 bước 4).

3. **Tính độ đồng thuận** bằng `scripts/annotator_agreement.py` (mã nguồn và test ở `P5b_classification_isic2017.md` mục 4.2):

```python
# cell: NB_cpu (máy cá nhân)
from nckh.runcard import new_run_id
RUN = f"{ROOT}/runs/{new_run_id('audit_agreement')}"
!cd {REPO} && {PY} scripts/annotator_agreement.py --dir-a {ROOT}/data/uq/audit/annotator_A --dir-b {ROOT}/data/uq/audit/annotator_B --out {RUN}/agreement.csv
```

   Trước khi chạy, kéo mask của hai người gán về máy: `rclone copy gdrive:NCKH_PanDerm/data/uq/audit ~/nckh_drive/data/uq/audit` (terminal VS Code).
   Script chỉ so các ảnh có ở **cả hai** thư mục. Nếu lệch tên file, nó dừng và liệt kê; khi đó hãy copy riêng phần ảnh chung vào hai thư mục con trước khi chạy.

4. **Không xoá** mask độc lập sau buổi thống nhất. Mask thống nhất lưu ở `data/uq/audit/consensus/`; hai thư mục độc lập là bằng chứng inter-annotator agreement.

**Sản phẩm SV B nộp:** `audit_pairs.csv` (+ seed), thư mục `annotator_B/`, `agreement.csv` (Dice, IoU, chênh lệch tỷ lệ diện tích từng ảnh) và một dòng tóm tắt (trung vị + min Dice).

**Checklist**
- [ ] Mẫu audit chọn từ participant test hoặc cohort tách riêng, có seed.
- [ ] Gán độc lập, không xem mask người kia.
- [ ] `agreement.csv` và các mask độc lập được lưu, không bị ghi đè.
- [ ] Nếu bất đồng lớn (Dice thấp ở nhiều ảnh), báo giảng viên; không gọi mask là "ground truth lâm sàng".

## P11 — Viết bài: phụ lục kỹ thuật (tuần 12–14)

**SV A chủ trì:** điều phối bản thảo (Methods, Results, Discussion).

**SV B làm:** viết **phụ lục kỹ thuật** và đoạn mô tả triển khai trong Methods, lấy số liệu **từ run card và file kết quả**, không gõ tay. Dàn ý phụ lục:

1. **Môi trường:** máy cá nhân (VS Code, .venv Python 3.10, CPU) cho xử lý dữ liệu/đánh giá; Colab nối qua extension VS Code (loại GPU thật từ run card) cho train/suy luận; ba môi trường (seg/cls trên Colab, cpu trên máy) và lý do tách.
2. **Phiên bản:** bảng phiên bản gói lấy từ `run_card.json` → `packages` của từng run chính; commit PanDerm upstream `fd7a807` + patch đã áp.
3. **Checkpoint và hash:** checkpoint pretrained (nguồn, ngày tải, SHA-256), checkpoint fine-tune seg/cls (run_id, SHA-256), `ridge.joblib` (SHA-256 của `pairs.csv` lưu bên trong).
4. **Run card:** danh sách run_id của mọi kết quả trong bài, mỗi run gắn với bảng/hình nào.
5. **Quy trình tái lập:** các lệnh theo thứ tự P3 → P5a → P5b → P6 → P7 → P8 → P10, tham chiếu notebook.
6. **Giới hạn kỹ thuật:** metric seg tính ở 224×224 và giữ thành phần liên thông lớn nhất (theo upstream); tiền xử lý PIL vs OpenCV; chế độ suy luận không có TTA/có TTA; độ không xác định của GPU (nondeterminism).

**Checklist**
- [ ] Mọi số trong phụ lục truy ngược được tới một `run_id`.
- [ ] Không có đường dẫn cá nhân, token, hay tên file ảnh UQ cụ thể.
