# Kế hoạch triển khai nghiên cứu khoa học end-to-end

## Đề tài

**Ứng dụng PanDerm để phân đoạn tổn thương, đưa ra nhóm bệnh tham khảo và dự báo thay đổi tỷ lệ diện tích mask ở lần tái khám kế tiếp từ ảnh dermoscopy**

**Nhóm thực hiện:** 2 sinh viên  
**Thời lượng dự kiến:** 14 tuần  
**Phiên bản kế hoạch:** 28/09/2026  
**Căn cứ:** đề cương PanDerm đã thống nhất.

---

## 1. Mục tiêu và giới hạn cần giữ xuyên suốt

### 1.1. Câu hỏi nghiên cứu

1. PanDerm Base ViT-B/16 có thể phân đoạn tổn thương trên bộ ISIC 2018 Task 1 và chuyển sang ảnh UQ ở mức nào?
2. Từ ảnh hiện tại, mask dự đoán, đặc trưng hình dạng, điểm nhóm bệnh tham khảo và khoảng thời gian Δt, hồi quy Ridge có dự báo được tỷ lệ diện tích mask ở lần chụp kế tiếp tốt hơn baseline “không thay đổi” không?
3. Các thay đổi độ sáng, độ mờ, cân bằng trắng và lông che khuất ảnh hưởng ra sao đến phân đoạn, nhãn tham khảo và dự báo?

### 1.2. Đầu ra dự báo được định nghĩa thế nào

Đầu ra của nghiên cứu là **thay đổi tỷ lệ diện tích mask trên ảnh** giữa hai lần chụp cùng tổn thương. Đây là một chỉ dấu hình ảnh; không được diễn giải thành kích thước vật lý của tổn thương, tiến triển ác tính, chẩn đoán, tiên lượng điều trị hoặc khuyến nghị y khoa.

Với ảnh tại lần khám t và ảnh kế tiếp t+1:

- a_t = số pixel thuộc mask / tổng số pixel ảnh.
- Δa_t = a_(t+1) − a_t.
- Mô hình dự báo Δa_t từ dữ liệu có tại thời điểm t và khoảng thời gian Δt.
- Tỷ lệ diện tích dự báo ở lần kế tiếp = clip(a_t + Δa_dự_báo, 0, 1).
- Baseline dự báo không thay đổi: Δa_dự_báo = 0.

**Ranh giới khoa học quan trọng:** nếu không có mask chuyên gia cho ảnh UQ, tỷ lệ diện tích tính từ mask PanDerm là một phép đo do mô hình tạo ra. Khi ấy kết quả chỉ là “dự báo tỷ lệ mask PanDerm sẽ tạo ra ở ảnh sau”, chưa phải dự báo thay đổi sinh học thật. Kế hoạch yêu cầu audit mask thủ công trên một tập UQ được giữ riêng; nếu không đủ nguồn lực để gán mask, bài báo phải giữ cách diễn giải thận trọng này.

### 1.3. Phạm vi mô hình

- Mô hình ảnh chính: PanDerm Base ViT-B/16.
- Hai nhánh được tinh chỉnh riêng theo mã nguồn tương ứng: phân đoạn và phân loại tham khảo.
- Mô hình dự báo: Ridge Regression một bước.
- Đối chứng dự báo: baseline không thay đổi diện tích mask.
- Không làm bảng xếp hạng nhiều kiến trúc. Không thêm GRU, NAS, diffusion hoặc nhiều nhánh mô hình.
- DermFM-Zero không nằm trong thực nghiệm chính của kế hoạch này; đưa vào tổng quan tài liệu nếu phù hợp, không tích hợp thêm vào pipeline.

### 1.4. Nguồn dữ liệu dự kiến

| Dữ liệu | Dùng cho | Điều cần xác minh |
|---|---|---|
| ISIC 2018 Task 1 | Tinh chỉnh và đánh giá phân đoạn | Tải đủ ảnh và mask tham chiếu; giữ thông tin nhiều người gán mask; đọc giấy phép/cách trích dẫn |
| ISIC 2017 Task 3 | Tinh chỉnh và đánh giá nhãn nhóm bệnh tham khảo | Đối chiếu đúng mã ảnh, nhãn, split và giấy phép |
| UQ Longitudinal Skin Image Dataset | Ghép ảnh cùng tổn thương theo thời gian và đánh giá dự báo | Quyền truy cập/sử dụng, participant_id, lesion_id, thời điểm chụp, loại ảnh, khả năng công bố chỉ số/tệp dẫn xuất và số cặp ảnh hợp lệ thực tế |

> Bài báo mô tả UQ báo cáo **35.909 ảnh dermoscopy dọc thuộc 7.038 tổn thương**. Nhóm MYM và nhánh can thiệp HOPS được chụp theo dõi định kỳ 6 tháng/lần trong lần lượt 3 năm và 2 năm; nhánh đối chứng HOPS chỉ có một lần chụp bổ sung ở mốc 24 tháng. Đây là tổng quan của dataset, không bảo đảm mọi tổn thương có đủ các lần chụp hoặc có một cặp “khám kế tiếp” dùng được. Nhóm phải đếm cặp liên tiếp thật từ metadata và ảnh được cấp quyền trước khi chốt phạm vi dự báo. Nguồn: [bài báo mô tả UQ](https://doi.org/10.1038/s41597-025-05880-2).

Số lượng công bố trong bài báo của bộ dữ liệu không đồng nghĩa nhóm chắc chắn được tải hoặc được quyền sử dụng toàn bộ dữ liệu. Mọi con số cuối cùng phải lấy từ manifest thực tế sau lọc.

### 1.5. Dùng công cụ và nền tảng ở đâu

| Công việc | Nơi chạy đề xuất | Công cụ/mã nguồn | Ghi chú triển khai |
|---|---|---|---|
| Quản lý mã, protocol, manifest và phân tích metadata | Máy cá nhân hoặc máy trường, CPU | Git; Python; pandas; Jupyter | Chỉ đẩy mã và tài liệu được phép lên GitHub; không đẩy ảnh, metadata định danh hoặc checkpoint bị hạn chế |
| Fine-tune phân đoạn | Máy GPU local, venv riêng cho nhánh | `PanDerm/segmentation/run.sh`; ISIC 2018 Task 1 | Dùng PanDerm pretrained; train/validation để chọn checkpoint, test khóa riêng |
| Fine-tune phân loại tham khảo | Máy GPU local, venv riêng cho nhánh | `PanDerm/classification/script/finetune_train.sh`; ISIC 2017 Task 3 | Tách venv khỏi phân đoạn vì hai nhánh trong repo yêu cầu bộ phiên bản PyTorch/torchvision khác nhau |
| Suy luận PanDerm trên UQ và xuất mask/đặc trưng | Máy trường/máy cá nhân được phép | Checkpoint đã khóa, inference script, Python | Protocol chính dùng UQ cho đánh giá dọc, không fine-tune PanDerm trên UQ; không đưa UQ vào máy trước khi xác minh quyền |
| Ghép cặp ảnh và huấn luyện Ridge | Máy cá nhân/máy trường, CPU | pandas, NumPy, scikit-learn | Ridge không cần GPU; xử lý dữ liệu UQ tại nơi được phép |
| Tạo ảnh suy giảm và đánh giá độ bền | CPU để biến đổi ảnh; GPU cho suy luận PanDerm nếu có | Pillow hoặc OpenCV, script đã khóa | Tạo biến thể sau khi chia tập; giữ mask/target tham chiếu cố định |
| Thống kê, biểu đồ và demo | Máy cá nhân, CPU | pandas, scikit-learn, Matplotlib; Streamlit chạy cục bộ | Demo dùng ảnh được phép; không đưa ảnh UQ hạn chế lên web công khai |

**Cách dùng máy GPU:** fine-tune trên ISIC và các lượt suy luận ảnh cần GPU chạy trên máy GPU local; mỗi nhánh PanDerm một venv riêng theo hướng dẫn của repo; chạy trong `tmux`, lưu cấu hình, log và checkpoint định kỳ vào `runs/` để khôi phục khi bị dừng. Nếu máy GPU không đủ tài nguyên, tiếp tục xử lý dữ liệu/Ridge trên CPU và chuyển huấn luyện sang GPU của trường. Không đổi mô hình hay dùng tập test để bù cho giới hạn phần cứng.

Trong protocol chính, PanDerm được tinh chỉnh trên ISIC; UQ được giữ cho kiểm tra chuyển miền, tạo cặp thời gian và đánh giá dự báo. Không dùng participant test UQ để tinh chỉnh mô hình, chọn checkpoint hoặc điều chỉnh ngưỡng.

**Quyền dữ liệu:** kiểm tra điều khoản ISIC trước khi dùng. Với UQ, chỉ xử lý trên máy được đơn vị dữ liệu cho phép; không tải lên dịch vụ đám mây. Kiểm tra riêng điều khoản PanDerm/checkpoint trước khi chia sẻ checkpoint tinh chỉnh hoặc công khai demo.

---

## 2. Phân công cho hai người

Trong kế hoạch, gọi hai thành viên là **SV A** và **SV B**. Thay tên sau khi nhóm thống nhất.

| Mảng | SV A – dữ liệu, đo lường, bài báo | SV B – mô hình, pipeline, demo | Cùng chịu trách nhiệm |
|---|---|---|---|
| Nền tảng nghiên cứu | Câu hỏi, tiêu chí chọn dữ liệu, giấy phép, tài liệu nền | Kiểm tra repo, checkpoint, môi trường, giới hạn phần cứng | Khóa protocol trước khi xem test |
| Dữ liệu | Tạo data dictionary, kiểm tra metadata, tạo manifest và báo cáo số lượng | Viết loader, kiểm tra ảnh-mask, tạo pipeline tiền xử lý | Kiểm tra trùng lặp, split và rò rỉ |
| Phân đoạn | Viết quy trình gán mask UQ; gán mask audit và kiểm tra chéo | Chạy PanDerm segmentation, lưu checkpoint/log, xuất mask | Xem lỗi phân đoạn trên tập validation và audit |
| Phân loại | Đối chiếu nhãn, tính phân bố lớp, chuẩn bị bảng kết quả | Fine-tune nhánh classification và xuất điểm lớp | Diễn giải nhãn là tham khảo, không phải chẩn đoán |
| Dự báo | Rà soát định nghĩa cặp thời gian, kiểm tra bảng phân tích | Huấn luyện Ridge, chọn alpha, xuất dự báo | Xác nhận không dùng thông tin tương lai làm đầu vào |
| Đánh giá | Viết mã bootstrap/thống kê, tổng hợp bảng và hình | Tạo test tự động, chạy đánh giá và phân tích lỗi | Đọc chéo kết quả từ log và mã nguồn |
| Bài báo/demo | Dẫn dắt Methods, Results, Discussion và tài liệu tham khảo | Dẫn dắt hệ thống demo và mô tả triển khai | Cùng viết Introduction, Abstract, kiểm tra CLAIM, bảo vệ kết quả |

**Nguyên tắc kiểm tra chéo**

- Không để một người vừa chọn mô hình bằng tập test vừa tự báo cáo kết quả test đó.
- SV A giữ bảng split và nhãn test; SV B chỉ chạy lệnh đánh giá cuối sau khi cấu hình được khóa.
- Mỗi bảng/biểu đồ trong bài báo phải truy ngược được tới một script, một file log và một phiên bản dữ liệu.
- Người tạo kết quả không phải người duy nhất kiểm tra số liệu đưa vào bài.

---

## 3. Lịch tổng thể 14 tuần

Các phase có thể chạy song song. Nếu vướng quyền dữ liệu hoặc checkpoint, ưu tiên xử lý Go/No-Go trước khi đầu tư vào huấn luyện dài.

| Tuần | Phase chính | SV A | SV B | Nơi chạy chính | Mốc cần có |
|---|---|---|---|---|---|
| 1 | Khóa câu hỏi và protocol | Rà soát dữ liệu, giấy phép, tài liệu | Clone repo, kiểm tra yêu cầu môi trường | Máy cá nhân, CPU | Protocol v1, checklist rủi ro |
| 1–2 | Go/No-Go UQ và pilot mô hình | Kiểm tra quyền, metadata, ID, thời gian | Thử nạp checkpoint và suy luận 20–50 ảnh | Metadata UQ tại nơi được phép; pilot GPU trên máy GPU local với ISIC | Quyết định có đủ điều kiện làm nhánh dự báo |
| 2–3 | Thu thập, kiểm kê dữ liệu | Data dictionary, số lượng, missingness | Loader và smoke test | Máy cá nhân/máy trường, CPU | Manifest nguồn và báo cáo dữ liệu |
| 3–4 | Làm sạch, ghép cặp, chia tập | Kiểm tra cặp thời gian và split | Viết unit tests cho loader/pair builder | Máy cá nhân/máy trường, CPU | Split khóa, không rò rỉ |
| 4–6 | Tinh chỉnh PanDerm | Kiểm tra nhãn, mask tham chiếu, bảng log | Tinh chỉnh segmentation và classification | Máy GPU local; venv riêng cho mỗi nhánh | Checkpoint chọn từ validation |
| 5–7 | Chạy mask UQ và audit thủ công | Gán mask tập audit, đánh giá chéo | Chạy suy luận UQ và xuất features | Máy/GPU được phép | Báo cáo Dice/IoU và area error UQ |
| 7–9 | Dự báo một bước | Kiểm tra đơn vị phân tích, khoảng Δt | Huấn luyện Ridge, chọn alpha, so baseline | Máy cá nhân/máy trường, CPU | Kết quả trên participant test |
| 9–10 | Robustness và tiền xử lý | Kiểm tra tính hợp lệ biến đổi ảnh | Chạy phép suy giảm/tiền xử lý đã khóa | CPU tạo biến thể; GPU chạy inference | Bảng độ bền, biểu đồ lỗi |
| 10–11 | Demo và kiểm thử hệ thống | Rà soát thông điệp giới hạn sử dụng | Tích hợp giao diện và pipeline | Streamlit chạy cục bộ | Demo chạy được bằng dữ liệu cho phép |
| 11–12 | Phân tích và hình/bảng | Viết kết quả, tạo sơ đồ dòng dữ liệu | Xuất kết quả tái lập, ví dụ lỗi | Máy cá nhân, CPU | Bộ bảng/hình phiên bản cuối |
| 12–14 | Bài báo, poster, rà soát | Viết và chỉnh sửa bản thảo | Tài liệu cài đặt, demo, phụ lục kỹ thuật | Máy cá nhân; GitHub chỉ chứa nội dung được phép | Bài nghiên cứu hoàn chỉnh và gói tái lập |

---

## 4. Các phase triển khai chi tiết

## Phase 0 — Khóa protocol trước khi chạy mô hình

**Thời gian:** tuần 1  
**Chủ trì:** SV A; SV B phản biện; giảng viên hướng dẫn duyệt.

### Việc cần làm

1. Chốt tên đề tài và dùng nhất quán cụm “dự báo thay đổi tỷ lệ diện tích mask trên ảnh”.
2. Rà soát tài liệu gốc về PanDerm, ISIC, UQ dataset, theo dõi dọc và đánh giá AI ảnh y khoa. Tạo bảng trích xuất gồm câu hỏi, dữ liệu, mô hình, endpoint, split, metrics, giới hạn và phần có thể tái lập. Ưu tiên bài báo gốc, trang dataset và tài liệu chính thức.
3. Viết lại câu hỏi nghiên cứu, mục tiêu chính, mục tiêu phụ và giới hạn sử dụng. Phần “khoảng trống” phải dựa trên bảng tài liệu, không chỉ dựa vào việc mô hình còn mới.
4. Xác định đơn vị phân tích:
   - segmentation/classification: một ảnh hoặc một lesion_id tùy dữ liệu;
   - dự báo: một cặp ảnh liên tiếp của cùng lesion_id;
   - đơn vị bootstrap/chia tập cho UQ: participant_id.
5. Chốt mô hình cố định: PanDerm Base, hai nhánh phân đoạn/phân loại; Ridge một bước.
6. Chốt baseline dự báo: giữ nguyên diện tích mask, Δa = 0.
7. Chốt các chỉ số ở Mục 7 và phép suy giảm ảnh ở Mục 8 trước khi mở tập test.
8. Tạo protocol.md hoặc một tài liệu tương đương, ghi:
   - ngày chốt;
   - thành viên duyệt;
   - phiên bản mã nguồn;
   - split seed;
   - các phép thử chính/phụ;
   - tiêu chí dừng và phương án dự phòng.

### Sản phẩm

- Protocol v1 đã được nhóm và giảng viên xem.
- Nhật ký quyết định: mỗi thay đổi phương pháp sau khi khóa protocol phải có ngày, lý do và ảnh hưởng tới kết quả.

### Điều kiện hoàn thành

- Cả hai thành viên có thể giải thích cùng một câu: mô hình dự báo cái gì, dữ liệu nào tạo nhãn, và điều gì tuyệt đối không được kết luận.

---

## Phase 1 — Go/No-Go quyền dữ liệu và tính khả thi UQ

**Thời gian:** tuần 1–2  
**Chủ trì:** SV A; SV B hỗ trợ kiểm tra cấu trúc và dung lượng.

### Kiểm tra quyền sử dụng

1. Đọc trang UQ, bài mô tả dataset, giấy phép/điều kiện đi kèm từng tệp.
2. Ghi rõ nhóm được phép:
   - tải và lưu dữ liệu ở đâu;
   - huấn luyện/tinh chỉnh mô hình;
   - lưu và xử lý trên máy của thành viên nhóm;
   - tạo và lưu mask/embedding;
   - công bố số liệu, hình minh họa hoặc mã nguồn;
   - chia sẻ checkpoint đã tinh chỉnh.
3. Nếu mục “access” và mục “license” không nhất quán hoặc điều khoản không nói rõ nghiên cứu học thuật, gửi câu hỏi cho đơn vị lưu trữ/giảng viên. Không tự suy ra từ chữ “open”.
4. Hỏi giảng viên/đơn vị dữ liệu xem nghiên cứu thứ cấp trên ảnh đã khử định danh có cần phê duyệt đạo đức hoặc văn bản miễn trừ của trường hay không; lưu câu trả lời.
5. Không đẩy ảnh, metadata cá nhân, token truy cập hoặc checkpoint bị hạn chế lên GitHub công khai.
6. Lưu bản ghi ngày truy cập, URL, điều khoản và người xác minh.

### Kiểm kê UQ

Tạo bảng data dictionary với tên cột gốc, ý nghĩa, kiểu dữ liệu, tỷ lệ thiếu và cách dùng. Tối thiểu xác nhận:

- mã ảnh;
- đường dẫn ảnh;
- participant_id hoặc mã người tham gia tương đương;
- lesion_id hoặc mã tổn thương;
- ngày/giờ chụp hoặc visit order;
- loại ảnh (dermoscopy/clinical/tile);
- nhãn/histology/outcome nếu có và nguồn xác nhận;
- trường nào bị ẩn, không được dùng hoặc không được xuất.

Tạo thống kê ban đầu:

- số ảnh đọc được;
- số participant, lesion và visit;
- số lesion có ít nhất hai lần chụp;
- số cặp liên tiếp hợp lệ, định nghĩa “lần kế tiếp” là ảnh dermoscopy kế tiếp có thứ tự thời gian rõ ràng của cùng lesion, không mặc định mọi khoảng cách đều đúng 6 tháng;
- phân phối khoảng Δt theo ngày/tháng;
- số cặp thiếu ID hoặc timestamp;
- số ảnh trùng chính xác/trùng gần;
- số cặp bị loại và lý do.

> Không suy ra số cặp bằng cách lấy 35.909 ảnh chia cho 7.038 tổn thương hoặc giả định mỗi tổn thương có đủ lịch chụp. Số cặp phân tích phải được tính sau khi lọc theo ID, thời điểm, loại ảnh, chất lượng ảnh và điều kiện quyền sử dụng; báo riêng số tổn thương có ảnh lặp và tổng số cặp.

### Go/No-Go cho nhánh dự báo

**GO** khi đồng thời có:

1. Quyền sử dụng cho nghiên cứu được xác nhận.
2. Có thể nối các ảnh thuộc cùng lesion_id.
3. Có thứ tự thời gian đáng tin cậy để lập cặp t → t+1.
4. Có participant_id để chia tập chống rò rỉ; nếu không có thì phải đổi tuyên bố và giới hạn đánh giá.
5. Có đủ cặp sau lọc để tạo train/validation/test theo participant.

Nhóm sẽ kiểm kê toàn bộ cặp hợp lệ trước khi quyết định cỡ test. Đây là nghiên cứu hồi cứu/khả thi, không tự tuyên bố đủ lực thống kê cho kết luận lâm sàng. Nếu cỡ test nhỏ, báo cỡ mẫu hiệu dụng và khoảng tin cậy, đồng thời gọi kết quả là thăm dò.

**NO-GO hoặc đổi phạm vi** nếu quyền chưa rõ, ID/timestamp không đủ, hoặc dữ liệu không thể tách nhóm an toàn. Khi đó:

- Không báo cáo dự báo UQ như kết quả đã xác nhận.
- Trình giảng viên hai lựa chọn: tìm dataset dọc khác có điều khoản rõ; hoặc sửa đề tài thành đánh giá độ bền phân đoạn/phân loại trên ISIC và bỏ mục dự báo dọc.
- Ghi mọi thay đổi trong protocol và đề cương trước khi tiếp tục. Không âm thầm thay bộ dữ liệu hay tự tạo nhãn “tiến triển bệnh”.

### Sản phẩm

- Data dictionary.
- Bảng thống kê trước/sau lọc.
- Biên bản Go/No-Go.

---

## Phase 2 — Cài môi trường và chạy thử repo PanDerm

**Thời gian:** tuần 1–3  
**Chủ trì:** SV B; SV A xác minh license và lưu lại nguồn.

### Các bước

1. Tạo bản clone repo PanDerm và ghi commit hash, ngày clone, URL.
2. Đọc hướng dẫn riêng của classification và segmentation; chúng có thể cần môi trường/phụ thuộc khác nhau.
3. Cài môi trường theo README; lưu file khóa môi trường hoặc danh sách phiên bản:
   - Python;
   - PyTorch/CUDA;
   - thư viện segmentation;
   - phiên bản driver/GPU.
4. Tải checkpoint PanDerm Base từ liên kết chính thức sau khi đọc điều kiện sử dụng.
5. Kiểm tra kích thước input, normalization, số lớp, thứ tự mapping lớp và cách chuyển logits thành mask.
6. Chạy smoke test trên 20–50 ảnh, chưa huấn luyện:
   - checkpoint load thành công;
   - đầu ra phân đoạn đúng kích thước;
   - xác suất phân loại hữu hạn, tổng xác suất hợp lý;
   - không lỗi khi ảnh khác tỷ lệ;
   - ghi thời gian suy luận và VRAM.
7. Lưu 10 ảnh minh họa gồm ảnh gốc, mask overlay và xác suất lớp để kiểm tra mắt thường.
8. Chạy tối đa 1 epoch hoặc một pilot nhỏ để xác minh backward pass/checkpoint save/load. Pilot này không dùng để báo cáo hiệu năng.

### Kiểm tra tương thích

- Nếu checkpoint Base không nạp được vào segmentation config, ghi rõ lỗi và kiểm tra các phiên bản, đường dẫn, kiến trúc.
- Repo PanDerm hướng dẫn segmentation và classification với bộ phụ thuộc PyTorch/torchvision khác nhau. Dựng hai runtime riêng thay vì cài hai bộ vào cùng một môi trường; ghi lại phiên bản thực tế đã chạy trong mỗi notebook.
- Hướng dẫn repo hiện ghi segmentation dùng Python 3.10, PyTorch 2.2.1/torchvision 0.17.1 và MMSegmentation; classification dùng PyTorch 2.4.1/torchvision 0.19.1. Xác minh lại lệnh cài và tương thích với driver máy GPU tại thời điểm chạy; không trộn tùy tiện các phiên bản.
- Repo nêu mã nguồn/mô hình theo giấy phép CC BY-NC-ND 4.0 và mục đích nghiên cứu học thuật phi thương mại. Xác minh điều kiện áp dụng cụ thể trước khi sửa đổi, chia sẻ checkpoint tinh chỉnh hoặc công khai demo.
- Cho phép một vòng thử có giới hạn. Không thay sang model khác theo cảm tính.
- Nếu không giải quyết được trong tuần 2, trao đổi giảng viên để chọn hướng giảm phạm vi trong PanDerm theo repo hoặc dùng GPU của trường; không tự chuyển sang mô hình khác.
- Trên máy GPU, chạy pilot ngắn để kiểm tra VRAM và thời gian trước khi huấn luyện chính. Lưu checkpoint/log vào `runs/` và thử resume sau khi dừng train giữa chừng.
- Nếu máy GPU chưa sẵn sàng, vẫn làm các bước CPU như ghép cặp, Ridge, thống kê và viết bài.

### Sản phẩm và điều kiện đạt

- environment file hoặc hướng dẫn dựng môi trường sạch;
- script inference tối thiểu;
- log smoke test;
- checkpoint có nguồn và điều kiện sử dụng được ghi;
- thành viên khác có thể làm theo README nội bộ và chạy lại một mẫu.

---

## Phase 3 — Tải, kiểm tra và khóa dữ liệu

**Thời gian:** tuần 2–4  
**Chủ trì:** SV A quản lý manifest; SV B viết loader/test.

### 3.1. Tạo cấu trúc dự án

- protocol/
- manifests/
- configs/
- src/
- logs/
- checkpoints/
- predictions/
- figures/
- manuscript/

Ảnh UQ và dữ liệu bị hạn chế lưu ở nơi được phép; không đưa vào repository công khai. Manifest công khai chỉ giữ trường được phép chia sẻ, không ghi thông tin có thể tái định danh.

### 3.2. Kiểm tra dữ liệu

- Đối chiếu danh sách tệp với metadata.
- Tính hash cho ảnh nguồn; phát hiện ảnh hỏng, rỗng, lặp chính xác.
- Kiểm tra mode màu, kích thước, tỷ lệ ảnh, giá trị pixel, orientation.
- Đối chiếu mỗi ảnh với mask/nhãn bằng khóa ID chứ không ghép theo thứ tự tên tệp.
- Tạo báo cáo lỗi; không lặng lẽ xóa ảnh bất thường.
- Lưu số lượng bị loại và lý do trong flow table.

### 3.3. Chia tập

Ưu tiên split chính thức nếu bộ dữ liệu có split được tài liệu hóa và đủ nhãn đánh giá. Nếu tự chia:

- ISIC: chia theo lesion_id nếu có; nếu không, gom ảnh trùng bằng hash trước khi chia.
- UQ: chia theo participant_id, ví dụ 70/15/15 cho train/validation/test, dùng seed khóa trong protocol.
- Tất cả lần khám, lesion và ảnh của một participant chỉ xuất hiện trong đúng một split.
- Tạo cặp thời gian **sau khi** gán participant vào split.
- Không tạo ảnh tăng cường trước khi chia.
- Khóa manifest và lưu hash của manifest. Không đổi test sau khi đã xem kết quả.

### 3.4. Unit tests dữ liệu bắt buộc

1. Mỗi ảnh trong manifest mở được.
2. Mỗi mask ISIC khớp ID và kích thước sau resize.
3. Không có participant_id giao giữa các split UQ.
4. Không có lesion_id giao giữa các split ISIC nếu dùng cùng một lesion.
5. Mỗi cặp UQ có cùng lesion_id, participant_id và timestamp tăng.
6. Δt dương, hữu hạn, đúng đơn vị.
7. Mọi cột đầu vào khi dự báo đều sẵn có tại thời điểm t; không có nhãn, mask hoặc metadata của t+1 trong features.
8. Tạo lại manifest với cùng seed cho kết quả giống hệt.

**Điều kiện đạt:** các test chống leakage đều bằng 0 lỗi; mọi ngoại lệ được liệt kê và duyệt trước khi huấn luyện.

---

## Phase 4 — Quy trình gán mask UQ và kiểm soát chất lượng nhãn

**Thời gian:** tuần 3–7, song song với fine-tuning  
**Chủ trì:** SV A; SV B kiểm tra chéo.

### Quy trình

1. Viết hướng dẫn một trang: ranh giới tổn thương, cách xử lý vùng mờ, lông che, bọt khí, vật đánh dấu và vùng ngoài ảnh.
2. Chọn trước tối đa khoảng 100 ảnh UQ, ưu tiên tạo thành các cặp cùng lesion và có đủ ảnh ở cả hai lần khám. Mục tiêu thực tế là khoảng 50 cặp/100 ảnh nếu số liệu và thời gian cho phép.
3. Chọn từ participant test hoặc chọn một audit cohort tách biệt; không dùng các ảnh này để tinh chỉnh segmentation checkpoint.
4. Hai thành viên cùng gán thử 10 ảnh, rà soát điểm bất đồng, cập nhật hướng dẫn.
5. SV A gán mask; SV B gán độc lập một phần ngẫu nhiên đã định trước. Lưu mask gốc của từng người trước khi adjudicate.
6. Tính Dice giữa người gán, chênh lệch tỷ lệ diện tích mask và số pixel khác biệt.
7. Thảo luận và lưu mask thống nhất; không xóa các phiên bản độc lập vì cần báo inter-annotator agreement.
8. Nếu hai người bất đồng nhiều, trình giảng viên hoặc chuyên gia được phép hỗ trợ; nếu không có người phân xử, báo bất đồng như giới hạn, không gọi mask là ground truth lâm sàng.

Chọn ngẫu nhiên có phân tầng theo khoảng Δt và diện tích mask sơ bộ để audit không chỉ gồm ảnh dễ. Lưu seed và danh sách chọn. Nếu chỉ có thể gán ít hơn 50 cặp, ghi rõ đây là audit pilot; không tự khẳng định mẫu đại diện hoặc đủ cho kiểm định lâm sàng.

### Sản phẩm

- Hướng dẫn gán mask có phiên bản.
- Mask audit và bảng người gán/phiên bản.
- Inter-annotator Dice/IoU và chênh lệch area.

**Lưu ý:** mẫu 100 ảnh chỉ đủ cho một audit khả thi ở cấp sinh viên; cần báo số lượng thật, độ bất định và không suy rộng thành thẩm định lâm sàng.

---

## Phase 5 — Tinh chỉnh và đánh giá hai nhánh PanDerm

**Thời gian:** tuần 4–6  
**Chủ trì:** SV B; SV A kiểm tra nhãn, split và bảng thống kê.

**Nơi chạy:** máy GPU local cho fine-tune và suy luận ISIC; hai venv riêng cho segmentation và classification theo môi trường PanDerm tương ứng. Nếu máy GPU không đủ hoặc không tương thích phụ thuộc, dùng GPU của trường có môi trường đã khóa. Lưu checkpoint tốt nhất theo validation, log và cấu hình ở nơi lưu trữ được phép.

### 5.1. Phân đoạn — ISIC 2018 Task 1

1. Đọc hướng dẫn PanDerm segmentation và tải ảnh/mask từ nguồn ISIC chính thức.
2. Trong venv segmentation riêng trên máy GPU, cài phụ thuộc theo `PanDerm/Segmentation.md`, sửa đường dẫn dữ liệu/checkpoint trong `segmentation/run.sh`, rồi chạy lệnh từ thư mục `segmentation/`. Đây là nơi fine-tune nhánh phân đoạn trên ISIC 2018 Task 1.
3. Kiểm tra nhiều mask tham chiếu cho từng ảnh. Chọn cách đánh giá trước:
   - cách chính: so prediction với mask đồng thuận đa số; hoặc
   - báo Dice/IoU trung bình trên các người gán.
   Cách nào chọn phải được giữ nhất quán và mô tả trong Methods.
4. Chia train/validation/test theo lesion/hash trước tăng cường.
5. Chỉ áp dụng augmentation trong train; không tạo ảnh biến đổi để tăng số test.
6. Fine-tune PanDerm Base segmentation. Lưu:
   - cấu hình;
   - seed;
   - log từng epoch;
   - checkpoint tốt nhất theo metric validation đã chốt;
   - số epoch, learning rate, batch size, thời gian/GPU.
7. Chạy 3 seed nếu tài nguyên cho phép để đánh giá độ dao động huấn luyện. Nếu không, dùng một seed đã chốt và nêu đây là giới hạn.
8. Đánh giá đúng một lần trên test ISIC sau khi chọn checkpoint và ngưỡng mask từ validation.

### 5.2. Phân loại tham khảo — ISIC 2017 Task 3

1. Kiểm tra mapping nhãn melanoma, nevus, seborrheic keratosis và số ảnh từng lớp.
2. Giữ split chính thức nếu có đủ nhãn; nếu tự chia, nhóm theo lesion_id/hash.
3. Trong venv classification riêng trên máy GPU, cài phụ thuộc theo README PanDerm, cập nhật CSV/đường dẫn ảnh/checkpoint và chạy `bash script/finetune_train.sh` từ thư mục `classification/`. Đây là nơi fine-tune nhánh phân loại trên ISIC 2017 Task 3.
4. Nếu dùng weighted sampler, chỉ áp dụng trên train.
5. Chọn checkpoint và mọi ngưỡng từ validation; không dùng test để chọn prompt, augmentation hay class threshold.
6. Lưu logits/probabilities của từng ảnh test để có thể tái tạo confusion matrix, AUROC và calibration.

### 5.3. Không tuyên bố vượt benchmark từ bài báo khác

Không so sánh trực tiếp số của nhóm với ISIC leaderboard hoặc bài PanDerm nếu split, xử lý mask, phiên bản mã nguồn, metric hoặc dữ liệu khác nhau. Có thể dẫn kết quả công bố trong Related Work để đặt bối cảnh, nhưng không gọi đó là đối chứng trực tiếp.

### Điều kiện đạt phase

- Có checkpoint hợp lệ và script suy luận tái lập.
- Tập test chưa bị dùng để lựa cấu hình.
- Metrics không NaN/Inf, confusion matrix khớp số lượng ảnh.
- Báo cả kết quả theo lớp và phân bố mẫu; không chỉ báo accuracy tổng.

---

## Phase 6 — Tạo dự báo Ridge cho ảnh dọc UQ

**Thời gian:** tuần 5–9  
**Chủ trì:** SV B xây dựng; SV A duyệt logic dữ liệu và phân tích.

**Nơi chạy:** ghép cặp và Ridge chạy trên CPU bằng pandas/scikit-learn; GPU không cần thiết. Chỉ chạy PanDerm inference để tạo mask/đặc trưng trên UQ trong môi trường được điều khoản dữ liệu cho phép.

### 6.1. Xây các cặp liên tiếp

1. Sắp xếp ảnh theo lesion_id rồi timestamp.
2. Tạo cặp kề nhau t → t+1; t+1 là ảnh dermoscopy tiếp theo thật sự có trong dữ liệu của cùng lesion, không mặc định là lần khám sau đúng 6 tháng; không nối hai lesion khác nhau.
3. Ghi Δt thực tế theo ngày; nếu chuyển sang tháng/năm thì ghi phép chuyển đổi chính xác.
4. Loại các cặp không xác định được thứ tự, Δt bằng 0, ảnh lỗi hoặc loại ảnh không phù hợp; ghi số lượng/lý do.
5. Nếu một lesion có nhiều visits, có thể dùng các cặp liên tiếp; mọi cặp của participant vẫn nằm trong cùng một split.

### 6.2. Tạo features

Features chỉ dùng tại thời điểm t:

- tỷ lệ diện tích mask PanDerm;
- circularity và eccentricity từ mask, nếu tính ổn định;
- điểm/xác suất ba nhãn tham khảo từ ISIC;
- khoảng thời gian Δt.

Chuẩn hóa features bằng thống kê tính trên train. Không đưa mask t+1, nhãn hậu kiểm, chẩn đoán sau này, histology hoặc ngày tương lai vào features trừ khi được định nghĩa rõ là có sẵn tại thời điểm sử dụng.

### 6.3. Huấn luyện

1. Tạo target Δa từ lần kế tiếp bằng cùng phép đo mask đã khóa.
2. Fit Ridge trên train.
3. Chọn alpha trên validation; có thể dùng một lưới nhỏ cố định và không mở rộng sau khi xem test.
4. Giữ baseline persistence: dự báo Δa = 0.
5. Khóa pipeline và checkpoint; mở participant test đúng một lần cho phân tích chính.
6. Tạo prediction file gồm participant, lesion, visit t, Δt, target, dự báo Ridge, baseline, trạng thái mask và cờ chất lượng. Chỉ giữ ID được phép.

### 6.4. Hai mức đánh giá cần phân biệt

**Mức A — dự báo mask tự động:** trên các cặp UQ, Ridge dự báo diện tích mask do PanDerm tạo ra ở lần sau; đánh giá với mask PanDerm của lần sau trên participant test. Đây là phép đo nhất quán nhưng mục tiêu được tạo bởi chính pipeline AI.

**Mức B — audit diện tích tham chiếu thủ công:** trên các cặp đã gán mask tay, tính area theo mask thủ công và báo sai số mask tự động so với thủ công. Nếu số lượng cho phép, đánh giá dự báo đã khóa trên cặp thủ công test. Đây là phân tích hỗ trợ; mô tả rõ số lượng nhỏ và quy trình gán.

Không gộp hai mức thành một kết quả hoặc gọi mức A là ground truth. Nếu chỉ làm được mức A, kết luận phải nói “mô hình dự báo kích thước mask tự động”, không nói “bệnh sẽ lớn lên/nhỏ đi”.

### 6.5. Kiểm tra chống rò rỉ dự báo

- Participant test không xuất hiện ở train/validation.
- Features chỉ lấy từ visit t.
- Không fit scaler hoặc chọn alpha trên test.
- Không dùng t+1 để chọn pair, features, threshold hoặc checkpoint.
- Không ghép cặp sau khi chia bằng quy tắc có thể làm lẫn participant.
- Cặp bị loại được thống kê trước khi mở test.
- Chạy kiểm thử tự động xác nhận mọi điều kiện trên.

---

## Phase 7 — Bộ kiểm thử và chỉ số bắt buộc

**Thời gian:** xây trong tuần 3–8; chạy final tuần 9–10.

### 7.1. Kiểm tra dữ liệu và hệ thống

| Nhóm test | Phép kiểm | Kết quả phải lưu |
|---|---|---|
| Tệp | Ảnh/mask mở được, kích thước đúng, hash trùng được xử lý | Tỷ lệ tệp lỗi, danh sách loại |
| Metadata | ID, ngày chụp, nhãn, số thiếu | Data dictionary và missingness |
| Split | Không trùng participant/lesion giữa tập | Test leakage bằng 0 hoặc giải thích ngoại lệ |
| Thời gian | Cặp cùng lesion, timestamp tăng, Δt hợp lệ | Số cặp hợp lệ và phân bố khoảng cách |
| Pipeline | Input/output đúng shape, output hữu hạn | Log smoke test và phiên bản môi trường |
| Tái lập | Chạy lại cùng cấu hình | Cùng manifest, cùng seed; chênh lệch số do nondeterminism được ghi |
| Demo | Ảnh thiếu/không đọc được, mask rỗng, Δt sai | Thông báo lỗi dễ hiểu, không crash âm thầm |

### 7.2. Phân đoạn

**Chỉ số chính**

- Dice coefficient.
- Intersection over Union (IoU).

**Báo cáo**

- Macro trung bình theo ảnh, không chỉ cộng tất cả pixel thành một mask lớn.
- Cách xử lý 5 mask tham chiếu ISIC.
- Trung bình và khoảng tin cậy 95%.
- Tỷ lệ mask rỗng/lỗi và ví dụ thất bại.
- Trên UQ audit: Dice, IoU và sai số tuyệt đối tỷ lệ diện tích so với mask thủ công.
- Nếu đủ khả năng, thêm boundary F1 hoặc HD95 như chỉ số phụ; không để chỉ số phụ thay thế Dice/IoU.

### 7.3. Phân loại nhóm tham khảo

**Chỉ số chính**

- Macro-F1.
- Balanced accuracy.
- AUROC one-vs-rest cho mỗi lớp và macro average.

**Bắt buộc kèm**

- Confusion matrix có số lượng.
- Sensitivity/recall từng lớp.
- Số mẫu mỗi lớp và tỷ lệ mất cân bằng.
- Brier score hoặc calibration plot nếu giao diện trình bày đầu ra như xác suất.
- Ngưỡng phân loại được chọn trên validation và khóa trước test.

Không gọi output là chẩn đoán. Trong demo và bài báo ghi là điểm/xác suất lớp theo nhãn dữ liệu.

### 7.4. Dự báo thay đổi diện tích

**Chỉ số chính**

- MAE của Δa.
- RMSE của Δa.
- Sai lệch trung bình (mean signed error).
- MAE/RMSE của Ridge so với baseline Δa = 0.

**Chỉ số phụ**

- R², nếu phân bố target cho phép diễn giải.
- Độ đúng hướng tăng/giảm/ổn định; định nghĩa ngưỡng “ổn định” từ validation hoặc độ bất đồng người gán, không lấy từ test.
- Sai số theo nhóm khoảng Δt và mức diện tích ban đầu, chỉ khi mỗi nhóm có đủ mẫu.
- Biểu đồ predicted-vs-observed và residual-vs-Δt.

Diễn giải MAE theo điểm phần trăm diện tích ảnh để người đọc hiểu quy mô sai số. Không gọi đây là sai số tăng trưởng vật lý.

### 7.5. Độ bất định và thống kê

- Với UQ, resample theo participant, không resample từng cặp độc lập.
- Báo khoảng tin cậy bootstrap 95% cho metrics và chênh lệch Ridge − baseline; dùng cùng tập test cho hai phương pháp.
- Dùng 2.000 bootstrap replicates nếu chạy được; lưu seed.
- Với ISIC, bootstrap theo lesion nếu có ID lesion đáng tin cậy; nếu không, giải thích đơn vị lấy mẫu.
- Nếu chạy 3 seed, báo trung bình ± độ lệch chuẩn qua seed và khoảng tin cậy theo mẫu riêng; không trộn hai loại biến thiên.
- Chỉ định trước một kết quả dự báo chính; các metric còn lại là phụ. Không chọn metric tốt nhất sau khi xem test.
- Nghiên cứu vẫn có kết quả khoa học nếu Ridge không tốt hơn baseline; cần báo trung thực và phân tích vì sao.

---

## Phase 8 — Thử nghiệm độ bền ảnh và xử lý tương phản

**Thời gian:** tuần 8–10  
**Chủ trì:** SV B tạo biến đổi; SV A duyệt mức và lập bảng kết quả.

**Nơi chạy:** tạo biến đổi brightness/blur/white balance/hair bằng Pillow hoặc OpenCV trên CPU; chạy lại PanDerm inference bằng máy GPU local hoặc GPU trường. Sinh biến thể theo batch để tránh lưu nhiều bản sao ảnh. Chỉ xử lý ảnh UQ trên máy được phép giữ UQ.

### 8.1. Phép suy giảm ảnh

| Phép thử | Mức dự kiến | Quy tắc |
|---|---|---|
| Độ sáng | α = 0,8 và 1,2 | Cắt pixel về dải hợp lệ; giữ ảnh gốc đối chiếu |
| Độ mờ | Gaussian blur σ = 1,0 và 1,5 pixel | Không làm thay đổi nhãn hoặc mask |
| Cân bằng trắng | Gain kênh đỏ/xanh ±10% và ±20% | Ghi rõ công thức và seed |
| Lông che | Che khoảng 1% và 3% diện tích ảnh | Gắn nhãn đây là vật cản tổng hợp, không phải dữ liệu lâm sàng thật |

### 8.2. Quy trình không rò rỉ

1. Chia tập ảnh gốc trước.
2. Tạo biến thể suy giảm sau khi split.
3. Test chỉ là các biến thể của ảnh trong test; không đưa các biến thể sang train.
4. Giữ target và mask tham chiếu gốc cố định khi đánh giá robustness đầu vào.
5. Đánh giá cùng ảnh clean và degraded theo cặp.
6. Báo mức giảm tuyệt đối của Dice, Macro-F1, MAE/RMSE và khoảng tin cậy paired bootstrap.
7. Không thử toàn bộ tích phương pháp xử lý × suy giảm; chạy từng loại một.

### 8.3. Tăng cường tương phản

- Cấu hình chính: ảnh RGB cùng normalization chính thức PanDerm.
- Ben Graham và Gamma là thí nghiệm phụ; chọn tối đa một biến thể dựa trên validation.
- Nếu chọn được, khóa biến thể rồi đánh giá một lần trên test.
- Gabor chỉ là phân tích kết cấu bổ sung nếu còn thời gian. Không gọi Gabor là phép tăng tương phản và không ghép thêm kênh thứ tư vào PanDerm RGB trong cấu hình chính.
- Báo kết quả âm tính nếu xử lý không cải thiện hoặc làm xấu metric; không loại khỏi báo cáo chỉ vì không có lợi.

---

## Phase 9 — Tích hợp demo và kiểm thử end-to-end

**Thời gian:** tuần 10–11  
**Chủ trì:** SV B; SV A duyệt câu chữ và giới hạn.

**Nơi chạy:** làm demo bằng Streamlit chạy cục bộ trên máy nhóm. Không cần triển khai website công khai; nếu sau này muốn chia sẻ, chỉ dùng ảnh và checkpoint được phép công bố.

### Demo tối thiểu

Đầu vào:

- một ảnh dermoscopy;
- khoảng thời gian dự kiến đến lần tái khám kế tiếp.

Đầu ra:

- ảnh gốc;
- mask overlay;
- ba điểm/xác suất nhóm bệnh tham khảo;
- tỷ lệ diện tích mask hiện tại;
- tỷ lệ diện tích mask dự báo và Δa;
- cảnh báo nếu ảnh không đạt điều kiện hoặc mask có chất lượng thấp.

Thông báo cố định:

> Đây là demo nghiên cứu trên ảnh dermoscopy. Nhóm bệnh là nhãn tham khảo theo dữ liệu; dự báo là thay đổi tỷ lệ diện tích mask trên ảnh, không phải chẩn đoán, tiên lượng bệnh hoặc khuyến nghị điều trị.

### Kiểm thử end-to-end

1. Dựng môi trường sạch theo hướng dẫn.
2. Nạp checkpoint segmentation, classification, Ridge và cấu hình.
3. Chạy một ảnh hợp lệ từ đầu đến cuối.
4. Kiểm tra mask overlay đúng vị trí, kích thước và màu.
5. Kiểm tra xác suất hữu hạn; tổng xác suất đúng quy ước.
6. Kiểm tra đầu ra area nằm trong [0,1] sau clip.
7. Kiểm tra Δt âm, bằng 0, thiếu hoặc không phải số đều bị từ chối.
8. Kiểm tra ảnh hỏng và mask rỗng có thông báo; không sinh kết luận giả.
9. Kiểm tra cùng ảnh, cùng checkpoint và cùng Δt cho đầu ra tái lập trong mức sai số số học.
10. Chạy demo với dữ liệu công khai được phép; không đưa ảnh UQ hạn chế quyền lên web công khai.

### Sản phẩm

- Demo cục bộ.
- Mã giao diện Streamlit và lệnh chạy cục bộ.
- Hướng dẫn chạy.
- Ảnh chụp demo không chứa dữ liệu bị hạn chế.
- File test và log kiểm thử.

---

## Phase 10 — Phân tích lỗi, khóa kết quả và tạo hình/bảng

**Thời gian:** tuần 10–12  
**Chủ trì:** SV A phân tích; SV B tái tạo số liệu từ code.

### Phân tích lỗi có kế hoạch

- Chọn ví dụ theo quy tắc định trước: đúng tốt, lỗi vừa, lỗi nặng; không chỉ chọn ảnh đẹp.
- Ghi nguyên nhân quan sát được: blur, hair, low contrast, tổn thương nhỏ, biên mờ, màu lệch, mask cắt biên.
- Phân biệt lỗi segmentation dẫn tới sai target với lỗi Ridge dự báo.
- Báo riêng cặp bị thiếu mask, mask rỗng và cặp không tạo được dự báo.
- Nếu UQ có biến tuổi/giới/vùng cơ thể và quyền cho phép, chỉ báo phân tầng khi nhóm đủ mẫu và không lộ thông tin cá nhân. Không suy luận fairness nếu dữ liệu thiếu hoặc cỡ mẫu nhỏ.

### Hình và bảng tối thiểu cho bài báo

1. **Sơ đồ dòng dữ liệu:** ảnh nguồn → lọc → split theo nhóm → PanDerm → mask/features → Ridge → đánh giá.
2. **Bảng đặc điểm dữ liệu:** số ảnh, participant, lesion, pair, Δt, missingness theo split.
3. **Bảng segmentation:** Dice/IoU ISIC và audit UQ, khoảng tin cậy.
4. **Bảng classification:** Macro-F1, balanced accuracy, AUROC và confusion matrix.
5. **Bảng forecast:** MAE/RMSE/bias Ridge và baseline, mức A tự động và mức B thủ công tách riêng.
6. **Biểu đồ robustness:** hiệu năng clean so với từng loại/mức suy giảm.
7. **Biểu đồ dự báo:** predicted-vs-observed, residual-vs-Δt.
8. **Hình định tính:** ảnh, mask tham chiếu, mask dự đoán; cả ví dụ thành công và thất bại.

### Bảng trống để nhóm điền kết quả thật

| Thử nghiệm | N test | Metric | Kết quả | CI 95% | Ghi chú |
|---|---:|---|---:|---|---|
| PanDerm segmentation — ISIC 2018 | [điền] | Dice / IoU | [điền sau khi chạy] | [điền] | Cách tổng hợp nhiều mask: [điền] |
| PanDerm segmentation — UQ audit | [điền] | Dice / IoU / area error | [điền sau khi chạy] | [điền] | Số ảnh gán tay: [điền] |
| Nhãn tham khảo — ISIC 2017 | [điền] | Macro-F1 / BAcc / AUROC | [điền sau khi chạy] | [điền] | N test mỗi lớp: [điền] |
| Ridge dự báo Δa — UQ | [điền cặp, [điền] người] | MAE / RMSE / bias | [điền sau khi chạy] | [điền] | Tách mức A/B |
| Baseline không thay đổi — UQ | [điền cặp, [điền] người] | MAE / RMSE / bias | [điền sau khi chạy] | [điền] | Cùng participant test |
| Độ bền ảnh | [điền] | ΔDice / ΔMacro-F1 / ΔMAE | [điền sau khi chạy] | [điền] | Từng phép suy giảm |

Không điền trước số kỳ vọng. Nếu kết quả không vượt baseline hoặc có CI rộng, để nguyên kết quả thật và điều chỉnh kết luận.

---

## Phase 11 — Viết bài nghiên cứu hoàn chỉnh

**Thời gian:** tuần 11–14  
**Chủ trì:** SV A điều phối bản thảo; SV B cung cấp kỹ thuật và tái lập; cả hai kiểm duyệt.

### Cấu trúc bài báo

1. **Tiêu đề:** nêu rõ “dự báo thay đổi mask/diện tích trên ảnh” nếu dùng endpoint hiện tại.
2. **Tóm tắt có cấu trúc:** Bối cảnh, Mục tiêu, Phương pháp, Kết quả, Kết luận. Chỉ viết số liệu sau khi khóa kết quả.
3. **Giới thiệu:** vấn đề theo dõi ảnh; sai khác chất lượng ảnh; khoảng trống dự báo một bước; mục tiêu cụ thể.
4. **Phương pháp:**
   - thiết kế nghiên cứu;
   - nguồn dữ liệu, quyền sử dụng và tiêu chí chọn/loại;
   - chuẩn hóa nhãn và tạo cặp;
   - chia tập theo participant/lesion;
   - PanDerm Base và cách fine-tune;
   - Ridge, features, Δt, target và baseline;
   - tăng cường tương phản và suy giảm;
   - metrics, bootstrap và phần mềm/phần cứng.
5. **Kết quả:** sơ đồ chọn dữ liệu, đặc điểm split, segmentation, nhãn tham khảo, forecast, robustness và phân tích lỗi.
6. **Thảo luận:** ý nghĩa trong phạm vi nghiên cứu ảnh, so với tài liệu liên quan, khả năng áp dụng và giới hạn.
7. **Kết luận:** trả lời đúng câu hỏi; không đổi “thay đổi mask” thành “tiến triển bệnh”.
8. **Tuyên bố:** đạo đức/quyền dữ liệu, nguồn tài trợ nếu có, đóng góp tác giả, xung đột lợi ích, chia sẻ code/data.
9. **Tài liệu tham khảo và phụ lục:** cấu hình, split logic, data dictionary được phép, checklist reporting.

### Các giới hạn phải ghi trong Discussion

- UQ có thể không có mask phân đoạn chuyên gia; mask tự động có thể sai ngoài miền ISIC.
- Nếu forecast target dựa trên mask do PanDerm sinh, đây là dự báo một phép đo của mô hình, không phải ground truth sinh học.
- Thay đổi góc chụp, độ phóng đại, ánh sáng hoặc framing có thể đổi area theo pixel.
- Nhãn nhóm bệnh là nhãn tham chiếu theo dataset, không dùng chẩn đoán cá nhân.
- Dữ liệu hồi cứu và điều khoản dataset giới hạn khả năng khái quát.
- Cỡ tập audit nhỏ, nếu có, cần báo khoảng bất định và không tuyên bố kiểm định lâm sàng.
- Kiểm tra overlap pretraining/test nếu đủ thông tin; nếu không xác minh được, nêu rõ nguy cơ.

### Checklist trước khi nộp

- [ ] Tên dataset, phiên bản, URL, ngày truy cập và giấy phép đúng.
- [ ] Mọi số lượng trong Abstract/Results khớp với flow diagram và manifest.
- [ ] Mọi metric được tạo lại từ script; bảng báo đúng split.
- [ ] Không có participant/lesion leakage.
- [ ] Mọi lựa chọn checkpoint/ngưỡng chỉ dùng train/validation.
- [ ] Baseline dự báo dùng đúng participant test như Ridge.
- [ ] Các phép suy giảm chỉ nằm trên test sau khi cấu hình đã khóa.
- [ ] Hình minh họa không gây hiểu nhầm về chẩn đoán hoặc tăng trưởng sinh học.
- [ ] Ghi rõ model/version, checkpoint, seed, hardware, package versions.
- [ ] Rà theo CLAIM 2024 cho báo cáo AI hình ảnh y khoa.
- [ ] Kiểm tra trích dẫn, chính tả, bảng/hình, định dạng cuộc thi.
- [ ] Giảng viên hướng dẫn đã đọc bản cuối.

---

## 5. Tiêu chí nghiệm thu theo phase

| Phase | Được xem là hoàn thành khi |
|---|---|
| 0. Protocol | Câu hỏi/target/split/metrics/baseline khóa bằng văn bản |
| 1. Quyền và UQ | Có Go/No-Go có căn cứ; nếu No-Go, đã trình phương án đổi đề tài |
| 2. Repo/model | Môi trường sạch nạp checkpoint và chạy smoke test |
| 3. Dữ liệu | Manifest, data dictionary, split và kiểm tra leakage hoàn tất |
| 4. Mask UQ | Có protocol gán, audit subset và inter-annotator result hoặc nêu rõ không thực hiện |
| 5. PanDerm | Checkpoint segmentation/classification được chọn trên validation và test đã khóa |
| 6. Ridge | Cặp thời gian, features, target, baseline và participant test kiểm tra xong |
| 7. Đo lường | Tất cả metric chính, CI và log có thể tái tạo |
| 8. Robustness | Biến thể ảnh có seed, split đúng, kết quả clean/degraded đầy đủ |
| 9. Demo | Chạy end-to-end, xử lý lỗi, disclaimer và không lộ dữ liệu |
| 10. Phân tích | Bảng/hình được sinh từ script và có cả kết quả bất lợi |
| 11. Bài báo | Bản thảo đủ Methods/Results/Discussion, trích dẫn và checklist |

Tiêu chí nghiệm thu đánh giá **tính hoàn thành và tính hợp lệ của quy trình**, không đặt một ngưỡng hiệu năng giả trước khi chạy. Kết quả thấp hoặc không hơn baseline vẫn là kết quả cần báo cáo.

---

## 6. Phương án dự phòng

### UQ không cấp quyền hoặc thiếu ID/thời gian

- Dừng nhánh dự báo UQ.
- Trình giảng viên chọn dataset dọc khác có quyền và ID đủ, hoặc đổi phạm vi thành nghiên cứu độ bền segmentation/classification trên ISIC.
- Nếu đổi phạm vi, viết lại protocol và đề cương; không giữ kết luận về dự báo.

### Không chạy được PanDerm Base segmentation

- Dùng tối đa hai tuần đầu để kiểm tra version/config/checkpoint.
- Thử cấu hình PanDerm-Large chỉ khi tài nguyên thực tế đủ và được giảng viên duyệt.
- Nếu không, sử dụng lựa chọn ít công hơn đã có trong repo (linear probe cho phân loại) và điều chỉnh câu hỏi; không tự ý ghép U-Net/ViT mới vào cuối kỳ.

### PanDerm segmentation chuyển miền UQ kém

- Báo Dice/IoU và area error audit.
- Không diễn giải Ridge như dự báo thay đổi lesion nếu phép đo đầu vào không đủ tin cậy.
- Có thể thu hẹp bài thành đánh giá độ bền/chuyển miền; chỉ giữ nhánh forecast nếu audit cho thấy có cơ sở và số lượng đủ.

### Không đủ thời gian gán mask UQ

- Giữ đánh giá forecast mức A là dự báo mask tự động.
- Bỏ mọi tuyên bố về diện tích thật hoặc tiến triển sinh học.
- Ghi số lượng thủ công thực hiện được và lý do giới hạn.

### GPU không đủ hoặc không ổn định

- Chạy pilot ngắn trước khi huấn luyện dài.
- Lưu checkpoint/log định kỳ vào nơi nhóm được phép dùng.
- Hạ batch size hoặc chạy encoder đóng băng/linear probe trong cùng PanDerm nếu pipeline hỗ trợ.
- Không chuyển checkpoint/dữ liệu hạn chế qua dịch vụ đám mây nếu điều khoản dataset không cho phép.

---

## 7. Quản lý phiên bản và lưu trữ kết quả

Mỗi lần chạy có một run card gồm:

- run_id;
- ngày/giờ;
- người chạy;
- git commit;
- phiên bản data manifest và hash;
- split seed;
- checkpoint gốc và hash;
- config;
- random seed;
- GPU/VRAM;
- package versions;
- log;
- metric;
- đường dẫn prediction file.

Không ghi đè kết quả. Dùng thư mục riêng theo ngày/run_id. Chỉ lưu bản sao dữ liệu theo quyền cho phép. Mỗi figure/table được tạo bằng script từ prediction/log, không nhập số thủ công.

### Kiểm tra tái lập cuối

Một thành viên chưa chạy thí nghiệm chính cần làm theo hướng dẫn từ môi trường sạch để:

1. đọc manifest;
2. nạp checkpoint;
3. chạy một inference sample;
4. tạo một metric nhỏ;
5. xác nhận định dạng bảng cuối.

Nếu không tái tạo được, sửa hướng dẫn hoặc ghi giới hạn trước khi nộp.

---

## 8. Sản phẩm cuối cùng

Nhóm chỉ được xem là hoàn thành khi có:

1. Bài nghiên cứu khoa học hoàn chỉnh với kết quả thật.
2. Bảng mô tả dữ liệu, manifest/split được phép chia sẻ và sơ đồ flow.
3. Checkpoint/config/log/prediction files có thể truy lại phiên bản và không vi phạm quyền dữ liệu.
4. Bộ script tạo lại metrics và hình/bảng.
5. Báo cáo Dice/IoU, Macro-F1/Balanced Accuracy/AUROC, MAE/RMSE/bias và phân tích độ bền ảnh theo kế hoạch.
6. Demo cục bộ kèm hướng dẫn và thông báo giới hạn.
7. Poster/tóm tắt cuộc thi nếu yêu cầu.
8. Checklist CLAIM 2024 đã rà soát.

**Không được xem là hoàn thành** nếu chỉ có demo mà không có đánh giá trên test; nếu có số kết quả nhưng không biết sinh từ split nào; hoặc nếu bài viết gọi thay đổi mask là tiến triển bệnh.

---

## 9. Nguồn và tài liệu nên dùng

Các URL dưới đây là điểm bắt đầu. Nhóm cần đọc lại điều khoản tại thời điểm tải và ghi ngày truy cập vào bài báo.

1. **PanDerm — repo chính thức:** [github.com/SiyuanYan1/PanDerm](https://github.com/SiyuanYan1/PanDerm)
2. **PanDerm — hướng dẫn segmentation:** [Segmentation.md](https://github.com/SiyuanYan1/PanDerm/blob/main/Segmentation.md)
3. **PanDerm — bài báo:** [Nature Medicine, DOI 10.1038/s41591-025-03747-y](https://doi.org/10.1038/s41591-025-03747-y)
4. **UQ Longitudinal Skin Image Dataset:** [UQ espace record](https://espace.library.uq.edu.au/view/UQ:a13deaf)
5. **UQ dataset description:** [Research Data Australia](https://researchdata.edu.au/a-longitudinal-dataset-skin-cancers/3750284)
6. **Bài báo mô tả UQ dataset:** [Scientific Data, DOI 10.1038/s41597-025-05880-2](https://doi.org/10.1038/s41597-025-05880-2)
7. **ISIC Challenge datasets:** [data page](https://challenge.isic-archive.com/data/)
8. **ISIC 2018 Task 1 — segmentation:** [official task page](https://challenge.isic-archive.com/landing/2018/45/)
9. **ISIC 2017 Challenge:** [official landing page](https://challenge.isic-archive.com/landing/2017/)
10. **CLAIM 2024 Update:** [Radiology: Artificial Intelligence, DOI 10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300)
11. **Ridge regression:** Hoerl & Kennard, 1970, [DOI 10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
12. **DermFM-Zero — tài liệu nền liên quan:** [repo](https://github.com/SiyuanYan1/DermFM-Zero), [bài arXiv 2602.10624](https://arxiv.org/abs/2602.10624). Không đưa vào pipeline chính nếu chưa sửa câu hỏi và protocol.

---

## 10. Nhật ký quyết định

| Ngày | Quyết định/thay đổi | Lý do | Ảnh hưởng tới protocol/kết quả | Người duyệt |
|---|---|---|---|---|
| 2026-10-07 | Trong `patches/panderm_base_seg.patch` (file `segmentation/models/cae_seg.py`), đổi dòng `new_state_dict = {k.replace('encoder.', ''): v for k, v in cae_weight.items() if 'encoder' in k}` thành `new_state_dict = dict(cae_weight)` (dòng context → cặp `-`/`+`, header hunk `@@ -13,18 +15,32 @@` giữ nguyên) | `inspect_checkpoint.py` trên `panderm_bb_data6_checkpoint-499.pth`: `wrapped_in: null`, 186 key, `prefix_counts` = `blocks` 180, `patch_embed` 2, `norm` 2, `cls_token` 1, `pos_embed` 1; `patch_embed_shape` [768, 3, 16, 16]. Key không có prefix `encoder.` nên bộ lọc `if 'encoder' in k` bỏ hết trọng số | Không đổi protocol. Backbone nạp trọng số PanDerm Base thay vì khởi tạo ngẫu nhiên; kiểm tra bằng dòng `ViT coverage` ≥ 90% khi dựng model | [điền] |
| [điền] | [điền] | [điền] | [điền] | [điền] |
