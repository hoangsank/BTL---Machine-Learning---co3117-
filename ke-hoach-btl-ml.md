# Kế hoạch Bài tập lớn Machine Learning (CO3117)

## 1. Dataset đã chọn

**Gas Turbine CO and NOx Emission Data Set** — UCI Machine Learning Repository, ID 551.

- Quy mô: 36.733 dòng, 11 biến (9 đặc trưng + 2 target: CO, NOx), dữ liệu cảm biến theo giờ từ một tuabin khí, thu thập 2011–2015.
- Bài toán: hồi quy (regression), dự đoán nồng độ khí thải CO hoặc NOx từ các thông số vận hành.

| Ký hiệu | Ý nghĩa | Đơn vị |
|---|---|---|
| AT | Nhiệt độ môi trường | °C |
| AP | Áp suất môi trường | mbar |
| AH | Độ ẩm môi trường | % |
| AFDP | Chênh áp bộ lọc khí | mbar |
| GTEP | Áp suất khí thải tuabin | mbar |
| TIT | Nhiệt độ khí vào tuabin | °C |
| TAT | Nhiệt độ khí ra tuabin | °C |
| CDP | Áp suất xả máy nén | mbar |
| TEY | Sản lượng điện của tuabin | MWH |

Target: **CO** và **NOx** (mg/m³) — chọn 1 target chính để tập trung, có thể thử target còn lại như phần mở rộng.

> Dataset không nằm trong danh sách gợi ý sẵn của môn, cần email xin GVHD duyệt trước khi bắt tay làm, nêu rõ lý do phù hợp (quy mô, tính chất hồi quy, thách thức đa cộng tuyến giữa các cảm biến).

---

## 2. Phân loại công việc: Bắt buộc và Mở rộng

Phần **Bắt buộc** là điều kiện đủ để đáp ứng yêu cầu đề bài. Phần **Mở rộng** được xếp theo thứ tự ưu tiên (chi phí thấp/giá trị cao trước) để nhóm cân nhắc làm thêm tùy thời gian thực tế còn lại, không bắt buộc phải làm hết.

### 2.1. Bắt buộc

| Hạng mục | Mô tả |
|---|---|
| Mô tả & biện minh dataset | Giải thích vì sao dataset phù hợp để áp dụng ML, nêu rõ thách thức chính (đa cộng tuyến giữa các cảm biến, có thể có drift theo thời gian) |
| EDA đầy đủ | Phân bố dữ liệu, số lượng mẫu, tương quan giữa các biến, kiểm tra missing/outlier |
| Tiền xử lý | Xử lý outlier (nếu có), chuẩn hoá dữ liệu cho các model cần scale |
| Áp dụng ≥ 2 thuật toán khác họ | Không dùng 2 biến thể cùng một họ thuật toán (ví dụ chỉ LightGBM và XGBoost đều là gradient boosting) — cần có ít nhất 1 model tuyến tính (Linear/Ridge) và 1 model dạng cây/ensemble (Random Forest hoặc Gradient Boosting) để thể hiện sự đa dạng thực sự |
| So sánh & phân tích kết quả | Bảng so sánh các model theo metric đã chọn, có diễn giải bằng lời chứ không chỉ liệt kê số |
| Giải thích lý do chọn metric | Nêu rõ vì sao chọn RMSE/MAE/R² thay vì chỉ báo cáo con số |
| Báo cáo tổng kết | Theo outline: Giới thiệu → Dữ liệu & tiền xử lý → Mô hình & phương pháp → Thực nghiệm & kết quả → Thảo luận → Kết luận & hướng phát triển |
| Slide thuyết trình | Chuẩn bị sẵn phòng trường hợp nhóm được gọi lên trình bày |

### 2.2. Mở rộng — Ưu tiên 1 (chi phí thấp, giá trị cao)

| Hạng mục | Mô tả | Lý do nên làm |
|---|---|---|
| Baseline "ngây thơ" | So sánh với một quy tắc đơn giản (ví dụ dự đoán bằng giá trị trung bình hoặc giá trị kỳ trước) | Trả lời câu hỏi "model ML có thực sự tốt hơn một cách làm đơn giản không" — nâng chất lượng phần Thảo luận mà không tốn nhiều công |
| Feature importance / permutation importance | Xác định đặc trưng nào ảnh hưởng nhiều nhất đến target | Hỗ trợ trực tiếp cho phần diễn giải kết quả |
| Diễn giải có cơ sở vật lý | Liên hệ kết quả model với hiện tượng thực tế: nhiệt độ buồng đốt (TIT) cao làm đốt cháy hoàn toàn hơn nên CO giảm, nhưng đồng thời thúc đẩy phản ứng tạo NOx (cơ chế Zeldovich) nên NOx tăng | Nếu model học được đúng xu hướng ngược chiều này giữa TIT với CO và NOx, đó là bằng chứng mạnh cho phần Thảo luận, chi phí chỉ là một đoạn văn ngắn |

### 2.3. Mở rộng — Ưu tiên 2 (chi phí trung bình)

| Hạng mục | Mô tả |
|---|---|
| Hyperparameter tuning có ghi log | GridSearchCV/RandomizedSearchCV, ghi lại quá trình tìm kiếm chứ không chỉ báo kết quả cuối cùng |
| Chia tập theo thời gian (rolling-origin) | Vì dữ liệu trải dài 5 năm, thử chia train/test theo từng năm (train 2011–2013, test 2014; rồi mở rộng dần) để đánh giá tính ổn định của model qua thời gian, thay vì chỉ 1 lần chia ngẫu nhiên |
| Kiểm định thống kê giữa các model | Paired t-test hoặc Wilcoxon trên phần dư (residual) để xác nhận model A tốt hơn model B có ý nghĩa thống kê, không chỉ dựa vào một con số nhỏ hơn |

### 2.4. Mở rộng — Ưu tiên 3 (chi phí cao hơn, làm nếu còn nhiều thời gian)

| Hạng mục | Mô tả |
|---|---|
| SHAP values | Giải thích đóng góp từng biến vào dự đoán, kiểm chứng lại quan hệ vật lý đã nêu ở mục 2.2 bằng số liệu thực nghiệm |
| Multi-output modeling | Dự đoán đồng thời cả CO và NOx bằng một model, so sánh với việc dùng 2 model tách biệt |
| Ensemble/stacking | Kết hợp các model tốt nhất, xem có cải thiện so với model đơn lẻ không |
| Robustness test | Thêm nhiễu Gaussian nhỏ vào input, đo mức tăng của RMSE để đánh giá độ nhạy của model — phù hợp bối cảnh dữ liệu cảm biến công nghiệp có sai số đo thực tế |

---

## 3. Phân công vai trò

| Vai trò | Người phụ trách | Công việc chính |
|---|---|---|
| **Data Lead** | Thành viên 1 | EDA, làm sạch dữ liệu, xử lý outlier, viết phần "Dữ liệu & tiền xử lý" |
| **Feature Engineer** | Thành viên 2 | Feature engineering, feature selection/importance, phần mở rộng liên quan diễn giải vật lý |
| **Modeling Lead 1** | Thành viên 3 | Baseline (Linear/Ridge), baseline ngây thơ, Random Forest |
| **Modeling Lead 2** | Thành viên 4 | Gradient Boosting, hyperparameter tuning, các phần mở rộng nâng cao (SHAP, ensemble) nếu triển khai |

Nguyên tắc chung:
- Mỗi người chủ trì 1 mảng nhưng tất cả đều cần hiểu toàn bộ pipeline.
- Dùng chung một module tiền xử lý, không viết lại rời rạc giữa các phần.
- Review chéo code trước khi gộp vào nhánh chính.

---

## 4. Lộ trình theo giai đoạn

Lộ trình chia theo **giai đoạn công việc** thay vì tuần lịch cố định, vì thời gian hiệu quả thực tế thường không trải đều theo 2.5 tháng lịch (xen kẽ các môn học khác, giữa kỳ...). Mỗi giai đoạn ước lượng theo tuần làm việc thực tế.

### Giai đoạn 1 — Hiểu bài toán & EDA (~1 tuần) — Bắt buộc
- Đọc mô tả dataset, tìm 1–2 tài liệu liên quan để có mốc so sánh.
- EDA: phân bố target, tương quan giữa các biến, kiểm tra outlier.
- Thống nhất cách xử lý outlier, ghi lại lý do.

### Giai đoạn 2 — Tiền xử lý & Feature Engineering (~1 tuần) — Bắt buộc
- Chuẩn hoá dữ liệu cho các model cần scale.
- Xây dựng module tiền xử lý dùng chung cho cả nhóm.
- Nếu chọn hướng chia theo thời gian (mở rộng ưu tiên 2), chuẩn bị cấu trúc chia theo năm ngay từ giai đoạn này.

### Giai đoạn 3 — Model chính (~1–1.5 tuần) — Bắt buộc
- Modeling Lead 1: Linear/Ridge Regression, Random Forest.
- Modeling Lead 2: Gradient Boosting.
- Đảm bảo có ít nhất 2 họ thuật toán khác nhau, không chỉ 2 biến thể cùng họ.

### Giai đoạn 4 — Đánh giá & so sánh cơ bản (~1 tuần) — Bắt buộc + Mở rộng ưu tiên 1
- Thống nhất bộ metric, viết rõ lý do chọn từng metric.
- Bổ sung baseline ngây thơ để so sánh.
- Chạy feature importance, viết đoạn diễn giải có cơ sở vật lý.

### Giai đoạn 5 — Mở rộng ưu tiên 2 (~1–1.5 tuần, tùy thời gian còn lại)
- Hyperparameter tuning có ghi log.
- Rolling-origin split theo năm.
- Kiểm định thống kê giữa các model.

### Giai đoạn 6 — Mở rộng ưu tiên 3 (tùy chọn, nếu còn effort)
- SHAP, multi-output modeling, ensemble, robustness test — chọn 1–2 mục phù hợp nhất với thời gian còn lại, không cần làm hết.

### Giai đoạn 7 — Viết báo cáo & chuẩn bị nộp (~1–1.5 tuần) — Bắt buộc
- Viết theo outline đề bài.
- Chuẩn bị README.md, requirements.txt/environment.yml.
- Đóng gói theo đúng định dạng nộp bài.
- Chuẩn bị slide.

### Giai đoạn 8 — Buffer
- Kiểm tra lại tính tái hiện của code, sửa lỗi phát sinh trước hạn nộp.

---

## 5. Checklist trước khi nộp

- [ ] Source code + data (hoặc link Google Drive/GitHub nếu file lớn)
- [ ] README.md mô tả cách chạy, môi trường, ghi chú
- [ ] requirements.txt / environment.yml (khuyến khích kèm Dockerfile)
- [ ] Báo cáo tổng kết (đúng outline)
- [ ] Slide thuyết trình
- [ ] Đặt tên file zip đúng format: `MSSV1-MSSV2-MSSV3-MSSV4.zip`
- [ ] Đã áp dụng ≥ 2 họ thuật toán thực sự khác nhau (không chỉ 2 biến thể cùng họ)
- [ ] Đã giải thích rõ lý do chọn metric, không chỉ liệt kê số liệu
- [ ] Không dùng API thương mại thay thế toàn bộ pipeline

---

## 6. Ghi chú làm việc chung

- Dùng chung 1 repo Git, mỗi người 1 branch theo tên vai trò (`data-eda`, `feature-eng`, `modeling-1`, `modeling-2`), merge về `main` qua Pull Request.
- Họp ngắn cuối mỗi giai đoạn (15–20 phút) để đồng bộ tiến độ.
- Nếu ai bị trễ tiến độ ở phần nào, báo sớm trong nhóm để người khác hỗ trợ kịp — tránh dồn việc vào giai đoạn cuối.
- Ưu tiên hoàn thành trọn vẹn phần Bắt buộc trước khi bắt đầu bất kỳ mục Mở rộng nào, kể cả khi còn nhiều thời gian trên lịch.
