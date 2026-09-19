# BTL Machine Learning (CO3117) - Gas Turbine CO and NOx Emission

## 📌 Tổng quan dự án
Dự án nhằm giải quyết bài toán hồi quy (regression) để dự đoán nồng độ khí thải **CO** và **NOx** dựa trên các thông số vận hành của một tuabin khí. Bộ dữ liệu (Dataset ID 551 từ UCI) bao gồm 36.733 dòng và 11 biến (9 đặc trưng, 2 target)[cite: 1]. 

Mục tiêu chính: Áp dụng các thuật toán Machine Learning đa dạng (Linear, Tree-based/Ensemble) để có mô hình dự đoán chính xác nhất, đồng thời giải thích được các yếu tố vật lý ảnh hưởng đến kết quả[cite: 1].

---

## 📂 Cấu trúc thư mục & Tác dụng của từng file

Dự án được chia thành các thư mục với chức năng rõ ràng để làm việc nhóm hiệu quả, không dẫm chân lên nhau.

*   **`data/`**: Nơi chứa dữ liệu. 
    *   *Lưu ý:* Thư mục này đã được đưa vào `.gitignore` để tránh push nhầm dữ liệu hàng chục ngàn dòng lên GitHub[cite: 1]. Hãy tự tải dataset từ UCI và chép vào `data/raw/` trên máy cá nhân.
*   **`notebooks/`**: Nơi các thành viên làm "nháp", vẽ biểu đồ, chạy thử nghiệm nghiệm mô hình[cite: 4]. 
    *   `01_eda.ipynb`: (**Data Lead**) Làm EDA, kiểm tra phân bố, outlier và tương quan (đa cộng tuyến)[cite: 1, 4].
    *   `02_feature_eng.ipynb`: (**Feature Engineer**) Làm Feature Engineering, feature importance, phân tích cơ sở vật lý[cite: 1, 4].
    *   `03_model_baseline.ipynb`: (**Modeling Lead 1**) Huấn luyện mô hình Baseline, Linear/Ridge Regression, Random Forest[cite: 1, 4].
    *   `04_model_boosting.ipynb`: (**Modeling Lead 2**) Huấn luyện Gradient Boosting, hyperparameter tuning, phân tích SHAP[cite: 1, 4].
*   **`src/`**: Mã nguồn (source code) chuẩn để cả nhóm dùng chung[cite: 4].
    *   Quy tắc: Code thử nghiệm bên `notebooks/` khi đã chạy tốt phải được đóng gói thành các "hàm" (functions) và đưa vào đây[cite: 1].
    *   `data_prep.py`: Module tiền xử lý dữ liệu chung, tránh tình trạng mỗi người viết một kiểu[cite: 1, 4].
    *   `evaluation.py`: Hàm tính toán các metric thống nhất (RMSE, MAE, R²)[cite: 1, 4].
*   **`docs/`**: Nơi lưu trữ báo cáo tổng kết và slide thuyết trình cuối kỳ[cite: 1, 4].
*   **`requirements.txt`**: Danh sách thư viện môi trường để chạy code[cite: 4].

---

## 🔄 Quy trình làm việc nhóm (Git Workflow)

Để tránh conflict code (đụng độ code) và mất dữ liệu, toàn bộ nhóm **tuyệt đối không commit trực tiếp lên nhánh `main`**. Hãy làm theo quy trình sau:

1. **Phân nhánh (Branching):** Mỗi người làm việc trên một nhánh riêng tương ứng với vai trò[cite: 1]:
   - Data Lead: `git checkout -b data-eda`[cite: 1]
   - Feature Engineer: `git checkout -b feature-eng`[cite: 1]
   - Modeling 1: `git checkout -b modeling-1`[cite: 1]
   - Modeling 2: `git checkout -b modeling-2`[cite: 1]
2. **Push & Pull Request (PR):** Sau khi hoàn thành một giai đoạn, push code lên nhánh cá nhân và tạo Pull Request (PR) trên GitHub để gộp vào nhánh `main`[cite: 1].
3. **Review chéo:** Các thành viên khác phải vào đọc code, thống nhất review chéo trước khi đồng ý gộp vào `main`[cite: 1].
4. **Họp đồng bộ:** Cần họp ngắn (15-20 phút) sau mỗi giai đoạn để đồng bộ. Ai trễ deadline cần báo ngay để được hỗ trợ[cite: 1].

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án

1. **Clone repo về máy:**
   ```bash
   git clone <link-repo-github>
   cd btl-ml-co3117



Dưới đây là giải thích chi tiết về vai trò của thư mục notebooks và cách phân bổ cho từng file:

1. Đặc điểm của file Notebook (.ipynb)
Khác với file Python thuần (.py) thường chạy một mạch từ đầu đến cuối, Notebook cho phép bạn chia code thành từng ô (cell) nhỏ. Khi chạy một ô code, kết quả (như bảng dữ liệu báo cáo, biểu đồ, hay con số) sẽ hiện ra ngay bên dưới ô đó. Điều này cực kỳ tiện lợi để vừa code, vừa xem dữ liệu, vừa ghi chú lại bằng văn bản (Markdown).


Trong kế hoạch có nhấn mạnh: "Dùng chung một module tiền xử lý, không viết lại rời rạc giữa các phần".   

Notebooks là nơi bạn thử nghiệm. Ví dụ: Data Lead viết code xử lý outlier bên 01_eda.ipynb.Khi code xử lý outlier đó đã chạy tốt, Data Lead sẽ chuyển đoạn code đó thành một "hàm" (function) chuẩn chỉnh và lưu vào file src/data_prep.py.   Sau đó, 3 thành viên còn lại ở các notebook 02, 03, 04 chỉ cần dùng lệnh from src.data_prep import ... để gọi hàm đó ra xài, thay vì copy-paste lại cả đoạn code dài vào notebook của mình.Cách làm này giúp code sạch sẽ, dễ tìm lỗi và đúng chuẩn làm việc nhóm chuyên nghiệp.