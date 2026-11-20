# Bài tập lớn Machine Learning — CO3117

Dự đoán khí thải **CO và NOx của tuabin khí** từ các thông số môi trường và vận hành. Dự án gồm tiền xử lý, phân tích đặc trưng, so sánh mô hình hồi quy và giải thích mô hình bằng SHAP.

## 1. Dữ liệu và phương pháp

Dữ liệu: **Gas Turbine CO and NOx Emission**, UCI Machine Learning Repository, ID **551**. CSV gốc được gửi kèm tại `data/raw/gas_turbine_data.csv`, không cần tải lại để chạy thí nghiệm.

- CSV gốc: **36.733 dòng**, gồm cột `year`, 9 đặc trưng và 2 biến mục tiêu `CO`, `NOX`.
- Loại 7 dòng trùng lặp, còn **36.726 dòng**.
- Train: năm **2011–2013**, **22.191 dòng**. Test: năm **2014–2015**, **14.535 dòng**.
- Chặn trên CO tại phân vị 99% tính riêng từ train (**10,7166**), lưu thành `CO_clipped`; áp dụng cùng ngưỡng cho test. `NOX` giữ nguyên.
- `StandardScaler` chỉ fit trên 9 đặc trưng của train, sau đó áp dụng cho test.

| Đặc trưng | Mô tả |
| --- | --- |
| AT | Nhiệt độ môi trường |
| AP | Áp suất môi trường |
| AH | Độ ẩm tương đối môi trường |
| AFDP | Chênh áp bộ lọc khí |
| GTEP | Áp suất khí thải tuabin |
| TIT | Nhiệt độ khí vào tuabin |
| TAT | Nhiệt độ khí ra tuabin |
| TEY | Sản lượng điện tuabin |
| CDP | Áp suất xả máy nén |

Các mô hình: dự đoán trung bình (`DummyRegressor`), Linear Regression, Ridge, Random Forest, KNN, MLP và XGBoost. Baseline tập trung vào `CO_clipped`; boosting dự đoán cả `CO_clipped` và `NOX`.

Đánh giá bằng **RMSE** (nhấn mạnh sai số lớn), **MAE** (sai số tuyệt đối trung bình) và **R²** (mức độ giải thích biến thiên). RMSE/MAE càng thấp càng tốt, R² càng cao càng tốt. Metric CO trong luồng chính được tính trên **CO đã chặn ngưỡng**, không phải toàn bộ giá trị CO gốc.

## 2. Cấu trúc thư mục

```text
.
├── README.md
├── requirements.txt
├── ke-hoach-btl-ml.md            # Kế hoạch và phân công ban đầu
├── data/
│   ├── raw/
│   │   ├── gas_turbine_data.csv  # Dữ liệu gốc có cột year
│   │   └── data.py               # Script tải từ UCI khi cần
│   └── processed/
│       ├── train_clean.csv       # Train, 9 feature đã chuẩn hóa
│       ├── test_clean.csv        # Test, dùng scaler của train
│       └── gas_turbine_clean.csv # Bản dữ liệu sạch tổng hợp có sẵn
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_model_boosting.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Chia theo năm, chặn CO, chuẩn hóa
│   ├── feature_eng.py            # Đặc trưng vật lý, VIF, CV theo năm
│   ├── models.py                 # Mô hình, huấn luyện, tìm siêu tham số
│   ├── evaluation.py             # Metric, so sánh thống kê
│   └── visualization.py          # Hàm vẽ biểu đồ
├── models/
│   ├── scaler.joblib
│   ├── preprocessing_meta.joblib
│   ├── xgboost_tuned_co.pkl
│   └── xgboost_tuned_nox.pkl
└── reports/
    ├── figures/                 # Biểu đồ phân tích và kết quả
    └── log/                     # tuning_log_co.csv, tuning_log_nox.csv
```

## 3. Cài đặt môi trường

Sử dụng **Python 3.12**. Các phiên bản trong `requirements.txt` lấy từ môi trường `.venv-vscode` hiện có của dự án. Đây là danh sách thư viện trực tiếp, không phải bản khóa toàn bộ phụ thuộc gián tiếp. Không cần GPU.

Mở terminal tại thư mục gốc của bài nộp (nơi chứa README). Trên Windows PowerShell:

```powershell
py -3.12 -m venv .venv-submit
.\.venv-submit\Scripts\python.exe -m pip install --upgrade pip
.\.venv-submit\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-submit\Scripts\python.exe -m pip check
.\.venv-submit\Scripts\python.exe -m jupyter lab
```

Trên Linux/macOS:

```bash
python3.12 -m venv .venv-submit
source .venv-submit/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m jupyter lab
```

Cần Internet khi cài thư viện. Sau đó các notebook dùng CSV đi kèm, không cần tải dữ liệu trực tuyến. Trong Jupyter, chọn kernel của môi trường vừa tạo.

## 4. Thứ tự chạy

Mở từng notebook trong `notebooks/`, chọn **Restart Kernel and Run All Cells**. Thư mục làm việc của kernel cần là `notebooks/`, đặc biệt với notebook 04 vì sử dụng đường dẫn `../data/...`.

| Thứ tự | Notebook | Nội dung và đầu ra |
| --- | --- | --- |
| 1 | `01_eda.ipynb` | Bản hiện tại đọc dữ liệu, loại trùng, chia theo năm, chặn CO, chuẩn hóa; tạo lại train/test CSV, scaler và metadata. |
| 2 | `02_feature_eng.ipynb` | Đọc CSV gốc theo đơn vị vật lý; tương quan/VIF, đặc trưng dẫn xuất, so sánh tập đặc trưng, drift; lưu hình `fe_*.png`. |
| 3 | `03_model_baseline.ipynb` | Đọc train/test đã xử lý; huấn luyện baseline cho CO, so sánh metric, phân tích phần dư, kiểm định thống kê, tuning Random Forest. |
| 4 | `04_model_boosting.ipynb` | Đọc train/test đã xử lý; huấn luyện/tuning XGBoost cho CO và NOX, đối chiếu Random Forest, SHAP; lưu mô hình, log và biểu đồ. |

Notebook 03 và 04 cần đầu ra notebook 01; notebook 02 tự đọc CSV gốc. Random Forest nhiều seed, tìm siêu tham số và SHAP có thể tốn nhiều thời gian tùy CPU. Chạy lại có thể ghi đè dữ liệu xử lý, mô hình, log và hình cùng tên.

Nếu chỉ cần tạo lại dữ liệu xử lý, chạy tại thư mục gốc bằng Python của môi trường đã cài:

```powershell
.\.venv-submit\Scripts\python.exe -c "from src.data_prep import prepare_data; prepare_data('data/raw/gas_turbine_data.csv', 'data/processed', 'models')"
```

Muốn tải lại dữ liệu UCI, cài thêm thư viện tải dữ liệu rồi chạy tại thư mục gốc:

```powershell
.\.venv-submit\Scripts\python.exe -m pip install ucimlrepo
.\.venv-submit\Scripts\python.exe data/raw/data.py
```

Script này ghi đè CSV gốc. CSV dùng cho pipeline phải có cột `year`, 9 đặc trưng ở mục 1 và hai cột `CO`, `NOX`.

## 5. Kết quả và lưu ý tái lập

- Có thể xem đầu ra đã lưu trong notebook và ảnh trong `reports/figures/` trước khi huấn luyện lại.
- `reports/log/tuning_log_co.csv` và `tuning_log_nox.csv` lưu tham số, điểm R² kiểm định chéo và thời gian fit của tuning XGBoost. Luồng này dùng 15 tổ hợp × 3 fold cho mỗi target với `KFold(shuffle=True, random_state=42)` trên train; khác CV theo năm ở notebook 02.
- Hai mô hình XGBoost đã lưu dùng **9 đặc trưng đã chuẩn hóa**, theo thứ tự `AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP`. Khi suy luận từ dữ liệu gốc, áp dụng `models/scaler.joblib` trước. Không chuẩn hóa lần nữa nếu dùng train/test CSV đã xử lý.
- Đặc trưng vật lý ở notebook 02 phải được tính trên dữ liệu **đơn vị gốc**, trước chuẩn hóa. Hai đặc trưng bổ sung giữ trong `FS_ENGINEERED`: `CDP_GTEP_ratio`, `AH_abs`.
- `gas_turbine_clean.csv` là tệp tổng hợp có sẵn; notebook 01 hiện chỉ tạo lại train/test CSV và hai tệp tiền xử lý trong `models/`.
- Notebook 03 còn đầu ra `KeyboardInterrupt` tại ô cuối tuning Random Forest. Cần chạy lại ô này đến khi hoàn tất để có log tuning mới.
- Bảng so sánh bổ sung ở notebook 03 dùng phép chia theo vị trí 80/20; tên cột `_random`/`_time` còn là nhãn cũ. Kết quả chính phía trước dùng cách chia theo năm ở mục 1.
- Hình `model_comparison_rmse.png` ở ô cuối notebook 04 dùng danh sách RMSE ghi trực tiếp trong mã. Sau khi chạy lại, cập nhật danh sách này theo kết quả thực tế trước khi đưa hình vào báo cáo.
- Notebook baseline lưu toàn bộ biểu đồ vào `reports/figures/`: `baseline_rmse_comparison.png`, `hinh_4_2.png`, `hinh_4_3.png`, `hinh_4_5.png`, `permutation_importance_rf.png`. Các tệp CSV như `bang_4_5.csv`, `cv_results_RandomForest.csv` vẫn được lưu tại thư mục làm việc của kernel.
- Các notebook từng chạy bằng nhiều phiên bản Python/thư viện. Kết quả chạy lại có thể chênh lệch. Nếu mô hình cũ không tương thích, chạy lại notebook 01 và 04 để tạo scaler và mô hình trong cùng môi trường.

## 6. Đóng gói bài nộp

Giữ nguyên cấu trúc và gửi kèm:

- `README.md`, `requirements.txt`, `src/`, đủ 4 notebook.
- `data/raw/gas_turbine_data.csv`, `data/processed/` để người chấm có dữ liệu chạy ngay.
- `models/`, `reports/figures/`, `reports/log/` để đối chiếu kết quả.
- Báo cáo tổng kết và slide theo yêu cầu môn học. **Thư mục hiện chưa có báo cáo tổng kết hoặc slide**; bổ sung trước khi nộp nếu được yêu cầu.

Không cần nén `.git/`, `.venv/`, `.venv-notebook/`, `.venv-vscode/`, `.venv-submit/`, `__pycache__/`, `.ipynb_checkpoints/`. `ke-hoach-btl-ml.md` là tài liệu kế hoạch, có thể kèm nếu cần.

Đặt tên ZIP theo quy định lớp; kế hoạch nhóm hiện ghi dạng `MSSV1-MSSV2-MSSV3-MSSV4.zip`. Giải nén thử sang thư mục khác để kiểm tra đủ dữ liệu, mã nguồn, mô hình và kết quả.
