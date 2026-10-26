from ucimlrepo import fetch_ucirepo
import pandas as pd
import os


# 1. Tải dataset từ UCI
dataset = fetch_ucirepo(id=551)

# 2. Lấy toàn bộ dữ liệu (features + targets) hoặc lấy theo dataframe
X = dataset.data.features
y = dataset.data.targets

# Gộp chung features và targets thành 1 DataFrame hoàn chỉnh
df = dataset.data.original

# 3. Tạo thư mục lưu trữ nếu chưa có và xuất ra file CSV
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'gas_turbine_data.csv')
df.to_csv(output_path, index=False)

print(f"Lưu dữ liệu thành công vào: {output_path}")
print(f"Kích thước dữ liệu: {df.shape}")