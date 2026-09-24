import pandas as pd
import numpy as np

def load_and_clean_data(raw_data_path):
    """
    Đọc dữ liệu gốc, xoá trùng lặp và xử lý Outlier cho biến CO.
    """
    df = pd.read_csv(raw_data_path)
    df = df.drop_duplicates()
    
    # Clipping CO ở p99
    p99_co = df['CO'].quantile(0.99)
    df['CO_clipped'] = df['CO'].clip(upper=p99_co)
    
    return df

def split_train_test(df):
    """
    Chia tập Train (2011-2013) và Test (2014-2015) theo quy chuẩn.
    """
    train_df = df[df['year'].isin([2011, 2012, 2013])]
    test_df = df[df['year'].isin([2014, 2015])]
    
    feature_cols = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
    target_cols = ['CO_clipped', 'NOX']
    
    return train_df[feature_cols], train_df[target_cols], test_df[feature_cols], test_df[target_cols]

def save_processed_data(df, output_csv_path):
    """
    Lưu dữ liệu đã làm sạch vào thư mục data/processed.
    """
    df.to_csv(output_csv_path, index=False)
    print(f"[THÀNH CÔNG] Đã lưu dữ liệu sạch vào: {output_csv_path}")