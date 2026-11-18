import pandas as pd
import numpy as np
from pathlib import Path

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


def load_model_dataset(project_dir=None, feature_cols=None, target_col="CO_clipped"):
    """Load the year-split train/test data used for model experiments."""
    base_dir = Path(project_dir) if project_dir is not None else Path.cwd()
    if base_dir.name == "notebooks":
        base_dir = base_dir.parent

    processed_dir = base_dir / "data" / "processed"
    train_path = processed_dir / "train_clean.csv"
    test_path = processed_dir / "test_clean.csv"

    missing_paths = [path for path in (train_path, test_path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Không tìm thấy train/test đã chia theo năm: "
            + ", ".join(str(path) for path in missing_paths)
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if feature_cols is None:
        feature_cols = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    df = pd.concat([train_df, test_df], ignore_index=True).sort_values("year").reset_index(drop=True)
    X = df[feature_cols]
    y = df[target_col]

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "train_path": train_path,
        "test_path": test_path,
    }


__all__ = [
    "load_and_clean_data",
    "split_train_test",
    "save_processed_data",
    "load_model_dataset",
]