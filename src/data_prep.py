from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
TARGET_COLS = ['CO_clipped', 'NOX']
TRAIN_YEARS = [2011, 2012, 2013]
TEST_YEARS = [2014, 2015]


def _split_by_year(df):
    train_df = df[df['year'].isin(TRAIN_YEARS)].copy()
    test_df = df[df['year'].isin(TEST_YEARS)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError('Cần có train 2011–2013 và test 2014–2015.')
    return train_df, test_df


def load_and_clean_data(raw_data_path):
    df = pd.read_csv(raw_data_path).drop_duplicates().copy()
    train_df, _ = _split_by_year(df)
    co_p99_threshold = train_df['CO'].quantile(0.99)
    df['CO_clipped'] = df['CO'].clip(upper=co_p99_threshold)
    return df


def _prepare_frames(df):
    train_df, test_df = _split_by_year(df)
    co_p99_threshold = train_df['CO'].quantile(0.99)
    train_df['CO_clipped'] = train_df['CO'].clip(upper=co_p99_threshold)
    test_df['CO_clipped'] = test_df['CO'].clip(upper=co_p99_threshold)

    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    return train_df, test_df, scaler, {'co_p99_threshold': co_p99_threshold}


def split_train_test(df):
    train_df, test_df, _, _ = _prepare_frames(df)
    return (train_df[FEATURE_COLS], train_df[TARGET_COLS],
            test_df[FEATURE_COLS], test_df[TARGET_COLS])


def save_processed_data(df, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f'[THÀNH CÔNG] Đã lưu dữ liệu sạch vào: {output_csv_path}')


def prepare_data(raw_data_path, output_dir, models_dir):
    df = pd.read_csv(raw_data_path).drop_duplicates().copy()
    train_df, test_df, scaler, metadata = _prepare_frames(df)
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    save_processed_data(train_df, output_dir / 'train_clean.csv')
    save_processed_data(test_df, output_dir / 'test_clean.csv')
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / 'scaler.joblib')
    joblib.dump(metadata, models_dir / 'preprocessing_meta.joblib')
    return train_df, test_df, scaler, metadata
