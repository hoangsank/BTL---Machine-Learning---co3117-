import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
TARGETS = ['CO_clipped', 'NOX']

ENGINEERED_FEATURES = ['CDP_GTEP_ratio', 'AH_abs']
CANDIDATE_FEATURES = ['TIT_TAT_diff', 'TEY_per_TIT', 'CDP_AP_ratio',
                      'CDP_GTEP_ratio', 'AH_abs', 'AT_x_AH']

FS_BASE = BASE_FEATURES
FS_COMPACT = ['AT', 'AP', 'AH', 'AFDP', 'TIT', 'TAT']
FS_ENGINEERED = BASE_FEATURES + ENGINEERED_FEATURES


def absolute_humidity(temp_c, rel_humidity_pct):
    """Độ ẩm tuyệt đối (g/m³) từ nhiệt độ (°C) và độ ẩm tương đối (%).

    Áp suất hơi bão hoà theo công thức Magnus (hPa), nhân độ ẩm tương đối ra áp suất
    hơi nước thực tế e, rồi đổi sang khối lượng hơi nước trên 1 m³ không khí:
    AH_abs = 216.7 * e / (T + 273.15).
    """
    sat_pressure = 6.112 * np.exp(17.62 * temp_c / (243.12 + temp_c))
    vapor_pressure = sat_pressure * rel_humidity_pct / 100.0
    return 216.7 * vapor_pressure / (temp_c + 273.15)


def add_physical_features(X):
    """Thêm các feature dẫn xuất từ 9 cột gốc. Trả về bản sao, không sửa X."""
    X = X.copy()
    X['TIT_TAT_diff'] = X['TIT'] - X['TAT']
    X['TEY_per_TIT'] = X['TEY'] / X['TIT']
    X['CDP_AP_ratio'] = X['CDP'] / X['AP']
    X['CDP_GTEP_ratio'] = X['CDP'] / X['GTEP']
    X['AH_abs'] = absolute_humidity(X['AT'], X['AH'])
    X['AT_x_AH'] = X['AT'] * X['AH']
    return X


def compute_vif(X):
    """VIF của từng cột: 1 / (1 - R2) khi hồi quy cột đó theo các cột còn lại."""
    vif = {}
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        vif[col] = 1.0 / max(1.0 - r2, 1e-9)
    return pd.Series(vif)


def prune_by_vif(X, threshold=10.0):
    """Loại lần lượt cột có VIF lớn nhất cho đến khi mọi VIF <= threshold."""
    X = X.copy()
    dropped = []
    while True:
        vif = compute_vif(X)
        if vif.max() <= threshold:
            return list(X.columns), dropped
        worst = vif.idxmax()
        dropped.append((worst, float(vif[worst])))
        X = X.drop(columns=worst)


def expanding_year_splits(df, years):
    """Chia theo năm mở rộng dần: train các năm trước, validate năm kế tiếp."""
    for i in range(1, len(years)):
        train_idx = np.flatnonzero(df['year'].isin(years[:i]).to_numpy())
        val_idx = np.flatnonzero((df['year'] == years[i]).to_numpy())
        yield train_idx, val_idx


def time_cv_rmse(model, df, cols, target, years):
    """RMSE trung bình qua các fold chia theo năm."""
    scores = []
    for train_idx, val_idx in expanding_year_splits(df, years):
        fitted = clone(model).fit(df.iloc[train_idx][cols], df.iloc[train_idx][target])
        pred = fitted.predict(df.iloc[val_idx][cols])
        scores.append(np.sqrt(mean_squared_error(df.iloc[val_idx][target], pred)))
    return float(np.mean(scores))


def group_permutation_importance(model, X, y, groups, n_repeats=5, seed=0):
    """Tăng RMSE khi xáo trộn đồng thời tất cả cột trong mỗi nhóm."""
    rng = np.random.default_rng(seed)
    base = np.sqrt(mean_squared_error(y, model.predict(X)))
    result = {}
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            perm = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[perm]
            increases.append(np.sqrt(mean_squared_error(y, model.predict(shuffled))) - base)
        result[name] = float(np.mean(increases))
    return pd.Series(result).sort_values(ascending=False)


def scale_features(X_train, *others):
    """Fit StandardScaler trên train, áp dụng cho các tập còn lại. Trả về (scaler, train, *others)."""
    scaler = StandardScaler().fit(X_train)
    scaled = [pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)]
    for X in others:
        scaled.append(pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns))
    return (scaler, *scaled)
