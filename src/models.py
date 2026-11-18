import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, ParameterSampler, RandomizedSearchCV
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

SEED = 42


def build_model():
    """Tạo bộ mô hình baseline cho bài toán hồi quy CO."""
    return {
        "Naive (mean)": DummyRegressor(strategy="mean"),
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED,
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=500,
            random_state=SEED,
        ),
    }


def fit_eval(models, X_train, X_test, y_train, y_test):
    """Huấn luyện các mô hình và trả về bảng metric + dự đoán trên test."""
    rows = []
    preds = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        p_test = model.predict(X_test)
        p_train = model.predict(X_train)

        preds[name] = p_test
        rows.append(
            {
                "Model": name,
                "RMSE": float(np.sqrt(mean_squared_error(y_test, p_test))),
                "MAE": float(mean_absolute_error(y_test, p_test)),
                "R2": float(r2_score(y_test, p_test)),
                "R2_train": float(r2_score(y_train, p_train)),
                "Times": float(time.time() - t0),
            }
        )

    result = pd.DataFrame(rows).set_index("Model")
    return result, preds


def compute_vif(X):
    """Tính VIF cho từng biến dựa trên ma trận tương quan."""
    corr = np.corrcoef(X.values, rowvar=False)
    inv_corr = np.linalg.inv(corr)
    vif = np.diag(inv_corr)
    return pd.Series(vif, index=X.columns, name="VIF")


def permutation_importance_report(model, X_test, y_test, n_repeats=5):
    """Tính độ quan trọng biến theo phương pháp permutation importance."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values()


def compare_models(a, b, preds, y_test):
    """So sánh sai số tuyệt đối giữa 2 mô hình bằng Wilcoxon signed-rank."""
    ea = np.abs(y_test.to_numpy() - preds[a])
    eb = np.abs(y_test.to_numpy() - preds[b])
    p_value = wilcoxon(ea, eb).pvalue

    return {
        "model_a": a,
        "model_b": b,
        "mae_a": float(ea.mean()),
        "mae_b": float(eb.mean()),
        "p_value": float(p_value),
    }

def build_xgboost_model(tuned=False, target="CO"):
    """Tạo mô hình XGBoost (Baseline hoặc Tuned với tham số tối ưu)."""
    if not tuned:
        return xgb.XGBRegressor(random_state=SEED, n_jobs=-1)

    # Siêu tham số tối ưu tìm được từ RandomizedSearchCV
    if target == "CO":
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.01,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
        }
    else:  # target == "NOX"
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    return xgb.XGBRegressor(**params, random_state=SEED, n_jobs=-1)


def tune_xgboost(
    X_train, y_train, n_iter=15, cv=3, random_state=42, n_jobs=-1, clip_p99=False
):
    """Dò siêu tham số XGBoost; nếu clip_p99=True thì tính P99 riêng trên y_train của từng fold."""
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train).ravel()
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))

    records = []
    for params in param_list:
        fold_scores, fold_times = [], []
        for tr_idx, val_idx in kf.split(X_arr):
            X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            if clip_p99:
                p99_fold = np.quantile(y_tr, 0.99)
                y_tr_fit = np.clip(y_tr, None, p99_fold)
            else:
                y_tr_fit = y_tr

            t0 = time.perf_counter()
            model = xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs, **params)
            model.fit(X_tr, y_tr_fit)
            fold_times.append(time.perf_counter() - t0)

            preds = model.predict(X_val)
            fold_scores.append(r2_score(y_val, preds))  # Chấm trên y_val gốc

        records.append({
            "params": params,
            "mean_test_score": float(np.mean(fold_scores)),
            "std_test_score": float(np.std(fold_scores)),
            "mean_fit_time": float(np.mean(fold_times)),
        })

    cv_results_df = pd.DataFrame(records).sort_values("mean_test_score", ascending=False)
    best_params = cv_results_df.iloc[0]["params"]

    if clip_p99:
        p99_train = np.quantile(y_arr, 0.99)
        y_train_fit = np.clip(y_arr, None, p99_train)
    else:
        y_train_fit = y_arr

    best_model = xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs, **best_params)
    best_model.fit(X_arr, y_train_fit)

    return best_model, best_params, cv_results_df


__all__ = [
    "SEED",
    "build_model",
    "fit_eval",
    "compute_vif",
    "permutation_importance_report",
    "compare_models",
    "build_xgboost_model",
    "tune_xgboost",
]


