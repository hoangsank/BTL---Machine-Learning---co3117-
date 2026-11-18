import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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


def tune_and_save_log(
    model,
    param_space,
    X_train,
    y_train,
    model_name,
    search_type="grid",
    cv=5,
    output_dir=None,
):
    """Tune a model and save the cross-validation log to CSV."""
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=20,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=SEED,
            verbose=1,
        )

    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_)

    output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"cv_results_{model_name}.csv"
    cv_results.to_csv(csv_path, index=False)

    return search.best_estimator_, csv_path


__all__ = [
    "SEED",
    "build_model",
    "fit_eval",
    "compute_vif",
    "permutation_importance_report",
    "compare_models",
    "tune_and_save_log",
]
