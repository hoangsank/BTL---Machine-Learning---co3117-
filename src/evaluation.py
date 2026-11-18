import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred, model_name="Mô hình"):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"=== KẾT QUẢ ĐÁNH GIÁ: {model_name} ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    print("=" * 35)
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def compare_model_table(pairs, preds, y_test, n_boot=1000, alpha=0.05, seed=42, save_path=None):
    """Compare absolute errors between model pairs using bootstrap and Wilcoxon tests."""
    y = np.asarray(y_test, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.integers(0, n, size=(n_boot, n))

    rows = []
    for model_a, model_b in pairs:
        error_a = np.abs(y - preds[model_a])
        error_b = np.abs(y - preds[model_b])
        difference = error_a - error_b

        try:
            p_value = wilcoxon(error_a, error_b).pvalue
        except ValueError:
            p_value = 1.0

        bootstrap = difference[indices].mean(axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5])

        rows.append({
            "Cặp": f"{model_a} vs {model_b}",
            "MAE A": error_a.mean(),
            "MAE B": error_b.mean(),
            "ΔMAE (A−B)": difference.mean(),
            "Δ%": 100 * difference.mean() / error_b.mean(),
            "KTC 95% thấp": low,
            "KTC 95% cao": high,
            "p Wilcoxon": p_value,
            "A tốt hơn B (% mẫu)": 100 * (difference < 0).mean(),
        })

    table = pd.DataFrame(rows)
    table["p hiệu chỉnh"] = np.minimum(1.0, table["p Wilcoxon"] * len(pairs))

    def verdict(row):
        if row["p hiệu chỉnh"] >= alpha or (
            row["KTC 95% thấp"] <= 0 <= row["KTC 95% cao"]
        ):
            return "không phân biệt được"
        model_a, model_b = row["Cặp"].split(" vs ")
        return f"{model_a} tốt hơn" if row["ΔMAE (A−B)"] < 0 else f"{model_b} tốt hơn"

    table["Kết luận"] = table.apply(verdict, axis=1)

    if save_path:
        table.to_csv(save_path, index=False)

    return table


def bootstrap_coefficient_report(models, X_train, y_train, n_bootstraps=200, seed=42):
    """Estimate coefficient stability for standardized Linear and Ridge models."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    y = np.asarray(y_train, dtype=float).ravel()

    alpha = float(models["Ridge"].alpha_)
    linear_full = LinearRegression().fit(X_scaled, y)
    ridge_full = Ridge(alpha=alpha).fit(X_scaled, y)

    rng = np.random.default_rng(seed)
    linear_coefficients = []
    ridge_coefficients = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(X_scaled), len(X_scaled))
        X_boot = X_scaled.iloc[indices]
        y_boot = y[indices]
        linear_coefficients.append(
            LinearRegression().fit(X_boot, y_boot).coef_
        )
        ridge_coefficients.append(
            Ridge(alpha=alpha).fit(X_boot, y_boot).coef_
        )

    linear_coefficients = np.asarray(linear_coefficients)
    ridge_coefficients = np.asarray(ridge_coefficients)

    report = pd.DataFrame({
        "Linear": linear_full.coef_,
        "Ridge": ridge_full.coef_,
        "SD boot Linear": linear_coefficients.std(axis=0),
        "SD boot Ridge": ridge_coefficients.std(axis=0),
        "% đổi dấu Linear": (
            100
            * (np.sign(linear_coefficients) != np.sign(linear_full.coef_)).mean(axis=0)
        ),
    }, index=X_train.columns)

    return report, alpha


__all__ = [
    "evaluate_regression",
    "compare_model_table",
    "bootstrap_coefficient_report",
]