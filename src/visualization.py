from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


SCATTER_COLOR = "#3498db"
LINE_COLOR = "#e74c3c"


def configure_plot_theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _remove_top_right_spines(axes):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_actual_vs_predicted(
    y_test,
    y_pred,
    model_name="Random Forest",
    output_path=None,
):
    configure_plot_theme()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        y_test,
        y_pred,
        s=15,
        alpha=0.5,
        color=SCATTER_COLOR,
        edgecolor="none",
    )

    limits = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    ax.plot(
        limits,
        limits,
        color=LINE_COLOR,
        linestyle="--",
        linewidth=2,
        label="Dự đoán hoàn hảo (y=x)",
    )

    ax.set_title(
        f"So sánh CO thực tế và CO dự đoán ({model_name})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("CO thực tế (mg/m³)", fontsize=11)
    ax.set_ylabel("CO dự đoán (mg/m³)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_residual_diagnostics(
    y_test,
    y_pred,
    bins=60,
    output_path=None,
):
    configure_plot_theme()
    residuals = y_test.to_numpy() - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        residuals,
        bins=bins,
        color=SCATTER_COLOR,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    axes[0].axvline(0, color=LINE_COLOR, linestyle="--", linewidth=2)
    axes[0].set_title(
        "Phân phối của phần dư",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Phần dư (mg/m³)", fontsize=11)
    axes[0].set_ylabel("Tần suất", fontsize=11)

    axes[1].scatter(
        y_pred,
        residuals,
        s=15,
        alpha=0.5,
        color=SCATTER_COLOR,
        edgecolor="none",
    )
    axes[1].axhline(0, color=LINE_COLOR, linestyle="--", linewidth=2)
    axes[1].set_title(
        "Phần dư vs Giá trị dự đoán",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("CO dự đoán (mg/m³)", fontsize=11)
    axes[1].set_ylabel("Phần dư (mg/m³)", fontsize=11)

    _remove_top_right_spines(axes)
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_rmse_comparison(result, output_path=None):
    configure_plot_theme()
    summary = result.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        summary.index,
        summary["RMSE"],
        color="#50aae6",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_title("So sánh RMSE của các mô hình", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mô hình", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_linear_residual_diagnostics(y_test, y_pred, output_path=None):
    configure_plot_theme()
    residuals = y_test.to_numpy(dtype=float).ravel() - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_pred, residuals, s=4, alpha=0.3)
    axes[0].axhline(0, color="r", lw=1)
    axes[0].set_xlabel("CO dự đoán (mg/m³)")
    axes[0].set_ylabel("Phần dư (mg/m³)")
    axes[0].set_title("Phần dư trên tập test")

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot của phần dư")
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    return fig, axes


def plot_permutation_importance(importance, output_path=None):
    configure_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.sort_values().plot.barh(
        ax=ax,
        title="Permutation importance - Random Forest (CO)",
    )
    ax.set_xlabel("Mức giảm R² khi xáo trộn biến")
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


__all__ = [
    "configure_plot_theme",
    "plot_actual_vs_predicted",
    "plot_residual_diagnostics",
    "plot_rmse_comparison",
    "plot_linear_residual_diagnostics",
    "plot_permutation_importance",
]
