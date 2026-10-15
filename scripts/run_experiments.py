"""Run VaR/ES comparison, window sensitivity, and stress scenarios.

Toggle USE_REAL for yfinance (SPY/QQQ/TLT) vs synthetic Gaussian returns.
Outputs tables under reports/ and figures under reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_risk.historical import historical_es, historical_var
from market_risk.monte_carlo import monte_carlo_var_es
from market_risk.parametric import (
    estimate_mean_cov,
    parametric_es_gaussian,
    parametric_var_gaussian,
    portfolio_mean_std,
)
from market_risk.backtest import rolling_kupiec
from market_risk.data import load_returns_yfinance
from market_risk.portfolio import portfolio_losses
from market_risk.stress import apply_parallel_shock, scale_cov_vol

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

USE_REAL = True
TICKERS = ["SPY", "QQQ", "TLT"]
WEIGHTS = np.array([0.5, 0.3, 0.2])

# Dark theme matched to GitHub dark canvas (~#0d1117)
BG = "#0d1117"
AX_BG = "#161b22"
FG = "#e6edf3"
GRID = "#30363d"
COL_LOSS = "#8b949e"
COL_HIST = "#58a6ff"      # blue
COL_PARAM = "#3fb950"     # green
COL_MC = "#d2a8ff"        # soft purple
COL_EXC = "#ff7b72"       # coral red
COL_VAR95 = "#79c0ff"
COL_VAR99 = "#ffa657"     # amber
COL_STRESS = "#f85149"


def apply_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "axes.edgecolor": GRID,
            "xtick.color": FG,
            "ytick.color": FG,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "legend.facecolor": AX_BG,
            "legend.edgecolor": GRID,
            "legend.labelcolor": FG,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def save_dark_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


apply_dark_style()


def make_synthetic_returns(n_assets: int = 3, n_days: int = 500, seed: int = 42):
    """Small daily means; diagonal vols ~1%/day; mild correlation."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.0002, 0.0001, 0.00015][:n_assets])
    vols = np.array([0.01, 0.012, 0.015][:n_assets])
    corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    returns = rng.multivariate_normal(mu, cov, size=n_days)
    return returns, mu, cov


def build_results_table(returns, weights, alphas=(0.95, 0.99), n_sims=100_000, seed=0):
    losses = portfolio_losses(returns, weights)
    mu_hat, cov_hat = estimate_mean_cov(returns)
    mu_p, sigma_p = portfolio_mean_std(mu_hat, cov_hat, weights)

    rows = []
    for alpha in alphas:
        var_h = historical_var(losses, alpha)
        es_h = historical_es(losses, alpha)
        var_p = parametric_var_gaussian(mu_p, sigma_p, alpha)
        es_p = parametric_es_gaussian(mu_p, sigma_p, alpha)
        var_mc, es_mc = monte_carlo_var_es(
            mu_hat, cov_hat, weights, alpha=alpha, n_sims=n_sims, seed=seed
        )
        rows.extend(
            [
                {"alpha": alpha, "method": "historical", "VaR": var_h, "ES": es_h},
                {"alpha": alpha, "method": "parametric", "VaR": var_p, "ES": es_p},
                {"alpha": alpha, "method": "monte_carlo", "VaR": var_mc, "ES": es_mc},
            ]
        )
    return pd.DataFrame(rows), losses, mu_hat, cov_hat, mu_p, sigma_p


def plot_loss_hist_var(losses, var_h, var_p, var_mc, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(
        losses,
        bins=40,
        density=True,
        alpha=0.85,
        color=COL_HIST,
        edgecolor=BG,
        linewidth=0.4,
    )
    ax.axvline(var_h, color=COL_VAR99, lw=2.4, label=f"VaR_H={var_h:.4f}")
    ax.axvline(var_p, color=COL_PARAM, lw=2.4, ls="--", label=f"VaR_P={var_p:.4f}")
    ax.axvline(var_mc, color=COL_MC, lw=2.4, ls=":", label=f"VaR_MC={var_mc:.4f}")
    ax.set_xlabel("Portfolio loss L = -R_p")
    ax.set_ylabel("Density")
    ax.set_title("Loss histogram with VaR lines (alpha=0.95)")
    ax.grid(True, axis="y")
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    save_dark_fig(fig, path)


def plot_var_exceptions(
    losses: np.ndarray,
    dates,
    window: int,
    path: Path,
    alpha: float = 0.95,
):
    """
    Two panels:
      top  — full-sample historical VaR (definition: ~ (1-alpha) of points above)
      bottom — rolling VaR (honest backtest exceptions)
    """
    from market_risk.backtest import exception_indicator, rolling_historical_var

    var_full = historical_var(losses, alpha)
    hits_full = exception_indicator(losses, var_full)
    hit_full = float(np.mean(hits_full))

    var_roll = rolling_historical_var(losses, window=window, alpha=alpha)
    valid = ~np.isnan(var_roll)
    hits_roll = exception_indicator(losses[valid], var_roll[valid])
    hit_roll = float(np.mean(hits_roll)) if len(hits_roll) else float("nan")

    if dates is None:
        x = np.arange(len(losses))
        x_label = "Day index"
    else:
        x = np.asarray(dates)
        x_label = "Date"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    ax0.plot(x, losses, color=COL_LOSS, lw=0.9, label="Loss L_t")
    ax0.axhline(
        var_full, color=COL_VAR99, lw=2.0, label=f"Full-sample VaR 95%={var_full:.4f}"
    )
    ax0.scatter(
        x[hits_full == 1],
        losses[hits_full == 1],
        color=COL_EXC,
        s=18,
        zorder=3,
        label="Exception",
        edgecolors=BG,
        linewidths=0.3,
    )
    ax0.set_ylabel("Loss")
    ax0.set_title(
        f"Full-sample VaR (definition check): hit rate={hit_full:.1%} vs target {1 - alpha:.0%}"
    )
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax0.grid(True)

    ax1.plot(x, losses, color=COL_LOSS, lw=0.9, label="Loss L_t")
    ax1.plot(x, var_roll, color=COL_PARAM, lw=1.8, label=f"Rolling VaR (W={window})")
    if np.any(valid):
        xv = x[valid]
        lv = losses[valid]
        ax1.scatter(
            xv[hits_roll == 1],
            lv[hits_roll == 1],
            color=COL_EXC,
            s=18,
            zorder=3,
            label="Exception",
            edgecolors=BG,
            linewidths=0.3,
        )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss")
    ax1.set_title(
        f"Rolling VaR backtest: hit rate={hit_roll:.1%} vs target {1 - alpha:.0%}"
    )
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax1.grid(True)

    fig.tight_layout()
    save_dark_fig(fig, path)
    return {"hit_full": hit_full, "hit_roll": hit_roll, "var_full": var_full}


def plot_window_sensitivity(returns, weights, windows, path: Path, alpha: float = 0.95):
    var_h_list = []
    var_p_list = []
    for w_len in windows:
        R_w = returns[-w_len:]
        L_w = portfolio_losses(R_w, weights)
        var_h_list.append(historical_var(L_w, alpha))
        mu_w, cov_w = estimate_mean_cov(R_w)
        mu_p, sigma_p = portfolio_mean_std(mu_w, cov_w, weights)
        var_p_list.append(parametric_var_gaussian(mu_p, sigma_p, alpha))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        windows,
        var_h_list,
        "o-",
        color=COL_VAR99,
        lw=2.2,
        ms=8,
        label="Historical VaR",
    )
    ax.plot(
        windows,
        var_p_list,
        "s--",
        color=COL_PARAM,
        lw=2.2,
        ms=8,
        label="Parametric VaR",
    )
    ax.set_xlabel("Window length W (days)")
    ax.set_ylabel(f"VaR ({alpha:.0%})")
    ax.set_title("Sensitivity of VaR to historical window length")
    ax.legend(framealpha=0.95)
    ax.grid(True)
    fig.tight_layout()
    save_dark_fig(fig, path)


def plot_var_bars(df: pd.DataFrame, path: Path):
    """Grouped bars: methods on x, two alphas as series."""
    methods = ["historical", "parametric", "monte_carlo"]
    x = np.arange(len(methods))
    width = 0.35

    var_95 = [df[(df.alpha == 0.95) & (df.method == m)]["VaR"].iloc[0] for m in methods]
    var_99 = [df[(df.alpha == 0.99) & (df.method == m)]["VaR"].iloc[0] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        x - width / 2,
        var_95,
        width,
        label="alpha=0.95",
        color=COL_VAR95,
        edgecolor=BG,
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        var_99,
        width,
        label="alpha=0.99",
        color=COL_VAR99,
        edgecolor=BG,
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("VaR")
    ax.set_title("VaR by method and confidence level")
    ax.legend(framealpha=0.95)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_dark_fig(fig, path)


def run_stress_experiments(
    weights, mu_p, sigma_p, cov_hat, var_p_95, path: Path, alpha: float = 0.95, suffix: str = ""
):
    """
    Two scenarios:
      1) parallel -10% on all assets -> immediate portfolio loss
      2) vol x2 -> recompute parametric VaR with stressed covariance
    """
    shock = -0.10
    parallel_loss = apply_parallel_shock(weights, shock=shock)

    cov_stressed = scale_cov_vol(cov_hat, vol_mult=2.0)
    sigma_p_stressed = float(np.sqrt(weights @ cov_stressed @ weights))
    var_p_vol2 = parametric_var_gaussian(mu_p, sigma_p_stressed, alpha)

    stress_df = pd.DataFrame(
        [
            {"scenario": "baseline_parametric_VaR_95", "value": var_p_95},
            {"scenario": "parallel_shock_loss_-10pct", "value": parallel_loss},
            {"scenario": "parametric_VaR_95_vol_x2", "value": var_p_vol2},
        ]
    )

    print("\n=== Stress / what-if (VaR is not worst case) ===\n")
    print(stress_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        f"\nParallel -10% loss ({parallel_loss:.4f}) vs baseline VaR 95% ({var_p_95:.4f}): "
        f"stress/VaR = {parallel_loss / var_p_95:.1f}x"
    )
    print(
        f"Vol x2 parametric VaR ({var_p_vol2:.4f}) vs baseline ({var_p_95:.4f}): "
        f"ratio = {var_p_vol2 / var_p_95:.2f}x"
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["Baseline VaR 95%", "Parallel -10% loss", "VaR 95% (vol x2)"]
    values = [var_p_95, parallel_loss, var_p_vol2]
    colors = [COL_HIST, COL_STRESS, COL_VAR99]
    bars = ax.bar(labels, values, color=colors, edgecolor=BG, linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=9,
        )
    ax.set_ylabel("Loss / VaR")
    ax.set_title("Stress scenarios vs baseline parametric VaR")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_dark_fig(fig, path)

    csv_path = ROOT / "reports" / f"stress_table{suffix}.csv"
    stress_df.to_csv(csv_path, index=False)
    print(f"Saved stress table -> {csv_path}")
    print(f"Saved figure -> {path}")
    print(
        "\nTakeaway: VaR describes a quantile under a model/history; "
        "stress asks 'what if X happens?' and can exceed VaR by a lot."
    )
    return stress_df


def main():
    weights = WEIGHTS
    suffix = "_real" if USE_REAL else "_synth"
    dates = None

    if USE_REAL:
        returns, dates, tickers = load_returns_yfinance(TICKERS, start="2021-01-01")
        print(
            f"Real data: {tickers}, T={len(returns)}, "
            f"{dates[0].date()} -> {dates[-1].date()}"
        )
    else:
        returns, _, _ = make_synthetic_returns(n_assets=3, n_days=500, seed=42)
        print(f"Synthetic data: T={len(returns)}, N={returns.shape[1]}")

    df, losses, mu_hat, cov_hat, mu_p, sigma_p = build_results_table(
        returns, weights, alphas=(0.95, 0.99), n_sims=100_000, seed=0
    )

    # A) table
    print("\n=== VaR / ES comparison (same portfolio, same data) ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    csv_path = ROOT / "reports" / f"var_es_table{suffix}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table -> {csv_path}")

    # values for alpha=0.95 lines on histogram
    row = lambda method: df[(df.alpha == 0.95) & (df.method == method)].iloc[0]
    var_h = row("historical")["VaR"]
    var_p = row("parametric")["VaR"]
    var_mc = row("monte_carlo")["VaR"]

    # B) hist
    p1 = FIG_DIR / f"loss_hist_var{suffix}.png"
    plot_loss_hist_var(losses, var_h, var_p, var_mc, p1)
    print(f"Saved figure -> {p1}")

    # C) window sensitivity (cap windows by available T)
    t_len = len(returns)
    windows = [w for w in (60, 120, 250, 500) if w <= t_len]
    p2 = FIG_DIR / f"window_sensitivity{suffix}.png"
    plot_window_sensitivity(returns, weights, windows, p2, alpha=0.95)
    print(f"Saved figure -> {p2}")

    # D) bar chart
    p3 = FIG_DIR / f"var_bars{suffix}.png"
    plot_var_bars(df, p3)
    print(f"Saved figure -> {p3}")

    print(
        "\nTakeaway: H, P and MC disagree on the same sample - "
        "different assumptions (empirical past vs Gaussian model vs finite MC noise). "
        "That difference is an educational feature, not a bug."
    )

    # Rolling Kupiec backtest (historical VaR, no look-ahead)
    window = 250 if len(losses) > 250 else max(60, len(losses) // 3)

    # Exceptions over time: definition check vs rolling backtest
    p_exc = FIG_DIR / f"var_exceptions{suffix}.png"
    exc_stats = plot_var_exceptions(losses, dates, window=window, path=p_exc, alpha=0.95)
    print(f"Saved figure -> {p_exc}")
    print(
        f"Exception rates: full-sample={exc_stats['hit_full']:.2%}, "
        f"rolling={exc_stats['hit_roll']:.2%} (target 5%)"
    )

    kupiec = rolling_kupiec(losses, window=window, alpha=0.95)
    print("\n=== Rolling Kupiec POF (historical VaR, alpha=0.95) ===\n")
    print(
        f"window={window}, n={kupiec['n']}, hits={kupiec['x']}, "
        f"hit_ratio={kupiec['hit_ratio']:.4f}, expected_p={kupiec['expected_p']:.4f}"
    )
    print(
        f"LR={kupiec['lr_stat']:.4f}, p_value={kupiec['p_value']:.4f}, "
        f"reject_5pct={kupiec['reject_5pct']}"
    )

    # E) stress / what-if
    p4 = FIG_DIR / f"stress_vs_var{suffix}.png"
    run_stress_experiments(
        weights=weights,
        mu_p=mu_p,
        sigma_p=sigma_p,
        cov_hat=cov_hat,
        var_p_95=var_p,
        path=p4,
        alpha=0.95,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()
