"""
Correlation Regime Analysis
What happens when the stock-bond correlation breaks down (as it did in 2022).

Key insight: in normal times bonds hedge equities (negative correlation).
In inflationary regimes the Fed hikes aggressively, hammering both bonds
AND stocks simultaneously. The 60/40 loses its shock absorber.
The Target Portfolio owns no bonds, so it is structurally immune to this
specific correlation flip — though it still carries concentrated equity risk.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from portfolio_lab import MarketData, PortfolioOptimizer, UNIVERSE

# ── Portfolio weights (11-asset order: VTI VB VEA VWO BND TIP TLT HYG GLD VNQ DJP)
TARGET_WEIGHTS = np.array([0.588, 0.06, 0.154, 0.098, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05])
TARGET_COLOR   = "#f97316"
BENCH_COLOR    = "#6b7280"
RP_COLOR       = "#3b82f6"
MS_COLOR       = "#10b981"

TICKERS    = [a.ticker for a in UNIVERSE]
EQUITY_IDX = [0, 1, 2, 3]   # VTI VB VEA VWO
BOND_IDX   = [4, 5, 6, 7]   # BND TIP TLT HYG

# ── Actual 2022 full-year total returns (ETF-level, dividends reinvested)
ACTUAL_2022 = np.array([
    -0.195,  # VTI  — US total market
    -0.160,  # VB   — US small cap
    -0.145,  # VEA  — Int'l developed
    -0.175,  # VWO  — Emerging markets
    -0.130,  # BND  ← bonds fell WITH stocks
    -0.115,  # TIP  ← TIPS also lost
    -0.290,  # TLT  ← worst year for long bonds on record
    -0.110,  # HYG  — high yield
    -0.003,  # GLD  — gold nearly flat
    -0.260,  # VNQ  — REITs crushed by rate hikes
    +0.140,  # DJP  — commodities: one of the few winners
])


def build_crisis_cov(cov_annual: np.ndarray) -> np.ndarray:
    """
    Build a 2022-like covariance matrix.
    Flips bond-equity correlation from ~-0.1 (normal) to +0.4 (crisis/inflation).
    Everything else unchanged — this isolates the single structural shift.
    """
    vols = np.sqrt(np.diag(cov_annual))
    crisis = cov_annual.copy()
    for i in EQUITY_IDX:
        for j in BOND_IDX:
            crisis[i, j] = 0.40 * vols[i] * vols[j]
            crisis[j, i] = crisis[i, j]

    # Ensure positive-definite after modification
    min_eig = np.linalg.eigvalsh(crisis).min()
    if min_eig < 0:
        crisis += np.eye(len(crisis)) * (abs(min_eig) + 1e-8)
    return crisis


def simulate_paths(weights, mu_daily, cov_daily, n_sims=4000, horizon_years=10, seed=42):
    """Parametric Monte Carlo using multivariate normal daily returns."""
    np.random.seed(seed)
    n_assets = len(weights)
    n_steps  = horizon_years * 252
    raw = np.random.multivariate_normal(mu_daily, cov_daily, size=n_sims * n_steps)
    port_rets = raw.reshape(n_sims, n_steps, n_assets) @ weights  # (n_sims, n_steps)
    return np.cumprod(1 + port_rets, axis=1) * 100                # start at $100


def fan(ax, paths, color, label):
    years = np.linspace(0, 10, paths.shape[1])
    p5,  p25, p50, p75, p95 = [np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)]
    ax.fill_between(years, p5,  p95, alpha=0.07, color=color)
    ax.fill_between(years, p25, p75, alpha=0.15, color=color)
    ax.plot(years, p50, color=color, linewidth=2,
            label=f"{label}  median ${p50[-1]:.0f}  |  5th ${p5[-1]:.0f}")
    ax.axhline(100, color="#4b5563", linewidth=0.7, linestyle="--")


def apply_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0f0f1a", "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#2a2a4a",   "grid.color": "#2a2a4a",
        "text.color": "#e0e0f0",       "font.size": 10,
    })


def run():
    apply_style()
    out = os.path.dirname(__file__)

    print("=" * 55)
    print("  CORRELATION REGIME ANALYSIS")
    print("=" * 55)

    # ── Data & strategies ──────────────────────────────────────
    print("\nLoading market data & optimising benchmarks...")
    data = MarketData(UNIVERSE, lookback_years=10)
    data._generate_synthetic_data()
    opt  = PortfolioOptimizer(data)

    strategy_weights = {
        "60/40 Benchmark": opt.benchmark_60_40(),
        "Risk Parity":     opt.risk_parity(),
        "Max Sharpe":      opt.max_sharpe(),
        "Target Portfolio": TARGET_WEIGHTS,
    }
    strat_colors = {
        "60/40 Benchmark": BENCH_COLOR,
        "Risk Parity":     RP_COLOR,
        "Max Sharpe":      MS_COLOR,
        "Target Portfolio": TARGET_COLOR,
    }

    cov_normal = data.cov_matrix.values
    cov_crisis = build_crisis_cov(cov_normal)
    mu_daily   = data.annualized_returns.values / 252

    # ──────────────────────────────────────────────────────────
    # Chart 1 — 2022 actual asset returns
    # ──────────────────────────────────────────────────────────
    print("\n[1/4] 2022 actual asset returns...")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#ef4444" if r < 0 else "#10b981" for r in ACTUAL_2022]
    bars = ax.barh(TICKERS, ACTUAL_2022 * 100, color=colors, edgecolor="none", height=0.6)

    for i in BOND_IDX:
        ax.annotate(
            "bond — fell with stocks" if i == BOND_IDX[1] else "bond",
            xy=(ACTUAL_2022[i] * 100, i),
            xytext=(2, 0), textcoords="offset points",
            va="center", fontsize=8, color="#a5b4fc",
        )

    ax.axvline(0, color="#4b5563", linewidth=0.8)
    ax.set_xlabel("2022 Full-Year Total Return (%)")
    ax.set_title(
        "2022 — Bonds Fell WITH Stocks: The Correlation Regime Broke Down",
        fontsize=13, fontweight="bold", color="#a5b4fc",
    )
    for bar, v in zip(bars, ACTUAL_2022):
        ax.text(
            v * 100 + (0.4 if v >= 0 else -0.4),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1%}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8.5,
        )
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    p = os.path.join(out, "regime_2022_assets.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   Saved: {p}")

    # ──────────────────────────────────────────────────────────
    # Chart 2 — 2022 portfolio performance
    # ──────────────────────────────────────────────────────────
    print("[2/4] 2022 portfolio performance...")
    port_2022 = {n: float(w @ ACTUAL_2022) * 100 for n, w in strategy_weights.items()}
    for n, v in port_2022.items():
        print(f"   {n:22s}: {v:+.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(port_2022.keys())
    vals  = list(port_2022.values())
    cols  = [strat_colors[n] for n in names]
    bars  = ax.bar(names, vals, color=cols, width=0.5, edgecolor="none")
    ax.axhline(0, color="#4b5563", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v - 0.4, f"{v:.1f}%",
                ha="center", va="top", fontsize=12, fontweight="bold")
    ax.set_ylabel("2022 Full-Year Return (%)")
    ax.set_title(
        "2022 Portfolio Performance — Bonds Gave No Shelter",
        fontsize=13, fontweight="bold", color="#a5b4fc",
    )
    ax.grid(True, alpha=0.2, axis="y")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    p = os.path.join(out, "regime_2022_portfolios.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   Saved: {p}")

    # ──────────────────────────────────────────────────────────
    # Chart 3 — Correlation matrices: normal vs crisis
    # ──────────────────────────────────────────────────────────
    print("[3/4] Correlation regime matrices...")
    vols = np.sqrt(np.diag(cov_normal))
    corr_normal = cov_normal / np.outer(vols, vols)
    corr_crisis = cov_crisis / np.outer(vols, vols)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    kw = dict(annot=True, fmt=".2f", cmap="RdBu_r", center=0,
              vmin=-1, vmax=1, linewidths=0.5, cbar=False)
    sns.heatmap(pd.DataFrame(corr_normal, index=TICKERS, columns=TICKERS), ax=ax1, **kw)
    ax1.set_title("Normal Regime\nBonds hedge equities (negative correlation)",
                  fontsize=11, color="#10b981")
    sns.heatmap(pd.DataFrame(corr_crisis, index=TICKERS, columns=TICKERS), ax=ax2, **kw)
    ax2.set_title("2022-style Inflation Regime\nBonds move WITH equities (+0.40 correlation)",
                  fontsize=11, color="#ef4444")
    fig.suptitle("Correlation Regime Shift — The Hidden Risk Inside 60/40",
                 fontsize=14, fontweight="bold", color="#a5b4fc", y=1.02)
    fig.tight_layout()
    p = os.path.join(out, "regime_correlation_matrices.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   Saved: {p}")

    # ──────────────────────────────────────────────────────────
    # Chart 4 — Monte Carlo: normal vs crisis regime
    # The Target Portfolio barely changes between regimes because
    # it holds no bonds — there's nothing to re-correlate.
    # 60/40 deteriorates because its bond buffer evaporates.
    # ──────────────────────────────────────────────────────────
    print("[4/4] Monte Carlo under both regimes (this takes ~30s)...")
    regimes = [
        (cov_normal / 252, "Normal Regime",          "#10b981"),
        (cov_crisis / 252, "2022-style Inflation Regime", "#ef4444"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (cov_d, title, regime_color) in zip(axes, regimes):
        paths_bench  = simulate_paths(strategy_weights["60/40 Benchmark"], mu_daily, cov_d, seed=42)
        paths_target = simulate_paths(TARGET_WEIGHTS, mu_daily, cov_d, seed=43)

        fan(ax, paths_bench,  BENCH_COLOR,  "60/40 Benchmark")
        fan(ax, paths_target, TARGET_COLOR, "Target Portfolio")

        ax.set_xlabel("Years")
        ax.set_ylabel("Portfolio Value ($100 initial)")
        ax.set_title(title, fontsize=12, color=regime_color, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Monte Carlo (10-yr) — Target Portfolio is Regime-Agnostic on Bond Correlation\n"
        "60/40 median outcome falls when bonds stop hedging",
        fontsize=12, fontweight="bold", color="#a5b4fc",
    )
    fig.tight_layout()
    p = os.path.join(out, "regime_monte_carlo.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   Saved: {p}")

    print("\n✅ Done — 4 regime charts written to", out)


if __name__ == "__main__":
    run()
