"""
Target Portfolio Analysis
Compares the rebalanced target portfolio against key benchmarks.

Proxy mapping summary:
  Core 60%  : IWDA→VTI/VEA/VWO (60/30/10), STOXX+Japan→VEA, EIMI→VWO,
               WSML→VB, IWQU+IWMO→VTI
  Thematic 30%: semis, AI, biotech, defense, uranium, clean energy, robotics → VTI
  Alt 10%   : PHAU→GLD, BTCE→DJP (Bitcoin vol is ~3x higher in practice)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from portfolio_lab import (
    MarketData, PortfolioOptimizer, RiskAnalyzer,
    PortfolioVisualizer, UNIVERSE
)

# Weights align to 11-asset UNIVERSE order:
# [VTI, VB, VEA, VWO, BND, TIP, TLT, HYG, GLD, VNQ, DJP]
TARGET_WEIGHTS = np.array([
    0.588,  # VTI  — US large cap (core + all thematic mapped here)
    0.060,  # VB   — US small cap (WSML small cap)
    0.154,  # VEA  — Int'l developed (STOXX, Japan, IWDA-Europe/Japan slice)
    0.098,  # VWO  — Emerging markets (EIMI + IWDA-EM slice)
    0.000,  # BND  — zero bonds
    0.000,  # TIP  — zero
    0.000,  # TLT  — zero
    0.000,  # HYG  — zero
    0.050,  # GLD  — gold (PHAU)
    0.000,  # VNQ  — zero REITs
    0.050,  # DJP  — commodities proxy for Bitcoin ETP (BTCE)
])

TARGET_COLOR = "#f97316"   # orange


def run_analysis():
    print("=" * 55)
    print("  TARGET PORTFOLIO ANALYSIS")
    print("=" * 55)

    # ── 1. Data ──────────────────────────────────────────────
    print("\n[1/5] Generating synthetic market data (Yahoo Finance not available)...")
    data = MarketData(UNIVERSE, lookback_years=10)
    data._generate_synthetic_data()  # Force synthetic so we don't need network

    # ── 2. Benchmark strategies ──────────────────────────────
    print("[2/5] Optimising benchmark strategies...")
    optimizer = PortfolioOptimizer(data)
    analyzer  = RiskAnalyzer(data)

    strategy_weights = {
        "60/40 Benchmark": optimizer.benchmark_60_40(),
        "Risk Parity":     optimizer.risk_parity(),
        "Max Sharpe":      optimizer.max_sharpe(),
        "Target Portfolio": TARGET_WEIGHTS,
    }

    results      = {}
    risk_contribs = {}
    for name, w in strategy_weights.items():
        results[name]       = analyzer.compute_metrics(name, w)
        risk_contribs[name] = analyzer.risk_contribution(w)

    # ── 3. Print metrics ─────────────────────────────────────
    print("[3/5] Portfolio metrics:\n")
    for m in results.values():
        print(m.summary())

    # ── 4. Scenario stress test ──────────────────────────────
    print("[4/5] Running scenario stress tests...")
    scenarios = {name: analyzer.scenario_analysis(w) for name, w in strategy_weights.items()}

    # ── 5. Monte Carlo ───────────────────────────────────────
    print("[5/5] Monte Carlo (5 000 paths, 10-year horizon)...")
    mc_results = {
        name: analyzer.monte_carlo(w, n_simulations=5000, horizon_years=10)
        for name, w in strategy_weights.items()
    }
    for name, mc in mc_results.items():
        print(f"  {name}: median ${mc['percentiles'][50]:.0f}  "
              f"| 5th ${mc['percentiles'][5]:.0f}  "
              f"| 95th ${mc['percentiles'][95]:.0f}")

    # ── Charts ───────────────────────────────────────────────
    print("\nGenerating charts...")

    # Patch COLORS so the visualiser knows about Target Portfolio
    PortfolioVisualizer.COLORS["Target Portfolio"] = TARGET_COLOR
    viz = PortfolioVisualizer()

    out_dir = os.path.dirname(__file__)

    # Chart 1 — Efficient frontier
    frontier = optimizer.efficient_frontier(n_points=40)
    viz.plot_efficient_frontier(
        frontier, results,
        save_path=os.path.join(out_dir, "target_efficient_frontier.png"),
    )

    # Chart 2 — Scenario comparison
    viz.plot_scenario_comparison(
        scenarios,
        save_path=os.path.join(out_dir, "target_scenarios.png"),
    )

    # Chart 3 — Monte Carlo fan chart
    viz.plot_monte_carlo(
        mc_results,
        save_path=os.path.join(out_dir, "target_monte_carlo.png"),
    )

    # Chart 4 — Allocation stacked bars
    viz.plot_allocation_comparison(
        results,
        save_path=os.path.join(out_dir, "target_allocations.png"),
    )

    # Chart 5 — Risk decomposition heatmap
    viz.plot_risk_decomposition(
        risk_contribs,
        save_path=os.path.join(out_dir, "target_risk_decomposition.png"),
    )

    print("\n✅ Done — 5 charts written to", out_dir)


if __name__ == "__main__":
    run_analysis()
