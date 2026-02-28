"""
Analyze the actual portfolio from Portfolio.csv using the existing framework.
Maps European-listed holdings to the closest US ETF proxies for analysis.
"""
from portfolio_lab import (
    MarketData, PortfolioOptimizer, RiskAnalyzer, PortfolioVisualizer,
    UNIVERSE, AssetClass, Asset
)
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
#  ACTUAL PORTFOLIO FROM Portfolio.csv
# ═══════════════════════════════════════════════════════════

HOLDINGS = {
    "Cash (EUR)":                          78.70,
    "Alphabet (GOOGL)":                  1042.26,
    "Amundi STOXX Europe 600":           8555.40,
    "Fox Factory (FOXF)":                1151.72,
    "VanEck Semiconductor ETF":         51285.70,
    "WisdomTree Physical Gold":          4936.56,
    "Xtrackers MSCI Japan":             13297.37,
    "iShares $ Treasury 20+yr":          5239.21,
    "iShares Global Agg Bond":          31135.09,
    "iShares MSCI World":              115619.35,
    "iShares EUR Ultrashort Bond":     155693.72,
}

TOTAL = sum(HOLDINGS.values())

print("=" * 65)
print("  YOUR PORTFOLIO ANALYSIS")
print(f"  Total Value: EUR {TOTAL:,.2f}")
print("=" * 65)

# ── Breakdown ──
print("\n📋 Holdings Breakdown:")
print(f"  {'Holding':<35s} {'Value (EUR)':>12s} {'Weight':>8s}")
print("  " + "─" * 58)
for name, val in sorted(HOLDINGS.items(), key=lambda x: -x[1]):
    pct = val / TOTAL * 100
    print(f"  {name:<35s} {val:>12,.2f} {pct:>7.1f}%")

# ── Category Allocation ──
categories = {
    "Equity": [
        ("iShares MSCI World", 115619.35),
        ("VanEck Semiconductor ETF", 51285.70),
        ("Xtrackers MSCI Japan", 13297.37),
        ("Amundi STOXX Europe 600", 8555.40),
        ("Alphabet (GOOGL)", 1042.26),
        ("Fox Factory (FOXF)", 1151.72),
    ],
    "Fixed Income": [
        ("iShares EUR Ultrashort Bond", 155693.72),
        ("iShares Global Agg Bond", 31135.09),
        ("iShares $ Treasury 20+yr", 5239.21),
    ],
    "Alternatives": [
        ("WisdomTree Physical Gold", 4936.56),
    ],
    "Cash": [
        ("Cash (EUR)", 78.70),
    ],
}

print("\n📊 Category Breakdown:")
for cat, items in categories.items():
    total_cat = sum(v for _, v in items)
    pct = total_cat / TOTAL * 100
    print(f"  {cat:<18s}: EUR {total_cat:>12,.2f}  ({pct:>5.1f}%)")

# ── Key Concentration Risks ──
semi_pct = 51285.70 / TOTAL * 100
ultrashort_pct = 155693.72 / TOTAL * 100
equity_total = sum(v for _, v in categories["Equity"])
semi_of_equity = 51285.70 / equity_total * 100

print("\n⚠️  Concentration Analysis:")
print(f"  Semiconductor ETF:     {semi_pct:.1f}% of total portfolio")
print(f"  Semiconductor ETF:     {semi_of_equity:.1f}% of equity allocation")
print(f"  EUR Ultrashort Bond:   {ultrashort_pct:.1f}% of total portfolio (cash-equivalent)")
print(f"  Top 3 holdings:        {(115619.35 + 155693.72 + 51285.70) / TOTAL * 100:.1f}% of portfolio")

# ═══════════════════════════════════════════════════════════
#  MAP TO PROXY UNIVERSE & RUN FRAMEWORK ANALYSIS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FRAMEWORK ANALYSIS (proxy-mapped to US ETFs)")
print("=" * 65)

# Map to the 11-asset universe used by the framework
# MSCI World breakdown: ~60% US, ~25% Int'l Dev, ~5% EM, ~10% other
msci_world_val = 115619.35
us_from_world = msci_world_val * 0.60
intl_from_world = msci_world_val * 0.25
em_from_world = msci_world_val * 0.05

proxy_map = {
    "VTI": us_from_world + 51285.70 + 1042.26 + 1151.72,  # US equities + Semis + GOOGL + FOXF
    "VB":  0,
    "VEA": intl_from_world + 8555.40 + 13297.37,  # Int'l developed
    "VWO": em_from_world,  # Emerging
    "BND": 31135.09 + 155693.72,  # Bonds (ultrashort + global agg)
    "TIP": 0,
    "TLT": 5239.21,
    "HYG": 0,
    "GLD": 4936.56,
    "VNQ": 0,
    "DJP": 0,
}

proxy_total = sum(proxy_map.values())
tickers = [a.ticker for a in UNIVERSE]
my_weights = np.array([proxy_map.get(t, 0) / proxy_total for t in tickers])

print(f"\n  Proxy weights (mapped to {len(tickers)}-asset universe):")
for t, w in zip(tickers, my_weights):
    if w > 0.001:
        print(f"    {t:>5s}: {w*100:>6.1f}%")

# Load data and run analysis
print("\n📡 Loading market data...")
data = MarketData(UNIVERSE, lookback_years=10)
_ = data.prices

optimizer = PortfolioOptimizer(data)
analyzer = RiskAnalyzer(data)

# ── My Portfolio Metrics ──
my_metrics = analyzer.compute_metrics("My Portfolio", my_weights)
print(my_metrics.summary())

# ── Compare with optimized strategies ──
strategies = {
    "60/40 Benchmark":     optimizer.benchmark_60_40,
    "Max Sharpe":          optimizer.max_sharpe,
    "Min Variance":        optimizer.min_variance,
    "Risk Parity":         optimizer.risk_parity,
    "Max Diversification": optimizer.max_diversification,
    "CVaR Optimized":      optimizer.cvar_optimized,
}

print("\n" + "=" * 65)
print("  STRATEGY COMPARISON")
print("=" * 65)
print(f"\n  {'Strategy':<22s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'CVaR95':>8s}")
print("  " + "─" * 65)

# My portfolio first
print(f"  {'→ MY PORTFOLIO':<22s} {my_metrics.expected_return:>7.2%} {my_metrics.volatility:>7.2%} "
      f"{my_metrics.sharpe_ratio:>7.2f} {my_metrics.sortino_ratio:>7.2f} "
      f"{my_metrics.max_drawdown:>7.2%} {my_metrics.cvar_95:>7.2%}")

all_weights = {"My Portfolio": my_weights}
all_metrics = {"My Portfolio": my_metrics}

for name, fn in strategies.items():
    w = fn()
    m = analyzer.compute_metrics(name, w)
    all_weights[name] = w
    all_metrics[name] = m
    print(f"  {name:<22s} {m.expected_return:>7.2%} {m.volatility:>7.2%} "
          f"{m.sharpe_ratio:>7.2f} {m.sortino_ratio:>7.2f} "
          f"{m.max_drawdown:>7.2%} {m.cvar_95:>7.2%}")

# ── Risk Contribution ──
print("\n" + "=" * 65)
print("  RISK CONTRIBUTION DECOMPOSITION — MY PORTFOLIO")
print("=" * 65)
my_rc = analyzer.risk_contribution(my_weights)
for ticker, rc in my_rc.items():
    if abs(rc) > 0.001:
        bar = "█" * int(abs(rc) * 200)
        print(f"  {ticker:>5s}: {rc*100:>6.1f}% {bar}")

# ── Scenario Analysis ──
print("\n" + "=" * 65)
print("  SCENARIO STRESS TEST — MY PORTFOLIO")
print("=" * 65)
my_scenarios = analyzer.scenario_analysis(my_weights)
for scenario, row in my_scenarios.iterrows():
    val = row["Portfolio Return"]
    if not np.isnan(val):
        arrow = "🟢" if val > 0 else "🔴"
        print(f"  {arrow} {scenario:<35s}: {val:>8.2%}")

# ── Monte Carlo ──
print("\n" + "=" * 65)
print("  MONTE CARLO SIMULATION — MY PORTFOLIO (5,000 paths, 10Y)")
print("=" * 65)
mc = analyzer.monte_carlo(my_weights, n_simulations=5000, horizon_years=10)
print(f"  Starting value:     $100")
print(f"  Median 10Y value:   ${mc['percentiles'][50]:.0f}")
print(f"  5th percentile:     ${mc['percentiles'][5]:.0f}  (worst case)")
print(f"  95th percentile:    ${mc['percentiles'][95]:.0f}  (best case)")
print(f"  Median max DD:      {mc['median_max_dd']:.1%}")
print(f"  t-distribution df:  {mc['t_df']:.1f}  (lower = fatter tails)")

# ── Opportunity Cost Analysis ──
print("\n" + "=" * 65)
print("  OPPORTUNITY COST ANALYSIS")
print("=" * 65)

max_sharpe_w = all_weights["Max Sharpe"]
max_sharpe_m = all_metrics["Max Sharpe"]
risk_parity_w = all_weights["Risk Parity"]
risk_parity_m = all_metrics["Risk Parity"]

ret_gap_sharpe = max_sharpe_m.expected_return - my_metrics.expected_return
ret_gap_rp = risk_parity_m.expected_return - my_metrics.expected_return

portfolio_eur = TOTAL
years = 10

my_10y = portfolio_eur * (1 + my_metrics.expected_return) ** years
sharpe_10y = portfolio_eur * (1 + max_sharpe_m.expected_return) ** years
rp_10y = portfolio_eur * (1 + risk_parity_m.expected_return) ** years

print(f"  Your expected 10Y return:       {my_metrics.expected_return:.2%} p.a.")
print(f"  Max Sharpe expected:             {max_sharpe_m.expected_return:.2%} p.a.")
print(f"  Risk Parity expected:            {risk_parity_m.expected_return:.2%} p.a.")
print(f"")
print(f"  Your EUR {portfolio_eur:,.0f} in 10 years:")
print(f"    Current allocation:            EUR {my_10y:>12,.0f}")
print(f"    Max Sharpe allocation:         EUR {sharpe_10y:>12,.0f}  (+EUR {sharpe_10y - my_10y:>10,.0f})")
print(f"    Risk Parity allocation:        EUR {rp_10y:>12,.0f}  (+EUR {rp_10y - my_10y:>10,.0f})")

print("\n" + "=" * 65)
print("  ✅ ANALYSIS COMPLETE")
print("=" * 65)
