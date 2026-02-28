# Portfolio Construction Lab — Project Summary

## What This Is

A two-part quantitative portfolio analysis system that goes beyond the traditional 60/40 stock/bond allocation. It combines an interactive exploration dashboard (React) with a production-grade optimization engine (Python), designed to help construct, stress-test, and deeply analyze modern investment portfolios.

---

## Components

### 1. Interactive Portfolio Lab (`portfolio-lab.jsx`)

A React dashboard with 6 analytical views, plus dynamic "My Portfolio" loaded from CSV:

| Tab | Purpose |
|-----|---------|
| **Overview** | Sharpe ratio leaderboard across 8 strategies + full asset universe (12 assets across equity, bonds, alternatives) |
| **Strategies** | Side-by-side allocation breakdowns with stacked weight bars, risk/return stats, and strategy descriptions |
| **Scenarios** | Stress testing against 7 market scenarios (2008 GFC, stagflation, rate shock, tech crash, bull run, inflation spike) with heatmap comparison |
| **Monte Carlo** | 2,000-path simulation with configurable horizons (5–30 years), percentile distributions, max drawdown estimates, and sparkline path previews |
| **Efficient Frontier** | 4,000 random portfolios plotted on risk-return space with named strategies overlaid — uses Dirichlet-like sampling for realistic spread |
| **Deep Dive** | Single-strategy focus: category allocation pie, risk contribution decomposition, full scenario bar chart |

**8 Strategies Implemented:**
- 60/40 Benchmark (classic)
- Risk Parity (Bridgewater-inspired equal risk contribution)
- All Weather (balanced across economic regimes)
- Max Sharpe (mean-variance optimized)
- Min Variance (lowest total volatility)
- Factor Tilt (value, momentum, quality)
- Trend + Carry (crisis alpha seeker)
- My Portfolio (dynamically loaded from Portfolio.csv via ISIN→proxy mapping)

**12-Asset Universe:**
- Equity: VTI, VB, VEA, VWO
- Fixed Income: BND, VTIP, TLT, HYG
- Alternatives: GLD, VNQ, DJP, DBMF

**Technical Details:**
- Seed-able PRNG (Mulberry32) for reproducible Monte Carlo
- Box-Muller transform for Gaussian random generation
- Full correlation matrix (12×12) with realistic cross-asset relationships
- Parametric scenario shocks with correlation shift modeling
- Dark theme UI with recharts visualizations

### 2. Python Optimization Framework (`portfolio_lab.py`)

A 5-stage quantitative pipeline:

**Stage 1 — Data Ingestion (`MarketData`)**
- Fetches 10 years of daily prices from Yahoo Finance via `yfinance`
- Falls back to synthetic data generation (Cholesky-correlated) when offline
- Computes log returns, Ledoit-Wolf shrinkage covariance, correlation matrix
- Lazy-loaded properties with caching

**Stage 2 — Portfolio Optimization (`PortfolioOptimizer`)**
- **Max Sharpe**: Tangency portfolio via negative Sharpe minimization
- **Min Variance**: Global minimum volatility
- **Risk Parity**: Equal risk contribution (minimize squared RC deviations)
- **Max Diversification**: Maximize diversification ratio (weighted vol / portfolio vol)
- **CVaR Optimized**: Minimize Conditional Value at Risk (Expected Shortfall at 95%)
- **60/40 Benchmark**: Equal-weighted within equity and bond buckets
- Efficient frontier computation (40-point sweep with return targeting)
- All use scipy SLSQP with weight bounds (0–35/40% per asset)

**Stage 3 — Risk Analytics (`RiskAnalyzer`)**
- Full metric suite: Sharpe, Sortino, max drawdown, CVaR, Calmar ratio, skewness, excess kurtosis
- Risk contribution decomposition: RC_i = w_i × (Σw)_i / σ_p
- Per-asset marginal risk contribution

**Stage 4 — Scenario Stress Testing**
- Historical crisis windows: COVID crash, Fed taper tantrum, China devaluation, Q4 2018, 2022 rate shock, SVB banking crisis
- Actual historical return data (not parametric)

**Stage 5 — Fat-Tailed Monte Carlo**
- Student-t distribution fitted to historical returns (captures real-world fat tails)
- 5,000 simulation paths with max drawdown tracking per path
- Percentile distribution: 5th, 10th, 25th, 50th, 75th, 90th, 95th

**Stage 6 — Visualization (`PortfolioVisualizer`)**
- 6 publication-quality charts: efficient frontier, allocation comparison, risk decomposition heatmap, scenario comparison, Monte Carlo fan charts, correlation matrix
- Dark theme matplotlib with seaborn heatmaps

**Python Learning Annotations:**
The code is annotated with `PYTHON TIP` comments mapping Python patterns to Swift equivalents — dataclasses vs structs, lazy properties, vectorized operations, dict comprehensions, callable dictionaries replacing switch statements, Enum patterns, and composition over inheritance.

---

## Intended Workflow

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  React Lab          │     │  Python Framework     │     │  Iterate            │
│  (explore/intuit)   │ ──► │  (validate w/ real    │ ──► │  (adjust weights,   │
│                     │     │   market data)        │     │   add constraints)  │
│  • Compare strategies│     │  • Historical returns │     │  • Sector limits    │
│  • Scenario stress  │     │  • Fat-tail MC        │     │  • ESG screens      │
│  • Build intuition  │     │  • True risk decomp   │     │  • Turnover budgets │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

---

## Key Quantitative Concepts Used

| Concept | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Risk Parity** | Equalizes risk contribution across assets | 60/40 has ~90% equity risk despite 60% equity dollars |
| **Ledoit-Wolf Shrinkage** | Regularizes covariance estimation | Raw sample covariance is noisy with many assets |
| **CVaR / Expected Shortfall** | Average loss in worst 5% of outcomes | Coherent risk measure, better than VaR for fat tails |
| **Student-t Monte Carlo** | Simulates with heavier tails than normal | Markets crash harder/more often than Gaussian predicts |
| **Diversification Ratio** | Measures correlation benefit captured | Higher = more free lunch from diversification |
| **Efficient Frontier** | Upper-left boundary of risk-return space | Any portfolio below it takes unnecessary risk |
| **Scenario Stress Testing** | Replays historical crises | Reveals hidden correlations that spike during crashes |
| **Sharpe / Sortino / Calmar** | Risk-adjusted return metrics | Sharpe uses total vol, Sortino only downside, Calmar uses max drawdown |

---

## Bug Fixes Applied

**Efficient Frontier Display (React):**
- Added `type="number"` to both ScatterChart axes — without this, recharts treated numeric data as categorical, collapsing all 4,000 points into a single vertical line
- Replaced uniform random weight generation with Dirichlet-like sampling (varying concentration parameter via `-log(uniform)` raised to a power) to produce realistic portfolio spread from concentrated to diversified
- Standardized data keys to `x`/`y` across frontier points and strategy markers for consistent axis mapping

---

## Dependencies

**React Dashboard:**
- React 18+ with hooks
- recharts (charting library)
- Fetches Portfolio.csv at runtime (DEGIRO export, European number format)
- Maps holdings by ISIN to 12-asset proxy universe (MSCI World splits 60/30/10 across VTI/VEA/VWO)
- Falls back gracefully with error banner when CSV is missing

**Python Framework:**
```
pip install yfinance numpy pandas scipy matplotlib seaborn scikit-learn
```

---

## Future Extensions

- **Black-Litterman model**: Incorporate personal market views into optimization
- **Regime detection**: Hidden Markov Models to switch strategies based on market regime
- **Factor exposure analysis**: Fama-French 5-factor decomposition
- **Transaction costs & rebalancing**: Realistic turnover modeling
- **Walk-forward backtesting**: Out-of-sample validation with rolling windows
- **Hierarchical Risk Parity (HRP)**: Tree-based clustering for more stable allocations

---

## Quick Reference

### Commands
- `python3` — use instead of `python` (system has no `python` alias)
- `cd dashboard && npm run dev` — start Vite dev server on port 5173
- `python3 portfolio_lab.py` — run full quant pipeline (generates PNGs)
- `python3 analyze_my_portfolio.py` — analyze Portfolio.csv against framework

### Project Structure
- `portfolio-lab.jsx` — **single source of truth** for the React dashboard
- `dashboard/` — Vite scaffold only; `App.jsx` imports `../../portfolio-lab.jsx`
- `dashboard/public/Portfolio.csv` — symlink to `../../Portfolio.csv`
- Do NOT create copies of `portfolio-lab.jsx` inside `dashboard/src/`

### Data Handling
- `Portfolio.csv` contains personal financial data — NEVER commit
- CSV uses European format: commas as decimal separators, quoted fields
- `.gitignore` + `pre-commit` hook block CSV check-in
- PNGs are generated output, safe to delete and regenerate

### Key Gotchas
- Recharts tooltips need explicit `labelStyle`/`itemStyle` for dark mode readability
- Recharts ScatterChart axes require `type="number"` or data renders as categorical
- EUR ultrashort bond (0.5% vol) proxied to BND (4.5% vol) overstates risk ~9x
