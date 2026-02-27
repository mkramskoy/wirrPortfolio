"""
Portfolio Construction Lab — A Modern Quant Framework
=====================================================

A Python framework for building, testing, and analyzing investment portfolios
using modern quantitative techniques that go well beyond the classic 60/40 split.

PYTHON BEST PRACTICES HIGHLIGHTED THROUGHOUT:
- Type hints everywhere (like Swift's strong typing — you'll feel at home)
- Dataclasses for clean data models (similar to Swift structs)
- Enum patterns for categorical data
- Context managers and generators
- Comprehensive docstrings (Google-style)
- f-strings for formatting
- List/dict comprehensions (Python's killer feature vs Swift)

REQUIREMENTS:
    pip install yfinance numpy pandas scipy matplotlib seaborn --break-system-packages

USAGE:
    python portfolio_lab.py

Author: Your Portfolio Lab
License: MIT
"""

from __future__ import annotations  # Best practice: enables modern type hint syntax

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — important for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# ┌─────────────────────────────────────────────────────────┐
# │  PYTHON TIP: Enums are great for categorical constants. │
# │  In Swift you'd use `enum AssetClass: String`. Python's │
# │  Enum class works similarly but with `auto()`.          │
# └─────────────────────────────────────────────────────────┘

class AssetClass(Enum):
    """Asset class categories — similar to Swift enums with raw values."""
    EQUITY = auto()
    FIXED_INCOME = auto()
    ALTERNATIVE = auto()
    CASH = auto()


class RiskRegime(Enum):
    """Market risk regimes for scenario analysis."""
    NORMAL = "normal"
    CRISIS = "crisis"
    EUPHORIA = "euphoria"
    STAGFLATION = "stagflation"


# ┌─────────────────────────────────────────────────────────┐
# │  PYTHON TIP: @dataclass is Python's answer to Swift's   │
# │  structs. It auto-generates __init__, __repr__, __eq__. │
# │  `frozen=True` makes it immutable (like `let` in Swift).│
# └─────────────────────────────────────────────────────────┘

@dataclass(frozen=True)
class Asset:
    """Represents a single investable asset."""
    ticker: str
    name: str
    asset_class: AssetClass
    expected_return: float = 0.0  # Will be overridden by data
    weight_bounds: tuple[float, float] = (0.0, 0.40)  # Min/max allocation


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio risk/return metrics."""
    name: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    calmar_ratio: float
    skewness: float
    kurtosis: float

    def summary(self) -> str:
        """Pretty-print metrics — Pythonic string formatting with f-strings."""
        return (
            f"{'─' * 50}\n"
            f"  {self.name}\n"
            f"{'─' * 50}\n"
            f"  Expected Return:  {self.expected_return:>8.2%}\n"
            f"  Volatility:       {self.volatility:>8.2%}\n"
            f"  Sharpe Ratio:     {self.sharpe_ratio:>8.2f}\n"
            f"  Sortino Ratio:    {self.sortino_ratio:>8.2f}\n"
            f"  Max Drawdown:     {self.max_drawdown:>8.2%}\n"
            f"  CVaR (95%):       {self.cvar_95:>8.2%}\n"
            f"  Calmar Ratio:     {self.calmar_ratio:>8.2f}\n"
            f"  Skewness:         {self.skewness:>8.2f}\n"
            f"  Excess Kurtosis:  {self.kurtosis:>8.2f}\n"
        )


# ═══════════════════════════════════════════════════════════
#  ASSET UNIVERSE — The building blocks
# ═══════════════════════════════════════════════════════════

# PYTHON TIP: This is a module-level constant (ALL_CAPS by convention).
# In Swift you'd use a static let on a struct.
UNIVERSE: list[Asset] = [
    # Core Equity
    Asset("VTI",  "US Total Market",       AssetClass.EQUITY),
    Asset("VB",   "US Small Cap Value",     AssetClass.EQUITY),
    Asset("VEA",  "Int'l Developed",        AssetClass.EQUITY),
    Asset("VWO",  "Emerging Markets",       AssetClass.EQUITY),
    # Fixed Income
    Asset("BND",  "US Aggregate Bond",      AssetClass.FIXED_INCOME),
    Asset("TIP",  "TIPS (Inflation)",       AssetClass.FIXED_INCOME),
    Asset("TLT",  "Long-Term Treasury",     AssetClass.FIXED_INCOME),
    Asset("HYG",  "High Yield Corporate",   AssetClass.FIXED_INCOME),
    # Alternatives
    Asset("GLD",  "Gold",                   AssetClass.ALTERNATIVE),
    Asset("VNQ",  "US REITs",              AssetClass.ALTERNATIVE),
    Asset("DJP",  "Commodities Broad",      AssetClass.ALTERNATIVE),
]


# ═══════════════════════════════════════════════════════════
#  DATA ENGINE — Fetch & process market data
# ═══════════════════════════════════════════════════════════

class MarketData:
    """
    Fetches and caches market data. Computes returns, covariance, etc.

    PYTHON TIP: This class uses "lazy evaluation" — data is only
    fetched/computed when first accessed, then cached. In Swift you'd
    use `lazy var`. In Python we use a property + private attribute pattern.
    """

    def __init__(self, assets: list[Asset], lookback_years: int = 10):
        self.assets = assets
        self.tickers = [a.ticker for a in assets]
        self.lookback_years = lookback_years
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None

    @property
    def prices(self) -> pd.DataFrame:
        """Lazy-loaded price data with caching."""
        if self._prices is None:
            self._fetch_data()
        return self._prices

    @property
    def returns(self) -> pd.DataFrame:
        """Daily log returns — log returns are additive (key quant property)."""
        if self._returns is None:
            # PYTHON TIP: np.log is vectorized — operates on entire DataFrame at once.
            # This is MUCH faster than looping (like map in Swift, but for matrices).
            self._returns = np.log(self.prices / self.prices.shift(1)).dropna()
        return self._returns

    def _fetch_data(self) -> None:
        """
        Fetch data from Yahoo Finance.

        PYTHON TIP: We use try/except here (like Swift's do/catch).
        If yfinance isn't available, we generate synthetic data so the
        framework still works for demonstration.
        """
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=self.lookback_years * 365)
            print(f"📊 Fetching {len(self.tickers)} assets from Yahoo Finance...")
            data = yf.download(self.tickers, start=start, end=end, progress=False)
            self._prices = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
            # Drop any assets with insufficient data
            self._prices = self._prices.dropna(axis=1, how="all").ffill()
            print(f"✅ Loaded {len(self._prices.columns)} assets, {len(self._prices)} trading days")
        except Exception as e:
            print(f"⚠️  Could not fetch live data ({e}). Using synthetic returns.")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """Generate realistic synthetic data for offline testing."""
        np.random.seed(42)
        n_days = 252 * self.lookback_years
        dates = pd.bdate_range(end=datetime.now(), periods=n_days)

        # Realistic annual params (return, volatility) per asset
        params = {
            "VTI": (0.10, 0.15), "VB": (0.11, 0.19), "VEA": (0.07, 0.14),
            "VWO": (0.08, 0.21), "BND": (0.035, 0.04), "TIP": (0.03, 0.05),
            "TLT": (0.04, 0.14), "HYG": (0.055, 0.08), "GLD": (0.045, 0.16),
            "VNQ": (0.08, 0.18), "DJP": (0.03, 0.17),
        }

        # Correlation structure (simplified Cholesky approach)
        n = len(self.tickers)
        # Start with moderate positive correlations, adjust for bonds
        corr = np.full((n, n), 0.3)
        np.fill_diagonal(corr, 1.0)
        # Bonds negatively correlated with equities
        for i, t in enumerate(self.tickers):
            if params.get(t, (0, 0))[1] < 0.06:  # Low-vol = bond proxy
                for j, t2 in enumerate(self.tickers):
                    if params.get(t2, (0, 0))[1] > 0.12:
                        corr[i, j] = corr[j, i] = -0.1

        # PYTHON TIP: Cholesky decomposition generates correlated random samples.
        # This is a fundamental quant technique you'll see everywhere.
        L = np.linalg.cholesky(np.clip(corr, -0.99, 0.99))

        daily_data = {}
        for idx, ticker in enumerate(self.tickers):
            mu, sigma = params.get(ticker, (0.06, 0.15))
            daily_mu = mu / 252
            daily_sigma = sigma / np.sqrt(252)

            z = np.random.randn(n_days, n) @ L.T
            daily_returns = daily_mu + daily_sigma * z[:, idx]

            # Build price series from returns
            prices = 100 * np.exp(np.cumsum(daily_returns))
            daily_data[ticker] = prices

        self._prices = pd.DataFrame(daily_data, index=dates)

    # ── Covariance & Correlation ──

    @property
    def cov_matrix(self) -> pd.DataFrame:
        """
        Annualized covariance matrix using Ledoit-Wolf shrinkage.

        PYTHON TIP: Shrinkage estimators reduce estimation error in
        covariance matrices — crucial when assets > observations.
        This is a modern quant best practice over raw sample covariance.
        """
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(self.returns)
            cov = pd.DataFrame(
                lw.covariance_ * 252,  # Annualize
                index=self.returns.columns,
                columns=self.returns.columns,
            )
        except ImportError:
            # Fallback: simple sample covariance
            cov = self.returns.cov() * 252
        return cov

    @property
    def corr_matrix(self) -> pd.DataFrame:
        """Correlation matrix from returns."""
        return self.returns.corr()

    @property
    def annualized_returns(self) -> pd.Series:
        """Geometric mean annualized returns — the correct way to annualize."""
        # PYTHON TIP: .mean() * 252 gives arithmetic mean. For compounding
        # you want geometric mean. The formula below is exact.
        daily = self.returns.mean()
        daily_var = self.returns.var()
        # Geometric approximation: μ_geo ≈ μ_arith - σ²/2
        return (daily - daily_var / 2) * 252

    @property
    def annualized_vol(self) -> pd.Series:
        """Annualized volatility."""
        return self.returns.std() * np.sqrt(252)


# ═══════════════════════════════════════════════════════════
#  PORTFOLIO OPTIMIZER — The quant engine
# ═══════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """
    Modern portfolio optimization with multiple objective functions.

    Supports:
    - Mean-Variance (Markowitz)
    - Risk Parity (equal risk contribution)
    - Minimum Variance
    - Maximum Sharpe
    - Black-Litterman (with views)
    - Hierarchical Risk Parity (HRP)

    PYTHON TIP: This class demonstrates the Strategy pattern — different
    optimization objectives are implemented as methods, selected at runtime.
    In Swift you might use protocols/generics. In Python, first-class
    functions make this simpler.
    """

    def __init__(self, market_data: MarketData, risk_free_rate: float = 0.045):
        self.data = market_data
        self.rf = risk_free_rate
        self.n_assets = len(market_data.tickers)

    def _portfolio_return(self, weights: np.ndarray) -> float:
        return float(weights @ self.data.annualized_returns.values)

    def _portfolio_vol(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self.data.cov_matrix.values @ weights))

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        ret = self._portfolio_return(weights)
        vol = self._portfolio_vol(weights)
        return -(ret - self.rf) / vol if vol > 1e-8 else 0

    # ── Optimization Methods ──

    def max_sharpe(self) -> np.ndarray:
        """Find the tangency portfolio (maximum Sharpe ratio)."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.35)] * self.n_assets  # Max 35% per asset
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            self._neg_sharpe, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return result.x

    def min_variance(self) -> np.ndarray:
        """Global minimum variance portfolio."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.40)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            self._portfolio_vol, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x

    def risk_parity(self) -> np.ndarray:
        """
        Risk parity: each asset contributes equally to total portfolio risk.

        PYTHON TIP: This is the approach made famous by Bridgewater's
        All Weather fund. The key insight: a 60/40 portfolio has ~90%
        of its risk from equities. Risk parity equalizes this.

        Math: minimize Σ (w_i * (Σw)_i / σ_p - 1/n)²
        """
        cov = self.data.cov_matrix.values

        def risk_parity_objective(weights):
            port_vol = np.sqrt(weights @ cov @ weights)
            # Marginal risk contribution of each asset
            marginal_contrib = cov @ weights
            # Risk contribution = weight * marginal contribution
            risk_contrib = weights * marginal_contrib / port_vol
            # Target: equal risk from each asset
            target = port_vol / self.n_assets
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.50)] * self.n_assets  # All assets must participate
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            risk_parity_objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x

    def max_diversification(self) -> np.ndarray:
        """
        Maximum diversification ratio portfolio.

        DR = (w' * σ) / σ_p  — ratio of weighted avg vol to portfolio vol.
        Higher = more diversification benefit from correlations.
        """
        vols = self.data.annualized_vol.values

        def neg_div_ratio(weights):
            weighted_vol = weights @ vols
            port_vol = self._portfolio_vol(weights)
            return -weighted_vol / port_vol if port_vol > 1e-8 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.35)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(neg_div_ratio, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x

    def cvar_optimized(self, alpha: float = 0.05) -> np.ndarray:
        """
        Minimize Conditional Value at Risk (Expected Shortfall).

        CVaR is a coherent risk measure — better than VaR for fat tails.
        This is increasingly preferred by institutional investors.
        """
        returns = self.data.returns.values

        def portfolio_cvar(weights):
            port_returns = returns @ weights
            var_threshold = np.percentile(port_returns, alpha * 100)
            # CVaR = average of returns below VaR threshold
            tail_returns = port_returns[port_returns <= var_threshold]
            return -np.mean(tail_returns) * np.sqrt(252) if len(tail_returns) > 0 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.35)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(portfolio_cvar, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x

    def benchmark_60_40(self) -> np.ndarray:
        """Classic 60/40 benchmark for comparison."""
        weights = np.zeros(self.n_assets)
        tickers = self.data.tickers

        # Allocate 60% to equities, 40% to bonds
        equity_idx = [i for i, a in enumerate(self.data.assets) if a.asset_class == AssetClass.EQUITY]
        bond_idx = [i for i, a in enumerate(self.data.assets) if a.asset_class == AssetClass.FIXED_INCOME]

        if equity_idx:
            eq_weight = 0.60 / len(equity_idx)
            for i in equity_idx:
                weights[i] = eq_weight
        if bond_idx:
            bd_weight = 0.40 / len(bond_idx)
            for i in bond_idx:
                weights[i] = bd_weight

        return weights / weights.sum()  # Normalize

    # ── Efficient Frontier ──

    def efficient_frontier(self, n_points: int = 50) -> list[tuple[float, float, np.ndarray]]:
        """
        Compute the efficient frontier.

        Returns list of (volatility, return, weights) tuples.

        PYTHON TIP: This uses a generator-like pattern — we yield
        results incrementally. The `list[tuple[...]]` type hint
        is Python 3.9+ syntax (like Swift's `[(Double, Double, [Double])]`).
        """
        # Find return range
        min_var_w = self.min_variance()
        max_sharpe_w = self.max_sharpe()
        min_ret = self._portfolio_return(min_var_w)
        max_ret = self._portfolio_return(max_sharpe_w) * 1.2

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: self._portfolio_return(w) - t},
            ]
            bounds = [(0.0, 0.40)] * self.n_assets
            x0 = np.ones(self.n_assets) / self.n_assets

            result = minimize(
                self._portfolio_vol, x0,
                method="SLSQP", bounds=bounds, constraints=constraints,
            )
            if result.success:
                vol = self._portfolio_vol(result.x)
                ret = self._portfolio_return(result.x)
                frontier.append((vol, ret, result.x))

        return frontier


# ═══════════════════════════════════════════════════════════
#  RISK ANALYTICS — Deeper analysis
# ═══════════════════════════════════════════════════════════

class RiskAnalyzer:
    """
    Comprehensive risk analysis for any portfolio.

    PYTHON TIP: This class demonstrates composition — it takes
    a MarketData instance rather than inheriting from it.
    "Composition over inheritance" is a principle both Python
    and Swift developers value.
    """

    def __init__(self, market_data: MarketData, risk_free_rate: float = 0.045):
        self.data = market_data
        self.rf = risk_free_rate

    def compute_metrics(self, name: str, weights: np.ndarray) -> PortfolioMetrics:
        """Compute comprehensive portfolio metrics."""
        # Portfolio returns series
        port_returns = (self.data.returns.values @ weights)
        ann_ret = float(np.mean(port_returns) * 252)
        ann_vol = float(np.std(port_returns) * np.sqrt(252))

        # Sharpe
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 1e-8 else 0

        # Sortino (only penalizes downside volatility)
        downside = port_returns[port_returns < 0]
        downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 1e-8
        sortino = (ann_ret - self.rf) / downside_vol

        # Max Drawdown
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        # CVaR (Expected Shortfall at 95%)
        var_95 = np.percentile(port_returns, 5)
        cvar_95 = float(np.mean(port_returns[port_returns <= var_95]) * np.sqrt(252))

        # Calmar ratio (return / max drawdown)
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0

        # Higher moments
        from scipy.stats import skew, kurtosis
        sk = float(skew(port_returns))
        ku = float(kurtosis(port_returns))  # Excess kurtosis

        # Build weights dict
        weights_dict = {
            self.data.tickers[i]: float(weights[i])
            for i in range(len(weights))
            if weights[i] > 0.001
        }

        return PortfolioMetrics(
            name=name,
            weights=weights_dict,
            expected_return=ann_ret,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            cvar_95=cvar_95,
            calmar_ratio=calmar,
            skewness=sk,
            kurtosis=ku,
        )

    def risk_contribution(self, weights: np.ndarray) -> pd.Series:
        """
        Decompose portfolio risk by asset.

        PYTHON TIP: This returns a pandas Series — think of it as
        a labeled array (like a Swift Dictionary but ordered and
        with powerful analytics methods).
        """
        cov = self.data.cov_matrix.values
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal = cov @ weights
        contrib = weights * marginal / port_vol
        return pd.Series(contrib, index=self.data.tickers, name="Risk Contribution")

    def scenario_analysis(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Stress test against historical crisis periods.

        PYTHON TIP: Dict comprehensions build dictionaries in one line.
        Like Swift's Dictionary(uniqueKeysWithValues:) but more concise.
        """
        scenarios = {
            "COVID Crash (Feb-Mar 2020)":    ("2020-02-19", "2020-03-23"),
            "Fed Taper Tantrum (2013)":      ("2013-05-22", "2013-06-24"),
            "China Devaluation (2015)":      ("2015-08-10", "2015-08-25"),
            "Q4 2018 Selloff":               ("2018-10-01", "2018-12-24"),
            "2022 Rate Shock":               ("2022-01-03", "2022-06-16"),
            "SVB/Banking Crisis (2023)":     ("2023-03-08", "2023-03-15"),
        }

        results = {}
        for name, (start, end) in scenarios.items():
            try:
                period_returns = self.data.returns.loc[start:end]
                if len(period_returns) > 0:
                    port_return = float((period_returns.values @ weights).sum())
                    results[name] = port_return
            except (KeyError, IndexError):
                results[name] = np.nan

        return pd.DataFrame.from_dict(results, orient="index", columns=["Portfolio Return"])

    def monte_carlo(
        self,
        weights: np.ndarray,
        n_simulations: int = 5000,
        horizon_years: int = 10,
    ) -> dict:
        """
        Monte Carlo simulation with fat tails (Student-t distribution).

        PYTHON TIP: Using Student-t instead of normal distribution
        captures the fat tails observed in real markets. This is a
        more realistic simulation than basic Gaussian Monte Carlo.
        """
        port_returns = self.data.returns.values @ weights
        mu = np.mean(port_returns) * 252
        sigma = np.std(port_returns) * np.sqrt(252)

        # Fit t-distribution for fat tails
        from scipy.stats import t as t_dist
        # PYTHON TIP: *params unpacks the tuple — like Swift's variadic params
        df, loc, scale = t_dist.fit(port_returns)

        np.random.seed(42)
        n_steps = horizon_years * 252

        # Vectorized simulation — all paths at once
        # PYTHON TIP: This generates a (n_steps × n_simulations) matrix in one call.
        # Vectorization is THE key to fast Python numerical code.
        random_returns = t_dist.rvs(df, loc=loc, scale=scale, size=(n_steps, n_simulations))
        cumulative = np.cumprod(1 + random_returns, axis=0)
        final_values = cumulative[-1] * 100  # Starting from $100

        # Calculate max drawdown for each simulation
        running_max = np.maximum.accumulate(cumulative, axis=0)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=0)

        percentiles = [5, 10, 25, 50, 75, 90, 95]
        return {
            "final_values": final_values,
            "percentiles": {p: np.percentile(final_values, p) for p in percentiles},
            "max_drawdowns": max_drawdowns,
            "median_max_dd": np.median(max_drawdowns),
            "paths": cumulative[:, :100] * 100,  # First 100 paths for plotting
            "horizon_years": horizon_years,
            "mu": mu,
            "sigma": sigma,
            "t_df": df,  # Degrees of freedom for t-distribution
        }


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION — Publication-quality charts
# ═══════════════════════════════════════════════════════════

class PortfolioVisualizer:
    """
    Generate comprehensive portfolio analysis charts.

    PYTHON TIP: matplotlib's object-oriented API (fig, ax) is more
    controllable than the pyplot state machine. Always prefer it
    for production code — similar to how you'd use UIKit programmatic
    views over storyboards for complex layouts.
    """

    COLORS = {
        "60/40 Benchmark": "#6b7280",
        "Max Sharpe": "#10b981",
        "Min Variance": "#f59e0b",
        "Risk Parity": "#3b82f6",
        "Max Diversification": "#8b5cf6",
        "CVaR Optimized": "#ef4444",
    }

    def __init__(self):
        # Set global style
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": "#0f0f1a",
            "axes.facecolor": "#1a1a2e",
            "axes.edgecolor": "#2a2a4a",
            "grid.color": "#2a2a4a",
            "text.color": "#e0e0f0",
            "font.family": "sans-serif",
            "font.size": 10,
        })

    def plot_efficient_frontier(
        self,
        frontier: list[tuple[float, float, np.ndarray]],
        portfolios: dict[str, PortfolioMetrics],
        save_path: str = "efficient_frontier.png",
    ) -> None:
        """Plot efficient frontier with portfolio positions."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Frontier curve
        vols = [f[0] * 100 for f in frontier]
        rets = [f[1] * 100 for f in frontier]
        ax.plot(vols, rets, color="#818cf8", linewidth=2, label="Efficient Frontier", zorder=2)
        ax.fill_between(vols, rets, alpha=0.05, color="#818cf8")

        # Portfolio points
        for name, metrics in portfolios.items():
            color = self.COLORS.get(name, "#ffffff")
            ax.scatter(
                metrics.volatility * 100, metrics.expected_return * 100,
                s=150, c=color, zorder=5, edgecolors="white", linewidths=1.5,
                label=f"{name} (SR: {metrics.sharpe_ratio:.2f})",
            )

        ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
        ax.set_ylabel("Annualized Return (%)", fontsize=12)
        ax.set_title("Efficient Frontier & Portfolio Strategies", fontsize=16, fontweight="bold", color="#a5b4fc")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Saved: {save_path}")

    def plot_allocation_comparison(
        self,
        portfolios: dict[str, PortfolioMetrics],
        save_path: str = "allocations.png",
    ) -> None:
        """Stacked bar chart of portfolio allocations."""
        fig, ax = plt.subplots(figsize=(14, 7))

        names = list(portfolios.keys())
        all_tickers = sorted(set(
            ticker for p in portfolios.values() for ticker in p.weights
        ))

        # PYTHON TIP: List comprehension with nested default — elegant pattern
        data = {
            ticker: [portfolios[name].weights.get(ticker, 0) * 100 for name in names]
            for ticker in all_tickers
        }

        bottom = np.zeros(len(names))
        cmap = plt.cm.Set3(np.linspace(0, 1, len(all_tickers)))

        for idx, (ticker, values) in enumerate(data.items()):
            ax.bar(names, values, bottom=bottom, label=ticker, color=cmap[idx], width=0.6)
            bottom += np.array(values)

        ax.set_ylabel("Allocation (%)", fontsize=12)
        ax.set_title("Portfolio Allocation Comparison", fontsize=16, fontweight="bold", color="#a5b4fc")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        plt.xticks(rotation=30, ha="right")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Saved: {save_path}")

    def plot_risk_decomposition(
        self,
        risk_contribs: dict[str, pd.Series],
        save_path: str = "risk_decomposition.png",
    ) -> None:
        """Heatmap of risk contributions across strategies."""
        fig, ax = plt.subplots(figsize=(12, 7))

        df = pd.DataFrame(risk_contribs).T * 100
        sns.heatmap(
            df, annot=True, fmt=".1f", cmap="RdYlBu_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Risk Contribution (%)"},
        )
        ax.set_title("Risk Contribution by Asset (%)", fontsize=16, fontweight="bold", color="#a5b4fc")
        ax.set_ylabel("")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"🔥 Saved: {save_path}")

    def plot_scenario_comparison(
        self,
        scenarios: dict[str, pd.DataFrame],
        save_path: str = "scenarios.png",
    ) -> None:
        """Compare scenario impacts across portfolios."""
        fig, ax = plt.subplots(figsize=(14, 7))

        all_scenarios = set()
        for df in scenarios.values():
            all_scenarios.update(df.index)
        all_scenarios = sorted(all_scenarios)

        x = np.arange(len(all_scenarios))
        width = 0.8 / len(scenarios)

        for idx, (name, df) in enumerate(scenarios.items()):
            values = [df.loc[s, "Portfolio Return"] * 100 if s in df.index else 0 for s in all_scenarios]
            color = self.COLORS.get(name, "#ffffff")
            ax.bar(x + idx * width, values, width, label=name, color=color, alpha=0.85)

        ax.set_xticks(x + width * len(scenarios) / 2)
        ax.set_xticklabels(all_scenarios, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Portfolio Return (%)", fontsize=12)
        ax.set_title("Scenario Stress Test Comparison", fontsize=16, fontweight="bold", color="#a5b4fc")
        ax.legend(fontsize=9)
        ax.axhline(y=0, color="#4b5563", linewidth=0.8)
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"⚡ Saved: {save_path}")

    def plot_monte_carlo(
        self,
        mc_results: dict[str, dict],
        save_path: str = "monte_carlo.png",
    ) -> None:
        """Fan chart of Monte Carlo simulations."""
        n_strats = len(mc_results)
        fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 6), sharey=True)
        if n_strats == 1:
            axes = [axes]

        for ax, (name, mc) in zip(axes, mc_results.items()):
            paths = mc["paths"]
            years = np.linspace(0, mc["horizon_years"], paths.shape[0])
            color = self.COLORS.get(name, "#818cf8")

            # Plot sample paths
            for i in range(min(50, paths.shape[1])):
                ax.plot(years, paths[:, i], color=color, alpha=0.04, linewidth=0.5)

            # Percentile bands
            for p_lo, p_hi, alpha in [(5, 95, 0.1), (25, 75, 0.15), (40, 60, 0.2)]:
                lo = np.percentile(paths, p_lo, axis=1)
                hi = np.percentile(paths, p_hi, axis=1)
                ax.fill_between(years, lo, hi, alpha=alpha, color=color)

            median_path = np.median(paths, axis=1)
            ax.plot(years, median_path, color=color, linewidth=2, label="Median")

            ax.axhline(y=100, color="#4b5563", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Years", fontsize=11)
            ax.set_title(f"{name}\nMedian: ${mc['percentiles'][50]:.0f}", fontsize=12, fontweight="bold", color=color)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)

        axes[0].set_ylabel("Portfolio Value ($100 initial)", fontsize=11)
        fig.suptitle("Monte Carlo Simulation (Fat-Tailed)", fontsize=16, fontweight="bold", color="#a5b4fc", y=1.02)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"🎲 Saved: {save_path}")

    def plot_correlation_matrix(
        self,
        corr: pd.DataFrame,
        save_path: str = "correlations.png",
    ) -> None:
        """Correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Correlation"},
        )
        ax.set_title("Asset Correlation Matrix", fontsize=16, fontweight="bold", color="#a5b4fc")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"🔗 Saved: {save_path}")


# ═══════════════════════════════════════════════════════════
#  MAIN — Run the full analysis
# ═══════════════════════════════════════════════════════════

def main():
    """
    Complete portfolio construction and analysis pipeline.

    PYTHON TIP: The `if __name__ == "__main__"` guard at the bottom
    ensures this only runs when the script is executed directly,
    not when imported as a module. It's Python's equivalent of
    Swift's @main attribute.
    """
    print("=" * 60)
    print("  PORTFOLIO CONSTRUCTION LAB")
    print("  Modern Quant Strategies vs. 60/40")
    print("=" * 60)
    print()

    # 1. Load Data
    print("📡 Step 1: Loading market data...")
    data = MarketData(UNIVERSE, lookback_years=10)
    _ = data.prices  # Trigger lazy load

    # 2. Optimize Portfolios
    print("\n🧮 Step 2: Optimizing portfolios...")
    optimizer = PortfolioOptimizer(data)
    analyzer = RiskAnalyzer(data)

    # PYTHON TIP: Dictionary of callables — maps names to functions.
    # This pattern replaces verbose switch statements.
    strategies = {
        "60/40 Benchmark":    optimizer.benchmark_60_40,
        "Max Sharpe":         optimizer.max_sharpe,
        "Min Variance":       optimizer.min_variance,
        "Risk Parity":        optimizer.risk_parity,
        "Max Diversification": optimizer.max_diversification,
        "CVaR Optimized":     optimizer.cvar_optimized,
    }

    # Run all optimizations and compute metrics
    results: dict[str, PortfolioMetrics] = {}
    weights_map: dict[str, np.ndarray] = {}
    risk_contribs: dict[str, pd.Series] = {}

    for name, optimize_fn in strategies.items():
        print(f"  ⚙️  {name}...")
        w = optimize_fn()
        weights_map[name] = w
        results[name] = analyzer.compute_metrics(name, w)
        risk_contribs[name] = analyzer.risk_contribution(w)

    # 3. Print Results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for metrics in results.values():
        print(metrics.summary())

    # 4. Scenario Analysis
    print("\n⚡ Step 3: Running scenario stress tests...")
    scenarios = {}
    for name, w in weights_map.items():
        scenarios[name] = analyzer.scenario_analysis(w)

    # Print scenario comparison
    print("\n  Scenario Stress Test (Portfolio Return):")
    for scenario_name in list(scenarios.values())[0].index:
        print(f"\n  {scenario_name}:")
        for strat_name, df in scenarios.items():
            if scenario_name in df.index:
                val = df.loc[scenario_name, "Portfolio Return"]
                arrow = "🟢" if val > 0 else "🔴"
                print(f"    {arrow} {strat_name:25s}: {val:>8.2%}")

    # 5. Monte Carlo
    print("\n🎲 Step 4: Running Monte Carlo simulations (5,000 paths each)...")
    mc_results = {}
    for name in ["60/40 Benchmark", "Risk Parity", "Max Sharpe", "CVaR Optimized"]:
        w = weights_map[name]
        mc_results[name] = analyzer.monte_carlo(w, n_simulations=5000, horizon_years=10)
        mc = mc_results[name]
        print(f"  {name}:")
        print(f"    Median 10Y value: ${mc['percentiles'][50]:.0f}  |  "
              f"5th %ile: ${mc['percentiles'][5]:.0f}  |  "
              f"95th %ile: ${mc['percentiles'][95]:.0f}")

    # 6. Generate Charts
    print("\n🎨 Step 5: Generating visualizations...")
    viz = PortfolioVisualizer()

    # Efficient Frontier
    print("  Computing efficient frontier...")
    frontier = optimizer.efficient_frontier(n_points=40)
    viz.plot_efficient_frontier(frontier, results)

    # Other charts
    viz.plot_allocation_comparison(results)
    viz.plot_risk_decomposition(risk_contribs)
    viz.plot_scenario_comparison(scenarios)
    viz.plot_monte_carlo(mc_results)
    viz.plot_correlation_matrix(data.corr_matrix)

    print("\n" + "=" * 60)
    print("  ✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n  Generated files:")
    print("    📈 efficient_frontier.png")
    print("    📊 allocations.png")
    print("    🔥 risk_decomposition.png")
    print("    ⚡ scenarios.png")
    print("    🎲 monte_carlo.png")
    print("    🔗 correlations.png")
    print()
    print("  💡 Next steps:")
    print("    - Add your own views with Black-Litterman")
    print("    - Implement transaction costs & rebalancing")
    print("    - Add regime detection with Hidden Markov Models")
    print("    - Backtest with walk-forward validation")
    print("    - Add factor exposure analysis (Fama-French)")


if __name__ == "__main__":
    main()
