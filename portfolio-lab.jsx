import { useState, useMemo, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ScatterChart, Scatter, Cell, PieChart, Pie } from "recharts";

// ─── Seed-able PRNG for reproducible Monte Carlo ───
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ─── Asset Universe ───
const ASSETS = {
  "US Large Cap (VTI)": { ret: 0.098, vol: 0.155, category: "equity" },
  "US Small Cap (VB)": { ret: 0.108, vol: 0.195, category: "equity" },
  "Int'l Developed (VEA)": { ret: 0.072, vol: 0.145, category: "equity" },
  "Emerging Markets (VWO)": { ret: 0.085, vol: 0.21, category: "equity" },
  "US Agg Bond (BND)": { ret: 0.035, vol: 0.045, category: "bond" },
  "TIPS (VTIP)": { ret: 0.032, vol: 0.05, category: "bond" },
  "Long Treasury (TLT)": { ret: 0.038, vol: 0.14, category: "bond" },
  "High Yield (HYG)": { ret: 0.055, vol: 0.085, category: "bond" },
  "Gold (GLD)": { ret: 0.045, vol: 0.16, category: "alt" },
  "REITs (VNQ)": { ret: 0.078, vol: 0.185, category: "alt" },
  "Commodities (DJP)": { ret: 0.035, vol: 0.17, category: "alt" },
  "Managed Futures (DBMF)": { ret: 0.06, vol: 0.12, category: "alt" },
};

const ASSET_NAMES = Object.keys(ASSETS);

// Correlation matrix (simplified, realistic approximations)
const CORR_BASE = [
  [1.0, 0.88, 0.78, 0.72, -0.05, 0.05, -0.25, 0.55, 0.05, 0.65, 0.35, -0.15],
  [0.88, 1.0, 0.72, 0.68, -0.08, 0.02, -0.28, 0.58, 0.02, 0.7, 0.38, -0.12],
  [0.78, 0.72, 1.0, 0.82, 0.02, 0.08, -0.15, 0.5, 0.15, 0.55, 0.4, -0.1],
  [0.72, 0.68, 0.82, 1.0, 0.05, 0.1, -0.1, 0.52, 0.18, 0.5, 0.45, -0.08],
  [-0.05, -0.08, 0.02, 0.05, 1.0, 0.75, 0.85, 0.35, 0.15, 0.15, 0.05, 0.0],
  [0.05, 0.02, 0.08, 0.1, 0.75, 1.0, 0.55, 0.25, 0.3, 0.12, 0.35, 0.05],
  [-0.25, -0.28, -0.15, -0.1, 0.85, 0.55, 1.0, 0.1, 0.2, -0.05, -0.05, 0.1],
  [0.55, 0.58, 0.5, 0.52, 0.35, 0.25, 0.1, 1.0, 0.1, 0.45, 0.3, -0.05],
  [0.05, 0.02, 0.15, 0.18, 0.15, 0.3, 0.2, 0.1, 1.0, 0.1, 0.4, 0.15],
  [0.65, 0.7, 0.55, 0.5, 0.15, 0.12, -0.05, 0.45, 0.1, 1.0, 0.3, -0.1],
  [0.35, 0.38, 0.4, 0.45, 0.05, 0.35, -0.05, 0.3, 0.4, 0.3, 1.0, 0.2],
  [-0.15, -0.12, -0.1, -0.08, 0.0, 0.05, 0.1, -0.05, 0.15, -0.1, 0.2, 1.0],
];

// ─── Scenario Shocks ───
const SCENARIOS = {
  "Base Case": { label: "Base Case", shocks: Array(12).fill(0), corrShift: 0 },
  "2008 GFC": {
    label: "2008-style Crisis",
    shocks: [-0.38, -0.45, -0.42, -0.52, 0.05, 0.02, 0.2, -0.25, 0.25, -0.38, -0.35, 0.12],
    corrShift: 0.3,
  },
  "Rate Shock": {
    label: "Rates +300bp",
    shocks: [-0.12, -0.15, -0.08, -0.1, -0.1, -0.02, -0.28, -0.08, 0.05, -0.18, 0.05, 0.08],
    corrShift: 0.15,
  },
  "Stagflation": {
    label: "Stagflation",
    shocks: [-0.2, -0.25, -0.18, -0.15, -0.08, 0.08, -0.15, -0.12, 0.2, -0.15, 0.25, 0.15],
    corrShift: 0.2,
  },
  "Tech Crash": {
    label: "Tech Bubble Pop",
    shocks: [-0.32, -0.22, -0.15, -0.2, 0.08, 0.04, 0.15, -0.08, 0.12, -0.1, 0.02, 0.1],
    corrShift: 0.2,
  },
  "Bull Run": {
    label: "Bull Market",
    shocks: [0.25, 0.3, 0.18, 0.22, 0.02, 0.01, -0.05, 0.1, -0.02, 0.2, 0.08, -0.05],
    corrShift: -0.1,
  },
  "Inflation Spike": {
    label: "Inflation >6%",
    shocks: [-0.08, -0.05, -0.02, 0.02, -0.12, 0.06, -0.2, -0.05, 0.18, -0.05, 0.3, 0.12],
    corrShift: 0.15,
  },
};

// ─── Portfolio Strategies ───
function buildPortfolios() {
  return {
    "60/40 Benchmark": {
      weights: [0.36, 0.06, 0.12, 0.06, 0.30, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
      color: "#6b7280",
      description: "Classic 60% equity / 40% bond split",
    },
    "Risk Parity": {
      weights: [0.08, 0.04, 0.06, 0.04, 0.22, 0.12, 0.15, 0.05, 0.08, 0.04, 0.06, 0.06],
      color: "#3b82f6",
      description: "Equal risk contribution across asset classes (Bridgewater-inspired)",
    },
    "All Weather": {
      weights: [0.18, 0.03, 0.06, 0.03, 0.12, 0.1, 0.18, 0.0, 0.08, 0.04, 0.1, 0.08],
      color: "#8b5cf6",
      description: "Balanced across economic regimes: growth, recession, inflation, deflation",
    },
    "Max Sharpe": {
      weights: [0.32, 0.08, 0.1, 0.05, 0.15, 0.05, 0.0, 0.05, 0.05, 0.08, 0.02, 0.05],
      color: "#10b981",
      description: "Mean-variance optimized for highest risk-adjusted return",
    },
    "Min Variance": {
      weights: [0.05, 0.02, 0.05, 0.02, 0.35, 0.2, 0.08, 0.03, 0.05, 0.02, 0.03, 0.1],
      color: "#f59e0b",
      description: "Minimizes total portfolio volatility",
    },
    "Factor Tilt": {
      weights: [0.2, 0.12, 0.08, 0.08, 0.1, 0.05, 0.0, 0.07, 0.05, 0.1, 0.05, 0.1],
      color: "#ef4444",
      description: "Tilts toward value, momentum, and quality factors",
    },
    "Trend + Carry": {
      weights: [0.15, 0.05, 0.08, 0.05, 0.1, 0.08, 0.05, 0.04, 0.08, 0.05, 0.1, 0.17],
      color: "#ec4899",
      description: "Trend-following with carry overlay — crisis alpha seeker",
    },
  };
}

// ─── Math Helpers ───
function portfolioStats(weights, assets, corrMatrix) {
  const n = weights.length;
  let ret = 0;
  for (let i = 0; i < n; i++) ret += weights[i] * assets[i].ret;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      variance += weights[i] * weights[j] * assets[i].vol * assets[j].vol * corrMatrix[i][j];
    }
  }
  const vol = Math.sqrt(Math.max(variance, 0.0001));
  const sharpe = (ret - 0.045) / vol; // risk-free ~4.5% currently
  return { ret, vol, sharpe };
}

function scenarioReturn(weights, scenario) {
  let r = 0;
  for (let i = 0; i < weights.length; i++) r += weights[i] * scenario.shocks[i];
  return r;
}

function runMonteCarlo(weights, assetData, corrMatrix, years = 10, sims = 2000) {
  const stats = portfolioStats(weights, assetData, corrMatrix);
  const mu = stats.ret;
  const sigma = stats.vol;
  const rng = mulberry32(42);

  const paths = [];
  const finalValues = [];
  const drawdowns = [];

  for (let s = 0; s < sims; s++) {
    let value = 100;
    let peak = 100;
    let maxDD = 0;
    const path = [100];

    for (let y = 0; y < years; y++) {
      const shock = gaussianRandom(rng);
      const yearReturn = mu + sigma * shock;
      value *= 1 + yearReturn;
      peak = Math.max(peak, value);
      const dd = (peak - value) / peak;
      maxDD = Math.max(maxDD, dd);
      path.push(value);
    }
    if (s < 50) paths.push(path);
    finalValues.push(value);
    drawdowns.push(maxDD);
  }

  finalValues.sort((a, b) => a - b);
  drawdowns.sort((a, b) => a - b);

  return {
    median: finalValues[Math.floor(sims * 0.5)],
    p5: finalValues[Math.floor(sims * 0.05)],
    p25: finalValues[Math.floor(sims * 0.25)],
    p75: finalValues[Math.floor(sims * 0.75)],
    p95: finalValues[Math.floor(sims * 0.95)],
    medianDD: drawdowns[Math.floor(sims * 0.5)],
    p95DD: drawdowns[Math.floor(sims * 0.95)],
    paths,
    years,
  };
}

// ─── UI Components ───
const TABS = ["Overview", "Strategies", "Scenarios", "Monte Carlo", "Efficient Frontier", "Deep Dive"];

const categoryColors = { equity: "#3b82f6", bond: "#10b981", alt: "#f59e0b" };

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{ background: "#1a1a2e", borderRadius: 12, padding: "16px 20px", minWidth: 140, border: "1px solid #2a2a4a" }}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || "#e0e0ff", fontVariantNumeric: "tabular-nums" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function TabButton({ active, label, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "8px 18px",
        borderRadius: 8,
        border: active ? "1px solid #6366f1" : "1px solid transparent",
        background: active ? "#2a2a4e" : "transparent",
        color: active ? "#c4b5fd" : "#6b7280",
        fontSize: 13,
        fontWeight: active ? 600 : 400,
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

function StrategyToggle({ name, color, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "6px 14px", borderRadius: 8,
        border: active ? `1px solid ${color}` : "1px solid #2a2a4a",
        background: active ? `${color}15` : "#1a1a2e",
        color: active ? color : "#6b7280",
        fontSize: 12, fontWeight: active ? 600 : 400,
        cursor: "pointer", transition: "all 0.15s",
      }}
    >
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: active ? color : "#444" }} />
      {name}
    </button>
  );
}

// ─── Main App ───
export default function PortfolioLab() {
  const [tab, setTab] = useState("Overview");
  const [activeStrategies, setActiveStrategies] = useState(["60/40 Benchmark", "Risk Parity", "All Weather"]);
  const [selectedScenario, setSelectedScenario] = useState("Base Case");
  const [mcYears, setMcYears] = useState(10);
  const [deepDiveStrategy, setDeepDiveStrategy] = useState("Risk Parity");

  const portfolios = useMemo(() => buildPortfolios(), []);
  const assetData = ASSET_NAMES.map((n) => ASSETS[n]);

  const toggleStrategy = useCallback((name) => {
    setActiveStrategies((prev) =>
      prev.includes(name) ? prev.filter((s) => s !== name) : [...prev, name]
    );
  }, []);

  // ── Compute all strategy stats ──
  const allStats = useMemo(() => {
    const out = {};
    for (const [name, p] of Object.entries(portfolios)) {
      const base = portfolioStats(p.weights, assetData, CORR_BASE);
      const scenarioResults = {};
      for (const [sKey, sc] of Object.entries(SCENARIOS)) {
        scenarioResults[sKey] = scenarioReturn(p.weights, sc);
      }
      out[name] = { ...base, scenarios: scenarioResults, weights: p.weights, color: p.color, description: p.description };
    }
    return out;
  }, [portfolios, assetData]);

  // ── Monte Carlo for active strategies ──
  const mcResults = useMemo(() => {
    const out = {};
    for (const name of activeStrategies) {
      out[name] = runMonteCarlo(portfolios[name].weights, assetData, CORR_BASE, mcYears, 2000);
    }
    return out;
  }, [activeStrategies, mcYears, portfolios, assetData]);

  // ── Efficient Frontier ──
  const frontier = useMemo(() => {
    const rng = mulberry32(99);
    const points = [];
    // Generate diverse portfolios using Dirichlet-like sampling
    // TECHNIQUE: Use -log(uniform) to get exponential draws, then normalize.
    // Varying the concentration parameter creates portfolios from very
    // concentrated (high vol) to very diversified (low vol).
    for (let i = 0; i < 4000; i++) {
      // Vary concentration: low = concentrated, high = diversified
      const concentration = 0.1 + rng() * 3.0;
      const raw = ASSET_NAMES.map(() => {
        const u = Math.max(rng(), 1e-10);
        return Math.pow(-Math.log(u), concentration);
      });
      const sum = raw.reduce((a, b) => a + b, 0);
      const normalized = raw.map((v) => v / sum);
      const s = portfolioStats(normalized, assetData, CORR_BASE);
      if (s.vol > 0.001) {
        points.push({ x: +(s.vol * 100).toFixed(2), y: +(s.ret * 100).toFixed(2) });
      }
    }
    return points;
  }, [assetData]);

  const strategyPoints = useMemo(() => {
    return Object.entries(allStats).map(([name, s]) => ({
      name,
      x: +(s.vol * 100).toFixed(2),
      y: +(s.ret * 100).toFixed(2),
      sharpe: +s.sharpe.toFixed(3),
      color: s.color,
    }));
  }, [allStats]);

  // ─── RENDER ───
  return (
    <div style={{ background: "#0f0f1a", color: "#e0e0f0", minHeight: "100vh", fontFamily: "'Inter', -apple-system, sans-serif", padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 28, fontWeight: 800, margin: 0, background: "linear-gradient(135deg, #818cf8, #c084fc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Portfolio Construction Lab
        </h1>
        <p style={{ color: "#6b7280", fontSize: 14, margin: "6px 0 0" }}>Quant-driven portfolio analysis — beyond 60/40</p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 24, overflowX: "auto", paddingBottom: 4 }}>
        {TABS.map((t) => (
          <TabButton key={t} active={tab === t} label={t} onClick={() => setTab(t)} />
        ))}
      </div>

      {/* Strategy Toggles (shown on relevant tabs) */}
      {["Strategies", "Scenarios", "Monte Carlo"].includes(tab) && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 20 }}>
          {Object.entries(portfolios).map(([name, p]) => (
            <StrategyToggle key={name} name={name} color={p.color} active={activeStrategies.includes(name)} onClick={() => toggleStrategy(name)} />
          ))}
        </div>
      )}

      {/* ═══════ OVERVIEW TAB ═══════ */}
      {tab === "Overview" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 28 }}>
            {Object.entries(allStats).map(([name, s]) => (
              <MetricCard
                key={name}
                label={name.replace(" Benchmark", "")}
                value={`${(s.sharpe).toFixed(2)}`}
                sub={`${(s.ret * 100).toFixed(1)}% ret · ${(s.vol * 100).toFixed(1)}% vol`}
                color={s.color}
              />
            ))}
          </div>

          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", marginBottom: 12 }}>Sharpe Ratio Comparison (higher = better risk-adjusted return)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={Object.entries(allStats).map(([name, s]) => ({ name: name.replace(" Benchmark", ""), sharpe: +s.sharpe.toFixed(3), fill: s.color }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
              <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
              <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }} />
              <Bar dataKey="sharpe" radius={[6, 6, 0, 0]}>
                {Object.entries(allStats).map(([name, s], i) => (
                  <Cell key={i} fill={s.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", margin: "28px 0 12px" }}>Asset Universe</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 8 }}>
            {ASSET_NAMES.map((name) => {
              const a = ASSETS[name];
              return (
                <div key={name} style={{ background: "#1a1a2e", borderRadius: 8, padding: "10px 14px", borderLeft: `3px solid ${categoryColors[a.category]}` }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#d0d0f0" }}>{name}</div>
                  <div style={{ fontSize: 11, color: "#6b7280", marginTop: 3 }}>
                    E[r]: {(a.ret * 100).toFixed(1)}% · σ: {(a.vol * 100).toFixed(1)}% · <span style={{ color: categoryColors[a.category] }}>{a.category}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ═══════ STRATEGIES TAB ═══════ */}
      {tab === "Strategies" && (
        <div>
          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", marginBottom: 16 }}>Portfolio Allocations</h3>
          {activeStrategies.map((name) => {
            const s = allStats[name];
            return (
              <div key={name} style={{ marginBottom: 24, background: "#1a1a2e", borderRadius: 12, padding: 20, border: `1px solid ${s.color}30` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                  <h4 style={{ margin: 0, color: s.color, fontSize: 15, fontWeight: 700 }}>{name}</h4>
                  <div style={{ fontSize: 12, color: "#8888aa" }}>Sharpe: {s.sharpe.toFixed(2)} · Ret: {(s.ret * 100).toFixed(1)}% · Vol: {(s.vol * 100).toFixed(1)}%</div>
                </div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 12 }}>{s.description}</div>
                <div style={{ display: "flex", gap: 2, height: 28, borderRadius: 6, overflow: "hidden" }}>
                  {s.weights.map((w, i) =>
                    w > 0.005 ? (
                      <div
                        key={i}
                        title={`${ASSET_NAMES[i]}: ${(w * 100).toFixed(1)}%`}
                        style={{
                          width: `${w * 100}%`,
                          background: categoryColors[ASSETS[ASSET_NAMES[i]].category],
                          opacity: 0.4 + w * 1.5,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 9, color: "#fff", fontWeight: 600,
                        }}
                      >
                        {w >= 0.06 ? `${(w * 100).toFixed(0)}%` : ""}
                      </div>
                    ) : null
                  )}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
                  {s.weights.map((w, i) =>
                    w > 0.005 ? (
                      <span key={i} style={{ fontSize: 10, color: "#8888aa" }}>
                        <span style={{ color: categoryColors[ASSETS[ASSET_NAMES[i]].category] }}>●</span> {ASSET_NAMES[i].split("(")[0].trim()}: {(w * 100).toFixed(1)}%
                      </span>
                    ) : null
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ═══════ SCENARIOS TAB ═══════ */}
      {tab === "Scenarios" && (
        <div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 20 }}>
            {Object.entries(SCENARIOS).map(([key, sc]) => (
              <button
                key={key}
                onClick={() => setSelectedScenario(key)}
                style={{
                  padding: "6px 14px", borderRadius: 8, fontSize: 12, cursor: "pointer",
                  border: selectedScenario === key ? "1px solid #a78bfa" : "1px solid #2a2a4a",
                  background: selectedScenario === key ? "#2a2a4e" : "#1a1a2e",
                  color: selectedScenario === key ? "#c4b5fd" : "#6b7280",
                  fontWeight: selectedScenario === key ? 600 : 400,
                }}
              >
                {sc.label}
              </button>
            ))}
          </div>

          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", marginBottom: 12 }}>
            Scenario Impact: {SCENARIOS[selectedScenario].label}
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={activeStrategies.map((name) => ({
                name: name.replace(" Benchmark", ""),
                return: +(allStats[name].scenarios[selectedScenario] * 100).toFixed(2),
                fill: allStats[name].color,
              }))}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
              <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
              <YAxis dataKey="name" type="category" tick={{ fill: "#9ca3af", fontSize: 12 }} width={110} />
              <Tooltip
                contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }}
                formatter={(v) => [`${v}%`, "Return"]}
              />
              <ReferenceLine x={0} stroke="#4b5563" />
              <Bar dataKey="return" radius={[0, 6, 6, 0]}>
                {activeStrategies.map((name, i) => (
                  <Cell key={i} fill={allStats[name].return < 0 ? "#ef4444" : allStats[name].color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", margin: "28px 0 12px" }}>All Scenarios Heatmap</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "8px 12px", color: "#6b7280", borderBottom: "1px solid #2a2a4a" }}>Strategy</th>
                  {Object.entries(SCENARIOS)
                    .filter(([k]) => k !== "Base Case")
                    .map(([k, sc]) => (
                      <th key={k} style={{ padding: "8px 6px", color: "#6b7280", borderBottom: "1px solid #2a2a4a", fontSize: 10 }}>{sc.label}</th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {activeStrategies.map((name) => (
                  <tr key={name}>
                    <td style={{ padding: "8px 12px", color: allStats[name].color, fontWeight: 600, borderBottom: "1px solid #1a1a2e" }}>{name.replace(" Benchmark", "")}</td>
                    {Object.entries(SCENARIOS)
                      .filter(([k]) => k !== "Base Case")
                      .map(([k]) => {
                        const val = allStats[name].scenarios[k];
                        const intensity = Math.min(Math.abs(val) * 3, 1);
                        const bg = val >= 0 ? `rgba(16, 185, 129, ${intensity * 0.4})` : `rgba(239, 68, 68, ${intensity * 0.4})`;
                        return (
                          <td key={k} style={{ padding: "8px 6px", textAlign: "center", background: bg, color: val >= 0 ? "#6ee7b7" : "#fca5a5", borderBottom: "1px solid #1a1a2e", fontWeight: 500 }}>
                            {(val * 100).toFixed(1)}%
                          </td>
                        );
                      })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════ MONTE CARLO TAB ═══════ */}
      {tab === "Monte Carlo" && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
            <span style={{ fontSize: 13, color: "#8888aa" }}>Horizon:</span>
            {[5, 10, 15, 20, 30].map((y) => (
              <button
                key={y}
                onClick={() => setMcYears(y)}
                style={{
                  padding: "4px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                  border: mcYears === y ? "1px solid #a78bfa" : "1px solid #2a2a4a",
                  background: mcYears === y ? "#2a2a4e" : "transparent",
                  color: mcYears === y ? "#c4b5fd" : "#6b7280",
                }}
              >
                {y}Y
              </button>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 24 }}>
            {activeStrategies.map((name) => {
              const mc = mcResults[name];
              if (!mc) return null;
              return (
                <div key={name} style={{ background: "#1a1a2e", borderRadius: 12, padding: 16, border: `1px solid ${allStats[name].color}30` }}>
                  <h4 style={{ margin: "0 0 12px", color: allStats[name].color, fontSize: 14, fontWeight: 700 }}>{name}</h4>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12 }}>
                    <div>
                      <div style={{ color: "#6b7280" }}>Median ($100→)</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: "#e0e0f0" }}>${mc.median.toFixed(0)}</div>
                    </div>
                    <div>
                      <div style={{ color: "#6b7280" }}>5th %ile (worst)</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: "#fca5a5" }}>${mc.p5.toFixed(0)}</div>
                    </div>
                    <div>
                      <div style={{ color: "#6b7280" }}>95th %ile (best)</div>
                      <div style={{ fontSize: 18, fontWeight: 600, color: "#6ee7b7" }}>${mc.p95.toFixed(0)}</div>
                    </div>
                    <div>
                      <div style={{ color: "#6b7280" }}>Median Max DD</div>
                      <div style={{ fontSize: 18, fontWeight: 600, color: "#fbbf24" }}>-{(mc.medianDD * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                  {/* Sparkline of sample paths */}
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart>
                      <XAxis dataKey="x" hide />
                      <YAxis hide domain={["auto", "auto"]} />
                      {mc.paths.slice(0, 15).map((path, pi) => (
                        <Line
                          key={pi}
                          data={path.map((v, xi) => ({ x: xi, y: v }))}
                          dataKey="y"
                          stroke={allStats[name].color}
                          strokeOpacity={0.15}
                          strokeWidth={1}
                          dot={false}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
          </div>

          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", marginBottom: 12 }}>Distribution of Outcomes ($100 invested over {mcYears} years)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={activeStrategies.map((name) => {
                const mc = mcResults[name];
                return mc ? { name: name.replace(" Benchmark", ""), p5: +mc.p5.toFixed(0), p25: +mc.p25.toFixed(0), median: +mc.median.toFixed(0), p75: +mc.p75.toFixed(0), p95: +mc.p95.toFixed(0) } : null;
              }).filter(Boolean)}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
              <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} angle={-15} textAnchor="end" height={55} />
              <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} tickFormatter={(v) => `$${v}`} />
              <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }} formatter={(v) => [`$${v}`, ""]} />
              <Legend />
              <Bar dataKey="p5" name="5th %ile" fill="#ef444466" radius={[2, 2, 0, 0]} />
              <Bar dataKey="p25" name="25th %ile" fill="#f59e0b66" radius={[2, 2, 0, 0]} />
              <Bar dataKey="median" name="Median" fill="#10b981" radius={[2, 2, 0, 0]} />
              <Bar dataKey="p75" name="75th %ile" fill="#3b82f666" radius={[2, 2, 0, 0]} />
              <Bar dataKey="p95" name="95th %ile" fill="#8b5cf666" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ═══════ EFFICIENT FRONTIER TAB ═══════ */}
      {tab === "Efficient Frontier" && (
        <div>
          <h3 style={{ fontSize: 16, fontWeight: 600, color: "#a5b4fc", marginBottom: 4 }}>Risk–Return Landscape</h3>
          <p style={{ fontSize: 12, color: "#6b7280", marginBottom: 16 }}>3,000 random portfolios (grey) vs. named strategies (colored). The frontier's upper-left edge is efficient.</p>
          <ResponsiveContainer width="100%" height={450}>
            <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
              <XAxis
                type="number"
                dataKey="x"
                name="Volatility"
                tick={{ fill: "#6b7280", fontSize: 11 }}
                tickFormatter={(v) => `${v}%`}
                label={{ value: "Volatility (%)", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 12 }}
                domain={["auto", "auto"]}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Return"
                tick={{ fill: "#6b7280", fontSize: 11 }}
                tickFormatter={(v) => `${v}%`}
                label={{ value: "Expected Return (%)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 12 }}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }}
                formatter={(val, name) => [`${val}%`, name === "x" ? "Volatility" : "Return"]}
                labelFormatter={() => ""}
              />
              <Scatter name="Random Portfolios" data={frontier} fill="#ffffff" fillOpacity={0.08} r={2} isAnimationActive={false} />
              {strategyPoints.map((sp) => (
                <Scatter key={sp.name} name={`${sp.name} (SR: ${sp.sharpe})`} data={[sp]} fill={sp.color} r={9} strokeWidth={2} stroke="#fff" isAnimationActive={false} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ═══════ DEEP DIVE TAB ═══════ */}
      {tab === "Deep Dive" && (
        <div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 20 }}>
            {Object.keys(portfolios).map((name) => (
              <button
                key={name}
                onClick={() => setDeepDiveStrategy(name)}
                style={{
                  padding: "6px 14px", borderRadius: 8, fontSize: 12, cursor: "pointer",
                  border: deepDiveStrategy === name ? `1px solid ${allStats[name].color}` : "1px solid #2a2a4a",
                  background: deepDiveStrategy === name ? `${allStats[name].color}20` : "#1a1a2e",
                  color: deepDiveStrategy === name ? allStats[name].color : "#6b7280",
                  fontWeight: deepDiveStrategy === name ? 600 : 400,
                }}
              >
                {name}
              </button>
            ))}
          </div>

          {(() => {
            const s = allStats[deepDiveStrategy];
            const mc = runMonteCarlo(portfolios[deepDiveStrategy].weights, assetData, CORR_BASE, 10, 2000);

            // Risk contribution
            const riskContrib = s.weights.map((w, i) => {
              let marginal = 0;
              for (let j = 0; j < s.weights.length; j++) {
                marginal += s.weights[j] * assetData[i].vol * assetData[j].vol * CORR_BASE[i][j];
              }
              return { name: ASSET_NAMES[i].split("(")[0].trim(), contrib: w * marginal / (s.vol * s.vol), weight: w };
            }).filter((r) => r.weight > 0.005);

            // Allocation by category
            const catAlloc = { equity: 0, bond: 0, alt: 0 };
            s.weights.forEach((w, i) => { catAlloc[assetData[i].category] += w; });

            return (
              <div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                  <MetricCard label="Expected Return" value={`${(s.ret * 100).toFixed(1)}%`} color="#10b981" />
                  <MetricCard label="Volatility" value={`${(s.vol * 100).toFixed(1)}%`} color="#f59e0b" />
                  <MetricCard label="Sharpe Ratio" value={s.sharpe.toFixed(2)} color="#818cf8" />
                  <MetricCard label="Median 10Y" value={`$${mc.median.toFixed(0)}`} sub="per $100 invested" color="#c084fc" />
                  <MetricCard label="95% Max DD" value={`-${(mc.p95DD * 100).toFixed(1)}%`} color="#ef4444" />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
                  <div>
                    <h4 style={{ color: "#a5b4fc", fontSize: 14, marginBottom: 8 }}>Category Allocation</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={[
                            { name: "Equity", value: +(catAlloc.equity * 100).toFixed(1), fill: categoryColors.equity },
                            { name: "Bonds", value: +(catAlloc.bond * 100).toFixed(1), fill: categoryColors.bond },
                            { name: "Alts", value: +(catAlloc.alt * 100).toFixed(1), fill: categoryColors.alt },
                          ]}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}%`}
                          labelLine={false}
                        >
                          {[categoryColors.equity, categoryColors.bond, categoryColors.alt].map((c, i) => (
                            <Cell key={i} fill={c} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <h4 style={{ color: "#a5b4fc", fontSize: 14, marginBottom: 8 }}>Risk Contribution</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={riskContrib.sort((a, b) => b.contrib - a.contrib)} layout="vertical">
                        <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                        <YAxis dataKey="name" type="category" tick={{ fill: "#9ca3af", fontSize: 10 }} width={95} />
                        <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }} formatter={(v) => [`${(v * 100).toFixed(1)}%`, "Risk %"]} />
                        <Bar dataKey="contrib" fill={s.color} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <h4 style={{ color: "#a5b4fc", fontSize: 14, marginBottom: 8 }}>Scenario Stress Test</h4>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart
                    data={Object.entries(SCENARIOS)
                      .filter(([k]) => k !== "Base Case")
                      .map(([k, sc]) => ({ name: sc.label, return: +(s.scenarios[k] * 100).toFixed(2) }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                    <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#e0e0f0" }} formatter={(v) => [`${v}%`, "Return"]} />
                    <ReferenceLine y={0} stroke="#4b5563" />
                    <Bar dataKey="return" radius={[4, 4, 0, 0]}>
                      {Object.entries(SCENARIOS)
                        .filter(([k]) => k !== "Base Case")
                        .map(([k], i) => (
                          <Cell key={i} fill={s.scenarios[k] >= 0 ? "#10b981" : "#ef4444"} />
                        ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })()}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 40, padding: "16px 0", borderTop: "1px solid #1e1e3a", fontSize: 11, color: "#4b5563" }}>
        Expected returns and correlations are forward-looking estimates based on historical premia and current conditions. This is an analytical tool, not investment advice. Risk-free rate assumed at 4.5%. Monte Carlo uses 2,000 simulations with geometric Brownian motion.
      </div>
    </div>
  );
}
