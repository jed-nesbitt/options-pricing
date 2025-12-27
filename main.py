from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try importing scipy for normal CDF, fallback to erf
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ------------------------
# Defaults (easy to adjust)
# ------------------------
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_N_SIMS = 100_000
DEFAULT_INPUT_PATH = "data/options.csv"
DEFAULT_OUTPUT_DIR = "outputs"
CHARTS_SUBFOLDER = "charts"


# ------------------------
# Utility functions
# ------------------------
def normal_cdf(x: float) -> float:
    if SCIPY_AVAILABLE:
        return float(norm.cdf(x))
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pdf(x: float) -> float:
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------
# Black-Scholes core formulas
# ------------------------
def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    option_type = option_type.lower()
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    else:
        return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    d1 = _d1(S, K, T, r, sigma)
    return normal_cdf(d1) if option_type == "call" else normal_cdf(d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    return normal_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    return S * math.sqrt(T) * normal_pdf(d1)


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    first = -(S * normal_pdf(d1) * sigma) / (2 * math.sqrt(T))

    if option_type == "call":
        second = -r * K * math.exp(-r * T) * normal_cdf(d2)
        return first + second
    else:
        second = r * K * math.exp(-r * T) * normal_cdf(-d2)
        return first + second


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * math.exp(-r * T) * normal_cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * normal_cdf(-d2)


# ------------------------
# Monte Carlo pricer
# ------------------------
def monte_carlo_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    Z = rng.normal(0.0, 1.0, n_sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    discounted = np.exp(-r * T) * payoffs
    mc_price = float(discounted.mean())
    mc_se = float(discounted.std(ddof=1) / math.sqrt(n_sims))
    return mc_price, mc_se, payoffs, ST


# ------------------------
# Input loading + column mapping
# ------------------------
def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def load_options_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm_to_original: Dict[str, str] = {_norm_col(c): c for c in df.columns}

    def pick(*keys: str) -> str | None:
        for k in keys:
            if k in norm_to_original:
                return norm_to_original[k]
        return None

    # Supports your headers:
    # underlying price (S) -> underlyingprices
    # strike price (K) -> strikepricek
    # time to maturity in years (T) -> timetomaturityinyearst
    # volitility -> volitility
    # type -> type
    col_S = pick("underlyingprices", "underlyingprice", "spotprice", "spot", "s", "underlying")
    col_K = pick("strikepricek", "strikeprice", "strike", "k")
    col_T = pick("timetomaturityinyearst", "timetomaturityinyears", "timetomaturity", "maturity", "t")
    col_sig = pick("volitility", "volatility", "sigma", "vol", "impliedvol", "stdev")
    col_type = pick("type", "optiontype", "callput", "cp", "option")

    missing = []
    if not col_S:
        missing.append("S")
    if not col_K:
        missing.append("K")
    if not col_T:
        missing.append("T")
    if not col_sig:
        missing.append("sigma")
    if not col_type:
        missing.append("option_type")

    if missing:
        raise ValueError(
            "Missing required columns. Need columns for S, K, T, sigma, option type.\n"
            f"Missing: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Normalized: {list(norm_to_original.keys())}"
        )

    return df.rename(
        columns={
            col_S: "S",
            col_K: "K",
            col_T: "T",
            col_sig: "sigma",
            col_type: "option_type",
        }
    ).copy()


def normalize_option_type(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"call", "c"}:
        return "call"
    if s in {"put", "p"}:
        return "put"
    raise ValueError(f"Invalid option type: {x!r} (expected Call/Put)")


def safe_tag(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


# ------------------------
# Plot helpers (save to outputs/<run>/charts/)
# ------------------------
def save_payoff_hist(payoffs: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(payoffs, bins=50, edgecolor="black")
    plt.title("Monte Carlo Distribution of Option Payoffs")
    plt.xlabel("Payoff at Maturity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_convergence_plot(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    analytic_price: float,
    n_sims_max: int,
    out_path: Path,
    rng: np.random.Generator,
) -> None:
    max_pow = max(2.0, math.log10(max(100, n_sims_max)))
    sim_counts = np.unique(np.logspace(2, max_pow, 20).astype(int))
    sim_counts = sim_counts[sim_counts <= n_sims_max]
    if sim_counts.size == 0:
        sim_counts = np.array([max(100, n_sims_max)], dtype=int)

    mc_estimates = []
    for n in sim_counts:
        mc_p, _, _, _ = monte_carlo_european(S, K, T, r, sigma, option_type, int(n), rng=rng)
        mc_estimates.append(mc_p)

    plt.figure(figsize=(10, 5))
    plt.plot(sim_counts, mc_estimates, marker="o", label="Monte Carlo Price")
    plt.axhline(analytic_price, linestyle="--", label="Analytical Price")
    plt.xscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Price Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------
# Main
# ------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Black-Scholes + Monte Carlo pricer (CSV/Excel input).")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to input CSV/Excel (default: data/options.csv)")
    parser.add_argument("--r", type=float, default=DEFAULT_RISK_FREE_RATE, help="Risk-free rate (default: 0.05)")
    parser.add_argument("--sims", type=int, default=DEFAULT_N_SIMS, help="Monte Carlo sims per option (default: 100000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving payoff/convergence plots")
    parser.add_argument("--max-plot-rows", type=int, default=25, help="Only plot first N rows (default: 25)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Base output folder (default: outputs)")
    args = parser.parse_args()

    run_started_at = dt.datetime.now()

    in_path = Path(args.input)

    # Base output folder
    base_out_dir = Path(args.out)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Create a new run folder each time (timestamped)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # charts inside the run folder
    charts_dir = out_dir / CHARTS_SUBFOLDER
    charts_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = run_started_at.strftime("%Y-%m-%d %H:%M:%S")

    df_raw = load_options_table(in_path)
    df = standardize_columns(df_raw)

    rng = np.random.default_rng(args.seed)

    results_rows = []
    plot_count = 0

    for idx, row in df.iterrows():
        try:
            S = float(row["S"])
            K = float(row["K"])
            T = float(row["T"])
            sigma = float(row["sigma"])
            option_type = normalize_option_type(row["option_type"])

            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                raise ValueError(f"Invalid numeric inputs: S={S}, K={K}, T={T}, sigma={sigma} (must be > 0)")

            r = float(args.r)
            n_sims = int(args.sims)

            analytic_price = black_scholes_price(S, K, T, r, sigma, option_type)
            mc_price, mc_se, payoffs, ST = monte_carlo_european(S, K, T, r, sigma, option_type, n_sims, rng=rng)

            # Greeks
            d = delta(S, K, T, r, sigma, option_type)
            g = gamma(S, K, T, r, sigma)
            v = vega(S, K, T, r, sigma)
            th = theta(S, K, T, r, sigma, option_type)
            rh = rho(S, K, T, r, sigma, option_type)

            payoff_p5, payoff_p50, payoff_p95 = np.percentile(payoffs, [5, 50, 95])

            results_rows.append(
                {
                    "timestamp": timestamp_str,
                    "row_index": int(idx),
                    "option_type": option_type,
                    "S": S,
                    "K": K,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                    "n_sims": n_sims,
                    "analytic_price": analytic_price,
                    "mc_price": mc_price,
                    "mc_standard_error": mc_se,
                    "delta": d,
                    "gamma": g,
                    "vega": v,
                    "theta": th,
                    "rho": rh,
                    "payoff_p5": float(payoff_p5),
                    "payoff_p50": float(payoff_p50),
                    "payoff_p95": float(payoff_p95),
                    "ST_mean": float(ST.mean()),
                    "ST_std": float(ST.std(ddof=1)),
                    "error": "",
                }
            )

            # Plots saved into outputs/<run>/charts/
            if (not args.no_plots) and (plot_count < args.max_plot_rows):
                tag = safe_tag(f"{option_type}_S{S}_K{K}_T{T}_sig{sigma}")
                payoff_path = charts_dir / f"{run_id}_row{idx}_{tag}_mc_payoffs.png"
                conv_path = charts_dir / f"{run_id}_row{idx}_{tag}_mc_convergence.png"

                save_payoff_hist(payoffs, payoff_path)
                save_convergence_plot(S, K, T, r, sigma, option_type, analytic_price, n_sims, conv_path, rng=rng)

                plot_count += 1

        except Exception as e:
            results_rows.append(
                {
                    "timestamp": timestamp_str,
                    "row_index": int(idx),
                    "option_type": str(row.get("option_type", "")),
                    "S": row.get("S", ""),
                    "K": row.get("K", ""),
                    "T": row.get("T", ""),
                    "r": args.r,
                    "sigma": row.get("sigma", ""),
                    "n_sims": args.sims,
                    "analytic_price": "",
                    "mc_price": "",
                    "mc_standard_error": "",
                    "delta": "",
                    "gamma": "",
                    "vega": "",
                    "theta": "",
                    "rho": "",
                    "payoff_p5": "",
                    "payoff_p50": "",
                    "payoff_p95": "",
                    "ST_mean": "",
                    "ST_std": "",
                    "error": str(e),
                }
            )

    # Write results CSV into this run folder (no appending needed now)
    out_csv = out_dir / "option_mc_results.csv"
    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(out_csv, index=False)

    # Metadata
    run_finished_at = dt.datetime.now()
    duration_s = (run_finished_at - run_started_at).total_seconds()

    rows_total = int(out_df.shape[0])
    rows_failed = int((out_df["error"].astype(str).str.len() > 0).sum()) if "error" in out_df.columns else 0
    rows_ok = rows_total - rows_failed

    metadata = {
        "run_id": run_id,
        "run_started_at": run_started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "run_finished_at": run_finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration_s,
        "input": {
            "path": str(in_path.resolve()),
            "sha256": file_sha256(in_path),
        },
        "config": {
            "risk_free_rate": float(args.r),
            "n_sims": int(args.sims),
            "seed": args.seed,
            "plots_enabled": (not args.no_plots),
            "max_plot_rows": int(args.max_plot_rows),
        },
        "output": {
            "run_folder": str(out_dir.resolve()),
            "results_csv": str(out_csv.resolve()),
            "charts_folder": str(charts_dir.resolve()),
        },
        "stats": {
            "rows_total": rows_total,
            "rows_ok": rows_ok,
            "rows_failed": rows_failed,
            "charts_saved_rows": int(plot_count),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "scipy_available": bool(SCIPY_AVAILABLE),
        },
    }

    metadata_path = out_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Done.")
    print(f"Input:    {in_path.resolve()}")
    print(f"Run dir:  {out_dir.resolve()}")
    print(f"CSV:      {out_csv.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    if args.no_plots:
        print("Charts:   disabled (--no-plots)")
    else:
        print(f"Charts:   {charts_dir.resolve()} (saved for up to {args.max_plot_rows} rows)")


if __name__ == "__main__":
    main()
