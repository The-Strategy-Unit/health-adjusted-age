"""Functions for checking fitted distributions."""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metalog_jax.metalog import Metalog

from health_adjusted_age.config import PathsConfig
from health_adjusted_age.io_sampling import load_metalogs_json


# ==============================================================================
# Helper: create plots
# ==============================================================================
def create_diagnostic_plots(
    year: int, sex: str, mix_arr: np.ndarray, metalog: Metalog, out_dir: Path
):
    """
    Create three diagnostic plots for a fitted metalog:
    1. Quantile comparison
    2. Residuals
    3. QQ plot
    """
    # Probability grid for evaluation
    P_GRID = np.linspace(0.001, 0.999, 1000)

    # Get fitted quantiles and empirical quantiles
    fitted_quantiles = metalog.ppf(P_GRID)
    empirical_quantiles = jnp.quantile(mix_arr, P_GRID)

    # Calculate residuals
    residuals = empirical_quantiles - fitted_quantiles
    residual_std = np.std(residuals)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Quantile Comparison
    ax1.plot(
        P_GRID,
        empirical_quantiles,
        label="Original Data",
        marker="o",
        markersize=3,
        linewidth=2,
        alpha=0.7,
        markevery=100,
    )
    ax1.plot(
        P_GRID,
        fitted_quantiles,
        label="Metalog Fit",
        marker="x",
        markersize=3,
        linewidth=2,
        alpha=0.7,
        markevery=100,
    )
    ax1.fill_between(
        P_GRID, empirical_quantiles, fitted_quantiles, alpha=0.2, color="gray"
    )
    ax1.set_xlabel("Probability", fontsize=12)
    ax1.set_ylabel("Quantile Value", fontsize=12)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"Year {year}, Sex {sex}: Fit vs Original Data",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    # Plot 2: Residuals
    ax2.plot(
        P_GRID,
        residuals,
        color="red",
        marker="o",
        markersize=3,
        linewidth=1.5,
        markevery=100,
    )
    ax2.axhline(
        y=0, color="black", linestyle="--", linewidth=2, alpha=0.7, label="Perfect Fit"
    )
    ax2.axhline(
        y=residual_std,
        color="orange",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
        label=f"±1 SD ({residual_std:.3f})",
    )
    ax2.axhline(
        y=-residual_std, color="orange", linestyle=":", alpha=0.5, linewidth=1.5
    )
    ax2.set_xlabel("Probability", fontsize=12)
    ax2.set_ylabel("Residual (Original - Fitted)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best")
    ax2.set_title("Residuals", fontsize=13, fontweight="bold", loc="left")

    # Plot 3: QQ Plot
    ax3.scatter(
        empirical_quantiles,
        fitted_quantiles,
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.plot(
        [empirical_quantiles.min(), empirical_quantiles.max()],
        [empirical_quantiles.min(), empirical_quantiles.max()],
        "r--",
        linewidth=2,
        label="Perfect Fit (y=x)",
    )
    ax3.set_xlabel("Original Quantiles", fontsize=12)
    ax3.set_ylabel("Fitted Quantiles", fontsize=12)
    ax3.legend(fontsize=11, loc="best")
    ax3.set_title("QQ Plot", fontsize=13, fontweight="bold", loc="left")
    ax3.grid(True, alpha=0.3)

    # Add metrics text box to QQ plot
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    r_squared = 1 - (
        np.sum(residuals**2)
        / np.sum((empirical_quantiles - np.mean(empirical_quantiles)) ** 2)
    )
    max_error = np.max(np.abs(residuals))

    metrics_text = (
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"R²: {r_squared:.5f}\n"
        f"Max Error: {max_error:.4f}"
    )

    ax3.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()

    # Save figure
    filename = f"qa_metalog_{year}_{sex}.png"
    filepath = out_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "year": year,
        "sex": sex,
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "max_error": float(max_error),
        "plot_file": filename,
    }


# ==============================================================================
# QA fitted distributions
# ==============================================================================
def run_qa(paths: PathsConfig):
    """Generate QA plots for all fitted metalogs."""

    print("=" * 70)
    print("QUALITY ASSURANCE: GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    qa_dir = Path(paths.qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Load summary file to get all fitted combinations
    summary_path = paths.fitted_dir / "metalogs_summary.json"

    if not summary_path.exists():
        print(f"ERROR: Summary file not found at {summary_path}")
        print("Please run the metalog fitting script first.")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"\nFound {summary['successful_fits']} fitted metalogs")
    print(f"QA plots will be saved to: {qa_dir}\n")

    # Load all metalogs from combined file once upfront
    metalogs_path = paths.fitted_dir / "metalogs.json"

    if not metalogs_path.exists():
        print(f"ERROR: Combined metalogs file not found at {metalogs_path}")
        return

    metalogs = load_metalogs_json(metalogs_path)
    print(
        f"✓ Loaded combined metalogs file: {sum(len(v) for v in metalogs.values())} year-sex combinations\n"
    )

    # Load original data
    print(f"Loading original data from: {paths.mix_dist_path}")
    df = pd.read_parquet(paths.mix_dist_path)

    qa_results = []

    for idx, fit_info in enumerate(summary["fits"], 1):
        year = fit_info["year"]
        sex = fit_info["sex"]

        print(f"{'-' * 70}")
        print(f"[{idx}/{len(summary['fits'])}] Processing: Year={year}, Sex={sex}")

        # Look up metalog from combined dict
        if year not in metalogs or sex not in metalogs[year]:
            print(
                f"  ✗ WARNING: Metalog not found for year={year}, sex={sex}, skipping..."
            )
            continue

        try:
            metalog = metalogs[year][sex]
            print(f"  ✓ Loaded metalog for year={year}, sex={sex}")
        except Exception as e:
            print(f"  ✗ ERROR accessing metalog: {e}")
            continue

        # Get original data for this combination
        df_filtered = df[(df["year"] == year) & (df["sex"] == sex)]
        mix_arr = df_filtered["mix_vals"].explode().astype(float).to_numpy()

        print(f"  Data points: {len(mix_arr)}")

        try:
            result = create_diagnostic_plots(year, sex, mix_arr, metalog, qa_dir)
            qa_results.append(result)
            print(f"  ✓ Created QA plot: {result['plot_file']}")
            print(f"    RMSE: {result['rmse']:.4f}, R²: {result['r_squared']:.5f}")
        except Exception as e:
            print(f"  ✗ ERROR creating plots: {e}")
            continue

    # Save QA summary
    qa_summary = {
        "total_qa_plots": len(qa_results),
        "results": qa_results,
        "statistics": {
            "mean_rmse": float(np.mean([r["rmse"] for r in qa_results])),
            "mean_r_squared": float(np.mean([r["r_squared"] for r in qa_results])),
            "max_rmse": float(np.max([r["rmse"] for r in qa_results])),
            "min_r_squared": float(np.min([r["r_squared"] for r in qa_results])),
        },
    }

    qa_summary_path = qa_dir / "qa_summary.json"
    with open(qa_summary_path, "w") as f:
        json.dump(qa_summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("QA COMPLETE")
    print(f"{'=' * 70}")
    print(f"Successfully created {len(qa_results)} diagnostic plots")
    print(f"QA plots saved to: {qa_dir}")
    print("\nOverall Statistics:")
    print(f"  Mean RMSE: {qa_summary['statistics']['mean_rmse']:.4f}")
    print(f"  Mean R²: {qa_summary['statistics']['mean_r_squared']:.5f}")
    print(f"  Worst RMSE: {qa_summary['statistics']['max_rmse']:.4f}")
    print(f"  Worst R²: {qa_summary['statistics']['min_r_squared']:.5f}")
    print(f"\nQA summary saved to: {qa_summary_path.name}")
    print("=" * 70)

    return qa_results
