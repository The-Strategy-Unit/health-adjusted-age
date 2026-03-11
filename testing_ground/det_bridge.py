import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from health_adjusted_age.config import ModelConfig, PathsConfig
from health_adjusted_age.io_sampling import save_hsa_age_samples_to_parquet
from health_adjusted_age.sampling import (
    calculate_hsa_ages,
    calculate_model_inputs,
    filter_ex_data,
    filter_ex_data_all_ages,
)


# ==============================================================================
# Compute and apply a deterministic bridge
# ==============================================================================
def compute_deterministic_bridge(haa_samples, model_baseline_year, sexes, ages):
    """
    For each sex/age, compute mean HAA at model_baseline_year.

    Args:
        haa_samples (dict): HAA sample arrays keyed by (year, sex, age).
        model_baseline_year (int): The baseline year at which to compute the
            mean HAA offset.
        sexes (iterable): Sex categories to iterate over (e.g. ['M', 'F']).
        ages (iterable): Age values to iterate over (e.g. range(0, 101)).

    Returns:
        dict: Mean HAA at baseline keyed by (sex, age). Used as a deterministic
            offset to anchor HAA distributions in apply_deterministic_bridge_absolute.

    Warns:
        Prints a warning if no samples exist for a given (year, sex, age) key,
            meaning that key will be absent from the returned dict.
    """
    bridge = {}
    for sex in sexes:
        for age in ages:
            key = (model_baseline_year, sex, age)
            if key in haa_samples:
                bridge[(sex, age)] = float(np.mean(haa_samples[key]))
            else:
                print(f"  ✗ WARNING: No HAA samples for bridge key {key}")
    return bridge


def apply_deterministic_bridge_absolute(haa_samples, bridge, model_baseline_year):
    """
    Rebase absolute HAA distributions to model_baseline_year.

    Subtracts mean HAA at baseline and adds back chronological age — giving
    absolute HAA consistent with a baseline_year anchor where a person's
    HAA ≈ their chronological age at baseline.

    Args:
        haa_samples (dict): HAA sample arrays keyed by (year, sex, age). Years
            before model_baseline_year are skipped.
        bridge (dict): Mean HAA at baseline keyed by (sex, age), used as the
            subtracted offset to anchor the distribution.
        model_baseline_year (int): The anchor year; only entries from this year
            onward are rebased.

    Returns:
        dict: Rebased HAA arrays keyed by (year, sex, age), where each array
            has been shifted so HAA ≈ chronological age at baseline.
    """
    rebased = {}
    for (year, sex, age), samples in haa_samples.items():
        if year < model_baseline_year:
            continue
        key = (sex, age)
        if key not in bridge:
            continue
        rebased[(year, sex, age)] = samples - bridge[key] + age
    return rebased


# ==============================================================================
# Plot examples of rebased HAA samples
# ==============================================================================
PLOT_AGES = [65, 75, 85]


def plot_rebased_haa_panel(
    rebased_haa, sexes, baseline_year, n_paths=30, plot_ages=PLOT_AGES
):
    """
    Panel plot of rebased absolute HAA fan charts.

    Produces one figure per sex, with one panel per age in plot_ages (default:
    65, 75, 85). Each panel shows a fan chart (P10-P90, P25-P75, median),
    sample paths, a chronological-age reference line, and a baseline-year
    marker.

    Args:
        rebased_haa (dict): Rebased HAA arrays keyed by (year, sex, age), as
            returned by apply_deterministic_bridge_absolute.
        sexes (iterable): Sex categories to plot (e.g. ['M', 'F']).
        baseline_year (int): Anchor year; used to filter plot_years and draw
            a vertical reference line on each panel.
        n_paths (int): Number of random sample paths to overlay on each panel.
            Defaults to 30.
        plot_ages (list[int]): Ages to plot panels for. Defaults to [65, 75, 85].

    Returns:
        None: Figures are saved to data/det_bridge/rebased_haa_panel_{sex}.png
            and displayed inline.
    """
    years_all = sorted(set(y for (y, s, a) in rebased_haa.keys()))
    plot_years = [y for y in years_all if y >= baseline_year]

    for sex in sexes:
        ncols = 3
        nrows = int(np.ceil(len(plot_ages) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, nrows * 3.5),
            sharey=False,
            sharex=True,
        )
        axes = axes.flatten()
        sex_label = "Female" if sex == "f" else "Male"
        fig.suptitle(
            f"Rebased Absolute HAA from {baseline_year} Baseline — {sex_label}",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        for i, age in enumerate(plot_ages):
            ax = axes[i]

            samples_matrix = []
            valid_years = []
            for year in plot_years:
                key = (year, sex, age)
                if key in rebased_haa:
                    samples_matrix.append(rebased_haa[key])
                    valid_years.append(year)

            if not samples_matrix:
                ax.set_visible(False)
                continue

            traj = np.array(samples_matrix).T  # (n_samples, n_years)
            years_arr = np.array(valid_years)

            # fan chart
            p10, p25, p75, p90 = [
                np.percentile(traj, p, axis=0) for p in [10, 25, 75, 90]
            ]
            p50 = np.percentile(traj, 50, axis=0)

            ax.fill_between(
                years_arr, p10, p90, alpha=0.15, color="steelblue", label="P10–P90"
            )
            ax.fill_between(
                years_arr, p25, p75, alpha=0.25, color="steelblue", label="P25–P75"
            )
            ax.plot(
                years_arr,
                p50,
                color="steelblue",
                linewidth=2,
                linestyle="--",
                label="Median",
            )

            # sample paths
            idx = np.random.choice(traj.shape[0], size=n_paths, replace=False)
            for j, path_idx in enumerate(idx):
                ax.plot(
                    years_arr,
                    traj[path_idx],
                    alpha=0.2,
                    linewidth=0.5,
                    color="coral",
                    label="Sample paths" if j == 0 else None,
                )

            # reference line at chronological age
            ax.axhline(
                age,
                color="black",
                linewidth=1.0,
                linestyle="-",
                label=f"Chronological age ({age})",
            )
            ax.axvline(baseline_year, color="grey", linewidth=0.8, linestyle=":")

            ax.set_title(f"Age {age}", fontsize=11)
            ax.set_ylabel("HAA (years)", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=8)

            if i == 0:
                ax.legend(fontsize=7, loc="lower left")

        for j in range(len(plot_ages), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

        out = Path("data/det_bridge")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"rebased_haa_panel_{sex}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        plt.show()


# ==============================================================================
# Test deterministic bridge
# ==============================================================================
def run(
    model_baseline_year: int = 2025,
    n_samples: int = 10_000,
    sexes: Optional[list] = None,
    ages: Optional[list] = None,
):
    """
    Run the deterministic bridge test end-to-end.

    Generates HAA samples, computes a deterministic bridge offset at
    model_baseline_year, rebases the distributions, saves outputs, and
    produces fan-chart plots. Intended as the main entry point for
    testing and validating the deterministic bridge approach.

    Args:
        model_baseline_year (int): Anchor year for the bridge; HAA distributions
            are rebased so HAA ≈ chronological age at this year. Defaults to 2025.
        n_samples (int): Number of Monte Carlo samples to generate. Defaults
            to 10,000.
        sexes (list[str], optional): Sex categories to process. Defaults to
            ['f', 'm'].
        ages (list[int], optional): Ages to process. Defaults to range(55, 91).

    Returns:
        tuple:
            - rebased_haa (dict): Rebased HAA arrays keyed by (year, sex, age).
            - bridge (dict): Mean HAA at baseline keyed by (sex, age).
            - summary_df (pd.DataFrame): Per-(year, sex, age) summary statistics
            (mean, std, P10, P50, P90) of the rebased HAA distributions.
            - haa_samples (dict): Raw (pre-rebase) HAA arrays keyed by
            (year, sex, age).

    Outputs:
        Writes the following to data/det_bridge/:
            - bridge_values.csv: Bridge offsets per (sex, age).
            - rebased_haa_summary.csv: Summary statistics of rebased HAA.
            - rebased_haa_samples.parquet: Full rebased HAA sample arrays.
            - rebased_haa_panel_{sex}.png: Fan-chart plots for each sex.
    """
    sexes = sexes or ["f", "m"]
    ages = ages or list(range(55, 91))

    target_years = (
        2025,
        2045,
    )

    config = ModelConfig(
        n_samples=n_samples,
        target_years=target_years,
    )
    paths = PathsConfig()

    print("=" * 70)
    print("TEST: deterministic_bridge")
    print(f"  model_baseline_year : {model_baseline_year}")
    print(f"  n_samples           : {n_samples}")
    print("=" * 70)

    # generate HAA samples
    ex_df = filter_ex_data(path=paths.ex_data_path, model_config=config)

    _, delta_dfle_samples = calculate_model_inputs(
        ex_df=ex_df,
        paths=paths,
        model_config=config,
    )

    ex_df_all_ages = filter_ex_data_all_ages(
        path=paths.ex_data_path, model_config=config
    )

    _, haa_samples = calculate_hsa_ages(
        ex_df_all_ages=ex_df_all_ages,
        delta_dfle_per_ly_samples=delta_dfle_samples,
        model_config=config,
    )

    # compute bridge and rebase
    print(f"\nComputing deterministic bridge at {model_baseline_year}...")
    bridge = compute_deterministic_bridge(haa_samples, model_baseline_year, sexes, ages)
    print(f"  Bridge values (mean HAA at {model_baseline_year}):")
    for (sex, age), val in sorted(bridge.items()):
        print(f"    sex={sex}, age={age}: {val:.3f}")

    rebased_haa = apply_deterministic_bridge_absolute(
        haa_samples, bridge, model_baseline_year
    )

    # save outputs
    out = Path("data/det_bridge")
    out.mkdir(parents=True, exist_ok=True)

    # bridge values
    bridge_df = pd.DataFrame(
        [{"sex": s, "age": a, "bridge_haa": v} for (s, a), v in sorted(bridge.items())]
    )
    bridge_df.to_csv(out / "bridge_values.csv", index=False)

    # rebased HAA summary
    rows = []
    for (year, sex, age), samples in rebased_haa.items():
        rows.append(
            {
                "year": year,
                "sex": sex,
                "age": age,
                "mean": round(float(np.mean(samples)), 4),
                "std": round(float(np.std(samples)), 4),
                "p10": round(float(np.percentile(samples, 10)), 4),
                "p50": round(float(np.percentile(samples, 50)), 4),
                "p90": round(float(np.percentile(samples, 90)), 4),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(["sex", "age", "year"])
    summary_df.to_csv(out / "rebased_haa_summary.csv", index=False)

    save_hsa_age_samples_to_parquet(rebased_haa, out / "rebased_haa_samples.parquet")

    print(f"\nOutputs saved to {out}")
    print("\nRebased HAA summary (female, age 65):")
    print(
        summary_df[(summary_df["sex"] == "f") & (summary_df["age"] == 65)].to_string(
            index=False
        )
    )

    # produce plots
    plot_rebased_haa_panel(
        rebased_haa=rebased_haa,
        sexes=sexes,
        baseline_year=model_baseline_year,
        n_paths=30,
    )

    return rebased_haa, bridge, summary_df, haa_samples


# ==============================================================================
# Entry point — run from CLI or import run() directly
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic bridge test")
    parser.add_argument("--model-baseline-year", type=int, default=2025)
    parser.add_argument("--n-samples", type=int, default=10_000)
    args = parser.parse_args()

    rebased_haa, bridge, summary_df, haa_samples = run(
        model_baseline_year=args.model_baseline_year,
        n_samples=args.n_samples,
    )

    # sanity check
    age = 65
    sex = "m"
    year = 2045

    mean_original = np.mean(haa_samples[(year, sex, age)])
    mean_rebased = np.mean(rebased_haa[(year, sex, age)])
    bridge_scalar = bridge[(sex, age)]

    print(f"mean HAA original 2045:     {mean_original:.4f}")
    print(f"mean HAA rebased 2045:      {mean_rebased:.4f}")
    print(f"difference:                 {mean_original - mean_rebased:.4f}")
    print(f"bridge scalar - age:        {bridge_scalar - age:.4f}")
    # difference and bridge scalar - age should be identical
