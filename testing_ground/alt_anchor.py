# experiments/alternative_anchor.py
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from metalog_jax.base import MetalogRandomVariableParameters
from metalog_jax.metalog import Metalog
from metalog_jax.utils import JaxUniformDistributionParameters

from health_adjusted_age.config import ModelConfig, PathsConfig
from health_adjusted_age.io_sampling import (
    load_hsa_age_samples_from_parquet,
    save_hsa_age_samples_to_parquet,
)
from health_adjusted_age.sampling import (
    calculate_hsa_ages,
    calculate_model_inputs,
    filter_ex_data,
    filter_ex_data_all_ages,
)


# ==============================================================================
# Estimate DFLE at baseline
# ==============================================================================
def estimate_dfle_at_baseline(paths, model_baseline_year, model_config, ex_df, sexes):
    """
    Estimate Disability-Free Life Expectancy (DFLE) at the baseline year.

    Loads the fitted QOI metalog for each sex at model_baseline_year, draws
    samples, and computes DFLE as:

        DFLE_baseline = mean(QOI_baseline) / 100 * ex_baseline

    where QOI is the Quality of Life Index (%) and ex_baseline is life
    expectancy at hsa_ref_age in the baseline year.

    Args:
        paths (PathsConfig): Project path configuration, used to locate fitted
            metalog files in paths.fitted_dir.
        model_baseline_year (int): The baseline year at which DFLE is estimated.
        model_config (ModelConfig): Model configuration containing n_samples,
            seed, and hsa_ref_age.
        ex_df (pd.DataFrame): Life expectancy data with columns
            [year, sex, age, ex].
        sexes (iterable): Sex categories to process (e.g. ['f', 'm']).

    Returns:
        dict: DFLE estimate at baseline keyed by sex (e.g. {'f': 12.3, 'm': 11.1}).
            Sexes with missing metalog or ex data are excluded with a warning.

    Warns:
        Prints a warning and skips the sex if:
            - No fitted metalog file exists at paths.fitted_dir for that sex/year.
            - No ex data is found in ex_df for that sex/year/hsa_ref_age combination.
    """
    dfle_baseline = {}

    for sex in sexes:
        # load baseline year metalog
        metalog_path = paths.fitted_dir / f"metalog_{model_baseline_year}_{sex}.pkl"
        if not metalog_path.exists():
            print(f"  ✗ WARNING: Metalog not found: {metalog_path}")
            continue

        metalog = Metalog.load(metalog_path)
        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(seed=model_config.seed),
            size=model_config.n_samples,
        )
        qoi_samples = metalog.rvs(rv_params)

        # get ex at baseline year
        ex_val = ex_df.loc[
            (ex_df["year"] == model_baseline_year)
            & (ex_df["sex"] == sex)
            & (ex_df["age"] == model_config.hsa_ref_age),
            "ex",
        ].values
        if len(ex_val) == 0:
            print(f"  ✗ WARNING: No ex data for sex={sex}, year={model_baseline_year}")
            continue

        ex_base = float(ex_val[0])
        dfle_baseline[sex] = float(np.mean(qoi_samples) / 100 * ex_base)

        print(
            f"  sex={sex}: mean_qoi={float(np.mean(qoi_samples)):.2f}%, "
            f"ex_base={ex_base:.4f}, "
            f"dfle_baseline={dfle_baseline[sex]:.4f}"
        )

    return dfle_baseline


# ==============================================================================
# Calculate rebased model inputs
# ==============================================================================
def calculate_model_inputs_rebased(
    ex_df, dfle_baseline, model_baseline_year, model_config, paths, sexes
):
    """
    Recalculate delta DFLE per life-expectancy year, anchored to model_baseline_year.

    Draws fresh QOI samples from fitted metalogs and computes the rebased delta
    for each (year, sex) using:

        Δ_new = (QOI_y / 100 * ex_y - DFLE_baseline) / (ex_y - ex_baseline)

    Replicates the seed sequence and iteration order of calculate_model_inputs
    exactly, incrementing sample_counter for every row (including skipped ones)
    to keep downstream results reproducible.

    Args:
        ex_df (pd.DataFrame): Life expectancy data with columns
            [year, sex, age, ex], covering base year, baseline year, and
            target years.
        dfle_baseline (dict): DFLE estimate at baseline keyed by sex, as
            returned by estimate_dfle_at_baseline (e.g. {'f': 12.3, 'm': 11.1}).
        model_baseline_year (int): Anchor year for the rebasing; rows at this
            year are skipped but still increment the seed counter.
        model_config (ModelConfig): Model configuration containing n_samples,
            seed, base_year, target_years, and hsa_ref_age.
        paths (PathsConfig): Project path configuration, used to locate fitted
            metalog files in paths.fitted_dir.
        sexes (iterable): Sex categories to process (e.g. ['f', 'm']).

    Returns:
        dict: Rebased delta DFLE per LE year keyed by (year, sex). Each value
            is a np.ndarray of length n_samples. Entries are excluded with a
            warning if baseline data, metalog, or ex data is missing, or if
            the denominator (ex_y - ex_baseline) is near zero.

    Warns:
        Prints a warning and skips the entry if:
            - No ex_baseline found in ex_df for a given sex.
            - dfle_baseline or ex_baseline is missing for a given sex.
            - The denominator (ex_y - ex_baseline) is near zero (< 1e-10).
            - No fitted metalog file exists for a given (year, sex).
    """
    # get ex at baseline year for each sex/age filtered
    ex_baseline = {}
    for sex in sexes:
        val = ex_df.loc[
            (ex_df["year"] == model_baseline_year)
            & (ex_df["sex"] == sex)
            & (ex_df["age"] == model_config.hsa_ref_age),
            "ex",
        ].values
        if len(val) > 0:
            ex_baseline[sex] = float(val[0])
        else:
            print(f"  ✗ WARNING: No ex_baseline for sex={sex}")

    # filter to target years + baseline - same as calculate_model_inputs
    if model_config.target_years is not None:
        if isinstance(model_config.target_years, int):
            target_years = [model_config.target_years]
        else:
            target_years = list(model_config.target_years)
        ex_df_filtered = ex_df[
            ex_df["year"].isin([model_config.base_year] + target_years)
        ]
    else:
        ex_df_filtered = ex_df

    # exclude base year (2021) rows from processing - same as calculate_model_inputs
    ex_df_to_process = ex_df_filtered[ex_df_filtered["year"] != model_config.base_year]

    rebased_delta = {}
    sample_counter = 0

    for idx, row in ex_df_to_process.iterrows():
        year = row["year"]
        sex = row["sex"]
        ex_year = row["ex"]

        # increment counter but skip baseline year - seed must stay in sync
        if year == model_baseline_year:
            sample_counter += 1
            continue

        if sex not in dfle_baseline or sex not in ex_baseline:
            print(f"  ✗ WARNING: Missing baseline data for sex={sex}, skipping...")
            sample_counter += 1
            continue

        denominator = ex_year - ex_baseline[sex]
        if abs(denominator) < 1e-10:
            print(f"  ✗ WARNING: Denominator near zero year={year}, sex={sex}")
            sample_counter += 1
            continue

        # load metalog
        metalog_path = paths.fitted_dir / f"metalog_{year}_{sex}.pkl"
        if not metalog_path.exists():
            print(f"  ✗ WARNING: Metalog not found: {metalog_path}")
            sample_counter += 1
            continue

        metalog = Metalog.load(metalog_path)

        # match seed exactly to calculate_model_inputs
        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(
                seed=model_config.seed + sample_counter
            ),
            size=model_config.n_samples,
        )
        qoi_samples = metalog.rvs(rv_params)
        sample_counter += 1

        # equation 3 with baseline year anchor
        numerator = qoi_samples / 100 * ex_year - dfle_baseline[sex]
        rebased_delta[(year, sex)] = np.array(numerator / denominator)

        print(
            f"  year={year}, sex={sex}: "
            f"dfle_baseline={dfle_baseline[sex]:.4f}, "
            f"ex_base={ex_baseline[sex]:.4f}, "
            f"ex_year={ex_year:.4f}, "
            f"denominator={denominator:.4f}, "
            f"mean_delta_new={float(np.mean(rebased_delta[(year, sex)])):.4f}"
        )

    return rebased_delta


# ==============================================================================
# Test alternative anchor
# ==============================================================================
def run(
    model_baseline_year: int = 2025,
    n_samples: int = 10_000,
    sexes: Optional[list] = None,
    ages: Optional[list] = None,
):
    """
    Run the alternative anchor experiment end-to-end.

    Implements an alternative rebasing approach where DFLE at model_baseline_year
    is estimated directly from the QOI metalog, and delta DFLE per LE year is
    recalculated using model_baseline_year as the anchor. Results are compared
    numerically against experiment 001 (deterministic bridge) to verify the two
    methods are equivalent.

    Pipeline steps:
        1. Run original pipeline to get delta DFLE samples.
        2. Estimate DFLE at model_baseline_year from QOI metalog.
        3. Recalculate delta DFLE per LE year anchored to model_baseline_year.
        4. Recalculate HAA using model_baseline_year as base_year.
        5. Compare against experiment 001 bridge results if available.
        6. Save outputs to data/alt_anchor/.

    Args:
        model_baseline_year (int): Anchor year for rebasing; used as both the
            DFLE reference point and the base_year for HAA recalculation.
            Defaults to 2025.
        n_samples (int): Number of Monte Carlo samples to generate. Defaults
            to 10,000.
        sexes (list[str], optional): Sex categories to process. Defaults to
            ['f', 'm'].
        ages (list[int], optional): Ages to process. Defaults to range(55, 91).

    Returns:
        tuple:
            - haa_samples_rebased (dict): Rebased HAA arrays keyed by
              (year, sex, age).
            - summary_df (pd.DataFrame): Per-(year, sex, age) summary statistics
              (mean, std, P10, P50, P90) of the rebased HAA distributions.
            - delta_dfle_samples (dict): Original delta DFLE samples from the
              standard pipeline, keyed by (year, sex).
            - rebased_delta (dict): Recalculated delta DFLE per LE year anchored
              to model_baseline_year, keyed by (year, sex).
            - dfle_baseline (dict): Estimated DFLE at baseline keyed by sex.
            - ex_df_all_ages (pd.DataFrame): Life expectancy data across all
              ages used for HAA calculation.

    Outputs:
        Writes the following to data/alt_anchor/:
            - alternative_anchor_summary.csv: Summary statistics of rebased HAA.
            - alternative_anchor_samples.parquet: Full rebased HAA sample arrays.
            - comparison_vs_bridge.csv: Per-(year, sex, age) comparison against
              experiment 001 results (only written if bridge results are found).

    Warns:
        Prints a warning if experiment 001 outputs are not found at
            data/det_bridge/rebased_haa_samples.parquet — comparison is skipped
            but the experiment continues.
        Prints per-entry differences where abs(mean_alt - mean_bridge) > 0.001,
            and a summary indicating whether the methods are numerically
            equivalent.
    """
    sexes = sexes or ["f", "m"]
    ages = ages or list(range(55, 91))
    target_years = (
        model_baseline_year,
        2045,
    )

    config = ModelConfig(
        n_samples=n_samples,
        target_years=target_years,
    )
    paths = PathsConfig()

    print("=" * 70)
    print("EXPERIMENT: alternative_anchor")
    print(f"  model_baseline_year : {model_baseline_year}")
    print(f"  n_samples           : {n_samples}")
    print("=" * 70)

    # run original pipeline
    ex_df = filter_ex_data(path=paths.ex_data_path, model_config=config)

    _, delta_dfle_samples = calculate_model_inputs(
        ex_df=ex_df,
        paths=paths,
        model_config=config,
    )

    ex_df_all_ages = filter_ex_data_all_ages(
        path=paths.ex_data_path, model_config=config
    )

    # estimate DFLE at baseline
    print(f"\nEstimating DFLE at {model_baseline_year}...")
    dfle_baseline = estimate_dfle_at_baseline(
        paths=paths,
        model_baseline_year=model_baseline_year,
        model_config=config,
        ex_df=ex_df_all_ages,
        sexes=sexes,
    )

    # rebase delta dfle
    print(f"\nRecalculating delta DFLE per LE year from {model_baseline_year}...")
    rebased_delta = calculate_model_inputs_rebased(
        ex_df=ex_df,  # not ex_df_all_ages
        dfle_baseline=dfle_baseline,
        model_baseline_year=model_baseline_year,
        model_config=config,
        paths=paths,
        sexes=sexes,
    )

    # recalculate HAA
    # create a config with base_year = model_baseline_year for equation 4
    config_rebased = ModelConfig(**{**vars(config), "base_year": model_baseline_year})

    _, haa_samples_rebased = calculate_hsa_ages(
        ex_df_all_ages=ex_df_all_ages,
        delta_dfle_per_ly_samples=rebased_delta,
        model_config=config_rebased,  # uses 2025 as base year for ex_diff
    )

    # load experiment 001 bridge results for comparison
    print("\nLoading experiment 001 results for comparison...")
    bridge_path = Path("data/det_bridge/rebased_haa_samples.parquet")
    if not bridge_path.exists():
        print("  ✗ WARNING: experiment 001 outputs not found — run 001 first")
        bridge_samples = None
    else:
        bridge_samples = load_hsa_age_samples_from_parquet(bridge_path)

    # numerical comparison
    print("\nNumerical comparison — alternative anchor vs deterministic bridge:")
    differences = []
    for (year, sex, age), samples_alt in haa_samples_rebased.items():
        if bridge_samples and (year, sex, age) in bridge_samples:
            samples_bridge = bridge_samples[(year, sex, age)]
            mean_alt = float(np.mean(samples_alt))
            mean_bridge = float(np.mean(samples_bridge))
            diff = abs(mean_alt - mean_bridge)
            differences.append(
                {
                    "year": year,
                    "sex": sex,
                    "age": age,
                    "mean_alternative": round(mean_alt, 4),
                    "mean_bridge": round(mean_bridge, 4),
                    "abs_diff": round(diff, 6),
                }
            )
            if diff > 0.001:
                print(
                    f"  DIFFERENCE at {year},{sex},{age}: "
                    f"alt={mean_alt:.4f} vs bridge={mean_bridge:.4f} "
                    f"diff={diff:.6f}"
                )

    if differences:
        diff_df = pd.DataFrame(differences)
        max_diff = diff_df["abs_diff"].max()
        print(f"\n  Max absolute difference: {max_diff:.6f}")
        if max_diff < 0.001:
            print("  ✓ Methods are numerically equivalent (diff < 0.001)")
        else:
            print("  ✗ Methods differ — investigate")
    else:
        print("  No experiment 001 results available for comparison")

    # save outputs
    out = Path("data/alt_anchor")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (year, sex, age), samples in haa_samples_rebased.items():
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
    summary_df.to_csv(out / "alternative_anchor_summary.csv", index=False)

    save_hsa_age_samples_to_parquet(
        haa_samples_rebased, out / "alternative_anchor_samples.parquet"
    )

    if differences:
        diff_df.to_csv(out / "comparison_vs_bridge.csv", index=False)

    print(f"\nOutputs saved to {out}")

    # return haa_samples_rebased, summary_df
    return (
        haa_samples_rebased,
        summary_df,
        delta_dfle_samples,
        rebased_delta,
        dfle_baseline,
        ex_df_all_ages,
    )


# ==============================================================================
# Entry point — run from CLI or import run() directly
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternative anchor experiment")
    parser.add_argument("--model-baseline-year", type=int, default=2025)
    parser.add_argument("--n-samples", type=int, default=10_000)
    args = parser.parse_args()

    (
        haa_rebased,
        summary_df,
        delta_dfle_samples,
        rebased_delta,
        dfle_baseline,
        ex_df_all_ages,
    ) = run(
        model_baseline_year=args.model_baseline_year,
        n_samples=args.n_samples,
    )

    config = ModelConfig(
        n_samples=10000,
        target_years=(2025, 2045),
    )
