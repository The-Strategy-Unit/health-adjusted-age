"""Functions for generating samples of DFLE per LE year and health adjusted age (HAA)."""  # noqa: E501

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from metalog_jax.base import MetalogRandomVariableParameters
from metalog_jax.metalog import Metalog
from metalog_jax.utils import JaxUniformDistributionParameters

from health_adjusted_age.config import ModelConfig, PathsConfig


# ==============================================================================
# Filtering LE data helpers
# ==============================================================================
def filter_ex_data(path: Path, model_config: ModelConfig):
    """
    Filter life expectancy data for a specific age.

    Parameters:
    -----------
    path : str or Path
        Path to life tables CSV
    base_year : int
        Starting year for filtering
    hsa_ref_age : int
        Specific age to filter for
    target_years : tuple of ints
        Specific year(s) to filter for

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with one row per year-sex combination at the reference age
    """
    # Read CSV
    print(f"\nReading data from: {path}")
    ex_dat = pd.read_csv(path)

    # Apply basic filters using boolean indexing
    ex_dat = ex_dat[
        (ex_dat["type"] == "period") & (ex_dat["age"] == model_config.hsa_ref_age)
    ]

    # Apply year filter if specified
    if model_config.target_years is not None:
        # Convert single year to list
        if isinstance(model_config.target_years, int):
            target_years = [model_config.target_years]
        else:
            target_years = list(model_config.target_years)

        # Filter for specific years (plus base year which is always needed)
        years_to_keep = list(set([model_config.base_year] + target_years))
        ex_dat = ex_dat[ex_dat["year"].isin(years_to_keep)]

        print(f"Filtering for specific years: {target_years}")
        print(f"(Including base year {model_config.base_year} for calculations)")

    return ex_dat


def filter_ex_data_all_ages(path: Path, model_config: ModelConfig):
    """
    Filter life expectancy data for all ages >= hsa_start_age.
    Used for hsa_age calculations.

    Parameters:
    -----------
    path : str or Path
        Path to life tables CSV
    base_year : int
        Starting year for filtering
    hsa_start_age : int
        Minimum age to include
    target_years : tuple of ints
        Specific year(s) to filter for

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with all ages >= hsa_start_age
    """
    # Read CSV
    print(f"\nReading data from: {path}")
    ex_dat = pd.read_csv(path)

    # Apply basic filters using boolean indexing
    ex_dat = ex_dat[
        (ex_dat["type"] == "period") & (ex_dat["age"] >= model_config.hsa_start_age)
    ]

    # Apply year filter if specified
    if model_config.target_years is not None:
        if isinstance(model_config.target_years, int):
            target_years = [model_config.target_years]
        else:
            target_years = list(model_config.target_years)

        years_to_keep = list(set([model_config.base_year] + target_years))
        ex_dat = ex_dat[ex_dat["year"].isin(years_to_keep)]

    return ex_dat


# ==============================================================================
# Generate samples for change in DFLE per LE year
# ==============================================================================
def calculate_model_inputs(
    ex_df: pd.DataFrame, paths: PathsConfig, model_config: ModelConfig
):
    """
    Calculate model inputs using the formula:
    Model_input_{y,s} = (QOI_{y,s}/100 * ex_{y,s} - dfle_{s}) / (ex_{y,s} - ex_{base,s})

    Returns both summary statistics and full sample arrays.

    Parameters:
    -----------
    ex_df : pd.DataFrame
        Filtered life expectancy data (output from filter_ex_data)
    metalogs_dir : Path
        Directory containing fitted metalog files
    base_year : int
        Base year for ex_base calculation
    hsa_ref_age : int
        Reference age (should match the age in ex_df)
    dfle_f : float
        Disability-free life expectancy constant for females
    dfle_m : float
        Disability-free life expectancy constant for males
    n_samples : int
        Number of samples to draw from each metalog
    seed : int
        Random seed for reproducibility
    target_years : tuple of ints
        Only calculate for these specific years

    Returns:
    --------
    tuple: (summary_df, samples_dict)
        summary_df: DataFrame with summary statistics
        samples_dict: Dict with full sample arrays {(year, sex): samples_array}
    """

    print("=" * 70)
    print("CALCULATING MODEL INPUTS")
    print("=" * 70)

    # Filter for target years if specified
    if model_config.target_years is not None:
        if isinstance(model_config.target_years, int):
            target_years = [model_config.target_years]
        else:
            target_years = list(model_config.target_years)

        # Filter ex_df but keep base year data
        ex_df_filtered = ex_df[
            ex_df["year"].isin([model_config.base_year] + target_years)
        ]
        print(f"\nCalculating for specific years: {target_years}")
    else:
        ex_df_filtered = ex_df
        print("\nCalculating for all years")

    print(f"\nInput data shape: {ex_df_filtered.shape}")
    print(f"Years: {sorted(ex_df_filtered['year'].unique())}")
    print(f"Sexes: {sorted(ex_df_filtered['sex'].unique())}")
    print(f"Age: {model_config.hsa_ref_age}")

    # Get base year ex values for each sex
    base_data = ex_df_filtered[ex_df_filtered["year"] == model_config.base_year]

    if len(base_data) == 0:
        raise ValueError(f"No data found for base year {model_config.base_year}")

    print(
        f"\nBase year ({model_config.base_year}) ex values at age {model_config.hsa_ref_age}:"  # noqa: E501
    )
    for _, row in base_data.iterrows():
        print(f"  Sex={row['sex']}: ex={row['ex']:.4f}")

    # Create lookup dict for base year ex values by sex
    ex_base_dict = {}
    for _, row in base_data.iterrows():
        ex_base_dict[row["sex"]] = row["ex"]

    # Filter out base year from processing
    ex_df_to_process = ex_df_filtered[ex_df_filtered["year"] != model_config.base_year]

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(ex_df_to_process)} YEAR-SEX COMBINATIONS")
    print(f"{'=' * 70}\n")

    summary_results = []
    samples_dict = {}  # Store full samples: {(year, sex): array}

    for idx, row in ex_df_to_process.iterrows():
        year = row["year"]
        sex = row["sex"]
        ex_current = row["ex"]

        print(f"Processing: Year={year}, Sex={sex}")

        # Get base year ex for this sex
        if sex not in ex_base_dict:
            print(f"  ✗ WARNING: No base year data for sex={sex}, skipping...")
            continue

        ex_base = ex_base_dict[sex]

        # Get dfle constant for this sex
        dfle = model_config.dfle_f if sex == "f" else model_config.dfle_m

        # Load metalog for this year-sex combination
        metalog_path = paths.fitted_dir / f"metalog_{year}_{sex}.pkl"

        if not metalog_path.exists():
            print(f"  ✗ WARNING: Metalog not found: {metalog_path}, skipping...")
            continue

        try:
            metalog = Metalog.load(metalog_path)
            print("  ✓ Loaded metalog")
        except Exception as e:
            print(f"  ✗ ERROR loading metalog: {e}, skipping...")
            continue

        # Sample QOI from metalog using JAX random sampling
        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(
                seed=model_config.seed + len(summary_results)
            ),
            size=model_config.n_samples,
        )
        qoi_samples = metalog.rvs(rv_params)

        # Calculate model input for each sample
        # Model_input = (QOI/100 * ex - dfle) / (ex - ex_base)
        numerator = qoi_samples / 100 * ex_current - dfle
        denominator = ex_current - ex_base

        if abs(denominator) < 1e-10:
            print(
                f"  ✗ WARNING: Denominator near zero (ex={ex_current:.4f} ≈ ex_base={ex_base:.4f}), skipping..."  # noqa: E501
            )
            continue

        model_input_samples = numerator / denominator

        # Store full samples for later use
        samples_dict[(year, sex)] = np.array(model_input_samples)

        # Store summary statistics
        summary_result = {
            "base": row["base"],
            "type": row["type"],
            "id": row["id"],
            "sex": sex,
            "year": year,
            "age": model_config.hsa_ref_age,
            "ex": ex_current,
            "ex_base": ex_base,
            "dfle": dfle,
            "qoi_mean": float(jnp.mean(qoi_samples)),
            "qoi_median": float(jnp.median(qoi_samples)),
            "qoi_std": float(jnp.std(qoi_samples)),
            "model_input_mean": float(jnp.mean(model_input_samples)),
            "model_input_median": float(jnp.median(model_input_samples)),
            "model_input_std": float(jnp.std(model_input_samples)),
            "model_input_q25": float(jnp.quantile(model_input_samples, 0.25)),
            "model_input_q75": float(jnp.quantile(model_input_samples, 0.75)),
        }

        summary_results.append(summary_result)

        print(f"  ex={ex_current:.4f}, ex_base={ex_base:.4f}")
        print(f"  Model input (mean): {summary_result['model_input_mean']:.4f}")
        print(f"  Model input (median): {summary_result['model_input_median']:.4f}")

    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary_results)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successfully calculated {len(summary_df)} model inputs")
    print(f"Stored {len(samples_dict)} sample distributions")
    print("\nSummary results:")
    print(
        summary_df[
            ["year", "sex", "age", "ex", "model_input_mean", "model_input_median"]
        ]
    )

    return summary_df, samples_dict


# ==============================================================================
# Generate samples for health adjusted age (HAA)
# ==============================================================================
def calculate_hsa_ages(
    ex_df_all_ages: pd.DataFrame,
    delta_dfle_per_ly_samples: dict,
    model_config: ModelConfig,
):
    """
    Calculate hsa_age distributions for all ages >= hsa_start_age using the formula:
    hsa_age_{y,s,i} = age_i - (model_input_{y,s} * (ex_{y,s,i} - ex_{base,s,i}))

    Parameters:
    -----------
    ex_df_all_ages : pd.DataFrame
        Life expectancy data for all ages >= hsa_start_age
    model_inputs_samples : dict
        Dictionary of model input samples {(year, sex): samples_array}
    base_year : int
        Base year for ex_base lookups
    hsa_start_age : int
        Minimum age for calculations

    Returns:
    --------
    pd.DataFrame
        HSA ages with summary statistics for all year-sex-age combinations
    """

    print("=" * 70)
    print("CALCULATING HSA AGES")
    print("=" * 70)

    print(f"\nAge range: {model_config.hsa_start_age} to {ex_df_all_ages['age'].max()}")

    total_combinations = len(
        ex_df_all_ages[ex_df_all_ages["year"] != model_config.base_year]
    )
    print(f"Total combinations: {total_combinations}")

    # Get base year data for ex_base lookups
    base_data = ex_df_all_ages[ex_df_all_ages["year"] == model_config.base_year]

    # Create lookup dict for ex_base by sex and age
    ex_base_lookup = {}
    for _, row in base_data.iterrows():
        key = (row["sex"], row["age"])
        ex_base_lookup[key] = row["ex"]

    # Filter out base year
    ex_df_to_process = ex_df_all_ages[ex_df_all_ages["year"] != model_config.base_year]

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(ex_df_to_process)} YEAR-SEX-AGE COMBINATIONS")
    print(f"{'=' * 70}\n")

    results = []
    hsa_samples_dict = {}  # Store full samples: {(year, sex, age): array}

    for row_idx, (idx, row) in enumerate(ex_df_to_process.iterrows()):
        year = row["year"]
        sex = row["sex"]
        age = row["age"]
        ex_current = row["ex"]

        # Get ex_base for this sex-age combination
        ex_base_key = (sex, age)
        if ex_base_key not in ex_base_lookup:
            if row_idx % 100 == 0:
                print(
                    f"  ✗ WARNING: No base year data for sex={sex}, age={age}, skipping..."  # noqa: E501
                )
            continue

        ex_base = ex_base_lookup[ex_base_key]

        # Get model_input samples for this year-sex combination
        model_input_key = (year, sex)
        if model_input_key not in delta_dfle_per_ly_samples:
            if row_idx % 100 == 0:
                print(
                    f"  ✗ WARNING: No model input samples for year={year}, sex={sex}, skipping..."  # noqa: E501
                )
            continue

        model_input_samples_array = delta_dfle_per_ly_samples[model_input_key]

        # Calculate hsa_age distribution
        # hsa_age = age - (model_input * (ex - ex_base))
        ex_diff = ex_current - ex_base
        hsa_age_samples = age - (model_input_samples_array * ex_diff)

        # Store full samples for later use
        hsa_samples_dict[(year, sex, age)] = np.array(hsa_age_samples)

        # Store result with summary statistics
        result = {
            "base": row["base"],
            "type": row["type"],
            "id": row["id"],
            "sex": sex,
            "year": year,
            "age": age,
            "ex": ex_current,
            "ex_base": ex_base,
            "ex_diff": ex_diff,
            "hsa_age_mean": float(np.mean(hsa_age_samples)),
            "hsa_age_median": float(np.median(hsa_age_samples)),
            "hsa_age_std": float(np.std(hsa_age_samples)),
            "hsa_age_q25": float(np.quantile(hsa_age_samples, 0.25)),
            "hsa_age_q75": float(np.quantile(hsa_age_samples, 0.75)),
            "hsa_age_q05": float(np.quantile(hsa_age_samples, 0.05)),
            "hsa_age_q95": float(np.quantile(hsa_age_samples, 0.95)),
        }

        results.append(result)

        if len(results) % 100 == 0:
            print(f"Processed {len(results)} combinations...")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successfully calculated {len(results_df)} hsa_age distributions")
    print("\nSample results:")
    print(
        results_df[
            ["year", "sex", "age", "hsa_age_mean", "hsa_age_median", "hsa_age_std"]
        ].head(20)
    )  # noqa: E501

    return results_df, hsa_samples_dict
