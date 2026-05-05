"""Functions for generating samples of DFLE per LE year and health adjusted age (HAA)."""  # noqa: E501

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from metalog_jax.base import MetalogRandomVariableParameters
from metalog_jax.utils import JaxUniformDistributionParameters

from health_adjusted_age.config import PathsConfig, SamplingConfig
from health_adjusted_age.io_sampling import load_metalogs_json


# ==============================================================================
# Filtering LE data helpers
# ==============================================================================
def filter_ex_data(path: Path, sampling_config: SamplingConfig):
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
        (ex_dat["type"] == "period") & (ex_dat["age"] == sampling_config.hsa_ref_age)
    ]

    # Apply year filter if specified
    if sampling_config.target_years is not None:
        # Convert single year to list
        if isinstance(sampling_config.target_years, int):
            target_years = [sampling_config.target_years]
        else:
            target_years = list(sampling_config.target_years)

        # Filter for specific years (plus base year which is always needed)
        years_to_keep = list(set([sampling_config.base_year] + target_years))
        ex_dat = ex_dat[ex_dat["year"].isin(years_to_keep)]

        print(f"Filtering for specific years: {target_years}")
        print(f"(Including base year {sampling_config.base_year} for calculations)")

    return ex_dat


def filter_ex_data_all_ages(path: Path, sampling_config: SamplingConfig):
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
        (ex_dat["type"] == "period") & (ex_dat["age"] >= sampling_config.hsa_start_age)
    ]

    # Apply year filter if specified
    if sampling_config.target_years is not None:
        if isinstance(sampling_config.target_years, int):
            target_years = [sampling_config.target_years]
        else:
            target_years = list(sampling_config.target_years)

        years_to_keep = list(set([sampling_config.base_year] + target_years))
        ex_dat = ex_dat[ex_dat["year"].isin(years_to_keep)]

    return ex_dat


# ==============================================================================
# Generate samples for change in DFLE per LE year
# ==============================================================================
def calculate_model_inputs(
    ex_df: pd.DataFrame, paths: PathsConfig, sampling_config: SamplingConfig
):
    """
    Calculate model inputs using the formula:
    Model_input_{y,s} = (QOI_{y,s}/100 * ex_{y,s} - dfle_{s}) / (ex_{y,s} - ex_{base,s})

    Returns both summary statistics and full sample arrays.

    Parameters:
    -----------
    ex_df : pd.DataFrame
        Filtered life expectancy data (output from filter_ex_data)
    paths : PathsConfig
        Paths configuration including fitted_dir for metalogs
    sampling_config : SamplingConfig
        Configuration with base_year, hsa_ref_age, dfle_f, dfle_m, n_samples, seed,
        target_years

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
    if sampling_config.target_years is not None:
        if isinstance(sampling_config.target_years, int):
            target_years = [sampling_config.target_years]
        else:
            target_years = list(sampling_config.target_years)

        # Filter ex_df but keep base year data
        ex_df_filtered = ex_df[
            ex_df["year"].isin([sampling_config.base_year] + target_years)
        ]
        print(f"\nCalculating for specific years: {target_years}")
    else:
        ex_df_filtered = ex_df
        target_years = sorted(ex_df["year"].unique().tolist())
        print("\nCalculating for all years")

    print(f"\nInput data shape: {ex_df_filtered.shape}")
    print(f"Years: {sorted(ex_df_filtered['year'].unique())}")
    print(f"Sexes: {sorted(ex_df_filtered['sex'].unique())}")
    print(f"Age: {sampling_config.hsa_ref_age}")

    # Get base year ex values for each sex
    base_data = ex_df_filtered[ex_df_filtered["year"] == sampling_config.base_year]

    if len(base_data) == 0:
        raise ValueError(f"No data found for base year {sampling_config.base_year}")

    print(
        f"\nBase year ({sampling_config.base_year}) ex values at age "
        f"{sampling_config.hsa_ref_age}:"
    )
    for _, row in base_data.iterrows():
        print(f"  Sex={row['sex']}: ex={row['ex']:.4f}")

    # Create lookup dict for base year ex values by sex
    ex_base_dict = {}
    for _, row in base_data.iterrows():
        ex_base_dict[row["sex"]] = row["ex"]

    # Filter out base year from processing
    ex_df_to_process = ex_df_filtered[
        ex_df_filtered["year"] != sampling_config.base_year
    ]

    # Determine which years we actually need to process
    years_to_process = sorted(ex_df_to_process["year"].unique().tolist())

    print(f"\n{'=' * 70}")
    print("LOADING METALOGS")
    print(f"{'=' * 70}")

    # Load combined metalogs file and filter to only years needed
    all_metalogs = load_metalogs_json(paths.fitted_dir / "metalogs.json")

    metalogs = {
        year: all_metalogs[year] for year in years_to_process if year in all_metalogs
    }

    missing_years = [y for y in years_to_process if y not in all_metalogs]
    if missing_years:
        print(f"  ✗ WARNING: No metalogs found for years: {missing_years}")

    print(f"  ✓ Loaded metalogs for years: {sorted(metalogs.keys())}")
    print(f"  ✓ Total year-sex combinations: {sum(len(v) for v in metalogs.values())}")

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
        dfle = sampling_config.dfle_f if sex == "f" else sampling_config.dfle_m

        # Look up metalog from filtered dict
        if year not in metalogs or sex not in metalogs[year]:
            print(
                f"  ✗ WARNING: Metalog not found for year={year}, sex={sex},"
                " skipping..."
            )
            continue

        try:
            metalog = metalogs[year][sex]
            print("  ✓ Loaded metalog")
        except Exception as e:
            print(f"  ✗ ERROR loading metalog: {e}, skipping...")
            continue

        # Sample QOI from metalog using JAX random sampling
        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(
                seed=sampling_config.seed + len(summary_results)
            ),
            size=sampling_config.n_samples,
        )
        qoi_samples = metalog.rvs(rv_params)

        # Calculate model input for each sample
        # Model_input = (QOI/100 * ex - dfle) / (ex - ex_base)
        numerator = qoi_samples / 100 * ex_current - dfle
        denominator = ex_current - ex_base

        if abs(denominator) < 1e-10:
            print(
                f"  ✗ WARNING: Denominator near zero "
                f"(ex={ex_current:.4f} ≈ ex_base={ex_base:.4f}), skipping..."
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
            "age": sampling_config.hsa_ref_age,
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
    sampling_config: SamplingConfig,
) -> dict:
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

    print(
        f"\nAge range: {sampling_config.hsa_start_age} to {ex_df_all_ages['age'].max()}"
    )

    total_combinations = len(
        ex_df_all_ages[ex_df_all_ages["year"] != sampling_config.base_year]
    )
    print(f"Total combinations: {total_combinations}")

    # Get base year data for ex_base lookups
    base_data = ex_df_all_ages[ex_df_all_ages["year"] == sampling_config.base_year]

    # Create lookup dict for ex_base by sex and age
    ex_base_lookup = {}
    for _, row in base_data.iterrows():
        key = (row["sex"], row["age"])
        ex_base_lookup[key] = row["ex"]

    # Filter out base year
    ex_df_to_process = ex_df_all_ages[
        ex_df_all_ages["year"] != sampling_config.base_year
    ]

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(ex_df_to_process)} YEAR-SEX-AGE COMBINATIONS")
    print(f"{'=' * 70}\n")

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
                    f"  ✗ WARNING: No base year data for sex={sex}, age={age}",
                    " skipping...",
                )
            continue

        ex_base = ex_base_lookup[ex_base_key]

        # Get model_input samples for this year-sex combination
        model_input_key = (year, sex)
        if model_input_key not in delta_dfle_per_ly_samples:
            if row_idx % 100 == 0:
                print(
                    f"  ✗ WARNING: No model input samples for year={year}, sex={sex}"
                    " skipping..."
                )
            continue

        model_input_samples_array = delta_dfle_per_ly_samples[model_input_key]

        # Calculate hsa_age distribution
        # hsa_age = age - (model_input * (ex - ex_base))
        ex_diff = ex_current - ex_base
        hsa_age_samples = age - (model_input_samples_array * ex_diff)

        # Store full samples for later use
        hsa_samples_dict[(year, sex, age)] = np.array(hsa_age_samples)

    return hsa_samples_dict
