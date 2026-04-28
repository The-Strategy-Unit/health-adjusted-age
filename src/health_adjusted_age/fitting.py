"""Functions for fitting distributions to expert (probabilistic) judgements."""

from pathlib import Path

import numpy as np
import pandas as pd
from metalog_jax.base import MetalogInputData, MetalogParameters
from metalog_jax.metalog import fit
from metalog_jax.utils import DEFAULT_Y

from health_adjusted_age.config import MetalogConfig
from health_adjusted_age.io_fitting import (
    create_metadata_summary,
    save_metalogs_json,
)


# ==============================================================================
# Fit metalog distributions to a single year-sex pair
# ==============================================================================
def fit_pair_metalog(mix_vals: np.ndarray, metalog_config: MetalogConfig):
    """
    Fit a metalog distribution to a single year-sex combination.

    Parameters
    ----------
    mix_vals : np.ndarray
        Values to fit
    metalog_config : MetalogConfig
        Configuration options with num_terms, lower_bound, upper_bound

    Returns
    -------
    Metalog
        Fitted metalog distribution
    """
    data = MetalogInputData.from_values(
        mix_vals, DEFAULT_Y, precomputed_quantiles=False
    )

    params = MetalogParameters(
        boundedness=metalog_config.boundedness,
        lower_bound=metalog_config.lower_bound,
        upper_bound=metalog_config.upper_bound,
        method=metalog_config.method,
        num_terms=metalog_config.num_terms,
    )

    return fit(data, params)


# ==============================================================================
# Fit metalog distributions to all year-sex pairs
# ==============================================================================
def fit_all_metalogs(mixture_path: Path, out_dir: Path, metalog_config: MetalogConfig):
    """
    Fit metalog distributions to all year-sex combinations.

    Parameters
    ----------
    mixture_path : Path
        Path to mixtures parquet file
    out_dir : Path
        Location for saving fitted distributions
    metalog_config : MetalogConfig
        Configuration options with num_terms, lower_bound, upper_bound

    Returns
    -------
    list
        List of results with metadata for each fit
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FITTING DISTRIBUTIONS TO ALL YEAR-SEX COMBINATIONS")
    print("=" * 90)

    # Read data
    print(f"  \nReading data from: {mixture_path}")
    df = pd.read_parquet(mixture_path)
    # don't fit to baseline 2021
    df = df[df["year"] > 2021]

    # Get unique combinations
    combinations = df.groupby(["year", "sex"]).size().reset_index(name="count")
    print(f"\nFound {len(combinations)} unique year-sex combinations:")
    print(combinations)
    results = []
    failures = []
    dists = {}  # collect all metalogs: dists[year][sex]

    for _, row in combinations.iterrows():
        year = row["year"]
        sex = row["sex"]

        print(f"  \n{'-' * 90}")
        print(f"  Processing: Year={year}, Sex={sex}")
        print(f"  {'-' * 90}")

        # Filter and process
        df_filtered = df[(df["year"] == year) & (df["sex"] == sex)]
        mix_arr = df_filtered["mix_vals"].explode().astype(float).to_numpy()
        try:
            metalog = fit_pair_metalog(mix_arr, metalog_config)

            # Collect into nested dict
            if year not in dists:
                dists[year] = {}
            dists[year][sex] = metalog

            results.append(
                {
                    "year": year,
                    "sex": sex,
                    "success": True,
                    "n_obs": len(mix_arr),
                    "data_min": float(mix_arr.min()),
                    "data_max": float(mix_arr.max()),
                    "metalog": metalog,
                }
            )
        except Exception as e:
            print(f"  ✗ ERROR fitting metalog: {e}")

            # Store failure information
            failures.append(
                {
                    "year": year,
                    "sex": sex,
                    "success": False,
                    "n_obs": len(mix_arr),
                    "data_min": float(mix_arr.min()),
                    "data_max": float(mix_arr.max()),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            continue

    # Save all metalogs to single combined JSON after the loop
    metalog_file = save_metalogs_json(dists, out_dir / "metalogs.json")
    print(f"\n✓ Saved combined metalogs: {metalog_file.name}")

    # Count successes in the list
    successful = len(results)
    failed = len(failures)
    attempts = len(results) + len(failures)

    if failed > 0:
        print("\nFailed combinations:")
        for f in failures:
            print(f"  Year={f['year']}, Sex={f['sex']}: {f['error']}")

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Create summary file
    print(f"\n{'=' * 90}")
    print("CREATING SUMMARY")
    print(f"{'=' * 90}")

    summary_path = create_metadata_summary(results, failures, out_dir, metalog_config)
    print(f"\n✓ Created summary: {summary_path.name}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"{'=' * 70}")
    print(f"Successful: {successful}/{attempts}")
    print(f"Failed: {failed}/{attempts}")
    print(f"Output directory: {out_dir}")
    print("\nFiles created:")
    print(f"  - 1 metalogs file ({metalog_file.name})")
    print(f"  - 1 summary file ({summary_path.name})")

    return results
