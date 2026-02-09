"""Functions for fitting distributions to expert (probabilistic) judgements."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from metalog_jax.base import MetalogInputData, MetalogParameters
from metalog_jax.metalog import fit
from metalog_jax.utils import DEFAULT_Y

from health_adjusted_age.config import MetalogConfig


# ==============================================================================
# I/O Helpers
# ==============================================================================
def save_metalog_pickle(metalog, year: int, sex: str, output_dir: Path):
    """Save metalog using pickle format."""
    filename = f"metalog_{year}_{sex}.pkl"
    filepath = output_dir / filename
    metalog.save(str(filepath))
    return filepath


def save_metalog_coefficients(metalog, year: int, sex: str, output_dir: Path):
    """Save metalog coefficients and metadata as JSON."""
    metalog_info = {
        "year": year,
        "sex": sex,
        "coefficients": metalog.a.tolist(),
        "num_terms": int(metalog.num_terms),
        "boundedness": str(metalog.boundedness),
        "lower_bound": float(metalog.lower_bound),
        "upper_bound": float(metalog.upper_bound),
        "method": str(metalog.method),
    }

    filename = f"metalog_{year}_{sex}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(metalog_info, f, indent=2)

    return filepath


def create_metadata_summary(
    results: list, failures: list, output_dir: Path, metalog_config: MetalogConfig
):
    """Create a summary file with all fitted distributions."""
    summary = {
        "successful_fits": len(results),
        "config": {
            "boundedness": metalog_config.boundedness,
            "lower_bound": metalog_config.lower_bound,
            "upper_bound": metalog_config.upper_bound,
            "method": metalog_config.method,
            "num_terms": metalog_config.num_terms,
        },
        "fits": [],
        "failures": [],
    }

    for result in results:
        summary["fits"].append(
            {
                "year": result["year"],
                "sex": result["sex"],
                "success": result["success"],
                "n_obs": result["n_obs"],
                "data_min": result["data_min"],
                "data_max": result["data_max"],
                "pickle_file": result["pickle_file"].name,
                "json_file": result["json_file"].name,
            }
        )

    for failure in failures:
        summary["failures"].append(
            {
                "year": failure["year"],
                "sex": failure["sex"],
                "success": failure["success"],
                "n_obs": failure["n_obs"],
                "data_min": failure["data_min"],
                "data_max": failure["data_max"],
                "error": failure["error"],
                "error_type": failure["error_type"],
                "pickle_file": failure["pickle_file"],
                "json_file": failure["json_file"],
            }
        )

    summary_path = output_dir / "metalog_fits_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path


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
    metalog_config : (Dict[str, Any])
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
    metalog_config : (Dict[str, Any])
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

    # Get unique combinations
    combinations = df.groupby(["year", "sex"]).size().reset_index(name="count")
    print(f"\nFound {len(combinations)} unique year-sex combinations:")
    print(combinations)

    results = []
    failures = []

    for _, row in combinations.iterrows():
        year = row["year"]
        sex = row["sex"]

        print(f"  \n{'-' * 90}")
        print(f"  Processing: Year={year}, Sex={sex}")
        print(f"  {'-' * 90}")

        # Filter and process
        df_filtered = df[(df["year"] == year) & (df["sex"] == sex)]
        mix_arr = df_filtered["mix_vals"].explode().astype(float).to_numpy()

        # Fit metalog
        try:
            metalog = fit_pair_metalog(mix_arr, metalog_config)

            # Save metalog (pickle and JSON formats)
            pickle_file = save_metalog_pickle(metalog, year, sex, out_dir)
            json_file = save_metalog_coefficients(metalog, year, sex, out_dir)

            print(f"  ✓ Saved pickle: {pickle_file.name}")
            print(f"  ✓ Saved JSON: {json_file.name}")

            # Store result (for successes)
            results.append(
                {
                    "year": year,
                    "sex": sex,
                    "success": True,
                    "n_obs": len(mix_arr),
                    "data_min": float(mix_arr.min()),
                    "data_max": float(mix_arr.max()),
                    "metalog": metalog,
                    "pickle_file": pickle_file,
                    "json_file": json_file,
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
                    "pickle_file": None,
                    "json_file": None,
                }
            )
            continue

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
    print(f"  - {successful} pickle files (.pkl)")
    print(f"  - {successful} JSON files (.json)")
    print(f"  - 1 summary file ({summary_path.name})")

    return results
