"""Functions for rebasing HAA distributions to a *new" model baseline."""

import numpy as np
import pandas as pd


# =====================================================================================
# Summary stats for rebased HAA distributions
# =====================================================================================
def compute_haa_summary(haa_samples: dict) -> pd.DataFrame:
    """
    Compute HAA summary statistics from a samples dict.

    Args:
        haa_samples (dict): HAA distributions keyed by (year, sex, age).

    Returns:
        pd.DataFrame: Summary statistics for each year/sex/age combination.
    """
    summary = []
    for (year, sex, age), samples in haa_samples.items():
        summary.append(
            {
                "sex": sex,
                "year": year,
                "age": age,
                "hsa_age_mean": float(np.mean(samples)),
                "hsa_age_median": float(np.median(samples)),
                "hsa_age_std": float(np.std(samples)),
                "hsa_age_q25": float(np.quantile(samples, 0.25)),
                "hsa_age_q75": float(np.quantile(samples, 0.75)),
                "hsa_age_q05": float(np.quantile(samples, 0.05)),
                "hsa_age_q95": float(np.quantile(samples, 0.95)),
            }
        )
    return pd.DataFrame(summary)


# =====================================================================================
# Compute HAA means for a provided baseline year
# =====================================================================================
def compute_baseline_haa_means(rebase_year: int, haa_samples: dict) -> dict:
    """
    Compute mean HAA for provided year.

    Args:
        rebase_year (int): The year to use as the new baseline.
        haa_samples (dict): HAA distributions keyed by (year, sex, age).

    Returns:
        dict: Mean HAA for baseline year keyed by (sex, age). For use as a scalar
            offset input to rebase_haa_distributions().

    Warns:
        Prints a warning if no HAA distributions exist for the supplied rebase year.
    """
    haa_means = {}
    baseline_keys = [
        (year, sex, age) for year, sex, age in haa_samples if year == rebase_year
    ]

    if not baseline_keys:
        print(
            f"  ✗ WARNING: No HAA distributions exist for baseline year {rebase_year}"
        )
        return haa_means

    for year, sex, age in baseline_keys:
        haa_means[(sex, age)] = float(np.mean(haa_samples[(year, sex, age)]))

    return haa_means


def rebase_haa_distributions(
    baseline_haa_means: dict, haa_samples: dict, rebase_year: int
) -> dict:
    """
    Rebase HAA distributions to a provided year.

    Subtracts mean HAA at baseline and adds back chronological age - giving
    HAA consistent with a baseline anchor year where a person's
    HAA = their chronological age.

    Args:
        baseline_haa_means (dict): Mean HAA at baseline, keyed by (sex, age).
            Output of compute_baseline_haa_means().
        haa_samples (dict): HAA distributions keyed by (year, sex, age).
        rebase_year (int): Anchor year; entries before this year are excluded.

    Returns:
        dict: Rebased HAA distributions keyed by (year, sex, age), such
            that HAA = chronological age in the rebase year.
    """
    rebased_haas = {}
    for (year, sex, age), samples in haa_samples.items():
        if year < rebase_year:
            continue
        key = (sex, age)
        if key not in baseline_haa_means:
            continue
        rebased_haas[(year, sex, age)] = samples - baseline_haa_means[key] + age
    return rebased_haas
