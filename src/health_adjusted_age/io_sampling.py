"""Input-Output sampling helper functions."""

from pathlib import Path

import pandas as pd


# ==============================================================================
# Save 'change in DFLE per LE year' samples to parquet file
# ==============================================================================
def save_samples_to_parquet(samples_dict, path: Path):
    """Save samples dictionary to parquet file."""
    # Convert dict to long-format DataFrame
    records = []
    for (year, sex), samples in samples_dict.items():
        for sample_idx, value in enumerate(samples):
            records.append(
                {
                    "year": year,
                    "sex": sex,
                    "sample_idx": sample_idx,
                    "model_input": value,
                }
            )

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"✓ Saved {len(samples_dict)} sample distributions to: {path}")


# ==============================================================================
# Load 'change in DFLE per LE year' samples
# ==============================================================================
def load_samples_from_parquet(path: Path):
    """Load samples from parquet file back into dictionary."""
    df = pd.read_parquet(path)

    samples_dict = {}
    for (year, sex), group in df.groupby(["year", "sex"]):
        samples_dict[(year, sex)] = group["model_input"].values

    print(f"✓ Loaded {len(samples_dict)} sample distributions from: {path}")
    return samples_dict


# ==============================================================================
# Save HAA samples to parquet file
# ==============================================================================
def save_hsa_age_samples_to_parquet(hsa_samples_dict, path: Path):
    """Save samples dictionary to parquet file."""
    # Convert dict to long-format DataFrame
    records = []
    for (year, sex, age), samples in hsa_samples_dict.items():
        for sample_idx, value in enumerate(samples):
            records.append(
                {
                    "year": year,
                    "sex": sex,
                    "age": age,
                    "sample_idx": sample_idx,
                    "hsa_age": value,
                }
            )

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"✓ Saved {len(hsa_samples_dict)} sample distributions to: {path}")


# ==============================================================================
# Load HAA samples
# ==============================================================================
def load_hsa_age_samples_from_parquet(path: Path):
    """Load samples from parquet file back into dictionary."""
    df = pd.read_parquet(path)

    hsa_samples_dict = {}
    for (year, sex, age), group in df.groupby(["year", "sex", "age"]):
        hsa_samples_dict[(year, sex, age)] = group["hsa_age"].values

    print(f"✓ Loaded {len(hsa_samples_dict)} sample distributions from: {path}")
    return hsa_samples_dict
