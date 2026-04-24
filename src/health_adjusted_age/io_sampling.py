"""Input-Output sampling helper functions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from metalog_jax.metalog import Metalog


# ==============================================================================
# Load metalogs from combined JSON file into nested dict: dists[year][sex]
# ==============================================================================
def load_metalogs_json(input_path: Path) -> dict:
    """
    Load combined metalogs JSON into nested dict: dists[year][sex]

    Args:
        input_path: Path to the combined metalogs JSON file

    Returns:
        Nested dict with integer year keys: dists[2020]["male"]
    """
    with open(input_path, "r") as f:
        combined = json.load(f)

    return {
        int(year): {
            sex: Metalog.loads(json.dumps(metalog_data))
            for sex, metalog_data in sex_dict.items()
        }
        for year, sex_dict in combined.items()
    }


# ==============================================================================
# Save 'change in DFLE per LE year' samples to parquet file
# ==============================================================================
def save_samples_to_parquet(samples_dict: dict, path: Path):
    """Save samples dictionary to parquet file."""
    frames = []
    for (year, sex), samples in samples_dict.items():
        n = len(samples)
        frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    "sex": sex,
                    "sample_idx": np.arange(n),
                    "model_input": np.array(samples, dtype=np.float64),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
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
