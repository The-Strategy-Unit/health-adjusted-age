"""Input-Output fitting helper functions."""

import json
from pathlib import Path

from health_adjusted_age.config import MetalogConfig


# ==============================================================================
# Serialize metalog as JSON
# ==============================================================================
def save_metalogs_json(dists: dict, output_path: Path) -> Path:
    """
    Save multiple metalog distributions to a single combined JSON file.

    Args:
        dists: Nested dict of metalog objects: dists[year][sex]
        output_path: Path for the combined output JSON file

    Returns:
        Path to the written file
    """
    combined = {
        str(year): {
            sex: json.loads(metalog.dumps()) for sex, metalog in sex_dict.items()
        }
        for year, sex_dict in dists.items()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    return output_path


# def save_metalog_json(metalog, year: int, sex: str, output_dir: Path):
#     """Save metalog using JSON format."""
#     filename = f"metalog_{year}_{sex}.json"
#     filepath = output_dir / filename
#     metalog.save(str(filepath))
#     return filepath


# ==============================================================================
# Save metalog snapshot as JSON
# ==============================================================================
def save_metalog_snapshot(metalog, year: int, sex: str, output_dir: Path):
    """Save metalog snapshot as JSON."""
    metalog_snapshot = {
        "year": year,
        "sex": sex,
        "coefficients": metalog.a.tolist(),
        "num_terms": int(metalog.num_terms),
        "boundedness": str(metalog.boundedness),
        "lower_bound": float(metalog.lower_bound),
        "upper_bound": float(metalog.upper_bound),
        "method": str(metalog.method),
    }

    filename = f"metalog_snapshot_{year}_{sex}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(metalog_snapshot, f, indent=2)

    return filepath


# ==============================================================================
# Save summary of all fitted metalog models and failures
# ==============================================================================
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
                "snapshot_file": result["snapshot_file"].name,
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
                "snapshot_file": failure["snapshot_file"],
            }
        )

    summary_path = output_dir / "metalogs_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path
