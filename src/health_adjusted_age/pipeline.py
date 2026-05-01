import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tomli_w

from health_adjusted_age.config import ModelConfig
from health_adjusted_age.fitting import fit_all_metalogs
from health_adjusted_age.io_sampling import (
    save_hsa_age_samples_to_parquet,
    save_samples_to_parquet,
)
from health_adjusted_age.qa_fitting import run_qa
from health_adjusted_age.rebase import (
    compute_baseline_haa_means,
    compute_haa_summary,
    rebase_haa_distributions,
)
from health_adjusted_age.sampling import (
    calculate_hsa_ages,
    calculate_model_inputs,
    filter_ex_data,
    filter_ex_data_all_ages,
)


# ==============================================================================
# Helper: Log the config used for a run
# ==============================================================================
def save_run_config(config: ModelConfig, path: Path) -> None:
    d = asdict(config)
    d["timestamp"] = datetime.now().isoformat()
    d["paths"] = {k: str(v) for k, v in d["paths"].items()}
    d["fitting"]["metalog"]["boundedness"] = d["fitting"]["metalog"][
        "boundedness"
    ].value
    d["fitting"]["metalog"]["method"] = d["fitting"]["metalog"]["method"].value
    # tomli_w can't serialise None - convert to empty string
    if d["sampling"]["rebase_year"] is None:
        d["sampling"]["rebase_year"] = ""
    with open(path, "wb") as f:
        tomli_w.dump(d, f)


# ==============================================================================
# Helper: Check for existence of fitted metalogs
# ==============================================================================
def _assert_fitted_dir_exists(config: ModelConfig) -> None:
    if not config.paths.fitted_dir.exists() or not any(
        config.paths.fitted_dir.iterdir()
    ):
        raise FileNotFoundError(
            f"Fitted metalogs not found at {config.paths.fitted_dir}. "
            "Run run_fitting() first."
        )


# ==============================================================================
# Helper: Config fingerprinting - save a hash of the metalog config in act 1,
# verify it matches at the start of act 2
# ==============================================================================
def _compute_metalog_hash(config: ModelConfig) -> str:
    d = asdict(config.fitting.metalog)
    # normalise enums to their values so the string is stable
    d["boundedness"] = d["boundedness"].value
    d["method"] = d["method"].value
    serialised = json.dumps(d, sort_keys=True)  # sort_keys ensures stable ordering
    return hashlib.sha256(serialised.encode()).hexdigest()


def _save_metalog_hash(config: ModelConfig) -> None:
    hash_path = config.paths.fitted_dir / "metalog_config.hash"
    hash_path.write_text(_compute_metalog_hash(config))


def _check_metalog_hash(config: ModelConfig) -> None:
    hash_path = config.paths.fitted_dir / "metalog_config.hash"
    if not hash_path.exists():
        raise FileNotFoundError(
            "No config hash found in fitted_dir. Re-run run_fitting() to generate one."
        )
    saved = hash_path.read_text().strip()
    current = _compute_metalog_hash(config)
    if saved != current:
        raise ValueError(
            "Metalog config has changed since fitted outputs were generated. "
            "Re-run run_fitting() before run_sampling()."
        )


# ==============================================================================
# Act 1: Fit metalogs (run once)
# ==============================================================================
def run_metalog_fitting(config: ModelConfig):
    """
    Act 1: Fit metalogs and optionally run QA.
    Outputs are saved to config.paths.fitted_dir for reuse.
    """
    save_run_config(config, config.paths.data_dir / "run_config_fitting.toml")

    fit_all_metalogs(
        mixture_path=config.paths.mix_dist_path,
        out_dir=config.paths.fitted_dir,
        metalog_config=config.fitting.metalog,
    )
    _save_metalog_hash(config)  # <-- save hash at end of act 1

    if config.fitting.run_qa:
        run_qa(paths=config.paths)


# ==============================================================================
# Act 2: Generate HAA samples (run as needed)
# ==============================================================================
def run_haa_sampling(config: ModelConfig):
    """
    Act 2: Run sampling and calculate health-adjusted ages.
    Requires fitted metalogs already exist in config.paths.fitted_dir.
    """
    _assert_fitted_dir_exists(config)
    _check_metalog_hash(config)  # <-- verify hash at start of act 2

    save_run_config(config, config.paths.data_dir / "run_config_sampling.toml")

    ex_df = filter_ex_data(
        path=config.paths.ex_data_path,
        sampling_config=config.sampling,
    )

    summary, delta_dfle_per_ly_samples = calculate_model_inputs(
        ex_df=ex_df, paths=config.paths, sampling_config=config.sampling
    )

    summary.to_csv(config.paths.delta_dfle_per_ly_summary_path, index=False)
    save_samples_to_parquet(
        delta_dfle_per_ly_samples, config.paths.delta_dfle_per_ly_samples_path
    )

    ex_df_all_ages = filter_ex_data_all_ages(
        path=config.paths.ex_data_path, sampling_config=config.sampling
    )

    haa_samples = calculate_hsa_ages(
        ex_df_all_ages=ex_df_all_ages,
        delta_dfle_per_ly_samples=delta_dfle_per_ly_samples,
        sampling_config=config.sampling,
    )

    if config.sampling.rebase_year is not None:
        baseline_means = compute_baseline_haa_means(
            config.sampling.rebase_year, haa_samples
        )
        haa_samples = rebase_haa_distributions(
            baseline_means, haa_samples, config.sampling.rebase_year
        )
        # drop the rebase year itself - needed for baseline calculation but not a
        # meaningful output
        output_years = set(config.sampling.target_years) - {config.sampling.rebase_year}
        # guard
        if not output_years:
            raise ValueError(
                "No output years remaining after excluding rebase year "
                f"{config.sampling.rebase_year}. "
                "Specify at least one --year other than the rebase year."
            )
        haa_samples = {
            (year, sex, age): samples
            for (year, sex, age), samples in haa_samples.items()
            if year in output_years
        }

        haa_df = compute_haa_summary(haa_samples)
        haa_df.to_csv(config.paths.haa_summary_path, index=False)
        save_hsa_age_samples_to_parquet(haa_samples, config.paths.haa_samples_path)

    return haa_df, haa_samples


# ==============================================================================
# Convenience: Run both acts (preserves original behaviour)
# ==============================================================================
def run_pipeline(config: ModelConfig):
    """Run both acts end-to-end."""
    run_metalog_fitting(config)
    return run_haa_sampling(config)
