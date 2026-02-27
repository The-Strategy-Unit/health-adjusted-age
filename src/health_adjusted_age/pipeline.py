from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tomli_w

from health_adjusted_age.config import ModelConfig
from health_adjusted_age.fitting import fit_all_metalogs
from health_adjusted_age.qa_fitting import run_qa
from health_adjusted_age.sampling import (
    calculate_hsa_ages,
    calculate_model_inputs,
    filter_ex_data,
    filter_ex_data_all_ages,
    save_hsa_age_samples_to_parquet,
    save_samples_to_parquet,
)


# ==============================================================================
# Helper: validate the config
# ==============================================================================
def validate_config(config: ModelConfig) -> None:
    if not config.target_years:
        raise ValueError("target_years must not be empty")

    valid_years = range(2022, 2051)
    if invalid := [y for y in config.target_years if y not in valid_years]:
        raise ValueError(f"target_years contains out-of-range values: {invalid}")

    if config.n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    if not config.paths.ex_data_path.exists():
        raise FileNotFoundError(config.paths.ex_data_path)

    if not config.paths.mix_dist_path.exists():
        raise FileNotFoundError(config.paths.mix_dist_path)


# ==============================================================================
# Helper: log the config used for a run
# ==============================================================================
def save_run_config(config: ModelConfig, path: Path) -> None:
    d = asdict(config)
    d["timestamp"] = datetime.now().isoformat()
    d["paths"] = {k: str(v) for k, v in d["paths"].items()}
    d["metalog"]["boundedness"] = d["metalog"]["boundedness"].value
    d["metalog"]["method"] = d["metalog"]["method"].value
    with open(path, "wb") as f:
        tomli_w.dump(d, f)


# ==============================================================================
# Run full pipeline
# ==============================================================================
def run_pipeline(config: ModelConfig = ModelConfig()):
    """
    Run the health adjusted ages pipeline.

    Args:
        config: ModelConfig dataclass instance containing all parameters.

    Returns:
        Tuple of outputs
    """
    validate_config(config)
    save_run_config(config, config.paths.data_dir / "run_config_log.toml")

    fit_all_metalogs(
        mixture_path=config.paths.mix_dist_path,
        out_dir=config.paths.fitted_dir,
        metalog_config=config.metalog,
    )

    if config.run_qa:
        run_qa(paths=config.paths)

    ex_df = filter_ex_data(path=config.paths.ex_data_path, model_config=config)

    summary, delta_dfle_per_ly_samples = calculate_model_inputs(
        ex_df=ex_df, paths=config.paths, model_config=config
    )

    summary.to_csv(config.paths.delta_dfle_per_ly_summary_path, index=False)
    save_samples_to_parquet(
        delta_dfle_per_ly_samples, config.paths.delta_dfle_per_ly_samples_path
    )

    ex_df_all_ages = filter_ex_data_all_ages(
        path=config.paths.ex_data_path, model_config=config
    )

    haa_df, haa_samples = calculate_hsa_ages(
        ex_df_all_ages=ex_df_all_ages,
        delta_dfle_per_ly_samples=delta_dfle_per_ly_samples,
        model_config=config,
    )

    haa_df.to_csv(config.paths.haa_summary_path, index=False)
    save_hsa_age_samples_to_parquet(haa_samples, config.paths.haa_samples_path)

    return haa_df, haa_samples
