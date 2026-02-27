from dataclasses import dataclass
from pathlib import Path

from metalog_jax.base import MetalogBoundedness, MetalogFitMethod


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(__file__).parent.parent.parent
    # data dirs
    raw_data_dir: Path = project_root / "data_raw"
    data_dir: Path = project_root / "data"
    # output dirs
    fitted_dir = data_dir / "fitted_dist"
    qa_dir = fitted_dir / "qa"
    # input files
    ex_data_path: Path = raw_data_dir / "life_tables_2022b.csv"
    mix_dist_path: Path = raw_data_dir / "mixtures.parquet"
    # output files
    delta_dfle_per_ly_summary_path = data_dir / "haa_inputs_summary.csv"
    delta_dfle_per_ly_samples_path = data_dir / "haa_inputs_samples.parquet"
    haa_summary_path = data_dir / "haa_summary.csv"
    haa_samples_path = data_dir / "haa_samples.parquet"


@dataclass(frozen=True)
class MetalogConfig:
    boundedness: MetalogBoundedness = MetalogBoundedness.BOUNDED
    lower_bound: float = 0.0
    upper_bound: float = 100.0
    method: MetalogFitMethod = MetalogFitMethod.OLS
    num_terms: int = 9


@dataclass(frozen=True)
class ModelConfig:
    target_years: tuple[int, ...] = (2035,)
    base_year: int = 2021
    hsa_ref_age: int = 65  # Reference age for HAA calculations
    hsa_start_age: int = 55  # Generate HAA for all ages >= hsa_start_age
    n_samples: int = 10000  # Number of HAA samples to generate for each age/sex/year combination  # noqa: E501
    seed: int = 42  # RNG seed for reproducibility
    dfle_f: float = 10.66  # DFLE for females age 65 in base year (2021)
    dfle_m: float = 10.45  # DFLE for males age 65 in base year (2021)
    run_qa: bool = False  # Whether to run QA checks on fitted distributions
    paths: PathsConfig = PathsConfig()
    metalog: MetalogConfig = MetalogConfig()


DEFAULT_CONFIG = ModelConfig()
