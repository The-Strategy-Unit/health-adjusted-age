from dataclasses import dataclass, replace
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

    def __post_init__(self):
        # Create output directories if they don't exist
        self.fitted_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
# important! config can't be mutated at runtime, so the hash saved at the end of act 1
# is guaranteed to represent what was actually used for fitting
class MetalogConfig:
    boundedness: MetalogBoundedness = MetalogBoundedness.BOUNDED
    lower_bound: float = 0.0
    upper_bound: float = 100.0
    method: MetalogFitMethod = MetalogFitMethod.OLS
    num_terms: int = 9


@dataclass(frozen=True)
class FittingConfig:
    run_qa: bool = False  # Whether to run QA checks on fitted distributions
    metalog: MetalogConfig = MetalogConfig()


@dataclass(frozen=True)
class SamplingConfig:
    target_years: tuple[int, ...] = (2035,)
    base_year: int = 2021
    hsa_ref_age: int = 65  # Reference age for HAA calculations
    hsa_start_age: int = 55  # Generate HAA for all ages >= hsa_start_age
    n_samples: int = 10_000  # Number of HAA samples to generate for each age/sex/year
    seed: int = 42  # RNG seed for reproducibility
    dfle_f: float = 10.66  # DFLE for females age 65 in base year (2021)
    dfle_m: float = 10.45  # DFLE for males age 65 in base year (2021)

    def __post_init__(self):
        # Validate that target years are >= base year
        if not self.target_years:
            raise ValueError("target_years must not be empty")

        valid_years = range(2022, 2051)
        if invalid := [y for y in self.target_years if y not in valid_years]:
            raise ValueError(f"target_years contains out-of-range values: {invalid}")

        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")


@dataclass(frozen=True)
class ModelConfig:
    paths: PathsConfig = PathsConfig()
    fitting: FittingConfig = FittingConfig()
    sampling: SamplingConfig = SamplingConfig()


# ==============================================================================
# Helpers for updating nested frozen dataclasses
# ==============================================================================
def with_fitting(config: ModelConfig, **kwargs) -> ModelConfig:
    return replace(config, fitting=replace(config.fitting, **kwargs))


def with_sampling(config: ModelConfig, **kwargs) -> ModelConfig:
    return replace(config, sampling=replace(config.sampling, **kwargs))
