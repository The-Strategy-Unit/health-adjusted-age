from dataclasses import dataclass, replace
from pathlib import Path

from metalog_jax.base import MetalogBoundedness, MetalogFitMethod


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(__file__).parent.parent.parent
    raw_data_dir: Path | None = None
    data_dir: Path | None = None
    fitted_dir: Path | None = None
    qa_dir: Path | None = None
    ex_data_path: Path | None = None
    mix_dist_path: Path | None = None
    delta_dfle_per_ly_summary_path: Path | None = None
    delta_dfle_per_ly_samples_path: Path | None = None
    haa_summary_path: Path | None = None
    haa_samples_path: Path | None = None

    def __post_init__(self):
        raw: Path = self.raw_data_dir or self.project_root / "data_raw"
        data: Path = self.data_dir or self.project_root / "data"
        fitted: Path = data / "fitted_dist"
        qa: Path = fitted / "qa"

        object.__setattr__(self, "raw_data_dir", raw)
        object.__setattr__(self, "data_dir", data)
        object.__setattr__(self, "fitted_dir", fitted)
        object.__setattr__(self, "qa_dir", qa)
        object.__setattr__(self, "ex_data_path", raw / "life_tables_2022b.csv")
        object.__setattr__(self, "mix_dist_path", raw / "mixtures.parquet")
        object.__setattr__(
            self, "delta_dfle_per_ly_summary_path", data / "haa_inputs_summary.csv"
        )
        object.__setattr__(
            self, "delta_dfle_per_ly_samples_path", data / "haa_inputs_samples.parquet"
        )
        object.__setattr__(self, "haa_summary_path", data / "haa_summary.csv")
        object.__setattr__(self, "haa_samples_path", data / "haa_samples.parquet")

        fitted.mkdir(parents=True, exist_ok=True)
        qa.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)


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
    target_years: tuple[int, ...] = (2045,)
    base_year: int = 2021
    rebase_year: int | None = None
    # Year to rebase HAA distributions to (default: None)
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
def with_metalog(config: ModelConfig, **kwargs) -> ModelConfig:
    return with_fitting(config, metalog=replace(config.fitting.metalog, **kwargs))


def with_fitting(config: ModelConfig, **kwargs) -> ModelConfig:
    return replace(config, fitting=replace(config.fitting, **kwargs))


def with_sampling(config: ModelConfig, **kwargs) -> ModelConfig:
    return replace(config, sampling=replace(config.sampling, **kwargs))
