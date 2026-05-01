![status: complete](https://img.shields.io/badge/status-in_progress-yellow) ![Last Commit](https://img.shields.io/github/last-commit/The-Strategy-Unit/health-adjusted-age) ![license: MIT](https://img.shields.io/badge/license-MIT-blue)

# Health Adjusted Age

An analytical pipeline for calculating health-adjusted ages (HAA) from life tables and probabilistic forecasts of health status. Developed by the [Strategy Unit](https://www.strategyunitwm.nhs.uk/).

The pipeline uses [projections of period life expectancy](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/bulletins/pastandprojecteddatafromtheperiodandcohortlifetables/2022baseduk1981to2072) from the Office for National Statistics (ONS) and forecasts of health status as inputs to generate estimates of health-adjusted age. HAA is the age of a person adjusted to reflect their health status; someone in better-than-average health will have a HAA lower than their chronological age, and vice versa.

Outputs include summary statistics and full probabilistic HAA distributions by sex, age, and future year.

The probabilistic forecasts of future health status used in the pipeline were obtained from an expert elicitation exercise in November 2025. For details see, [health status elicitation exercise](https://github.com/The-Strategy-Unit/health-status-ee-exercise).

For more information please contact paulseamer@nhs.net.

## Setup

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) installed

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation
```bash
git clone https://github.com/The-Strategy-Unit/health-adjusted-age.git
cd health-adjusted-age
uv sync --all-extras
```

## Running the pipeline

The pipeline runs in two stages.
1. Stage 1 fits [metalog distributions](https://en.wikipedia.org/wiki/Metalog_distribution) to the input forecasts of health status and only needs to be run once (or when the input data or metalog configuration changes)
2. Stage 2 generates HAA samples and can be re-run as needed with different configurations (e.g., for different years)

### Stage 1: fit metalogs (run once)
```bash
haa fit           # run with default config
haa fit --run-qa  # run and generate QA statistics & diagnostic plots
```

### Stage 2: generate HAA distributions (run as needed)
```bash
haa sample                              # run with default config (year=2035; n-samples = 10_000)
haa sample --year 2035 --year 2040      # multiple years
haa sample --year 2035 --n-samples 500  # fewer samples (faster, for testing)
```

### Rebasing HAA distributions

By default, HAA distributions are anchored to 2021&mdash;the base year for the forecasts of future health status. In this anchor year, HAA equals chronological age by definition.

If you want to anchor HAA to a different year (for example, to reflect a 
more recent starting point for planning purposes), you can rebase the 
distributions using `--rebase-year`. The rebase year must also be passed as 
a `--year` argument so that the pipeline can compute a new baseline:

```bash
# anchor HAA to 2025 - outputs for 2035 only (2025 is dropped after rebasing is performed)
haa sample --rebase-year 2025 --year 2025 --year 2035
```

Note that the rebase year itself is not included in the outputs&mdash;it is used 
internally to compute a new baseline and then dropped.

### Run both stages end-to-end
```bash
haa run --run-qa --year 2035
```

### When to re-run Stage 1
Re-run `haa fit` if:
* Input data changes (`data_raw/mixtures.parquet`)
* Any metalog config parameters change (`MetalogConfig` in `config.py`)

Stage 2 will raise an error if the metalog config has changed since the last fit, preventing silent mismatches between Stage 1 *fit* outputs and sampling config.

## Input data

Place the following files in `data_raw/` before running:

| File | Description |
|---|---|
| `life_tables_2022b.csv` | Life expectancy projections with columns `base, type, id, sex, year, age, ex` |
| `mixtures.parquet` | Health status [mixture distributions](https://en.wikipedia.org/wiki/Mixture_distribution) with columns `year, sex, mix_vals` |

## Configuration

Default configuration is defined in `src/health_adjusted_age/config.py`. The most commonly changed parameters are in `SamplingConfig`.

**Run parameters**&mdash;change these to customise each run:

| Parameter | Default | Description |
|---|---|---|
| `rebase_year` | `None` | Year to rebase HAA distributions to |
| `target_years` | `(2035,)` | Years to generate HAA estimates for |
| `n_samples` | `10_000` | Sample size per age/sex/year combination |

**Fixed parameters**&mdash;do not change unless working on model development.

| Parameter | Default | Description |
|---|---|---|
| `base_year` | `2021` | Anchor year for health status forecasts |
| `hsa_ref_age` | `65` | Reference age for HAA calculations |
| `hsa_start_age` | `55` | Generate HAA for all ages >= hsa_start_age |
| `seed` | `42` | RNG seed for reproducibility |
| `dfle_f` | `10.66` | DFLE for females age 65 in base year (2021) |
| `dfle_m` | `10.45` | DFLE for males age 65 in base year (2021) |

## Outputs

All outputs are written to `data/`:

| File | Description |
|---|---|
| `haa_summary.csv` | HAA distributions summary statistics |
| `haa_samples.parquet` | Full HAA distributions |
| `haa_inputs_summary.csv` | Intermediate distributions summary statistics |
| `haa_inputs_samples.parquet` | Intermediate distributions |
| `fitted_dist/` | Fitted metalog distributions (Stage 1 output) |
| `fitted_dist/QA/` | QA diagnostics for metalog distributions (Stage 1 output) |

## Project structure

```
data/      # pipeline outputs (created on first run, not versioned)
data_raw/  # input data files (not versioned, see Input data section)
docs/      # methods documentation
R/         # ...
src/health_adjusted_age/
    __init__.py    # public API (ModelConfig, run_fitting, run_sampling, run_pipeline)
    __main__.py    # enables python -m health_adjusted_age
    cli.py         # CLI (haa fit / sample / run)
    config.py      # configuration dataclasses
    fitting.py     # metalog fitting
    io_fitting.py  # fitting I/O utilities
    io_sampling.py # sampling I/O utilities
    pipeline.py    # two-stage pipeline entry point
    qa_fitting.py  # QA diagnostic plots
    rebase.py      # rebase HAA distributions
    sampling.py    # HAA distributions
testing_ground/  # sandbox for features under development
tests/
    fixtures/               # minimal synthetic data for testing
    test_hash.py            # config fingerprinting tests
    test_pipeline_smoke.py  # end-to-end pipeline tests
create_fixtures.py  # generates synthetic test fixtures
pyproject.toml      # project metadata and dependencies
uv.lock             # locked dependency versions (do not edit manually)
```

## Development

### Running tests
```bash
uv run pytest -v
```

### Code formatting and linting
```bash
ruff format .
ruff check .
ruff check --fix .
```

### VS Code setup

The `.vscode/` folder contains recommended settings and extensions. Install the recommended extensions when prompted.
