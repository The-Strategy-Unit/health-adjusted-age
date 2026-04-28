# Skeleton python project template

HEALTH-ADJUSTED-AGE/  
├── .gitignore              # Python & uv specific ignores  
├── .vscode/  
│   ├── settings.json       # VSCode settings (incl. python interpreter & formatting settings)
│   └── extensions.json     # recommended VSCode extensions  
├── pyproject.toml          # project dependencies & configuration  
├── README.md               # setup instructions  
├── src/  
│   └── health_adjusted_age/  
│       └── __init__.py     # a very simple module  
└── tests/
    └── test_main.py        # example testsource 
 
health-adjusted-age python package now has:
- a **library** (importable, testable)
- a **pipeline** (single entry point)
- a **config system** (immutable, overrideable)
- a **CLI / script surface**

```bash
# install the current project as an editable pkg
# editable pkgs do not need to be reinstalled for changes to their source code to be active
uv pip install -e .
```

```bash
# runs __main__.py directly - works but bypasses the entry point mechanism
uv run python src/health_adjusted_age/__main__.py run
# runs the package as a module - the canonical Python way
uv run python -m health_adjusted_age run
# uses the registered entry point from pyproject.toml - what you'd normally use
uv run haa run
```

```bash
# via CLI (see cli.py)
# inside project environment managed by uv
uv run haa fit --run-qa
# generates log run_config_fitting.toml
# data/fitted_dist/metalogs.json, metalogs_summary.json, metalog_config.hash
# data/fitted_dist/qa/qa_metalog_2022_f.png etc. qa_summary.json
uv run haa sample --year 2035 --n-samples 500
# generates log run_config_sampling.toml
# data/haa_inputs_samples.parquet, haa_inputs_summary.csv
# data/haa_samples.parquet, haa_summary.csv
uv run haa run --year 2035 --run-qa
# all above
# ask for help
uv run haa --help
uv run haa fit --help
uv run haa sample --help
uv run haa run --help
# error handling
uv run haa sample --year 9999 # should hit valid_years validation
uv run haa sample --year 2035 --n-samples 0  # should hit n_samples check
uv run haa sample # runs with defaults
uv run haa # should error: subcommand required
# run tests
uv run pytest
```

```python
# from a Python script - import and call directly
from health_adjusted_age.config import ModelConfig, with_fitting, with_sampling
from health_adjusted_age.pipeline import run_metalog_fitting, run_haa_sampling

config = ModelConfig()
config = with_sampling(config, target_years=(2035, 2040), n_samples=5000)
config = with_fitting(config, run_qa=True)

run_metalog_fitting(config)
haa_df, haa_samples, = run_haa_sampling(config)
haa_df, haa_samples, = run_pipeline(config)
```

```python
# from a Python script - simulate CLI args
from health_adjusted_age.cli import main
import sys

sys.argv = ["haa", "sample", "--year", "2035", "--n-samples", "5000"]
main()
```

```bash
# render nb without code cells
jupyter nbconvert --to html --no-input docs/haa_methods.ipynb --output haa_methods.html
```

```bash
# load packages in dev dependency group (pytest)
uv sync --dev
# run deterministic bridge tests
pytest tests/test_det_bridge.py -v
```
