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
# long-hand
uv run python src/health_adjusted_age/__main__.py
# canonical
python -m health_adjusted_age
# cli command - see [project.scripts] in pyproject.toml
uv run pipeline
# run tests
uv run pytest
```

```python
# run pipeline as a library
from health_adjusted_age import run_pipeline, DEFAULT_CONFIG
run_pipeline(DEFAULT_CONFIG)
```

```python
# changing config options the canonical way
from dataclasses import replace
from health_adjusted_age import run_pipeline, DEFAULT_CONFIG

config = replace(
    DEFAULT_CONFIG,
    target_years=(2035,),
    # target_years=(range(2022, 2051)),
    n_samples=500,
)

summary, samples = run_pipeline(config)
```

```bash
# *NEW* config options via cli - see cli.py
uv run pipeline --year 2030 --n-samples 500 --run-qa
uv run pipeline --year 2025 --year 2030 --year 2035 --n-samples 500
```

```bash
# render nb without code cells
jupyter nbconvert --to html --no-input docs/haa_methods.ipynb --output haa_methods.html
```
