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

```bash
uv pip install -e .
# long-hand
uv run python src/health_adjusted_age/__main__.py
# canonical
python -m health_adjusted_age
# [project.scripts] table CLI entry point (in pyproject.toml)
uv run pipeline
# tests
uv run pytest
```

health-adjusted-age python package now has:
- a **library** (importable, testable)
- a **pipeline** (single entry point)
- a **config system** (immutable, overrideable)
- a **CLI / script surface**

```bash
# run health-adjusted-age pipeline as a library
from health_adjusted_age import run_pipeline, DEFAULT_CONFIG
run_pipeline(DEFAULT_CONFIG)

# run as a module
python -m health_adjusted_age

# run as CLI command
uv run pipeline
```

```bash
# changing config options the canonical way
from dataclasses import replace
from health_adjusted_age import run_pipeline, DEFAULT_CONFIG

config = replace(
    DEFAULT_CONFIG,
    target_years=(2035,),
    n_samples=500,
)

summary, samples = run_pipeline(config)

# with CLI - NOT IMPLEMENTED YET!
uv run pipeline --n-samples 500
```