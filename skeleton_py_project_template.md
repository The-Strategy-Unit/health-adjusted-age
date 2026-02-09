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
pytest
```