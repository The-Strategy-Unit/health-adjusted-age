# health-adjusted-age

A Python project managed with [uv](https://github.com/astral-sh/uv).

## Setup
### Prerequisites
* Python 3.9 or higher
* [uv](https://github.com/astral-sh/uv) installed

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initial Setup
1. Clone the repository:

```bash
git clone https://github.com/The-Strategy-Unit/health-adjusted-age.git
cd health-adjusted-age
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
.venv\Scripts\activate  # Windows
# Quick, one-off installations only: uv pip install -e ".[dev]"
uv sync --all-extras  # Creates/updates a uv.lock lockfile 
```

## Usage

### Installing Dependencies

```bash
# Install a package
uv pip install <package-name>

# Add to pyproject.toml dependencies, then:
uv pip install -e .
```

### Running Tests

```bash
pytest
```

### Code Formatting & Linting

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

## Development

This project uses:
- [uv](https://github.com/astral-sh/uv) for fast Python package management
- [pytest](https://docs.pytest.org/en/stable/) for testing
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting

## VSCode Setup

The `.vscode/` folder contains recommended settings and extensions. Install the recommended extensions when prompted.