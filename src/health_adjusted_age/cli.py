import argparse
from dataclasses import replace

from health_adjusted_age.config import ModelConfig
from health_adjusted_age.pipeline import (
    run_haa_sampling,
    run_metalog_fitting,
    run_pipeline,
)


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments that are relevant to the sampling stage."""
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="target_years",
        help="Target year (repeatable, e.g. --year 2035 --year 2040)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples",
    )


def add_fitting_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments that are relevant to the fitting stage."""
    parser.add_argument(
        "--run-qa",
        action="store_true",
        dest="run_qa",
        help="Run QA step after fitting",
    )


def build_config(args: argparse.Namespace) -> ModelConfig:
    """Build a ModelConfig from parsed CLI args, overriding defaults as needed."""
    config = ModelConfig()

    if getattr(args, "target_years", None):
        config = replace(config, target_years=tuple(args.target_years))

    if getattr(args, "n_samples", None):
        config = replace(config, n_samples=args.n_samples)

    if getattr(args, "run_qa", False):
        config = replace(config, run_qa=True)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Health adjusted age pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
subcommands:
  fit     Fit metalogs (run once, or when metalog config changes)
  sample  Run sampling using existing fitted outputs
  run     Run both acts end-to-end (fit + sample)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- fit subcommand --
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit metalogs (act 1 — run once)",
    )
    add_fitting_args(fit_parser)

    # -- sample subcommand --
    sample_parser = subparsers.add_parser(
        "sample",
        help="Run sampling against existing fitted outputs (act 2)",
    )
    add_sampling_args(sample_parser)

    # -- run subcommand (both stages) --
    run_parser = subparsers.add_parser(
        "run",
        help="Run both acts end-to-end",
    )
    add_fitting_args(run_parser)
    add_sampling_args(run_parser)

    args = parser.parse_args()
    config = build_config(args)

    if args.command == "fit":
        run_metalog_fitting(config)
    elif args.command == "sample":
        run_haa_sampling(config)
    elif args.command == "run":
        run_pipeline(config)


if __name__ == "__main__":
    main()
