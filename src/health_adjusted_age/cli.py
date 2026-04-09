import argparse
from dataclasses import replace

from health_adjusted_age.config import ModelConfig
from health_adjusted_age.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="target_years",
        help="Target year (repeatable)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples",
    )

    parser.add_argument(
        "--run-qa",
        action="store_true",
        dest="run_qa",
        help="Run QA step",
    )

    args = parser.parse_args()

    config = ModelConfig()

    if args.target_years:
        config = replace(
            config,
            target_years=tuple(args.target_years),
        )

    if args.n_samples:
        config = replace(config, n_samples=args.n_samples)

    if args.run_qa:
        config = replace(config, run_qa=True)

    run_pipeline(config)
