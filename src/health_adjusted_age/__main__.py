from health_adjusted_age.config import DEFAULT_CONFIG
from health_adjusted_age.pipeline import run_pipeline


def main():
    run_pipeline(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
