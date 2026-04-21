from health_adjusted_age.config import ModelConfig
from health_adjusted_age.pipeline import run_pipeline


def main():
    run_pipeline(ModelConfig())


if __name__ == "__main__":
    main()
