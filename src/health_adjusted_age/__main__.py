from health_adjusted_age.config import DEFAULT_CONFIG
from health_adjusted_age.fitting import fit_all_metalogs


def main():
    cfg = DEFAULT_CONFIG

    fit_all_metalogs(
        mixture_path=cfg.paths.mix_dist_path,
        out_dir=cfg.paths.fitted_dir,
        metalog_config=cfg.metalog,
    )


if __name__ == "__main__":
    main()
