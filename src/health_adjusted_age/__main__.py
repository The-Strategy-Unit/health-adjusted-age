from health_adjusted_age.config import DEFAULT_CONFIG
from health_adjusted_age.fitting import fit_all_metalogs
from health_adjusted_age.qa_fitting import run_qa
from health_adjusted_age.sampling import (
    calculate_hsa_ages,
    calculate_model_inputs,
    filter_ex_data,
    filter_ex_data_all_ages,
    save_hsa_age_samples_to_parquet,
    save_samples_to_parquet,
)


def main():
    cfg = DEFAULT_CONFIG

    fit_all_metalogs(
        mixture_path=cfg.paths.mix_dist_path,
        out_dir=cfg.paths.fitted_dir,
        metalog_config=cfg.metalog,
    )

    run_qa(paths=cfg.paths)

    ex_df = filter_ex_data(path=cfg.paths.ex_data_path, model_config=cfg)

    summary, delta_dfle_per_ly_samples = calculate_model_inputs(
        ex_df=ex_df, paths=cfg.paths, model_config=cfg
    )

    summary.to_csv(cfg.paths.delta_dfle_per_ly_summary_path, index=False)
    save_samples_to_parquet(
        delta_dfle_per_ly_samples, cfg.paths.delta_dfle_per_ly_samples_path
    )

    ex_df_all_ages = filter_ex_data_all_ages(
        path=cfg.paths.ex_data_path, model_config=cfg
    )

    haa_df, haa_samples = calculate_hsa_ages(
        ex_df_all_ages=ex_df_all_ages,
        delta_dfle_per_ly_samples=delta_dfle_per_ly_samples,
        model_config=cfg,
    )

    haa_df.to_csv(cfg.paths.haa_summary_path, index=False)
    print(f"\n✓ HSA ages saved to: {cfg.paths.haa_summary_path}")

    save_hsa_age_samples_to_parquet(haa_samples, cfg.paths.haa_samples_path)


if __name__ == "__main__":
    main()
