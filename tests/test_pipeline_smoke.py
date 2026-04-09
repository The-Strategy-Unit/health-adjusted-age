from dataclasses import replace

from health_adjusted_age import ModelConfig, run_pipeline


def test_pipeline_smoke(tmp_path):
    default_config = ModelConfig()
    config = replace(
        default_config,
        paths=replace(
            default_config.paths,
            data_dir=tmp_path,
        ),
        target_years=(2035,),
        n_samples=10,
    )

    summary, samples = run_pipeline(config)

    assert not summary.empty
