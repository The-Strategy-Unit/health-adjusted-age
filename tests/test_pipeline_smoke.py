from dataclasses import replace

from health_adjusted_age import DEFAULT_CONFIG, run_pipeline


def test_pipeline_smoke(tmp_path):
    config = replace(
        DEFAULT_CONFIG,
        paths=replace(
            DEFAULT_CONFIG.paths,
            data_dir=tmp_path,
        ),
        target_years=(2035,),
        n_samples=10,
    )

    summary, samples = run_pipeline(config)

    assert not summary.empty
