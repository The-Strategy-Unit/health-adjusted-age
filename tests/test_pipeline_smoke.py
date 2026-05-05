from pathlib import Path

import pytest

from health_adjusted_age.config import (
    FittingConfig,
    ModelConfig,
    PathsConfig,
    SamplingConfig,
)
from health_adjusted_age.pipeline import (
    run_haa_sampling,
    run_metalog_fitting,
    run_pipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_config(tmp_path):
    """ModelConfig wired to fixture input data and tmp_path for outputs."""
    return ModelConfig(
        paths=PathsConfig(
            raw_data_dir=FIXTURES,
            data_dir=tmp_path,
        ),
        fitting=FittingConfig(run_qa=False),
        sampling=SamplingConfig(
            target_years=(2045,),
            n_samples=10,
        ),
    )


def test_run_fitting_creates_outputs(fixture_config):
    """Stage 1: fitted outputs should exist after run_metalog_fitting."""
    run_metalog_fitting(fixture_config)

    assert (fixture_config.paths.fitted_dir / "metalogs.json").exists()
    assert (fixture_config.paths.fitted_dir / "metalog_config.hash").exists()


def test_run_sampling_returns_results(fixture_config):
    """Stage 2: should return non-empty dataframe and samples dict."""
    run_metalog_fitting(fixture_config)
    haa_samples = run_haa_sampling(fixture_config)

    assert len(haa_samples) > 0


def test_run_sampling_fails_without_fitting(fixture_config):
    """Stage 2 should raise if stage 1 has not been run."""
    with pytest.raises(FileNotFoundError):
        run_haa_sampling(fixture_config)


def test_run_pipeline_matches_staged_run(fixture_config, tmp_path):
    """Full pipeline output should match running stages separately."""
    # staged run
    run_metalog_fitting(fixture_config)
    haa_df_staged, _ = run_haa_sampling(fixture_config)

    # full pipeline run in a fresh tmp dir
    fresh_config = fixture_config.__class__(
        paths=fixture_config.paths.__class__(
            raw_data_dir=FIXTURES,
            data_dir=tmp_path / "fresh",
        ),
        fitting=fixture_config.fitting,
        sampling=fixture_config.sampling,
    )
    haa_df_pipeline, _ = run_pipeline(fresh_config)

    assert haa_df_staged.shape == haa_df_pipeline.shape
    assert list(haa_df_staged.columns) == list(haa_df_pipeline.columns)
