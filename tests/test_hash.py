import pytest

from health_adjusted_age.config import ModelConfig, with_metalog
from health_adjusted_age.pipeline import (
    _check_metalog_hash,
    _compute_metalog_hash,
    _save_metalog_hash,
)


def test_hash_is_stable():
    config = ModelConfig()
    assert _compute_metalog_hash(config) == _compute_metalog_hash(config)


def test_hash_changes_when_config_changes():
    config1 = ModelConfig()
    config2 = with_metalog(config1, num_terms=7)
    assert _compute_metalog_hash(config1) != _compute_metalog_hash(config2)


def test_hash_roundtrip_passes_with_same_config(tmp_path):
    config = ModelConfig()
    # point fitted_dir at tmp_path so we don't write to real data dir
    config = config.__class__(
        paths=config.paths.__class__(
            project_root=tmp_path,
        ),
        fitting=config.fitting,
        sampling=config.sampling,
    )
    _save_metalog_hash(config)
    _check_metalog_hash(config)  # should not raise


def test_hash_roundtrip_fails_with_changed_config(tmp_path):
    config = ModelConfig()
    config = config.__class__(
        paths=config.paths.__class__(project_root=tmp_path),
        fitting=config.fitting,
        sampling=config.sampling,
    )
    _save_metalog_hash(config)
    config_changed = with_metalog(config, num_terms=7)
    with pytest.raises(ValueError, match="Metalog config has changed"):
        _check_metalog_hash(config_changed)


def test_hash_check_raises_if_no_hash_file(tmp_path):
    config = ModelConfig()
    config = config.__class__(
        paths=config.paths.__class__(project_root=tmp_path),
        fitting=config.fitting,
        sampling=config.sampling,
    )
    with pytest.raises(FileNotFoundError):
        _check_metalog_hash(config)
