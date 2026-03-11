import numpy as np
import pytest

from testing_ground.det_bridge import (
    apply_deterministic_bridge_absolute,
    compute_deterministic_bridge,
)

# ==============================================================================
# Fixtures
# ==============================================================================
BASELINE_YEAR = 2025
SEXES = ["f", "m"]
AGES = [55, 65, 75, 85]
YEARS = [2025, 2035, 2045]


@pytest.fixture
def haa_samples():
    """Synthetic HAA samples keyed by (year, sex, age). N(age, 5) per cell."""
    rng = np.random.default_rng(42)
    return {
        (year, sex, age): rng.normal(loc=age, scale=5.0, size=1_000)
        for year in YEARS
        for sex in SEXES
        for age in AGES
    }


@pytest.fixture
def bridge(haa_samples):
    """Bridge computed from synthetic samples."""
    return compute_deterministic_bridge(haa_samples, BASELINE_YEAR, SEXES, AGES)


@pytest.fixture
def rebased_haa(haa_samples, bridge):
    """Rebased HAA computed from synthetic samples and bridge."""
    return apply_deterministic_bridge_absolute(haa_samples, bridge, BASELINE_YEAR)


# ==============================================================================
# Bridge tests
# ==============================================================================
class TestComputeDeterministicBridge:
    def test_keys_are_sex_age_tuples(self, bridge):
        """Bridge should be keyed by (sex, age) only — no year."""
        for key in bridge.keys():
            assert len(key) == 2
            sex, age = key
            assert sex in SEXES
            assert age in AGES

    def test_bridge_values_are_scalars(self, bridge):
        """Each bridge value should be a plain Python float."""
        for val in bridge.values():
            assert isinstance(val, float)

    def test_bridge_value_close_to_age(self, bridge):
        """
        With samples drawn from N(age, 5), mean HAA at baseline should be
        close to chronological age within a reasonable tolerance.
        """
        for (sex, age), val in bridge.items():
            assert abs(val - age) < 1.0, (
                f"Bridge value {val:.3f} unexpectedly far from age {age} for sex={sex}"
            )

    def test_all_sex_age_combinations_present(self, bridge):
        """Bridge should contain an entry for every (sex, age) combination."""
        for sex in SEXES:
            for age in AGES:
                assert (sex, age) in bridge, f"Missing bridge key: ({sex}, {age})"

    def test_missing_key_excluded(self):
        """Keys absent from haa_samples should be silently excluded from bridge."""
        sparse = {(BASELINE_YEAR, "f", 65): np.ones(100)}
        bridge = compute_deterministic_bridge(sparse, BASELINE_YEAR, SEXES, AGES)
        assert list(bridge.keys()) == [("f", 65)]


# ==============================================================================
# Rebase tests
# ==============================================================================
class TestApplyDeterministicBridgeAbsolute:
    @pytest.mark.parametrize(
        "sex,age,year",
        [
            ("f", 65, 2025),
            ("f", 65, 2045),
            ("m", 75, 2035),
            ("m", 85, 2045),
            ("f", 55, 2025),
        ],
    )
    def test_rebase_identity(self, haa_samples, rebased_haa, bridge, sex, age, year):
        """
        Core identity: mean(original) - mean(rebased) == bridge(sex,age) - age.
        Holds for all (year, sex, age) combinations at or after baseline.
        """
        mean_original = np.mean(haa_samples[(year, sex, age)])
        mean_rebased = np.mean(rebased_haa[(year, sex, age)])
        bridge_scalar = bridge[(sex, age)]

        np.testing.assert_almost_equal(
            mean_original - mean_rebased,
            bridge_scalar - age,
            decimal=6,
            err_msg=f"Rebase identity failed for sex={sex}, age={age}, year={year}",
        )

    def test_pre_baseline_years_excluded(self, rebased_haa):
        """No entries before baseline_year should appear in rebased_haa."""
        for year, sex, age in rebased_haa.keys():
            assert year >= BASELINE_YEAR, (
                f"Pre-baseline entry found: ({year}, {sex}, {age})"
            )

    def test_output_shape_matches_input(self, haa_samples, rebased_haa):
        """Each rebased array should have the same length as the original."""
        for key, samples in rebased_haa.items():
            assert len(samples) == len(haa_samples[key])

    def test_missing_bridge_key_excluded(self, haa_samples):
        """Entries with no matching bridge key should be silently dropped."""
        partial_bridge = {("f", 65): 65.0}
        rebased = apply_deterministic_bridge_absolute(
            haa_samples, partial_bridge, BASELINE_YEAR
        )
        for year, sex, age in rebased.keys():
            assert (sex, age) in partial_bridge
