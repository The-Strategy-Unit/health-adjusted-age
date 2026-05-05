import numpy as np
import pandas as pd
import pytest

from health_adjusted_age.rebase import (
    compute_baseline_haa_means,
    compute_haa_summary,
    rebase_haa_distributions,
)

# ==============================================================================
# Fixtures
# ==============================================================================
REBASE_YEAR = 2025
SEXES = ["f", "m"]
AGES = [55, 65, 75]
YEARS = [2025, 2045]


@pytest.fixture
def haa_samples():
    """Synthetic HAA samples keyed by (year, sex, age). N(age, 2) per cell."""
    rng = np.random.default_rng(42)
    return {
        (year, sex, age): rng.normal(loc=age, scale=2.0, size=1_000)
        for year in YEARS
        for sex in SEXES
        for age in AGES
    }


@pytest.fixture
def baseline_means(haa_samples):
    """Baseline means computed from synthetic samples."""
    return compute_baseline_haa_means(REBASE_YEAR, haa_samples)


@pytest.fixture
def rebased_samples(haa_samples, baseline_means):
    """Rebased samples computed from synthetic samples and baseline means."""
    return rebase_haa_distributions(baseline_means, haa_samples, REBASE_YEAR)


# ==============================================================================
# compute_baseline_haa_means
# ==============================================================================
class TestComputeBaselineHaaMeans:
    # verify keyed by (sex, age) not (year, sex, age)
    def test_keys_are_sex_age_tuples(self, baseline_means):
        for key in baseline_means:
            assert len(key) == 2
            sex, age = key
            assert sex in SEXES
            assert age in AGES

    # verify all (sex, age) combinations are present
    def test_all_sex_age_combinations_present(self, baseline_means):
        for sex in SEXES:
            for age in AGES:
                assert (sex, age) in baseline_means

    # samples are drawn from N(age, 2). Therefore mean should be close to
    # chronological age.
    def test_mean_values_close_to_age(self, baseline_means):
        """With samples drawn from N(age, 2), means should be close to age."""
        for (sex, age), mean in baseline_means.items():
            assert abs(mean - age) < 0.5, (
                f"Mean {mean:.3f} unexpectedly far from age {age} for sex={sex}"
            )

    # verifies the means come specifically from REBASE_YEAR (2025) rows,
    # not from any other year
    def test_ignores_non_rebase_years(self, haa_samples):
        """Only the rebase year should contribute to baseline means."""
        means = compute_baseline_haa_means(REBASE_YEAR, haa_samples)
        # 2045 samples have loc=age too but means should match 2025 not 2045
        for (sex, age), mean in means.items():
            expected = np.mean(haa_samples[(REBASE_YEAR, sex, age)])
            np.testing.assert_almost_equal(mean, expected, decimal=6)

    # passes a year (9999) that doesn't exist in the samples dict and
    # checks the function returns {}
    def test_returns_empty_dict_for_missing_year(self, haa_samples):
        """Should return empty dict if rebase year not in samples."""
        result = compute_baseline_haa_means(9999, haa_samples)
        assert result == {}


# ==============================================================================
# rebase_haa_distributions
# ==============================================================================
class TestRebaseHaaDistributions:
    # verifies that at the rebase year the shift cancels perfectly
    def test_rebase_year_mean_equals_chronological_age(
        self, rebased_samples, baseline_means
    ):
        """At rebase year, mean rebased HAA should equal chronological age."""
        for sex in SEXES:
            for age in AGES:
                mean_rebased = np.mean(rebased_samples[(REBASE_YEAR, sex, age)])
                np.testing.assert_almost_equal(mean_rebased, age, decimal=6)

    # checks that any year before REBASE_YEAR is dropped from the output
    def test_years_before_rebase_excluded(self, rebased_samples):
        """No entries before rebase year should appear in output."""
        for year, sex, age in rebased_samples:
            assert year >= REBASE_YEAR

    # verifies each rebased array has the same number of samples as the original
    def test_output_shape_matches_input(self, haa_samples, rebased_samples):
        """Each rebased array should have same length as original."""
        for key, samples in rebased_samples.items():
            assert len(samples) == len(haa_samples[key])

    # verifies the function silently drops rather than errors on missing keys.
    def test_missing_baseline_key_excluded(self, haa_samples):
        """Entries with no matching baseline key should be silently dropped."""
        partial_means = {("f", 65): 65.0}
        rebased = rebase_haa_distributions(partial_means, haa_samples, REBASE_YEAR)
        for year, sex, age in rebased:
            assert (sex, age) in partial_means

    # for every (year, sex, age), the difference between original and rebased means
    # should equal baseline_mean - age
    def test_shift_is_consistent_across_years(self, haa_samples, baseline_means):
        """
        Core identity: mean(original) - mean(rebased) == baseline_mean - age.
        Should hold for all (year, sex, age) combinations.
        """
        rebased = rebase_haa_distributions(baseline_means, haa_samples, REBASE_YEAR)
        for (year, sex, age), samples in rebased.items():
            mean_original = np.mean(haa_samples[(year, sex, age)])
            mean_rebased = np.mean(samples)
            expected_shift = baseline_means[(sex, age)] - age
            np.testing.assert_almost_equal(
                mean_original - mean_rebased,
                expected_shift,
                decimal=6,
                err_msg=f"Shift identity failed for year={year}, sex={sex}, age={age}",
            )


# ==============================================================================
# compute_haa_summary
# ==============================================================================
class TestComputeHaaSummary:
    # confirms the function returns a pandas DataFrame not a dict or list
    def test_returns_dataframe(self, haa_samples):
        result = compute_haa_summary(haa_samples)
        assert isinstance(result, pd.DataFrame)

    # checks all nine expected columns exist
    def test_expected_columns_present(self, haa_samples):
        result = compute_haa_summary(haa_samples)
        expected = {
            "year",
            "sex",
            "age",
            "hsa_age_mean",
            "hsa_age_median",
            "hsa_age_std",
            "hsa_age_q25",
            "hsa_age_q75",
            "hsa_age_q05",
            "hsa_age_q95",
        }
        assert expected.issubset(set(result.columns))

    # confirms there's exactly one row per entry in the input dict
    def test_one_row_per_key(self, haa_samples):
        result = compute_haa_summary(haa_samples)
        assert len(result) == len(haa_samples)

    # spot checks the hsa_age_mean column against a direct np.mean calculation
    # for every key
    def test_mean_values_correct(self, haa_samples):
        result = compute_haa_summary(haa_samples)
        for (year, sex, age), samples in haa_samples.items():
            row = result[
                (result["year"] == year)
                & (result["sex"] == sex)
                & (result["age"] == age)
            ]
            assert len(row) == 1
            np.testing.assert_almost_equal(
                row["hsa_age_mean"].iloc[0],
                float(np.mean(samples)),
                decimal=6,
            )

    # edge case check with a single entry dict of constant values (all 63.5)
    def test_single_key(self):
        """Should handle a single key without errors."""
        samples = {(2045, "f", 65): np.ones(100) * 63.5}
        result = compute_haa_summary(samples)
        assert len(result) == 1
        assert result["hsa_age_mean"].iloc[0] == 63.5
