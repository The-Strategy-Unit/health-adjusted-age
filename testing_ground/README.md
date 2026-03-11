## 001 — Deterministic Bridge

**Status**: Complete  
**Date**: 11 March 2026

**Question**: The pipeline produces absolute HAA distributions anchored to 2021 
(most recent observed DFLE). If, for example the downstream NHP model baseline is 2025.
How should HAA distributions be rebased to be consistent with a *new* baseline?

**Approach**: Compute mean HAA at 2025 for each sex/age as a deterministic scalar offset.
Subtract this offset and add back chronological age, rebasing each year's absolute HAA
distribution so that a person's HAA ≈ their chronological age at 
the 2025 baseline.

**Key assumption**: Mean HAA at 2025 is treated as a deterministic anchor —  uncertainty
in the 2021–2025 period is discarded. Defensible given the gap is small relative to the
full projection horizon, and no updated DFLE data is available. Should be revisited when
updated DFLE data becomes available.

**Limitation**: As the model baseline advances, the bridge scalar must be recomputed. This
is a permanent feature of the pipeline given observed DFLE data will likely always lag the
NHP model baseline year.

**Future experiments**:
- 002: Is the deterministic bridge (001) mathematically equivalent to re-anchoring the
  pipeline to a *new* baseline from first principles — estimating DFLE_2025 directly from
  the 2025 metalog and recomputing equations 3 & 4 with a denominator calculated from the
  new baseline?
- 003: Sensitivity to bridge statistic — mean vs median vs mode
- 004: Replace deterministic bridge with sample-wise differencing using correlated 
  trajectories — propagates 2021–2025 uncertainty honestly rather than discarding it
- 005: 2×2 comparison — (independent vs correlated) × (bridge vs sample-wise)


## 002 — Alternative Anchor

**Status**: Complete  
**Date**: 11 March 2026

**Question**: Is the deterministic bridge (001) mathematically equivalent to re-anchoring
the pipeline to a *new* baseline from first principles e.g., estimating DFLE in 2025
directly from the 2025 metalog and recomputing equations 3 & 4 with a denominator
calculated from the new baseline?

**Finding**: Yes - the two methods are numerically equivalent (max abs diff  < 0.02 years,
consistent with Monte Carlo noise). The deterministic bridge (001) is the simpler
implementation of the same operation.

**Implementation note**: Both ex lookups in the alternative anchor must filter by
hsa_ref_age - missing this filter was the source of errors encountered during development.

**Conclusion**: Deterministic bridge (001) is validated and correct. Move to production.
