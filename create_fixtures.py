"""
Script to generate minimal fixture data for testing.
Run once from the project root: python create_fixtures.py
Outputs to tests/fixtures/
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path("tests/fixtures")
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ==============================================================================
# life_tables_fixture.csv
# Covers: base_year=2021, target_years=(2035, 2040), sexes=(f, m), ages 55-70
# Mirrors real structure: base, type, id, sex, year, age, ex
# ex values are plausible but synthetic
# ==============================================================================
rows = []
years = [2021, 2035, 2040]
sexes = ["f", "m"]
ages = range(55, 71)  # 55 to 70 inclusive

# Rough baseline ex values at age 55 (female slightly higher)
BASE_EX = {"f": 30.0, "m": 26.0}
# ex declines ~0.9 per year of age, increases ~0.1 per calendar year
for year in years:
    for sex in sexes:
        for age in ages:
            year_offset = (year - 2021) * 0.1
            age_offset = (age - 55) * -0.9
            noise = RNG.normal(0, 0.05)
            ex = BASE_EX[sex] + year_offset + age_offset + noise
            rows.append(
                {
                    "base": "2022b",
                    "type": "period",
                    "id": "ppp",
                    "sex": sex,
                    "year": year,
                    "age": age,
                    "ex": round(ex, 4),
                }
            )

life_tables = pd.DataFrame(rows)
life_tables_path = FIXTURE_DIR / "life_tables.csv"
life_tables.to_csv(life_tables_path, index=False)
print(f"✓ Created {life_tables_path} ({len(life_tables)} rows)")

# ==============================================================================
# mixtures_fixture.parquet
# Covers: target_years=(2035, 2040), sexes=(f, m) — no baseline 2021 row
# mix_vals is a list of 200 floats per row (real data has ~1000, 200 is enough)
# Values represent % of LE spent in good health — plausible range ~60-90
# ==============================================================================
mixture_rows = []
target_years = [2035, 2040]

for year in target_years:
    for sex in sexes:
        centre = 75.0 if sex == "f" else 72.0
        centre += (year - 2035) * 0.5  # slight improvement over time
        mix_vals = RNG.normal(loc=centre, scale=4.0, size=200).tolist()
        # clip to plausible bounds
        mix_vals = [min(max(v, 50.0), 95.0) for v in mix_vals]
        mixture_rows.append(
            {
                "year": np.int32(year),
                "sex": sex,
                "mix_vals": mix_vals,
            }
        )

mixtures = pd.DataFrame(mixture_rows)
mixtures_path = FIXTURE_DIR / "mixtures.parquet"
mixtures.to_parquet(mixtures_path, index=False)
print(f"✓ Created {mixtures_path} ({len(mixtures)} rows)")

print("\nFixture summary:")
print(
    f"  life_tables: {len(life_tables)} rows, years={years}, sexes={sexes}, ages=55-70"
)
print(
    f"  mixtures:    {len(mixtures)} rows, years={target_years}, sexes={sexes}, 200 mix_vals each"
)
print("\nUse in tests via: Path('tests/fixtures/')")
