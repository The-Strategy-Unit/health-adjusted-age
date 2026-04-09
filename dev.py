from dataclasses import replace

from health_adjusted_age import ModelConfig, run_pipeline
from health_adjusted_age.sampling import load_hsa_age_samples_from_parquet

config = replace(
    ModelConfig(),
    target_years=(2035,),
    # target_years=(range(2022, 2051)),
    n_samples=500,
)

summary, samples = run_pipeline(config)

df = load_hsa_age_samples_from_parquet(config.paths.haa_samples_path)
df[2035, "f", 80]
len(df[2035, "f", 80])
