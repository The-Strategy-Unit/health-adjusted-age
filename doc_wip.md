


health_adjusted_age/          ← package (library)
├─ __init__.py                ← public API
├─ config.py                  ← config + defaults
├─ sampling.py                ← generate samples
├─ fitting.py                 ← metalog / distributions
├─ qa.py                      ← validation / checks
└─ io.py                      ← input/output helpers



## Questions
- non-principal projections (population and life tables)
- new HLE release?
- what if fewer people are getting to 65 healthy?
- what is baseline - do we do anything different - do we always calculate mx as eg 2035 / 2021 (chron age)
 or if activity is eg 2024 should our multiplier be 2035/2024 both haa (i think this)
 - beyond 2035 is it valid?
- add multiplier to pipeline