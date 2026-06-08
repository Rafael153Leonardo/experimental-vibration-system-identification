# Data

Use this layout:

```text
data/sample/      Small approved CSV samples for reproducible examples
data/synthetic/   Generated synthetic signals
data/raw/         Local-only raw data, ignored by git
data/private/     Local-only private data, ignored by git
```

The public scripts use `data/sample/sample_vibration_18hz.csv` by default.
Synthetic data remains available for repeatable experiments and comparisons.

`material_trials.csv` records the material context for public samples and for
the original Euler-Bernoulli reference calculation. It deliberately separates
known material labels from incomplete geometry so the project does not classify
an inox run using plastic-ruler dimensions.
