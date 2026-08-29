[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22167083.svg)](https://doi.org/10.5281/zenodo.22167083)

# Suplementary materials for:
# Sex-Stratified Percentile Reference Values for Six Common Acoustic Measures of Voice from the Perceptual Voice Qualities Database

Lucero, J. C., submitted to *American Journal of Speech-Language Pathology*.

---

## Overview

These materials reproduce every table and figure in the paper from the publicly
available Perceptual Voice Qualities Database (PVQD; Walden, 2020). The acoustic
parameters were extracted from the PVQD sustained-vowel recordings using the
PhonaLab analysis engine (Parselmouth 0.4.x / Praat 6.1.38).

---

## Files included

| File | Description |
|---|---|
| `pvqd_vowel_params_production.csv` | Acoustic parameters for all 290 PVQD speakers, extracted with the PhonaLab production pipeline. |
| `pvqd_reference_paper_analysis.py` | Computes all reference values, bootstrap CIs, sex-stratified analyses, and sensitivity analyses. Outputs CSV files corresponding to the paper's tables. |
| `make_figures.py` | Generates Figures 1 (distribution panels) and 2 (sex-stratified forest plot). |
| `README.md` | This file. |

---

## Requirements

- Python 3.9+
- numpy
- pandas
- scipy
- matplotlib (for figures only)

---

## Quick start

Place the following files from this deposit and from the PVQD in the same
directory:

- `pvqd_vowel_params_production.csv` (from this deposit)
- `pvqd_merged_Final.csv` (from the PVQD — contains CAPE-V severity ratings)
- `Demographics_Final.CSV` (from the PVQD — contains speaker sex and age)

Then run:

```bash
python pvqd_reference_paper_analysis.py --vowel pvqd_vowel_params_production.csv
```

Output CSV files are written to `reference_paper_outputs/`:

| Output file | Corresponds to |
|---|---|
| `reference_values_main.csv` | Table 1 (pooled descriptives + percentile references) |
| `reference_values_sex_stratified.csv` | Table 2 (sex-stratified references, Cohen's d, Mann–Whitney U) |
| `reference_values_sensitivity.csv` | Table 3 (healthy-boundary sensitivity at CAPE-V < 5, 10, 15) |
| `healthy_subset_demographics.csv` | Demographics of the healthy subset (for the Methods section) |

To generate the figures:

```bash
python make_figures.py
```

This produces `Figure1_distributions.pdf`, `Figure1_distributions.png`,
`Figure2_sex_stratified.pdf`, and `Figure2_sex_stratified.png` in the current
directory. The figure script reads the same three CSV files listed above.

---

## Pipeline specification

The acoustic parameters were extracted with the PhonaLab Voice Analyzer pipeline:

- **Preprocessing:** resample to 22050 Hz if above; silence-trim (intensity
  threshold −30 dB, 50 ms padding); no amplitude normalization.
- **F0:** Praat autocorrelation pitch tracker, ±2.5 SD outlier trim.
- **Jitter / Shimmer (% and dB):** periodic point process at 75–600 Hz; both
  shimmer expressions from the same voicing basis.
- **HNR:** cross-correlation harmonicity.
- **GNE:** band 500–4500 Hz, bandwidth 1000 Hz, step 80 Hz; maximum of the GNE
  object.
- **CPPS:** PowerCepstrogram (60 Hz floor, 0.002 s step, 5000 Hz max, 50 Hz
  pre-emphasis); Get CPPS with settings identical to Murton et al. (2020) /
  Watts et al. (2017). Voiced-segment concatenation without the absolute power
  floor; honest missing (None) on insufficient voiced content.

The healthy subset is defined as CAPE-V overall severity < 10 (n = 74; 26 male,
48 female).

---

## Citing this material

If you use these reference values or analysis code, please cite this deposit:

> Lucero, J. C. (2026). Supplementary materials for acoustic reference values for healthy voices [Data set and analysis code]. Zenodo [DOI]

---

## License

The analysis code is released under the MIT License. The PVQD data are subject to
their original terms (Walden, 2020).
