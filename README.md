[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22167082.svg)](https://doi.org/10.5281/zenodo.22167082)

# Reproducibility materials for:
# Percentile Reference Values for Six Acoustic Measures of Voice From the Perceptual Voice Qualities Database: Sex Differences and Their Dependence on Fundamental Frequency

Lucero, J. C. (2026). *American Journal of Speech-Language Pathology*.
[DOI to be added upon publication]

Concept DOI of this deposit (always resolves to the latest version): https://doi.org/10.5281/zenodo.22167083
This version: [[version DOI]]

---

## Overview

These materials reproduce every table and figure in the paper and its supplemental
materials from the publicly available Perceptual Voice Qualities Database (PVQD;
Walden, 2022). The analysis has two stages:

1. **Acoustic extraction** — compute voice parameters from the PVQD sustained-vowel
   recordings using the PhonaLab analysis engine (Parselmouth 0.4.7 / Praat 6.1.38).
   This step requires a PhonaLab installation. The output is provided as a CSV so
   that most users can skip it.

2. **Reference-value computation** — from the extracted parameters, compute
   descriptive statistics, bootstrap percentile references with 95% CIs, sex
   stratification, boundary sensitivity (Tables 1–2, S2, Figures 1–2), and the
   revision analyses: diagnostic composition of the subset, alternative reference
   populations, f_o/age/diagnosis-adjusted models, and classification under pooled
   versus sex-specific limits (Table 3, S3, S4). This stage requires only Python
   with standard scientific libraries.

---

## Files included

| File | Description |
|---|---|
| `pvqd_vowel_params_production.csv` | Acoustic parameters for all 290 PVQD speakers with a sustained /ɑ/ recording, extracted with the PhonaLab production pipeline (stage 1 output). Provided so stage 2 can run without a PhonaLab installation. |
| `pvqd_reference_paper_analysis.py` | Stage 2, main analysis. Computes the pooled and sex-stratified reference values, bootstrap CIs, and boundary sensitivity. Outputs CSV files corresponding to Tables 1, 2 and S2. |
| `revision_analyses.py` | Stage 2, revision analyses (added in version 2 of this deposit). Computes the clinical composition of the subset, references under alternative population definitions (S3), the sex/f_o/age/diagnosis regressions (Table 3, S4a), age correlations (S4b), classification under pooled vs. sex-specific limits (S4c), and severity by recording site (S4d). Imports loading and bootstrap functions from `pvqd_reference_paper_analysis.py`. |
| `regenerate_pvqd_params_production.py` | Stage 1 script. Extracts acoustic parameters by importing PhonaLab's production analysis functions. Included for full-chain reproducibility; requires a PhonaLab API installation. At startup it verifies the pipeline state it expects (shimmer-dB measurement, silence trimming, and a purely relative voiced-segment gate for CPPS; see *Pipeline specification*). |
| `make_figures.py` | Generates Figures 1 (distribution panels) and 2 (sex-stratified forest plot). |
| `README.md` | This file. |

---

## Requirements

**For stage 2 (most users):**
- Python 3.9+
- numpy, pandas, scipy
- matplotlib (figures only)

**Additionally for stage 1 (full-chain reproduction):**
- praat-parselmouth 0.4.7 (embeds Praat 6.1.38)
- The PhonaLab API source (importable `services/` package)
- The raw PVQD sustained-vowel WAV files

---

## Quick start (stage 2 — reference values from the provided CSV)

Place the following files in one directory:

- `pvqd_vowel_params_production.csv` (from this deposit)
- `pvqd_merged_Final.csv` (from the PVQD — CAPE-V ratings; also supplies the
  `diagnosis` field used by `revision_analyses.py`)
- `Demographics_Final.CSV` (from the PVQD — speaker sex and age)

Then run:

```bash
python pvqd_reference_paper_analysis.py --vowel pvqd_vowel_params_production.csv
python revision_analyses.py
python make_figures.py
```

Outputs of `pvqd_reference_paper_analysis.py` (directory `reference_paper_outputs/`):

| Output file | Corresponds to |
|---|---|
| `reference_values_main.csv` | Table 1 (pooled descriptives and percentile references) |
| `reference_values_sex_stratified.csv` | Table 2 (sex-stratified references, Cohen's d, Mann–Whitney U) |
| `reference_values_sensitivity.csv` | Supplemental S2 (boundary sensitivity at CAPE-V < 5, 10, 15) |
| `healthy_subset_demographics.csv` | Demographics of the reference subset (Method) |

Outputs of `revision_analyses.py` (directory `revision_outputs/`):

| Output file | Corresponds to |
|---|---|
| `subset_A_composition.csv` | Per-speaker severity, sex, age and diagnosis for the reference subset (Table S3a) |
| `alt_population_references.csv` | References under population definitions A–E (Table S3b) |
| `sex_effect_f0_age_adjusted.csv` | Sex coefficients before and after adjustment (Table 3, S4a) |
| `age_correlations.csv` | Spearman correlations with age (Table S4b) |
| `reclassification.csv` | Classification under pooled vs. sex-specific limits (Table S4c) |

The console output of `revision_analyses.py` also prints the diagnosed-vs-non-diagnosed
comparison within the subset, the within-sex correlations with f_o, the
diagnosis-adjusted models, and the severity-by-site summary (Table S4d).

`make_figures.py` writes `Figure1_distributions.{pdf,png}` and
`Figure2_sex_stratified.{pdf,png}`.

---

## Full-chain reproduction (stage 1 — re-extract parameters from raw audio)

```bash
python regenerate_pvqd_params_production.py \
    --api-path /path/to/phonalab/api \
    --input-dir /path/to/PVQD/vowels \
    --output pvqd_vowel_params_production.csv \
    --pattern _vowel_a.wav
```

The script refuses to run against a PhonaLab installation that does not match the
pipeline described in the paper. With the matching installation it reproduces the
deposited CSV exactly.

---

## Pipeline specification

The acoustic parameters were extracted with the PhonaLab Voice Analyzer pipeline;
the complete Praat call syntax is in Supplemental Material S1 of the paper.

- **Preprocessing:** resample to 22050 Hz if above; silence-trim (intensity
  threshold −30 dB, 50 ms padding); no amplitude normalization.
- **f_o:** Praat autocorrelation pitch tracker, 75–600 Hz, ±2.5 SD outlier trim.
- **Jitter / Shimmer (% and dB):** periodic point process at 75–600 Hz; both
  shimmer expressions from the same voicing basis.
- **HNR:** cross-correlation harmonicity.
- **GNE:** band 500–4500 Hz, bandwidth 1000 Hz, step 80 Hz; maximum of the GNE
  object.
- **CPPS:** computed on the voiced portion of the trimmed signal, concatenated from
  30-ms windows whose power in air exceeds 30% of the mean power of the sounding
  intervals (purely relative criterion, after Maryn et al., 2010; no absolute
  power floor); PowerCepstrogram (60 Hz floor, 0.002 s step, 5000 Hz max, 50 Hz
  pre-emphasis); Get CPPS with the settings of Watts et al. (2017) / Murton et al.
  (2020). Missing (NaN) when the voiced portion is under 0.5 s.

The reference subset is the *perceptually near-normal* subset: speakers with mean
CAPE-V overall severity < 10 (n = 74; 26 male, 48 female). Of these, 37 have no
voice-disorder diagnosis in the PVQD demographics, 36 have a clinical diagnosis,
and 1 has no record (see the paper's Method and Supplemental S3). The subset is
defined perceptually and is not a healthy or normative sample.

---

## Version history

- **v1 (2026-08)** — original submission: `pvqd_vowel_params_production.csv`,
  `pvqd_reference_paper_analysis.py`, `regenerate_pvqd_params_production.py`,
  `make_figures.py`.
- **v2 (2026-11)** — revision: adds `revision_analyses.py` and its outputs (Table 3,
  Supplemental S3–S4); `regenerate_pvqd_params_production.py` now verifies the
  CPPS voiced-segment gate at startup and records CPPS as missing when the voiced
  portion is insufficient; README updated to the revised title and terminology.
  The parameter values for all speakers in the reference subset are unchanged
  from v1.

---

## Citing this material

Please cite both the paper and this deposit:

> Lucero, J. C. (2026). Percentile reference values for six acoustic measures of
> voice from the Perceptual Voice Qualities Database: Sex differences and their
> dependence on fundamental frequency. *American Journal of Speech-Language
> Pathology*. [DOI]

> Lucero, J. C. (2026). pvqd-acoustic-reference-values [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.22167083

---

## License

The analysis code is released under the MIT License. The PVQD data are subject to
their original terms (Walden, 2022).
