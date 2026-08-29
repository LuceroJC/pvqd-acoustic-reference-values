"""
PVQD healthy-subset reference values -- extended analysis for the AJSLP paper.

This script supersedes `pvqd_reference_bootstrap.py` for manuscript purposes.
It produces identical point estimates and 95% bootstrap CIs for the four
platform-critical values (jitter p90, shimmer p90, HNR p10, GNE p10), and adds
the analyses required for a defensible Research Note:

    1. Coverage extended to F0, jitter, shimmer (%), shimmer (dB), HNR, CPPS, GNE.
       F0 is descriptive only (no percentile-based reference).
    2. Full descriptive statistics per measure: n, mean, SD, median, IQR,
       min-max, and skewness (bias-corrected).
    3. Healthy-boundary sensitivity: percentile-based reference values recomputed
       at CAPE-V < 5, <10, <15 to demonstrate convergence/divergence.
    4. Sex stratification: sex-specific descriptive statistics and percentile
       references, Cohen's d, and Mann-Whitney U.
    5. Demographics of the healthy subset (n, age distribution, sex breakdown).

Outputs
-------
Console: a formatted summary of all analyses.
CSVs (in --output-dir):
    - reference_values_main.csv          (Table 1 in manuscript)
    - reference_values_sex_stratified.csv (Table 2 in manuscript)
    - reference_values_sensitivity.csv    (Table 3 in manuscript)
    - healthy_subset_demographics.csv     (Methods description)

Protocol
--------
- Healthy subset: PVQD speakers with CAPE-V Overall Severity < 10 (primary).
- Pipeline: PhonaLab acoustic engine (Parselmouth 0.4.7, Praat 6.1.38).
- Bootstrap: 2000 resamples with replacement, percentile method for the CI,
  numpy default_rng with fixed seed for reproducibility.
- Sex encoded as 'M' / 'F' after normalization (case-insensitive; also accepts
  'Male' / 'Female').

Inputs
------
Three CSVs, joined on speaker_id extracted from the vowel filename:
    - pvqd_vowel_params.csv    (PhonaLab acoustic parameters, one row per speaker)
    - pvqd_merged_Final.csv    (perceptual ratings; supplies cape_v_severity)
    - Demographics_Final.CSV   (speaker demographics; supplies sex and age)

The demographics file is semicolon-separated and has trailing whitespace in the
header row and the `Participant ID` column; both are handled at load. `Gender`
(M/F, mixed case) is normalized to `sex`, and `Age` is normalized to `age`. The
demographics join is case-insensitive: PVQD IDs differ in case between files
(e.g. vowel-derived 'SJ2003' vs demographics 'Sj2003'), so matching on raw case
silently drops demographics. Merging on an uppercased key recovers all speakers;
any speaker still lacking a demographics match after the join is printed.

If the demographics CSV is not provided, the script falls back to reading
`sex` and `age` from the ratings CSV when those columns are present.

Note on CPPS
------------
CPPS on the platform's Voice Analyzer card uses 13.33 dB (Lucero, 2026), the
discriminative cutoff on the full N = 288 PVQD sample. That is a separate
construct from the healthy-subset p10 computed here and should not be
conflated with it in the manuscript.

Usage
-----
    python pvqd_reference_paper_analysis.py \
        --vowel path/to/pvqd_vowel_params.csv \
        --merged path/to/pvqd_merged_Final.csv \
        --demographics path/to/Demographics_Final.CSV \
        --output-dir path/to/output_folder
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VOWEL_CSV = "pvqd_vowel_params.csv"
DEFAULT_MERGED_CSV = "pvqd_merged_Final.csv"
DEFAULT_DEMOGRAPHICS_CSV = "Demographics_Final.CSV"
DEFAULT_OUTPUT_DIR = "reference_paper_outputs"

CAPE_V_HEALTHY_MAX = 10                     # Primary healthy boundary
CAPE_V_CUTOFFS_SENSITIVITY = [5, 10, 15]    # For boundary sensitivity analysis
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 2000
CI_ALPHA = 0.05                             # -> 2.5 / 97.5 percentile bounds

# Column names in merged CSV -- change if PVQD file uses different labels.
SEX_COL = "sex"
AGE_COL = "age"

# (column, percentile_or_None, human label)
# percentile = None means descriptive-only (no reference floor/ceiling).
MEASURES = [
    ("f0_mean",           None, "F0 (Hz)"),
    ("jitter_local_pct",  90,   "Jitter local (%)"),
    ("shimmer_local_pct", 90,   "Shimmer local (%)"),
    ("shimmer_local_dB",  90,   "Shimmer local (dB)"),
    ("hnr",               10,   "HNR (dB)"),
    ("cpps",              10,   "CPPS (dB)"),
    ("gne",               10,   "GNE"),
]

SPEAKER_ID_RE = re.compile(r"^([A-Z]+\d+)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_speaker_id(filename: str) -> str | None:
    """Pull e.g. 'BL01' from 'BL01_ENSS_vowel_a.wav'."""
    if not isinstance(filename, str):
        return None
    m = SPEAKER_ID_RE.match(filename)
    return m.group(1) if m else None


def normalize_sex(val) -> str | None:
    """Standardize sex encoding to 'M' / 'F' / None."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    return None


def load_demographics(demo_csv: Path) -> pd.DataFrame:
    """Load PVQD demographics CSV and normalize column names.

    Handles the file's quirks: semicolon separator, trailing whitespace in the
    header row, and a trailing space in the `Participant ID` column. Returns a
    DataFrame keyed by `speaker_id` with `sex` and `age` columns as available.
    """
    df = pd.read_csv(demo_csv, sep=";")
    df.columns = [c.strip() for c in df.columns]

    if "Participant ID" not in df.columns:
        raise ValueError(
            f"'Participant ID' column not found in {demo_csv}. "
            f"Columns present: {list(df.columns)}"
        )
    df["Participant ID"] = df["Participant ID"].astype(str).str.strip()

    rename_map = {"Participant ID": "speaker_id"}
    if "Gender" in df.columns:
        rename_map["Gender"] = SEX_COL
    if "Age" in df.columns:
        rename_map["Age"] = AGE_COL
    df = df.rename(columns=rename_map)

    keep = ["speaker_id"] + [c for c in (SEX_COL, AGE_COL) if c in df.columns]
    return df[keep].copy()


def load_joined(vowel_csv: Path,
                merged_csv: Path,
                demographics_csv: Path | None = None) -> pd.DataFrame:
    """Return the full joined DataFrame with cape_v_severity, sex, age columns.

    Demographics (sex, age) are sourced from `demographics_csv` when provided.
    If demographics is not provided, the script falls back to reading `sex` and
    `age` from the merged CSV where those columns are present.
    """
    vowel = pd.read_csv(vowel_csv)
    merged = pd.read_csv(merged_csv)

    vowel["speaker_id"] = vowel["filename"].apply(extract_speaker_id)
    if vowel["speaker_id"].isna().any():
        n_bad = int(vowel["speaker_id"].isna().sum())
        raise ValueError(f"{n_bad} vowel filenames did not match the speaker-ID pattern")

    keep = ["speaker_id", "cape_v_severity"]
    # Only pull sex/age from ratings if we won't be overlaying demographics
    if demographics_csv is None:
        for col in (SEX_COL, AGE_COL):
            if col in merged.columns:
                keep.append(col)

    df = vowel.merge(merged[keep], on="speaker_id", how="left")

    if demographics_csv is not None:
        demo = load_demographics(demographics_csv)
        # Case-insensitive join: PVQD demographics IDs (e.g. 'Sj2003') do not
        # always match the case of the vowel-derived IDs (e.g. 'SJ2003'). Merge
        # on an uppercased key so casing differences don't silently drop
        # demographics, but keep the original speaker_id for display.
        df["_join_key"] = df["speaker_id"].str.upper()
        demo["_join_key"] = demo["speaker_id"].str.upper()
        demo_cols = ["_join_key"] + [c for c in (SEX_COL, AGE_COL) if c in demo.columns]
        df = df.merge(demo[demo_cols], on="_join_key", how="left")
        df = df.drop(columns="_join_key")

        # Report any speakers still without demographics after the join, so a
        # future coverage gap can't hide the way the SJ2003 casing bug did.
        if SEX_COL in df.columns:
            unmatched = sorted(df.loc[df[SEX_COL].isna(), "speaker_id"].unique())
            if unmatched:
                print(f"[demographics] {len(unmatched)} speaker(s) without a "
                      f"demographics match after case-insensitive join: {unmatched}")

    if SEX_COL in df.columns:
        df[SEX_COL] = df[SEX_COL].apply(normalize_sex)

    return df


def filter_healthy(df: pd.DataFrame, cape_v_max: float = CAPE_V_HEALTHY_MAX) -> pd.DataFrame:
    """Return the subset with CAPE-V Overall Severity strictly below `cape_v_max`."""
    return df[df["cape_v_severity"] < cape_v_max].copy()


# ---------------------------------------------------------------------------
# Bootstrap and descriptive statistics
# ---------------------------------------------------------------------------

def bootstrap_percentile(values: np.ndarray,
                         percentile: float,
                         seed: int = BOOTSTRAP_SEED,
                         n_boot: int = BOOTSTRAP_N,
                         alpha: float = CI_ALPHA) -> tuple[float, float, float]:
    """(point_estimate, ci_low, ci_high) for a percentile of `values` via resampling."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = np.percentile(sample, percentile)

    point = float(np.percentile(values, percentile))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def descriptive_stats(values: np.ndarray) -> dict:
    """n, mean, SD, median, Q1, Q3, IQR, min, max, skewness."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, iqr=np.nan,
                    min=np.nan, max=np.nan, skew=np.nan)
    q1, q3 = np.percentile(values, [25, 75])
    return dict(
        n=int(len(values)),
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        median=float(np.median(values)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        min=float(values.min()),
        max=float(values.max()),
        skew=float(stats.skew(values, bias=False)) if len(values) > 2 else np.nan,
    )


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent groups, pooled SD (denominator n1+n2-2)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyze_main(healthy: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics + percentile-with-CI for each measure (pooled)."""
    rows = []
    for col, pct, label in MEASURES:
        if col not in healthy.columns:
            continue
        vals = healthy[col].dropna().to_numpy()
        d = descriptive_stats(vals)
        row = {
            "parameter": label,
            "column": col,
            "n": d["n"],
            "mean": d["mean"],
            "sd": d["sd"],
            "median": d["median"],
            "q1": d["q1"],
            "q3": d["q3"],
            "iqr": d["iqr"],
            "min": d["min"],
            "max": d["max"],
            "skew": d["skew"],
            "percentile": pct,
        }
        if pct is not None and d["n"] > 0:
            point, lo, hi = bootstrap_percentile(vals, pct)
            row.update({"reference_value": point, "ci_low": lo, "ci_high": hi})
        else:
            row.update({"reference_value": np.nan,
                        "ci_low": np.nan, "ci_high": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_sex(healthy: pd.DataFrame) -> pd.DataFrame | None:
    """Sex-stratified descriptive stats, percentile refs, Cohen's d, Mann-Whitney U."""
    if SEX_COL not in healthy.columns:
        return None
    rows = []
    for col, pct, label in MEASURES:
        if col not in healthy.columns:
            continue
        male = healthy.loc[healthy[SEX_COL] == "M", col].dropna().to_numpy()
        female = healthy.loc[healthy[SEX_COL] == "F", col].dropna().to_numpy()
        dm = descriptive_stats(male)
        df_ = descriptive_stats(female)

        d = cohen_d(male, female)
        if len(male) > 0 and len(female) > 0:
            mw = stats.mannwhitneyu(male, female, alternative="two-sided")
            mw_u = float(mw.statistic)
            mw_p = float(mw.pvalue)
        else:
            mw_u = np.nan
            mw_p = np.nan

        row = {
            "parameter": label,
            "column": col,
            "n_M": dm["n"], "mean_M": dm["mean"], "sd_M": dm["sd"],
            "median_M": dm["median"],
            "n_F": df_["n"], "mean_F": df_["mean"], "sd_F": df_["sd"],
            "median_F": df_["median"],
            "cohen_d_M_minus_F": d,
            "mann_whitney_U": mw_u,
            "mann_whitney_p": mw_p,
            "percentile": pct,
        }
        if pct is not None:
            pm, lo_m, hi_m = bootstrap_percentile(male, pct) \
                if len(male) > 0 else (np.nan, np.nan, np.nan)
            pf, lo_f, hi_f = bootstrap_percentile(female, pct) \
                if len(female) > 0 else (np.nan, np.nan, np.nan)
            row.update({
                "reference_value_M": pm, "ci_low_M": lo_m, "ci_high_M": hi_m,
                "reference_value_F": pf, "ci_low_F": lo_f, "ci_high_F": hi_f,
            })
        else:
            row.update({
                "reference_value_M": np.nan, "ci_low_M": np.nan, "ci_high_M": np.nan,
                "reference_value_F": np.nan, "ci_low_F": np.nan, "ci_high_F": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_sensitivity(full_df: pd.DataFrame,
                        cutoffs=CAPE_V_CUTOFFS_SENSITIVITY) -> pd.DataFrame:
    """Percentile-based reference values at multiple CAPE-V healthy cutoffs."""
    rows = []
    for cutoff in cutoffs:
        subset = filter_healthy(full_df, cape_v_max=cutoff)
        n_total = len(subset)
        for col, pct, label in MEASURES:
            if col not in subset.columns or pct is None:
                continue
            vals = subset[col].dropna().to_numpy()
            if len(vals) == 0:
                point, lo, hi = np.nan, np.nan, np.nan
            else:
                point, lo, hi = bootstrap_percentile(vals, pct)
            rows.append({
                "parameter": label,
                "column": col,
                "percentile": pct,
                "cape_v_cutoff": cutoff,
                "n_subset": n_total,
                "n_measure": int(len(vals)),
                "reference_value": point,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def describe_demographics(healthy: pd.DataFrame) -> dict:
    """Age + sex breakdown of the healthy subset."""
    result = {"n_total": int(len(healthy))}
    if AGE_COL in healthy.columns:
        age = healthy[AGE_COL].dropna()
        result.update({
            "n_with_age": int(len(age)),
            "age_mean": float(age.mean()) if len(age) else np.nan,
            "age_sd": float(age.std(ddof=1)) if len(age) > 1 else np.nan,
            "age_min": float(age.min()) if len(age) else np.nan,
            "age_max": float(age.max()) if len(age) else np.nan,
        })
    else:
        result["n_with_age"] = 0
    if SEX_COL in healthy.columns:
        counts = healthy[SEX_COL].value_counts(dropna=True)
        result["n_male"] = int(counts.get("M", 0))
        result["n_female"] = int(counts.get("F", 0))
        result["n_sex_missing"] = int(healthy[SEX_COL].isna().sum())
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_header(text: str) -> None:
    print(f"\n=== {text} ===\n")


def print_demographics(demo: dict) -> None:
    print_header("Healthy subset demographics")
    print(f"Total speakers: {demo['n_total']}")
    if demo.get("n_with_age", 0) > 0:
        print(f"Age (n={demo['n_with_age']}): "
              f"mean = {demo['age_mean']:.1f}, sd = {demo['age_sd']:.1f}, "
              f"range = {demo['age_min']:.0f}-{demo['age_max']:.0f}")
    else:
        print("Age: not available in merged CSV")
    if "n_male" in demo:
        print(f"Sex: M = {demo['n_male']}, F = {demo['n_female']}, "
              f"missing = {demo.get('n_sex_missing', 0)}")
    else:
        print("Sex: not available in merged CSV")


def print_main(df: pd.DataFrame) -> None:
    print_header(f"Main reference values (healthy subset, CAPE-V < {CAPE_V_HEALTHY_MAX})")
    for _, r in df.iterrows():
        print(f"{r['parameter']:<20}  n = {r['n']:>3}   "
              f"mean = {r['mean']:>8.3f}   sd = {r['sd']:>7.3f}   "
              f"median = {r['median']:>8.3f}   "
              f"IQR = [{r['q1']:.3f}, {r['q3']:.3f}]   "
              f"range = [{r['min']:.3f}, {r['max']:.3f}]   "
              f"skew = {r['skew']:+.2f}")
        if pd.notna(r["reference_value"]):
            print(f"{'':<20}  p{int(r['percentile']):<2}"
                  f" = {r['reference_value']:.3f}   "
                  f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


def print_sex(df: pd.DataFrame | None) -> None:
    if df is None:
        print("\n(sex column not found -- skipping sex-stratified analysis)")
        return
    print_header("Sex-stratified analysis")
    for _, r in df.iterrows():
        d = r["cohen_d_M_minus_F"]
        p = r["mann_whitney_p"]
        d_str = f"{d:+.2f}" if pd.notna(d) else "NA"
        p_str = f"{p:.3g}" if pd.notna(p) else "NA"
        print(f"{r['parameter']:<20}  "
              f"M (n={r['n_M']:>2}): {r['mean_M']:>8.3f} ± {r['sd_M']:>6.3f}    "
              f"F (n={r['n_F']:>2}): {r['mean_F']:>8.3f} ± {r['sd_F']:>6.3f}    "
              f"d = {d_str}   MW p = {p_str}")
        if pd.notna(r.get("reference_value_M")):
            print(f"{'':<20}  p{int(r['percentile']):<2}  "
                  f"M: {r['reference_value_M']:.3f} "
                  f"[{r['ci_low_M']:.3f}, {r['ci_high_M']:.3f}]   "
                  f"F: {r['reference_value_F']:.3f} "
                  f"[{r['ci_low_F']:.3f}, {r['ci_high_F']:.3f}]")


def print_sensitivity(df: pd.DataFrame) -> None:
    print_header("Healthy-boundary sensitivity (percentile at each CAPE-V cutoff)")
    for col in df["column"].unique():
        sub = df[df["column"] == col]
        label = sub.iloc[0]["parameter"]
        pct = int(sub.iloc[0]["percentile"])
        print(f"{label} (p{pct}):")
        for _, r in sub.iterrows():
            print(f"  CAPE-V < {int(r['cape_v_cutoff']):>2}   "
                  f"n = {int(r['n_measure']):>3}   "
                  f"{r['reference_value']:.3f}   "
                  f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PVQD healthy-subset reference-value analyses for the AJSLP paper."
    )
    parser.add_argument("--vowel", default=DEFAULT_VOWEL_CSV,
                        help=f"PhonaLab vowel parameters CSV (default: {DEFAULT_VOWEL_CSV})")
    parser.add_argument("--merged", default=DEFAULT_MERGED_CSV,
                        help=f"PVQD merged ratings CSV (default: {DEFAULT_MERGED_CSV})")
    parser.add_argument("--demographics", default=DEFAULT_DEMOGRAPHICS_CSV,
                        help=f"PVQD demographics CSV (default: {DEFAULT_DEMOGRAPHICS_CSV}). "
                             f"Pass an empty string to disable and fall back to "
                             f"reading sex/age from --merged.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for CSV outputs (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    demo_path = Path(args.demographics) if args.demographics else None
    full = load_joined(Path(args.vowel), Path(args.merged), demo_path)
    healthy = filter_healthy(full, cape_v_max=CAPE_V_HEALTHY_MAX)

    demo = describe_demographics(healthy)
    main_df = analyze_main(healthy)
    sex_df = analyze_sex(healthy)
    sens_df = analyze_sensitivity(full)

    print_demographics(demo)
    print_main(main_df)
    print_sex(sex_df)
    print_sensitivity(sens_df)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_df.to_csv(out / "reference_values_main.csv", index=False)
    if sex_df is not None:
        sex_df.to_csv(out / "reference_values_sex_stratified.csv", index=False)
    sens_df.to_csv(out / "reference_values_sensitivity.csv", index=False)
    pd.DataFrame([demo]).to_csv(out / "healthy_subset_demographics.csv", index=False)

    print(f"\nCSV outputs written to: {out.resolve()}")
    print("\nNote: CPPS reference here is the healthy-subset p10, a descriptive "
          "typical-range floor. It is a separate construct from the discriminative "
          "dysphonia cutoff (13.33 dB, Lucero 2026) computed on the full PVQD sample "
          "and should be reported as such in the manuscript.")


if __name__ == "__main__":
    main()