#!/usr/bin/env python3
"""
Supplementary analyses for the AJSLP-26-00521 revision.

Addresses the reviewers' requests:
  R1/R2  - clinical composition of the CAPE-V < 10 subset (diagnosis labels)
  R2     - alternative reference populations: no-diagnosis ("Normal") speakers,
           Normal & CAPE-V < 10, Normal & CAPE-V <= 20; percentile references
           (pooled and by sex) for each, next to the original subset
  R1     - sex effects after adjustment for f0 (and age): OLS on log-scaled
           perturbation measures; partial Spearman within sex
  R1     - age: Spearman rho of each measure with age, pooled and within sex
  R1     - decision rule: reclassification of speakers under pooled vs
           sex-specific limits (within the subset and across the full corpus)
  R1     - precision of tail percentiles at n = 26
  R2     - rater/clinic nesting: between-site differences in CAPE-V severity

Reads the same three CSVs as pvqd_reference_paper_analysis.py and imports its
loading/bootstrap functions, so everything is on the production pipeline.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import pvqd_reference_paper_analysis as base

OUT = Path("revision_outputs"); OUT.mkdir(exist_ok=True)
MEAS = [m for m in base.MEASURES if m[1] is not None]          # six measures
LOGM = {"jitter_local_pct", "shimmer_local_pct", "shimmer_local_dB"}

def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ---------------------------------------------------------------- load -----
full = base.load_joined(Path("pvqd_vowel_params_production.csv"),
                        Path("pvqd_merged_Final.csv"),
                        Path("Demographics_Final.CSV"))
merged = pd.read_csv("pvqd_merged_Final.csv")[["speaker_id", "diagnosis", "has_pathology"]]
full = full.merge(merged, on="speaker_id", how="left")
full["diagnosis"] = full["diagnosis"].fillna("").astype(str).str.strip()
full["site"] = full["speaker_id"].str.extract(r"^([A-Z]+)")[0]
full["dx_group"] = np.where(full["diagnosis"].str.lower() == "normal", "none",
                   np.where(full["diagnosis"] == "", "missing", "diagnosed"))

A = full[full.cape_v_severity < 10].copy()                      # manuscript subset

# ------------------------------------------------ 1. clinical composition --
hr("1. Clinical composition of the CAPE-V < 10 subset (n=%d)" % len(A))
print(A.dx_group.value_counts().to_string())
print("\nBy sex:\n", pd.crosstab(A.sex, A.dx_group).to_string())
print("\nDiagnosis labels among diagnosed members:")
print(A.loc[A.dx_group == "diagnosed", "diagnosis"].value_counts().to_string())
A[["speaker_id", "sex", "age", "cape_v_severity", "diagnosis", "dx_group"]] \
    .sort_values("cape_v_severity").to_csv(OUT / "subset_A_composition.csv", index=False)

normal_all = full[full.dx_group == "none"]
s = normal_all.cape_v_severity.dropna()
print(f"\nSpeakers labelled 'Normal' in PVQD: n={len(normal_all)}; CAPE-V severity "
      f"M={s.mean():.1f} SD={s.std(ddof=1):.1f} Mdn={s.median():.1f} "
      f"min={s.min():.2f} max={s.max():.2f}; n<10 = {(s<10).sum()}, n<=20 = {(s<=20).sum()}")

# -------------------------------------- 2. alternative reference populations
hr("2. Percentile references under alternative reference-population definitions")
subsets = {
    "A: CAPE-V<10 (manuscript)":       A,
    "B: no diagnosis (Normal), any severity": normal_all,
    "C: Normal & CAPE-V<10":           normal_all[normal_all.cape_v_severity < 10],
    "D: Normal & CAPE-V<=20":          normal_all[normal_all.cape_v_severity <= 20],
    "E: CAPE-V<10 & diagnosed":        A[A.dx_group == "diagnosed"],
}
rows = []
for name, df in subsets.items():
    nM, nF = (df.sex == "M").sum(), (df.sex == "F").sum()
    print(f"\n{name}:  n={len(df)} (M={nM}, F={nF})")
    for col, pct, lab in MEAS:
        p, lo, hi = base.bootstrap_percentile(df[col].values, pct)
        line = f"  {lab:20s} p{pct} = {p:7.3f} [{lo:.3f}, {hi:.3f}]"
        rec = dict(subset=name, n=len(df), nM=nM, nF=nF, measure=lab, pct=pct,
                   pooled=p, pooled_lo=lo, pooled_hi=hi)
        for sx in ("M", "F"):
            v = df.loc[df.sex == sx, col].values
            if len(v) >= 10:
                ps, los, his = base.bootstrap_percentile(v, pct)
                line += f"   {sx}: {ps:.3f} [{los:.3f}, {his:.3f}]"
                rec.update({f"{sx}": ps, f"{sx}_lo": los, f"{sx}_hi": his, f"n{sx}": len(v)})
        print(line)
        rows.append(rec)
pd.DataFrame(rows).to_csv(OUT / "alt_population_references.csv", index=False)

# diagnosed vs non-diagnosed inside subset A: do they differ acoustically?
print("\nWithin subset A, diagnosed (n=%d) vs no-diagnosis (n=%d): Mann-Whitney"
      % ((A.dx_group == "diagnosed").sum(), (A.dx_group == "none").sum()))
for col, pct, lab in MEAS:
    a = A.loc[A.dx_group == "diagnosed", col].dropna(); b = A.loc[A.dx_group == "none", col].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = base.cohen_d(a.values, b.values)
    print(f"  {lab:20s} Mdn dx={a.median():.3f}  Mdn none={b.median():.3f}  d={d:+.2f}  p={p:.3f}")

# ------------------------------------------------- 3. f0 / age adjustment --
hr("3. Sex effect before and after adjusting for f0 (and age) — OLS, subset A")
class _OLS:
    """Minimal OLS with coefficient p-values (numpy/scipy only)."""
    def __init__(self, y, X):
        d = pd.concat([y.rename("y"), X], axis=1).dropna()
        Xm = np.column_stack([np.ones(len(d)), d[X.columns].values.astype(float)])
        yv = d["y"].values.astype(float)
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        resid = yv - Xm @ beta
        n, k = Xm.shape
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
        t = beta / se
        pv = 2 * stats.t.sf(np.abs(t), n - k)
        names = ["const"] + list(X.columns)
        self.params = pd.Series(beta, names); self.pvalues = pd.Series(pv, names)
        self.rsquared = 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))
class _SM:
    @staticmethod
    def OLS(y, X, missing=None):
        class F:
            def fit(self): return _OLS(y, X)
        return F()
    @staticmethod
    def add_constant(X): return X
sm = _SM()
A["sexM"] = (A.sex == "M").astype(int)
A["logf0"] = np.log(A.f0_mean)
rows = []
for col, pct, lab in MEAS:
    y = np.log(A[col]) if col in LOGM else A[col]
    ylab = f"log({lab})" if col in LOGM else lab
    out = {"measure": ylab}
    for tag, X in (("sex only", A[["sexM"]]),
                   ("sex + log f0", A[["sexM", "logf0"]]),
                   ("sex + log f0 + age", A[["sexM", "logf0", "age"]])):
        m = sm.OLS(y, sm.add_constant(X), missing="drop").fit()
        b, p = m.params["sexM"], m.pvalues["sexM"]
        out[f"b_sex[{tag}]"] = b; out[f"p_sex[{tag}]"] = p
        if "logf0" in X: out[f"p_f0[{tag}]"] = m.pvalues["logf0"]
        if "age" in X: out[f"p_age[{tag}]"] = m.pvalues["age"]
        out[f"R2[{tag}]"] = m.rsquared
    # standardized sex effect = b / residual-free SD of y  (cf. Cohen's d)
    print(f"  {ylab:26s} b_sex: {out['b_sex[sex only]']:+.3f} (p={out['p_sex[sex only]']:.3g})"
          f"  -> +f0: {out['b_sex[sex + log f0]']:+.3f} (p={out['p_sex[sex + log f0]']:.3g}; "
          f"p_f0={out['p_f0[sex + log f0]']:.3g})"
          f"  -> +f0+age: {out['b_sex[sex + log f0 + age]']:+.3f} (p={out['p_sex[sex + log f0 + age]']:.3g}; "
          f"p_age={out['p_age[sex + log f0 + age]']:.3g})")
    rows.append(out)
pd.DataFrame(rows).to_csv(OUT / "sex_effect_f0_age_adjusted.csv", index=False)
print(f"\n  VIF-style check: corr(sexM, log f0) = {A[['sexM','logf0']].corr().iloc[0,1]:+.2f}")

print("\n  Spearman rho of each measure with f0 WITHIN sex (subset A):")
for col, pct, lab in MEAS:
    s = ""
    for sx in ("M", "F"):
        d = A[A.sex == sx]
        r, p = stats.spearmanr(d.f0_mean, d[col], nan_policy="omit")
        s += f"   {sx}: rho={r:+.2f} (p={p:.3f})"
    print(f"  {lab:20s}{s}")

hr("4. Age: Spearman rho with age (subset A), pooled and within sex")
rows = []
for col, pct, lab in MEAS:
    r, p = stats.spearmanr(A.age, A[col], nan_policy="omit")
    s = f"  {lab:20s} pooled rho={r:+.2f} (p={p:.3f})"
    rec = dict(measure=lab, rho_pooled=r, p_pooled=p)
    for sx in ("M", "F"):
        d = A[A.sex == sx]; rs, ps = stats.spearmanr(d.age, d[col], nan_policy="omit")
        s += f"   {sx}: rho={rs:+.2f} (p={ps:.3f})"; rec[f"rho_{sx}"] = rs; rec[f"p_{sx}"] = ps
    print(s); rows.append(rec)
pd.DataFrame(rows).to_csv(OUT / "age_correlations.csv", index=False)
print("\n  Age distribution by sex:", A.groupby("sex").age.describe()[["count","mean","50%","min","max"]].round(1).to_string())
print("  Age bands:", pd.cut(A.age, [0, 29, 49, 100], labels=["<30", "30-49", "50+"]).value_counts().sort_index().to_dict())

# ---------------------------------------------------- 5. reclassification --
hr("5. Does stratification change interpretation? Pooled vs sex-specific limits")
def limits(df):
    L = {}
    for col, pct, lab in MEAS:
        L[col] = dict(pooled=np.percentile(df[col].dropna(), pct),
                      M=np.percentile(df.loc[df.sex == "M", col].dropna(), pct),
                      F=np.percentile(df.loc[df.sex == "F", col].dropna(), pct), pct=pct)
    return L
L = limits(A)
def flag(df, L, which):
    out = {}
    for col, pct, lab in MEAS:
        lim = L[col]["pooled"] if which == "pooled" else df.sex.map({"M": L[col]["M"], "F": L[col]["F"]})
        out[col] = (df[col] > lim) if pct == 90 else (df[col] < lim)
    return pd.DataFrame(out, index=df.index)
rows = []
for scope, df in (("subset A (n=74)", A), ("full corpus (n=%d)" % len(full), full)):
    fp, fs = flag(df, L, "pooled"), flag(df, L, "sex")
    print(f"\n  {scope}: speakers flagged beyond the limit")
    print(f"  {'measure':20s} {'pooled':>7s} {'sex-sp.':>8s} {'agree':>6s} {'pooled-only':>12s} {'sex-only':>9s}  by sex (pooled-only M/F, sex-only M/F)")
    for col, pct, lab in MEAS:
        a, b = fp[col], fs[col]
        po, so = (a & ~b), (~a & b)
        print(f"  {lab:20s} {a.sum():7d} {b.sum():8d} {(a==b).sum():6d} {po.sum():12d} {so.sum():9d}"
              f"   {po[df.sex=='M'].sum()}/{po[df.sex=='F'].sum()}, {so[df.sex=='M'].sum()}/{so[df.sex=='F'].sum()}")
        rows.append(dict(scope=scope, measure=lab, pooled=a.sum(), sex_specific=b.sum(),
                         discordant=(a != b).sum(), pooled_only=po.sum(), sex_only=so.sum(),
                         pooled_only_M=po[df.sex=='M'].sum(), pooled_only_F=po[df.sex=='F'].sum(),
                         sex_only_M=so[df.sex=='M'].sum(), sex_only_F=so[df.sex=='F'].sum()))
    if scope.startswith("full"):
        # discrimination of dysphonic (CAPE-V >= 10) speakers by either limit
        dys = df.cape_v_severity >= 10
        print(f"\n  Full corpus: proportion of CAPE-V>=10 speakers (n={dys.sum()}) flagged, pooled vs sex-specific")
        for col, pct, lab in MEAS:
            print(f"  {lab:20s} pooled {fp.loc[dys, col].mean():.2f}   sex-specific {fs.loc[dys, col].mean():.2f}")
pd.DataFrame(rows).to_csv(OUT / "reclassification.csv", index=False)
# proportion of the pooled distribution each sex exceeds
print("\n  Share of each sex beyond the POOLED limit (ideal = 0.10 if pooled limit were fair to both):")
fp = flag(A, L, "pooled")
for col, pct, lab in MEAS:
    print(f"  {lab:20s} M: {fp.loc[A.sex=='M', col].mean():.2f}   F: {fp.loc[A.sex=='F', col].mean():.2f}")

# ------------------------------------------------- 6. tail precision n=26 --
hr("6. Precision of tail percentiles at n = 26 vs 48 (bootstrap CI widths, relative to IQR)")
for col, pct, lab in MEAS:
    s = f"  {lab:20s}"
    for sx in ("M", "F"):
        v = A.loc[A.sex == sx, col].dropna().values
        p, lo, hi = base.bootstrap_percentile(v, pct)
        iqr = np.subtract(*np.percentile(v, [75, 25]))
        s += f"   {sx} (n={len(v)}): CI width {hi-lo:.3f} = {(hi-lo)/iqr:.2f} IQR"
    print(s)
print("\n  With n=26, the 10th/90th percentile sits between the 2nd and 3rd most extreme observations;"
      "\n  the estimate rests on ~3 speakers.  Expected SE of a sample quantile (Mosteller/Bahadur) ~ sqrt(p(1-p)/n)/f(x_p).")

# -------------------------------------------------- 7. rater/clinic nesting
hr("7. Site (clinic) as a nesting factor: CAPE-V severity by recording-site prefix")
print(full.groupby("site").cape_v_severity.describe()[["count", "mean", "50%", "min", "max"]].round(1).to_string())
print("\n  Site composition of subset A:", A.site.value_counts().to_dict())
kw = stats.kruskal(*[g.cape_v_severity.dropna() for _, g in full.groupby("site")])
print(f"  Kruskal-Wallis across sites (full corpus): H={kw.statistic:.1f}, p={kw.pvalue:.3g}")
print("\nOutputs written to", OUT.resolve())
