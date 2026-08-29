import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120,
})
C_HIST="#9ecae1"; C_TAIL="#fc9272"; C_REF="#c0392b"; C_MALE="#2c7fb8"; C_FEMALE="#d95f0e"

vowel=pd.read_csv("pvqd_vowel_params_production.csv")
vowel["speaker_id"]=vowel["filename"].str.extract(r"^([A-Za-z]+\d+)")[0].str.upper()
merged=pd.read_csv("pvqd_merged_Final.csv"); merged["speaker_id"]=merged["speaker_id"].astype(str).str.strip().str.upper()
demo=pd.read_csv("Demographics_Final.CSV",sep=";"); demo.columns=[c.strip() for c in demo.columns]
demo["speaker_id"]=demo["Participant ID"].astype(str).str.strip().str.upper()
def nsex(v):
    if pd.isna(v): return None
    s=str(v).strip().upper(); return "M" if s in ("M","MALE") else "F" if s in ("F","FEMALE") else None
demo["sex"]=demo["Gender"].apply(nsex)
df=(vowel.merge(merged[["speaker_id","cape_v_severity"]],on="speaker_id",how="left")
        .merge(demo[["speaker_id","sex"]],on="speaker_id",how="left"))
h=df[df["cape_v_severity"]<10].copy()

def boot(vals,pct,nb=2000):
    vals=np.asarray(vals,float); vals=vals[~np.isnan(vals)]
    rng=np.random.default_rng(42); n=len(vals)
    b=np.array([np.percentile(rng.choice(vals,n,True),pct) for _ in range(nb)])
    return float(np.percentile(vals,pct)),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))

MEAS=[("jitter_local_pct",90,"upper","Jitter (local)","%"),
      ("shimmer_local_pct",90,"upper","Shimmer (local)","%"),
      ("shimmer_local_dB",90,"upper","Shimmer (local)","dB"),
      ("hnr",10,"lower","HNR","dB"),("cpps",10,"lower","CPPS","dB"),("gne",10,"lower","GNE","")]

# ---- Figure 1 ----
fig,axes=plt.subplots(2,3,figsize=(9.0,5.4))
for ax,(col,pct,tail,lab,unit) in zip(axes.ravel(),MEAS):
    v=h[col].dropna().values; ref,lo,hi=boot(v,pct); sk=stats.skew(v,bias=False)
    n,bins,patches=ax.hist(v,bins=16,color=C_HIST,edgecolor="white",linewidth=0.5)
    bw=bins[1]-bins[0]
    for patch,left in zip(patches,bins[:-1]):
        c=left+bw/2; beyond=(c>=ref) if tail=="upper" else (c<=ref)
        if beyond: patch.set_facecolor(C_TAIL)
    ax.axvline(ref,color=C_REF,lw=1.6,ls="--")
    ax.annotate(f"p{pct} = {ref:.2f}",xy=(ref,ax.get_ylim()[1]*0.92),xytext=(4,0),
                textcoords="offset points",color=C_REF,fontsize=8,ha="left",va="top",fontweight="bold")
    ax.set_xlabel(f"{lab} ({unit})" if unit else lab); ax.set_ylabel("Count")
    ax.text(0.97,0.95,f"skew {sk:+.2f}",transform=ax.transAxes,ha="right",va="top",
            fontsize=7.5,color="#555",style="italic")
fig.suptitle("Distributions of acoustic measures in vocally healthy speakers (n = 74), with percentile reference",fontsize=10.5,y=0.995)
handles=[Patch(facecolor=C_HIST,edgecolor="white",label="Within reference"),
         Patch(facecolor=C_TAIL,edgecolor="white",label="Beyond reference (atypical tail)"),
         Line2D([0],[0],color=C_REF,lw=1.6,ls="--",label="Percentile reference")]
fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.03,1,0.97])
fig.savefig("Figure1_distributions.pdf",bbox_inches="tight")
fig.savefig("Figure1_distributions.png",dpi=300,bbox_inches="tight")
plt.close(fig); print("Fig1 done")

# ---- Figure 2 ----
fig,axes=plt.subplots(2,3,figsize=(9.0,5.4))
for ax,(col,pct,tail,lab,unit) in zip(axes.ravel(),MEAS):
    M=h[h.sex=="M"][col].dropna().values; F=h[h.sex=="F"][col].dropna().values
    rM,loM,hiM=boot(M,pct); rF,loF,hiF=boot(F,pct)
    d=(M.mean()-F.mean())/np.sqrt(((len(M)-1)*M.std(ddof=1)**2+(len(F)-1)*F.std(ddof=1)**2)/(len(M)+len(F)-2))
    ax.errorbar(rF,1.0,xerr=[[rF-loF],[hiF-rF]],fmt="o",color=C_FEMALE,capsize=3,ms=6,lw=1.6)
    ax.errorbar(rM,0.0,xerr=[[rM-loM],[hiM-rM]],fmt="s",color=C_MALE,capsize=3,ms=6,lw=1.6)
    ax.set_yticks([0,1]); ax.set_yticklabels(["M","F"]); ax.set_ylim(-0.6,1.6)
    ax.set_xlabel(f"{lab} ({unit})  \u2014  p{pct}" if unit else f"{lab}  \u2014  p{pct}")
    ax.text(0.97,0.95,f"d = {d:+.2f}",transform=ax.transAxes,ha="right",va="top",fontsize=7.5,color="#555",style="italic")
    ax.tick_params(axis="y",length=0)
fig.suptitle("Sex-stratified percentile references with bootstrap 95% CIs (vocally healthy speakers)",fontsize=10.5,y=0.995)
handles=[Line2D([0],[0],marker="s",color=C_MALE,lw=0,ms=7,label="Male (n = 26)"),
         Line2D([0],[0],marker="o",color=C_FEMALE,lw=0,ms=7,label="Female (n = 48)")]
fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.03,1,0.97])
fig.savefig("Figure2_sex_stratified.pdf",bbox_inches="tight")
fig.savefig("Figure2_sex_stratified.png",dpi=300,bbox_inches="tight")
plt.close(fig); print("Fig2 done")
