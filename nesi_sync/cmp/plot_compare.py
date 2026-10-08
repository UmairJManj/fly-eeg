"""Track C figures (Okabe-Ito, constrained layout, PDF+PNG): pooled SNR gain vs channel count per method and artifact,
and alpha-effect preservation. Reads cmp/results/compare.csv."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path("/nesi/project/aut04653/Manj/Fly/cmp/results"); P = R / "plots"; P.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True, "savefig.dpi": 300, "pdf.fonttype": 42})
OI = {"ica_iclabel": "#0072B2", "asr": "#E69F00", "gedai": "#009E73", "tcn_nobrain": "#D55E00", "fly_brain_tcn": "#CC79A7"}
LAB = {"ica_iclabel": "ICA + ICLabel (spatial)", "asr": "ASR (spatial)", "gedai": "GEDAI (spatial)", "tcn_nobrain": "TCN, single-channel (no brain)",
       "fly_brain_tcn": "fly reservoir + TCN, single-channel"}
MK = {"ica_iclabel": "o", "asr": "s", "gedai": "^", "tcn_nobrain": "D", "fly_brain_tcn": "v"}
ART = {"eog": "eye blinks (EOG)", "emg": "muscle (EMG)", "both": "EOG + EMG"}
d = pd.read_csv(R / "compare.csv"); d = d[(d.note.isna()) | (d.note == "")]
g = d.groupby(["artifact", "nch", "method"])[["snr_gain", "cc", "alpha_ratio_err"]].mean().reset_index()
def save(fig, n): fig.savefig(P / f"{n}.pdf"); fig.savefig(P / f"{n}.png"); plt.close(fig); print("  fig", P / n)
for metric, ylab, name, logy in (("snr_gain", "pooled SNR gain (dB)", "figC1_gain_vs_channels", False),
                                  ("alpha_ratio_err", "relative error of EC/EO alpha ratio", "figC2_effect_preservation", True)):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    for a, art in zip(ax, ("eog", "emg", "both")):
        for m in OI:
            s = g[(g.artifact == art) & (g.method == m)].sort_values("nch")
            if len(s): a.plot(s.nch, s[metric], marker=MK[m], ms=4, color=OI[m], label=LAB[m])
        a.set_xscale("log", base=2); a.set_xticks([8, 16, 32, 64]); a.set_xticklabels([8, 16, 32, 64]); a.set_xlabel("channels available")
        a.set_title(ART[art]); a.grid(alpha=0.25, lw=0.5)
        if logy: a.set_yscale("log")
    ax[0].set_ylabel(ylab); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="outside upper center", fontsize=7, columnspacing=1.0)
    for a in ax: a.tick_params(axis="x", labelsize=8)
    save(fig, name)
t = g.pivot_table(index=["artifact", "nch"], columns="method", values="snr_gain").round(2)
(P / "tableC.md").write_text("# Track C: pooled SNR gain (dB), mean over subjects and SNR levels\n\n" + t.to_markdown() + "\n\n" +
    "# alpha-ratio relative error\n\n" + g.pivot_table(index=["artifact", "nch"], columns="method", values="alpha_ratio_err").round(3).to_markdown() + "\n")
print(open(P / "tableC.md").read())
