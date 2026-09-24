"""Trend of the brain-only campaign: val SNR gain vs pass for every arm (history.json), reference lines, and a table
of finished arms in chronological order.  python fly_eeg_brain_trend.py  -> brain/plots_trend/"""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
B = Path("/nesi/project/aut04653/Manj/Fly/brain"); P = B / "plots_trend"; P.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True, "savefig.dpi": 300, "pdf.fonttype": 42})
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#999999"]
REF = {"eog": {"FIR": 8.09, "complex CNN (leak-free)": 16.2, "TCN (no brain)": 17.6}, "emg": {"FIR": 7.47, "complex CNN (leak-free)": 11.4, "TCN (no brain)": 11.6},
       "both": {"FIR": 7.29, "complex CNN (leak-free)": 11.5, "TCN (no brain)": 11.7}}
arms = []
for h in sorted(glob.glob(str(B / "*/history.json")), key=os.path.getmtime):
    d = Path(h).parent; tag = d.name
    if tag.startswith("smoke") or "FAILED" in tag: continue
    H = json.load(open(h)); s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else None
    art = "emg" if tag.startswith("emg") else "both" if tag.startswith("both") else "eog"
    arms.append(dict(tag=tag, art=art, passes=[x["pass_"] for x in H], gain=[x["val_gain"] for x in H], cc=[x["val_cc"] for x in H],
                     test=(s["brain"]["snr_gain"] if s else None), started=os.path.getmtime(h) - sum(x["sec"] for x in H)))
arms.sort(key=lambda a: a["started"])
SHORT = lambda t: t.replace("eog_", "").replace("emg_", "").replace("both_", "").replace("whole_random", "lr1e-3").replace("_biaslow", "").replace("D_", "D ").replace("E_", "E ").replace("F_", "F ")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
for a, art in zip(ax, ("eog", "emg", "both")):
    k = 0
    for arm in arms:
        if arm["art"] != art or (arm["test"] is None and len(arm["passes"]) <= 6) or "smoke" in arm["tag"]: continue   # skip retired/short arms
        a.plot(arm["passes"], arm["gain"], "-", lw=1.2, color=OI[k % len(OI)], label=SHORT(arm["tag"]) + (f" [test {arm['test']:+.1f}]" if arm["test"] is not None else " [running]")); k += 1
    for (name, v), ls, dy in zip(REF[art].items(), (":", "--", "-."), (0.2, -0.9, 0.2)): a.axhline(v, color="0.35", ls=ls, lw=0.8); a.text(30, v + dy, name, fontsize=5.5, color="0.35", ha="right")
    a.set_title({"eog": "eye blinks", "emg": "muscle", "both": "mixed"}[art]); a.set_xlabel("pass"); a.grid(alpha=0.25, lw=0.5); a.legend(frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=1)
ax[0].set_ylabel("validation SNR gain (dB)"); ax[0].set_ylim(-4, 19); ax[0].set_xlim(0, 31)
fig.savefig(P / "trend.pdf"); fig.savefig(P / "trend.png"); plt.close(fig)
rows = ["| # | arm | artifact | wiring | best val | test | status |", "|---|---|---|---|---|---|---|"]
for i, arm in enumerate(arms, 1):
    rows.append(f"| {i} | {arm['tag']} | {arm['art']} | {'shuffled' if 'shuffled' in arm['tag'] else 'real'} | {max(arm['gain']):+.2f} @ {arm['passes'][int(np.argmax(arm['gain']))]} | "
                f"{('%+.2f' % arm['test']) if arm['test'] is not None else '-'} | {'done' if arm['test'] is not None else 'running/stopped, pass %d' % arm['passes'][-1]} |")
(P / "trend_table.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
