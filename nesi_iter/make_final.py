"""Final table + publication plots for the Fly EEG iteration campaign.
Run from nfly:  uv run --no-sync python ../iter/make_final.py
Reads readouts_v2_<tag>.json from every states_* cache and results_*_centered.npz (FIR / v1 baseline),
writes iter/FINAL_TABLE.md and iter/plots/fig*.{pdf,png}."""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nesi/project/aut04653/Manj/Fly")
NB = Path("/nesi/nobackup/aut04653/Manj/Fly")
OUT = ROOT / "iter" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
ARTS = ["eog", "emg", "both"]
LABEL = {"eog": "EOG", "emg": "EMG", "both": "EOG+EMG"}
SPAR = {"eog": 8.28, "emg": 9.05, "both": 8.00}          # Shaikh et al. TNSRE 2026 Table II
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]   # Okabe-Ito
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True, "savefig.dpi": 300, "pdf.fonttype": 42})

def cache_label(d):
    n = d.name.replace("states_", "")
    a = n.split("_")[0]
    aug = re.search(r"aug(\d+)", n); xtr = "xtr" in n
    return a, f"{'all' if xtr else '2400'} rec x{aug.group(1) if aug else 1}"

rows = []                                                  # (artifact, tag, data label, config, metrics)
for d in sorted(NB.glob("states_*_centered*")):
    if "smoke" in d.name: continue
    a, data = cache_label(d)
    for f in sorted(d.glob("readouts_v2_r*.json")):
        res = json.loads(f.read_text())
        for k, r in res.items():
            if isinstance(r, dict) and k != "noisy":
                rows.append((a, f.stem.split("_")[-1], data, k, r))
# v1 baseline (round 0) from the brain-run npz
r0 = {}
for a in ARTS:
    z = np.load(ROOT / "fly-eeg" / f"results_{a}_centered.npz")
    lv = {}
    for name in ("fir", "fly"):
        ds, s0 = z[f"{name}_dsnr_region"], z[f"{name}_snr0"]
        lv[name] = {int(l): float(np.median(ds[s0 == l])) for l in np.unique(s0)}
    r0[a] = lv

# ---- table: best per round per artifact
def rnum(t): return int(re.sub(r"\D", "", t) or 0)
tags = sorted({t for _, t, _, _, _ in rows}, key=lambda t: (rnum(t), t))
lines = ["# Fly EEG cleaner: iteration campaign, final table", "",
         "Score = SPAR-EEG region dSNR (mean over 26 input-SNR levels of per-level medians), dB. RRMSE / CC are whole-epoch means.", "",
         "| round | data | best readout | " + " | ".join(f"{LABEL[a]} dSNR / RRMSE / CC" for a in ARTS) + " |",
         "|---|---|---|" + "---|" * len(ARTS)]
r0m = {}
for a in ARTS:                                             # round-0 readout (v1 mlp 5-shift) from readouts.npz
    z = np.load(NB / f"states_{a}_centered" / "readouts.npz"); s0 = np.load(NB / f"states_{a}_centered" / "te_snr.npy")
    ds = z["mlp_shifts_dsnr_region"]
    r0m[a] = dict(region_dsnr=float(np.mean([np.median(ds[s0 == l]) for l in np.unique(s0)])),
                  rrmse=float(z["mlp_shifts_rrmse"].mean()), cc=float(z["mlp_shifts_cc"].mean()))
lines.append("| r0 | 2400 rec x1 | mlp 5-shift 6k (v1) | " + " | ".join(
    f"{r0m[a]['region_dsnr']:.2f} / {r0m[a]['rrmse']:.3f} / {r0m[a]['cc']:.3f}" for a in ARTS) + " |")
best_final = {}
for t in tags:
    cells, data, cfgs = [], set(), set()
    for a in ARTS:
        cand = [r for r in rows if r[0] == a and r[1] == t]
        if not cand: cells.append("-"); continue
        b = max(cand, key=lambda r: r[4]["region_dsnr"])
        data.add(b[2]); cfgs.add(b[3])
        cells.append(f"{b[4]['region_dsnr']:.2f} / {b[4]['rrmse']:.3f} / {b[4]['cc']:.3f}")
        if a not in best_final or b[4]["region_dsnr"] > best_final[a][4]["region_dsnr"]: best_final[a] = b
    lines.append(f"| {t} | {', '.join(sorted(data))} | {', '.join(sorted(cfgs))} | " + " | ".join(cells) + " |")
lines += ["", "## Final best per artifact", "", "| artifact | round | data | readout | region dSNR | RRMSE | CC | SPAR-EEG | FIR (65 taps) |", "|---|---|---|---|---|---|---|---|---|"]
for a in ARTS:
    b = best_final[a]
    lines.append(f"| {LABEL[a]} | {b[1]} | {b[2]} | {b[3]} | {b[4]['region_dsnr']:.2f} | {b[4]['rrmse']:.3f} | {b[4]['cc']:.3f} | {SPAR[a]:.2f} | {np.mean(list(r0[a]['fir'].values())):.2f} |")
# ---- whole-protocol honesty check (EEGdenoiseNet protocol: artifact over the full epoch, SNR U(-7, 2)); score = whole-epoch SNR gain
wl = ["", "## Honesty check: EEGdenoiseNet whole-epoch protocol (artifact everywhere, no clean context)", "",
      "| artifact | FIR (65 taps) | fly ridge (v1) | ridge5, x4 | tcn d5, x4 | tcn d9, x4 | tcn d9, x12 | tcn d9 w384 60k, x12 (final) |", "|---|---|---|---|---|---|---|---|"]
for a in ARTS:
    r1 = NB / f"states_{a}_whole_aug4" / "readouts_v2_w1.json"; r2 = NB / f"states_{a}_whole_aug12" / "readouts_v2_w2.json"
    if not r1.exists(): wl.append(f"| {LABEL[a]} | pending |  |  |  |  |  |  |"); continue
    res = json.loads(r1.read_text()); res2 = json.loads(r2.read_text()) if r2.exists() else {}
    z = np.load(ROOT / "fly-eeg" / f"results_{a}.npz")
    fmtc = lambda r: f"{r['snr_gain']:+.2f} dB / {r['rrmse']:.3f} / {r['cc']:.3f}"
    cells = [fmtc(res[k]) if k in res else "-" for k in ("ridge5", "tcn+s20000", "tcn+d9+p0.3+s40000")]
    cells += [fmtc(res2[k]) if k in res2 else "pending" for k in ("tcn+d9+p0.3+s40000", "tcn+d9+w384+p0.3+s60000")]
    wl.append(f"| {LABEL[a]} | {z['fir_snr_gain'].mean():+.2f} dB | {z['fly_snr_gain'].mean():+.2f} dB | " + " | ".join(cells) + " |")
wl.append(""); wl.append("Cells: SNR gain / RRMSE / CC (whole epoch). The centered-protocol scores above are larger partly because the artifact occupies only the middle third there, so a whole-epoch readout also uses the clean outer thirds.")
lines += wl
(ROOT / "iter" / "FINAL_TABLE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- fig1: per-level region dSNR, final best vs FIR vs v1 ridge vs SPAR-EEG mean
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, a in zip(axes, ARTS):
    b = best_final[a]
    lv = {int(k): v for k, v in b[4]["per_level"].items()}
    L = sorted(lv)
    ax.plot(L, [r0[a]["fir"][l] for l in L], color=OI[6], lw=1.2, label="linear FIR (no brain)")
    ax.plot(L, [r0[a]["fly"][l] for l in L], color=OI[0], lw=1.2, ls="--", label="fly, ridge readout (v1)")
    ax.plot(L, [lv[l] for l in L], color=OI[1], lw=1.8, label="fly, TCN readout (final)")
    ax.axhline(SPAR[a], color=OI[2], lw=1, ls=":", label="SPAR-EEG (mean over levels)")
    ax.set_title(LABEL[a]); ax.set_xlabel("input SNR (dB)"); ax.set_xticks(range(-20, 6, 5))
axes[0].set_ylabel("artifact-region SNR gain (dB)")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig1_per_level.{ext}")

# ---- fig2: progression over rounds (best per round per artifact)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
for i, a in enumerate(ARTS):
    xs, ys = ["r0"], [r0m[a]["region_dsnr"]]
    for t in tags:
        cand = [r for r in rows if r[0] == a and r[1] == t]
        if cand: xs.append(t); ys.append(max(r[4]["region_dsnr"] for r in cand))
    ax.plot(xs, ys, "o-", color=OI[i], lw=1.4, ms=3.5, label=LABEL[a])
    ax.axhline(SPAR[a], color=OI[i], lw=0.8, ls=":")
ax.set_ylabel("best region dSNR (dB)"); ax.set_xlabel("round"); ax.legend(frameon=False, fontsize=7)
ax.text(0.99, 0.02, "dotted: SPAR-EEG", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig2_progression.{ext}")

# ---- fig3: lever ablation (readout / data / context), eog + emg + both
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
levers = [("ridge 5-shift", "2400 rec x1", "ridge5"), ("MLP 9-shift", "2400 rec x1", "mlp9"), ("TCN d5", "2400 rec x1", "tcn"),
          ("TCN d7", "2400 rec x1", "tcn+d7+p0.3+s20000"), ("TCN d7, x4 data", "2400 rec x4", "tcn+d7+p0.3+s20000"),
          ("TCN d8, x12 data", "2400 rec x12", "tcn+d8+p0.3+s30000"), ("TCN d9, all rec x12", "all rec x12", "tcn+d9+p0.3+s40000")]
for ax, a in zip(axes, ARTS):
    vals = []
    for name, data, cfg in levers:
        c = [r for r in rows if r[0] == a and r[2] == data and r[3] == cfg]
        vals.append(max((r[4]["region_dsnr"] for r in c), default=np.nan))
    ax.barh(range(len(levers)), vals, color=[OI[0]] * 3 + [OI[1]] + [OI[2]] * 2 + [OI[1]])
    ax.set_yticks(range(len(levers))); ax.set_yticklabels([l[0] for l in levers] if ax is axes[0] else [])
    ax.axvline(SPAR[a], color=OI[6], lw=0.8, ls=":"); ax.set_title(LABEL[a]); ax.set_xlabel("region dSNR (dB)")
    ax.invert_yaxis()
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig3_levers.{ext}")
print("plots ->", OUT)
