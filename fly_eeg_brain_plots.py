"""Publication figures for a fly_eeg_brain.py run (Okabe-Ito, constrained layout, PDF+PNG).
Usage: python fly_eeg_brain_plots.py OUT_DIR [OUT_DIR ...]   (several dirs -> comparison bars across runs)"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True, "savefig.dpi": 300, "pdf.fonttype": 42})
LABEL = {"noisy": "noisy input", "fir": "FIR + ridge (no brain)", "untrained": "untrained brain\nfixed readout", "r0": "untrained brain\nridge readout",
         "brain": "TRAINED brain\nfixed readout"}
def save(fig, d, n): fig.savefig(d / f"{n}.pdf"); fig.savefig(d / f"{n}.png"); plt.close(fig); print("  fig", d / n)

for d in [Path(s) for s in sys.argv[1:]]:
    P = d / "plots"; P.mkdir(exist_ok=True)
    H = json.load(open(d / "history.json")); S = json.load(open(d / "summary.json")); Z = np.load(d / "results.npz")
    art = S["args"]["artifact"].upper()
    # fig1 training curves
    ps = [h["pass_"] for h in H]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax[0].plot(ps, [h["train_rmse"] for h in H], "o-", ms=3, color=OI[0], label="train"); ax[0].plot(ps, [h["val_rmse"] for h in H], "s-", ms=3, color=OI[1], label="val")
    ax[0].set_xlabel("pass"); ax[0].set_ylabel("RMSE (unit-variance EEG)"); ax[0].set_title("Loss"); ax[0].legend(frameon=False)
    ax[1].plot(ps, [h["val_gain"] for h in H], "o-", ms=3, color=OI[2]); ax[1].axhline(S["untrained"]["snr_gain"], color=OI[3], ls="--", lw=0.8, label="untrained brain")
    if "r0" in S: ax[1].axhline(S["r0"]["snr_gain"], color=OI[0], ls=":", lw=0.8, label="untrained + ridge")
    if "fir" in S: ax[1].axhline(S["fir"]["snr_gain"], color="0.4", ls="-.", lw=0.8, label="FIR")
    ax[1].set_xlabel("pass"); ax[1].set_ylabel("val SNR gain (dB)"); ax[1].set_title("Cleaning learned by the brain"); ax[1].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.12), columnspacing=0.8, fontsize=7)
    ax[2].plot(ps, [h["gain_med"] for h in H], "o-", ms=3, color=OI[0], label="synapse gain (median)")
    ax[2].plot(ps, [h["alpha_med"] for h in H], "s-", ms=3, color=OI[1], label="leak alpha (median)")
    ax[2].plot(ps, [h["in_gain_mean"] for h in H], "^-", ms=3, color=OI[2], label="input gain (mean)")
    ax[2].set_xlabel("pass"); ax[2].set_ylabel("value"); ax[2].set_title("Brain parameters"); ax[2].legend(frameon=False)
    for a in ax: a.grid(alpha=0.25, lw=0.5)
    save(fig, P, "fig1_training")
    # fig2 comparison bars (test SNR gain, mean +- sem) + RRMSE
    keys = [k for k in ("fir", "untrained", "r0", "brain") if f"{k}_snr_gain" in Z]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for j, (met, yl) in enumerate([("snr_gain", "SNR gain (dB)"), ("rrmse", "RRMSE")]):
        v = [Z[f"{k}_{met}"] for k in keys]; m = [t.mean() for t in v]; e = [t.std() / np.sqrt(len(t)) for t in v]
        ax[j].bar(range(len(keys)), m, yerr=e, capsize=3, color=[OI[i] for i in range(len(keys))])
        ax[j].set_xticks(range(len(keys))); ax[j].set_xticklabels([LABEL[k] for k in keys], fontsize=7.5); ax[j].set_ylabel(yl); ax[j].grid(alpha=0.25, lw=0.5, axis="y")
        for i, mm in enumerate(m): ax[j].text(i, mm, f"{mm:+.2f}" if met == "snr_gain" else f"{mm:.3f}", ha="center", va="bottom" if mm >= 0 else "top", fontsize=7.5)
    ax[0].set_title(f"{art}: test SNR gain (n={len(Z['clean'])} epochs)"); ax[1].set_title("test RRMSE (lower is better)")
    save(fig, P, "fig2_comparison")
    # fig3 example epochs
    fs = 256; t = np.arange(Z["clean"].shape[1]) / fs
    snr_in = 20 * np.log10(np.sqrt((Z["clean"] ** 2).mean(1)) / np.sqrt(((Z["noisy"] - Z["clean"]) ** 2).mean(1)))
    picks = [int(np.argsort(snr_in)[q]) for q in (len(snr_in) // 10, len(snr_in) // 2, 9 * len(snr_in) // 10)]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    for a, i in zip(ax, picks):
        a.plot(t, Z["noisy"][i], color="0.7", lw=0.8, label="noisy input"); a.plot(t, Z["clean"][i], color=OI[6], lw=1.0, label="clean EEG (truth)")
        a.plot(t, Z["xhat_brain"][i], color=OI[2], lw=1.0, label="trained brain output")
        a.set_title(f"epoch {i}: input SNR {snr_in[i]:+.1f} dB, brain gain {Z['brain_snr_gain'][i]:+.1f} dB", fontsize=9); a.set_ylabel("amplitude (a.u.)"); a.grid(alpha=0.25, lw=0.5)
    ax[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.15)); ax[-1].set_xlabel("time (s)")
    save(fig, P, "fig3_examples")
    # fig4 per-epoch gain vs input SNR
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k, c in [("brain", OI[2]), ("r0", OI[0]), ("fir", "0.4")]:
        if f"{k}_snr_gain" in Z: ax.scatter(snr_in, Z[f"{k}_snr_gain"], s=5, alpha=0.5, color=c, label=LABEL[k].replace("\n", " "))
    ax.set_xlabel("input SNR (dB)"); ax.set_ylabel("SNR gain (dB)"); ax.set_title(f"{art}: per-epoch gain"); ax.legend(frameon=False, markerscale=2); ax.grid(alpha=0.25, lw=0.5)
    save(fig, P, "fig4_gain_vs_input")
    rows = [f"| {LABEL[k].replace(chr(10), ' ')} | {Z[f'{k}_snr_gain'].mean():+.2f} | {Z[f'{k}_rrmse'].mean():.3f} | {Z[f'{k}_cc'].mean():.3f} |" for k in keys]
    (P / "table.md").write_text(f"# {art} ({S['args']['protocol']}), brain-trained, readout={S['args']['readout']}, shuffle={S['args']['shuffle']}\n\n| method | SNR gain (dB) | RRMSE | CC |\n|---|---|---|---|\n" + "\n".join(rows) + f"\n\nbest pass {S['best_pass']}, {S['n_params']:,} brain parameters, {S['elapsed_s'] / 3600:.2f} h\n")
    print(open(P / "table.md").read())
