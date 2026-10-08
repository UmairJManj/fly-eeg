"""Track C comparison on the semi-simulated multichannel benchmark (make_dataset.py).
Methods: none | ICA+ICLabel | ASR | GEDAI  (+ 'fly' if predictions exist at cmp_data/fly_S<s>_<art>_<snr>_<nch>.npy)
Channel counts: 8 / 16 / 32 / 64 (standard 10-20 subsets); each method sees ONLY the subset.
Scores per (subject, artifact, SNR, nch, method): SNR gain, RRMSE, CC averaged over channels, and effect preservation =
relative error of the eyes-closed/eyes-open occipital alpha (8-12 Hz) power ratio vs the clean base.
Writes cmp/results/compare.csv and cmp/results/summary.md.   Usage: python run_compare.py [subjects...]"""
import sys, os, time, warnings, numpy as np, pandas as pd, mne, torch
from mne_icalabel import label_components
from scipy.signal import welch
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
DATA = "/nesi/nobackup/aut04653/Manj/Fly/cmp_data"; RES = "/nesi/project/aut04653/Manj/Fly/cmp/results"; os.makedirs(RES, exist_ok=True)
SUBSETS = {8: ["Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2"],
           16: ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "T7", "T8", "P3", "P4", "Pz", "O1", "O2", "Oz"],
           32: ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
                "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"], 64: None}
OCC = ["O1", "O2", "Oz"]

def make_raw(x, names, fs):
    info = mne.create_info(list(names), fs, "eeg"); raw = mne.io.RawArray(x * 1e-6, info); raw.set_montage("standard_1005"); return raw

def m_ica(x, names, fs):
    raw = make_raw(x, names, fs); n = min(len(names) - 1, 25)
    ica = mne.preprocessing.ICA(n_components=n, method="infomax", fit_params=dict(extended=True), random_state=0, max_iter="auto"); ica.fit(raw)
    lab = label_components(raw, ica, method="iclabel")
    bad = [i for i, (l, p) in enumerate(zip(lab["labels"], lab["y_pred_proba"])) if l not in ("brain", "other") and p > 0.5]
    return ica.apply(raw.copy(), exclude=bad).get_data() * 1e6

def m_asr(x, names, fs):
    from meegkit.asr import ASR                       # asrpy's reconstruction diverges (1e14) on >8 channels; meegkit's port is stable
    asr = ASR(sfreq=fs, cutoff=20, method="euclid"); asr.fit(x); return np.asarray(asr.transform(x), dtype=np.float64)

_REFCOV = {}
def m_gedai(x, names, fs, pos=None):
    import pygedai, io, contextlib
    key = tuple(names)
    if key not in _REFCOV:                                   # template lead-field covariance for THIS electrode set (metres, MNE head coords)
        df = pd.DataFrame({"channel_name": list(names), "X": pos[:, 0], "Y": pos[:, 1], "Z": pos[:, 2]})
        with contextlib.redirect_stdout(io.StringIO()):
            _REFCOV[key] = pygedai.interpolate_ref_cov(df)
    with contextlib.redirect_stdout(io.StringIO()):
        out = pygedai.gedai(torch.tensor(x, dtype=torch.float32), fs, leadfield=_REFCOV[key].to(torch.float32), skip_checks_and_return_cleaned_only=True)
    return np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)

METHODS = {"none": lambda x, n, f: x, "ica_iclabel": m_ica, "asr": m_asr, "gedai": m_gedai}

def alpha_ratio(x, names, run, fs):
    idx = [i for i, n in enumerate(names) if n in OCC]
    def apow(seg):
        f, p = welch(seg[idx], fs, nperseg=2 * fs, axis=-1); return p[:, (f >= 8) & (f <= 12)].mean()
    return apow(x[:, run == 1]) / apow(x[:, run == 0])          # eyes closed / eyes open

def scores(xhat, base, noisy, names, run, fs):
    err = np.sqrt(((xhat - base) ** 2).mean(1)); rr = err / np.sqrt((base ** 2).mean(1))
    cc = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(xhat, base)])
    p_art, p_res, p_sig = ((noisy - base) ** 2).mean(1), err ** 2, (base ** 2).mean(1)
    snr_in = 10 * np.log10(p_sig / p_art); snr_out = 10 * np.log10(p_sig / p_res)
    contam = p_art >= 0.1 * p_sig                                   # channels that actually carry artifact (>=10 % of signal power)
    a0, a1 = alpha_ratio(base, names, run, fs), alpha_ratio(xhat, names, run, fs)
    return dict(snr_gain=float(10 * np.log10(p_art.sum() / p_res.sum())),            # PRIMARY: pooled over channels
                snr_gain_chmean=float((snr_out - snr_in).mean()),                       # mean of per-channel gains (punishes touching clean channels)
                snr_gain_contam=float((snr_out - snr_in)[contam].mean()) if contam.any() else float("nan"), n_contam=int(contam.sum()),
                rrmse=float(rr.mean()), cc=float(cc.mean()), alpha_ratio_err=float(abs(a1 - a0) / a0), alpha_ratio_clean=float(a0), alpha_ratio_cleaned=float(a1))

def main():
    subjects = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]; rows = []
    for s in subjects:
        z = np.load(f"{DATA}/S{s}.npz"); base, names, run, fs, pos = z["base"].astype(np.float64), list(z["ch_names"]), z["run"], float(z["sfreq"]), z["pos"]
        for nch, subset in SUBSETS.items():
            idx = list(range(len(names))) if subset is None else [names.index(n) for n in subset if n in names]
            nm = [names[i] for i in idx]
            for key in [k for k in z.files if k.startswith("noisy_")]:
                _, art, snr = key.split("_"); noisy = z[key].astype(np.float64)[idx]; b = base[idx]
                for meth, fn in METHODS.items():
                    t0 = time.time()
                    try: xhat = fn(noisy, nm, fs, pos[idx]) if meth == "gedai" else fn(noisy, nm, fs); ok = ""
                    except Exception as e: xhat = noisy; ok = f"FAILED: {type(e).__name__}: {str(e)[:80]}"
                    r = dict(subject=s, artifact=art, snr=int(snr), nch=len(idx), method=meth, seconds=round(time.time() - t0, 1), note=ok, **scores(xhat, b, noisy, nm, run, fs))
                    rows.append(r); print(f"S{s} {art:4s} {snr:>3s} dB {len(idx):2d}ch {meth:12s} gain {r['snr_gain']:+6.2f} dB  CC {r['cc']:.3f}  alpha-ratio err {r['alpha_ratio_err']:.3f}  {r['seconds']}s {ok}", flush=True)
                for prefix, mname in (("brain", "retuned_fly_brain"), ("fly", "fly_brain_tcn"), ("tcn", "tcn_nobrain")):     # single-channel rows, if predictions exist
                    fp = f"{DATA}/{prefix}_S{s}_{art}_{snr}_{len(idx)}.npy"
                    if os.path.exists(fp):
                        r = dict(subject=s, artifact=art, snr=int(snr), nch=len(idx), method=mname, seconds=0, note="", **scores(np.load(fp).astype(np.float64), b, noisy, nm, run, fs)); rows.append(r)
                        print(f"S{s} {art:4s} {snr:>3s} dB {len(idx):2d}ch {mname:12s} gain {r['snr_gain']:+6.2f} dB  CC {r['cc']:.3f}  alpha-ratio err {r['alpha_ratio_err']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(f"{RES}/compare.csv", index=False)
    df = pd.DataFrame(rows); ok = df[df.note == ""]
    g = ok.groupby(["artifact", "nch", "method"])[["snr_gain", "snr_gain_contam", "cc", "alpha_ratio_err"]].mean().round(3)
    md = "# Track C: multichannel comparison (mean over subjects and SNR levels)\n\n" + g.to_markdown() + "\n"
    open(f"{RES}/summary.md", "w").write(md); print(md)

if __name__ == "__main__":
    main()
