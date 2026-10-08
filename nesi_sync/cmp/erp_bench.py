"""Track D: REAL-EEG utility benchmark, protocol of Delorme 2023 (Sci Rep 13:2372, "EEG is better left alone").
No clean ground truth: a cleaner is judged by how much brain-evoked signal survives = the percentage of channels with a
significant condition difference (unpaired t-test, p < 0.05) of the mean potential in the dataset's 100-ms window of maximum
effect, averaged over bootstrap resamples of 50 epochs per condition. Reference = data high-pass filtered at 0.5 Hz (the
paper's best single step). Datasets (OpenNeuro, as in the paper):
  gonogo  ds002680  14 subj, 31 ch, ses-01 all runs, correct animal targets vs correct distractors, window 350-450 ms
  face    ds002718  18 subj, 70 ch, famous_new vs scrambled_new,                                      window 250-350 ms
  oddball ds003061  13 subj, 64 ch, run 1 (run 2 if run 1 is truncated), oddball_with_reponse vs standard (no response), 400-500 ms
All data -> 256 Hz, HP 0.5 Hz (FIR), epochs -0.3..0.7 s, no baseline (as in the paper).
Every cleaner runs on the continuous HP-filtered recording; single-channel denoisers channel by channel (2 s windows, 50 % overlap).
  python erp_bench.py clean --ds face --sub 2 [--methods ...]      -> erp_cache/<ds>/sub-XX/<method>.npy
  python erp_bench.py score [--reps 2000]                           -> cmp/results/erp/erp_scores.csv + erp_summary.md"""
import argparse, os, sys, time, warnings, json, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
try:
    import mne; mne.set_log_level("ERROR")            # cmp/.venv (loading + spatial cleaners); the nfly env (GPU denoisers) has no mne
except ImportError: mne = None
sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/fly-eeg"); sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/cmp")
DATA = Path("/nesi/nobackup/aut04653/Manj/Fly/erp_data"); CACHE = Path("/nesi/nobackup/aut04653/Manj/Fly/erp_cache")
RES = Path("/nesi/project/aut04653/Manj/Fly/cmp/results/erp"); FLY = Path("/nesi/project/aut04653/Manj/Fly")
NB = Path("/nesi/nobackup/aut04653/Manj/Fly"); FS = 256
DS = {"gonogo": dict(id="ds002680", win=(0.35, 0.45)), "face": dict(id="ds002718", win=(0.25, 0.35)), "oddball": dict(id="ds003061", win=(0.40, 0.50))}
BRAINS = {"eog": "eog_AB_full", "emg": "emg_Z_resbidir", "both": "both_N_jo48"}       # finished retuned brains (best per artifact)


# ---------------------------------------------------------------- loading
def _eeg_names(tsv):
    ch = pd.read_csv(tsv, sep="\t"); return [n for n, t in zip(ch["name"], ch["type"]) if str(t).upper() in ("EEG", "MISC")]

def _pos(tsv, names):
    """electrodes.tsv -> MNE head-frame positions (m): EEGLAB/CTF x=nose, y=left -> MNE x=right, y=nose; centred, radius 9.5 cm."""
    e = pd.read_csv(tsv, sep="\t").set_index("name").loc[names, ["x", "y", "z"]].to_numpy(float)
    p = np.c_[-e[:, 1], e[:, 0], e[:, 2]]; p -= p.mean(0) * np.array([1, 1, 0]); p /= np.median(np.linalg.norm(p, axis=1)); return p * 0.095

def _read(setf, names):
    raw = mne.io.read_raw_eeglab(setf, preload=True); raw.pick([n for n in names if n in raw.ch_names])
    raw.set_channel_types({n: "eeg" for n in raw.ch_names}); return raw

def load(ds, sub):
    root = DATA / DS[ds]["id"]; s = f"sub-{sub:03d}"
    if ds == "gonogo":
        d = root / s / "ses-01" / "eeg"; runs = sorted(d.glob("*_eeg.set"), key=lambda p: int(p.stem.split("run-")[1].split("_")[0]))
        names = _eeg_names(str(runs[0]).replace("_eeg.set", "_channels.tsv")); A, B, X, off = [], [], [], 0
        for r in runs:
            raw = _read(r, names).resample(FS); ev = pd.read_csv(str(r).replace("_eeg.set", "_events.tsv"), sep="\t")
            st = ev[ev.trial_type == "stimulus"]
            rt = pd.to_numeric(st.response_time, errors="coerce")                                  # 'n/a' (no response) -> NaN
            A += list((st[(st.value == "animal_target") & rt.notna()].onset * FS).round().astype(int) + off)        # correct go
            B += list((st[(st.value == "animal_distractor") & rt.isna()].onset * FS).round().astype(int) + off)     # correct no-go
            X.append(raw.get_data() * 1e6); off += raw.n_times
        X = np.concatenate(X, 1); names = raw.ch_names; etsv = str(runs[0]).replace("_eeg.set", "_electrodes.tsv")
    elif ds == "face":
        f = root / s / "eeg" / f"{s}_task-FaceRecognition_eeg.set"; names = [n for n in _eeg_names(str(f).replace("_eeg.set", "_channels.tsv")) if n.startswith("EEG")]
        names = [n for n in names if n in pd.read_csv(str(f).replace("_eeg.set", "_channels.tsv"), sep="\t").query("type == 'EEG'").name.tolist()]
        raw = _read(f, names).resample(FS); ev = pd.read_csv(str(f).replace("_eeg.set", "_events.tsv"), sep="\t")
        A = list((ev[ev.trial_type == "famous_new"].onset * FS).round().astype(int)); B = list((ev[ev.trial_type == "scrambled_new"].onset * FS).round().astype(int))
        X = raw.get_data() * 1e6; names = raw.ch_names; etsv = str(f).replace("_eeg.set", "_electrodes.tsv")
    else:
        for run in (1, 2):
            f = root / s / "eeg" / f"{s}_task-P300_run-{run}_eeg.set"; tsv = str(f).replace("_eeg.set", "_events.tsv")
            ev = pd.read_csv(tsv, sep="\t")
            if (ev.value == "oddball_with_reponse").sum() >= 60 or run == 2: break
        names = pd.read_csv(str(f).replace("_eeg.set", "_channels.tsv"), sep="\t").query("type == 'EEG'").name.tolist()
        raw = _read(f, names).resample(FS); resp = ev[ev.value == "response"].onset.to_numpy()
        A = list((ev[ev.value == "oddball_with_reponse"].onset * FS).round().astype(int))
        std = ev[ev.value == "standard"].onset.to_numpy(); ok = [t for t in std if not ((resp > t) & (resp < t + 1.0)).any()]
        B = list((np.array(ok) * FS).round().astype(int)); X = raw.get_data() * 1e6; names = raw.ch_names; etsv = str(f).replace("_eeg.set", "_electrodes.tsv")
    pos = _pos(etsv, names)
    X = mne.filter.filter_data(X, FS, 0.5, None, verbose=False)            # the paper's reference step: HP 0.5 Hz
    return X.astype(np.float64), list(names), pos, np.array(A), np.array(B)


# ---------------------------------------------------------------- cleaners
def _raw(x, names, pos):
    raw = mne.io.RawArray(x * 1e-6, mne.create_info(list(names), FS, "eeg"))
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, pos)), coord_frame="head")); return raw

def m_ica(x, names, pos, thr=0.9):
    """ICA (extended infomax) + ICLabel: drop eye / muscle components with p > thr (EEGLAB default 0.9, as the paper's pipeline)."""
    from mne_icalabel import label_components
    raw = _raw(x, names, pos); raw.set_eeg_reference("average"); n = min(len(names) - 1, 30)
    ica = mne.preprocessing.ICA(n_components=n, method="infomax", fit_params=dict(extended=True), random_state=0, max_iter="auto")
    ica.fit(raw.copy().filter(1.0, None, verbose=False)); lab = label_components(raw, ica, method="iclabel")
    bad = [i for i, (l, p) in enumerate(zip(lab["labels"], lab["y_pred_proba"])) if l in ("eye blink", "muscle artifact") and p > thr]
    out = ica.apply(_raw(x, names, pos), exclude=bad).get_data() * 1e6; return out

def m_ica50(x, names, pos): return m_ica(x, names, pos, 0.5)

def m_asr(x, names, pos):
    import run_compare as rc; return rc.m_asr(x, names, FS)

def m_gedai(x, names, pos):
    import run_compare as rc; return rc.m_gedai(x, names, FS, pos)

def _channelwise(x, fn):
    return np.stack([fn(x[c]) for c in range(len(x))])

def m_flybrain(art):
    def f(x, names, pos):
        import fly_brain_apply as fba
        bd = fba.load_brain(str(FLY / "brain" / BRAINS[art]), "cuda"); return _channelwise(x, lambda s: fba.clean_channel(s, bd))
    return f

def m_tcn(kind, art):
    """kind 'nobrain' -> TCN on the raw window; 'fly' -> fixed fly reservoir + TCN (Track B)."""
    def f(x, names, pos):
        import fly_apply as fa
        nets = fa.load_tcns(str(NB / "tcn_models" / (f"nobrain_{art}" if kind == "nobrain" else art)), "cuda")
        res = None if kind == "nobrain" else fa.load_brain("cuda"); return _channelwise(x, lambda s: fa.clean_channel(s, res, nets, 256))
    return f

def m_ai(arch, art):
    def f(x, names, pos):
        import sota_apply as sa
        nets = sa.load(art, arch, "cuda"); return _channelwise(x, lambda s: sa.clean_channel(s, nets, "cuda"))
    return f

METHODS = {"hp": lambda x, n, p: x, "ica_iclabel": m_ica, "ica_iclabel50": m_ica50, "asr": m_asr, "gedai": m_gedai}
for art in ("eog", "emg", "both"):
    METHODS[f"fly_brain_{art}"] = m_flybrain(art); METHODS[f"tcn_nobrain_{art}"] = m_tcn("nobrain", art); METHODS[f"fly_tcn_{art}"] = m_tcn("fly", art)
    for arch in ("eegdir", "ccnn", "scnn", "xfmr", "rnn", "fcnn"): METHODS[f"{arch}_{art}"] = m_ai(arch, art)
CPU_METHODS = ["hp", "ica_iclabel", "ica_iclabel50", "asr", "gedai"]
GPU_METHODS = ["fly_brain_both", "fly_brain_eog", "fly_brain_emg", "tcn_nobrain_both", "fly_tcn_both"] + [f"{a}_both" for a in ("eegdir", "ccnn", "scnn", "xfmr", "rnn", "fcnn")] + ["eegdir_eog", "ccnn_eog", "tcn_nobrain_eog"]


def clean(a):
    out = CACHE / a.ds / f"sub-{a.sub:03d}"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if (out / "hp.npy").exists() and (out / "meta.npz").exists() and not a.remeta:
        X = np.load(out / "hp.npy").astype(np.float64); M = np.load(out / "meta.npz"); names, pos, A, B = list(M["names"]), M["pos"], M["A"], M["B"]
    else:
        X, names, pos, A, B = load(a.ds, a.sub); np.save(out / "hp.npy", X.astype(np.float32))
        np.savez(out / "meta.npz", names=np.array(names), pos=pos, A=A, B=B, n=X.shape[1])
    print(f"{a.ds} sub-{a.sub:03d}: {len(names)} ch, {X.shape[1] / FS / 60:.1f} min, {len(A)} vs {len(B)} trials  ({time.time() - t0:.0f}s load)", flush=True)
    for m in a.methods:
        f = out / f"{m}.npy"
        if f.exists() and not a.force: continue
        if not m.startswith(("hp", "ica", "asr", "gedai")):
            arch = m.split("_")[0]
            if arch in ("eegdir", "ccnn", "scnn", "xfmr", "rnn", "fcnn") and not list((NB / "tcn_models" / f"sota_{m.split('_')[-1]}").glob(f"{arch}_*.pt")):
                print(f"  {m}: models not saved yet, skipped", flush=True); continue
        t0 = time.time()
        try: Y = METHODS[m](X, names, pos); np.save(f, Y.astype(np.float32)); print(f"  {m:20s} {time.time() - t0:7.0f}s", flush=True)
        except Exception as e: print(f"  {m:20s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


# ---------------------------------------------------------------- scoring
def pct_sig(Y, A, B, win, reps, rng, n=50):
    """Delorme 2023: mean potential in the 100-ms window per epoch; draw n epochs per condition with replacement; unpaired
    t-test per channel; % channels p < .05; average over reps."""
    from scipy.stats import t as tdist
    i0, i1 = int(round(win[0] * FS)), int(round(win[1] * FS))
    def amp(on):
        on = on[(on + i0 >= 0) & (on + i1 <= Y.shape[1])]; return np.stack([Y[:, o + i0:o + i1].mean(1) for o in on])     # (trials, C)
    a, b = amp(A), amp(B); crit = tdist.ppf(0.975, 2 * n - 2); out = []
    for r0 in range(0, reps, 250):
        k = min(250, reps - r0); ia = rng.integers(0, len(a), (k, n)); ib = rng.integers(0, len(b), (k, n))
        xa, xb = a[ia], b[ib]                                                                                             # (k, n, C)
        sp = np.sqrt(((n - 1) * xa.var(1, ddof=1) + (n - 1) * xb.var(1, ddof=1)) / (2 * n - 2) * (2 / n))
        out.append((np.abs(xa.mean(1) - xb.mean(1)) / sp > crit).mean(1))
    return 100 * float(np.concatenate(out).mean()), len(a), len(b)

def score(a):
    RES.mkdir(parents=True, exist_ok=True); rows = []
    for ds in DS:
        for sd in sorted((CACHE / ds).glob("sub-*")):
            meta = np.load(sd / "meta.npz")
            for f in sorted(sd.glob("*.npy")):
                Y = np.load(f).astype(np.float64)
                if not np.isfinite(Y).all(): print(f"{ds} {sd.name} {f.stem}: non-finite, skipped"); continue
                rng = np.random.default_rng(0)                                          # same resamples for every method (paired)
                p, na, nb = pct_sig(Y, meta["A"], meta["B"], DS[ds]["win"], a.reps, rng)
                rows.append(dict(dataset=ds, subject=sd.name, method=f.stem, pct_sig=p, nA=na, nB=nb))
            print(f"{ds} {sd.name}: {len([r for r in rows if r['subject'] == sd.name and r['dataset'] == ds])} methods", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "erp_scores.csv", index=False)
    md = ["# Track D: real-EEG ERP utility (Delorme 2023 metric): % significant channels, change vs HP 0.5 Hz\n",
          "Paired bootstrap over subjects (20,000), 95 % CI; + = more brain signal survives than with the HP filter alone.\n"]
    rng = np.random.default_rng(1); summ = []
    for ds, g in df.groupby("dataset"):
        base = g[g.method == "hp"].set_index("subject").pct_sig
        for m, gm in g.groupby("method"):
            d = (gm.set_index("subject").pct_sig - base).dropna().to_numpy()
            if len(d) < 3: continue
            bs = rng.choice(d, (20000, len(d))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
            p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            summ.append(dict(dataset=ds, method=m, n=len(d), hp=base.mean(), pct_sig=gm.pct_sig.mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi, p=p))
    S = pd.DataFrame(summ).round(3); S.to_csv(RES / "erp_summary.csv", index=False)
    P = S.pivot(index="method", columns="dataset", values="delta"); P["mean"] = P.mean(1); P = P.sort_values("mean", ascending=False)
    md += ["\n## Delta % significant channels vs HP (mean over subjects)\n", P.round(2).to_markdown(), "\n\n## Full\n", S.to_markdown(index=False)]
    open(RES / "erp_summary.md", "w").write("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["clean", "score"]); p.add_argument("--ds"); p.add_argument("--sub", type=int)
    p.add_argument("--methods", nargs="+", default=None); p.add_argument("--set", choices=["cpu", "gpu", "all"], default="all")
    p.add_argument("--force", action="store_true"); p.add_argument("--remeta", action="store_true", help="reload raw + rebuild meta/hp (events)"); p.add_argument("--reps", type=int, default=2000)
    a = p.parse_args()
    if a.cmd == "clean":
        a.methods = a.methods or (CPU_METHODS if a.set == "cpu" else GPU_METHODS if a.set == "gpu" else list(METHODS)); clean(a)
    else: score(a)
