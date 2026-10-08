"""Track C benchmark: semi-simulated MULTICHANNEL EEG with ground truth.
Base = real 64-channel EEG (PhysioNet EEGBCI, runs 1 eyes-open, 2 eyes-closed, 4 and 8 motor), 1-40 Hz, 256 Hz,
average reference, pre-cleaned once with ICA+ICLabel (eye/muscle ICs removed) so that the "truth" is artifact-poor.
Artifacts = REAL EOG / EMG time courses from EEGdenoiseNet, projected onto the scalp with distance-decay topographies
(blink: one frontal source; muscle: 4 peripheral sources with sharper decay), added at a fixed global SNR.
Writes /nesi/nobackup/aut04653/Manj/Fly/cmp_data/S<subject>.npz: base (C,T), noisy_<art>_<snr> (C,T), ch_names, pos (C,3),
run (T,) run id per sample, sfreq.   Usage: python make_dataset.py [subjects...]"""
import sys, os, numpy as np, mne
from mne_icalabel import label_components
from mne.datasets import eegbci
MNE_DATA = "/nesi/nobackup/aut04653/Manj/Fly/mne_data"; OUT = "/nesi/nobackup/aut04653/Manj/Fly/cmp_data"
ART_DIR = "/nesi/project/aut04653/Manj/Fly/fly-eeg/data"
FS = 256; SNRS = (-5, 0, 5); RUNS = (1, 2, 4, 8)
rng = np.random.default_rng(0)

def load_subject(s):
    raws = []
    for r in RUNS:
        p = eegbci.load_data(subjects=s, runs=[r], path=MNE_DATA, update_path=False)[0]
        raw = mne.io.read_raw_edf(p, preload=True, verbose=False); eegbci.standardize(raw); raw.set_montage("standard_1005", verbose=False)
        raw.filter(1.0, 40.0, verbose=False).resample(FS, verbose=False); raw.set_eeg_reference("average", projection=False, verbose=False)
        raws.append(raw)
    run_id = np.concatenate([np.full(len(r.times), i) for i, r in enumerate(raws)])
    raw = mne.concatenate_raws(raws, verbose=False)
    return raw, run_id

def preclean(raw):
    ica = mne.preprocessing.ICA(n_components=25, method="infomax", fit_params=dict(extended=True), random_state=0, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)
    lab = label_components(raw, ica, method="iclabel")
    bad = [i for i, (l, p) in enumerate(zip(lab["labels"], lab["y_pred_proba"])) if l in ("eye blink", "muscle artifact", "heart beat", "channel noise", "line noise") and p > 0.6]
    print(f"  preclean: removing {len(bad)} ICs: {[lab['labels'][i] for i in bad]}", flush=True)
    return ica.apply(raw.copy(), exclude=bad, verbose=False)

def continuous_artifact(epochs, T, gap_range):
    """Concatenate random real artifact epochs with random silent gaps into a T-sample time course."""
    out = np.zeros(T); t = 0
    while t < T:
        e = epochs[rng.integers(len(epochs))]; e = e - e.mean()
        n = min(len(e), T - t); out[t:t + n] = e[:n]; t += n + rng.integers(*gap_range)
    return out

def topography(pos, centre, lam):
    d = np.linalg.norm(pos - pos[centre], axis=1); return np.exp(-d / lam)

def main():
    subjects = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    eog = np.load(f"{ART_DIR}/EOG_all_epochs.npy"); emg = np.load(f"{ART_DIR}/EMG_all_epochs.npy")
    os.makedirs(OUT, exist_ok=True)
    for s in subjects:
        print(f"== subject {s}", flush=True)
        raw, run_id = load_subject(s); raw = preclean(raw)
        base = raw.get_data() * 1e6; names = raw.ch_names; pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(names))]); C, T = base.shape
        arts = {}
        fp = names.index("Fpz") if "Fpz" in names else names.index("Fp1")
        arts["eog"] = np.outer(topography(pos, fp, 0.05), continuous_artifact(eog, T, (FS, 4 * FS)))
        src = [names.index(n) for n in ("T7", "T8", "Fp1", "Fp2", "O1", "O2") if n in names][:4]
        arts["emg"] = sum(np.outer(topography(pos, c, 0.025), continuous_artifact(emg, T, (FS // 2, 3 * FS))) for c in src)
        arts["both"] = arts["eog"] / arts["eog"].std() + arts["emg"] / arts["emg"].std()
        out = dict(base=base.astype(np.float32), ch_names=np.array(names), pos=pos, run=run_id, sfreq=FS)
        p_sig = (base ** 2).mean()
        for a, art in arts.items():
            for snr in SNRS:
                scale = np.sqrt(p_sig / (art ** 2).mean() / 10 ** (snr / 10))
                out[f"noisy_{a}_{snr:+d}"] = (base + scale * art).astype(np.float32)
        np.savez_compressed(f"{OUT}/S{s}.npz", **out)
        print(f"  saved {OUT}/S{s}.npz  {C} ch x {T} samples ({T / FS:.0f} s), artifacts {list(arts)} at SNR {SNRS} dB", flush=True)

if __name__ == "__main__":
    main()
