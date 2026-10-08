"""Export two extra clean-ground-truth EEG denoising datasets (2026-09-29, multi-dataset benchmark).

D3 motion : PhysioNet 'motion-artifact' (Sweeney et al. 2012). Two nearby EEG electrodes, one pulled to create motion
            artifacts, the other undisturbed (reference = clean). 2048 Hz -> 256 Hz, 1-80 Hz band-pass, 2 s windows
            (512 samples) that overlap the triggered artifact periods; noisy = moved channel, clean = reference channel,
            both divided by std(noisy) per window (EEGdenoiseNet convention). Split BY RECORDING (no leakage):
            ~65 % train / 10 % val / 25 % test of the recordings. Output: raw cache {tr,va,te}_{clean,noisy}.npy.
D4 ecg    : MIT-BIH arrhythmia (MLII lead) -> 256 Hz, 1-80 Hz, 2 s segments = ECG artifact pool, saved next to the
            EEGdenoiseNet pools as ECG_all_epochs.npy, so fly_eeg_brain.py / readout v2 can use --artifact ecg
            (EEGdenoiseNet mixing rule, SNR -7..2 dB, leak-free disjoint artifact split).
  python ext_export.py"""
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, resample_poly

EXT = Path("/nesi/nobackup/aut04653/Manj/Fly/ext_data")
DATA = Path("/nesi/project/aut04653/Manj/Fly/fly-eeg/data")
MOT_OUT = Path("/nesi/nobackup/aut04653/Manj/Fly/states_motion_raw")
FS, T = 256, 512
SOS = butter(4, [1, 80], btype="band", fs=FS, output="sos")


def read_header(hea):
    lines = [l for l in open(hea).read().splitlines() if l and not l.startswith("#")]
    n_sig, fs = int(lines[0].split()[1]), float(lines[0].split()[2].split("/")[0])
    sig = [l.split() for l in lines[1:1 + n_sig]]
    return fs, sig


def read_fmt16(rec):
    fs, sig = read_header(rec.with_suffix(".hea"))
    raw = np.fromfile(rec.with_suffix(".dat"), dtype="<i2")
    X = raw[: len(raw) // len(sig) * len(sig)].reshape(-1, len(sig)).T.astype(np.float64)
    return fs, {s[-1]: X[i] for i, s in enumerate(sig)}


def read_fmt212(rec):
    fs, sig = read_header(rec.with_suffix(".hea"))
    b = np.fromfile(rec.with_suffix(".dat"), dtype=np.uint8)
    b = b[: len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
    s0 = ((b[:, 1] & 0x0F) << 8) | b[:, 0]
    s1 = ((b[:, 1] & 0xF0) << 4) | b[:, 2]
    s0 = np.where(s0 > 2047, s0 - 4096, s0); s1 = np.where(s1 > 2047, s1 - 4096, s1)
    X = np.stack([s0, s1]).astype(np.float64)
    return fs, {s[-1]: X[i] for i, s in enumerate(sig)}


def prep(x, fs):
    up, down = FS, int(round(fs))
    g = np.gcd(up, down)
    return sosfiltfilt(SOS, resample_poly(x - x.mean(), up // g, down // g))

def read_ann(p):
    """WFDB (MIT format) annotation reader -> list of (sample, code)."""
    b = np.fromfile(p, dtype="<u2"); i = 0; t = 0; out = []
    while i < len(b):
        w = int(b[i]); A, I = w >> 10, w & 0x3FF; i += 1
        if A == 0 and I == 0: break
        if A == 59: t += (int(b[i]) << 16) | int(b[i + 1]); i += 2; continue
        if A == 63: i += (I + 1) // 2; continue
        if A in (60, 61, 62): continue
        t += I; out.append((t, A))
    return out


def motion_mask(ann, n_raw, fs):
    """code 6 = motion (artifact) period starts, code 3 = it ends (alternating every 60 s in this dataset)."""
    on = np.zeros(n_raw, bool); start = None
    for t, c in ann:
        if c == 6 and start is None: start = t
        elif c == 3 and start is not None: on[start:t] = True; start = None
    if start is not None: on[start:] = True
    return on


def motion():
    recs = [EXT / "motion" / r for r in open(EXT / "motion" / "RECORDS").read().split() if r.startswith("eeg_")]   # skip the fNIRS recordings
    per = []
    for r in recs:
        fs, S = read_fmt16(r)
        e1, e2 = prep(S["EEG1"], fs), prep(S["EEG2"], fs)
        on_raw = motion_mask(read_ann(r.with_suffix(".trigger")), len(S["EEG1"]), fs)
        on = on_raw[:: int(round(fs / FS))]
        n = min(len(e1), len(e2), len(on)); e1, e2, on = e1[:n], e2[:n], on[:n]
        # the MOVED channel is the one with more power inside the triggered periods
        moved, ref = (e1, e2) if e1[on].var() / e1[~on].var() > e2[on].var() / e2[~on].var() else (e2, e1)
        starts = np.arange(0, n - T, T // 2)
        keep = [s for s in starts if on[s:s + T].mean() > 0.25]
        if not keep: continue
        y = np.stack([moved[s:s + T] for s in keep]); x = np.stack([ref[s:s + T] for s in keep])
        sc = y.std(1, keepdims=True) + 1e-8
        per.append((r.name, (x / sc).astype(np.float32), (y / sc).astype(np.float32)))
        print(f"{r.name}: {len(keep)} artifact windows (moved={'EEG1' if moved is e1 else 'EEG2'})", flush=True)
    rng = np.random.default_rng(0); order = rng.permutation(len(per)); n = len(per)
    n_tr, n_va = int(round(0.65 * n)), max(1, int(round(0.10 * n)))
    split = {"tr": order[:n_tr], "va": order[n_tr:n_tr + n_va], "te": order[n_tr + n_va:]}
    MOT_OUT.mkdir(parents=True, exist_ok=True)
    for k, ids in split.items():
        x = np.concatenate([per[i][1] for i in ids]); y = np.concatenate([per[i][2] for i in ids])
        np.save(MOT_OUT / f"{k}_clean.npy", x); np.save(MOT_OUT / f"{k}_noisy.npy", y)
        S = np.lib.format.open_memmap(MOT_OUT / f"{k}_states.npy", "w+", np.float16, (len(y), T, 1)); S[:] = y[:, :, None]; S.flush()
        print(f"motion {k}: {len(x)} windows from recordings {[per[i][0] for i in ids]}", flush=True)


def ecg():
    recs = [EXT / "mitdb" / r for r in open(EXT / "mitdb" / "RECORDS").read().split()]
    segs = []
    for r in recs:
        if not r.with_suffix(".dat").exists(): continue
        fs, S = read_fmt212(r)
        lead = S.get("MLII", next(iter(S.values())))
        z = prep(lead, fs)
        segs.append(np.stack([z[s:s + T] for s in range(0, len(z) - T, T)])[::4])    # every 4th 2 s block per record
    pool = np.concatenate(segs).astype(np.float32)
    np.save(DATA / "ECG_all_epochs.npy", pool)
    print(f"ECG pool: {pool.shape} from {len(segs)} MIT-BIH records -> {DATA / 'ECG_all_epochs.npy'}", flush=True)


if __name__ == "__main__":
    motion()
    ecg()
