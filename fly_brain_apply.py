"""Apply a RETUNED fly brain (fly_eeg_brain.py output: brain_params.pt) to arbitrary EEG: one or many channels, continuous.
Each channel is cut into 2 s windows (50 % overlap), scaled to unit std, pushed through the retuned brain, read through
its fixed wire, rescaled and overlap-added. No decoder anywhere.
  python fly_brain_apply.py --brain DIR_eog [--brain DIR_emg ...] --npz cmp_data/S1.npz --keys noisy_eog_+0 ... --out cmp_data
Writes <out>/brain_<stem>_<art>_<snr>_<nch>.npy (the run dir whose name contains the artifact tag is used for that key)."""
import argparse, sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_eeg_brain import BrainDenoiser, shuffled_connectome
from nfly.connectome import load_malecns
T = 512

def load_brain(run_dir, dev):
    S = json.load(open(Path(run_dir) / "summary.json"))["args"]
    a = argparse.Namespace(**{k: v for k, v in S.items()}); a.device = dev; a.tbptt = 0
    for k, v in (("rho", 0.9), ("alpha_min", 0.02), ("alpha_max", 0.5), ("gain", 1.0), ("bias", 0.02), ("h_max", 10.0), ("lag", 24), ("warm", 64), ("wire", "dn"), ("jo_delays", 0), ("shuffle", False), ("train_edges", True)):
        if not hasattr(a, k): setattr(a, k, v)
    conn = fd.load_malecns(fd.NFLY_DATA)
    keep = conn.neurons.super_class.isin(fd.CENTRAL) | conn.neurons.cell_type.str.startswith("JO", na=False)
    conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy())))
    if a.shuffle: conn = shuffled_connectome(conn)
    ne = conn.neurons; is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    in_idx = torch.as_tensor(np.flatnonzero(is_jo))
    dn_idx = torch.as_tensor(np.flatnonzero(ne.super_class.isin(["descending_neuron"]).to_numpy() | (ne.flow == "efferent").to_numpy()))
    model = fd.make_model(conn, a, dev)
    if a.wire == "fastdn":
        al = model.alpha().detach().cpu()[dn_idx]; out_idx = dn_idx[torch.argsort(al, descending=True)[: len(dn_idx) // 2]]
    else: out_idx = dn_idx if a.wire == "dn" else torch.as_tensor(np.flatnonzero(~is_jo))
    bd = BrainDenoiser(model, in_idx, out_idx, a).to(dev); bd.register_buffer("dn_idx", dn_idx.to(dev))
    P = torch.load(Path(run_dir) / "brain_params.pt", map_location=dev, weights_only=False)
    missing = bd.load_state_dict(P, strict=False); bd.eval()
    print(f"loaded retuned brain from {run_dir} (test gain {json.load(open(Path(run_dir) / 'summary.json'))['brain']['snr_gain']:+.2f} dB); unexpected keys: {missing.unexpected_keys}", flush=True)
    return bd

@torch.no_grad()
def clean_channel(sig, bd, batch=128):
    hop = T // 2; n = len(sig); pad = (-(n - T)) % hop
    x = np.concatenate([sig, np.zeros(pad)]); starts = np.arange(0, len(x) - T + 1, hop)
    W = np.stack([x[s:s + T] for s in starts]).astype(np.float32); sc = W.std(1, keepdims=True) + 1e-8; Wn = W / sc
    Y = np.concatenate([bd(Wn[b:b + batch]).float().cpu().numpy() for b in range(0, len(Wn), batch)]) * sc
    win = np.hanning(T); acc = np.zeros(len(x)); wsum = np.zeros(len(x))
    for s, y in zip(starts, Y): acc[s:s + T] += y * win; wsum[s:s + T] += win
    return (acc / np.maximum(wsum, 1e-8))[:n]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--brain", action="append", required=True); p.add_argument("--npz", required=True)
    p.add_argument("--keys", nargs="*", default=None); p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--prefix", default="brain")
    a = p.parse_args(); dev = a.device
    z = np.load(a.npz); keys = a.keys or [k for k in z.files if k.startswith("noisy_")]; names = list(z["ch_names"]); stem = Path(a.npz).stem
    brains = {Path(d).name: load_brain(d, dev) for d in a.brain}
    SUB = {8: ["Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2"],
           16: ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "T7", "T8", "P3", "P4", "Pz", "O1", "O2", "Oz"],
           32: ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"], 64: None}
    for key in keys:
        _, art, snr = key.split("_"); bd = next((v for k, v in brains.items() if k.startswith(art)), next(iter(brains.values())))
        X = z[key].astype(np.float64); Y = np.stack([clean_channel(X[c], bd, a.batch) for c in range(len(X))])
        for nch, sub in SUB.items():
            idx = list(range(len(names))) if sub is None else [names.index(n) for n in sub if n in names]
            np.save(Path(a.out) / f"{a.prefix}_{stem}_{art}_{snr}_{len(idx)}.npy", Y[idx].astype(np.float32))
        print(f"{stem} {key}: cleaned {len(X)} channels with the retuned brain", flush=True)

if __name__ == "__main__":
    main()
