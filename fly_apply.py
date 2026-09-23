"""Apply the fly reservoir + saved TCN readouts to arbitrary EEG (continuous, one or many channels).
Each channel is cut into 2 s windows (512 samples @ 256 Hz, 50 % overlap), scaled to unit std like the EEGdenoiseNet
protocol, pushed through the FIXED brain (same settings as the training caches: rho 0.9, alpha 0.02-0.5, bias 0.02,
gain 1, lag 24, warm 64, raw-input skip feature), decoded by the TCN ensemble, rescaled and overlap-added (Hann).
  python fly_apply.py --models DIR [--models DIR2 ...] --npz cmp_data/S1.npz --keys noisy_eog_+0 ... --out cmp_data
Writes <out>/fly_<npz stem>_<art>_<snr>_<nch>.npy for nch in (8,16,32,64) using the model dir matching the artifact
(first dir whose name contains the artifact tag; else the first dir)."""
import argparse, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_eeg_readout_v2 import TCN
from nfly.connectome import load_malecns
FS, T = 256, 512

def load_brain(dev):
    a = argparse.Namespace(rho=0.9, alpha_min=0.02, alpha_max=0.5, gain=1.0, bias=0.02, h_max=10.0, lag=24, warm=64, no_skip=False, device=dev, save_states=None)
    conn = fd.load_malecns(fd.NFLY_DATA)
    keep = conn.neurons.super_class.isin(fd.CENTRAL) | conn.neurons.cell_type.str.startswith("JO", na=False)
    conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy()))); ne = conn.neurons
    is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    in_idx = torch.as_tensor(np.flatnonzero(is_jo))
    out_idx = torch.as_tensor(np.flatnonzero(ne.super_class.isin(["descending_neuron"]).to_numpy() | (ne.flow == "efferent").to_numpy()))
    model = fd.make_model(conn, a, dev)
    return fd.Reservoir(model, in_idx, out_idx, a)

def load_tcns(d, dev):
    nets = []
    for p in sorted(Path(d).glob("*.pt")):
        c = torch.load(p, map_location=dev, weights_only=False)
        net = TCN(c["mu"].to(dev), c["sd"].to(dev), c["K"], c["width"], tuple(c["dil"]), drop=c["drop"]).to(dev); net.load_state_dict(c["state"]); net.eval(); nets.append(net)
    assert nets, f"no models in {d}"; return nets

@torch.no_grad()
def clean_channel(sig, res, nets, batch=100):
    """sig (N,) -> cleaned (N,). Windows with 50 % overlap, Hann overlap-add."""
    hop = T // 2; n = len(sig); pad = (-(n - T)) % hop
    x = np.concatenate([sig, np.zeros(pad)]); starts = np.arange(0, len(x) - T + 1, hop)
    W = np.stack([x[s:s + T] for s in starts]).astype(np.float32); sc = W.std(1, keepdims=True) + 1e-8; Wn = W / sc
    out = []
    for b in range(0, len(Wn), batch):
        if nets[0].emb.in_features == 1:                                    # no-brain TCN: raw window is the only feature
            feats = torch.as_tensor(Wn[b:b + batch], device=nets[0].head.weight.device)[:, :, None]
        else:
            feats = res(Wn[b:b + batch])                                    # (B, T, K+1)
        out.append(torch.stack([net(feats) for net in nets]).mean(0).cpu().numpy())
    Y = np.concatenate(out) * sc; win = np.hanning(T); acc = np.zeros(len(x)); wsum = np.zeros(len(x))
    for s, y in zip(starts, Y): acc[s:s + T] += y * win; wsum[s:s + T] += win
    return (acc / np.maximum(wsum, 1e-8))[:n]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--models", action="append", required=True); p.add_argument("--npz", required=True)
    p.add_argument("--keys", nargs="*", default=None); p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=100); p.add_argument("--prefix", default="fly", help="output file prefix (fly | tcn for no-brain models)"); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.device
    z = np.load(a.npz); keys = a.keys or [k for k in z.files if k.startswith("noisy_")]; names = list(z["ch_names"]); stem = Path(a.npz).stem
    tcn = {Path(d).name: load_tcns(d, dev) for d in a.models}
    res = None if all(n[0].emb.in_features == 1 for n in tcn.values()) else load_brain(dev)   # skip the brain for no-brain models
    SUB = {8: ["Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2"],
           16: ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "T7", "T8", "P3", "P4", "Pz", "O1", "O2", "Oz"],
           32: ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"], 64: None}
    for key in keys:
        _, art, snr = key.split("_"); nets = next((v for k, v in tcn.items() if art in k), next(iter(tcn.values())))
        X = z[key].astype(np.float64); Y = np.stack([clean_channel(X[c], res, nets, a.batch) for c in range(len(X))])
        for nch, sub in SUB.items():
            idx = list(range(len(names))) if sub is None else [names.index(n) for n in sub if n in names]
            np.save(Path(a.out) / f"{a.prefix}_{stem}_{art}_{snr}_{len(idx)}.npy", Y[idx].astype(np.float32))
        print(f"{stem} {key}: cleaned {len(X)} channels with {len(nets)} TCN(s) [{[k for k in tcn if art in k] or list(tcn)[:1]}]", flush=True)

if __name__ == "__main__":
    main()
