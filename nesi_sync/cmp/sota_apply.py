"""Track C: apply the saved single-channel AI denoisers (EEGDiR / complex CNN / simple CNN, leak-free EEGdenoiseNet-trained,
3-seed ensemble each) channel-wise to the multichannel benchmark, exactly like fly_apply.py (2 s windows, 50 % overlap,
unit-std scaling, Hann overlap-add, artifact-matched model dir).  Writes cmp_data/<arch>_S<s>_<art>_<snr>_<nch>.npy.
  python sota_apply.py --npz cmp_data/S1.npz [--archs eegdir ccnn scnn]"""
import argparse, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/fly-eeg")
from fly_eeg_readout_v2 import BASELINES
T = 512; NB = Path("/nesi/nobackup/aut04653/Manj/Fly")
SUB = {8: ["Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2"],
       16: ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "T7", "T8", "P3", "P4", "Pz", "O1", "O2", "Oz"],
       32: ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"], 64: None}

def load(art, arch, dev):
    nets = []
    for pt in sorted((NB / "tcn_models" / f"sota_{art}").glob(f"{arch}_*.pt")):
        c = torch.load(pt, map_location=dev, weights_only=False)
        net = BASELINES[c["cfg"].split("+")[0]](c["mu"].to(dev), c["sd"].to(dev), c["K"]).to(dev); net.load_state_dict(c["state"]); net.eval(); nets.append(net)
    assert nets, f"no {arch} models for {art}"; return nets

@torch.no_grad()
def clean_channel(sig, nets, dev, batch=256):
    hop = T // 2; n = len(sig); pad = (-(n - T)) % hop
    x = np.concatenate([sig, np.zeros(pad)]); starts = np.arange(0, len(x) - T + 1, hop)
    W = np.stack([x[s:s + T] for s in starts]).astype(np.float32); sc = W.std(1, keepdims=True) + 1e-8; Wn = W / sc
    Y = np.concatenate([torch.stack([net(torch.as_tensor(Wn[b:b + batch], device=dev)[:, :, None]) for net in nets]).mean(0).float().cpu().numpy()
                        for b in range(0, len(Wn), batch)]) * sc
    win = np.hanning(T); acc = np.zeros(len(x)); wsum = np.zeros(len(x))
    for s, y in zip(starts, Y): acc[s:s + T] += y * win; wsum[s:s + T] += win
    return (acc / np.maximum(wsum, 1e-8))[:n]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--npz", required=True); p.add_argument("--archs", nargs="+", default=["eegdir", "ccnn", "scnn"])
    p.add_argument("--out", default=str(NB / "cmp_data")); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); dev = a.device; z = np.load(a.npz); names = list(z["ch_names"]); stem = Path(a.npz).stem
    for arch in a.archs:
        for key in [k for k in z.files if k.startswith("noisy_")]:
            _, art, snr = key.split("_"); nets = load(art, arch, dev); X = z[key].astype(np.float64)
            Y = np.stack([clean_channel(X[c], nets, dev) for c in range(len(X))])
            for nch, sub in SUB.items():
                idx = list(range(len(names))) if sub is None else [names.index(n) for n in sub if n in names]
                np.save(Path(a.out) / f"{arch}_{stem}_{art}_{snr}_{len(idx)}.npy", Y[idx].astype(np.float32))
            print(f"{stem} {arch} {key}: {len(X)} ch, {len(nets)} seeds", flush=True)

if __name__ == "__main__":
    main()
