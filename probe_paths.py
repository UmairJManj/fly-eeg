"""How much of the artifact could a path explain? Least-squares fit of (noisy - clean) from brain readouts on training epochs.
Paths: full-rate readout (r1) and the coarse --multirate K readout (r2) of the SAME brain; also the coarse path at several K."""
import argparse, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
p = argparse.ArgumentParser(); p.add_argument("--brain", required=True); p.add_argument("--n", type=int, default=200)
p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8]); p.add_argument("--device", default="cpu"); a = p.parse_args()
bd = load_brain(a.brain, a.device); S = bd.a
da = argparse.Namespace(artifact=S.artifact, protocol="whole", n_train=S.n_train, n_test=S.n_test, extra_train=0, aug=S.aug,
                        artifact_split=S.artifact_split, snr_lo=-7.0, snr_hi=2.0)
x, y, _ = fd.load_data(da, np.random.default_rng(0)); X, Y = x["tr"][:a.n], y["tr"][:a.n]
sc = Y.std(1, keepdims=True) + 1e-6; Yn, art = Y / sc, (Y - X) / sc            # same per-epoch scaling as predict()
def R2(feats):
    F = np.stack([f.ravel() for f in feats] + [np.ones(art.size)], 1); w, *_ = np.linalg.lstsq(F, art.ravel(), rcond=None)
    res = art.ravel() - F @ w; return 1 - res.var() / art.ravel().var(), w
with torch.no_grad():
    yt = torch.as_tensor(Yn, dtype=torch.float32, device=a.device)
    r1 = torch.cat([bd._brain_artifact(yt[i:i + 25]) for i in range(0, a.n, 25)]).cpu().numpy()
    print(f"full path only: R2 {R2([r1])[0]:.4f}", flush=True)
    for K in a.ks:
        S.multirate = K
        r2 = torch.cat([bd._coarse(yt[i:i + 25]) for i in range(0, a.n, 25)]).cpu().numpy()
        r2v, w = R2([r1, r2]); print(f"full + coarse K={K}: R2 {r2v:.4f}  weights {np.round(w, 4)}  | coarse alone R2 {R2([r2])[0]:.4f}", flush=True)
