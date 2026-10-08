"""Information ceiling of the trained brain's output neurons: ridge from neuron states to the artifact (train epochs),
scored on held-out epochs, vs the fixed random wire. Also a wider neuron pool (random non-JO subset)."""
import argparse, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
p = argparse.ArgumentParser(); p.add_argument("--brain", required=True); p.add_argument("--ntr", type=int, default=600); p.add_argument("--nte", type=int, default=200)
p.add_argument("--wide", type=int, default=4000); p.add_argument("--device", default="cuda"); a = p.parse_args()
bd = load_brain(a.brain, a.device); S = bd.a
da = argparse.Namespace(artifact=S.artifact, protocol="whole", n_train=S.n_train, n_test=S.n_test, extra_train=0, aug=S.aug,
                        artifact_split=S.artifact_split, snr_lo=-7.0, snr_hi=2.0)
x, y, _ = fd.load_data(da, np.random.default_rng(0))
def prep(X, Y): sc = Y.std(1, keepdims=True) + 1e-6; return Y / sc, (Y - X) / sc
ytr, atr = prep(x["tr"][:a.ntr], y["tr"][:a.ntr]); yte, ate = prep(x["te"][:a.nte], y["te"][:a.nte])
nonjo = torch.as_tensor(np.setdiff1d(np.arange(bd.model.n), bd.in_idx.cpu().numpy()))
wide = torch.sort(nonjo[torch.randperm(len(nonjo), generator=torch.Generator().manual_seed(0))[:a.wide]])[0].to(a.device)
@torch.no_grad()
def feats(Y, idx):
    out = []
    for i in range(0, len(Y), 20):
        yt = torch.as_tensor(Y[i:i + 20], dtype=torch.float32, device=a.device); yy = torch.cat([yt, torch.flip(yt, [1])], 0)
        hh = bd.states(yy, idx); B = len(yt); h = 0.5 * (hh[:B] + torch.flip(hh[B:], [1])); out.append(h.reshape(-1, h.shape[-1]).double().cpu())
    return torch.cat(out)
def ridge(Ftr, ttr, Fte, tte, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10)):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8; A, B = (Ftr - mu) / sd, (Fte - mu) / sd
    A = torch.cat([A, torch.ones(len(A), 1, dtype=A.dtype)], 1); B = torch.cat([B, torch.ones(len(B), 1, dtype=B.dtype)], 1)
    G, r = A.T @ A, A.T @ ttr; best = None
    for l in lams:
        w = torch.linalg.solve(G + l * len(A) * torch.eye(len(G), dtype=G.dtype), r); res = tte - B @ w
        R2 = 1 - res.var() / tte.var(); best = max(best or (-9, l), (float(R2), l))
    return best
ttr, tte = torch.as_tensor(atr.ravel(), dtype=torch.float64), torch.as_tensor(ate.ravel(), dtype=torch.float64)
with torch.no_grad():
    yt = torch.as_tensor(yte, dtype=torch.float32, device=a.device)
    r1 = torch.cat([bd._brain_artifact(yt[i:i + 20]) for i in range(0, len(yt), 20)]).double().cpu().ravel()
    w = torch.linalg.lstsq(torch.stack([r1, torch.ones_like(r1)], 1), tte[:, None]).solution; res = tte - torch.stack([r1, torch.ones_like(r1)], 1) @ w[:, 0]
    print(f"fixed random wire (best affine on test): R2 {float(1 - res.var() / tte.var()):.4f}", flush=True)
Ftr, Fte = feats(ytr, bd.out_idx), feats(yte, bd.out_idx); print(f"ridge over the {Ftr.shape[1]} DN/motor wire neurons: R2 {ridge(Ftr, ttr, Fte, tte)}", flush=True)
Ftr, Fte = feats(ytr, wide), feats(yte, wide); print(f"ridge over {Ftr.shape[1]} random non-JO neurons: R2 {ridge(Ftr, ttr, Fte, tte)}", flush=True)
