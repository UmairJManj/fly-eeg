"""Real retuned brain: loss/grad/prediction agreement and speed, gather path vs CSR path (NFLY_CSR toggle)."""
import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfly.brain.rnn as R
from fly_brain_apply import load_brain
run = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 32
bd = load_brain(run, "cuda"); bd.train()
z = np.load(Path(run) / "results.npz"); y = torch.tensor(z["noisy"][:B, :256]).float().cuda(); x = torch.tensor(z["clean"][:B, :256]).float().cuda()
ps = [p for p in bd.brain_params() if p.requires_grad]
out = {}
for csr in (False, True, False, True):
    R._USE_CSR = csr; bd.model._csr_w = None
    for p in ps: p.grad = None
    torch.cuda.synchronize(); t0 = time.time()
    L = ((bd(y) - x) ** 2).mean(); L.backward(); torch.cuda.synchronize(); dt = time.time() - t0
    out[csr] = (L.item(), [p.grad.clone() if p.grad is not None else None for p in ps], dt)
    print(f"csr={csr}: loss {L.item():.6f}  train step {dt:.2f}s  (B={B}, T=256, bidir={getattr(bd.a,'bidir',False)})", flush=True)
for (g0, g1, p) in zip(out[False][1], out[True][1], ps):
    if g0 is None: continue
    print(f"  param {tuple(p.shape)} grad rel err {((g0-g1).norm()/(g0.norm()+1e-30)).item():.2e}")
with torch.no_grad():
    yt = z["noisy"][:200]
    for csr in (False, True):
        R._USE_CSR = csr; bd.model._csr_w = None; torch.cuda.synchronize(); t0 = time.time()
        from fly_eeg_brain import predict
        xh = predict(bd, yt, 100); torch.cuda.synchronize()
        print(f"predict csr={csr}: {time.time()-t0:.2f}s, max|diff vs stored| {np.abs(xh - z['xhat_brain'][:200]).max():.2e}")
print("speedup train step", out[False][2] / out[True][2])
