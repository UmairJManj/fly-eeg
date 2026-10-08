"""Zero-shot evaluation on an independent raw cache (e.g. SS2016): saved readout models (tcn_models/<dir>/*.pt) and retuned
fly brains (brain/<dir>) are applied to the cache's TEST windows as-is (no retraining). Prints SNR gain / RRMSE / CC per model.
  python zeroshot_eval.py CACHE_DIR [--models DIR ...] [--brain DIR ...] [--out results.json]"""
import argparse, sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/fly-eeg")
from fly_eeg_denoise import metrics, fmt, fir_features, Ridge
from fly_eeg_readout_v2 import TCN, BASELINES
p = argparse.ArgumentParser(); p.add_argument("cache"); p.add_argument("--models", action="append", default=[]); p.add_argument("--brain", action="append", default=[])
p.add_argument("--out", default=None); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--n", type=int, default=0)
a = p.parse_args(); dev = a.device; C = Path(a.cache)
x = np.load(C / "te_clean.npy"); y = np.load(C / "te_noisy.npy")
if a.n: x, y = x[: a.n], y[: a.n]
res = {"noisy": metrics(y, x, y)}; print(f"{'noisy':40s} {fmt(res['noisy'])}", flush=True)
# FIR fitted on this cache's train split (the classical in-domain reference)
xtr, ytr = np.load(C / "tr_clean.npy"), np.load(C / "tr_noisy.npy"); r = Ridge(65, dev)
for b in range(0, len(ytr), 200): r.add("tr", fir_features(ytr[b:b + 200], dev).reshape(-1, 65), torch.as_tensor(xtr[b:b + 200], device=dev).reshape(-1))
xva, yva = np.load(C / "va_clean.npy"), np.load(C / "va_noisy.npy")
for b in range(0, len(yva), 200): r.add("va", fir_features(yva[b:b + 200], dev).reshape(-1, 65), torch.as_tensor(xva[b:b + 200], device=dev).reshape(-1))
r.fit(); xh = np.concatenate([r.predict(fir_features(y[b:b + 200], dev).reshape(-1, 65)).reshape(-1, y.shape[1]).float().cpu().numpy() for b in range(0, len(y), 200)])
res["fir_in_domain"] = metrics(xh, x, y); print(f"{'FIR (fitted here)':40s} {fmt(res['fir_in_domain'])}", flush=True)
with torch.no_grad():
    for d in a.models:
        for pt in sorted(Path(d).glob("*.pt")):
            c = torch.load(pt, map_location=dev, weights_only=False); cfg = c["cfg"].split("+")[0]
            net = (TCN(c["mu"].to(dev), c["sd"].to(dev), c["K"], c["width"], tuple(c["dil"]), drop=c["drop"]) if cfg == "tcn" else BASELINES[cfg](c["mu"].to(dev), c["sd"].to(dev), c["K"])).to(dev)
            net.load_state_dict(c["state"]); net.eval()
            xh = np.concatenate([net(torch.as_tensor(y[b:b + 200], device=dev)[:, :, None]).float().cpu().numpy() for b in range(0, len(y), 200)])
            name = f"{Path(d).name}/{pt.stem}"; res[name] = metrics(xh, x, y); print(f"{name:40s} {fmt(res[name])}", flush=True)
    if a.brain:
        from fly_brain_apply import load_brain
        for d in a.brain:
            bd = load_brain(d, dev); sc = y.std(1, keepdims=True) + 1e-8
            xh = np.concatenate([bd((y[b:b + 100] / sc[b:b + 100]).astype(np.float32)).float().cpu().numpy() for b in range(0, len(y), 100)]) * sc
            name = f"retuned_brain/{Path(d).name}"; res[name] = metrics(xh, x, y); print(f"{name:40s} {fmt(res[name])}", flush=True)
if a.out: json.dump({k: {m: float(v.mean()) for m, v in r_.items()} for k, r_ in res.items()}, open(a.out, "w"), indent=1)
