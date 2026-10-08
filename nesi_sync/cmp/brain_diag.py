"""Bottleneck diagnostics for a retuned brain on its own test set:
  (a) fixed +-1 wire (as trained)   (b) bidirectional: forward + time-reversed pass, averaged (is causality the limit?)
  (c) trained brain + ridge readout fitted on the DN neurons (is the fixed wire the limit?)
  python brain_diag.py --brain DIR [--brain DIR ...] [--out results.json]"""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fly-eeg"))
import fly_eeg_denoise as fd
from fly_eeg_brain import predict, ridge_on_brain
from fly_brain_apply import load_brain
p = argparse.ArgumentParser(); p.add_argument("--brain", action="append", required=True); p.add_argument("--out", default=None); a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; res = {}
for d in a.brain:
    S = json.load(open(Path(d) / "summary.json"))["args"]; da = argparse.Namespace(**S); da.device = dev; da.snr0 = None; da.eval_batch = 100
    x, y, _ = fd.load_data(da, np.random.default_rng(0)); name = Path(d).name; r = res[name] = {}
    bd = load_brain(d, dev)
    with torch.no_grad():
        xf = predict(bd, y["te"], 100); r["fixed_wire"] = fd.metrics(xf, x["te"], y["te"])
        xb = predict(bd, np.ascontiguousarray(y["te"][:, ::-1]), 100)[:, ::-1]; r["reversed_only"] = fd.metrics(xb, x["te"], y["te"])
        r["bidirectional_avg"] = fd.metrics((xf + xb) / 2, x["te"], y["te"])
        rg = ridge_on_brain(bd, x, y, da); rg.fit(); K = len(bd.dn_idx)
        xr = np.concatenate([rg.predict(bd.states(y["te"][b:b + 100], bd.dn_idx).reshape(-1, K)).reshape(-1, y["te"].shape[1]).float().cpu().numpy() for b in range(0, len(y["te"]), 100)])
        r["trained_brain_ridge_readout"] = fd.metrics(xr, x["te"], y["te"]); r["ridge_lambda"] = float(rg.lam)
    for k, v in r.items():
        if isinstance(v, dict): print(f"{name:26s} {k:30s} {fd.fmt(v)}", flush=True)
    del bd; torch.cuda.empty_cache()
if a.out: json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
