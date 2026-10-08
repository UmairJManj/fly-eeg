"""Average the outputs of several retuned brains (independent seeds / recipes) on the EEGdenoiseNet test set.
  python brain_ensemble_eval.py --brain DIR [--brain DIR ...] [--out results.json]
Test set = load_data with the FIRST brain's args (same artifact / split / n_test / rng 0), so scores are comparable
to each brain's own summary.json."""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fly-eeg"))
import fly_eeg_denoise as fd
from fly_eeg_brain import predict
from fly_brain_apply import load_brain
p = argparse.ArgumentParser(); p.add_argument("--brain", action="append", required=True); p.add_argument("--out", default=None); a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
S = json.load(open(Path(a.brain[0]) / "summary.json"))["args"]; da = argparse.Namespace(**S); da.device = dev; da.snr0 = None
x, y, snr = fd.load_data(da, np.random.default_rng(0))
res, xhs = {"noisy": fd.metrics(y["te"], x["te"], y["te"])}, []
print(f"{'noisy':40s} {fd.fmt(res['noisy'])}", flush=True)
for d in a.brain:
    bd = load_brain(d, dev); xh = predict(bd, y["te"], 100); xhs.append(xh)
    res[Path(d).name] = fd.metrics(xh, x["te"], y["te"]); print(f"{Path(d).name:40s} {fd.fmt(res[Path(d).name])}", flush=True)
    del bd; torch.cuda.empty_cache()
    if len(xhs) > 1:
        m = fd.metrics(np.mean(xhs, 0), x["te"], y["te"]); res[f"avg_{len(xhs)}"] = m; print(f"{'AVG of ' + str(len(xhs)) + ' brains':40s} {fd.fmt(m)}", flush=True)
if a.out: json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
