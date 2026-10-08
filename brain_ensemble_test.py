"""Average the test outputs of several retuned fly brains (same EEGdenoiseNet test set) and score every subset.
Run from nfly:  python ../fly-eeg-nesi/brain_ensemble_test.py --brain DIR1 --brain DIR2 ... --out DIR"""
import argparse, itertools, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
from fly_eeg_brain import predict

p = argparse.ArgumentParser(); p.add_argument("--brain", action="append", required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--batch", type=int, default=50); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
outs, xc, yn = {}, None, None
for run in a.brain:
    bd = load_brain(run, "cuda"); S = bd.a
    da = argparse.Namespace(artifact=S.artifact, protocol="whole", n_train=S.n_train, n_test=S.n_test, extra_train=0, aug=S.aug,
                            artifact_split=S.artifact_split, snr_lo=-7.0, snr_hi=2.0)
    x, y, _ = fd.load_data(da, np.random.default_rng(0))
    if xc is None: xc, yn = x["te"], y["te"]
    assert np.array_equal(xc, x["te"]), f"{run}: different test set"
    outs[Path(run).name] = predict(bd, yn, a.batch); del bd; torch.cuda.empty_cache()
    print(Path(run).name, fd.fmt(fd.metrics(outs[Path(run).name], xc, yn)), flush=True)
res = {}
for k in range(2, len(outs) + 1):
    for combo in itertools.combinations(outs, k):
        m = fd.metrics(np.mean([outs[c] for c in combo], 0), xc, yn); res[" + ".join(combo)] = {q: float(v.mean()) for q, v in m.items()}
        print(f"ENS {' + '.join(combo)}: {fd.fmt(m)}", flush=True)
json.dump(res, open(a.out / "ensemble.json", "w"), indent=1)
np.savez_compressed(a.out / "outputs.npz", clean=xc, noisy=yn, **outs)
