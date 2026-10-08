"""Test-time variants of a retuned fly brain on the EEGdenoiseNet test set (no training):
  base       brain(y)
  flip       0.5 * (brain(y) - brain(-y))           sign-symmetric average (JO +/- halves swap roles)
  shiftK     average of brain over circular-free time shifts +-k samples (reflect padded), realigned
  unroll2    brain(brain(y))                         second cleaning pass
Saves every variant's test output so ensembles across brains can be scored later.
Run from nfly:  python ../fly-eeg-nesi/brain_tta.py --brain RUN_DIR [--brain RUN_DIR2 ...] --out DIR
"""
import argparse, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
from fly_eeg_brain import predict


def shifted(bd, y, k, batch):
    yp = np.pad(y, ((0, 0), (abs(k), abs(k))), mode="reflect"); T = y.shape[1]
    ys = yp[:, abs(k) + k: abs(k) + k + T]
    out = predict(bd, np.ascontiguousarray(ys), batch)
    op = np.pad(out, ((0, 0), (abs(k), abs(k))), mode="reflect")
    return op[:, abs(k) - k: abs(k) - k + T]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--brain", action="append", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shifts", type=int, nargs="*", default=[-8, 8]); p.add_argument("--batch", type=int, default=50)
    p.add_argument("--device", default="cuda"); p.add_argument("--n-test", type=int, default=0)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); res = {}
    for run in a.brain:
        bd = load_brain(run, a.device); S = bd.a; tag = Path(run).name
        da = argparse.Namespace(artifact=S.artifact, protocol="whole", n_train=S.n_train, n_test=S.n_test, extra_train=0, aug=S.aug,
                                artifact_split=S.artifact_split, snr_lo=-7.0, snr_hi=2.0)
        x, y, _ = fd.load_data(da, np.random.default_rng(0)); xc, yn = x["te"], y["te"]
        if a.n_test: xc, yn = xc[:a.n_test], yn[:a.n_test]
        V = {"base": predict(bd, yn, a.batch)}
        V["neg"] = -predict(bd, -yn, a.batch)
        V["flip"] = 0.5 * (V["base"] + V["neg"])
        for k in a.shifts: V[f"shift{k:+d}"] = shifted(bd, yn, k, a.batch)
        V["shift_avg"] = np.mean([V["base"]] + [V[f"shift{k:+d}"] for k in a.shifts], 0)
        V["flip_shift"] = 0.5 * (V["shift_avg"] + np.mean([V["neg"]] + [-shifted(bd, -yn, k, a.batch) for k in a.shifts], 0))
        V["unroll2"] = predict(bd, V["base"], a.batch)
        for name, v in V.items():
            m = fd.metrics(v, xc, yn); res[f"{tag}:{name}"] = {k: float(q.mean()) for k, q in m.items()}
            print(f"{tag:28s} {name:12s} {fd.fmt(m)}", flush=True)
        np.savez_compressed(a.out / f"tta_{tag}.npz", clean=xc, noisy=yn, **V)
        json.dump(res, open(a.out / "tta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
