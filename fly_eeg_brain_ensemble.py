"""Average the test predictions of several brain runs ("three flies") and score the ensemble.
  python fly_eeg_brain_ensemble.py DIR1 DIR2 ...   (each holds results.npz from fly_eeg_brain.py on the same split)"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_eeg_denoise import metrics, fmt
dirs = [Path(d) for d in sys.argv[1:]]; Z = [np.load(d / "results.npz") for d in dirs]
clean, noisy = Z[0]["clean"], Z[0]["noisy"]
assert all(np.allclose(z["clean"], clean) for z in Z), "runs must share the test split"
for d, z in zip(dirs, Z): print(f"{d.name:30s} {fmt(metrics(z['xhat_brain'], clean, noisy))}")
ens = np.mean([z["xhat_brain"] for z in Z], 0); m = metrics(ens, clean, noisy)
print(f"{'ENSEMBLE of ' + str(len(Z)):30s} {fmt(m)}")
if "xhat_fir" in Z[0]: print(f"{'FIR reference':30s} {fmt(metrics(Z[0]['xhat_fir'], clean, noisy))}")
