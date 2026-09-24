"""Build a readout-v2 cache whose only feature is the RAW noisy signal (no brain), for any data protocol:
  python make_raw_cache.py OUT_DIR --artifact eog --protocol whole --aug 12 --artifact-split disjoint [--n-train 4000 --n-test 500]"""
import argparse, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_eeg_denoise import load_data
p = argparse.ArgumentParser(); p.add_argument("out", type=Path)
p.add_argument("--artifact", default="eog"); p.add_argument("--protocol", default="whole"); p.add_argument("--aug", type=int, default=12)
p.add_argument("--artifact-split", default="disjoint"); p.add_argument("--n-train", type=int, default=4000); p.add_argument("--n-test", type=int, default=500)
p.add_argument("--extra-train", type=int, default=0)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
x, y, snr = load_data(a, np.random.default_rng(0))
for k in ("tr", "va", "te"):
    np.save(a.out / f"{k}_clean.npy", x[k]); np.save(a.out / f"{k}_noisy.npy", y[k])
    S = np.lib.format.open_memmap(a.out / f"{k}_states.npy", "w+", np.float16, (len(y[k]), y[k].shape[1], 1)); S[:] = y[k][:, :, None]; S.flush()
    print(k, y[k].shape, flush=True)
if snr is not None: np.save(a.out / "te_snr.npy", snr["te"])
print("cache written to", a.out)
