"""Export EEGDiR's processed Klados SS2016 EEG/EOG set (arrow, x = clean, y = contaminated, std) into a readout-v2 raw cache:
{tr,va,te}_{clean,noisy}.npy + {k}_states.npy (raw noisy as the single feature). te = HF test split, va = last 10 % of HF train."""
import numpy as np, sys
from pathlib import Path
from datasets import load_from_disk
SRC = Path("/nesi/nobackup/aut04653/Manj/Fly/ss2016/SS2016_EOG"); OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/nesi/nobackup/aut04653/Manj/Fly/states_ss2016_eog_raw"); OUT.mkdir(parents=True, exist_ok=True)
tr = load_from_disk(str(SRC / "train")).with_format("numpy"); te = load_from_disk(str(SRC / "test")).with_format("numpy")
def arr(d, k): return np.asarray(d[k], dtype=np.float32)
xtr, ytr = arr(tr, "x"), arr(tr, "y"); n_va = len(xtr) // 10
splits = {"tr": (xtr[:-n_va], ytr[:-n_va]), "va": (xtr[-n_va:], ytr[-n_va:]), "te": (arr(te, "x"), arr(te, "y"))}
for k, (x, y) in splits.items():
    np.save(OUT / f"{k}_clean.npy", x); np.save(OUT / f"{k}_noisy.npy", y)
    S = np.lib.format.open_memmap(OUT / f"{k}_states.npy", "w+", np.float16, (len(y), y.shape[1], 1)); S[:] = y[:, :, None]; S.flush()
    print(k, x.shape, "clean std", x.std().round(3), "noisy std", y.std().round(3), flush=True)
print("SS2016 raw cache ->", OUT)
