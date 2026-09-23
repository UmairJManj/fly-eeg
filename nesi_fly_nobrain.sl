#!/bin/bash -e
# Control: the final TCN readout trained on the RAW noisy signal only (no brain). Builds a "states" cache whose
# single feature is the raw-input column of the real cache, then runs the identical readout recipe.
#   ARTIFACT=eog sbatch [--partition P --gpus-per-node G:1] fly-eeg/nesi_fly_nobrain.sl
#SBATCH --job-name=fly-nobrain
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-nobrain-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; NB=/nesi/nobackup/aut04653/Manj/Fly; SRC=$NB/states_${ARTIFACT}_whole_aug12; DST=${SRC}_nobrain; mkdir -p "$DST"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== $(date) building no-brain cache $DST from $SRC"
uv run --no-sync python - "$SRC" "$DST" <<'PY'
import sys, numpy as np, shutil
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for k in ("tr", "va", "te"):
    S = np.load(src / f"{k}_states.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(dst / f"{k}_states.npy", "w+", np.float16, (S.shape[0], S.shape[1], 1))
    for b in range(0, S.shape[0], 500):
        out[b:b + 500] = S[b:b + 500, :, -1:]        # last column = the raw noisy input (skip feature)
    out.flush()
    for f in (f"{k}_clean.npy", f"{k}_noisy.npy"):
        shutil.copy(src / f, dst / f)
    print(k, S.shape, "->", out.shape, flush=True)
PY
echo "== $(date) readout (identical recipe to the final result, no brain)"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py "$DST" --tag nobrain --configs "tcn+d9+w384+p0.3+s60000" --seeds 3 --save-model $NB/tcn_models/nobrain_$ARTIFACT
echo "== $(date) done"
