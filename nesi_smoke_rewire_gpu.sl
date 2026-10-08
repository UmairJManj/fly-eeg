#!/bin/bash -e
#SBATCH --job-name=fly-smoke-rwg
#SBATCH --account=aut04653
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-smoke-rwg-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$ROOT/brain/smoke_rw${V:+_$V}; rm -rf "$OUT"; mkdir -p "$OUT"; export OUT
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact emg --protocol whole --readout random --n-train 64 --n-test 16 --passes 1 --micro-batch 8 --train-len 128 --lr 3e-3 --lr-bias 3e-4 --warmup 2 --cosine --no-ref-ridge --no-fir --batch 8 --jo-delays 48 --residual --bidir $EXTRA --out "$OUT"
uv run --no-sync python - <<'P'
import sys; sys.path.insert(0, "../fly-eeg"); import numpy as np, torch
from fly_brain_apply import load_brain
import os; bd = load_brain(os.environ["OUT"], "cpu")
m = bd.model; print("reload OK;", "|w_new|>0:", int((m.w_new.abs() > 0).sum()) if hasattr(m, "w_new") else "-", "flipped signs:", int((torch.sign(m.sg) != m.sign).sum()) if hasattr(m, "sg") else "-")
with torch.no_grad(): print("out shape", tuple(bd(np.random.randn(2, 512).astype(np.float32)).shape))
P
echo "== $(date) smoke done"
