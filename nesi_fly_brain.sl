#!/bin/bash -e
# Train the fly brain itself to denoise (fixed readout). Submit from $ROOT:
#   ARTIFACT=eog READOUT=random [SHUFFLE=1] [SMOKE=1] [TAG=...] sbatch [--partition P --gpus-per-node G:1] fly-eeg/nesi_fly_brain.sl
#SBATCH --job-name=fly-brain
#SBATCH --account=aut04653
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-brain-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly
source $ROOT/env.sh
cd "$ROOT/nfly"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; PROTOCOL=${PROTOCOL:-whole}; READOUT=${READOUT:-random}
TAG=${TAG:-${ARTIFACT}_${PROTOCOL}_${READOUT}${SHUFFLE:+_shuffled}${SMOKE:+_smoke}}
OUT=$ROOT/brain/$TAG; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ -n "$SMOKE" ]; then NTR=64; NTE=16; PASSES=2; BATCH=${BATCH:-8}; else NTR=${NTR:-4000}; NTE=${NTE:-500}; PASSES=${PASSES:-30}; BATCH=${BATCH:-32}; fi
echo "== $(date) artifact=$ARTIFACT protocol=$PROTOCOL readout=$READOUT shuffle=${SHUFFLE:-0} smoke=${SMOKE:-0} n-train=$NTR batch=$BATCH passes=$PASSES out=$OUT"
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact $ARTIFACT --protocol $PROTOCOL --readout $READOUT \
    --n-train $NTR --n-test $NTE --batch $BATCH --passes $PASSES --out "$OUT" ${SHUFFLE:+--shuffle} ${EXTRA:-}
echo "== $(date) plots"
uv run --no-sync python ../fly-eeg/fly_eeg_brain_plots.py "$OUT"
echo "== $(date) done"
