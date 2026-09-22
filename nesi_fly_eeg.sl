#!/bin/bash -e
# Slurm script for NeSI (Mahuika). Layout: $ROOT/nfly and $ROOT/fly-eeg side by side.
# Submit:  ARTIFACT=eog sbatch fly-eeg/nesi_fly_eeg.sl      (PROTOCOL=centered|whole, SMOKE=1 for a tiny run)
#SBATCH --job-name=fly-eeg
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-eeg-%j.log

ROOT=/nesi/project/aut04653/Manj/Fly
source $ROOT/env.sh                      # uv + CUDA/12.6.3 modules, caches in the project (home is full)
cd "$ROOT/nfly"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ARTIFACT=${ARTIFACT:-eog}
PROTOCOL=${PROTOCOL:-centered}          # centered = SPAR-EEG protocol; whole = EEGdenoiseNet protocol
STATES=/nesi/nobackup/aut04653/Manj/Fly/states_${ARTIFACT}_${PROTOCOL}${SMOKE:+_smoke}
mkdir -p "$(dirname "$STATES")"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ -n "$SMOKE" ]; then
    NTR=60; NTE=4; STEPS=200
elif [ "$PROTOCOL" = centered ]; then
    NTR=2400; NTE=150; STEPS=6000
else
    NTR=4000; NTE=500; STEPS=6000
fi
EXTRA=""
[ "$PROTOCOL" = centered ] && EXTRA="--no-shuffle-control"

echo "== $(date) artifact=$ARTIFACT protocol=$PROTOCOL n-train=$NTR n-test=$NTE smoke=${SMOKE:-0}"
uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --protocol $PROTOCOL --artifact $ARTIFACT \
    --n-train $NTR --n-test $NTE --batch 100 $EXTRA --save-states "$STATES"
echo "== $(date) readout"
uv run --no-sync python ../fly-eeg/fly_eeg_readout.py "$STATES" --steps $STEPS
echo "== $(date) done"
