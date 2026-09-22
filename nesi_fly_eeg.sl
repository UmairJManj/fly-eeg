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
AUG=${AUG:-1}                            # centered protocol: draws per clean train/val record
XTR=${XTR:-0}                            # centered protocol: extra train records after the test block (-1 = all)
STATES=/nesi/nobackup/aut04653/Manj/Fly/states_${ARTIFACT}_${PROTOCOL}${NTR:+_n$NTR}$([ "$AUG" -gt 1 ] && echo _aug$AUG || true)$([ "$XTR" != 0 ] && echo _xtr$XTR || true)${SMOKE:+_smoke}
mkdir -p "$(dirname "$STATES")"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ -n "$SMOKE" ]; then
    NTR=${NTR:-60}; NTE=4; STEPS=200
elif [ "$PROTOCOL" = centered ]; then
    NTR=${NTR:-2400}; NTE=150; STEPS=6000
else
    NTR=${NTR:-4000}; NTE=500; STEPS=6000
fi
EXTRA=""
[ "$PROTOCOL" = centered ] && EXTRA="--no-shuffle-control"

echo "== $(date) artifact=$ARTIFACT protocol=$PROTOCOL n-train=$NTR n-test=$NTE aug=$AUG xtr=$XTR smoke=${SMOKE:-0} states=$STATES"
uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --protocol $PROTOCOL --artifact $ARTIFACT \
    --n-train $NTR --n-test $NTE --batch 100 --aug $AUG --extra-train $XTR $EXTRA --save-states "$STATES"
echo "== $(date) readout"
if [ -n "$CONFIGS" ]; then      # readout suite v2 (TAG, CONFIGS, RD_EXTRA as in nesi_fly_readout.sl)
    uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py "$STATES" --tag "${TAG:-r1}" --configs "$CONFIGS" ${RD_EXTRA:-} ${SMOKE:+--seeds 1 --tcn-steps 300 --mlp-steps 300}
else
    uv run --no-sync python ../fly-eeg/fly_eeg_readout.py "$STATES" --steps $STEPS
fi
echo "== $(date) done"
