#!/bin/bash -e
# Readout-only experiments on cached brain states.
# Submit: ARTIFACT=eog TAG=r1 CONFIGS=ridge5,ridge9,ridge17,mlp9,tcn sbatch [--partition P --gpus-per-node G:1] fly-eeg/nesi_fly_readout.sl
#SBATCH --job-name=fly-rd
#SBATCH --account=aut04653
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-rd-%j.log

ROOT=/nesi/project/aut04653/Manj/Fly
source $ROOT/env.sh
cd "$ROOT/nfly"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; PROTOCOL=${PROTOCOL:-centered}; TAG=${TAG:-r1}
STATES=${STATES:-/nesi/nobackup/aut04653/Manj/Fly/states_${ARTIFACT}_${PROTOCOL}}
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== $(date) artifact=$ARTIFACT tag=$TAG configs=${CONFIGS:-default} extra=${EXTRA:-}"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py "$STATES" --tag "$TAG" ${CONFIGS:+--configs $CONFIGS} ${EXTRA:-} ${RD_EXTRA:-}
echo "== $(date) done"
