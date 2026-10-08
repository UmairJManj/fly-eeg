#!/bin/bash -e
#SBATCH --job-name=fly-brain-diag
#SBATCH --account=aut04653
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-brain-diag-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(date) bottleneck diagnostics (fixed wire / bidirectional / ridge readout on the trained brain)"
uv run --no-sync python ../cmp/brain_diag.py --brain $ROOT/brain/eog_E_aug4 --brain $ROOT/brain/eog_D_biaslow --brain $ROOT/brain/emg_D_biaslow --out $ROOT/cmp/results/brain_diag.json
echo "== $(date) done"
