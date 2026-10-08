#!/bin/bash -e
#SBATCH --job-name=fly-brain-ens
#SBATCH --account=aut04653
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-brain-ens-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
B=$ROOT/brain
echo "== $(date) 4-brain average on the EOG test set (shared split)"
uv run --no-sync python ../cmp/brain_ensemble_eval.py --brain $B/eog_D_biaslow --brain $B/eog_E_aug4 --brain $B/eog_D_seed2 --brain $B/eog_D_biaslow_tbptt128 --out $ROOT/cmp/results/brain_ensemble_eog.json
echo "== $(date) done"
