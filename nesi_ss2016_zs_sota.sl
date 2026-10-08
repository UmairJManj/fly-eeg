#!/bin/bash -e
# Zero-shot on SS2016 for the leak-free saved SOTA models (ccnn/scnn/eegdir), rerun of 9300636 which timed out at 2 h.
#SBATCH --job-name=fly-ss2016zs2
#SBATCH --account=aut04653
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-ss2016zs2-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NB=/nesi/nobackup/aut04653/Manj/Fly
echo "== $(date) ZERO-SHOT on SS2016 test: leak-free saved SOTA models (sota_eog) + nobrain TCN"
uv run --no-sync python ../cmp/zeroshot_eval.py $NB/states_ss2016_eog_raw --models $NB/tcn_models/sota_eog --models $NB/tcn_models/nobrain_eog --out $ROOT/cmp/results/ss2016_zeroshot_sota.json
echo "== $(date) done"
