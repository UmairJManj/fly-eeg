#!/bin/bash -e
# Track D: single-channel denoisers (retuned fly brains, TCN, fly reservoir+TCN, EEGDiR/CNNs/Transformer/LSTM/FCNN), channel-wise.
#SBATCH --job-name=fly-erpgpu
#SBATCH --account=aut04653
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-erpgpu-%A_%a.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd $ROOT/nfly; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
read DS SUB < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ../cmp/erp_subjects.txt)
nvidia-smi --query-gpu=name --format=csv,noheader; echo "== $(date) $DS $SUB"
uv run --no-sync python ../cmp/erp_bench.py clean --ds $DS --sub $SUB --set gpu ${METHODS:+--methods $METHODS}; echo "== $(date) done"
