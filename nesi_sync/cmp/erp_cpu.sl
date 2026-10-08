#!/bin/bash -e
# Track D (real EEG, Delorme 2023 metric): load + HP 0.5 Hz + spatial cleaners (ICA+ICLabel, ASR, GEDAI). Array over erp_subjects.txt.
#SBATCH --job-name=fly-erpcpu
#SBATCH --account=aut04653
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --partition=milan
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-erpcpu-%A_%a.log
ROOT=/nesi/project/aut04653/Manj/Fly; cd $ROOT/cmp; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
read DS SUB < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" erp_subjects.txt)
echo "== $(date) $DS $SUB"; .venv/bin/python erp_bench.py clean --ds $DS --sub $SUB --set cpu; echo "== $(date) done"
