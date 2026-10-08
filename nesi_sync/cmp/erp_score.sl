#!/bin/bash -e
#SBATCH --job-name=fly-erpscore
#SBATCH --account=aut04653
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --partition=milan
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-erpscore-%j.log
cd /nesi/project/aut04653/Manj/Fly/cmp; export OMP_NUM_THREADS=16; .venv/bin/python erp_bench.py score --reps ${REPS:-2000}
