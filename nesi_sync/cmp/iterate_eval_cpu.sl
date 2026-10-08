#!/bin/bash -e
#SBATCH --job-name=fly-iter
#SBATCH --account=aut04653
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-iter-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=32
uv run --no-sync python ../cmp/iterate_eval.py $ROOT/brain/eog_Y_res_bidir --steps 3 --alphas 1.0 0.5 --n 100
