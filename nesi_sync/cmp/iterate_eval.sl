#!/bin/bash -e
#SBATCH --job-name=fly-iter
#SBATCH --account=aut04653
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-iter-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"
uv run --no-sync python ../cmp/iterate_eval.py $ROOT/brain/eog_Y_res_bidir --steps 4 --alphas 1.0 0.5
