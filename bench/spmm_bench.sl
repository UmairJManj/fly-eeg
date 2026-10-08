#!/bin/bash -e
#SBATCH --job-name=fly-bench
#SBATCH --account=aut04653
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-bench-%j.log
source /nesi/project/aut04653/Manj/Fly/env.sh; cd /nesi/project/aut04653/Manj/Fly/nfly
nvidia-smi --query-gpu=name --format=csv,noheader
uv run --no-sync python ../fly-eeg/bench/spmm_bench.py 32
uv run --no-sync python ../fly-eeg/bench/spmm_bench.py 64
cd /nesi/project/aut04653/Manj/Fly/nfly
uv run --no-sync python ../fly-eeg/bench/real_check.py ../brain/eog_Y_res_bidir 32
uv run --no-sync python ../fly-eeg/bench/real_check.py ../brain/emg_W_rewire_env 32
