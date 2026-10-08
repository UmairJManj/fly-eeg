#!/bin/bash -e
#SBATCH --job-name=fly-smoke-diff
#SBATCH --account=aut04653
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-smoke-diff-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=32
OUT=$ROOT/brain/smoke_diff; rm -rf "$OUT"; mkdir -p "$OUT"
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact eog --protocol whole --readout random --n-train 40 --n-test 8 --passes 1 --train-len 128 --batch 8 --eval-batch 8 --lr 3e-3 --lr-bias 3e-4 --warmup 2 --cosine --no-ref-ridge --no-fir --jo-delays 48 --residual --bidir --unroll 2 --snr-lo -15 --snr-hi 10 --device cpu --out "$OUT"
echo "== $(date) smoke done"
