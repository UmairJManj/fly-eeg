#!/bin/bash -e
#SBATCH --job-name=fly-smoke-jo
#SBATCH --account=aut04653
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-smoke-jo-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$ROOT/brain/smoke_jo16; rm -rf "$OUT"; mkdir -p "$OUT"
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact eog --protocol whole --readout random --n-train 64 --n-test 16 --passes 2 --train-len 256 --lr 3e-3 --lr-bias 3e-4 --warmup 20 --cosine --no-ref-ridge --no-fir --batch 8 --jo-delays 16 --out "$OUT"
echo "== $(date) smoke done"; grep "TRAINED brain" $OUT/train_*.log 2>/dev/null || grep "TRAINED brain" $ROOT/logs/fly-smoke-jo-$SLURM_JOB_ID.log
