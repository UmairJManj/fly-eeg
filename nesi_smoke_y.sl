#!/bin/bash -e
#SBATCH --job-name=fly-smoke-y
#SBATCH --account=aut04653
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-smoke-y-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$ROOT/brain/smoke_y; rm -rf "$OUT"; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader; done ) > $OUT/mem.log &
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact eog --protocol whole --readout random --n-train 256 --n-test 32 --passes 2 --train-len 256 --lr 3e-3 --lr-bias 3e-4 --warmup 20 --cosine --no-ref-ridge --no-fir --batch 32 --micro-batch 16 --jo-delays 48 --filterbank 8 --bidir --residual --train-readout --spec-loss 0.5 --cc-loss 0.5 --out "$OUT"
echo "== $(date) smoke done; peak mem MiB: $(sort -n $OUT/mem.log | tail -1)"
