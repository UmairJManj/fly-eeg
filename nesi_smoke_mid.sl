#!/bin/bash -e
#SBATCH --job-name=fly-smoke-mid
#SBATCH --account=aut04653
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-smoke-mid-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$ROOT/brain/smoke_mid; rm -rf "$OUT"; mkdir -p "$OUT"
ARGS="--device cpu --artifact emg --protocol whole --readout random --n-train 140 --n-test 16 --passes 2 --train-len 64 --lr 3e-3 --lr-bias 3e-4 --warmup 2 --cosine --no-ref-ridge --no-fir --batch 2 --jo-delays 48 --residual --out $OUT"
timeout 1200 bash -c "uv run --no-sync python ../fly-eeg/fly_eeg_brain.py $ARGS 2>&1 | tee $OUT/run1.log | grep --line-buffered -m1 'pass 1 step 51/' && sleep 1 && pkill -f smoke_mid/ ; true" || true
echo "== killed run 1; mid ckpt: $(ls $OUT/ckpt_mid.pt 2>/dev/null)"
uv run --no-sync python ../fly-eeg/fly_eeg_brain.py $ARGS 2>&1 | grep -E "resumed|pass [0-9]/2:|pass 1 step|TRAINED|Error|Traceback" | head -20
echo "== mid ckpt after: $(ls $OUT/ckpt_mid.pt 2>/dev/null || echo removed)"; echo "== $(date) smoke done"
