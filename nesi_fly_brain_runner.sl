#!/bin/bash
# Lane-agnostic work-queue runner for fly_eeg_brain.py arms (pattern: Compute/run/ua_v3_runner.sl).
# Submit several lanes on whatever GPU is free; each lane claims an unfinished arm (mkdir lock), runs it
# (resumable from OUT/ckpt.pt), then claims the next. Debug-QOS lanes self-requeue before the 2 h wall.
#   debug lane : sbatch --qos=debug -p milan --gpus-per-node=a100:1 fly-eeg/nesi_fly_brain_runner.sl
#   a100/h100/l4 lanes: sbatch -p milan --gpus-per-node=a100:1 | -p genoa --gpus-per-node=h100:1 | l4:1
#SBATCH --job-name=fly-lane
#SBATCH --account=aut04653
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-lane-%j.log
set -uo pipefail
ROOT=/nesi/project/aut04653/Manj/Fly
source $ROOT/env.sh
cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN=$ROOT/brain; LOCKS=$RUN/.locks; DONE=$RUN/.done; FAIL=$RUN/.fail; mkdir -p "$LOCKS" "$DONE" "$FAIL"
UNITS=(${UNITS:-eog:ridge0 eog:random:shuffled})
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *L4*) BATCH=16;; *) BATCH=32;; esac
GUARD_PID=""
if [ "${SLURM_JOB_QOS:-}" = "debug" ]; then ( sleep 6480 && echo "[guard] requeueing before wall" && scontrol requeue "$SLURM_JOB_ID" ) & GUARD_PID=$!; fi
for lock in "$LOCKS"/*; do [ -d "$lock" ] || continue; [ "$(cat "$lock/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -rf "$lock"; done
claim() { local c=$1 lock=$LOCKS/$1; [ -f "$DONE/$c" ] && return 1; [ -f "$FAIL/$c.2" ] && return 1
  if mkdir "$lock" 2>/dev/null; then echo "$SLURM_JOB_ID" > "$lock/owner"; return 0; fi
  local owner; owner=$(cat "$lock/owner" 2>/dev/null || true)
  if [ -z "$owner" ] || ! squeue -h -j "$owner" -t RUNNING 2>/dev/null | grep -q .; then rm -rf "$lock"
    if mkdir "$lock" 2>/dev/null; then echo "$SLURM_JOB_ID" > "$lock/owner"; sleep 5; [ "$(cat "$lock/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ] && return 0; fi; fi; return 1; }
release() { rm -rf "$LOCKS/$1"; }
mark_fail() { if [ -f "$FAIL/$1.1" ]; then touch "$FAIL/$1.2"; else touch "$FAIL/$1.1"; fi; }
echo "[lane] job=$SLURM_JOB_ID qos=${SLURM_JOB_QOS:-normal} gpu=$GPU batch=$BATCH units=${UNITS[*]}"
if [ -n "${TAKEOVER:-}" ] && squeue -h -j "$TAKEOVER" 2>/dev/null | grep -q .; then   # take over a slow plain job: cancel it, resume its arm from ckpt.pt
  echo "[lane] taking over job $TAKEOVER (cancelling it; its arm resumes here from its checkpoint)"; scancel "$TAKEOVER"; sleep 20; fi
while :; do
  CLAIMED=""; for c in "${UNITS[@]}"; do if claim "$c"; then CLAIMED=$c; break; fi; done
  [ -z "$CLAIMED" ] && { echo "[lane] no claimable work"; break; }
  IFS=: read -r ART RD SH <<< "$CLAIMED"; TAG=${ART}_whole_${RD}${SH:+_shuffled}; OUT=$RUN/$TAG; mkdir -p "$OUT"
  echo "[lane] === $CLAIMED -> $OUT  $(date +%T) ==="
  if uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact $ART --protocol whole --readout $RD --n-train 4000 --n-test 500 \
       --batch $BATCH --passes ${PASSES:-30} --out "$OUT" ${SH:+--shuffle} >> "$OUT/train_${SLURM_JOB_ID}.log" 2>&1 \
     && uv run --no-sync python ../fly-eeg/fly_eeg_brain_plots.py "$OUT" >> "$OUT/train_${SLURM_JOB_ID}.log" 2>&1; then
    touch "$DONE/$CLAIMED"; echo "[lane] done $CLAIMED  $(date +%T)"; grep "TRAINED brain" "$OUT/train_${SLURM_JOB_ID}.log" | tail -1
  else echo "[lane] FAILED: $CLAIMED"; mark_fail "$CLAIMED"; tail -5 "$OUT/train_${SLURM_JOB_ID}.log"; fi
  release "$CLAIMED"
done
[ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null
echo "[lane] exiting; done: $(ls "$DONE" 2>/dev/null | tr '\n' ' ')"; exit 0
