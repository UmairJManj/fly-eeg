#!/bin/bash
# Persistent lane for the fly-brain iteration campaign. Arms live in $ROOT/brain/units.txt, one per line:
#   <tag>  <fly_eeg_brain.py args except --out/--batch>        (lines starting with # are ignored)
# A lane claims the first unclaimed, unfinished arm (mkdir lock), runs it (resumable from ckpt.pt), plots it,
# then re-reads the file for the next arm. With nothing to do it idles (re-checking every 5 min) so new arms
# appended to the file start immediately. Debug-QOS lanes self-requeue 5 min before the wall (debug max is 1 h since 2026-10-01).
#SBATCH --job-name=fly-lane
#SBATCH --account=aut04653
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-lane-%j.log
set -uo pipefail
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN=$ROOT/brain; LOCKS=$RUN/.locks; DONE=$RUN/.done; FAIL=$RUN/.fail; UNITS_FILE=$RUN/units.txt; mkdir -p "$LOCKS" "$DONE" "$FAIL"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *L4*) BATCH=32;; *) BATCH=64;; esac      # bigger batches: fewer steps per pass (crop training keeps memory in check)
T_START=$(date +%s); WALL=$(( ${LANE_HOURS:-11} * 3600 )); [ "${SLURM_JOB_QOS:-}" = "debug" ] && WALL=6300
GUARD_PID=""; if [ "${SLURM_JOB_QOS:-}" = "debug" ]; then ( sleep $(( $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 300 )) && echo "[guard] requeueing before wall" && scontrol requeue "$SLURM_JOB_ID" ) & GUARD_PID=$!; fi
for lock in "$LOCKS"/*; do [ -d "$lock" ] || continue; [ "$(cat "$lock/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -rf "$lock"; done
claim() { local c=$1 lock=$LOCKS/$1; [ -f "$DONE/$c" ] && return 1; [ -f "$RUN/$c/summary.json" ] && return 1; [ -f "$FAIL/$c.2" ] && return 1
  if mkdir "$lock" 2>/dev/null; then echo "$SLURM_JOB_ID" > "$lock/owner"; return 0; fi
  local owner; owner=$(cat "$lock/owner" 2>/dev/null || true)
  if [ -z "$owner" ] || ! squeue -h -j "$owner" -t RUNNING 2>/dev/null | grep -q .; then rm -rf "$lock"
    if mkdir "$lock" 2>/dev/null; then echo "$SLURM_JOB_ID" > "$lock/owner"; sleep 5; [ "$(cat "$lock/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ] && return 0; fi; fi; return 1; }
release() { rm -rf "$LOCKS/$1"; }
mark_fail() { if [ -f "$FAIL/$1.1" ]; then touch "$FAIL/$1.2"; else touch "$FAIL/$1.1"; fi; }
echo "[lane] job=$SLURM_JOB_ID qos=${SLURM_JOB_QOS:-normal} gpu=$GPU batch=$BATCH"
while :; do
  CLAIMED=""; ARGS=""
  while read -r tag args; do [ -z "$tag" ] && continue; case "$tag" in \#*) continue;; esac
    [ -n "${ONLY:-}" ] && [ "$tag" != "$ONLY" ] && continue
    [ -z "${ALLOW_AUG:-}" ] && case " $args " in *" --batch "*) false;; *) true;; esac && case "$GPU" in *L4*) case " $args " in *" --aug "*) case " $args " in *" --passes "[1-5]" "*) ;; *) continue;; esac;; esac;; esac    # L4 is 3x slower per step: only short screening aug arms (<=5 passes) unless forced
    if claim "$tag"; then CLAIMED=$tag; ARGS=$args; break; fi; done < "$UNITS_FILE"
  if [ -z "$CLAIMED" ]; then
    if [ $(( $(date +%s) - T_START )) -gt $(( WALL - 600 )) ]; then echo "[lane] no work and wall near; exiting"; break; fi
    echo "[lane] idle $(date +%T)"; sleep 300; continue; fi
  OUT=$RUN/$CLAIMED; mkdir -p "$OUT"; B=$BATCH; case " $ARGS " in *" --batch "*) B="";; esac
  NB=/nesi/nobackup/aut04653/Manj/Fly/brain/$CLAIMED; mkdir -p "$NB"      # checkpoints live on nobackup (project quota); a symlink keeps the path
  [ -e "$OUT/ckpt.pt" ] && [ ! -L "$OUT/ckpt.pt" ] && mv "$OUT/ckpt.pt" "$NB/ckpt.pt"; [ -L "$OUT/ckpt.pt" ] || ln -s "$NB/ckpt.pt" "$OUT/ckpt.pt"
  echo "[lane] === $CLAIMED  ($ARGS)  $(date +%T) ==="
  if uv run --no-sync python ../fly-eeg/fly_eeg_brain.py $ARGS ${B:+--batch $B} --out "$OUT" >> "$OUT/train_${SLURM_JOB_ID}.log" 2>&1 \
     && uv run --no-sync python ../fly-eeg/fly_eeg_brain_plots.py "$OUT" >> "$OUT/train_${SLURM_JOB_ID}.log" 2>&1; then
    touch "$DONE/$CLAIMED"; echo "[lane] done $CLAIMED  $(date +%T): $(grep 'TRAINED brain' "$OUT/train_${SLURM_JOB_ID}.log" | tail -1)"
  else echo "[lane] FAILED: $CLAIMED"; mark_fail "$CLAIMED"; tail -5 "$OUT/train_${SLURM_JOB_ID}.log"; fi
  release "$CLAIMED"
done
[ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null; echo "[lane] exiting; done: $(ls "$DONE" 2>/dev/null | tr '\n' ' ')"; exit 0
