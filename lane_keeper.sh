#!/bin/bash
# One keeper cycle: (1) print new result lines from every arm / lane log (dedup via brain/.keeper_seen),
# (2) keep the lane pool full: 1 debug-QOS lane (if the debug slot is free) + NORMAL_TARGET normal lanes on
# whichever GPU type is free. Run in a loop by a Monitor.  Env: NORMAL_TARGET (default 3), NO_SUBMIT=1 to only report.
ROOT=/nesi/project/aut04653/Manj/Fly; cd $ROOT; SEEN=brain/.keeper_seen; touch $SEEN
report() { local src=$1 line=$2; local key; key=$(printf '%s' "$src|$line" | md5sum | cut -c1-16); grep -q "$key" $SEEN || { echo "$key" >> $SEEN; echo "[$src] ${line:0:190}"; }; }
for L in brain/*/train_*.log logs/fly-brain-9271509.log; do [ -f "$L" ] || continue; arm=$(basename $(dirname $L)); [ "$arm" = logs ] && arm=eog_whole_random
  grep -E "^pass [0-9]+/|TRAINED brain|early stop|Traceback|out of memory|resumed from" "$L" | while read -r line; do report "$arm" "$line"; done; done
for L in logs/fly-lane-*.log; do [ -f "$L" ] || continue; j=$(basename $L .log); grep -E "\[lane\] (===|done|FAILED)|taking over|requeueing" "$L" | while read -r line; do report "$j" "$line"; done; done
[ -n "${NO_SUBMIT:-}" ] && exit 0
n_norm=$(squeue -u $USER -h -n fly-lane -o "%q" | grep -vc debug); n_dbg=$(squeue -u $USER -h -o "%q" | grep -c debug)
if [ "$n_dbg" -eq 0 ]; then j=$(sbatch --parsable --qos=debug -p milan --gpus-per-node=a100:1 --time=02:00:00 fly-eeg/nesi_fly_lanes.sl 2>&1) && echo "[keeper] submitted debug lane $j"; fi
while [ "$n_norm" -lt "${NORMAL_TARGET:-3}" ]; do read PART TYPE < <(bash /nesi/project/aut04653/Manj/Foot/tools/pick_gpu.sh)
  j=$(sbatch --parsable -p $PART --gpus-per-node=$TYPE:1 fly-eeg/nesi_fly_lanes.sl 2>&1) && echo "[keeper] submitted normal lane $j on $PART/$TYPE"; n_norm=$((n_norm+1)); done
