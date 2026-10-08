#!/bin/bash -e
# Leak-free whole protocol (disjoint artifact pools): raw-signal cache + CNN/TCN baselines.
#   ARTIFACT=eog sbatch [-p P --gpus-per-node G:1] fly-eeg/nesi_fly_leakfree.sl
#SBATCH --job-name=fly-sotasave
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-sotasave-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; NB=/nesi/nobackup/aut04653/Manj/Fly; C=$NB/states_${ARTIFACT}_whole_aug12_disjoint_raw
[ -d $C ] || { echo "== $(date) cache"; uv run --no-sync python ../fly-eeg/make_raw_cache.py $C --artifact $ARTIFACT --aug 12 --artifact-split disjoint; }
echo "== $(date) baselines (leak-free)"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $C --tag sotasave --configs "${CONFIGS:-ccnn+s20000,scnn+s20000,eegdir+s20000}" --seeds 3 --save-model $NB/tcn_models/sota_$ARTIFACT
echo "== $(date) done"
