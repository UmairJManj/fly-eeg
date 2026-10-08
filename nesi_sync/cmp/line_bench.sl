#!/bin/bash -e
# 50 Hz line-noise benchmark: published SOTA (EEGDiR, complex CNN, simple CNN) trained on EEGdenoiseNet clean EEG + synthetic mains.
#SBATCH --job-name=fly-linebench
#SBATCH --account=aut04653
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-linebench-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; NB=/nesi/nobackup/aut04653/Manj/Fly; C=$NB/states_line_whole_aug12_disjoint_raw
source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ -d $C ] || uv run --no-sync python ../fly-eeg/make_raw_cache.py $C --artifact line --aug 12 --artifact-split disjoint
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $C --tag line --configs "ccnn+s20000,scnn+s20000,eegdir+s20000" --seeds 3 --save-model $NB/tcn_models/sota_line
echo "== $(date) done"
