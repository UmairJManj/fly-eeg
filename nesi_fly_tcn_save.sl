#!/bin/bash -e
# Retrain the final TCN recipe on the whole-protocol aug12 cache and SAVE the models (for fly_apply.py / Track C).
#   ARTIFACT=eog sbatch [--partition P --gpus-per-node G:1] fly-eeg/nesi_fly_tcn_save.sl
#SBATCH --job-name=fly-tcn-save
#SBATCH --account=aut04653
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-tcn-save-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; NB=/nesi/nobackup/aut04653/Manj/Fly
echo "== $(date) smoke (save path) on the smoke cache"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $NB/states_eog_centered_smoke --tag savesmoke --configs "tcn+d2+w32+s100" --seeds 1 --tcn-steps 100 --save-model $NB/tcn_models/smoke
ls $NB/tcn_models/smoke
echo "== $(date) full: $ARTIFACT whole aug12, final recipe, 3 seeds, saving to $NB/tcn_models/$ARTIFACT"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $NB/states_${ARTIFACT}_whole_aug12 --tag save --configs "tcn+d9+w384+p0.3+s60000" --seeds 3 --save-model $NB/tcn_models/$ARTIFACT
echo "== $(date) done"; ls -la $NB/tcn_models/$ARTIFACT
