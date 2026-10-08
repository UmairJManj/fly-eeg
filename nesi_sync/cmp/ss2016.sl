#!/bin/bash -e
# Independent dataset (Klados SS2016 EEG/EOG, via EEGDiR): export, zero-shot eval of saved deep models + retuned brains, then in-domain baselines.
#SBATCH --job-name=fly-ss2016
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-ss2016-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; NB=/nesi/nobackup/aut04653/Manj/Fly; C=$NB/states_ss2016_eog_raw
[ -f $C/te_states.npy ] || { echo "== $(date) export"; $ROOT/cmp/.venv/bin/python $ROOT/cmp/ss2016_export.py $C; }
source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(date) ZERO-SHOT on SS2016 test (models trained on EEGdenoiseNet EOG)"
uv run --no-sync python ../cmp/zeroshot_eval.py $C --models $NB/tcn_models/leakfree_eog --models $NB/tcn_models/sota_eog --brain $ROOT/brain/eog_D_biaslow --brain $ROOT/brain/eog_E_aug4 --out $ROOT/cmp/results/ss2016_zeroshot.json || true
echo "== $(date) IN-DOMAIN baselines trained on SS2016 train"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $C --tag ss2016 --configs "ccnn+s20000,scnn+s20000,eegdir+s20000" --seeds 3 --save-model $NB/tcn_models/ss2016_eog
echo "== $(date) done"
