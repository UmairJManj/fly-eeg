#!/bin/bash -e
# Multi-dataset benchmark, published SOTA side (2026-09-29): D3 PhysioNet motion-artifact EEG, D4 EEGdenoiseNet + MIT-BIH ECG.
# 1) export both datasets  2) zero-shot of EEGdenoiseNet-trained SOTA + fly brains on D3  3) in-domain SOTA
# (EEGDiR, complex CNN, simple CNN; 3 seeds, saved) on D3 and D4. Fly brains in-domain run as lane units (brain/units.txt).
#SBATCH --job-name=fly-extbench
#SBATCH --account=aut04653
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-extbench-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; NB=/nesi/nobackup/aut04653/Manj/Fly; B=$ROOT/brain
MOT=$NB/states_motion_raw; ECG=$NB/states_ecg_whole_aug12_disjoint_raw
[ -f $MOT/te_clean.npy ] && [ -f $ROOT/fly-eeg/data/ECG_all_epochs.npy ] || { echo "== $(date) export"; $ROOT/cmp/.venv/bin/python $ROOT/cmp/ext_export.py; }
source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ -d $ECG ] || { echo "== $(date) ECG cache"; uv run --no-sync python ../fly-eeg/make_raw_cache.py $ECG --artifact ecg --aug 12 --artifact-split disjoint; }
[ "${PART:-all}" = all -o "${PART}" = zs ] && { echo "== $(date) D3 motion ZERO-SHOT (EEGdenoiseNet-trained SOTA + brains)"
uv run --no-sync python ../cmp/zeroshot_eval.py $MOT --models $NB/tcn_models/sota_eog --models $NB/tcn_models/sota_emg \
  --brain $B/eog_N_jo48_s1 --brain $B/eog_N_jo48_s2 --brain $B/emg_N_jo48 --brain $B/eog_N_jo48_shuffled \
  --out $ROOT/cmp/results/motion_zeroshot.json || true; }
[ "${PART:-all}" = all -o "${PART}" = motion ] && { echo "== $(date) D3 motion IN-DOMAIN SOTA"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $MOT --tag motion --configs "ccnn+s20000,scnn+s20000,eegdir+s20000" --seeds 3 --save-model $NB/tcn_models/motion || true; }
[ "${PART:-all}" = all -o "${PART}" = ecg ] && { echo "== $(date) D4 ECG IN-DOMAIN SOTA"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $ECG --tag ecg --configs "ccnn+s20000,scnn+s20000,eegdir+s20000" --seeds 3 --save-model $NB/tcn_models/sota_ecg || true; }
echo "== $(date) done"
