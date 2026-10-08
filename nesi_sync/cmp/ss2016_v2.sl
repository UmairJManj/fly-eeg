#!/bin/bash -e
# INDEPENDENT test v2 (2026-09-29): Klados SS2016 EEG/EOG. Everything trained on EEGdenoiseNet EOG only, applied unchanged
# (zero-shot): published SOTA (EEGDiR, ccnn, scnn) + round-N fly brains. Also in-domain SOTA trained on SS2016 (upper bound).
# Metrics as in the papers: CC, RRMSE_t, RRMSE_s, SNR gain.
#SBATCH --job-name=fly-ss2016v2
#SBATCH --account=aut04653
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-ss2016v2-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; NB=/nesi/nobackup/aut04653/Manj/Fly; C=$NB/states_ss2016_eog_raw; B=$ROOT/brain
source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run --no-sync python ../cmp/zeroshot_eval.py $C --models $NB/tcn_models/sota_eog --models $NB/tcn_models/ss2016_eog \
  --brain $B/eog_J_jo48_disjoint --brain $B/eog_N_jo48_s1 --brain $B/eog_N_jo48_s2 --brain $B/eog_N_jo48_fastdn \
  --brain $B/eog_N_jo48_shuffled --brain $B/eog_N_jo48_shuffled_s1 --out $ROOT/cmp/results/ss2016_v2.json
echo "== $(date) done"
