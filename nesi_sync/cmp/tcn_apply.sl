#!/bin/bash -e
# Add the fly's row to Track C: brain + saved TCNs on every subject/artifact/SNR, then re-score everything.
#SBATCH --job-name=tcn-applyC
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/tcn-applyC-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NB=/nesi/nobackup/aut04653/Manj/Fly; nvidia-smi --query-gpu=name --format=csv,noheader
for s in ${SUBJECTS:-1 2 3 4}; do
  echo "== $(date) subject $s"
  uv run --no-sync python ../fly-eeg/fly_apply.py --models $NB/tcn_models/nobrain_eog --models $NB/tcn_models/nobrain_emg --models $NB/tcn_models/nobrain_both --npz $NB/cmp_data/S$s.npz --out $NB/cmp_data --batch 512 --prefix tcn
done
echo "== $(date) re-score with the fly row"; cd $ROOT/cmp && .venv/bin/python run_compare.py 1 2 3 4
echo "== $(date) done"
