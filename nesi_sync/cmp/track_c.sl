#!/bin/bash -e
#SBATCH --job-name=fly-trackC
#SBATCH --account=aut04653
#SBATCH --partition=milan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-trackC-%j.log
cd /nesi/project/aut04653/Manj/Fly/cmp; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MNE_DATA=/nesi/nobackup/aut04653/Manj/Fly/mne_data
[ -n "${SKIP_DATASET:-}" ] || { echo "== $(date) build dataset"; .venv/bin/python make_dataset.py ${SUBJECTS:-1 2 3 4}; }
echo "== $(date) compare"; .venv/bin/python run_compare.py ${SUBJECTS:-1 2 3 4}
echo "== $(date) done"
