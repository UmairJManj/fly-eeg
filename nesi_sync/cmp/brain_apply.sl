#!/bin/bash -e
# Independent-dataset validation of the RETUNED FLY BRAIN: apply the finished brains (eog/emg/both) to the EEGBCI benchmark, re-score.
#SBATCH --job-name=brain-applyC
#SBATCH --account=aut04653
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/brain-applyC-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NB=/nesi/nobackup/aut04653/Manj/Fly; nvidia-smi --query-gpu=name --format=csv,noheader
for s in ${SUBJECTS:-1 2}; do
  echo "== $(date) subject $s"
  uv run --no-sync python ../fly-eeg/fly_brain_apply.py --brain $ROOT/brain/${EOG:-eog_D_biaslow} --brain $ROOT/brain/${EMG:-emg_D_biaslow} --brain $ROOT/brain/${BOTH:-both_D_biaslow} \
      --npz $NB/cmp_data/S$s.npz --out $NB/cmp_data --keys ${KEYS:-noisy_eog_+0 noisy_emg_+0 noisy_both_+0}
done
echo "== $(date) re-score"; cd $ROOT/cmp && .venv/bin/python run_compare.py 1 2 3 4
echo "== $(date) done"
