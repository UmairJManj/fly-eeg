#!/bin/bash -e
# Track C: channel-wise EEGDiR / complex CNN / simple CNN rows, then score them into compare.csv.
#SBATCH --job-name=fly-sotaC
#SBATCH --account=aut04653
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-sotaC-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NB=/nesi/nobackup/aut04653/Manj/Fly; nvidia-smi --query-gpu=name --format=csv,noheader
for s in ${SUBJECTS:-1 2 3 4}; do echo "== $(date) subject $s"; uv run --no-sync python ../cmp/sota_apply.py --npz $NB/cmp_data/S$s.npz --archs ${ARCHS:-eegdir ccnn scnn}; done
echo "== $(date) score"; cd $ROOT/cmp && .venv/bin/python score_extra.py eegdir:eegdir ccnn:complex_cnn scnn:simple_cnn xfmr:transformer rnn:lstm fcnn:fcnn
echo "== $(date) done"
