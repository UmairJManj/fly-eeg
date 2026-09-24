#!/bin/bash -e
# Published single-channel deep-denoiser baselines on the RAW signal (no brain), same splits as everything else.
#   ARTIFACT=eog sbatch [-p P --gpus-per-node G:1] fly-eeg/nesi_fly_baselines.sl
#SBATCH --job-name=fly-cnn60k
#SBATCH --account=aut04653
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --partition=milan
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-cnn60k-%j.log
ROOT=/nesi/project/aut04653/Manj/Fly; source $ROOT/env.sh; cd "$ROOT/nfly"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARTIFACT=${ARTIFACT:-eog}; NB=/nesi/nobackup/aut04653/Manj/Fly; nvidia-smi --query-gpu=name --format=csv,noheader
echo "== $(date) smoke"; uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $NB/states_eog_whole_aug12_nobrain --tag blsmoke --configs "fcnn+s50,scnn+s50,ccnn+s50,rnn+s50,xfmr+s50" --seeds 1 --gpu-cache-gb 2 | tail -8
echo "== $(date) full: $ARTIFACT"
uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py $NB/states_${ARTIFACT}_whole_aug12_nobrain --tag cnn60k --configs "scnn+s60000,ccnn+s60000" --seeds 3 --save-model $NB/tcn_models/cnn60k_$ARTIFACT
echo "== $(date) done"
