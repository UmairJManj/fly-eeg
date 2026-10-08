#!/bin/bash -e
#SBATCH --job-name=fly-erpmeta
#SBATCH --account=aut04653
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=milan
#SBATCH --output=/nesi/project/aut04653/Manj/Fly/logs/fly-erpmeta-%j.log
cd /nesi/project/aut04653/Manj/Fly/cmp
for s in $(awk '$1=="gonogo"{print $2}' erp_subjects.txt); do .venv/bin/python erp_bench.py clean --ds gonogo --sub $s --methods hp --remeta; done
