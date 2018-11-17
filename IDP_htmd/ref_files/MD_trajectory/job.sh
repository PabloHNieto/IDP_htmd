#!/bin/bash
#
#SBATCH --job-name=1_equil_97119
#SBATCH --partition=multiscale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --priority=None
#SBATCH --workdir=/workspace8/p27_sj403/2_p27_charmm_unfold/1_equil
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err
#SBATCH --export=ACEMD_HOME,HTMD_LICENSE_FILE
#SBATCH --exclude=giallo

trap "touch /workspace8/p27_sj403/2_p27_charmm_unfold/1_equil/htmd.queues.done" EXIT SIGTERM


cd /workspace8/p27_sj403/2_p27_charmm_unfold/1_equil
/workspace8/p27_sj403/2_p27_charmm_unfold/1_equil/run.sh