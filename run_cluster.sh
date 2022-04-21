#!/bin/bash
#SBATCH --job-name=big-dreamer                            # Job name
#SBATCH --cpus-per-task=1                                 # Ask for 1 CPUs
#SBATCH --mem=1Gb                                         # Ask for 1 GB of RAM
#SBATCH --output=/scratch/<username>/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/<username>/logs/slurm-%j-%x.error  # log file

# Arguments
# $1: Path to code directory
# $2: Path to config file
# Copy code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
cd $SLURM_TMPDIR/"$1"

# Setup environment
module load python/3.7
source bigdreamer_env/bin/activate

export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin


python src/main.py --config_name="$2" # write over writes here