#!/bin/bash
#SBATCH --job-name=big-dreamer
#SBATCH --account=rrg-bengioy-ad
#SBATCH --mail-type=ALL
#SBATCH --mail-user=olivier.ethier@umontreal.ca
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32Gb
#SBATCH --output=logs_slurm/slurm-%j-%x.log

# You need to launch this script from the folder big-dreamer
# and have the environment in the grand-parent folder

# Arguments
# $1: algorithm: planet, dreamer or dreamerV2
# $2: environement to run on: HalfCheetah-v2, 

date
echo "Algorithm:" ${1:-planet}
echo "Env:" ${2:-HalfCheetah-v2}

# Setup environment
module load python/3.7
source ../../dreamer_env/bin/activate
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin

# copying code to tmp dir
rsync -av src $TMPDIR/
cd $TMPDIR

python src/main.py disable_cuda=False \
                    algorithm="${1:-planet}" \
                    env="${2:-HalfCheetah-v2}" \
                    action_repeat=2 \
                    episodes=100 \
                    collect_interval=50 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=10 \
                    log_video_freq=10

# # Use this for testing the algo:

# python src/main.py disable_cuda=False \
#                     algorithm="${1:-planet}" \
#                     env="${1:-HalfCheetah-v2}" \
#                     action_repeat=2 \
#                     episodes=1 \
#                     collect_interval=2 \
#                     hidden_size=32 \
#                     belief_size=32 \
#                     test_interval=3 \
#                     log_video_freq=3

# copying outputs to original folder (in scratch)
rsync -av outputs/ ~/scratch/dreamer_oli/big-dreamer/outputs/slurm-$SLURM_JOB_ID/