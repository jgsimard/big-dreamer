#!/bin/bash
#SBATCH --job-name=big-dreamer                            # Job name
#SBATCH --cpus-per-task=10                                 # Ask for 1 CPUs
#SBATCH --mem=32Gb                                         # Ask for 1 GB of RAM
#SBATCH --output=/scratch/<username>/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/<username>/logs/slurm-%j-%x.error  # log file

# Arguments

# Setup environment
module load python/3.7
source dreamer_env/bin/activate
cd dreamer_oli/big-dreamer

export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin


#python src/main.py disable_cuda=False \
#                    algorithm='planet' \
#                    env='HalfCheetah-v2' \
#                    action_repeat=2 \
#                    episodes=100 \
#                    collect_interval=50 \
#                    hidden_size=32 \
#                    belief_size=32 \
#                    test_interval=2 \
#                    log_video_freq=10

python src/main.py disable_cuda=False \
                    algorithm='planet' \
                    env='HalfCheetah-v2' \
                    action_repeat=2 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1
