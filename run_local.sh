#!/bin/bash


# Arguments
# $1: Path to config file

# python src/main.py disable_cuda=True \
#                     action_repeat=2 \
#                     episodes=10 \
#                     collect_interval=2 \
#                     hidden_size=32 \
#                     belief_size=32 \
#                     test_interval=2 \
#                     algorithm='dreamer' \
#                     log_video_freq=1

python src/main.py --config-name="$1"