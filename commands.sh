python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='Pendulum-v0' \
                    action_repeat=2 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1

python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='HalfCheetah-v2' \
                    action_repeat=2 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1

python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='HumanoidStandup-v2' \
                    action_repeat=2 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1

python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='Reacher-v2' \
                    action_repeat=2 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1



# dm_control
python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='cartpole-swingup' \
                    action_repeat=8 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1


python src/main.py disable_cuda=True \
                    algorithm='planet' \
                    env='cheetah-run' \
                    action_repeat=4 \
                    episodes=10 \
                    collect_interval=2 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=2 \
                    log_video_freq=1