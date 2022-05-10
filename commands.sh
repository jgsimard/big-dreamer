#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin
# salloc --time=1:0:0 --account=rrg-bengioy-ad --gres=gpu:1 -c 12 --mem=124G


#ENV=Ant-v2 # [-1, 1]
#ENV=HalfCheetah-v2 # [-1, 1]
ENV=Hopper-v2 # [-1, 1] # short
#ENV=HumanoiDd-v2 # [-0.4, 0.4]
#ENV=HumanoidStandup-v2 # [-0.4, 0.4]
#ENV=InvertedDoublePendulum-v2 # [-1, 1] # short
#ENV=InvertedPendulum-v2 # [-3, 3]
#ENV=Reacher-v2 # [-1, 1]
#ENV=Swimmer-v2 # [-1, 1]
#ENV=Walker2d-v2 # [-1, 1] # short

#python src/main.py disable_cuda=False \
#                    algorithm='dreamer' \
#                    env=${ENV} \
#                    action_repeat=2 \
#                    seed_steps=2000 \
#                    episodes=100 \
#                    collect_interval=5 \
#                    hidden_size=32 \
#                    belief_size=32 \
#                    test_interval=200 \
#                    log_video_freq=-1\
#                    log_freq=20 \
#                    use_discount=False \
#                    n_layers=3 \
#                    pixel_observation=False \
#                    embedding_size=256 \
#                    kl_balance=0.8 \
##                    latent_distribution=Gaussian\
#                    latent_distribution=Categorical\
#                    discrete_latent_dimensions=16 \
#                    discrete_latent_classes=16

python src/main.py disable_cuda=True \
                    env=${ENV} \
                    action_repeat=2 \
                    seed_steps=200 \
                    collect_interval=5 \
                    hidden_size=23 \
                    belief_size=42 \
                    test_interval=200 \
                    log_video_freq=-1\
                    log_freq=20 \
                    n_layers=2 \
                    pixel_observation=False \
                    embedding_size=256 \
                    latent_distribution=normal\
                    environment_steps_per_update=5


#                    kl_balance=0.8 \
#                    latent_distribution=normal\
