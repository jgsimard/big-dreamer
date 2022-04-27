import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tqdm import tqdm
from torchvision.utils import make_grid


from planet import Planet
from dreamer import Dreamer
from dreamerV2 import DreamerV2
from logger import Logger
from env import Env, EnvBatcher
from utils import init_gpu, device


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    Main function for running the experiments.
    """
    print("Command Dir:", os.getcwd())

    params = vars(cfg)
    for key, value in cfg.items():
        params[key] = value
    print("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logdir = os.path.join(
        data_path,
        f'project_{cfg.exp_name}_{cfg.env}_{time.strftime("%d-%m-%Y_%H-%M-%S")}',
    )
    params["logdir"] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open_dict(cfg):
        cfg.logdir = logdir

    print("\nLOGGING TO: ", logdir, "\n")

    #############
    # INIT Structure
    #############
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])

    init_gpu(use_gpu=not params["disable_cuda"])

    logger = Logger(params["logdir"], params=params)

    #############
    # ENV
    #############
    env = Env(params)

    #############
    # Model
    #############
    if params["algorithm"] == "planet":
        model = Planet(params, env)
    elif params["algorithm"] == "dreamer":
        model = Dreamer(params, env)
    elif params["algorithm"] == "dreamerV2":
        model = DreamerV2(params, env)
    else:
        raise NotImplementedError(
            f'algorithm {params["algorithm"]} is not yet implemented.'
        )

    env_steps, num_episodes = model.randomly_initialize_replay_buffer()
    print(f"Initialized with {num_episodes} episodes and {env_steps} steps")

    logs ={}
    observation = env.reset()
    past_time = time.time()
    # done = False
    episode_reward = 0
    last_episode_reward = 0
    episode_steps = 0
    belief = torch.zeros(1, params["belief_size"], device=device)
    posterior_state = torch.zeros(1, params["state_size"], device=device)
    action = torch.zeros(1, env.action_size, device=device)
    ###################
    # TRAINING LOOP
    ###################
    for step in range(env_steps, params['train_steps']):
        ###################
        # Weight updates
        ###################
        if step % params['environment_steps_per_update'] == 0:
            weight_update_start_time = time.time()
            for _ in range(params['collect_interval']):
                logs = model.train_step()
                elapsed_time = time.time() - weight_update_start_time
                logs["weight_update_per_sec"] = params['collect_interval'] / elapsed_time

        if params["algorithm"] in ['dreamer', 'dreamerV2']:
            if step % params['ActorCritic']['slow_critic_update_interval']:
                model.update_critic()

        ##########################
        # Environment interaction
        ##########################
        # print("Data collection")
        with torch.no_grad():
            # observation = env.reset()
            # total_reward = 0
            # belief = torch.zeros(1, params["belief_size"], device=device)
            # posterior_state = torch.zeros(1, params["state_size"], device=device)
            # action = torch.zeros(1, env.action_size, device=device)

            # pbar = tqdm(range(params["max_episode_length"] // params["action_repeat"]))
            # t = 0

            # for t in pbar:
            (
                belief,
                posterior_state,
                action,
                next_observation,
                reward,
                done,
            ) = model.update_belief_and_act(
                env,
                belief,
                posterior_state,
                action,
                observation.to(device=device),
                explore=True,
            )

            # store the new stuff
            model.buffer.append(observation, action, reward, done)

            episode_reward += reward
            episode_steps += 1
            observation = next_observation

            # print(episode_reward)

            if params["render"]:
                env.render()
            if done or episode_steps % params["max_episode_length"] == 0:
                # pbar.close()
                # env.close()
                # break
                observation = env.reset()
                # done = False
                last_episode_reward = episode_reward
                episode_reward = 0
                episode_steps = 0
                # todo use inplace stuff instead
                belief = torch.zeros(1, params["belief_size"], device=device)
                posterior_state = torch.zeros(1, params["state_size"], device=device)
                action = torch.zeros(1, env.action_size, device=device)

                num_episodes += 1
            # env_steps += t * params["action_repeat"]
            # num_episodes += 1
            logs["episode_total_reward"] = last_episode_reward

        ##########################
        # Logging
        ##########################
        if step % params["log_freq"] == 0:
            current_time = time.time()
            elapsed_time = current_time - past_time
            logs["env_update_per_sec"] = params['log_freq'] / elapsed_time
            past_time = current_time
            print("\nPerform Logging")
            for key, value in logs.items():
                print(f"{key} : {value}")
                logger.log_scalar(value, key, step)
            print("Done logging...")

            logger.flush()

        ##########################
        # Test model
        ##########################
        if step % params["test_interval"] == 0:
            print("\nTest model")
            model.eval()

            # Initialise parallelised test environments
            test_envs = EnvBatcher(Env, params, params["test_episodes"])

            with torch.no_grad():
                observation_test = test_envs.reset()
                total_rewards_test = np.zeros((params["test_episodes"],))
                video_frames_test = []

                belief_test = torch.zeros(
                    params["test_episodes"], params["belief_size"], device=device
                )
                posterior_state_test = torch.zeros(
                    params["test_episodes"], params["state_size"], device=device
                )
                action_test = torch.zeros(
                    params["test_episodes"], env.action_size, device=device
                )

                pbar_test = tqdm(
                    range(params["max_episode_length"] // params["action_repeat"])
                )
                for _ in pbar_test:
                    (
                        belief_test,
                        posterior_state_test,
                        action_test,
                        next_observation_test,
                        reward_test,
                        done_test,
                    ) = model.update_belief_and_act(
                        test_envs,
                        belief_test,
                        posterior_state_test,
                        action_test,
                        observation_test.to(device=device),
                    )

                    total_rewards_test += reward_test.numpy()

                    if params['pixel_observation']:
                        # Collect real vs. predicted frames for video
                        video_frames_test.append(
                            make_grid(
                                torch.cat(
                                    [
                                        observation,
                                        model.observation_model(
                                            belief_test, posterior_state_test
                                        ).cpu(),
                                    ],
                                    dim=3,
                                )
                                + 0.5,
                                nrow=5,
                            ).numpy()
                        )  # Decentre
                    observation_test = next_observation_test

                    pbar_test.set_description("Testing...")
                    if done_test.sum().item() == params["test_episodes"]:
                        pbar_test.close()
                        test_envs.close()
                        break
                # log test scalar
                test_logs = {
                    "Eval_min_return": total_rewards_test.min().item(),
                    "Eval_avg_return": total_rewards_test.mean().item(),
                    "Eval_max_return": total_rewards_test.max().item(),
                    "Eval_std_return": total_rewards_test.std().item(),
                }
                for key, value in test_logs.items():
                    print(f"{key} : {value}")
                    logger.log_scalar(value, key, step)
                model.eval()
            test_envs.close()

            # TODO : Save Model
            if step % params["log_video_freq"] == 0 \
                    and params["log_video_freq"] != -1 \
                    and params['pixel_observation']:
                # log eval videos
                logger.log_video(
                    np.expand_dims(np.stack(video_frames_test), axis=0),
                    name="Eval_rollout",
                    step=step,
                )
    # Close training environment
    env.close()

if __name__ == "__main__":
    import os

    print("Command Dir:", os.getcwd())
    my_app(None)
