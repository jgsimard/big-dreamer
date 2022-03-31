import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from planet import Planet
from dreamer import Dreamer
from logger import Logger
from env import Env, EnvBatcher
from utils import init_gpu, device


@hydra.main(config_path="conf", config_name="config")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    print("Command Dir:", os.getcwd())

    params = vars(cfg)
    # params.extend(env_args)
    for key, value in cfg.items():
        params[key] = value
    print("params: ", params)

    # ##################################
    # ### CREATE DIRECTORY FOR LOGGING
    # ##################################
    logdir_prefix = 'project_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + cfg.exp_name + '_' + cfg.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.logdir = logdir

    print("\nLOGGING TO: ", logdir, "\n")

    #############
    # INIT Structure
    #############
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    random.seed(params['seed'])

    init_gpu(use_gpu=not params['disable_cuda'])

    logger = Logger(params['logdir'])

    # metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
    #            'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

    #############
    # ENV
    #############
    env = Env(params['env'],
              params['symbolic_env'],
              params['seed'],
              params['max_episode_length'],
              params['action_repeat'],
              params['bit_depth'])

    #############
    # Model
    #############

    if params['algorithm'] == 'planet':
        model = Planet(params, env)
    elif params['algorithm'] == 'dreamer':
        model = Dreamer(params, env)
    elif params['algorithm'] == 'dreamerV2':
        raise NotImplementedError('DreamerV2 is not yet implemented')
    else:
        raise NotImplementedError(f'algorithm {params["algorithm"]} is not yet implemented.')

    env_steps, num_episodes = model.randomly_initialize_replay_buffer()

    ###################
    # RUN TRAINING
    ###################
    train_step = 0

    # Training
    for episode in tqdm(range(num_episodes+1, params['episodes']+1), total=params['episodes'], initial=num_episodes+1):
        # Model fitting
        print("training loop")
        for s in tqdm(range(params['collect_interval'])):
            logs = model.train_step()

            # if train_step % params['log_freq'] == 0:
            #     print("Perform Logging")
            #     # perform the logging
            #     for key, value in logs.items():
            #         print('{} : {}'.format(key, value))
            #         logger.log_scalar(value, key, train_step)
            #     print('Done logging...\n')
            #
            #     logger.flush()

            train_step += 1

        ##########################
        # Environment interaction
        ##########################
        print("Data collection")
        with torch.no_grad():
            observation = env.reset()
            total_reward = 0
            belief = torch.zeros(1, params['belief_size'], device=device)
            posterior_state = torch.zeros(1, params['state_size'], device=device)
            action = torch.zeros(1, env.action_size, device=device)

            pbar = tqdm(range(params['max_episode_length'] // params['action_repeat']))
            for t in pbar:
                outputs = model.update_belief_and_act(
                    env,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=device),
                    explore=True)
                belief, posterior_state, action, next_observation, reward, done = outputs

                # store the new stuff
                model.replay_buffer.append(observation, action, reward, done)

                total_reward += reward
                observation = next_observation

                if params['render']:
                    env.render()
                if done:
                    pbar.close()
                    break

            env_steps += t * params['action_repeat']
            num_episodes += 1
            logs['episodique_total_reward'] = total_reward

        ##########################
        # Test model
        ##########################

        # TODO : Test Model
        if episode % params['test_interval'] == 0:
            print("Test model")
            model.eval()

            # Initialise parallelised test environments
            test_envs = EnvBatcher(Env, (params['env'],
                                         params['symbolic_env'],
                                         params['seed'],
                                         params['max_episode_length'],
                                         params['action_repeat'],
                                         params['bit_depth']),
                                   {},
                                   params['test_episodes'])

            with torch.no_grad():
                observation = test_envs.reset()
                total_rewards = np.zeros((params['test_episodes'],))
                video_frames = []

                belief = torch.zeros(params['test_episodes'], params['belief_size'],  device=device)
                posterior_state = torch.zeros(params['test_episodes'], params['state_size'],device=device)
                action = torch.zeros(params['test_episodes'], env.action_size, device=device)

                pbar = tqdm(range(params['max_episode_length'] // params['action_repeat']))
                for t in pbar:
                    outputs = model.update_belief_and_act(
                        test_envs,
                        belief,
                        posterior_state,
                        action,
                        observation.to(device=device))
                    belief, posterior_state, action, next_observation, reward, done = outputs

                    total_rewards += reward.numpy()
                    # Collect real vs. predicted frames for video
                    if not params['symbolic_env']:
                        video_frames.append(make_grid(
                            torch.cat([observation, model.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                            nrow=5).numpy())  # Decentre
                    observation = next_observation
                    if done.sum().item() == params['test_episodes']:
                        pbar.close()
                        break

                logs['Eval_avg_return'] = total_rewards.mean()
                logs['Eval_std_return'] = total_rewards.std()

        # TODO : Save Model


        if train_step % params['log_freq'] == 0:
            print("Perform Logging")
            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, env_steps)
            print('Done logging...\n')

            logger.flush()

    # Close training environment
    env.close()


if __name__ == "__main__":
    import os

    print("Command Dir:", os.getcwd())
    my_main()
