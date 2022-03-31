import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from planet import Planet
from dreamer import Dreamer
from env import Env
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

    ###################
    ### RUN TRAINING
    ###################
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    random.seed(params['seed'])

    init_gpu(use_gpu=not params['disable_cuda'])

    metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
               'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

    # Initialise training environment
    env = Env(params['env'], params['symbolic_env'], params['seed'], params['max_episode_length'],
              params['action_repeat'], params['bit_depth'])


    if params['algorithm'] == 'planet':
        model = Planet(params, env)
    elif params['algorithm'] == 'dreamer':
        model = Dreamer(params, env)
    elif params['algorithm'] == 'dreamerV2':
        raise NotImplementedError('DreamerV2 is not yet implemented')
    else:
        raise NotImplementedError(f'algorithm {params["algorithm"]} is not yet implemented.')

    model.randomly_initilalize_replay_buffer(metrics)

    # Training
    for episode in tqdm(range(metrics['episodes'][-1] + 1, params['episodes'] + 1), total=params['episodes'],
                        initial=metrics['episodes'][-1] + 1):
        # Model fitting
        print("training loop")
        for s in tqdm(range(params['collect_interval'])):
            model.train_step()

        # Update and plot loss metrics
        # TODO : log the metrics

        # Data collection
        print("Data collection")
        with torch.no_grad():
            observation, total_reward = env.reset(), 0
            belief = torch.zeros(1, params['belief_size'], device=device)
            posterior_state = torch.zeros(1, params['state_size'], device=device)
            action = torch.zeros(1, env.action_size, device=device)

            pbar = tqdm(range(params['max_episode_length'] // params['action_repeat']))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = model.update_belief_and_act(
                    env,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=device),
                    explore=True)
                total_reward += reward
                observation = next_observation
                if params['render']:
                    env.render()
                if done:
                    pbar.close()
                    break

            # Update and plot train reward metrics
            # TODO : Log metrics

        # Test model
        print("Test model")
        # TODO : Test Model

        # TODO : Save Model

    # Close training environment
    env.close()


if __name__ == "__main__":
    import os

    print("Command Dir:", os.getcwd())
    my_main()
