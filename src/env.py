from typing import Tuple, List

# import cv2
import numpy as np
import torch

import gym
# from dm_control import suite
# from dm_control.suite.wrappers import pixels

from utils import images_to_observation

GYM_ENVS = [
    "Pendulum-v0",
    "MountainCarContinuous-v0",
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2",
]
CONTROL_SUITE_ENVS = [
    "cartpole-balance",
    "cartpole-swingup",
    "reacher-easy",
    "finger-spin",
    "cheetah-run",
    "ball_in_cup-catch",
    "walker-walk",
    "reacher-hard",
    "walker-run",
    "humanoid-stand",
    "humanoid-walk",
    "fish-swim",
    "acrobot-swingup",
]
CONTROL_SUITE_ACTION_REPEATS = {
    "cartpole": 8,
    "reacher": 4,
    "finger": 2,
    "cheetah": 4,
    "ball_in_cup": 6,
    "walker": 2,
    "humanoid": 2,
    "fish": 2,
    "acrobot": 4,
}


class BaseEnv:
    """
    Base class for our environments.
    """

    def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth) -> None:
        """
        Initialises base environment attributes.
        """
        self._env = env
        self.seed = seed
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

        self.bit_depth = bit_depth
        self.t = 0

        """
        Initialise the observation space.
        """
        self.obs_image_depth = 3
        self.obs_image_height = 64
        self.obs_image_width = 64

    def step(self, action):
        """
        Steps the environment.
        """

        raise NotImplementedError

    def reset(self):
        """
        Resets the environment.
        """

        raise NotImplementedError

    def render(self):
        """
        Renders the environment.
        """

        raise NotImplementedError

    def close(self):
        """
        Closes the environment.
        """

        raise NotImplementedError

    @property
    def observation_size(self) -> Tuple[int, int, int]:
        """
        Returns the size of the observation.
        """

        return (self.obs_image_depth, self.obs_image_height, self.obs_image_width)

    @property
    def action_size(self):
        """
        Returns the size of the action.
        """

        raise NotImplementedError

    def sample_random_action(self):
        """
        Samples an action randomly from a uniform distribution over all valid actions.
        """

        raise NotImplementedError


# class ControlSuiteEnv(BaseEnv):
#     """
#     Wrapper for the control suite.
#     """
#
#     def __init__(
#         self, env, seed, max_episode_length, action_repeat, bit_depth
#     ) -> None:
#         super().__init__(env, seed, max_episode_length, action_repeat, bit_depth)
#
#         domain, task = env.split("-")
#         self._env = suite.load(
#             domain_name=domain, task_name=task, task_kwargs={"random": seed}
#         )
#         self._env = pixels.Wrapper(self._env)
#
#         if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
#             print(
#                 f"WARNING: action repeat {action_repeat} is not recommended for {domain}."
#             )
#
#     def reset(self) -> np.ndarray:
#         """
#         Resets the environment.
#         """
#
#         self.t = 0
#         self._env.reset()
#
#         return images_to_observation(
#             self._env.physics.render(camera_id=0),
#             self.bit_depth,
#             (self.obs_image_height, self.obs_image_width)
#         )
#
#     def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
#         """
#         Steps the environment.
#         """
#
#         action = action.detach().numpy()
#         reward = 0
#
#         for _ in range(self.action_repeat):
#             state = self._env.step(action)
#             reward += state.reward
#             self.t += 1  # Increment internal timer
#             done = state.last() or self.t == self.max_episode_length
#             if done:
#                 break
#
#         observation = images_to_observation(
#             self._env.physics.render(camera_id=0),
#             self.bit_depth,
#             (self.obs_image_height, self.obs_image_width)
#         )
#
#         return observation, reward, done
#
#     def render(self) -> None:
#         """
#         Renders the environment.
#         """
#
#         cv2.imshow("screen", self._env.physics.render(camera_id=0)[:, :, ::-1])
#         cv2.waitKey(1)
#
#     def close(self) -> None:
#         """
#         Closes the environment.
#         """
#
#         cv2.destroyAllWindows()
#         self._env.close()
#
#     @property
#     def action_size(self) -> int:
#         """
#         Returns the size of the action.
#         """
#
#         return self._env.action_spec().shape[0]
#
#     def sample_random_action(self) -> torch.Tensor:
#         """
#         Samples an action randomly from a uniform distribution over all valid actions.
#         """
#
#         spec = self._env.action_spec()
#         return torch.from_numpy(
#             np.random.uniform(spec.minimum, spec.maximum, spec.shape)
#         )

class GymEnv(BaseEnv):
    """
    Wrapper for the OpenAI Gym environment.
    """

    def __init__(
        self, env, seed, max_episode_length, action_repeat, bit_depth
    ):
        super().__init__(env, seed, max_episode_length, action_repeat, bit_depth)

        self._env = gym.make(env)
        self._env.seed(seed)
        self._env.action_space.seed(seed)


    def reset(self) -> np.ndarray:
        """
        Resets the environment.
        """

        self.t = 0
        self._env.reset()

        return images_to_observation(
            self._env.render(mode="rgb_array"),
            self.bit_depth,
            (self.obs_image_height, self.obs_image_width)
        )

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        """
        Steps the environment.
        """

        action = action.detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            _, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break

        observation = images_to_observation(
            self._env.render(mode="rgb_array"),
            self.bit_depth,
            (self.obs_image_height, self.obs_image_width)
        )

        return observation, reward, done

    def render(self) -> None:
        """
        Renders the environment.
        """

        self._env.render()

    def close(self) -> None:
        """
        Closes the environment.
        """

        self._env.close()

    @property
    def action_size(self) -> int:
        """
        Returns the size of the action.
        """

        return self._env.action_space.shape[0]

    def sample_random_action(self) -> torch.Tensor:
        """
        Samples an action randomly from a uniform distribution over all valid actions.
        """

        return torch.from_numpy(self._env.action_space.sample())


def Env(params) -> BaseEnv:
    """
    Returns an environment wrapper.
    """

    env = params["env"]
    seed = params["seed"]
    max_episode_length = params["max_episode_length"]
    action_repeat = params["action_repeat"]
    bit_depth = params["bit_depth"]

    if env in GYM_ENVS:
        return GymEnv(env, seed, max_episode_length, action_repeat, bit_depth)

    # if env in CONTROL_SUITE_ENVS:
    #     return ControlSuiteEnv(
    #         env, seed, max_episode_length, action_repeat, bit_depth
    #     )

    raise ValueError(f"Unknown environment: {env}")


class EnvBatcher:
    """
    Wrapper for batching environments together.
    """

    def __init__(self, env_class, env_params, n) -> None:
        self.n = n
        self.envs = [env_class(env_params) for _ in range(n)]
        self.dones = [True] * n

    def reset(self) -> List[np.ndarray]:
        """
        Resets the environment.
        Returns: observation
        """

        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    def step(self, actions) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """
        Steps the environment.
        Returns: (observations, rewards, dones)
        """

        # Done mask to blank out observations and zero rewards for previously
        # terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, dones = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)]
        )
        dones = [
            d or prev_d for d, prev_d in zip(dones, self.dones)
        ]  # Env should remain terminated if previously terminated

        self.dones = dones
        observations = torch.cat(observations)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0

        return observations, rewards, dones

    def close(self) -> None:
        """
        Closes the environments.
        """

        for env in self.envs:
            env.close()
