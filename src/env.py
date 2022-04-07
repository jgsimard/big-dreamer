import cv2
import numpy as np
import torch

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


def preprocess_observation_(observation, bit_depth):
    """
    Preprocesses an observation inplace.
    (from float32 Tensor [0, 255] to [-0.5, 0.5])
    """

    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(0.5)
    # Dequantise:
    # (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2**bit_depth))


def postprocess_observation(observation, bit_depth):
    """
    Postprocess an observation for storage.
    (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """

    return np.clip(
        np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth),
        0,
        2**8 - 1,
    ).astype(np.uint8)


def _images_to_observation(images, bit_depth):
    # Resize and put channel first
    images = torch.tensor(
        cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
        dtype=torch.float32,
    )
    # Quantise, centre and dequantise inplace
    preprocess_observation_(images, bit_depth)
    # Add batch dimension
    return images.unsqueeze(dim=0)


class ControlSuiteEnv:
    """
    Wrapper for the control suite.
    """

    def __init__(
        self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth
    ):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        domain, task = env.split("-")
        self.symbolic = symbolic
        self._env = suite.load(
            domain_name=domain, task_name=task, task_kwargs={"random": seed}
        )

        if not symbolic:
            self._env = pixels.Wrapper(self._env)

        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print(
                f"WARNING: action repeat {action_repeat} is not recommended for {domain}."
            )

        self.bit_depth = bit_depth
        self.t = 0

    def reset(self):
        """
        Resets the environment.
        """

        self.t = 0  # Reset internal timer
        state = self._env.reset()

        if self.symbolic:
            return torch.tensor(
                np.concatenate(
                    [
                        np.asarray([obs]) if isinstance(obs, float) else obs
                        for obs in state.observation.values()
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
            ).unsqueeze(dim=0)

        return _images_to_observation(
            self._env.physics.render(camera_id=0), self.bit_depth
        )

    def step(self, action):
        """
        Steps the environment.
        """

        action = action.detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break

        if self.symbolic:
            observation = torch.tensor(
                np.concatenate(
                    [
                        np.asarray([obs]) if isinstance(obs, float) else obs
                        for obs in state.observation.values()
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
            ).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(
                self._env.physics.render(camera_id=0), self.bit_depth
            )

        return observation, reward, done

    def render(self):
        """
        Renders the environment.
        """

        cv2.imshow("screen", self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        """
        Closes the environment.
        """

        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        """
        Returns the size of the observation.
        """

        return (
            sum(
                [
                    (1 if len(obs.shape) == 0 else obs.shape[0])
                    for obs in self._env.observation_spec().values()
                ]
            )
            if self.symbolic
            else (3, 64, 64)
        )

    @property
    def action_size(self):
        """
        Returns the size of the action.
        """

        return self._env.action_spec().shape[0]

    def sample_random_action(self):
        """
        Samples an action randomly from a uniform distribution over all valid actions.
        """

        spec = self._env.action_spec()
        return torch.from_numpy(
            np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        )


class GymEnv:
    """
    Wrapper for the OpenAI Gym environment.
    """

    def __init__(
        self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth
    ):
        import gym

        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self._env.action_space.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.t = 0

    def reset(self):
        """
        Resets the environment.
        """

        self.t = 0  # Reset internal timer
        state = self._env.reset()

        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)

        return _images_to_observation(
            self._env.render(mode="rgb_array"), self.bit_depth
        )

    def step(self, action):
        """
        Steps the environment.
        """

        action = action.detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break

        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(
                self._env.render(mode="rgb_array"), self.bit_depth
            )

        return observation, reward, done

    def render(self):
        """
        Renders the environment.
        """

        self._env.render()

    def close(self):
        """
        Closes the environment.
        """

        self._env.close()

    @property
    def observation_size(self):
        """
        Returns the size of the observation.
        """

        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        """
        Returns the size of the action.
        """

        return self._env.action_space.shape[0]

    def sample_random_action(self):
        """
        Samples an action randomly from a uniform distribution over all valid actions.
        """

        return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    """
    Returns an environment wrapper.
    """

    if env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)

    if env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(
            env, symbolic, seed, max_episode_length, action_repeat, bit_depth
        )

    raise ValueError(f"Unknown environment: {env}")


class EnvBatcher:
    """
    Wrapper for batching environments together.
    """

    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    def reset(self):
        """
        Resets the environment.
        Returns: observation
        """

        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    def step(self, actions):
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

    def close(self):
        """
        Closes the environment.
        """

        [env.close() for env in self.envs]
