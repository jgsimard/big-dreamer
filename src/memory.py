import numpy as np
import torch

from utils import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    """
    Experience replay memory.
    """

    def __init__(self, size, action_size, bit_depth, pixel_observation, observation_size, device):
        self.device = device
        self.size = size
        self.pixel_observation = pixel_observation
        if self.pixel_observation:
            self.observations = np.empty((size, 3, 64, 64), dtype= np.uint8)
        else:
            print(f"observation_size={observation_size}")
            self.observations = np.empty((size, observation_size),dtype=np.float32)

        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = (
            0,
            0,
        )  # Tracks how much experience has been used in total
        self.bit_depth = bit_depth

    def append(self, observation, action, reward, done):
        """
        Append a new experience to the memory.
        """
        if self.pixel_observation:
            # Decentre and discretise visual observations (to save memory)
            self.observations[self.idx] = postprocess_observation(
                observation.numpy(), self.bit_depth
            )
        else:
            self.observations[self.idx] = observation.numpy()
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    def _sample_idx(self, L):
        """
        Sample a single sequence chunk uniformly from the memory.

        Returns an index for a valid single sequence chunk uniformly sampled from the memory
        """

        valid_idx = False
        while not valid_idx:
            # print("HEREE")
            # print(f'self.size={self.size}, self.idx={self.idx} L ={L}')

            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = (
                not self.idx in idxs[1:]
            )  # Make sure data does not cross the memory index
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        """
        Retrieve a batch of sequence chunks uniformly sampled from the memory.
        """

        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if self.pixel_observation:
            # Undo discretisation for visual observations
            preprocess_observation_(observations, self.bit_depth)
        return (
            observations.reshape(L, n, *observations.shape[1:]),
            self.actions[vec_idxs].reshape(L, n, -1),
            self.rewards[vec_idxs].reshape(L, n),
            self.nonterminals[vec_idxs].reshape(L, n, 1),
        )

    def sample(self, n, L):
        """
        Sample a batch of n sequence chunks of length L uniformly from the memory.
        """
        # print(self.size, n, L)

        batch = self._retrieve_batch(
            np.asarray([self._sample_idx(L) for _ in range(n)]), n, L
        )
        # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
        # [1578 1579 1580 ... 1625 1626 1627] | 0/100 [00:00<?, ?it/s]
        # [1049 1050 1051 ... 1096 1097 1098]
        # [1236 1237 1238 ... 1283 1284 1285]
        # ...
        # [2199 2200 2201 ... 2246 2247 2248]
        # [ 686  687  688 ...  733  734  735]
        # [1377 1378 1379 ... 1424 1425 1426]]
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
