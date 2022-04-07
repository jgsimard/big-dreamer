from typing import Dict


class BaseAgent(object):
    """
    Base class for agents.
    """

    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> Dict[str, float]:
        """
        Return a dictionary of logging information.
        """

        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        """
        Add a batch of paths to the replay buffer.
        """

        raise NotImplementedError

    def sample(self, batch_size):
        """
        Sample a batch of paths (trajectories) from the replay buffer.
        """

        raise NotImplementedError

    def save(self, path):
        """
        Save the agent to the given path.
        """

        raise NotImplementedError
