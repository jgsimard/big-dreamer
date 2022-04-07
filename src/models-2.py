from typing import Union

import torch
from torch import nn, Tensor

import torch.nn.functional as F
import torch.distributions as D

from torch.nn.common_types import *

Activation = Union[str, nn.Module]


def merge_belief_and_state(belief: Tensor, state: Tensor) -> Tensor:
    """
    Merge the belief and state into a single tensor.
    """

    return torch.cat([belief, state], dim=1)


class TransitionModel(nn.Module):
    """
    Transition model.
    """

    def __init__(self) -> None:
        super(TransitionModel, self).__init__()


class ObservationModel(nn.Module):
    """
    Observation model.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        embedding_size: int,
        activation: Activation = "relu",
    ) -> None:
        super(ObservationModel, self).__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)()

        self.embedding_size = embedding_size
        self.linear = nn.Linear(belief_size + state_size, embedding_size)
        self.deconvs = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride: _size_2_t = 1
            nn.ConvTranspose2d(embedding_size, 128, 5, 2),
            activation,
            nn.ConvTranspose2d(128, 64, 5, 2),
            activation,
            nn.ConvTranspose2d(64, 32, 6, 2),
            activation,
            nn.ConvTranspose2d(32, 3, 6, 2),
        )

    def forward(self, belief: Tensor, state: Tensor) -> Tensor:
        """
        Forward pass.
        """

        x = self.linear(merge_belief_and_state(belief, state))
        x = x.view(-1, self.embedding_size, 1, 1)
        return self.deconvs(x)


class RewardModel(nn.Module):
    """
    Reward model.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        activation: Activation = "relu",
    ) -> None:
        super(RewardModel, self).__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)()

        self.model = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1),
        )

    def forward(self, belief: Tensor, state: Tensor) -> Tensor:
        """
        Forward pass.
        """

        return self.model(merge_belief_and_state(belief, state)).squeeze(dim=1)


class CnnImageEncoder(nn.Module):
    """
    CNN image encoder.
    """

    def __init__(self, embedding_size: int, activation: Activation = "relu") -> None:
        super(CnnImageEncoder, self).__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)()

        self.model = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride
            nn.Conv2d(3, 32, 4, 2),
            activation,
            nn.Conv2d(32, 64, 4, 2),
            activation,
            nn.Conv2d(64, 128, 4, 2),
            activation,
            nn.Conv2d(128, 256, 4, 2),  # (B, 3, H, W) ->  (B, 256, H/16, W/16)
            activation,
            nn.Flatten(),
            nn.Identity()
            if embedding_size == 1024
            else nn.Linear(1024, embedding_size),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass.
        """

        return self.model(observation)


class LinearCombination(nn.Module):
    """
    Linear combination of two inputs.
    """

    def __init__(self, in1_size, in2_size, out_size):
        super(LinearCombination, self).__init__()
        self.in1_linear = nn.Linear(in1_size, out_size)
        self.in2_linear = nn.Linear(in2_size, bias=False)

    def forward(self, in1, in2):
        """
        Forward pass.
        """

        return self.in1_linear(in1) + self.in1_linear(in2)


def activation(x, layer_norm=False):
    """
    Activation function.
    """

    norm = nn.LayerNorm if layer_norm else nn.Identity()
    return F.elu(norm(x))


def diag_normal(x: Tensor, min_std=0.1, max_std=2.0):
    """
    Diagonal normal distribution.
    """

    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


# TODO Use Importance Weighted VAE to improve performance.
class RSSM(nn.Module):
    """
    RSSM.
    """

    def __init__(
        self,
        embedding_size: int,
        action_size: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        rnn_layers: int,
    ):
        super(RSSM, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=deterministic_size,
            num_layers=rnn_layers,
        )

        self.za_combination = LinearCombination(
            stochastic_size, action_size, hidden_size
        )

        self.prior_h = nn.Linear(deterministic_size, hidden_size)
        self.prior_out = nn.Linear(hidden_size, stochastic_size * 2)

        self.he_combination = LinearCombination(
            deterministic_size, embedding_size, hidden_size
        )
        self.posterior_parameters = nn.Linear(hidden_size, stochastic_size * 2)

    def prior(self, h):
        """
        Prior.
        """

        return self.prior_out(activation(self.prior_h(h)))

    def forward(
        self,
        embedded: Tensor,  # (T, B, embedding_size)
        action: Tensor,  # (T, B, action_size)
        reset: Tensor,  # (T, B)
        z_in: Tensor,  # (T, B, stochastic_size)
        h_in: Tensor,  # (T, B, deterministic_size)
    ):
        """
        Forward pass.
        """

        priors = []
        posts = []
        states_h = []
        samples = []

        for _ in range(T):
            za = activation(self.za_combination(z_in, action))
            h_out = self.rnn(za, h_in)
            he = activation(self.he_combination(h_out, embedded))
            posterior_parameters = self.posterior_parameters(he)
            posterior_distribution = diag_normal(posterior_parameters)
            sample = posterior_distribution.rsample().reshape(action.shape[0], -1)

            # record step
            posts.append(posterior_parameters)
            samples.append(sample)
            states_h.append(h_out)

            # update states for next step
            h_in = h_out
            z_in = sample

        posts = torch.stack(posts)  # (T,B,2S)
        states_h = torch.stack(states_h)  # (T,B,D)
        samples = torch.stack(samples)  # (T,B,S)
        priors = self.pri  # (T,B,2S)

        return (priors, posts, samples, states_h)


class Dreamer(nn.Module):
    """
    Dreamer.
    """

    def __init__(self):
        super(DreamerV1, self).__init__()


class Planet(nn.Module):
    """
    Planet.
    """

    def __init__(self):
        super(Planet, self).__init__()
