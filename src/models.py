from typing import Union, Optional, List, Tuple

import numpy as np
import torch
from torch import nn, Tensor

import torch.nn.functional as F
import torch.distributions as D

from utils import cat, stack, prefill


Activation = Union[str, nn.Module]


def bottle(f, x_tuple: tuple) -> Tensor:
    """
    Wraps the input tuple for a function to process a time x batch x features sequence in
    batch x features (assumes one output)

    Args:
        f: function to apply to the input tuple
        x_tuple (Tuple[belief, posterior_states]): tuple of tensors (belief, posterior_states)

    Returns:
        Tensor: output of f() of shape: (belief[0].size(), belief[1].size(), *f().size()[1:])
    """

    x_sizes = [x.size() for x in x_tuple]
    y = f(*[x[0].view(x[1][0] * x[1][1], *x[1][2:]) for x in zip(x_tuple, x_sizes)])
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

    return output


class GaussianBeliefModel(nn.Module):
    """
    Belief model
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            state_size: int,
            activation: nn.Module,
            min_std_dev: float
    ) -> None:
        super().__init__()
        self.min_std_dev = min_std_dev
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 2 * state_size)
        )

    def forward(
            self,
            belief: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Args:
            belief (Tensor): belief of shape: (batch_size, belief_size)

        Returns:
            Tensor: belief of shape: (batch_size, belief_size)
        """
        mean, _std_dev = torch.chunk(self.model(belief), 2, dim=1)
        std_dev = F.softplus(_std_dev) + self.min_std_dev
        state = mean + std_dev * torch.randn_like(mean)
        return state, (mean, std_dev)


class CategoricalBeliefModel(nn.Module):
    """
    Belief model
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            discrete_latent_dimensions: int,
            discrete_latent_classes: int,
            activation: nn.Module
    ) -> None:
        super().__init__()
        self.discrete_latent_dimensions = discrete_latent_dimensions
        self.discrete_latent_classes = discrete_latent_classes
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, discrete_latent_dimensions * discrete_latent_classes)
        )

    def forward(self, belief: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Args:
            belief (Tensor): belief of shape: (batch_size, belief_size)

        Returns:
            Tensor: belief of shape: (batch_size, belief_size)
        """
        logits = self.model(belief)
        shape = (*logits.shape[:-1], self.discrete_latent_dimension, self.discrete_latent_classes)
        logits = logits.reshape(shape)
        # use straight through gradient
        # https://arxiv.org/abs/1308.3432
        dist = D.OneHotCategoricalStraightThrough(logits=logits)
        state = dist.rsample()
        return state, (logits,)


class TransitionModel(nn.Module):
    """
    Transition model for the MDP.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int,
        embedding_size: int,
        activation: Optional[str] = "relu",
        min_std_dev: float = 0.1,
        latent_distribution: Optional[str] = "Gaussian",
        discrete_latent_dimensions: Optional[int] = 32,
        discrete_latent_classes: Optional[int] = 32
    ) -> None:
        super().__init__()

        # general attibutes
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.min_std_dev = min_std_dev

        assert latent_distribution in ["Gaussian", "Categorical"], f"{latent_distribution}"
        self.latent_distribution = latent_distribution

        # recurrent component
        self.rnn = nn.GRUCell(belief_size, belief_size)

        # model components
        self.fc_embed_state_action = nn.Sequential(
            nn.Linear(state_size + action_size, belief_size),
            activation()
        )

        # Belief models
        if self.latent_distribution == "Gaussian":
            self.belief_prior = GaussianBeliefModel(
                belief_size, hidden_size, state_size, activation, min_std_dev
            )
            self.belief_posterior = GaussianBeliefModel(
                belief_size + embedding_size,
                hidden_size,
                state_size,
                activation,
                min_std_dev,
            )
        elif self.latent_distribution == "Categorical":
            self.belief_prior = CategoricalBeliefModel(
                belief_size,
                hidden_size,
                discrete_latent_dimensions,
                discrete_latent_classes,
                activation
            )
            self.belief_posterior = CategoricalBeliefModel(
                belief_size + embedding_size,
                hidden_size,
                discrete_latent_dimensions,
                discrete_latent_classes,
                activation
            )

        self.modules = [
            self.fc_embed_state_action,
            self.belief_prior,
            self.belief_posterior,
        ]

    def forward(
        self,
        init_state: Tensor,
        actions: Tensor,
        init_belief: Tensor,
        observations: Optional[Tensor] = None,
        nonterminals: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...], Optional[Tensor], Optional[Tuple[Tensor, ...]]]:
        """
        L=Chunk size, B=Batch size, Hi=Hidden size, Be=Belief size, S=State Size, A=Action size
        Input:
            init_state:     (B, S)
            actions:        (L, B, A)
            init_belief:    (B, Be)
            observations:   (L, B, C, H, W)
            nonterminals:   (L, B, 1)
        Output:
            beliefs:            (L-1, B, Be)
            prior_states:       (L-1, B, S)
            prior_params:       (L-1, B, S)
            posterior_states:   (L-1, B, S)
            posterior_params:   (L-1, B, S)
        """

        # Create lists for hidden states
        # (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs = prefill(T)
        prior_states = prefill(T)
        posterior_states = prefill(T)

        if self.latent_distribution == "Gaussian":
            prior_means = prefill(T)
            prior_std_devs = prefill(T)
            posterior_means = prefill(T)
            posterior_std_devs = prefill(T)
        elif self.latent_distribution == "Categorical":
            prior_logits = prefill(T)
            posterior_logits = prefill(T)

        beliefs[0] = init_belief
        prior_states[0] = init_state
        posterior_states[0] = init_state

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]

            # Mask if previous transition was terminal
            _state = _state if nonterminals is None else _state * nonterminals[t]

            # Compute belief (deterministic hidden state)
            hidden = self.fc_embed_state_action(cat(_state, actions[t]))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            prior_states[t + 1], prior_params_ = self.belief_prior(beliefs[t + 1])
            if self.latent_distribution == "Gaussian":
                prior_means[t + 1], prior_std_devs[t + 1] = prior_params_
            elif self.latent_distribution == "Categorical":
                prior_logits[t + 1] = prior_params_

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using
                # current observation.
                t_ = t - 1  # Using t_ to deal with different time indexing for observations
                posterior_input = cat(beliefs[t + 1], observations[t_ + 1])
                posterior_states[t + 1], post_params_ = self.belief_posterior(posterior_input)
                if self.latent_distribution == "Gaussian":
                    posterior_means[t + 1], posterior_std_devs[t + 1] = post_params_
                elif self.latent_distribution == "Categorical":
                    posterior_logits[t + 1] = post_params_

        # Return new hidden states
        beliefs = stack(beliefs)
        prior_states = stack(prior_states)

        prior_params = None
        if self.latent_distribution == "Gaussian":
            prior_params = (stack(prior_means), stack(prior_std_devs))
        elif self.latent_distribution == "Categorical":
            prior_params = (stack(prior_logits),)

        # add posterior computations if observations were present
        if observations is not None:
            posterior_states = stack(posterior_states)
            # add posterior params
            posterior_params = None
            if self.latent_distribution == "Gaussian":
                posterior_params = (stack(posterior_means), stack(posterior_std_devs))
            elif self.latent_distribution == "Categorical":
                posterior_params = (stack(posterior_logits))
        else:
            posterior_states, posterior_params = None, None

        return beliefs, prior_states, prior_params, posterior_states, posterior_params


class Reshape(nn.Module):
    """
    Reshape module for PyTorch.
    """

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """
        Forward pass.
        """

        return x.view(self.shape)


class ObservationModel(nn.Module):
    """
    Observation model.
    Input : Latent -> Ouput : Recontructed Image
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        embedding_size: int,
        activation: Activation = "relu",
    ) -> None:
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        self.decoder = nn.Sequential(
            nn.Linear(belief_size + state_size, embedding_size),
            Reshape([-1, embedding_size, 1, 1]),
            nn.ConvTranspose2d(embedding_size, 128, 5, 2),
            activation(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            activation(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            activation(),
            nn.ConvTranspose2d(32, 3, 6, 2),
        )

    def forward(self, belief: Tensor, state: Tensor) -> Tensor:
        """
        Forward pass.
        """
        return self.decoder(cat(belief, state))


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
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        self.model = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, belief: Tensor, state: Tensor) -> Tensor:
        """
        Forward pass.
        """
        return self.model(cat(belief, state)).squeeze(dim=1)


class CriticModel(nn.Module):
    """
    Critic model.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        activation_function: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        if isinstance(activation_function, str):
            activation_function = getattr(nn, activation_function)
        self.model = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, belief, state):
        """
        Forward pass.
        Input: belief, state
        """

        return self.model(cat(belief, state)).squeeze(dim=1)


class ActorModel(nn.Module):
    """
    Actor model.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        action_size: int,
        activation_function: str = "elu",
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ) -> None:
        super().__init__()
        if isinstance(activation_function, str):
            activation_function = getattr(nn, activation_function)
        self.module = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, hidden_size),
            activation_function(),
            nn.Linear(hidden_size, 2 * action_size),
        )

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self.raw_init_std = torch.log(torch.exp(torch.tensor(self._init_std)) - 1)

    def forward(self, belief, state):
        """
        Forward pass.
        Input: belief, state
        """
        action = self.module(cat(belief, state)).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + self.raw_init_std) + self._min_std
        return action_mean, action_std


class CnnImageEncoder(nn.Module):
    """
    CNN image encoder.
    """

    def __init__(self, embedding_size: int, activation: Activation = "relu") -> None:
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        self.model = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride
            nn.Conv2d(3, 32, 4, 2),
            activation(),
            nn.Conv2d(32, 64, 4, 2),
            activation(),
            nn.Conv2d(64, 128, 4, 2),
            activation(),
            nn.Conv2d(128, 256, 4, 2),  # (B, 3, H, W) ->  (B, 256, H/16, W/16)
            activation(),
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

    def __init__(self, in1_size: int, in2_size: int, out_size: int):
        super().__init__()
        self.in1_linear = nn.Linear(in1_size, out_size)
        self.in2_linear = nn.Linear(in2_size, out_size, bias=False)

    def forward(self, in1, in2):
        """
        Forward pass.
        """

        return self.in1_linear(in1) + self.in1_linear(in2)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    """
    Inverse hyperbolic tangent.
    """

    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    """
    Bijector for the tanh function.
    """

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        """
        Sign of the bijector.
        """
        return 1.0

    def _call(self, x):
        """
        Forward pass.
        Input: x
        """

        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        """
        Inverse pass.
        Input: y
        """

        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        """
        Log of the absolute determinant of the Jacobian.
        """

        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    """
    Sample from a distribution.
    """

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        """
        Name of the distribution.
        """

        return "SampleDist"

    def __getattr__(self, name):
        """
        Get an attribute.
        """

        return getattr(self._dist, name)

    def mean(self):
        """
        Mean of the distribution.
        """
        # TODO: need to be defined. Is dist here supposed to be _dist?
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        """
        Mode of the distribution.
        """

        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        """
        Entropy of the distribution.
        """

        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        """
        Sample from the distribution.
        """

        return self._dist.sample()


def activation_(x, layer_norm=False):
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
        super().__init__()
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

        return self.prior_out(activation_(self.prior_h(h)))

    def forward(
        self,
        embedded: Tensor,  # (T, B, embedding_size)
        action: Tensor,  # (T, B, action_size)
        # reset: Tensor,  # (T, B)
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
        T = action.shape[0]
        for _ in range(T):
            za = activation_(self.za_combination(z_in, action))
            h_out = self.rnn(za, h_in)
            he = activation_(self.he_combination(h_out, embedded))
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

        return priors, posts, samples, states_h
