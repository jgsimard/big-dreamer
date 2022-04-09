from typing import Union, Optional, List

import numpy as np
import torch
from torch import nn, Tensor

import torch.nn.functional as F
import torch.distributions as D
from torch.distributions.transformed_distribution import TransformedDistribution

# from torch.nn.common_types import *

Activation = Union[str, nn.Module]


def merge_belief_and_state(belief: Tensor, state: Tensor) -> Tensor:
    """
    Merge the belief and state into a single tensor.
    """

    return torch.cat([belief, state], dim=1)

def bottle(f, x_tuple: tuple # (belief, posterior_states)
    ) -> Tensor: # shape: (belief[0].size(), belief[1].size(), *f().size()[1:])
    """
    Wraps the input tuple for a function to process a time x batch x features sequence in
    batch x features (assumes one output)
    """

    x_sizes = tuple([x.size() for x in x_tuple])
    y = f(
        *[x[0].view(x[1][0] * x[1][1], *x[1][2:]) for x in zip(x_tuple, x_sizes)]
    )
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

    return output

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
        activation_function: Optional[str]="relu",
        min_std_dev: float=0.1,
    ) -> None:
        super(TransitionModel, self).__init__()

        # general attibutes
        if isinstance(activation_function, str):
            activation_function = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        # recurrent component
        self.rnn = nn.GRUCell(belief_size, belief_size)
        # model components
        self.fc_embed_state_action = nn.Sequential(
            nn.Linear(state_size + action_size, belief_size),
            activation_function
        )
        self.fc_embed_belief_prior = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            activation_function
        )
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Sequential(
            nn.Linear(belief_size + embedding_size, hidden_size),
            activation_function
        )
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]
    
    def forward(
        self,
        prev_state: Tensor,
        actions: Tensor,
        prev_belief: Tensor,
        observations: Optional[Tensor] = None,
        nonterminals: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states,
                posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30])
                torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
                torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        # Create lists for hidden states
        # (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T

        beliefs[0], prior_states[0], posterior_states[0] = (
            prev_belief,
            prev_state,
            prev_state,
        )
        # Loop over time sequence
        for t in range(T - 1):
            _state = (
                prior_states[t] if observations is None else posterior_states[t]
            )  # Select appropriate previous state
            _state = (
                _state if nonterminals is None else _state * nonterminals[t]
            )  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.fc_embed_state_action(
                torch.cat([_state, actions[t]], dim=1)
            )
            
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.fc_embed_belief_prior(beliefs[t + 1])
            prior_means[t + 1], _prior_std_dev = torch.chunk(
                self.fc_state_prior(hidden), 2, dim=1
            )
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[
                t + 1
            ] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using
                # current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1
                )
                posterior_std_devs[t + 1] = (
                    F.softplus(_posterior_std_dev) + self.min_std_dev
                )
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[
                    t + 1
                ] * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden


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
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)

        self.embedding_size = embedding_size
        self.linear = nn.Linear(belief_size + state_size, embedding_size)
        self.deconvs = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride: _size_2_t = 1
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
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)

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

        return self.model(merge_belief_and_state(belief, state)).squeeze(dim=1)


class CriticModel(nn.Module):
    """
    Critic model.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int, 
        hidden_size: int,
        activation_function: Optional[str]="relu"
    ) -> None:
        super().__init__()
        if isinstance(activation_function, str):
            activation_function = getattr(F, activation_function)
        self.model = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, belief, state):
        """
        Forward pass.
        Input: belief, state
        """

        return self.model(merge_belief_and_state(belief, state)).squeeze(dim=1)


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
        activation_function: str="elu",
        min_std: float=1e-4,
        init_std: float=5,
        mean_scale: float=5,
    ) -> None:
        super().__init__()
        if isinstance(activation_function):
            activation_function = getattr(F, activation_function)
        self.module = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, 2 * action_size)
        )

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, belief, state):
        """
        Forward pass.
        Input: belief, state
        """

        raw_init_std = torch.log(torch.exp(self._init_std) - 1)

        x = merge_belief_and_state(belief, state)
        action = self.model(x).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, deterministic=False):
        """
        Get action.
        """

        action_mean, action_std = self.forward(belief, state)
        dist = D.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)

        if deterministic:
            return dist.mode()

        return dist.rsample()

class CnnImageEncoder(nn.Module):
    """
    CNN image encoder.
    """

    def __init__(self, embedding_size: int, activation: Activation = "relu") -> None:
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(F, activation)

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
