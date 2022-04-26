from typing import Union, Optional, List, Tuple

import numpy as np
import torch
from torch import nn, Tensor

import torch.nn.functional as F
import torch.distributions as D


from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from utils import cat, stack, prefill, build_mlp


Activation = Union[str, nn.Module]

patch_typeguard()



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
    Gaussian belief model (for PlaNet and DreamerV1)
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            state_size: int,
            activation: str,
            min_std_dev: float
    ) -> None:
        super().__init__()
        self.min_std_dev = min_std_dev
        self.model = build_mlp(input_size, hidden_size, 2*state_size, 1, activation)

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
    CategoricalBeliefModel
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            discrete_latent_dimensions: int,
            discrete_latent_classes: int,
            activation: str
    ) -> None:
        super().__init__()
        self.discrete_latent_dimensions = discrete_latent_dimensions
        self.discrete_latent_classes = discrete_latent_classes
        self.dim = discrete_latent_classes * discrete_latent_dimensions
        self.model = build_mlp(
            input_size,
            hidden_size,
            self.dim,
            1,
            activation
        )
        assert discrete_latent_classes

    def forward(self, belief: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Args:
            belief (Tensor): belief of shape: (batch_size, belief_size)
        Returns:
            Tensor: belief of shape: (batch_size, belief_size)
        """
        logits = self.model(belief)
        batch_shape = logits.shape[:-1]
        shape = (*batch_shape, self.discrete_latent_dimensions, self.discrete_latent_classes)
        logits = logits.reshape(shape)
        # use straight through gradient
        # https://arxiv.org/abs/1308.3432
        dist = D.OneHotCategoricalStraightThrough(logits=logits)
        state = dist.rsample()
        state = state.view(*batch_shape, self.dim)
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
        activation: Optional[str] = "ELU",
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
        self.fc_embed_state_action = build_mlp(
            state_size+action_size, -1, belief_size, 0, output_activation=activation
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

    @typechecked
    def forward(
        self,
        init_state: TensorType['batch_size', 'state_size'],
        actions: TensorType['seq_len', 'batch_size', 'action_size'],
        init_belief: TensorType['batch_size', 'belief_size'],
        embeddings: Optional[TensorType['seq_len', 'batch_size', 'embedding_size']] = None,
        nonterminals: Optional[TensorType['seq_len', 'batch_size', 1]] = None,
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
        # print("Twerk 1")
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

        # print("Twerk 2")
        beliefs[0] = init_belief
        prior_states[0] = init_state
        posterior_states[0] = init_state

        # print("Twerk 3")
        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if embeddings is None else posterior_states[t]

            # print("Twerk 4")
            # print(f'_state.shape={_state.shape}')
            # print(f'nonterminals[t].shape={nonterminals[t].shape}')
            # Mask if previous transition was terminal
            _state = _state if nonterminals is None else _state * nonterminals[t]

            # print("Twerk 5")
            # Compute belief (deterministic hidden state)
            hidden = self.fc_embed_state_action(cat(_state, actions[t]))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # print("Twerk 6")
            # Compute state prior by applying transition dynamics
            prior_states[t + 1], prior_params_ = self.belief_prior(beliefs[t + 1])
            if self.latent_distribution == "Gaussian":
                prior_means[t + 1], prior_std_devs[t + 1] = prior_params_
            elif self.latent_distribution == "Categorical":
                prior_logits[t + 1] = prior_params_

            if embeddings is not None:
                # Compute state posterior by applying transition dynamics and using
                # current observation.
                t_ = t - 1  # Using t_ to deal with different time indexing for observations
                posterior_input = cat(beliefs[t + 1], embeddings[t_ + 1])
                posterior_states[t + 1], post_params_ = self.belief_posterior(posterior_input)
                if self.latent_distribution == "Gaussian":
                    posterior_means[t + 1], posterior_std_devs[t + 1] = post_params_
                elif self.latent_distribution == "Categorical":
                    posterior_logits[t + 1] = post_params_

        # print("Twerk 7")
        # print(len(beliefs), [b.shape for b in beliefs])
        # print(len(prior_states), [b.shape for b in prior_states])
        # Return new hidden states
        beliefs = stack(beliefs)
        prior_states = stack(prior_states)

        prior_params = None
        if self.latent_distribution == "Gaussian":
            prior_params = (stack(prior_means), stack(prior_std_devs))
        elif self.latent_distribution == "Categorical":
            prior_params = (stack(prior_logits),)

        # print("Twerk 8")
        # add posterior computations if observations were present
        if embeddings is not None:
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
        self.output_shape = (3, 64, 64)

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

    def forward(
            self,
            belief: Tensor,
            state: Tensor
    ) -> Tensor:
        """
        Forward pass.
        """
        batch_shape = belief.shape[:-1]  # L,B or just B
        x = torch.cat([belief, state], dim=-1)  # (L, B , S+Be)
        x = self.decoder(x)
        x = x.view(*batch_shape, *self.output_shape)
        return x


class DenseModel(nn.Module):
    """
    Dense model.
    """

    def __init__(
            self,
            input_size: int,
            # belief_size: int,
            # state_size: int,
            hidden_size: int,
            output_size: int = 1,
            activation: str = "ELU",
            n_layers: int = 4,
            distribution: str = 'normal'
    ) -> None:
        super().__init__()
        self.model = build_mlp(input_size, hidden_size, output_size, n_layers, activation)
        # self.model = build_mlp(belief_size + state_size, hidden_size, 1, n_layers, activation)
        self.distribution = distribution

    # def forward(self, belief: Tensor, state: Tensor) -> Tensor:
    #     """
    #     Forward pass.
    #     """
    #     x = torch.cat([belief, state], dim=-1)  # (L, B , S+Be)
    #     x = self.model(x)
    #     return x
    def forward(self, *args: Tuple[Tensor]) -> Tensor:
        """
        Forward pass.
        """
        if len(args) == 2:
            # print("Twerk")
            belief, state = args
            # print("Twerk 2")
            # print(belief.shape, state.shape)
            x = torch.cat([belief, state], dim=-1)  # (L, B , S+Be)
            # print("Twerk 3")
        else:
            x, = args

        x = self.model(x)
        return x

# class RewardModel(nn.Module):
#     """
#     Reward model.
#     """
#
#     def __init__(
#         self,
#         belief_size: int,
#         state_size: int,
#         hidden_size: int,
#         activation: str = "ELU",
#         n_layers:int = 4
#     ) -> None:
#         super().__init__()
#         self.model = build_mlp(belief_size+state_size, hidden_size, 1, n_layers, activation)
#
#     def forward(self, belief: Tensor, state: Tensor) -> Tensor:
#         """
#         Forward pass.
#         """
#         x = torch.cat([belief, state], dim=-1)  # (L, B , S+Be)
#         x = self.model(x)
#         return x


# class CriticModel(nn.Module):
#     """
#     Critic model.
#     """
#
#     def __init__(
#         self,
#         belief_size: int,
#         state_size: int,
#         hidden_size: int,
#         activation: Optional[str] = "ELU",
#         n_layers: int = 4
#     ) -> None:
#         super().__init__()
#
#         self.model = build_mlp(
#             input_size = belief_size + state_size,
#             output_size = 1,
#             n_layers = n_layers,
#             hidden_size = hidden_size,
#             activation = activation
#         )
#     def forward(self, belief, state):
#         """
#         Forward pass.
#         Input: belief, state
#         """
#
#         return self.model(cat(belief, state)).squeeze(dim=1)


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
        activation_function: str = "ELU",
        action_distribution: str = 'Gaussian',
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
        n_layers: int = 4
    ) -> None:
        super().__init__()
        if action_distribution == 'Gaussian':
            output_size = 2 * action_size
        elif action_distribution == 'Categorical':
            output_size = action_size
        else:
            NotImplementedError()

        self.model = build_mlp(
            input_size=belief_size + state_size,
            output_size= output_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            activation=activation_function
        )

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self.raw_init_std = torch.log(torch.exp(torch.tensor(self._init_std)) - 1)
        self.action_distribution = action_distribution

    def forward(self, belief, state):
        """
        Forward pass.
        Input: belief, state
        """
        out_model = self.model(cat(belief, state)).squeeze(dim=1)

        if self.action_distribution == 'Gaussian':
            action_mean, action_std_dev = torch.chunk(out_model, 2, dim=1)
            action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
            action_std = F.softplus(action_std_dev + self.raw_init_std) + self._min_std
            return action_mean, action_std
        if self.action_distribution == 'Categorical':
            action_dist = self.get_action_dist(out_model)
            action = action_dist.sample()
            action = action + action_dist.probs - action_dist.probs.detach()
            return action, action_dist

        raise NotImplementedError()


class CnnImageEncoder(nn.Module):
    """
    CNN image encoder.
    """

    def __init__(self, embedding_size: int, activation: Activation = "ELU") -> None:
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

        batch_shape = observation.shape[:-3]  # L,B or just B
        obs_shape = observation.shape[-3:]  # C, W, H

        embedding = self.model(observation.view(-1, *obs_shape))
        embedding = torch.reshape(embedding, (*batch_shape, -1))
        return embedding


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


class DiscountModel(nn.Module):
    """
    Discount model as in https://arxiv.org/pdf/2010.02193.pdf
    """
    def __init__(
            self,
            belief_size: int,
            embedding_size: int,
            hidden_size: int,
            n_layers: int,
            activation_function: Optional[str] = "relu"
    ) -> nn.Sequential:
        super().__init__()

        self.model = build_mlp(
            input_size=belief_size + embedding_size,
            output_size=1,
            n_layers=n_layers,
            hidden_size=hidden_size,
            activation=activation_function,
            output_activation='Identity'
        )

    def forward(self, belief: Tensor, embedding: Tensor):
        """
        Forward pass.
        """
        x = cat(belief, embedding)
        logits = self.model(x)
        dist = D.Bernoulli(logits=logits)
        return dist.rsample()




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
