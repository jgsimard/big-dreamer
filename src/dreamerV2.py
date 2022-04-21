from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence, OneHotCategoricalStraightThrough
# import torch.distributions as D
# from torch.distributions.transformed_distribution import TransformedDistribution
from utils import stack, cat, prefill


from dreamer import Dreamer


class DreamerV2(Dreamer):
    """
    Dreamer is the evolution of Planet. The model predictive controller is replaced by
    an actor-critic to select the next action.
    """

    def __init__(self, params, env):
        super().__init__(params, env)

        self.kl_balance = params['kl_balance']
        if self.kl_loss_weight == 1.0:
            self.kl_loss_weight = 0.1

        # self.discount_model =

    def _kl_loss(
            self,
            posterior_params: Tuple[Tensor, ...],
            prior_params: Tuple[Tensor, ...],
    ) -> Tensor:

        """
        Compute the KL loss.
        """
        if self.latent_distribution == "Gaussian":
            posterior_means, posterior_std_devs = posterior_params
            prior_means, prior_std_devs = prior_params

            dist_post = Normal(posterior_means, posterior_std_devs)
            dist_post_detach = Normal(posterior_means.detach(), posterior_std_devs.detach())

            dist_prior = Normal(prior_means, prior_std_devs)
            dist_prior_detach = Normal(prior_means.detach(), prior_std_devs.detach())
        elif self.latent_distribution == "Categorical":
            posterior_logits, = posterior_params
            prior_logits,  = prior_params

            dist_post = OneHotCategoricalStraightThrough(logits=posterior_logits)
            dist_post_detach = OneHotCategoricalStraightThrough(logits=posterior_logits.detach())

            dist_prior = OneHotCategoricalStraightThrough(logits=prior_logits)
            dist_prior_detach = OneHotCategoricalStraightThrough(logits=prior_logits.detach())
        else:
            raise NotImplementedError()

        kl_lhs = kl_divergence(dist_post_detach, dist_prior).sum(dim=2)
        kl_rhs = kl_divergence(dist_post, dist_prior_detach).sum(dim=2)

        # this is the free bits optimization presented in
        # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
        # https://arxiv.org/abs/1606.04934
        kl_lhs = torch.max(kl_lhs, self.free_nats).mean(dim=(0, 1))
        kl_rhs = torch.max(kl_rhs, self.free_nats).mean(dim=(0, 1))

        # balance
        kl_loss = self.kl_balance * kl_lhs + (1 - self.kl_balance) * kl_rhs

        return kl_loss

    def imagine_ahead(
            self,
            prev_state: Tensor,
            prev_belief: Tensor
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        """
        imagine_ahead is the function to draw the imaginary trajectory using the
        dynamics model, actor, critic. It differs from the same method of DreamerV1 by using
        the discount model to predict its own discounts.

        :param prev_state:
        :param prev_belief:
        :return: generated trajectory of features includes beliefs, prior_states, prior_means,
                prior_std_devs
        """
        flatten = lambda x: x.view([-1] + list(x.size()[2:]))
        prev_belief = flatten(prev_belief)
        prev_state = flatten(prev_state)

        # Create lists for hidden states
        beliefs = prefill(self.planning_horizon)
        prior_states = prefill(self.planning_horizon)

        if self.latent_distribution == "Gaussian":
            prior_means = prefill(self.planning_horizon)
            prior_std_devs = prefill(self.planning_horizon)
        elif self.latent_distribution == "Categorical":
            prior_logits = prefill(self.planning_horizon)

        beliefs[0] = prev_belief
        prior_states[0] = prev_state

        # Loop over time sequence
        for t in range(self.planning_horizon - 1):
            _state = prior_states[t]
            # discount = self.discount(beliefs[t], _state)
            actions = self.get_action(beliefs[t].detach(), _state.detach())

            # Compute belief (deterministic hidden state)
            hidden = self.transition_model.fc_embed_state_action(cat(_state, actions))
            beliefs[t + 1] = self.transition_model.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            prior_states[t + 1], prior_params_ = self.transition_model.belief_prior(beliefs[t + 1])
            if self.latent_distribution == "Gaussian":
                prior_means[t + 1], prior_std_devs[t + 1] = prior_params_
            elif self.latent_distribution == "Categorical":
                prior_logits[t + 1] = prior_params_

        beliefs = stack(beliefs)
        prior_states = stack(prior_states)

        if self.latent_distribution == "Gaussian":
            prior_params = (stack(prior_means), stack(prior_std_devs))
        elif self.latent_distribution == "Categorical":
            prior_params = (stack(prior_logits),)

        return beliefs, prior_states, prior_params
