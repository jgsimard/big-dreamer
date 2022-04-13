import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence
# import torch.distributions as D
# from torch.distributions.transformed_distribution import TransformedDistribution


from dreamer import Dreamer


class DreamerV2(Dreamer):
    """
    Dreamer is the evolution of Planet. The model predictive controller is replaced by
    an actor-critic to select the next action.
    """

    def __init__(self, params, env):
        super().__init__(params, env)

        self.kl_balance = params['kl_balance']


    def kl_loss(
        self,
        posterior_means: Tensor,
        posterior_std_devs: Tensor,
        prior_means: Tensor,
        prior_std_devs: Tensor,
    ) -> Tensor:
        """
        Compute the KL loss.
        """
        dist_post = Normal(posterior_means, posterior_std_devs)
        dist_post_detach = Normal(posterior_means.detach(), posterior_std_devs.detach())

        dist_prior = Normal(prior_means, prior_std_devs)
        dist_prior_detach = Normal(prior_means.detach(), prior_std_devs.detach())

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
