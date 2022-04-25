# from typing import Tuple
#
# import torch
# from torch import Tensor
# from torch.distributions import Normal, kl_divergence, OneHotCategoricalStraightThrough
# # import torch.distributions as D
# # from torch.distributions.transformed_distribution import TransformedDistribution
# from utils import stack, cat, prefill


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
