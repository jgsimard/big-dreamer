import torch
from torch import nn


class MPCPlanner(nn.Module):
    """
    Model-predictive control planner with cross-entropy method and learned transition model.
    """

    def __init__(
        self,
        action_size,
        planning_horizon,
        optimisation_iters,
        candidates,
        top_candidates,
        transition_model,
        reward_model,
    ):
        super().__init__()
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    def forward(self, belief, state):
        """
        Plan actions for the given belief and state.

        :param belief: (B, H, Z)
        :param state: (B, Z)
        :return: actions (B, H, A)
        """

        batch_size, horizon, state_size = belief.size(0), belief.size(1), state.size(1)
        belief = belief.unsqueeze(dim=1)\
            .expand(batch_size, self.candidates, horizon)\
            .reshape(-1, horizon)
        state = state.unsqueeze(dim=1)\
            .expand(batch_size, self.candidates, state_size)\
            .reshape(-1, state_size)

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean = torch.zeros(
            self.planning_horizon, batch_size, 1, self.action_size, device=belief.device
        )
        action_std_dev = torch.ones(
            self.planning_horizon, batch_size, 1, self.action_size, device=belief.device
        )

        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once,
            # batched over particles)
            # Sample actions (time x (batch x candidates) x actions)
            noise = torch.randn(
                self.planning_horizon,
                batch_size,
                self.candidates,
                self.action_size,
                device=action_mean.device,
            )
            actions = (action_mean + action_std_dev * noise).view(
                self.planning_horizon, batch_size * self.candidates, self.action_size
            )
            # Sample next states
            # [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates
            beliefs, states, _, _, _ = self.transition_model(state, actions, belief)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            # output from r-model[12000]->view[12, 1000]->sum[1000]
            returns = (
                self.reward_model(beliefs.view(-1, horizon), states.view(-1, state_size))
                .view(self.planning_horizon, -1)
                .sum(dim=0)
            )
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(batch_size, self.candidates).topk(
                self.top_candidates, dim=1, largest=True, sorted=False
            )
            # Fix indices for unrolled actions
            topk += self.candidates * torch.arange(
                0, batch_size, dtype=torch.int64, device=topk.device
            ).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(
                self.planning_horizon, batch_size, self.top_candidates, self.action_size
            )

            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(dim=2, keepdim=True)
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)

        # Return first action mean µ_t
        return action_mean[0].squeeze(dim=1)
