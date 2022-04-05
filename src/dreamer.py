import torch
from torch import optim, nn
from torch.distributions import Normal
from torch.nn import functional as F

from models import ActorModel, bottle, CriticModel
from planet import Planet
from utils import FreezeParameters, device


class Dreamer(Planet):
    """
    Dreamer is a Planet that learns to plan trajectories using a model-predictive control
    algorithm.
    """

    def __init__(self, params, env):
        super(Dreamer, self).__init__(params, env)

        self.actor_model = ActorModel(
            self.belief_size,
            self.state_size,
            self.hidden_size,
            self.action_size,
            self.dense_activation_function,
        ).to(device)
        self.planner = self.actor_model

        self.critic_model = CriticModel(
            self.belief_size,
            self.state_size,
            self.hidden_size,
            self.dense_activation_function,
        ).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor_model.parameters(),
            lr=params["actor_learning_rate"],
            eps=params["adam_epsilon"],
        )
        self.value_optimizer = optim.Adam(
            self.critic_model.parameters(),
            lr=params["value_learning_rate"],
            eps=params["adam_epsilon"],
        )

        self.discount = params["discount"]
        self.disclam = params["disclam"]

    def imagine_ahead(self, prev_state, prev_belief):
        """
        imagine_ahead is the function to draw the imaginary tracjectory using the
        dynamics model, actor, critic.

        Input:  current state (posterior), current belief (hidden), policy, transition_model
                torch.Size([50, 30]) torch.Size([50, 200])
        Output: generated trajectory of features includes beliefs, prior_states, prior_means,
                prior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30])
                torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        flatten = lambda x: x.view([-1] + list(x.size()[2:]))
        prev_belief = flatten(prev_belief)
        prev_state = flatten(prev_state)

        # Create lists for hidden states
        # (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = self.planning_horizon
        beliefs, prior_states, prior_means, prior_std_devs = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0] = prev_belief, prev_state

        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t]
            actions = self.planner.get_action(beliefs[t].detach(), _state.detach())
            # Compute belief (deterministic hidden state)
            hidden = self.transition_model.act_fn(
                self.transition_model.fc_embed_state_action(
                    torch.cat([_state, actions], dim=1)
                )
            )
            beliefs[t + 1] = self.transition_model.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.transition_model.act_fn(
                self.transition_model.fc_embed_belief_prior(beliefs[t + 1])
            )
            prior_means[t + 1], _prior_std_dev = torch.chunk(
                self.transition_model.fc_state_prior(hidden), 2, dim=1
            )
            prior_std_devs[t + 1] = (
                F.softplus(_prior_std_dev) + self.transition_model.min_std_dev
            )
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[
                t + 1
            ] * torch.randn_like(prior_means[t + 1])
            # Return new hidden states

        # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
        return (
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        )

    def update_actor_critic(self, posterior_states, beliefs) -> dict:
        """
        update_actor_critic is the function to update the actor and critic models.
        """

        logs = {}
        ####################
        # BEHAVIOUR LEARNING
        ####################

        # Imagine trajectories
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()

        with FreezeParameters(self.model_modules):
            (
                imged_beliefs,
                imged_prior_states,
                imged_prior_means,
                imged_prior_std_devs,
            ) = self.imagine_ahead(actor_states, actor_beliefs)

        # Predict Rewards & Values
        with FreezeParameters(self.model_modules + self.critic_model.modules):
            imged_reward = bottle(
                self.reward_model, (imged_beliefs, imged_prior_states)
            )
            value_pred = bottle(self.critic_model, (imged_beliefs, imged_prior_states))

        # Compute Values estimates
        returns = lambda_return(
            imged_reward,
            value_pred,
            bootstrap=value_pred[-1],
            discount=self.discount,
            lambda_=self.disclam,
        )

        # Update Actor weights
        actor_loss = -torch.mean(returns)
        logs["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_model.parameters(), self.grad_clip_norm, norm_type=2
        )
        self.actor_optimizer.step()

        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        # detach the input tensor from the transition network.
        value_dist = Normal(
            bottle(self.critic_model, (value_beliefs, value_prior_states)), 1
        )
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        logs["value_loss"] = value_loss.item()

        # Update model parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_model.parameters(), self.grad_clip_norm, norm_type=2
        )
        self.value_optimizer.step()

        return logs

    def train_step(self) -> dict:
        ####################
        # DYNAMICS LEARNING
        ####################
        logs = super(Dreamer, self).train_step()  # planet
        ####################
        # BEHAVIOUR LEARNING
        ####################
        logs.update(
            self.update_actor_critic(self.posterior_states, self.beliefs)
        )  # dreamer addition
        return logs


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    """
    Compute the lambda-return for a given trajectory.

    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
    """

    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns
