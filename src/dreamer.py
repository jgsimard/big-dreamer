from typing import Tuple, Dict, Any
import copy

import torch
from torch import optim, nn, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal, kl_divergence, OneHotCategoricalStraightThrough
import torch.distributions as D
from torch.distributions.transformed_distribution import TransformedDistribution

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from models import ActorModel, TanhBijector, SampleDist, DenseModel
from planet import Planet
from utils import FreezeParameters, device, cat, prefill, stack, polyak_update


patch_typeguard()


class Dreamer(Planet):
    """
    Dreamer is the evolution of Planet. The model predictive controller is replaced by
    an actor-critic to select the next action.
    """

    def __init__(self, params: Dict[str, Any], env):
        super().__init__(params, env)

        self.entropy_weight = params['ActorCritic']["entropy_weight"]
        self.gradient_mixing = params['ActorCritic']["gradient_mixing"]
        self.polyak_avg = params['ActorCritic']["polyak_avg"]
        self.use_target = params['ActorCritic']['use_target']
        self.discount = params["discount"]
        self.disclam = params["disclam"]
        self.kl_balance = params['kl_balance']
        self.use_discount = params["use_discount"]
        self.discount_weight = params['discount_weight']

        self.actor = ActorModel(
            self.belief_size,
            self.state_size,
            self.hidden_size,
            self.action_size,
            self.dense_activation_function,
            params['action_distribution']
        ).to(device)

        self.critic = DenseModel(
            self.belief_size + self.state_size,
            self.hidden_size,
            activation=self.dense_activation_function,
        ).to(device)

        self.critic_modules = [self.critic]
        if self.use_target:
            self.critic_target = DenseModel(
                self.belief_size + self.state_size,
                self.hidden_size,
                activation=self.dense_activation_function,
            ).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_modules += [self.critic_target]

        if params['jit']:
            self.actor = torch.jit.script(self.actor)
            self.critic = torch.jit.script(self.critic)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=params['ActorCritic']["actor_learning_rate"],
            eps=params["adam_epsilon"],
            weight_decay=params['weight_decay']
        )
        self.value_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=params['ActorCritic']["value_learning_rate"],
            eps=params["adam_epsilon"],
            weight_decay=params['weight_decay']
        )



        if self.use_discount:
            self.discount_model = DenseModel(
                self.belief_size + self.state_size,
                self.hidden_size,
                activation=self.dense_activation_function,
            ).to(device)
        else:
            # just to shut up pylint
            self.discount_model = nn.Identity()

        self._initialize_optimizers()

    def _get_dist(
            self,
            distribution_parameters: Tuple[Tensor, ...],
            detach: bool=False
    ) -> torch.distributions.Distribution:
        if self.latent_distribution == "normal":
            means, std_devs = distribution_parameters
            if detach:
                means, std_devs = means.detach(), std_devs.detach()
            return Normal(means, std_devs)
        if self.latent_distribution == "categorical":
            logits, = distribution_parameters
            if detach:
                logits = logits.detach()
            return OneHotCategoricalStraightThrough(logits=logits)
        # if it gets here, there is a problem
        raise NotImplementedError(f'{self.latent_distribution}  is yet yet implemented')

    def _kl_loss(
            self,
            posterior_params: Tuple[Tensor, ...],
            prior_params: Tuple[Tensor, ...],
    ) -> Tensor:

        """
        Compute the KL loss.
        """
        prior_dist = self._get_dist(prior_params)
        posterior_dist = self._get_dist(posterior_params)

        if self.kl_balance == -1:
            div = kl_divergence(posterior_dist, prior_dist).sum(dim=2)

            # this is the free bits optimization presented in
            # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
            # https://arxiv.org/abs/1606.04934
            kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))

        else:
            prior_dist_detach = self._get_dist(prior_params, detach=True)
            posterior_dist_detach = self._get_dist(posterior_params, detach=True)

            kl_lhs = kl_divergence(posterior_dist_detach, prior_dist).mean()
            kl_rhs = kl_divergence(posterior_dist, prior_dist_detach).mean()

            # this is the free bits optimization presented in
            # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
            # https://arxiv.org/abs/1606.04934
            kl_lhs = torch.max(kl_lhs, self.free_nats)
            kl_rhs = torch.max(kl_rhs, self.free_nats)

            # balance
            kl_loss = self.kl_balance * kl_lhs + (1 - self.kl_balance) * kl_rhs

        return kl_loss

    def _initialize_optimizers(self) -> None:
        """
        Initialize the optimizers.
        """

        self.model_modules = [
            self.transition_model,
            self.observation_model,
            self.reward_model,
            self.encoder,
        ]

        self.model_params = (
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.encoder.parameters())
        )

        if self.use_discount:
            self.model_modules += [self.discount_model]
            self.model_params += list(self.discount_model.parameters())

        self.model_optimizer = optim.Adam(
            self.model_params,
            lr=self.model_learning_rate,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay
        )

    @typechecked
    def imagine_ahead(
            self,
            prev_state: TensorType['seq_len', 'batch_size', 'state_size'],
            prev_belief: TensorType['seq_len', 'batch_size', 'belief_size']
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...], Tensor]:
        """
        imagine_ahead is the function to draw the imaginary trajectory using the
        dynamics model, actor, critic.

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
        action_entropy = prefill(self.planning_horizon)

        if self.latent_distribution == "normal":
            prior_means = prefill(self.planning_horizon)
            prior_std_devs = prefill(self.planning_horizon)
        elif self.latent_distribution == "categorical":
            prior_logits = prefill(self.planning_horizon)

        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        action_entropy[0] = torch.zeros(len(beliefs[0]), device=device)

        # Loop over time sequence
        for t in range(self.planning_horizon - 1):
            _state = prior_states[t]
            actions, action_entropy[t+1] = self.get_action(beliefs[t].detach(), _state.detach())


            # Compute belief (deterministic hidden state)
            hidden = self.transition_model.fc_embed_state_action(cat(_state, actions))
            beliefs[t + 1] = self.transition_model.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            prior_states[t + 1], prior_params_ = self.transition_model.belief_prior(beliefs[t + 1])
            if self.latent_distribution == "normal":
                prior_means[t + 1], prior_std_devs[t + 1] = prior_params_
            elif self.latent_distribution == "categorical":
                prior_logits[t + 1] = prior_params_

        beliefs = stack(beliefs)
        prior_states = stack(prior_states)
        action_entropy = stack(action_entropy)
        if self.latent_distribution == "normal":
            prior_params = (stack(prior_means), stack(prior_std_devs))
        elif self.latent_distribution == "categorical":
            prior_params = (stack(prior_logits),)

        return beliefs, prior_states, prior_params, action_entropy

    def _discount_loss(
            self,
            beliefs: TensorType['seq_len', 'batch_size', 'belief_size'],
            posterior_states: TensorType['seq_len', 'batch_size', 'state_size'],
            nonterminals: TensorType['seq_len', 'batch_size', 1]
    ) -> Tensor:
        discount_target = nonterminals.float()
        discount_logits = self.discount_model(beliefs, posterior_states)
        discount_dist = D.Independent(D.Bernoulli(logits=discount_logits), 1)

        discount_loss = -torch.mean(discount_dist.log_prob(discount_target))

        return discount_loss

    def train_step(self) -> Dict[str, float]:
        """
        Used to train the model.
        """
        logs = {}

        ####################
        # DYNAMICS LEARNING
        ####################
        # print("HELLO 1")
        # Draw sequences chunks
        obs, actions, rewards, nonterminals = self.buffer.sample(self.batch_size, self.seq_len)

        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(self.batch_size, self.belief_size, device=device)
        init_state = torch.zeros(self.batch_size, self.state_size, device=device)

        # print("HELLO 2")
        # compute image embeddings
        embeddings = self.encoder(obs[1:])

        # print("HELLO 3")
        beliefs, _, prior_params, posterior_states, posterior_params = self.transition_model(
            init_state,
            actions[:-1],
            init_belief,
            embeddings,
            nonterminals[:-1],
        )
        # sum over final dims, average over batch and time

        # print("HELLO 4")
        # compute losses
        observation_loss = self._observation_loss(beliefs, posterior_states, obs[1:])
        # print("HELLO 5")
        reward_loss = self._reward_loss(beliefs, posterior_states, rewards[:-1])
        # print("HELLO 6")
        kl_loss = self._kl_loss(posterior_params, prior_params)
        model_loss = observation_loss + reward_loss + kl_loss * self.kl_loss_weight

        if self.use_discount:
            discount_loss = self._discount_loss(beliefs, posterior_states, nonterminals[:-1])
            model_loss += discount_loss * self.discount_weight
            logs['discount_loss'] = discount_loss

        # log stuff
        logs["observation_loss"] = observation_loss.item()
        logs["reward_loss"] = reward_loss.item()
        logs["kl_loss"] = kl_loss.item()
        logs["model_loss"] = model_loss.item()

        # Update model parameters
        self.model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self.model_params, self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()

        ####################
        # BEHAVIOUR LEARNING
        ####################

        # Imagine trajectories
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()

        with FreezeParameters(self.model_modules):
            imged_belief, imged_prior_state, _, action_entropy = self.imagine_ahead(
                actor_states, actor_beliefs)

        # Predict Rewards & Values
        with FreezeParameters(self.model_modules + [self.critic_modules]):
            imged_reward = self.reward_model(imged_belief, imged_prior_state)
            value_pred = self.critic_target(imged_belief, imged_prior_state)
            if self.use_discount:
                discount_logits = self.discount_model(imged_belief, imged_prior_state)
                discount_dist = D.Independent(D.Bernoulli(logits=discount_logits), 1)
                discount_arr = self.discount * torch.round(discount_dist.base_dist.probs)

        # Compute Values estimates
        returns = lambda_return(
            imged_reward,
            value_pred,
            bootstrap=value_pred[-1],
            discount=self.discount,
            lambda_=self.disclam,
        )
        if self.gradient_mixing == -1:
            objective = returns
        else:
            raise NotImplementedError("gradient_mixing not yet implemented ")

        # Update Actor weights
        policy_entropy = action_entropy.unsqueeze(-1)

        if self.entropy_weight != -1:
            # print(objective.shape, policy_entropy.shape)
            objective = objective + self.entropy_weight * policy_entropy
        if self.use_discount:
            # discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]),
            # discount_arr[:, 1:]], dim=1)
            discount_arr[:, 0, 0] = 1.0  # at least the first one of each trajectory =1
            discount = torch.cumprod(discount_arr, 0)
            objective = discount * objective
        actor_loss = -objective.mean()
        # actor_loss = - objective.mean(dim=1).sum()


        # actor_loss = -torch.mean(returns)
        # logs["actor_returns"] = returns.item()
        logs["actor_loss"] = actor_loss.item()
        logs["policy_entropy"] = torch.mean(policy_entropy).item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm, norm_type=2)
        self.actor_optimizer.step()

        # detach the input tensor from the transition network
        with torch.no_grad():
            value_beliefs = imged_belief.detach()
            value_prior_states = imged_prior_state.detach()
            target_return = returns.detach()
            if self.use_discount:
                value_discount = discount.detach()

        value_dist = Normal(self.critic(value_beliefs, value_prior_states), 1)

        if self.use_discount:
            value_loss = -(value_discount * value_dist.log_prob(target_return)).mean()
        else:
            value_loss = -value_dist.log_prob(target_return).mean()
        logs["value_loss"] = value_loss.item()

        # Update model parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm, norm_type=2)
        self.value_optimizer.step()

        return logs

    def eval(self) -> None:
        """
        Set the models to evaluation mode.
        """

        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        if self.use_discount:
            self.discount_model.eval()

    def train(self) -> None:
        """
        Set the models to training mode.
        """

        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
        self.actor.train()
        self.critic.train()
        if self.use_discount:
            self.discount_model.train()

    def update_critic(self) -> None:
        """
        Update the critic weights using polyack averaging (soft update : see DDPG)
        """
        polyak_update(self.critic_target, self.critic, self.polyak_avg)

    def get_action(self, belief: Tensor, state: Tensor, deterministic: bool = False) -> Tensor:
        """
        Get action.
        """

        action_mean, action_std = self.actor(belief, state)
        dist = D.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())  # clip [-1,1]
        dist = D.Independent(dist, 1)
        dist = SampleDist(dist)

        if deterministic:
            sample = dist.mode()
        else:
            sample = dist.rsample()
        return sample, dist.entropy()


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
