from typing import Tuple, Dict, Any, Union
import copy
import os

import torch
from torch import optim, nn, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal, kl_divergence, OneHotCategoricalStraightThrough, Independent
import torch.distributions as D
#from torch.distributions.transformed_distribution import TransformedDistribution

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from memory import ExperienceReplay
from utils import FreezeParameters, device, cat, prefill, stack, polyak_update
from models import \
    RSSM, ObservationModel, DenseModel, CnnImageEncoder, ActorModel#, TanhBijector, SampleDist
from planner import MPCPlanner
from env import EnvBatcher


patch_typeguard()


def _get_dist(
        distribution_parameters: Tuple[Tensor, ...],
        distribution: str = "normal",
        detach: bool = False
) -> torch.distributions.Distribution:
    if distribution == "normal":
        means, std_devs = distribution_parameters
        if detach:
            means, std_devs = means.detach(), std_devs.detach()
        return Normal(means, std_devs)
    if distribution == "categorical":
        logits, = distribution_parameters
        if detach:
            logits = logits.detach()
        return OneHotCategoricalStraightThrough(logits=logits)
    # if it gets here, there is a problem
    raise NotImplementedError(f'{distribution}  is not yet implemented')


class Agent(nn.Module):
    """
    TODO.
    """

    def __init__(self, params: Dict[str, Any], env):
        super().__init__()
        self.env = env

        # Initialize base parameters from the config file.
        self.belief_size = params["belief_size"]
        self.state_size = params["state_size"]
        self.action_size = env.action_size
        self.hidden_size = params["hidden_size"]
        self.embedding_size = params["embedding_size"]
        self.n_layers = params['n_layers']

        self.dense_activation_function = params["dense_activation_function"]
        self.cnn_activation_function = params["cnn_activation_function"]

        self.batch_size = params["batch_size"]
        self.seq_len = params["seq_len"]

        self.seed_steps = params["seed_steps"]

        self.experience_size = params["experience_size"]
        self.bit_depth = params["bit_depth"]
        self.kl_loss_weight = params['kl_loss_weight']
        self.latent_distribution = params['latent_distribution']
        self.discrete_latent_dimensions = params['discrete_latent_dimensions']
        self.discrete_latent_classes = params['discrete_latent_classes']


        self.planning_horizon = params["planning_horizon"]
        self.pixel_observation = params['pixel_observation']

        self.planner_algorithm = params['planner_algorithm']

        if self.planner_algorithm == 'mpc':
            self.optimisation_iters = params["MPC"]["optimisation_iters"]
            self.candidates = params["MPC"]["candidates"]
            self.top_candidates = params["MPC"]["top_candidates"]
        elif self.planner_algorithm == 'actor_critic':
            ac_params = params['ActorCritic']
            self.actor_learning_rate = ac_params['actor_learning_rate']
            self.value_learning_rate = ac_params['value_learning_rate']
            self.entropy_weight = ac_params["entropy_weight"]
            self.gradient_mixing = ac_params["gradient_mixing"]
            self.polyak_avg = ac_params["polyak_avg"]
            self.use_target = ac_params['use_target']
        else:
            raise NotImplementedError(f"{self.planner_algorithm} is not a valid planner")

        self.discount = params["discount"]
        self.lambda_ = params["disclam"]
        self.kl_balance = params['kl_balance']
        self.use_discount = params["use_discount"]
        self.discount_weight = params['discount_weight']

        self.model_learning_rate = params["model_learning_rate"]
        self.adam_epsilon = params["adam_epsilon"]
        self.weight_decay = params['weight_decay']
        self.action_noise = params["action_noise"]
        self.grad_clip_norm = params["grad_clip_norm"]
        self.action_repeat = params["action_repeat"]

        self.action_distribution = params['action_distribution']

        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1,), params["free_nats"], device=device)

        self.buffer = ExperienceReplay(
            self.experience_size,
            self.env.action_size,
            self.bit_depth,
            self.pixel_observation,
            self.env.observation_size,
            device,
        )

        self.initialize_models()
        self.initialize_optimizers()
        self.load(params)

    def initialize_models(self) -> None:
        """
        Initialize the different models.
        """
        self.rssm = RSSM(
            self.belief_size,
            self.state_size,
            self.action_size,
            self.hidden_size,
            self.embedding_size,
            self.dense_activation_function,
            latent_distribution=self.latent_distribution,
            discrete_latent_dimensions=self.discrete_latent_dimensions,
            discrete_latent_classes=self.discrete_latent_classes
        ).to(device=device)

        self.reward_model = DenseModel(
            self.belief_size + self.state_size,
            self.hidden_size,
            activation=self.dense_activation_function,
            n_layers=self.n_layers
        ).to(device=device)

        # Encoder / Decoders
        if self.pixel_observation:
            self.encoder = CnnImageEncoder(
                    self.embedding_size,
                    self.cnn_activation_function,
                ).to(device=device)
            self.observation_model = ObservationModel(
                self.belief_size,
                self.state_size,
                self.embedding_size,
                self.cnn_activation_function,
            ).to(device=device)
        else:
            self.encoder = DenseModel(
                self.env.observation_size,
                self.hidden_size,
                self.embedding_size,
                self.dense_activation_function,
                self.n_layers
            )
            self.observation_model = DenseModel(
                self.belief_size + self.state_size,
                self.hidden_size,
                self.env.observation_size,
                self.dense_activation_function,
                self.n_layers
            )

        # PLANNING ALGORITHM
        if self.planner_algorithm == 'mpc':
            self.planner = MPCPlanner(
                self.action_size,
                self.planning_horizon,
                self.optimisation_iters,
                self.candidates,
                self.top_candidates,
                self.rssm,
                self.reward_model,
            )
        if self.planner_algorithm == 'actor_critic':
            self.actor = ActorModel(
                self.belief_size,
                self.state_size,
                self.hidden_size,
                self.action_size,
                self.dense_activation_function,
                self.action_distribution
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

            if self.use_discount:
                self.discount_model = DenseModel(
                    self.belief_size + self.state_size,
                    self.hidden_size,
                    activation=self.dense_activation_function,
                ).to(device)

    def initialize_optimizers(self) -> None:
        """
        Initialize the optimizers.
        """

        self.model_modules = [
            self.rssm,
            self.observation_model,
            self.reward_model,
            self.encoder,
        ]

        self.model_params = (
            list(self.rssm.parameters())
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

        if self.planner_algorithm == 'actor_critic':
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(),
                lr=self.actor_learning_rate,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=self.value_learning_rate,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )

    def load(self, params: Dict[str, Any]) -> None:
        """
        Load models if they exist.
        """
        if params["models"] != "" and os.path.exists(params["models"]):
            print("Loading models...")
            model_dicts = torch.load(params["models"])
            self.transition_model.load_state_dict(model_dicts["transition_model"])
            self.observation_model.load_state_dict(model_dicts["observation_model"])
            self.reward_model.load_state_dict(model_dicts["reward_model"])
            self.encoder.load_state_dict(model_dicts["encoder"])
            self.model_optimizer.load_state_dict(model_dicts["model_optimizer"])
            raise NotImplementedError()

    def randomly_initialize_replay_buffer(self) -> Tuple[int, int]:
        """
        Initialize the replay buffer with random transitions.
        """

        n_steps = 0
        n_epidodes = 0
        while n_steps < self.seed_steps:
            done = False
            observation = self.env.reset()
            while not done:
                action = self.env.sample_random_action()
                next_observation, reward, done = self.env.step(action)
                self.buffer.append(observation, action, reward, done)
                observation = next_observation
                n_steps += self.action_repeat
            n_epidodes += 1

        self.env.close()
        return n_steps, n_epidodes

    def step(
        self, env, belief, posterior_state, action, observation, explore=False
    ):
        """
        Update belief and action given an observation.
        """
        # print("update_belief_and_act 1")
        embedding = self.encoder(observation).unsqueeze(dim=0)
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # Action and observation need extra time dimension
        # print("update_belief_and_act 2")
        belief, _, _, posterior_state, _ = self.rssm(
            posterior_state,
            action.unsqueeze(dim=0),
            belief,
            embedding,
        )
        # print("update_belief_and_act 3")
        # Remove time dimension from belief/state
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(dim=0)

        # Get action from planner(q(s_t|o≤t,a<t), p)
        # print("update_belief_and_act 4")
        action, _ = self.get_action(belief, posterior_state)

        # print("update_belief_and_act 5")
        if explore:
            # Add gaussian exploration noise on top of the sampled action
            # print(self.action_noise, action)
            action = torch.clamp(Normal(action, self.action_noise).rsample(), -1, 1)
        # print("update_belief_and_act 6")

        # Perform environment step (action repeats handled internally)
        next_observation, reward, done = env.step(
            action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
        )

        # self.replay_buffer.append(observation, action, reward, done)
        return belief, posterior_state, action, next_observation, reward, done

    @typechecked
    def _observation_loss(
            self,
            beliefs: TensorType['seq_len', 'batch_size', 'belief_size'],
            posterior_states: TensorType['seq_len', 'batch_size', 'state_size'],
            observations: Union[TensorType['seq_len', 'batch_size', 'channels', 'height', 'width'],
                                TensorType['seq_len', 'batch_size', 'observation_size']]
    ) -> Tensor:
        """
        Compute the observation loss.
        """
        means = self.observation_model(beliefs, posterior_states)
        # print(observations.shape, "TWWWWRWR")
        if self.pixel_observation:
            observation_dist = Independent(Normal(means, 1), 3)  # independent for matching dims
        else:
            observation_dist = Independent(Normal(means, 1), 1)
        observation_loss = -observation_dist.log_prob(observations).mean()
        return observation_loss

    @typechecked
    def _reward_loss(
            self,
            beliefs: TensorType['seq_len', 'batch_size', 'belief_size'],
            posterior_states: TensorType['seq_len', 'batch_size', 'state_size'],
            rewards: TensorType['seq_len', 'batch_size']
    ) -> Tensor:
        """
        Compute the reward loss
        """
        means = self.reward_model(beliefs, posterior_states)
        reward_dist = Independent(Normal(means, 1), 1)
        reward_loss = -reward_dist.log_prob(rewards.unsqueeze(dim=-1)).mean()
        return reward_loss

    def _kl_loss(
            self,
            posterior_params: Tuple[Tensor, ...],
            prior_params: Tuple[Tensor, ...],
    ) -> Tensor:

        """
        Compute the KL loss.
        """
        prior_dist = _get_dist(prior_params, self.latent_distribution)
        post_dist = _get_dist(posterior_params, self.latent_distribution)

        if self.kl_balance == -1:
            div = kl_divergence(post_dist, prior_dist).sum(dim=2)

            # this is the free bits optimization presented in
            # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
            # https://arxiv.org/abs/1606.04934
            kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))

        else:
            prior_dist_detach = _get_dist(prior_params, self.latent_distribution, detach=True)
            post_dist_detach = _get_dist(posterior_params, self.latent_distribution, detach=True)

            kl_lhs = kl_divergence(post_dist_detach, prior_dist).mean()
            kl_rhs = kl_divergence(post_dist, prior_dist_detach).mean()

            # this is the free bits optimization presented in
            # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
            # https://arxiv.org/abs/1606.04934
            kl_lhs = torch.max(kl_lhs, self.free_nats)
            kl_rhs = torch.max(kl_rhs, self.free_nats)

            # balance
            kl_loss = self.kl_balance * kl_lhs + (1 - self.kl_balance) * kl_rhs

        return kl_loss

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
        for t in range(self.planning_horizon-1):
            _state = prior_states[t]
            actions, action_entropy[t+1] = self.get_action(beliefs[t].detach(), _state.detach())

            # Compute belief (deterministic hidden state)
            hidden = self.rssm.fc_embed_state_action(cat(_state, actions))
            beliefs[t + 1] = self.rssm.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            prior_states[t + 1], prior_params_ = self.rssm.belief_prior(beliefs[t + 1])
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
        # use transition model to predict next step state
        beliefs, _, prior_params, posterior_states, posterior_params = self.rssm(
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
        clip_grad_norm_(self.model_params, self.grad_clip_norm)
        self.model_optimizer.step()

        ####################
        # BEHAVIOUR LEARNING
        ####################

        actor_loss, critic_loss, ac_logs = self.actor_critic_loss(posterior_states, beliefs)
        logs.update(ac_logs)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return logs

    def actor_critic_loss(self, posterior_states, beliefs):
        """
        actor critic loss

        :param posterior_states:
        :param beliefs:
        :return:
        """
        logs = {}
        # Imagine trajectories
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()

        with FreezeParameters(self.model_modules):
            imged_belief, imged_prior_state, _, action_entropy = self.imagine_ahead(
                actor_states, actor_beliefs)

        # Predict Rewards & Values
        with FreezeParameters(self.model_modules + self.critic_modules):
            imged_reward = self.reward_model(imged_belief, imged_prior_state)
            critic = self.critic_target if self.use_target else self.critic
            value_pred = critic(imged_belief, imged_prior_state)
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
            lambda_=self.lambda_,
        )
        if self.gradient_mixing == -1:
            objective = returns
        else:
            raise NotImplementedError("gradient_mixing not yet implemented ")

        # Update Actor weights
        if self.entropy_weight != -1:
            # print(objective.shape, policy_entropy.shape)
            objective = objective + self.entropy_weight * action_entropy.unsqueeze(-1)
        if self.use_discount:
            # discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]),
            # discount_arr[:, 1:]], dim=1)
            discount_arr[:, 0, 0] = 1.0  # at least the first one of each trajectory =1
            discount = torch.cumprod(discount_arr, 0)
            objective = discount * objective
        actor_loss = -objective.mean()

        logs["actor_loss"] = actor_loss.item()
        logs["action_entropy"] = torch.mean(action_entropy).item()

        # detach the input tensor from the transition network
        with torch.no_grad():
            value_beliefs = imged_belief.detach()
            value_prior_states = imged_prior_state.detach()
            target_return = returns.detach()
            if self.use_discount:
                value_discount = discount.detach()

        value_dist = Normal(self.critic(value_beliefs, value_prior_states), 1)
        value_xentropy = value_dist.log_prob(target_return)
        if self.use_discount:
            value_xentropy = value_discount * value_xentropy

        critic_loss = -torch.mean(value_xentropy)
        logs["critic_loss"] = critic_loss.item()

        return actor_loss, critic_loss, logs

    def update_critic(self) -> None:
        """
        Update the critic weights using polyack averaging (soft update : see DDPG)
        """
        polyak_update(self.critic_target, self.critic, self.polyak_avg)

    def get_action(self, belief: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get action.
        """

        action_mean, action_std = self.actor(belief, state)
        dist = D.Normal(action_mean, action_std)
        dist = D.Independent(dist, 1)
        return dist.rsample(), dist.entropy()


@typechecked
def lambda_return(
        reward: TensorType['seq_len', 'batch_size', 1],
        value: TensorType['seq_len', 'batch_size', 1],
        bootstrap: TensorType['batch_size', 1],
        discount=0.99,
        lambda_=0.95
) -> TensorType['seq_len', 'batch_size', 1]:
    """
    Compute the lambda-return for a given trajectory.

    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
    """

    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(reward)  # pcont
    inputs = reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []

    for index in indices:
        inp = inputs[index]
        disc = discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)

    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs

    return returns
