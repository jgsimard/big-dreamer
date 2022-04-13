import os

from typing import Any, Dict, List

import torch
from torch import Tensor, optim, nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from env import EnvBatcher
from memory import ExperienceReplay
from models import (
    TransitionModel,
    ObservationModel,
    RewardModel,
    CnnImageEncoder,
    bottle,
)
from utils import device
from planner import MPCPlanner
from base_agent import BaseAgent


class Planet(BaseAgent):
    """
    A planet-based agent.
    """

    def __init__(self, params: Dict[str, Any], env):
        self.env = env

        """
        Initialize base parameters from the config file.
        """

        self.belief_size = params["belief_size"]
        self.state_size = params["state_size"]
        self.action_size = env.action_size
        self.hidden_size = params["hidden_size"]
        self.embedding_size = params["embedding_size"]
        self.dense_activation_function = params["dense_activation_function"]

        self.observation_size = env.observation_size
        self.embedding_size = params["embedding_size"]
        self.cnn_activation_function = params["cnn_activation_function"]

        self.batch_size = params["batch_size"]
        self.chunk_size = params["chunk_size"]

        self.seed_episodes = params["seed_episodes"]

        self.learning_rate_schedule = params["learning_rate_schedule"]

        self.worldmodel_LogProbLoss = params["worldmodel_LogProbLoss"]

        self.experience_size = params["experience_size"]
        self.bit_depth = params["bit_depth"]
        self.kl_loss_weight = params['kl_loss_weight']

        """
        Initialize MPC parameters.
        """

        self.planning_horizon = params["planning_horizon"]
        self.optimisation_iters = params["optimisation_iters"]
        self.candidates = params["candidates"]
        self.top_candidates = params["top_candidates"]

        self.model_learning_rate = params["model_learning_rate"]
        self.adam_epsilon = params["adam_epsilon"]
        self.global_kl_beta = params["global_kl_beta"]
        self.action_noise = params["action_noise"]
        self.grad_clip_norm = params["grad_clip_norm"]
        self.action_repeat = params["action_repeat"]

        self.posterior_states = None
        self.beliefs = None

        """
        Initialize the replay buffer and models.
        """

        self.replay_buffer = ExperienceReplay(
            self.experience_size,
            env.action_size,
            self.bit_depth,
            device,
        )

        self.initialize_models()

        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1,), params["free_nats"], device=device)

        self.initialize_optimizers()
        self.load(params)

    def load(self, params: Dict[str, Any]) -> None:
        """
        Load models if they exist.
        """
        if params["models"] != "" and os.path.exists(params["models"]):
            model_dicts = torch.load(params["models"])
            self.transition_model.load_state_dict(model_dicts["transition_model"])
            self.observation_model.load_state_dict(model_dicts["observation_model"])
            self.reward_model.load_state_dict(model_dicts["reward_model"])
            self.encoder.load_state_dict(model_dicts["encoder"])
            self.model_optimizer.load_state_dict(model_dicts["model_optimizer"])

    def eval(self) -> None:
        """
        Set the models to evaluation mode.
        """

        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()

    def train(self) -> None:
        """
        Set the models to training mode.
        """

        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()

    def randomly_initialize_replay_buffer(self) -> List[int]:
        """
        Initialize the replay buffer with random transitions.
        """

        total_steps = 0
        for s in range(1, self.seed_episodes + 1):
            done = False
            t = 0
            observation = self.env.reset()
            while not done:
                action = self.env.sample_random_action()
                next_observation, reward, done = self.env.step(action)
                self.replay_buffer.append(observation, action, reward, done)
                observation = next_observation
                t += 1

            total_steps += t * self.action_repeat
        self.env.close()
        return total_steps, s

    def initialize_models(self) -> None:
        """
        Initialize the different models.
        """

        self.transition_model = torch.jit.script(
            TransitionModel(
                self.belief_size,
                self.state_size,
                self.action_size,
                self.hidden_size,
                self.embedding_size,
                self.dense_activation_function,
            ).to(device=device)
        )

        self.observation_model = torch.jit.script(
            ObservationModel(
                self.belief_size,
                self.state_size,
                self.embedding_size,
                self.cnn_activation_function,
            ).to(device=device)
        )

        self.reward_model = torch.jit.script(
            RewardModel(
                self.belief_size,
                self.state_size,
                self.hidden_size,
                self.dense_activation_function,
            ).to(device=device)
        )

        self.encoder = torch.jit.script(
            CnnImageEncoder(
                self.embedding_size,
                self.cnn_activation_function,
            ).to(device=device)
        )

        self.planner = MPCPlanner(
            self.action_size,
            self.planning_horizon,
            self.optimisation_iters,
            self.candidates,
            self.top_candidates,
            self.transition_model,
            self.reward_model,
        )

    def initialize_optimizers(self) -> None:
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
        self.model_optimizer = optim.Adam(
            self.model_params, lr=self.model_learning_rate, eps=self.adam_epsilon
        )

    def observation_loss(
        self, beliefs: Tensor, posterior_states: Tensor, observations: Tensor
    ) -> Tensor:
        """
        Compute the observation loss.
        Args:
            beliefs:            (L-1, B, S)
            posterior_states:   (L-1, B, S)
            observations:       (L, B, C, H, W)
        Returns:
            reward_loss: Tensor
        """

        if self.worldmodel_LogProbLoss:
            observation_dist = Normal(
                bottle(self.observation_model, (beliefs, posterior_states)), 1
            )
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=(2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(
                    bottle(self.observation_model, (beliefs, posterior_states)),
                    observations[1:],
                    reduction="none",
                )
                .sum(dim=(2, 3, 4))
                .mean(dim=(0, 1))
            )
        return observation_loss

    def reward_loss(
        self, beliefs: Tensor, posterior_states: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        Compute the reward loss. L=Chunk size, B=Batch Size

        Args:
            beliefs: Tensor (L-1, B, )
            posterior_states: Tensor
            rewards: Tensor
        Returns:
            reward_loss: Tensor
        """

        if self.worldmodel_LogProbLoss:
            reward_dist = Normal(
                bottle(self.reward_model, (beliefs, posterior_states)), 1
            )
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(
                bottle(self.reward_model, (beliefs, posterior_states)),
                rewards[:-1],
                reduction="none",
            ).mean(dim=(0, 1))
        return reward_loss

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

        div = kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means, prior_std_devs),
        ).sum(dim=2)

        # this is the free bits optimization presented in
        # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
        # https://arxiv.org/abs/1606.04934
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))

        return kl_loss

    def train_step(self) -> dict:
        """
        Train the model for one step.
        """

        log = {}
        ####################
        # DYNAMICS LEARNING
        ####################

        # 1) Draw Sequences

        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly
        # at random from the dataset (including terminal flags)
        # Transitions start at time t = 0
        observations, actions, rewards, nonterminals = self.replay_buffer.sample(
            self.batch_size, self.chunk_size
        )

        # 2) Compute model States
        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(self.batch_size, self.belief_size, device=device)
        init_state = torch.zeros(self.batch_size, self.state_size, device=device)

        # Update belief/state using posterior from previous belief/state,
        # previous action and current observation (over entire sequence at once)
        (
            beliefs,
            _, # prior_states, just in case
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = self.transition_model(
            init_state,
            actions[:-1],
            init_belief,
            bottle(self.encoder, (observations[1:],)),
            nonterminals[:-1],
        )

        # 3) Update model weights

        # Calculate observation likelihood, reward likelihood and KL losses
        # sum over final dims, average over batch and time
        observation_loss = self.observation_loss(
            beliefs, posterior_states, observations
        )
        reward_loss = self.reward_loss(beliefs, posterior_states, rewards)
        kl_loss = self.kl_loss(
            posterior_means, posterior_std_devs, prior_means, prior_std_devs
        )
        model_loss = observation_loss + reward_loss + kl_loss * self.kl_loss_weight

        log["observation_loss"] = observation_loss.item()
        log["reward_loss"] = reward_loss.item()
        log["kl_loss"] = kl_loss.item()
        log["model_loss"] = model_loss.item()

        # TODO: add a learning rate schedule
        # Update model parameters
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()

        # for dreamer, ugly for the moment
        self.posterior_states = posterior_states
        self.beliefs = beliefs

        return log

    def update_belief_and_act(
        self, env, belief, posterior_state, action, observation, explore=False
    ):
        """
        Update belief and action given an observation.
        """

        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            posterior_state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0),
        )
        # Remove time dimension from belief/state
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(dim=0)

        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = self.get_action(belief, posterior_state)

        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(Normal(action, self.action_noise).rsample(), -1, 1)

        # Perform environment step (action repeats handled internally)
        next_observation, reward, done = env.step(
            action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
        )

        # self.replay_buffer.append(observation, action, reward, done)
        return belief, posterior_state, action, next_observation, reward, done

    def get_action(self, belief, state):
        """
        Get action for the given belief and posterior state.
        """

        return self.planner(belief, state)
