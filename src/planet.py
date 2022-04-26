import os

from typing import Any, Dict, List, Tuple, Union
from collections import namedtuple


import torch
from torch import Tensor, optim, nn
from torch.distributions import Normal, kl_divergence, Independent
# from torch.nn import functional as F

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from env import EnvBatcher
from memory import ExperienceReplay
from models import TransitionModel, ObservationModel, DenseModel, CnnImageEncoder
from utils import device
from planner import MPCPlanner
from base_agent import BaseAgent

patch_typeguard()

DiscreteState = namedtuple('DiscreteState', [])

class Planet(BaseAgent):
    """
    A planet-based agent.
    """

    def __init__(self, params: Dict[str, Any], env):
        self.env = env

        print(f'Environement min {self.env._env.action_space.low}')
        print(f'Environement max {self.env._env.action_space.high}')

        # Initialize base parameters from the config file.

        self.belief_size = params["belief_size"]

        self.state_size = params["state_size"]
        self.action_size = env.action_size
        self.hidden_size = params["hidden_size"]
        self.embedding_size = params["embedding_size"]

        self.dense_activation_function = params["dense_activation_function"]
        self.cnn_activation_function = params["cnn_activation_function"]

        self.batch_size = params["batch_size"]
        self.seq_len = params["seq_len"]

        self.seed_episodes = params["seed_episodes"]
        self.seed_steps = params["seed_steps"]

        self.experience_size = params["experience_size"]
        self.bit_depth = params["bit_depth"]
        self.kl_loss_weight = params['kl_loss_weight']
        self.latent_distribution = params['latent_distribution']
        self.discrete_latent_dimensions = params['discrete_latent_dimensions']
        self.discrete_latent_classes = params['discrete_latent_classes']
        if self.latent_distribution == 'Categorical':
            self.state_size = self.discrete_latent_dimensions * self.discrete_latent_classes

        self.jit = params['jit']

        self.planning_horizon = params["planning_horizon"]

        self.pixel_observation = params['pixel_observation']

        """
                Initialize MPC parameters.
        """
        self.optimisation_iters = params["MPC"]["optimisation_iters"]
        self.candidates = params["MPC"]["candidates"]
        self.top_candidates = params["MPC"]["top_candidates"]

        self.model_learning_rate = params["model_learning_rate"]
        self.adam_epsilon = params["adam_epsilon"]
        self.weight_decay = params['weight_decay']
        self.action_noise = params["action_noise"]
        self.grad_clip_norm = params["grad_clip_norm"]
        self.action_repeat = params["action_repeat"]

        self.posterior_states = None
        self.beliefs = None

        """
        Initialize the replay buffer and models.
        """

        self.buffer = ExperienceReplay(
            self.experience_size,
            self.env.action_size,
            self.bit_depth,
            self.pixel_observation,
            self.env.observation_size,
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
            print("Loading models...")
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
        s = 0
        while total_steps < self.seed_steps:
        # for s in range(1, self.seed_episodes + 1):
            done = False
            t = 0
            observation = self.env.reset()
            # print(f'observation.shape={observation.shape}')
            while not done:
                action = self.env.sample_random_action()
                next_observation, reward, done = self.env.step(action)
                self.buffer.append(observation, action, reward, done)
                observation = next_observation
                t += 1
            s += 1

            total_steps += t * self.action_repeat
        self.env.close()
        return total_steps, s

    def initialize_models(self) -> None:
        """
        Initialize the different models.
        """
        self.transition_model = TransitionModel(
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
        ).to(device=device)

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
                self.dense_activation_function
            )
            self.observation_model = DenseModel(
                self.belief_size + self.state_size,
                self.hidden_size,
                self.env.observation_size,
                self.dense_activation_function
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

        if self.jit:
            self.transition_model = torch.jit.script(self.transition_model)
            self.observation_model = torch.jit.script(self.observation_model)
            self.reward_model = torch.jit.script(self.reward_model)
            self.encoder = torch.jit.script(self.encoder)
            self.planner = torch.jit.script(self.planner)

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
            self.model_params,
            lr=self.model_learning_rate,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay
        )

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
        # extract Gaussian params
        posterior_means, posterior_std_devs = posterior_params
        prior_means, prior_std_devs = prior_params

        div = kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means, prior_std_devs),
        ).sum(dim=2)

        # this is the free bits optimization presented in
        # Improved Variational Inference with Inverse Autoregressive Flow : C.8.
        # https://arxiv.org/abs/1606.04934
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))

        return kl_loss

    def train_step(self) -> Dict[str, float]:
        """
        Train the model for one step.
        """

        log = {}
        ####################
        # DYNAMICS LEARNING
        ####################

        # 1) Draw sequences chunks
        observations, actions, rewards, nonterminals = self.buffer.sample(
            self.batch_size, self.seq_len
        )

        # 2) Compute model States
        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(self.batch_size, self.belief_size, device=device)
        init_state = torch.zeros(self.batch_size, self.state_size, device=device)

        # Update belief/state using posterior from previous belief/state,
        # previous action and current observation (over entire sequence at once)
        embeddings = self.encoder(observations[1:])
        (
            beliefs,
            _,  # prior_states, just in case
            prior_params,
            posterior_states,
            posterior_params,
        ) = self.transition_model(
            init_state,
            actions[:-1],
            init_belief,
            embeddings,
            nonterminals[:-1],
        )

        # 3) Update model weights

        # Calculate observation likelihood, reward likelihood and KL losses
        # sum over final dims, average over batch and time
        observation_loss = self._observation_loss(beliefs, posterior_states, observations[1:])
        reward_loss = self._reward_loss(beliefs, posterior_states, rewards[:-1])
        kl_loss = self._kl_loss(posterior_params, prior_params)
        model_loss = observation_loss + reward_loss + kl_loss * self.kl_loss_weight

        # log stuff
        log["observation_loss"] = observation_loss.item()
        log["reward_loss"] = reward_loss.item()
        log["kl_loss"] = kl_loss.item()
        log["model_loss"] = model_loss.item()

        # Update model parameters
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()

        return log

    def update_belief_and_act(
        self, env, belief, posterior_state, action, observation, explore=False
    ):
        """
        Update belief and action given an observation.
        """
        embedding = self.encoder(observation).unsqueeze(dim=0)
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # Action and observation need extra time dimension
        belief, _, _, posterior_state, _ = self.transition_model(
            posterior_state,
            action.unsqueeze(dim=0),
            belief,
            embedding,
        )
        # Remove time dimension from belief/state
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(dim=0)

        # Get action from planner(q(s_t|o≤t,a<t), p)
        action, _ = self.get_action(belief, posterior_state)

        if explore:
            # Add gaussian exploration noise on top of the sampled action
            # print(self.action_noise, action)
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
