import os

import torch
from torch import optim, nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

import utils
from env import EnvBatcher
from memory import ExperienceReplay
from models import TransitionModel, ObservationModel, RewardModel, Encoder, bottle
from planner import MPCPlanner
from base_agent import BaseAgent


class Planet(BaseAgent):
    def __init__(self, params, env):
        self.env = env
        self.belief_size = params["belief_size"]
        self.state_size = params["state_size"]
        self.action_size = env.action_size
        self.hidden_size = params["hidden_size"]
        self.embedding_size = params["embedding_size"]
        self.dense_activation_function = params["dense_activation_function"]

        self.symbolic_env = params["symbolic_env"]
        self.observation_size = env.observation_size
        self.embedding_size = params["embedding_size"]
        self.cnn_activation_function = params["cnn_activation_function"]

        self.batch_size = params["batch_size"]
        self.chunk_size = params["chunk_size"]

        self.seed_episodes = params["seed_episodes"]

        self.overshooting_distance = params["overshooting_distance"]

        self.learning_rate_schedule = params["learning_rate_schedule"]

        self.worldmodel_LogProbLoss = params["worldmodel_LogProbLoss"]

        self.experience_size = params["experience_size"]
        self.bit_depth = params["bit_depth"]

        # for the MPC planner
        self.planning_horizon = params["planning_horizon"]
        self.optimisation_iters = params["optimisation_iters"]
        self.candidates = params["candidates"]
        self.top_candidates = params["top_candidates"]

        self.model_learning_rate = params["model_learning_rate"]
        self.adam_epsilon = params["adam_epsilon"]
        self.global_kl_beta = params["global_kl_beta"]
        self.overshooting_kl_beta = params["overshooting_kl_beta"]
        self.overshooting_reward_scale = params["overshooting_reward_scale"]
        self.action_noise = params["action_noise"]
        self.grad_clip_norm = params["grad_clip_norm"]
        self.action_repeat = params["action_repeat"]

        self.replay_buffer = ExperienceReplay(
            self.experience_size,
            self.symbolic_env,
            env.observation_size,
            env.action_size,
            self.bit_depth,
            utils.device,
        )

        # initialize the different models
        self.initialize_models()

        # optimization priors
        # Global prior N(0, I)
        self.global_prior = Normal(
            torch.zeros(
                params["batch_size"], params["state_size"], device=utils.device
            ),
            torch.ones(params["batch_size"], params["state_size"], device=utils.device),
        )
        # Allowed deviation in KL divergence
        self.free_nats = torch.full(
            (1,), params["free_nats"], device=utils.device
        )  # Allowed deviation in KL divergence

        # initialize the optimizers
        self.model_optimizer, self.actor_optimizer, self.value_optimizer = (
            None,
            None,
            None,
        )
        self.initialize_optimizers()

        # load the models if possible
        if params["models"] is not "" and os.path.exists(params["models"]):
            model_dicts = torch.load(params["models"])
            self.transition_model.load_state_dict(model_dicts["transition_model"])
            self.observation_model.load_state_dict(model_dicts["observation_model"])
            self.reward_model.load_state_dict(model_dicts["reward_model"])
            self.encoder.load_state_dict(model_dicts["encoder"])
            self.model_optimizer.load_state_dict(model_dicts["model_optimizer"])

    def eval(self):
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()

    def train(self):
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()

    def randomly_initialize_replay_buffer(self):
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

    def initialize_models(self):
        self.transition_model = TransitionModel(
            self.belief_size,
            self.state_size,
            self.action_size,
            self.hidden_size,
            self.embedding_size,
            self.dense_activation_function,
        ).to(device=utils.device)

        self.observation_model = ObservationModel(
            self.symbolic_env,
            self.observation_size,
            self.belief_size,
            self.state_size,
            self.embedding_size,
            self.cnn_activation_function,
        ).to(device=utils.device)

        self.reward_model = RewardModel(
            self.belief_size,
            self.state_size,
            self.hidden_size,
            self.dense_activation_function,
        ).to(device=utils.device)

        self.encoder = Encoder(
            self.symbolic_env,
            self.observation_size,
            self.embedding_size,
            self.cnn_activation_function,
        ).to(device=utils.device)

        self.planner = MPCPlanner(
            self.action_size,
            self.planning_horizon,
            self.optimisation_iters,
            self.candidates,
            self.top_candidates,
            self.transition_model,
            self.reward_model,
        )

    def initialize_optimizers(self):
        self.model_modules = (
            self.transition_model.modules
            + self.observation_model.modules
            + self.reward_model.modules
            + self.encoder.modules
        )

        self.model_params = (
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.encoder.parameters())
        )
        self.model_optimizer = optim.Adam(
            self.model_params, lr=self.model_learning_rate, eps=self.adam_epsilon
        )

    def compute_observation_loss(self, beliefs, posterior_states, observations):
        if self.worldmodel_LogProbLoss:
            observation_dist = Normal(
                bottle(self.observation_model, (beliefs, posterior_states)), 1
            )
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if self.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(
                    bottle(self.observation_model, (beliefs, posterior_states)),
                    observations[1:],
                    reduction="none",
                )
                .sum(dim=2 if self.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
            # print(observation_loss)
        return observation_loss

    def compute_reward_loss(self, beliefs, posterior_states, rewards):
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

    def compute_kl_loss(
        self, posterior_means, posterior_std_devs, prior_means, prior_std_devs
    ):
        div = kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means, prior_std_devs),
        ).sum(dim=2)
        # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
        if self.global_kl_beta != 0:
            kl_loss += self.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), self.global_prior
            ).sum(dim=2).mean(dim=(0, 1))
        return kl_loss

    def compute_overshooting_losses(
        self,
        kl_loss,
        reward_loss,
        actions,
        nonterminals,
        rewards,
        beliefs,
        prior_states,
        posterior_means,
        posterior_std_devs,
    ):
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.chunk_size - 1):
            d = min(
                t + self.overshooting_distance, self.chunk_size - 1
            )  # Overshooting distance
            t_, d_ = (
                t - 1,
                d - 1,
            )  # Use t_ and d_ to deal with different time indexing for latent states
            # Calculate sequence padding so overshooting terms can be calculated in one batch
            seq_pad = (0, 0, 0, 0, 0, t - d + self.overshooting_distance)
            # Store
            # (0) actions,
            # (1) nonterminals,
            # (2) rewards,
            # (3) beliefs,
            # (4) prior states,
            # (5) posterior means,
            # (6) posterior standard deviations and
            # (7) sequence masks
            # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars.append(
                (
                    F.pad(actions[t:d], seq_pad),
                    F.pad(nonterminals[t:d], seq_pad),
                    F.pad(rewards[t:d], seq_pad[2:]),
                    beliefs[t_],
                    prior_states[t_],
                    F.pad(posterior_means[t_ + 1 : d_ + 1].detach(), seq_pad),
                    F.pad(
                        posterior_std_devs[t_ + 1 : d_ + 1].detach(), seq_pad, value=1
                    ),
                    F.pad(
                        torch.ones(
                            d - t, self.batch_size, self.state_size, device=utils.device
                        ),
                        seq_pad,
                    ),
                )
            )
        overshooting_vars = tuple(zip(*overshooting_vars))

        # Update belief/state using prior from previous belief/state
        # and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(
            torch.cat(overshooting_vars[4], dim=0),
            torch.cat(overshooting_vars[0], dim=1),
            torch.cat(overshooting_vars[3], dim=0),
            None,
            torch.cat(overshooting_vars[1], dim=1),
        )
        seq_mask = torch.cat(overshooting_vars[7], dim=1)

        # Calculate overshooting KL loss with sequence mask
        # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        p = Normal(
            torch.cat(overshooting_vars[5], dim=1),
            torch.cat(overshooting_vars[6], dim=1),
        )
        q = Normal(prior_means, prior_std_devs)
        kl_loss = (
            (1 / self.overshooting_distance)
            * self.overshooting_kl_beta
            * torch.max(
                (kl_divergence(p, q) * seq_mask).sum(dim=2), self.free_nats
            ).mean(dim=(0, 1))
            * (self.chunk_size - 1)
        )
        # Calculate overshooting reward prediction loss with sequence mask
        # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        if self.overshooting_reward_scale != 0:
            reward_overshoothing_weight = (
                1 / self.overshooting_distance
            ) * self.overshooting_reward_scale
            reward_loss = (
                reward_overshoothing_weight
                * F.mse_loss(
                    bottle(self.reward_model, (beliefs, prior_states))
                    * seq_mask[:, :, 0],
                    torch.cat(overshooting_vars[2], dim=1),
                    reduction="none",
                ).mean(dim=(0, 1))
                * (self.chunk_size - 1)
            )
        return (kl_loss, reward_loss), (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
        )

    def train_step(self) -> dict:
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
        init_belief = torch.zeros(
            self.batch_size, self.belief_size, device=utils.device
        )
        init_state = torch.zeros(self.batch_size, self.state_size, device=utils.device)

        # Update belief/state using posterior from previous belief/state, previous action and current observation
        # (over entire sequence at once)
        (
            beliefs,
            prior_states,
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

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting);
        # sum over final dims, average over batch and time
        # (original implementation, though paper seems to miss 1/T scaling?)
        observation_loss = self.compute_observation_loss(
            beliefs, posterior_states, observations
        )
        log["observation_loss"] = observation_loss.item()

        reward_loss = self.compute_reward_loss(beliefs, posterior_states, rewards)
        log["reward_loss"] = reward_loss.item()

        # transition loss
        kl_loss = self.compute_kl_loss(
            posterior_means, posterior_std_devs, prior_means, prior_std_devs
        )
        log["kl_loss"] = kl_loss.item()

        # Calculate latent overshooting objective for t > 0
        if self.overshooting_kl_beta != 0:
            (kl_over, reward_over), state_updates = self.compute_overshooting_losses(
                actions,
                nonterminals,
                rewards,
                beliefs,
                prior_states,
                posterior_means,
                posterior_std_devs,
            )
            kl_loss += kl_over
            reward_loss += reward_over
            beliefs, prior_states, prior_means, prior_std_devs = state_updates

            # log overshoot losses
            log["kl_loss_overshoot"] = kl_over.item()
            log["reward_loss_overshoot"] = reward_over.item()

        model_loss = observation_loss + reward_loss + kl_loss
        log["model_loss"] = model_loss.item()

        # TODO : add a learning rate schedule
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
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # print("action size: ",action.size()) torch.Size([1, 6])
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            posterior_state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0),
        )
        # Remove time dimension from belief/state
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = self.planner.get_action(
            belief, posterior_state, deterministic=not explore
        )
        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(Normal(action, self.action_noise).rsample(), -1, 1)
            # action = action + self.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        # Perform environment step (action repeats handled internally)
        next_observation, reward, done = env.step(
            action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
        )

        # self.replay_buffer.append(observation, action, reward, done)
        return belief, posterior_state, action, next_observation, reward, done
