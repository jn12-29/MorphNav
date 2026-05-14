from __future__ import annotations

from typing import cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import BaseModel
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from components.path_integration import PlaceCellTargetEncoder, PathIntegrationHead, PathIntegrationOutputs


class PathIntegrationRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """Recurrent actor-critic policy with a parallel path-integration auxiliary head."""

    def __init__(self, *args, **kwargs):
        self.pi_dropout_rate = kwargs.pop("pi_dropout_rate", kwargs.pop("dropout_rate", 0.5))
        self.pi_bottleneck_dim = kwargs.pop("pi_bottleneck_dim", 256)
        self.pi_n_place_cells = kwargs.pop("pi_n_place_cells", kwargs.pop("n_pc", 256))
        self.pi_place_cell_scale = kwargs.pop("pi_place_cell_scale", 0.01)
        self.pi_pos_min = kwargs.pop("pi_pos_min", -2.5)
        self.pi_pos_max = kwargs.pop("pi_pos_max", 2.5)
        self.pi_neurons_seed = kwargs.pop("pi_neurons_seed", 8341)
        self.pi_bottleneck_has_bias = kwargs.pop("pi_bottleneck_has_bias", False)
        super().__init__(*args, **kwargs)

        self.path_integration_head = PathIntegrationHead(
            lstm_output_dim=self.lstm_output_dim,
            bottleneck_dim=self.pi_bottleneck_dim,
            n_place_cells=self.pi_n_place_cells,
            dropout_rate=self.pi_dropout_rate,
            bottleneck_has_bias=self.pi_bottleneck_has_bias,
        )
        self.path_integration_target_encoder = PlaceCellTargetEncoder(
            n_cells=self.pi_n_place_cells,
            stdev=self.pi_place_cell_scale,
            pos_min=self.pi_pos_min,
            pos_max=self.pi_pos_max,
            seed=self.pi_neurons_seed,
        )
        self._ensure_path_integration_head_in_optimizer()

    def _ensure_path_integration_head_in_optimizer(self) -> None:
        head_params = [param for param in self.path_integration_head.parameters() if param.requires_grad]
        if not head_params:
            return
        optimizer_param_ids = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
        missing = [param for param in head_params if id(param) not in optimizer_param_ids]
        if missing:
            self.optimizer.add_param_group({"params": missing})

    def _extract_actor_features(self, obs) -> th.Tensor:
        return BaseModel.extract_features(self, obs, self.pi_features_extractor)

    def _forward_actor_lstm(
        self,
        obs,
        lstm_states_pi: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        pi_features = self._extract_actor_features(obs)
        latent_pi, new_states = self._process_sequence(
            pi_features,
            lstm_states_pi,
            episode_starts,
            self.lstm_actor,
        )
        return latent_pi, cast(tuple[th.Tensor, th.Tensor], new_states)

    def forward_pi(
        self,
        obs,
        lstm_states_pi: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[PathIntegrationOutputs, tuple[th.Tensor, th.Tensor]]:
        latent_pi, lstm_states_pi = self._forward_actor_lstm(obs, lstm_states_pi, episode_starts)
        return self.path_integration_head(latent_pi), lstm_states_pi

    def evaluate_actions_with_pi(
        self,
        obs,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
        )

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features,
                lstm_states.vf,
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        pi_outputs = self.path_integration_head(latent_pi)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return values, log_prob, distribution.entropy(), pi_outputs

    def get_distribution_with_pi(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...], PathIntegrationOutputs]:
        latent_pi, lstm_states = self._forward_actor_lstm(
            obs,
            lstm_states,
            episode_starts,
        )
        pi_outputs = self.path_integration_head(latent_pi)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states, pi_outputs

    def _predict_with_pi(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
        distribution, lstm_states, pi_outputs = self.get_distribution_with_pi(
            observation,
            lstm_states,
            episode_starts,
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states, pi_outputs

    def predict_with_pi(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ):
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states, pi_outputs = self._predict_with_pi(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())
            pc_logits = pi_outputs.pc_logits.cpu().numpy()
            bottleneck = pi_outputs.bottleneck.cpu().numpy()

        actions = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)
            pc_logits = pc_logits.squeeze(axis=0)
            bottleneck = bottleneck.squeeze(axis=0)

        return actions, states, {"pc_logits": pc_logits, "bottleneck": bottleneck}
