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
        self.pi_init_state_key = kwargs.pop("pi_init_state_key", None)
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
        self.path_integration_state_init: th.nn.Linear | None = None
        self.path_integration_cell_init: th.nn.Linear | None = None
        if self.pi_init_state_key is not None:
            if not isinstance(self.observation_space, spaces.Dict) or self.pi_init_state_key not in self.observation_space.spaces:
                raise ValueError(f"pi_init_state_key {self.pi_init_state_key!r} is missing from observation_space")
            init_space = self.observation_space.spaces[self.pi_init_state_key]
            if len(init_space.shape) == 0 or init_space.shape[-1] < 2:
                raise ValueError(f"pi_init_state_key {self.pi_init_state_key!r} must expose at least 2 coordinates")
            init_state_dim = self.lstm_actor.num_layers * self.lstm_output_dim
            self.path_integration_state_init = th.nn.Linear(self.pi_n_place_cells, init_state_dim)
            self.path_integration_cell_init = th.nn.Linear(self.pi_n_place_cells, init_state_dim)
        self._ensure_path_integration_modules_in_optimizer()

    def _path_integration_trainable_modules(self) -> list[th.nn.Module | None]:
        return [
            self.path_integration_head,
            self.path_integration_state_init,
            self.path_integration_cell_init,
        ]

    def _ensure_path_integration_modules_in_optimizer(self) -> None:
        pi_params = [
            param
            for module in self._path_integration_trainable_modules()
            if module is not None
            for param in module.parameters()
            if param.requires_grad
        ]
        if not pi_params:
            return
        optimizer_param_ids = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
        missing = [param for param in pi_params if id(param) not in optimizer_param_ids]
        if missing:
            self.optimizer.add_param_group({"params": missing})

    def _extract_actor_features(self, obs) -> th.Tensor:
        return BaseModel.extract_features(self, obs, self.pi_features_extractor)

    def _extract_critic_features(self, obs) -> th.Tensor:
        return BaseModel.extract_features(self, obs, self.vf_features_extractor)

    def _initial_lstm_states_from_obs(self, obs) -> tuple[th.Tensor, th.Tensor] | None:
        if self.pi_init_state_key is None:
            return None
        if self.path_integration_state_init is None or self.path_integration_cell_init is None:
            return None
        if not isinstance(obs, dict) or self.pi_init_state_key not in obs:
            raise KeyError(f"PI initial-state key {self.pi_init_state_key!r} is missing from observations")

        init_pos = obs[self.pi_init_state_key].float()
        init_code = self.path_integration_target_encoder(init_pos).to(dtype=self.path_integration_state_init.weight.dtype)
        batch_size = init_code.shape[0]
        n_layers = self.lstm_actor.num_layers
        h0 = self.path_integration_state_init(init_code).view(batch_size, n_layers, self.lstm_output_dim)
        c0 = self.path_integration_cell_init(init_code).view(batch_size, n_layers, self.lstm_output_dim)
        return h0.swapaxes(0, 1).contiguous(), c0.swapaxes(0, 1).contiguous()

    def _process_sequence_with_initial_states(
        self,
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: th.nn.LSTM,
        initial_lstm_states: tuple[th.Tensor, th.Tensor] | None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        if initial_lstm_states is None:
            return self._process_sequence(features, lstm_states, episode_starts, lstm)

        n_seq = lstm_states[0].shape[1]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts_sequence = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if th.all(episode_starts_sequence == 0.0):
            return self._process_sequence(features, lstm_states, episode_starts, lstm)

        n_layers, _n_seq, hidden_size = lstm_states[0].shape
        flat_batch_size = features.shape[0]
        max_len = flat_batch_size // n_seq
        expected_shape = (n_layers, flat_batch_size, hidden_size)
        if initial_lstm_states[0].shape != expected_shape or initial_lstm_states[1].shape != expected_shape:
            raise ValueError(
                f"initial_lstm_states shapes must be {expected_shape}, got "
                f"{tuple(initial_lstm_states[0].shape)} and {tuple(initial_lstm_states[1].shape)}"
            )

        init_h_sequence = initial_lstm_states[0].reshape(n_layers, n_seq, max_len, hidden_size).permute(2, 0, 1, 3)
        init_c_sequence = initial_lstm_states[1].reshape(n_layers, n_seq, max_len, hidden_size).permute(2, 0, 1, 3)

        lstm_output = []
        for features_step, episode_start, init_h, init_c in zip(
            features_sequence,
            episode_starts_sequence,
            init_h_sequence,
            init_c_sequence,
            strict=True,
        ):
            reset_mask = episode_start.view(1, n_seq, 1)
            hidden, lstm_states = lstm(
                features_step.unsqueeze(dim=0),
                (
                    (1.0 - reset_mask) * lstm_states[0] + reset_mask * init_h,
                    (1.0 - reset_mask) * lstm_states[1] + reset_mask * init_c,
                ),
            )
            lstm_output.append(hidden)

        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, cast(tuple[th.Tensor, th.Tensor], lstm_states)

    def _forward_actor_lstm(
        self,
        obs,
        lstm_states_pi: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        pi_features = self._extract_actor_features(obs)
        latent_pi, new_states = self._process_sequence_with_initial_states(
            pi_features,
            lstm_states_pi,
            episode_starts,
            self.lstm_actor,
            self._initial_lstm_states_from_obs(obs),
        )
        return latent_pi, cast(tuple[th.Tensor, th.Tensor], new_states)

    def forward(
        self,
        obs,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        initial_lstm_states = self._initial_lstm_states_from_obs(obs)
        latent_pi, lstm_states_pi = self._process_sequence_with_initial_states(
            pi_features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
            initial_lstm_states,
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence_with_initial_states(
                vf_features,
                lstm_states.vf,
                episode_starts,
                self.lstm_critic,
                initial_lstm_states,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

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

        initial_lstm_states = self._initial_lstm_states_from_obs(obs)
        latent_pi, _ = self._process_sequence_with_initial_states(
            pi_features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
            initial_lstm_states,
        )

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence_with_initial_states(
                vf_features,
                lstm_states.vf,
                episode_starts,
                self.lstm_critic,
                initial_lstm_states,
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

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
        latent_pi, lstm_states = self._forward_actor_lstm(obs, lstm_states, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        features = self._extract_critic_features(obs)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence_with_initial_states(
                features,
                lstm_states,
                episode_starts,
                self.lstm_critic,
                self._initial_lstm_states_from_obs(obs),
            )
        elif self.shared_lstm:
            latent_pi, _ = self._forward_actor_lstm(obs, lstm_states, episode_starts)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

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
