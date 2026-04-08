import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class AuxPositionHead(nn.Module):
    """辅助定位头，输出位置预测"""

    def __init__(self, lstm_output_dim: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(lstm_output_dim, 2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(self.dropout(x))


class AuxRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """带辅助定位头的 Recurrent 策略"""

    def __init__(self, *args, **kwargs):
        self.dropout_rate = kwargs.pop("dropout_rate", 0.5)
        super().__init__(*args, **kwargs)

        # 辅助定位头：
        # - 训练时在 evaluate_actions_with_aux() 中用于辅助损失
        # - 推理/分析时在 predict_with_aux() 中输出位置预测
        self.aux_position_head = AuxPositionHead(
            self.lstm_output_dim, dropout_rate=self.dropout_rate
        )

    def evaluate_actions_with_aux(
        self,
        obs,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ):
        """
        训练用：
        基于 evaluate_actions() 额外返回 pred_pos，供 AuxRecurrentPPO.train() 计算辅助定位损失
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        # 位置预测直接基于 actor LSTM 输出
        pred_pos = self.aux_position_head(latent_pi)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return values, log_prob, distribution.entropy(), pred_pos

    def get_distribution_with_aux(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...], th.Tensor]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states and the position prediction
        """
        # Call the method from the parent of the parent class
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor
        )
        pred_pos = self.aux_position_head(latent_pi)  # 取 hidden state
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states, pred_pos

    def _predict_with_aux(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...], th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN and the position prediction
        """
        distribution, lstm_states, pred_pos = self.get_distribution_with_aux(
            observation, lstm_states, episode_starts
        )
        return (
            distribution.get_actions(deterministic=deterministic),
            lstm_states,
            pred_pos,
        )

    def predict_with_aux(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None, np.ndarray]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state and the position prediction
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(
                state[0], dtype=th.float32, device=self.device
            ), th.tensor(state[1], dtype=th.float32, device=self.device)
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self.device
            )
            actions, states, pred_pos = self._predict_with_aux(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())
            pred_pos = pred_pos.cpu().numpy()

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
            pred_pos = pred_pos.squeeze(axis=0)

        return actions, states, pred_pos
