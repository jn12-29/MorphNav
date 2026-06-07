"""PPO-LSTM with an online path-integration auxiliary loss.

`achieved_goal` remains in rollout observations as the PI target. Drop it only
from policy features through the configured extractor; do not remove it before
`train()`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
from sb3_contrib.ppo_recurrent.ppo_recurrent import BasePolicy, RecurrentPPO

from components.path_integration import recurrent_first_step_loss_weights, soft_place_cell_cross_entropy
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


class PathIntegrationRecurrentPPO(RecurrentPPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "PathIntegrationMultiInputLstmPolicy": PathIntegrationRecurrentActorCriticPolicy,
    }

    def __init__(
        self,
        *args,
        pi_loss_coef: float = 1.0,
        pi_first_step_loss_weight: float = 10.0,
        pi_target_key: str = "achieved_goal",
        pi_n_place_cells: int = 256,
        pi_place_cell_scale: float = 0.01,
        pi_pos_min: float = -2.5,
        pi_pos_max: float = 2.5,
        pi_neurons_seed: int = 8341,
        **kwargs,
    ):
        policy_kwargs = dict(kwargs.get("policy_kwargs") or {})
        policy_kwargs.setdefault("pi_n_place_cells", pi_n_place_cells)
        policy_kwargs.setdefault("pi_place_cell_scale", pi_place_cell_scale)
        policy_kwargs.setdefault("pi_pos_min", pi_pos_min)
        policy_kwargs.setdefault("pi_pos_max", pi_pos_max)
        policy_kwargs.setdefault("pi_neurons_seed", pi_neurons_seed)
        kwargs["policy_kwargs"] = policy_kwargs

        super().__init__(*args, **kwargs)
        self.pi_loss_coef = float(pi_loss_coef)
        if pi_first_step_loss_weight <= 0.0:
            raise ValueError("pi_first_step_loss_weight must be positive")
        self.pi_first_step_loss_weight = float(pi_first_step_loss_weight)
        self.pi_target_key = str(pi_target_key)

    def train(self) -> None:
        self.policy: PathIntegrationRecurrentActorCriticPolicy
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, pi_losses = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                mask = rollout_data.mask > 1e-8
                values, log_prob, entropy, pi_outputs = self.policy.evaluate_actions_with_pi(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                if not isinstance(rollout_data.observations, dict) or self.pi_target_key not in rollout_data.observations:
                    raise KeyError(
                        f"Path-integration target key {self.pi_target_key!r} is missing from rollout observations"
                    )
                target_pos = rollout_data.observations[self.pi_target_key].float()
                pc_targets = self.policy.path_integration_target_encoder(target_pos).to(dtype=pi_outputs.pc_logits.dtype)
                sequence_count = int(rollout_data.lstm_states.pi[0].shape[1])
                loss_weights = recurrent_first_step_loss_weights(
                    mask,
                    sequence_count=sequence_count,
                    first_step_weight=self.pi_first_step_loss_weight,
                )
                pi_loss = soft_place_cell_cross_entropy(
                    pi_outputs.pc_logits,
                    pc_targets,
                    mask=mask,
                    weights=loss_weights,
                )
                pi_losses.append(pi_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.pi_loss_coef * pi_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/pi_loss", np.mean(pi_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/pi_loss_coef", self.pi_loss_coef)
        self.logger.record("train/pi_first_step_loss_weight", self.pi_first_step_loss_weight)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict_with_pi(self, observation, state=None, episode_start=None, deterministic: bool = False):
        return self.policy.predict_with_pi(observation, state, episode_start, deterministic)
