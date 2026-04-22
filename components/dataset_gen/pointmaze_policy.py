from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from components.dataset_gen.pointmaze_config import PointMazePolicyConfig


@dataclass
class WeakRandomPolicyDriver:
    config: PointMazePolicyConfig
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._current_action: np.ndarray | None = None
        self._segment_steps_left = 0
        self._prev_xy: np.ndarray | None = None
        self._stuck_counter = 0

    def _sample_segment_length(self) -> int:
        low, high = self.config.segment_length_range
        if high < low:
            raise ValueError("segment_length_range must satisfy low <= high")
        return int(self._rng.integers(low, high + 1))

    def _sample_action(self) -> np.ndarray:
        base = np.array([self.config.forward_bias, self.config.turn_bias], dtype=np.float32)
        if self.config.action_noise_scale > 0:
            noise = self._rng.normal(loc=0.0, scale=self.config.action_noise_scale, size=2).astype(np.float32)
            base = base + noise
        return np.clip(base, -1.0, 1.0).astype(np.float32)

    def _resample(self) -> None:
        self._current_action = self._sample_action()
        self._segment_steps_left = self._sample_segment_length()
        self._stuck_counter = 0

    def act(self, agent_xy: np.ndarray, heading: float, collision: bool) -> np.ndarray:
        del heading
        xy = np.asarray(agent_xy, dtype=np.float32)

        if self._current_action is None:
            self._resample()

        if self._prev_xy is not None:
            displacement = float(np.linalg.norm(xy - self._prev_xy))
            if displacement <= self.config.stuck_threshold:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._prev_xy = xy.copy()

        should_resample = False
        if collision:
            should_resample = True
        elif self._segment_steps_left <= 0:
            should_resample = True
        elif self._stuck_counter >= self.config.stuck_patience:
            should_resample = True
        elif self.config.subgoal_resample_prob > 0 and self._rng.random() < self.config.subgoal_resample_prob:
            should_resample = True

        if should_resample:
            self._resample()

        self._segment_steps_left -= 1
        return self._current_action.copy()
