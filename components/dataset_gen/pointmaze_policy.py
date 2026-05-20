"""Collection policy for PointMaze offline PI datasets.

`GridCellRandomWalkForceDriver` is a grid-cells-style smooth velocity random
walk tracked through the environment's global x/y motor action space. Touch
recovery uses observable touch sensors with tangent-biased motion and bounded
jitter; it does not use egocentric commands, hard-coded arena reflection,
synthetic collisions, or random heading resampling in the touch branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from components.dataset_gen.pointmaze_config import PointMazePolicyConfig


@dataclass
class GridCellRandomWalkForceDriver:
    config: PointMazePolicyConfig
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._heading = float(self._rng.uniform(-np.pi, np.pi))
        self._angular_velocity = 0.0
        self._prev_xy: np.ndarray | None = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._desired_velocity = np.zeros(2, dtype=np.float32)
        self._stuck_counter = 0
        if self.config.motion_dt <= 0.0:
            raise ValueError("motion_dt must be positive")
        if not 0.0 <= self.config.angular_velocity_decay <= 1.0:
            raise ValueError("angular_velocity_decay must be in [0, 1]")
        if self.config.angular_velocity_std < 0.0:
            raise ValueError("angular_velocity_std must be >= 0")
        if self.config.speed_mean <= 0.0:
            raise ValueError("speed_mean must be positive")
        if self.config.speed_std < 0.0:
            raise ValueError("speed_std must be >= 0")
        if self.config.speed_max <= 0.0:
            raise ValueError("speed_max must be positive")
        if self.config.velocity_tracking_gain <= 0.0:
            raise ValueError("velocity_tracking_gain must be positive")
        if not 0.0 <= self.config.action_smoothing < 1.0:
            raise ValueError("action_smoothing must be in [0, 1)")
        if self.config.max_action_delta <= 0.0:
            raise ValueError("max_action_delta must be positive")
        if self.config.touch_tangent_weight < 0.0:
            raise ValueError("touch_tangent_weight must be >= 0")
        if self.config.touch_away_weight < 0.0:
            raise ValueError("touch_away_weight must be >= 0")
        if self.config.touch_tangent_weight == 0.0 and self.config.touch_away_weight == 0.0:
            raise ValueError("at least one touch response weight must be positive")
        if self.config.touch_jitter_angle < 0.0:
            raise ValueError("touch_jitter_angle must be >= 0")
        if self.config.stuck_threshold < 0.0:
            raise ValueError("stuck_threshold must be >= 0")
        if self.config.stuck_patience <= 0:
            raise ValueError("stuck_patience must be positive")

    @staticmethod
    def _wrap_to_pi(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    def _sample_speed(self) -> float:
        if self.config.speed_std == 0.0:
            return min(float(self.config.speed_mean), float(self.config.speed_max))

        speed = -1.0
        while speed <= 0.0:
            speed = float(self._rng.normal(self.config.speed_mean, self.config.speed_std))
        return min(speed, float(self.config.speed_max))

    def _update_stuck_state(self, xy: np.ndarray) -> None:
        if self._prev_xy is not None:
            displacement = float(np.linalg.norm(xy - self._prev_xy))
            if displacement <= self.config.stuck_threshold:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._prev_xy = xy.copy()

    def _resample_heading(self) -> None:
        self._heading = float(self._rng.uniform(-np.pi, np.pi))
        self._angular_velocity = 0.0
        self._stuck_counter = 0

    @staticmethod
    def _rotate_vector(vector: np.ndarray, angle: float) -> np.ndarray:
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        return np.array(
            [
                cos_angle * float(vector[0]) - sin_angle * float(vector[1]),
                sin_angle * float(vector[0]) + cos_angle * float(vector[1]),
            ],
            dtype=np.float32,
        )

    def _turn_from_touch(self, touch: np.ndarray, qvel: np.ndarray) -> None:
        touch = np.asarray(touch, dtype=np.float32).reshape(-1)
        if touch.size < 4 or float(np.max(touch[:4])) <= 0.0:
            return

        front, back, left, right = touch[:4]
        away = np.array(
            [
                float(back - front),
                float(right - left),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(away))
        if norm <= 1e-8:
            return

        away /= norm
        tangent = np.array([-away[1], away[0]], dtype=np.float32)
        reference_velocity = qvel
        if float(np.linalg.norm(reference_velocity)) <= 1e-8:
            reference_velocity = self._desired_velocity
        if float(np.dot(tangent, reference_velocity)) < 0.0:
            tangent = -tangent

        direction = (
            float(self.config.touch_tangent_weight) * tangent
            + float(self.config.touch_away_weight) * away
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-8:
            return
        direction /= direction_norm

        jitter = float(
            self._rng.uniform(
                -float(self.config.touch_jitter_angle),
                float(self.config.touch_jitter_angle),
            )
        )
        direction = self._rotate_vector(direction, jitter)
        away_dot = float(np.dot(direction, away))
        if away_dot <= 0.0:
            direction = direction - away_dot * away + 1e-6 * away
            direction /= float(np.linalg.norm(direction))
        heading = float(np.arctan2(direction[1], direction[0]))
        self._heading = self._wrap_to_pi(heading)
        self._angular_velocity = 0.0
        self._stuck_counter = 0

    def _advance_heading(self) -> None:
        decay = float(self.config.angular_velocity_decay)
        self._angular_velocity = (1.0 - decay) * self._angular_velocity + decay * float(
            self._rng.normal(
                self.config.angular_velocity_mean,
                self.config.angular_velocity_std,
            )
        )
        self._heading = self._wrap_to_pi(
            self._heading + self._angular_velocity * float(self.config.motion_dt)
        )

    def _rate_limit_action(self, action: np.ndarray) -> np.ndarray:
        delta = np.clip(
            action - self._prev_action,
            -float(self.config.max_action_delta),
            float(self.config.max_action_delta),
        )
        return self._prev_action + delta

    def act(self, agent_xy: np.ndarray, agent_qvel: np.ndarray, touch: np.ndarray | None = None) -> np.ndarray:
        xy = np.asarray(agent_xy, dtype=np.float32)[:2]
        qvel = np.asarray(agent_qvel, dtype=np.float32)[:2]
        self._update_stuck_state(xy)
        touch_arr = np.asarray([] if touch is None else touch, dtype=np.float32).reshape(-1)
        has_touch = touch_arr.size > 0 and bool(np.any(touch_arr > 0.0))

        if has_touch:
            self._turn_from_touch(touch_arr, qvel)
        elif self._stuck_counter >= self.config.stuck_patience:
            self._resample_heading()
        else:
            self._advance_heading()

        speed = self._sample_speed()
        desired_velocity = speed * np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float32)
        self._desired_velocity = desired_velocity.astype(np.float32)

        action = float(self.config.velocity_tracking_gain) * (self._desired_velocity - qvel)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action = self.config.action_smoothing * self._prev_action + (1.0 - self.config.action_smoothing) * action
        action = self._rate_limit_action(action)
        self._prev_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return self._prev_action.copy()
