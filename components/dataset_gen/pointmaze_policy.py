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
        if self.config.arena_max <= self.config.arena_min:
            raise ValueError("arena_max must be greater than arena_min")
        if self.config.boundary_margin < 0.0:
            raise ValueError("boundary_margin must be >= 0")
        if self.config.boundary_lookahead_time < 0.0:
            raise ValueError("boundary_lookahead_time must be >= 0")
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

    def _reflect_heading(self, xy: np.ndarray) -> None:
        direction = np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float32)
        reflected = False
        lower = float(self.config.arena_min) + float(self.config.boundary_margin)
        upper = float(self.config.arena_max) - float(self.config.boundary_margin)

        if xy[0] <= lower and direction[0] < 0.0:
            direction[0] *= -1.0
            reflected = True
        elif xy[0] >= upper and direction[0] > 0.0:
            direction[0] *= -1.0
            reflected = True
        if xy[1] <= lower and direction[1] < 0.0:
            direction[1] *= -1.0
            reflected = True
        elif xy[1] >= upper and direction[1] > 0.0:
            direction[1] *= -1.0
            reflected = True

        if not reflected:
            direction *= -1.0
        self._heading = self._wrap_to_pi(float(np.arctan2(direction[1], direction[0])))
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

    def _apply_boundary_reflection(self, xy: np.ndarray, desired_velocity: np.ndarray) -> np.ndarray:
        velocity = desired_velocity.copy()
        lower = float(self.config.arena_min) + float(self.config.boundary_margin)
        upper = float(self.config.arena_max) - float(self.config.boundary_margin)
        projected = xy + velocity * float(self.config.boundary_lookahead_time)
        reflected = False

        if (xy[0] <= lower or projected[0] <= self.config.arena_min) and velocity[0] < 0.0:
            velocity[0] *= -1.0
            reflected = True
        elif (xy[0] >= upper or projected[0] >= self.config.arena_max) and velocity[0] > 0.0:
            velocity[0] *= -1.0
            reflected = True
        if (xy[1] <= lower or projected[1] <= self.config.arena_min) and velocity[1] < 0.0:
            velocity[1] *= -1.0
            reflected = True
        elif (xy[1] >= upper or projected[1] >= self.config.arena_max) and velocity[1] > 0.0:
            velocity[1] *= -1.0
            reflected = True

        if reflected:
            self._heading = self._wrap_to_pi(float(np.arctan2(velocity[1], velocity[0])))
            self._angular_velocity = 0.0
        return velocity

    def _rate_limit_action(self, action: np.ndarray) -> np.ndarray:
        delta = np.clip(
            action - self._prev_action,
            -float(self.config.max_action_delta),
            float(self.config.max_action_delta),
        )
        return self._prev_action + delta

    def act(self, agent_xy: np.ndarray, agent_qvel: np.ndarray, collision: bool) -> np.ndarray:
        xy = np.asarray(agent_xy, dtype=np.float32)[:2]
        qvel = np.asarray(agent_qvel, dtype=np.float32)[:2]
        self._update_stuck_state(xy)

        if collision:
            self._reflect_heading(xy)
        elif self._stuck_counter >= self.config.stuck_patience:
            self._resample_heading()
        else:
            self._advance_heading()

        speed = self._sample_speed()
        desired_velocity = speed * np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float32)
        desired_velocity = self._apply_boundary_reflection(xy, desired_velocity)
        self._desired_velocity = desired_velocity.astype(np.float32)

        action = float(self.config.velocity_tracking_gain) * (self._desired_velocity - qvel)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action = self.config.action_smoothing * self._prev_action + (1.0 - self.config.action_smoothing) * action
        action = self._rate_limit_action(action)
        self._prev_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return self._prev_action.copy()
