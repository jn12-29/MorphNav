from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from components.pi_eval_visualization import export_online_pi_eval_visualization


class OnlinePIEvalVisualizationCallback(BaseCallback):
    def __init__(
        self,
        output_root: str | Path,
        *,
        n_eval_episodes: int = 4,
        target_key: str | None = None,
        deterministic: bool = True,
        n_bins: int = 32,
        max_units: int | None = None,
        top_k: int = 8,
        max_total_steps: int | None = None,
        gridscore_positive_activations: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_root = Path(output_root)
        self.n_eval_episodes = int(n_eval_episodes)
        self.target_key = target_key
        self.deterministic = bool(deterministic)
        self.n_bins = int(n_bins)
        self.max_units = max_units
        self.top_k = int(top_k)
        self.max_total_steps = max_total_steps
        self.gridscore_positive_activations = bool(gridscore_positive_activations)

    def _init_callback(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _record_summary(self, summary: dict[str, Any]) -> None:
        localization = summary["localization"]
        gridscore = summary["gridscore"]
        self.logger.record("eval/pi/localization_mse", float(localization["mse"]))
        self.logger.record("eval/pi/localization_rmse", float(localization["rmse"]))
        self.logger.record("eval/pi/localization_mae", float(localization["mae"]))
        self.logger.record("eval/pi/gridscore_best", float(gridscore["best_grid_score"]))
        self.logger.record("eval/pi/gridscore_mean", float(gridscore["mean_grid_score"]))
        self.logger.record("eval/pi/gridscore_valid_units", float(gridscore["valid_units"]))
        self.logger.dump(self.num_timesteps)

    def _resolved_target_key(self) -> str:
        if self.target_key is not None:
            return str(self.target_key)
        return str(getattr(self.model, "pi_target_key", "achieved_goal"))

    def _on_step(self) -> bool:
        eval_env = getattr(self.parent, "eval_env", None)
        if eval_env is None or not callable(getattr(self.model, "predict_with_pi", None)):
            return True

        output_dir = self.output_root / f"step_{self.num_timesteps:08d}"
        try:
            summary = export_online_pi_eval_visualization(
                self.model,
                eval_env,
                output_dir,
                n_eval_episodes=self.n_eval_episodes,
                target_key=self._resolved_target_key(),
                deterministic=self.deterministic,
                n_bins=self.n_bins,
                max_units=self.max_units,
                top_k=self.top_k,
                max_total_steps=self.max_total_steps,
                gridscore_positive_activations=self.gridscore_positive_activations,
            )
        except Exception as exc:
            if self.verbose > 0:
                print(f"PI eval visualization skipped: {exc}")
            return True

        self._record_summary(summary)
        if self.verbose > 0:
            print(f"PI eval visualization saved to {output_dir}")
        return True
