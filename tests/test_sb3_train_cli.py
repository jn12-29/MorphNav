import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ZOO_ROOT = REPO_ROOT / "rl-baselines3-zoo"


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], False),
        (["--gridscore-positive-activations"], True),
    ],
)
def test_sb3_train_cli_passes_gridscore_positive_activations(monkeypatch, extra_args, expected):
    monkeypatch.syspath_prepend(str(ZOO_ROOT))
    from rl_zoo3 import train as train_module

    captured = {}

    class FakeExperimentManager:
        def __init__(self, *args, **kwargs):
            captured["gridscore_positive_activations"] = kwargs["gridscore_positive_activations"]

        def setup_experiment(self):
            return None

        def hyperparameters_optimization(self):
            captured["optimized"] = True

    monkeypatch.setattr(train_module, "ExperimentManager", FakeExperimentManager)
    monkeypatch.setattr(train_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(train_module.np.random, "randint", lambda *args, **kwargs: SimpleNamespace(item=lambda: 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--algo",
            "pi_ppo_lstm",
            "--env",
            "CartPole-v1",
            *extra_args,
        ],
    )
    monkeypatch.setitem(gym.envs.registry, "CartPole-v1", gym.envs.registry["CartPole-v1"])

    train_module.train()

    assert captured["gridscore_positive_activations"] is expected
    assert captured["optimized"] is True
