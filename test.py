import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
paradir = os.path.dirname(curdir)

sys.path.append(paradir)

import custom_envs
import gymnasium as gym

env = gym.make("DMCAnt-v0", task_name="Floor")

env = gym.make("DMCHumanoid-v0", task_name="Floor")

env = gym.make("DMCHumanoid-v0", task_name="Gaps")

env.reset()
