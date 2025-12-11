import os
import sys

# paradir = os.path.dirname(curdir)
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
import gymnasium as gym
import custom_envs

env = gym.make("DMCAnt-v0", task_name="Floor")

env = gym.make("DMCHumanoid-v0", task_name="Floor")

env = gym.make("DMCHumanoid-v0", task_name="Gaps")

env = gym.make("DMCHumanoid-v0", task_name="Maze")

env.reset()
