import torch

acts = torch.load("runs/sb3/ppo_lstm/PointMaze_4/rollouts/best/data/activations.pt")

print(acts)

import pdb

pdb.set_trace()
