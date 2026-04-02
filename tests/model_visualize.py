from sb3_contrib import RecurrentPPO  # 请替换为模型实际使用的算法类，如 SAC, DQN 等
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import components

model = RecurrentPPO.load(
    "/home/xh/ai4neuron/MorphNav/logs/aux_ppo_lstm/PointMaze_6/best_model.zip"
)
print(model.policy)
