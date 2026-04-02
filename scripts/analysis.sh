# 网格模式（自适应，每个单元独立颜色范围）
python tests/analyze_recorded_data.py --max-display 1286 --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 单独保存模式（每个单元独立自适应）
python tests/analyze_recorded_data.py --plot-mode individual --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 随机采样模式（自适应）
python tests/analyze_recorded_data.py --plot-mode sample --max-display 64 --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 可视化轨迹比较
python tests/analyze_pos_data.py --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos 
