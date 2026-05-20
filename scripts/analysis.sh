# 网格模式（自适应，每个单元独立颜色范围）
python tests/analyze_recorded_data.py --max-display 1286 --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 单独保存模式（每个单元独立自适应）
python tests/analyze_recorded_data.py --plot-mode individual --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 随机采样模式（自适应）
python tests/analyze_recorded_data.py --plot-mode sample --max-display 64 --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos

# 可视化轨迹比较
python tests/analyze_pos_data.py --data-dir /home/xh/ai4neuron/MorphNav/recorded_data/logs/aux_ppo_lstm/PointMaze_6/videos 

# Offline PI bottleneck grid-score analysis example
python scripts/analyze_offline_pi_representations.py --model-path logs/offline_pi/pointmaze_phase1_seed0/final_model.zip --dataset-root recorded_data/pointmaze_mujoco_pi_probe --output-dir logs/offline_pi/pointmaze_phase1_seed0/analysis --max-seq-len 1000

# PointMaze dataset distribution validation before offline PI training
python scripts/analyze_pointmaze_dataset.py --dataset-root recorded_data/pointmaze_mujoco_pi_rehearsal --output-dir logs/offline_pi/pointmaze_phase1_seed0/dataset_analysis
