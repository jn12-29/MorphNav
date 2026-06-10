# 网格模式（自适应，每个单元独立颜色范围）
python tests/analyze_rollout_data.py --max-display 1286 --data-dir runs/sb3/aux_ppo_lstm/PointMaze_6/rollouts/best/data

# 单独保存模式（每个单元独立自适应）
python tests/analyze_rollout_data.py --plot-mode individual --data-dir runs/sb3/aux_ppo_lstm/PointMaze_6/rollouts/best/data

# 随机采样模式（自适应）
python tests/analyze_rollout_data.py --plot-mode sample --max-display 64 --data-dir runs/sb3/aux_ppo_lstm/PointMaze_6/rollouts/best/data

# 可视化轨迹比较
python tests/analyze_pos_data.py --data-dir runs/sb3/aux_ppo_lstm/PointMaze_6/rollouts/best/data

# Offline PI bottleneck grid-score analysis example
python scripts/analyze_offline_pi_representations.py --model-path runs/offline_pi/pointmaze_phase1_seed0_YYYYMMDD_HHMMSS/models/final_model.zip --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 --max-seq-len 1000

# PointMaze dataset distribution validation before offline PI training
python scripts/analyze_pointmaze_dataset.py --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0

# PointMaze dataset trajectory replay render
MUJOCO_GL=egl python scripts/render_pointmaze_trajectory.py --mode dataset --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 --episodes 0 1 2 --output-dir runs/offline_pi/render_examples/dataset_replay_YYYYMMDD_HHMMSS

# Offline PI decoded prediction render on held-out dataset episodes
MUJOCO_GL=egl python scripts/render_pointmaze_trajectory.py --mode probe --model-path runs/offline_pi/pointmaze_phase1_seed0_YYYYMMDD_HHMMSS/models/final_model.zip --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 --episodes 0 1 2 --output-dir runs/offline_pi/render_examples/probe_predictions_YYYYMMDD_HHMMSS

# Recurrent policy rollout render from an offline PI checkpoint
MUJOCO_GL=egl python scripts/render_pointmaze_trajectory.py --mode rollout --model-path runs/offline_pi/pointmaze_phase1_seed0_YYYYMMDD_HHMMSS/models/final_model.zip --seed 0 --steps 1000 --output-dir runs/offline_pi/render_examples/rollout_seed0_YYYYMMDD_HHMMSS
