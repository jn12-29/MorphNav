CUDA_VISIBLE_DEVICES=2 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"


CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"


CUDA_VISIBLE_DEVICES=2 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" --hyperparams n_timesteps:1e7

# "maze_map_name": "U_MAZE",
CUDA_VISIBLE_DEVICES=0 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'U_MAZE'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.45 --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=1 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'U_MAZE'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.2 --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=2 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'U_MAZE'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.3 --hyperparams n_timesteps:1e7

# "maze_map_name": "MEDIUM_MAZE",
CUDA_VISIBLE_DEVICES=6 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'MEDIUM_MAZE'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" --hyperparams n_timesteps:1e7

# "maze_map_name": "OPEN",
CUDA_VISIBLE_DEVICES=6 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.2 --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=7 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:True achieved_goal_aware:False target_aware:False sensor_aware:False maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.2 --hyperparams n_timesteps:1e7

# 20260312 PointMaze EMPTY
CUDA_VISIBLE_DEVICES=0 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:True achieved_goal_aware:False target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7

# Ant

CUDA_VISIBLE_DEVICES=6 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env AntMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env AntMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:False target_aware:False --hyperparams n_timesteps:1e6




# 20260402 AuxRecurrentPPO PointMaze
CUDA_VISIBLE_DEVICES=1 python ./rl-baselines3-zoo/train.py --algo aux_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:True target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=1 python ./rl-baselines3-zoo/train.py --algo aux_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:True target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7 aux_loss_coef:0.1

CUDA_VISIBLE_DEVICES=5 python ./rl-baselines3-zoo/train.py --algo aux_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:True target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7 aux_loss_coef:0.01

CUDA_VISIBLE_DEVICES=5 python ./rl-baselines3-zoo/train.py --algo aux_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:True target_aware:True sensor_aware:False start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7 aux_loss_coef:0.0

# 20260408 AuxRecurrentPPO PointMaze with dropout and fix max speed

# Path-integration auxiliary PPO-LSTM (place-cell prediction branch, actor path unchanged)
# Offline Phase 1 PI rehearsal uses scripts/offline_pi_rehearsal.py with --preset phase1_pointmaze_pi datasets generated by the grid-cells-style force-control random-walk driver.
CUDA_VISIBLE_DEVICES=0 python ./rl-baselines3-zoo/train.py --algo pi_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze_pi.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False achieved_goal_aware:True target_aware:True sensor_aware:True start_pos_aware:True maze_map_name:"'OPEN'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" success_radius:0.4 --hyperparams n_timesteps:1e7
