CUDA_VISIBLE_DEVICES=2 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 4  -n 1000 --load-best  --env-kwargs max_episode_steps:1000

CUDA_VISIBLE_DEVICES=2 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 7  -n 3000 --load-best  --env-kwargs max_episode_steps:300 maze_map_name:"'MEDIUM_MAZE'" continuing_task:True

MUJOCO_GL='egl' python -m rl_zoo3.record_training --algo ppo --env dmc-Rat-escape-v0 -f ./logs --exp-id 0  -n 1000 --deterministic

CUDA_VISIBLE_DEVICES=2 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 8  -n 3000 --load-best  --env-kwargs max_episode_steps:1000 maze_map_name:"'OPEN'" continuing_task:False