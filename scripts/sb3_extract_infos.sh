# test extract acts
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video_with_extraction.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 4  -n 1000 --load-best  --env-kwargs max_episode_steps:1000 continuing_task:True cur_pos_aware:False target_aware:False sensor_aware:False xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"

# test extract data
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 5  -n 10000 --load-best  --env-kwargs max_episode_steps:10000 continuing_task:True cur_pos_aware:False target_aware:False sensor_aware:False xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"

CUDA_VISIBLE_DEVICES=5 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 5  -n 10000 --load-best  --env-kwargs max_episode_steps:1000 continuing_task:False maze_map_name:"'U_MAZE'" cur_pos_aware:False target_aware:False sensor_aware:False xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"

CUDA_VISIBLE_DEVICES=5 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 2  -n 10000 --load-best  --env-kwargs max_episode_steps:300 continuing_task:False maze_map_name:"'U_MAZE'" cur_pos_aware:False target_aware:False sensor_aware:False xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"


CUDA_VISIBLE_DEVICES=3 MUJOCO_GL='egl' python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id 3  -n 12000 --load-best  --env-kwargs max_episode_steps:300