CUDA_VISIBLE_DEVICES=2 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"


CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False sensor_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'"


CUDA_VISIBLE_DEVICES=2 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False time_penalty:0.001 xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" --hyperparams n_timesteps:1e7


CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False sensor_aware:False xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" --hyperparams n_timesteps:1e7

# "maze_map_name": "MEDIUM_MAZE",
CUDA_VISIBLE_DEVICES=6 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False sensor_aware:False maze_map_name:"'MEDIUM_MAZE'" xml_file_path:"'/home/xh/ai4neuron/MorphNav/envs/assets/point_v1.xml'" --hyperparams n_timesteps:1e7



CUDA_VISIBLE_DEVICES=6 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env AntMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False --hyperparams n_timesteps:1e7

CUDA_VISIBLE_DEVICES=3 python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env AntMaze -conf ./rl-baselines3-zoo/conf/maze.yml --vec-env subproc -P --tensorboard-log ./logs --eval-freq 10_000 --eval-episodes 32 --n-eval-envs 8 --save-freq 100_000 --env-kwargs continuing_task:False cur_pos_aware:False target_aware:False --hyperparams n_timesteps:1e6