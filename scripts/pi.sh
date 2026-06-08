## phase1_pointmaze_pi preset keeps sensor_aware=True and uses tangent-biased touch response.
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --num-episodes 10000 \
  --episodes-per-shard 1000 \
  --dataset-seed 0
  
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --dataset-name phase1_pi/probe_seed1 \
  --num-episodes 4000 \
  --episodes-per-shard 1000 \
  --dataset-seed 1


# Offline Phase 1 PI rehearsal. maze_pi.yml encodes obs/start_pos into LSTM initial states.
# PI loss weights timestep 0 of each recurrent sequence by 10x.
# Metrics include full-sequence and first-step localization errors.
# Fresh models use rl-baselines3-zoo/conf/maze_pi.yml unless --config-path overrides it.
# Offline optimizer defaults to Adam; use --optimizer adamw/rmsprop/sgd,
# --weight-decay, and --momentum for SGD/RMSprop when needed.
# eval-every-epochs controls probe frequency.
# eval-artifact-every-epochs writes decoded-vs-target localization visualizations.
# eval-gridscore-every-epochs writes bottleneck ratemap/SAC/grid-score visualizations.
CUDA_VISIBLE_DEVICES=4 conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 300 \
  --first-step-loss-weight 10.0 \
  --eval-every-epochs 1 \
  --eval-artifact-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --checkpoint-every-epochs 5
  
CUDA_VISIBLE_DEVICES=1 conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 300 \
  --first-step-loss-weight 10.0 \
  --eval-every-epochs 1 \
  --eval-artifact-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --eval-artifact-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --checkpoint-every-epochs 5
