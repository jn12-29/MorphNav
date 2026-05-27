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
# Metrics include full-sequence and first-step localization errors.
# Probe-time grid-score analysis is enabled with bounded cost for experiment monitoring.
CUDA_VISIBLE_DEVICES=4 conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 300 \
  --eval-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --checkpoint-every-epochs 5
  
CUDA_VISIBLE_DEVICES=1 conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 300 \
  --eval-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --checkpoint-every-epochs 5