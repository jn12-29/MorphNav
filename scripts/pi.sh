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


# Offline Phase 1 PI rehearsal. TensorBoard is enabled by default and falls back to JSON/text logs if unavailable.
CUDA_VISIBLE_DEVICES=0 conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 10 \
  --eval-every-epochs 1 \
  --checkpoint-every-epochs 1