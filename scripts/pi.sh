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
