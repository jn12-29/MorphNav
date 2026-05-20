## phase1_pointmaze_pi preset keeps sensor_aware=True and uses tangent-biased touch response.
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --output-dir recorded_data \
  --dataset-name pointmaze_mujoco_pi_rehearsal_gcf_10k \
  --num-episodes 10000 \
  --episodes-per-shard 1000 \
  --dataset-seed 0
  
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --output-dir recorded_data \
  --dataset-name pointmaze_mujoco_pi_probe_gcf_4k \
  --num-episodes 4000 \
  --episodes-per-shard 1000 \
  --dataset-seed 1
