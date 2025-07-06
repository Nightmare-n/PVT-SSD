#!/usr/bin/env bash
set -euo pipefail

# 2. Activate your PCDet virtualenv
source pvt/bin/activate

# 3. Copy Waymo TFRecords
gsutil -m cp -r \
  gs://waymo_open_dataset_v_1_4_3/individual_files/{training,validation} \
  data/waymo/raw_data/

# 4. Flatten into raw_data/
mv data/waymo/raw_data/training/* data/waymo/raw_data/
mv data/waymo/raw_data/validation/* data/waymo/raw_data/
rmdir data/waymo/raw_data/training data/waymo/raw_data/validation

# 5. Generate .pkl infos
python -m pcdet.datasets.waymo.waymo_dataset \
  --func create_waymo_infos \
  --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

echo "Waymo prep complete—by the grace of Shri Radha Rani."

echo "Waymo training starting the grace of Shri Radha Rani."

cd tools/

bash ./scripts/dist_train.sh

echo "Waymo training complete—by the grace of Shri Radha Rani."
