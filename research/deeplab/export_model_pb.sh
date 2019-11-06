# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
DATASET_PATH="deeplab/datasets/CelebAMask-HQ/CelebAMask-HQ-skin-neck-hair"
CHECKPOINT_PATH="${DATASET_PATH}/init_models/face_3_mobilenetv2_full_scrutch/model.ckpt-29454"
OUTPUT_DIR="${DATASET_PATH}/tflite"

mkdir -p "${OUTPUT_DIR}"

python deeplab/export_model.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --num_classes=19 \
  --export_crop_size="512,512" \
  --export_path=${OUTPUT_DIR}/frozen_inference_graph.pb