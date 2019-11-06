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

tflite_convert \
  --graph_def_file=${OUTPUT_DIR}/frozen_inference_graph.pb \
  --output_file=${OUTPUT_DIR}/celeb_skin_neck_hair_full_scrutch.tflite \
  --output_format=TFLITE \
  --input_shape=1,512,512,3 \
  --input_arrays="MobilenetV2/MobilenetV2/input" \
  --output_arrays="SemanticPredictions"