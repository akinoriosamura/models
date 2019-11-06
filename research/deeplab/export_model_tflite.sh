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
DATASET_PATH="deeplab/datasets/CelebAMask-HQ/CelebAMask-HQ"
OUTPUT_DIR="${DATASET_PATH}/tflite"

tflite_convert \
  --graph_def_file=${OUTPUT_DIR}/frozen_inference_graph.pb \
  --output_file=${OUTPUT_DIR}/face_19_sample.tflite \
  --output_format=TFLITE \
  --input_shape=1,512,512,3 \
  --inference_input_type=FLOAT \
  --inference_type=FLOAT \
  --input_arrays="MobilenetV2/MobilenetV2/input" \
  --output_arrays="ArgMax"