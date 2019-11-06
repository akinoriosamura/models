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
DATASET_PATH="deeplab/datasets/pascal_voc_seg"
CHECKPOINT_PATH="${DATASET_PATH}/exp/tflite_sample/train/model.ckpt-643"
OUTPUT_DIR="${DATASET_PATH}/exp/tflite_sample/tflite"

mkdir -p ${OUTPUT_DIR}

python deeplab/export_model.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --quantize_delay_step=0 \
  --export_path=${OUTPUT_DIR}/frozen_inference_graph.pb

tflite_convert \
  --graph_def_file=${OUTPUT_DIR}/frozen_inference_graph.pb \
  --output_file=${OUTPUT_DIR}/sample_pascal.tflite \
  --output_format=TFLITE \
  --input_shape=1,513,513,3 \
  --input_arrays="MobilenetV2/MobilenetV2/input" \
  --inference_type=QUANTIZED_UINT8 \
  --inference_input_type=QUANTIZED_UINT8 \
  --std_dev_values=128 \
  --mean_values=128 \
  --change_concat_input_ranges=true \
  --output_arrays="ArgMax"