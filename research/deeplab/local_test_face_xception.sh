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
DATASET_DIR="datasets/CelebAMask-HQ"
CELEB_FOLDER="CelebAMask-HQ-skin-eye-lips"
EXP_FOLDER="exp/face_3_xception_pretrained"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

CELEB_DATASET="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/tfrecord"

DATASET_NAME="celebamask_hq_customized"

echo "===========train=============="
# Train 10 iterations.
# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
# See core/feature_extractor.py for supported model variants.

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
NUM_ITERATIONS=1000000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --train_crop_size="513,513" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --fine_tune_batch_norm=true \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CELEB_DATASET}" \
  --dataset="${DATASET_NAME}" \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --initialize_last_layer=False

echo "===========eval=============="
# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.

# python "${WORK_DIR}"/eval.py \
#   --logtostderr \
#   --eval_split="val" \
#   --model_variant="xception_65" \
#   --eval_crop_size="513,513" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --eval_logdir="${EVAL_LOGDIR}" \
#   --dataset_dir="${CELEB_DATASET}" \
#   --dataset="${DATASET_NAME}" \
#   --max_number_of_evaluations=1

echo "===========vis=============="
# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --vis_crop_size="513,513" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${CELEB_DATASET}" \
  --dataset="${DATASET_NAME}" \
  --max_number_of_iterations=1

# echo "===========export=============="
# # Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
# 
# python "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="mobilenet_v2" \
#   --num_classes=19 \
#   --crop_size=512 \
#   --crop_size=512 \
#   --dataset="${DATASET_NAME}" \
#   --inference_scales=1.0
# 
# # Run inference with the exported checkpoint.
# # Please refer to the provided deeplab_demo.ipynb for an example.
# echo "=================tflite=================="
# # Set up the working directories.
# tflite_convert \
#   --graph_def_file=${EXPORT_PATH} \
#   --output_file=${EXPORT_DIR}/celeba_skin_eye_lips_1024.tflite \
#   --output_format=TFLITE \
#   --input_shape=1,1024,1024,3 \
#   --inference_input_type=FLOAT \
#   --inference_type=FLOAT \
#   --input_arrays="MobilenetV2/MobilenetV2/input" \
#   --output_arrays="ArgMax"