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
PASCAL_FOLDER="pascal_voc_seg"
EXP_FOLDER="exp/voc_mobilev2_full_scrach"
DATASET_DIR="datasets"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
#TF_INIT_ROOT="http://download.tensorflow.org/models"
# CKPT_NAME="face_19_mobilenetv2_dm1.0_pre_mobilenetv2_coco_voc_trainaug"
#TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
#cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
#cd "${CURRENT_DIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

echo "===========train=============="
# From tensorflow/models/research/
# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
# See core/feature_extractor.py for supported model variants.

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note

nohup python "${WORK_DIR}"/train.py \
    --logtostderr \
    --training_number_of_steps=1000000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --train_crop_size="513,513" \
    --fine_tune_batch_norm=False \
    --train_batch_size=8 \
    --dataset="pascal_voc_seg" \
    --train_logdir=${TRAIN_LOGDIR} \
    --dataset_dir=${PASCAL_DATASET} &
#  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt-3589"
#  --initialize_last_layer=False
#  --depth_multiplier=0.5 \