cd ../
# Set up the working environment.
CURRENT_DIR=$(pwd)
echo "noe work dir"
echo ${CURRENT_DIR}
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
CELEB_FOLDER="CelebAMask-HQ"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${CELEB_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

NUM_ITERATIONS=20000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --decoder_output_stride=4 \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --resize_factor=16 \
  --train_crop_size="512,512" \
  --train_batch_size=4 \
  --dataset="celebamask_hq" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"