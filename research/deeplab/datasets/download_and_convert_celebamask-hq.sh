#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the CelebAMask-HQ dataset.
#
# Usage:
#   bash ./download_and_convert_CelebAMask-HQ.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_CelebAMask-HQ_data.py
#     - download_and_convert_CelebAMask-HQ.sh
#     + CelebAMask-HQ
#       + tfrecord
#       + ADEChallengeData2016
#         + annotations
#           + training
#           + validation
#         + images
#           + training
#           + validation

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./CelebAMask-HQ"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack CelebAMask-HQ dataset.
download_and_uncompress() {
  local DATASET_URL=${1}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${DATASET_URL}"
  fi
  echo "Uncompressing ${FILENAME}"
  unzip "${FILENAME}"
}

# Download the images.
DATASET_URL="https://drive.google.com/uc?export=download&confirm=oS6Z&id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv"

# download_and_uncompress "${DATASET_URL}"

cd "${CURRENT_DIR}"
echo "${CURRENT_DIR}"
echo "${WORK_DIR}/CelebAMask-HQ"
# Root path for CelebAMask-HQ dataset.
CELEBAMASK_HQ_ROOT="${WORK_DIR}/CelebAMask-HQ"

# create trainig labels
# split train and val
python ./convert_celebamask-hq.py  \
  --image_folder="${CELEBAMASK_HQ_ROOT}/CelebA-HQ-img" \
  --image_label_folder="${CELEBAMASK_HQ_ROOT}/CelebAMask-HQ-mask-anno" \
  --mask_folder="${CELEBAMASK_HQ_ROOT}/mask" \
  --output_dir="${CELEBAMASK_HQ_ROOT}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Converting CelebAMask-HQ dataset..."
python ./build_CelebAMask-HQ_data.py  \
  --train_image_folder="${CELEBAMASK_HQ_ROOT}/images/train/" \
  --train_image_label_folder="${CELEBAMASK_HQ_ROOT}/annotations/train/" \
  --val_image_folder="${CELEBAMASK_HQ_ROOT}/images/val/" \
  --val_image_label_folder="${CELEBAMASK_HQ_ROOT}/annotations/val/" \
  --output_dir="${WORK_DIR}/tfrecord" \
