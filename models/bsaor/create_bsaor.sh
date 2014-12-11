#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

ROOT=/home/guillem/Github/VA_GiC/
MODEL=${ROOT}/models/bsaor
DATA=${ROOT}/data/bsaor
BUILD=/usr/local/caffe/tools

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $MODEL/bsaor_train_${BACKEND}
rm -rf $MODEL/bsaor_test_${BACKEND}

$BUILD/convert_imageset --shuffle --resize_width=100 --resize_height=100 --backend=${BACKEND} /. $DATA/train-images.txt $MODEL/bsaor_train_${BACKEND}
$BUILD/convert_imageset --shuffle --resize_width=100 --resize_height=100 --backend=${BACKEND} /. $DATA/test-images.txt $MODEL/bsaor_test_${BACKEND}

$BUILD/compute_image_mean --backend=${BACKEND} $MODEL/bsaor_train_${BACKEND} $MODEL/bsaor_train_${BACKEND}.binaryproto
#$BUILD/compute_image_mean --backend=${BACKEND} $MODEL/bsaor_test_${BACKEND} $MODEL/bsaor_test_${BACKEND}.binaryproto

echo "Done."
