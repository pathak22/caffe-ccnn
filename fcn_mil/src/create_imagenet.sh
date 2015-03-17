#!/usr/bin/env sh

# This file is copied from caffe/examples/imagenet/create_imagenet.sh

# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OUTDIR=/mnt/a/pathak/fcn_mil_cache/lmdb_dataset_cache
DATA=/mnt/a/pathak/fcn_mil_cache/lmdb_dataset_cache
TOOLS=/home/pathak/caffe_fcn_mil/build/tools

TRAIN_DATA_ROOT=/mnt/x/ilsvrc2012/train/
VAL_DATA_ROOT=/mnt/x/ilsvrc2012/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/ilsvrc12_pascal_train.txt \
    $OUTDIR/datum_ilsvrc12_pascal_train

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/ilsvrc12_pascal_val.txt \
    $OUTDIR/datum_ilsvrc12_pascal_val

echo "Done."
