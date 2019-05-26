#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml
LOGDIR=/raid/bac/kaggle/logs/landmark/resnet50
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose