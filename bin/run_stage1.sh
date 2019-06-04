#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_CONFIG=config.yml
LOGDIR=/raid/bac/kaggle/logs/landmark/resume0/se_resnext50_32x4d
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose