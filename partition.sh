#!/bin/bash

python partition/partition.py \
       --dataset custom_s3dis \
       --ROOT_PATH data/custom_S3DIS_augmented \
       --voxel_width 0.03 \
       --reg_strength 0.03