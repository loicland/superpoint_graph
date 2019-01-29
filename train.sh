#!/bin/bash

CUDA_VISIBLE_DEVICES=0



python learning/main.py \
        --dataset 'custom_s3dis' \
        --S3DIS_PATH 'data/custom_S3DIS_augmented' \
        --cvfold '1' \
        --epochs '350' \
        --lr_steps '[275,320]' \
        --batch_size '4' \
        --test_nth_epoch '30' \
        --model_config 'gru_10_0,f_14' \
        --ptn_nfeat_stn '8' \
        --pc_attribs 'xyzelpsv' \
        --nworkers '8' \
        --odir 'results/s3dis/bw/cv1_aug_bs4' \
        --resume RESUME