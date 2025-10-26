#!/usr/bin/env bash
set -e
set -x
BASE_DATA_DIR='/data1/yhyun225/dataset/marigold_dataset/eval'
ckpt='checkpoints'
subfolder='output/eval'

for step in 8
do
    CUDA_VIISBLE_DEVICES=0 python script/depth/infer.py \
        --checkpoint $ckpt \
        --seed 1234 \
        --base_data_dir $BASE_DATA_DIR \
        --sampling_steps ${step} \
        --dataset_config config/dataset_depth/data_kitti_eigen_test.yaml \
        --output_dir ${subfolder}/kitti_eigen_test/${step}_steps/prediction
    
    CUDA_VISIBLE_DEVICES=0 python script/depth/eval.py \
        --base_data_dir $BASE_DATA_DIR \
        --dataset_config config/dataset_depth/data_kitti_eigen_test.yaml \
        --alignment least_square_log_depth \
        --prediction_dir ${subfolder}/kitti_eigen_test/${step}_steps/prediction \
        --output_dir ${subfolder}/kitti_eigen_test/${step}_steps/eval_metric
done
