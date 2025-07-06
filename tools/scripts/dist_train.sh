#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

EPOCH=epoch_30
CFG_NAME=waymo_models/pvt_ssd
TAG_NAME=default
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# single‐GPU training
python train.py \
  --cfg_file cfgs/${CFG_NAME}.yaml \
  --batch_size 8 \
  --epochs 30 \
  --workers 16 \
  --extra_tag ${TAG_NAME} \
  --max_ckpt_save_num 30 \
  --num_epochs_to_eval 1 \

# evaluation
GT=../data/waymo/gt.bin
# make sure this path points at the Waymo eval binary you built
EVAL=../data/waymo/compute_detection_metrics_main  
DT_DIR=../output/${CFG_NAME}/${TAG_NAME}/eval/eval_with_train/${EPOCH}/val/final_result/data

${EVAL} ${DT_DIR}/detection_pred.bin ${GT}
