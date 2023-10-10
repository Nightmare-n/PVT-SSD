#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=4

EPOCH=epoch_30

CFG_NAME=waymo_models/pvt_ssd
TAG_NAME=default

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 2 --extra_tag $TAG_NAME --max_ckpt_save_num 1 --num_epochs_to_eval 1

GT=../data/waymo/gt.bin
EVAL=../data/waymo/compute_detection_metrics_main
DT_DIR=../output/$CFG_NAME/$TAG_NAME/eval/eval_with_train/$EPOCH/val/final_result/data

$EVAL $DT_DIR/detection_pred.bin $GT
