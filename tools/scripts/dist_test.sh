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

CFG_NAME=waymo_models/pvt_ssd_3f
TAG_NAME=default

CKPT=../data/ckpts/pvt_ssd.pth

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 4 --extra_tag $TAG_NAME --ckpt $CKPT

GT=../data/waymo/gt.bin
EVAL=../data/waymo/compute_detection_metrics_main
DT_DIR=../output/$CFG_NAME/$TAG_NAME/eval/$EPOCH/val/default/final_result/data

$EVAL $DT_DIR/detection_pred.bin $GT
