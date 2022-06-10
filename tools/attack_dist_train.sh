#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
    exit
fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3
LOAD_FROM=$4

#echo '-----------'
#
#echo ${@}

PORT=${PORT:-29501}
#resume=result/ctw1500/psenet_r50_bifpnf_resasapp_coordconv_after_600e_ctw1500_8sampler_2/epoch_290.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} ${@:4}
else
#    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#        $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --resume-from=${LOAD_FROM} --launcher pytorch ${@:4}

    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --load-from=$LOAD_FROM --launcher pytorch
fi