#!/bin/sh

MASTER_PORT=25564
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port ${MASTER_PORT} \
    target_adapt_dbscan.py \
    -ds dukemtmc \
    -dt msmt17 \
    -a vit_base_patch16_224 \
	--num-instances 4 \
    --lr 0.008 \
    --iters 400 \
    -b 64 \
    --epochs 70 \
    --dropout 0 \
    --lambda-value 0.0 \
    --lintype lora \
	--init logs/pretrain/dukemtmc/vitbase/model_best.pth.tar \
	--logs-dir logs/adapt/dukemtmcTOmsmt17/vitbase \
	--rank 8 \
    --lora_alpha 32 \
	#--rr-gpu \

