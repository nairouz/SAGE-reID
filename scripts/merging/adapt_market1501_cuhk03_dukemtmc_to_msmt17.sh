#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3
python target_adapt_dbscan_merge.py \
    -ds1 market1501 \
    -ds2 cuhk03 \
    -ds3 dukemtmc \
    -dt msmt17 \
    -a vit_base_patch16_224 \
	--num-instances 4 \
    --lr 0.08 \
    --iters 400 \
    -b 64 \
    --epochs 1 \
    --dropout 0 \
    --lambda-value 0.3 \
    --lintype gate \
	--logs-dir logs/adapt/market1501_cuhk03_dukemtmcTOmsmt17/vitbase \
	--rank 8 \
    --lora_alpha 32 \
	#--rr-gpu \

