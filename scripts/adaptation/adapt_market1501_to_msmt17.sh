#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3
python target_adapt_dbscan.py \
    -ds market1501 \
    -dt msmt17 \
    -a vit_base_patch16_224 \
	--num-instances 4 \
    --lr 0.008 \
    --iters 400 \
    -b 64 \
    --epochs 70 \
    --dropout 0 \
    --lambda-value 0.3 \
    --lintype lora \
	--init logs/pretrain/market1501/vitbase/model_best.pth.tar \
	--logs-dir logs/adapt/market1501TOmsmt17/vitbase_testtime \
	--rank 8 \
    --lora_alpha 32 \
	#--rr-gpu \

