#!/bin/sh

MASTER_PORT=25388
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port ${MASTER_PORT} \
    source_pretrain_vit.py \
    -ds cuhk03 \
    -dt msmt17 \
    --lintype full \
    --arch vit_base_patch16_224 \
    --seed 0 \
    --margin 0.0 \
    --num-instances 4 \
    -b 64 \
    -j 4 \
    --warmup-step 5 \
    --lr 0.008 \
    --weight-decay 1e-4 \
    --iters 200 \
    --epochs 120 \
	--eval-step 10 \
    --logs-dir logs/pretrain/cuhk03/vitbase

