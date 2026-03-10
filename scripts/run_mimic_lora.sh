#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd "$(dirname "$0")/.."

python scripts/pipeline.py \
    -r "$runname-r-1" \
    -d vqav2,ok_vqa,coco \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    --train-args "encoder=mimic peft=mini_lora training.ce_loss_weight=0.5 peft.r=1 training.lr=5e-3" \
    --eval-args "encoder=mimic peft=mini_lora peft.r=1"
