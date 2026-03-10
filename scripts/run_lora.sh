#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd "$(dirname "$0")/.."

python scripts/pipeline.py \
    -r "$runname-idev2-r-16" \
    -d vqav2,ok_vqa,coco \
    -m idefics2-8b-base \
    -q 8000 \
    -s 0 \
    --devices 0,1,2,3 \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=lora peft=lora training.batch_size=2 training.accumulate_grad_batches=8" \
    --eval-args "encoder=lora peft=lora eval.batch_size=8"
