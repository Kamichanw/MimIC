#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../src/

python pipeline.py \
    -r "$runname-idev2-r-16" \
    -d vqav2,ok_vqa,coco \
    -m idefics2-8b-base \
    -q 8000 \
    -s 0 \
    -tea \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=lora peft=lora batch_size=2 accumulate_grad_batches=8" \
    --eval-args "encoder=lora peft=lora eval.batch_size=8"
