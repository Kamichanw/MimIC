#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd "$(dirname "$0")/.."

python scripts/pipeline.py \
    -r "$runname" \
    -d vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 16 \
    --devices 0,1,2,3 \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=mimic peft=mimic" \
    --eval-args "encoder=mimic peft=mimic"
