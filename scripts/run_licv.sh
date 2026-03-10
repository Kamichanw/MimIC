#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd "$(dirname "$0")/.."

python scripts/pipeline.py \
    -r "$runname" \
    -d vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    -a \
    --requires_memory 20000 \
    --train-args "encoder=licv peft=licv" \
    --eval-args "encoder=licv peft=licv"