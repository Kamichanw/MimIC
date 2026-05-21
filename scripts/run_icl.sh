#!/bin/bash
set -euo pipefail

# Edit this block to change the run.
CUDA_VISIBLE_DEVICES="0,1,2,3"
DATASETS=(vqav2)
MODEL_NAME="idefics-9b"
NUM_SHOT=0
BATCH_SIZE=2
LIMIT=10

export CUDA_VISIBLE_DEVICES
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
launcher=(conda run -n mimic accelerate)

launch_args=(--num_processes "${#devices[@]}")
if (( ${#devices[@]} > 1 )); then
    launch_args+=(--multi_gpu)
fi
eval_config=configs/accelerate/single_gpu.yaml
if (( ${#devices[@]} > 1 )); then
    eval_config=configs/accelerate/multi_gpu.yaml
fi

for dataset in "${DATASETS[@]}"; do
    "${launcher[@]}" launch --config_file "$eval_config" "${launch_args[@]}" eval.py \
        ckpt_path=null \
        model_name="$MODEL_NAME" \
        lmms_eval.tasks="$dataset" \
        data.num_shot="$NUM_SHOT" \
        lmms_eval.batch_size="$BATCH_SIZE" \
        lmms_eval.limit="$LIMIT"
done
