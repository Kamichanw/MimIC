#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit this block to change the run.
MODE=train
CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME="idefics-9b"
DATASET="vqav2"
NUM_QUERY_SAMPLES=1000
NUM_SHOT=16
CKPT_PATH=""

runname=$(basename "$0" .sh | sed 's/^run_//')
export CUDA_VISIBLE_DEVICES
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
launch_args=(--num_processes "${#devices[@]}")
if (( ${#devices[@]} > 1 )); then
    launch_args+=(--multi_gpu)
fi
train_config=configs/accelerate/zero2.yaml
eval_config=configs/accelerate/multi_gpu.yaml
if (( ${#devices[@]} == 1 )); then
    eval_config=configs/accelerate/single_gpu.yaml
fi

case "$MODE" in
    train)
        accelerate launch --config_file "$train_config" "${launch_args[@]}" train.py \
            runname="$runname" \
            model_name="$MODEL_NAME" \
            data.name="$DATASET" \
            data.num_query_samples="$NUM_QUERY_SAMPLES" \
            data.num_shot="$NUM_SHOT" \
            encoder=mimic \
            peft=mimic
        ;;
    eval)
        if [[ -z "$CKPT_PATH" ]]; then
            echo "Set CKPT_PATH in scripts/run_mimic.sh for eval."
            exit 1
        fi
        accelerate launch --config_file "$eval_config" "${launch_args[@]}" eval.py \
            ckpt_path="$CKPT_PATH" \
            model_name="$MODEL_NAME" \
            lmms_eval.tasks="$DATASET" \
            data.num_shot="$NUM_SHOT" \
            encoder=mimic \
            peft=mimic
        ;;
    *)
        echo "Set MODE to train or eval in scripts/run_mimic.sh."
        exit 1
        ;;
esac
