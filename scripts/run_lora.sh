#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit this block to change the run.
MODE=train
CUDA_VISIBLE_DEVICES="0,1,2,3"
DATASETS=(vqav2 ok_vqa coco)
EVAL_DATASET="vqav2"
MODEL_NAME="idefics2-8b-base"
NUM_QUERY_SAMPLES=8000
NUM_SHOT=0
CKPT_PATH=""

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')
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
        for dataset in "${DATASETS[@]}"; do
            accelerate launch --config_file "$train_config" "${launch_args[@]}" train.py \
                runname="$runname-idev2-r-16" \
                model_name="$MODEL_NAME" \
                data.name="$dataset" \
                data.num_query_samples="$NUM_QUERY_SAMPLES" \
                data.num_shot="$NUM_SHOT" \
                encoder=lora \
                peft=lora \
                training_args.per_device_train_batch_size=2 \
                training_args.gradient_accumulation_steps=8
        done
        ;;
    eval)
        if [[ -z "$CKPT_PATH" ]]; then
            echo "Set CKPT_PATH in scripts/run_lora.sh for eval."
            exit 1
        fi
        accelerate launch --config_file "$eval_config" "${launch_args[@]}" eval.py \
            ckpt_path="$CKPT_PATH" \
            model_name="$MODEL_NAME" \
            lmms_eval.tasks="$EVAL_DATASET" \
            data.num_shot="$NUM_SHOT" \
            encoder=lora \
            peft=lora \
            lmms_eval.batch_size=8
        ;;
    *)
        echo "Set MODE to train or eval in scripts/run_lora.sh."
        exit 1
        ;;
esac
