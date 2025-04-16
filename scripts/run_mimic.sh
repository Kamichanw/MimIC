runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../src/

python pipeline.py \
    -r "$runname" \
    -d vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 16 \
    -tea \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=mimic peft=mimic" \
    --eval-args "encoder=mimic peft=mimic"
