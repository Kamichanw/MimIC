runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../src/

python pipeline.py \
    -r "icl" \
    -d vqav2,coco \
    -m idefics-9b \
    -e \
    -s 0 \
    -q 1000 \
    --requires_memory 20000 \
    --eval-args "ckpt_path=null batch_size=2 iterations=10 resume=False"