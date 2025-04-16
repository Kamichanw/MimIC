import torch
import torch_npu
from pathlib import Path
import subprocess
import sys

# Add the parent directory to the Python path
sys.path.insert(0, "..")

# Paths
testbed_dir = str(Path(__file__).parent.parent / "testbed")
result_dir = str(Path(__file__).parent.parent / "results")

# Dataset paths
coco_dir = "/opt/dpcvol/datasets/mscoco2014"
vqav2_dir = "/opt/dpcvol/datasets/vqav2"
ok_vqa_dir = "/opt/dpcvol/datasets/okvqa"
karpathy_coco_caption_dir = "/opt/dpcvol/datasets/8163872184165546974/karpathy-split"
hateful_memes_dir = "/data1/share/dataset/hateful_memes"
flickr30k_dir = karpathy_coco_caption_dir
flickr30k_images_dir = "/data1/share/flickr30k"
ocr_vqa_dir = "/data1/share/dataset/OCR-VQA"
ocr_vqa_images_dir = "/data1/share/dataset/OCR-VQA/images"

# Model paths
idefics_9b_path = "/opt/dpcvol/models/idefics-9b"
llava_interleave_7b_path = "/data1/share/model_weight/llava/llava-interleave-qwen-7b-hf"
idefics2_8b_path = "/data1/share/model_weight/idefics/idefics2-8b"  # Not recommended for ICL
idefics2_8b_base_path = "/opt/dpcvol/models/idefics2-8b-base"

# Debugging information
print("Loading paths.py...")
print(f"vqav2_dir: {vqav2_dir}")
print(f"karpathy_coco_caption_dir: {karpathy_coco_caption_dir}")