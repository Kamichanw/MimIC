import evaluate
from evaluate.utils import launch_gradio_widget
import torch
import torch_npu

module = evaluate.load("Kamichanw/vqa_accuracy")
launch_gradio_widget(module)