import evaluate
from evaluate.utils import launch_gradio_widget
import torch
import torch_npu


module = evaluate.load("Kamichanw/CIDEr")
launch_gradio_widget(module)