import torch
import torch_npu
import os
import importlib.util
import inspect
from .interface import DatasetBase

# a mapping from dataset name to Dataset class
dataset_mapping = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["interface.py", "__init__.py"]:
        module_name = filename[:-3]
        filepath = os.path.join(os.path.dirname(__file__), filename)

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        DatasetClass = getattr(module, "Dataset", None)
        if DatasetClass is None:
            continue

        if inspect.isclass(DatasetClass) and issubclass(DatasetClass, DatasetBase):
            for dataset in DatasetClass.support_datasets:
                dataset_mapping[dataset] = DatasetClass
        else:
            raise ValueError(
                f"Module {__name__}.{module_name} does not contain a class that named Dataset and inherits from DatasetBase."
            )
            
__all__ = ["dataset_mapping", "DatasetBase"]
