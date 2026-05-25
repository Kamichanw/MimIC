import importlib
import inspect
import os

from .interface import DatasetBase


dataset_mapping = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["interface.py", "__init__.py"]:
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)

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
