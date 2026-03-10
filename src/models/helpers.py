import inspect
from typing import Any, Union
import torch
import copy
from collections.abc import Mapping, Sequence

from functools import partial


def try_inject_params(fn, **kwargs):
    """
    Try to inject positional arguments to fn, if applicable.
    """
    signature = inspect.signature(fn)
    supports_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    filtered_args = (
        {k: v for k, v in kwargs.items() if k in signature.parameters}
        if not supports_kwargs
        else kwargs
    )

    if filtered_args:
        return partial(fn, **filtered_args)
    return fn


def clone_to_device(
    obj: Any,
    device: Union[str, torch.device] = "cpu",
    check_cycles: bool = False,
    max_recur_depth: int = 1,
):
    """
    Recursively clone nested objects and move contained tensors to a device.
    """

    def move_to_device(obj, visited, depth):
        if check_cycles:
            if id(obj) in visited or depth > max_recur_depth:
                return obj
            visited.add(id(obj))
        if depth > max_recur_depth:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(device)

        elif isinstance(obj, Mapping):
            return {
                key: move_to_device(value, visited, depth + 1)
                for key, value in obj.items()
            }

        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return type(obj)(move_to_device(item, visited, depth + 1) for item in obj)  # type: ignore[call-arg]

        elif hasattr(obj, "__dict__"):
            cloned_obj = copy.copy(obj)
            for attr_name, attr_value in vars(obj).items():
                setattr(
                    cloned_obj,
                    attr_name,
                    move_to_device(attr_value, visited, depth + 1),
                )
            return cloned_obj

        elif getattr(obj, "__slots__", None) is not None:
            cloned_obj = copy.copy(obj)
            for slot in getattr(obj, "__slots__", ()):
                if hasattr(obj, slot):
                    setattr(
                        cloned_obj,
                        slot,
                        move_to_device(getattr(obj, slot), visited, depth + 1),
                    )
            return cloned_obj

        return obj

    return move_to_device(obj, set() if check_cycles else None, 0)
