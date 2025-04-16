import inspect
from typing import Any, Union
import torch
import copy
from collections.abc import Mapping, Sequence

from functools import partial
from collections.abc import Mapping, Sequence


def try_inject_params(fn, **kwargs):
    """
    Try to inject positional arguments to fn, if applicable.

    This function checks whether the function accepts keyword arguments (**kwargs) or
    explicitly defined arguments that are present in `kwargs`. If the function supports **kwargs,
    all arguments in `kwargs` are passed to the fn. Otherwise, only the arguments explicitly
    defined in the function's signature and present in `kwargs` are passed. If no matching
    arguments exist, the fn is returned unchanged.

    Args:
        fn (Callable): The function to be injected.
        **kwargs: Additional arguments that may be passed to the fn.

    Returns:
        Callable: A partially applied function if `kwargs` contains matching arguments;
                  otherwise, the original function.

    Example:
        If the function supports `**kwargs` or specific arguments from `kwargs`:

        >>> def hook_fn(module, input, output, module_name):
        >>>     print(f"Module name: {module_name}")

        This method will pass the `module_name` argument from `kwargs` to `fn`.
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
    Recursively traverse an object, clone it (if not inplace), and move all torch.Tensor attributes to a specified device.

    Args:
        obj (Any):
            The object to process. Can be a dict, list, tuple, set, or a custom class instance.
        device (Union[str, torch.device], optional):
            The device to move the tensors to. Default is 'cpu'.
        check_cycles (bool, optional):
            Whether to check for cycles during the traversal to prevent infinite recursion. Default is False.
        max_recur_depth (int, optional):
            Maximum recursion depth. If reached, further recursion will be skipped. Default is 1.

    Returns:
        Any:
            The processed object with all tensors moved to the specified device. If `inplace=True`, the original object is modified.
            If `inplace=False`, a new object is returned.
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

        elif hasattr(obj, "__slots__"):
            cloned_obj = copy.copy(obj)
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    setattr(
                        cloned_obj,
                        slot,
                        move_to_device(getattr(obj, slot), visited, depth + 1),
                    )
            return cloned_obj

        return obj

    return move_to_device(obj, set() if check_cycles else None, 0)
