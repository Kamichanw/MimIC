import torch
import torch.nn as nn
import weakref

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from types import FunctionType, MethodType
from torch.utils.hooks import RemovableHandle
from functools import partial, wraps
from bytecode import Bytecode, Instr

from testbed.utils import clone_to_device
import torch
import torch_npu

__all__ = ["ForwardTracker", "GradTracker", "LocalsTracker"]


@dataclass
class ModuleStatus:
    module_name: Optional[str]
    accessed: Optional[bool]


class TrackerBase(ABC):
    """
    The base class of all trackers.

    Args:
        data_props (list of str):
            A list of data properties to be tracked. Each property will be added to `_data` keys
            and registered as a property of the tracker.
        on_device (bool, str, torch.device):
            The device on which to store the data. If `True`, the data will be stored on the same device
            as the module. If `False`, the data will be stored on the CPU. If a string or a `torch.device`,
            the data will be stored on the specified device. Defaults to `False`.
    """

    def __init__(
        self, data_props: List[str], on_device: Union[bool, str, torch.device]
    ):
        super().__init__()
        self._auto_incre_index: bool = True
        self._data: Dict[str, List[List]] = {key: [] for key in data_props}

        if isinstance(on_device, bool):
            # if to_device is false, the data will be stored on the CPU
            # otherwise, the data will be stored on the same device as the module
            self.to_device: Optional[Union[str, torch.device]] = (
                "cpu" if on_device == False else None
            )
        else:
            self.to_device = on_device

        # the following variables need to be delete when removing
        self._next_index: int
        self._module_refs_dict: weakref.WeakKeyDictionary  # module refs to ModuleStatus
        self._hook_handles: Tuple
        self._myhandle: RemovableHandle

        def data_prop_getter(instance, key):
            if not instance.is_tracking:
                raise RuntimeError(
                    f"Attempting to get {key} from the tracker that was not attached to any model."
                )
            return instance._data[key]

        for key in self._data.keys():
            setattr(type(self), key, property(partial(data_prop_getter, key=key)))

    @property
    def is_tracking(self):
        return hasattr(self, "_module_refs_dict")

    @property
    def auto_incre_index(self):
        """
        Whether to automatically increase next_index. If `True`, when a tracked module is accessed
        for the second time, the data will be added to the position pointed to by `next_index` in data
        in the form of a list. If `False`, the user should manage `next_index` manually. Defaults to `True`.
        """
        return self._auto_incre_index

    @auto_incre_index.setter
    def auto_incre_index(self, value):
        if self.auto_incre_index != value:
            if self.is_tracking:
                for status in self._module_refs_dict.values():
                    status.accessed = True if value else None
            self._auto_incre_index = value

    @property
    def next_index(self):
        """
        The next index to place new data. It can only be incremented, and can only be increased by one at a time.
        When the tracker starts to track, `next_index` will be set to -1.
        """
        return self._next_index

    def incre_next_index(self):
        """
        Increase `next_index` by 1. It will raise an exception if `auto_incre_index` is `True`.
        """

        if self.auto_incre_index:
            raise RuntimeError(
                "next_index cannot be increased manually, since auto_incre_index is true."
            )
        self._next_index += 1
        for key in self._data:
            self._data[key].append([])

    def __enter__(self) -> "TrackerBase":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()

    @abstractmethod
    def _register_tracker(self, module: nn.Module):
        raise NotImplementedError()

    def _hook_wrapper(self, hook):
        """
        Wrap the hook to increase `next_index` automatically.
        """

        @wraps(hook)
        def wrapper(instance, *args, **kwargs):
            if self.auto_incre_index:
                if self._module_refs_dict[instance].accessed:
                    self._next_index += 1
                    for key in self._data:
                        self._data[key].append([])
                    for status in self._module_refs_dict.values():
                        status.accessed = False

                self._module_refs_dict[instance].accessed = True
            return hook(instance, *args, **kwargs)

        return wrapper

    def track(
        self,
        modules: List[nn.Module],
        trackers_dict: Dict[int, "TrackerBase"],
        extra_dict: Optional[Union[Dict[int, Any], List[Dict[int, Any]]]] = None,
    ):
        """
        Track a list of modules.

        Args:
            modules (List[nn.Module]):
                A list of modules to track.
            trackers_dict (Dict[int, TrackerBase, *optional*):
                A dictionary of trackers, indexed by tracker ``id``.
            extra_dict (Dict[int, Any] or List[Dict[int, Any]], *optional*): An additional dictionary or list of
                dictionaries whose keys will be deleted when the same keys are
                removed from ``trackers_dict``.
        """
        if not isinstance(modules, list) or not isinstance(modules[0], nn.Module):
            raise TypeError(
                f"modules should be a list of nn.Module, but got {type(modules)}"
            )

        self.remove()
        self._next_index = -1
        self._module_refs_dict = weakref.WeakKeyDictionary(
            {
                # module_name will be assigned when the tracker is attached to a model
                # accessed is set to True, because this will trigger the increment of next_index on the first trace.
                m: ModuleStatus(
                    module_name=None, accessed=True if self.auto_incre_index else None
                )
                for m in modules
            }
        )
        self._hook_handles = tuple(self._register_tracker(m) for m in modules)  # type: ignore[assignment]
        self._myhandle = RemovableHandle(trackers_dict, extra_dict=extra_dict)

    @property
    def id(self) -> Optional[int]:
        return self._myhandle.id if hasattr(self, "_myhandle") else None

    def clear(self):
        for v in self._data.values():
            v.clear()
        for status in self._module_refs_dict.values():
            status.accessed = True if self.auto_incre_index else None
        self._next_index = -1

    def remove(self) -> None:
        if not self.is_tracking:
            return

        for v in self._data.values():
            v.clear()
        for h in self._hook_handles:
            h.remove()
        self._myhandle.remove()

        del self._next_index
        del self._hook_handles
        del self._myhandle
        del self._module_refs_dict


class ForwardTracker(TrackerBase):
    """
    A tracker for monitoring the forward pass of specific modules in a model.

    This tracker automatically appends the outputs of specified modules during a single forward
    pass to the `inputs` and `outputs` attribute as a list. The length of this list corresponds
    to the number of specified modules that were passed through during the forward pass.

    Args:
        on_device (bool, str, torch.device):
                The device on which to store the data. If `True`, the data will be stored on the same device
                as the module. If `False`, the data will be stored on the CPU. If a string or a `torch.device`,
                the data will be stored on the specified device. Defaults to `False`.
    """

    def __init__(self, on_device: Union[bool, str, torch.device] = False):
        super().__init__(["inputs", "outputs"], on_device=on_device)

        # for type hint. actual properties are dynamically created in super().__init__
        self.inputs: Optional[List[List]]
        self.outputs: Optional[List[List]]

    def _register_tracker(self, module: nn.Module) -> RemovableHandle:
        @self._hook_wrapper
        def hook(m, args, output):
            self._data["inputs"][self.next_index].append(
                clone_to_device(args, device=self.to_device)
                if self.to_device is not None
                else args
            )
            self._data["outputs"][self.next_index].append(
                clone_to_device(output, device=self.to_device)
                if self.to_device is not None
                else args
            )

        return module.register_forward_hook(hook)


class GradTracker(TrackerBase):
    """
    A tracker for monitoring the gradients during the backward pass of specific modules in a model.

    This tracker automatically appends the gradients of the specified modules during the backward pass
    to the `grad_inputs` and `grad_outputs` attribute as a list. The length of this list corresponds to
    the number of modules whose gradients were tracked during the backward pass.

    Args:
        on_device (bool, str, torch.device):
                The device on which to store the data. If `True`, the data will be stored on the same device
                as the module. If `False`, the data will be stored on the CPU. If a string or a `torch.device`,
                the data will be stored on the specified device. Defaults to `False`.
    """

    def __init__(self, on_device: Union[bool, str, torch.device] = False):
        super().__init__(["grad_inputs", "grad_outputs"], on_device=on_device)

        # for type hint. actual properties are dynamically created in super().__init__
        self.grad_inputs: Optional[List[List]]
        self.grad_outputs: Optional[List[List]]

    def _register_tracker(self, module: nn.Module) -> RemovableHandle:
        @self._hook_wrapper
        def hook(m, grad_input, grad_output):
            self._data["grad_inputs"][self.next_index].append(
                clone_to_device(grad_input, device=self.to_device)
                if self.to_device
                else grad_input
            )
            self._data["grad_outputs"][self.next_index].append(
                clone_to_device(grad_output, device=self.to_device)
                if self.to_device
                else grad_output
            )

        return module.register_backward_hook(hook)


class LocalsTracker(TrackerBase):
    """
    A tracker for monitoring local variables in a specific method of modules during a forward pass.

    This tracker automatically appends the values of specified local variables during a forward
    pass to the `data` attribute as a list.

    Args:
        method_name (str):
            The name of the method to track.
        varnames (List[str]):
            A list of local variable names to track.
        on_device (bool, str, torch.device):
            The device on which to store the data. If `True`, the data will be stored on the same device
            as the module. If `False`, the data will be stored on the CPU. If a string or a `torch.device`,
            the data will be stored on the specified device. Defaults to `False`.
    """

    class ReplaceMethodHandle:
        def __init__(self, module: nn.Module, method_name: str, new_method: Callable):
            self.module = module
            self.method_name = method_name
            self.original_method = getattr(module, method_name)
            setattr(module, method_name, MethodType(new_method, self.module))

        def remove(self):
            setattr(self.module, self.method_name, self.original_method)

    def __init__(
        self,
        method_name: str,
        varnames: List[str],
        on_device: Union[bool, str, torch.device] = False,
    ):
        super().__init__([], on_device=on_device)
        self._data = {key: [] for key in varnames}
        self.method_name = method_name

    def get(self, key: str):
        if key not in self._data.keys():
            raise KeyError(f"{key} is not in data keys")
        if not self.is_tracking:
            raise RuntimeError(
                f"Attempting to get {key} from the tracker that was not attached to any model."
            )
        return self._data[key]

    def _register_tracker(self, module: nn.Module) -> ReplaceMethodHandle:
        def inject_wrapper(fn):
            code = Bytecode.from_code(fn.__code__)
            code[-1:-1] = [
                Instr("STORE_FAST", "__retval"),
                *[
                    instruction
                    for key in self._data.keys()
                    for instruction in [
                        Instr("LOAD_FAST", key),
                        Instr("STORE_FAST", f"__var_{key}"),
                    ]
                ],
                Instr("LOAD_FAST", "__retval"),
                *[Instr("LOAD_FAST", f"__var_{key}") for key in self._data.keys()],
                Instr("BUILD_TUPLE", 1 + len(self._data)),
                Instr("STORE_FAST", "__ret_tuple"),
                Instr("LOAD_FAST", "__ret_tuple"),
            ]

            new_fn = FunctionType(
                code=code.to_code(),
                globals=fn.__globals__,
                name=fn.__name__,
                argdefs=fn.__defaults__,
                closure=fn.__closure__,
            )
            new_method = MethodType(new_fn, module)

            @wraps(fn)
            @self._hook_wrapper
            def wrapper(instance, *args, **kwargs):
                retval, *vars = new_method(*args, **kwargs)
                for key, value in zip(self._data.keys(), vars):
                    self._data[key][self.next_index].append(
                        clone_to_device(value, device=self.to_device)
                        if self.to_device
                        else value
                    )
                return retval

            return wrapper

        fn = getattr(module, self.method_name)

        if not callable(fn):
            raise ValueError(
                f"{type(module).__name__}.{self.method_name} should be a callable object"
            )

        if hasattr(fn, "__func__"):
            fn = fn.__func__

        local_vars = fn.__code__.co_varnames
        missing_vars = [key for key in self._data.keys() if key not in local_vars]
        if missing_vars:
            raise ValueError(
                f"Missing required local variables in {type(module).__name__}.{self.method_name}: {', '.join(missing_vars)}"
            )

        return LocalsTracker.ReplaceMethodHandle(
            module, self.method_name, inject_wrapper(fn)
        )
