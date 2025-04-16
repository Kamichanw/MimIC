from abc import ABC
from collections import OrderedDict
from functools import partial
from PIL.Image import Image
import inspect
import os
import re
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union, overload
import warnings

import torch
from torch.utils.hooks import RemovableHandle
import torch.nn as nn

from ..utils.tracker import TrackerBase
from ..utils import try_inject_params
import torch
import torch_npu

class ModelBase(nn.Module, ABC):
    def __init__(
        self,
        model_root: str,
        processor_class: type,
        model_class: type,
        support_models: Optional[List[str]] = None,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__()

        processor_args = processor_args if processor_args else dict()
        model_args = model_args if model_args else dict()

        def instantiate(cls: type, **kwargs):
            is_auto_cls = cls.__name__.startswith("Auto")
            trust_remote_code = kwargs.pop("trust_remote_code", is_auto_cls)
            return cls.from_pretrained(  # type: ignore[attr-defined]
                model_root, trust_remote_code=trust_remote_code, **kwargs
            )

        self.processor = instantiate(
            processor_class, **{**processor_args, **common_args}
        )
        self.model = instantiate(model_class, **{**model_args, **common_args})

        self.model_name = os.path.basename(model_root)
        if support_models is not None and self.model_name not in support_models:
            warnings.warn(
                "The model name cannot be detected automatically in `model_root`, which may lead to unexpected behaviors. "
                f"make sure basename of model root is in {', '.join(support_models)}."
            )

        self.config = self.model.config
        self.prompt_template = None
        self._trackers_dict: OrderedDict = OrderedDict()

        def tracker_hook(m, args):
            for tracker in self._trackers_dict.values():
                tracker.incre_next_index()

        self.model.register_forward_pre_hook(tracker_hook, prepend=True)

    def _register_hook(self, register_fn_name, module_name, hook, **kwargs):
        if isinstance(module_name, str):
            # Use regex to match module names
            return [
                getattr(module, register_fn_name)(
                    try_inject_params(hook, module_name=name), **kwargs
                )
                for name, module in self.model.named_modules()
                if re.search(module_name, name)
            ]

        elif isinstance(module_name, list):
            # Exact match for each module name in the list
            return [
                getattr(module, register_fn_name)(
                    try_inject_params(hook, module_name=name), **kwargs
                )
                for name, module in self.model.named_modules()
                if name in module_name
            ]
        else:
            raise TypeError(
                f"module_name should be str or list of str, but got {type(module_name).__name__}"
            )

    def add_tracker(self, module_name: Union[str, List[str]], tracker: TrackerBase):
        """
        Add a tracker for modules specified by `target`.

        Args:
            module_name (str or List[str]):
                If str, then call add_tracker for the module named `module_name` using regex matching.
                If List[str], then add_tracker is called for each named `module_name` using exact matching.
            tracker (TrackerBase):
                A tracker to add.
        """
        if isinstance(module_name, str):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if re.search(module_name, name)
            }
        elif isinstance(module_name, list):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if name in module_name
            }
        else:
            matched_modules = None

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name}")

        tracker.track(list(matched_modules.values()), self._trackers_dict)
        tracker.auto_incre_index = False
        self._trackers_dict[tracker.id] = tracker
        for name, status in zip(
            matched_modules.keys(), tracker._module_refs_dict.values()
        ):
            status.module_name = name

    @overload  # type: ignore
    def register_forward_hook(
        self,
        module_name: Union[str, List[str]],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a forward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        for details. The hook will be called every time after forward() has computed an output.

        Args:
            module_name (str or List[str]):
                If str, then call register_forward_hook for the module named `module_name` using regex matching.
                If List[str], then register_forward_hook is called for each named `module_name` using exact matching.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args, output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload  # type: ignore
    def register_foward_hook(
        self,
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        """
        Register a forward hook on this model forward, same as standard one.
        """
        ...

    def register_forward_hook(  # type: ignore
        self,
        *args,
        prepend=False,
        with_kwargs=False,
        always_call=False,
    ):
        if callable(args[0]):
            return super().register_forward_hook(
                *args, prepend=prepend, with_kwargs=with_kwargs, always_call=always_call
            )
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_forward_hook",
                *args,
                prepend=prepend,
                with_kwargs=with_kwargs,
                always_call=always_call,
            )

    @overload
    def register_forward_pre_hook(
        self,
        module_name: Union[str, List[str]],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a forward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        for details. The hook will be called every time before forward() is invoked.

        Args:
            module_name (str, List[str], or HookType):
                If str, then call register_forward_pre_hook for the module named `module_name` using regex matching.
                If List[str], then register_forward_pre_hook is called for each named `module_name` using exact matching.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_forward_pre_hook(
        self, hook: Callable, *, prepend: bool = False, with_kwargs: bool = False
    ) -> RemovableHandle:
        """
        Register a forward pre-hook on this model forward, same as standard one.
        """
        ...

    def register_forward_pre_hook(self, *args, prepend=False, with_kwargs=False):
        if callable(args[0]):
            return super().register_forward_pre_hook(
                *args, prepend=prepend, with_kwargs=with_kwargs
            )
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_forward_pre_hook",
                *args,
                prepend=prepend,
                with_kwargs=with_kwargs,
            )

    @overload
    def register_full_backward_hook(
        self,
        module_name: Union[str, List[str]],
        hook: Callable,
        prepend: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a full backward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
        for details. The hook will be called every time the gradients with respect to a module are computed.

        Args:
            module_name (str, List[str], or HookType):
                If str, then call register_full_backward_hook for the module named `module_name` using regex matching.
                If List[str], then register_full_backward_hook is called for each named `module_name` using exact matching.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_input, grad_output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_full_backward_hook(
        self, hook: Callable, prepend: bool = False
    ) -> RemovableHandle:
        """
        Register a full_backward hook on this model, same as standard one.
        """
        ...

    def register_full_backward_hook(self, *args, prepend=False):
        if callable(args[0]):
            return super().register_full_backward_hook(args[0], prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_hook", *args, prepend=prepend
            )

    @overload
    def register_full_backward_pre_hook(
        self,
        module_name: Union[str, List[str]],
        hook: Callable,
        prepend: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a full backward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
        for details. The hook will be called every time before the gradients with respect to a module are computed.

        Args:
            module_name (str, List[str], or HookType):
                If str, then call register_full_backward_pre_hook for the module named `module_name` using regex matching.
                If List[str], then register_full_backward_pre_hook is called for each named `module_name` using exact matching.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_full_backward_pre_hook(
        self,
        hook: Callable,
        prepend: bool = False,
    ) -> RemovableHandle:
        """
        Register a full_backward pre-hook on this model, same as standard one.
        """
        ...

    def register_full_backward_pre_hook(self, *args, prepend=False):
        if callable(args[0]):
            return super().register_full_backward_pre_hook(args[0], prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_pre_hook", *args, prepend=prepend
            )

    @property
    def default_prompt_template(self) -> str:
        if hasattr(self.processor, "get_chat_template"):
            return self.processor.get_chat_template()
        if hasattr(self.processor.tokenizer, "get_chat_template"):
            return self.processor.tokenizer.get_chat_template()
        else:
            raise RuntimeError("No default prompt template found.")

    @property
    def device(self):
        return self.model.device

    def process_input(
        self,
        images: Union[List[Image], List[List[Image]]],
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            prompt_template (str, optional):
                A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)  # type: ignore[arg-type]

        return self.processor(
            images=images,
            text=text,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        *inputs,
        processor_args: Optional[Dict[str, Any]] = None,
        return_inputs: bool = False,
        return_generated_ids: bool = False,
        **generate_args,
    ):
        """
        Generates text using the model based on the provided inputs.

        Args:
            *inputs:
                Inputs that are further fed into `process_input`, see `process_input` docs for details.
            processor_args (Dict[str, Any], optional):
                Additional arguments for the `process_input` method. Defaults to None.
            return_inputs (bool, optional):
                Whether to include the processed inputs in the output dictionary. Defaults to False.
            return_generated_ids (bool, optional):
                Whether to include the generated IDs in the output dictionary. Defaults to False.
            **generate_args:
                Additional arguments to pass to the `generate` method of the model.

        Returns:

            Dict[str, Any]: A dictionary containing:
                - 'outputs': The decoded generated text sequences.
                - 'inputs' (optional): The processed inputs if `return_inputs` is True.
                - 'generated_ids' (optional): The generated token IDs if `return_generated_ids` is True.
        """
        processor_args = processor_args if processor_args else dict()

        inputs = self.process_input(*inputs, **processor_args).to(
            device=self.device, dtype=self.model.dtype
        )
        seq_len = inputs.input_ids.shape[-1]  # type: ignore[attr-defined]

        generated_ids = self.model.generate(**inputs, **generate_args)
        generated_ids = generated_ids[:, seq_len:]

        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if return_inputs == False and return_generated_ids == False:
            return outputs

        result = {"outputs": outputs}
        if return_inputs:
            result["inputs"] = inputs
        if return_generated_ids:
            result["generated_ids"] = generated_ids

        return result

    def forward(
        self,
        *processor_input,
        processor_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        processor_args = processor_args if processor_args else dict()
        inputs = self.process_input(*processor_input, **processor_args).to(self.device)
        return self.model(**inputs, **kwargs)

    def apply_prompt_template(
        self,
        conversation: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        prompt_template: Optional[str] = None,
        tokenize=False,
        **kwargs,
    ):
        if prompt_template is None:
            prompt_template = (
                self.default_prompt_template
                if self.prompt_template is None
                else self.prompt_template
            )

        apply_fn = (
            self.processor.apply_chat_template
            if hasattr(self.processor, "apply_chat_template")
            else self.processor.tokenizer.apply_chat_template
        )
        apply_fn = partial(
            apply_fn, chat_template=prompt_template, tokenize=tokenize, **kwargs
        )

        if isinstance(conversation[0], list):
            return [apply_fn(con) for con in conversation]
        else:
            return apply_fn(conversation)

    @overload
    def replace_module(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        new_module_cls: nn.Module,
        *,
        strict: bool = True,
        **init_args,
    ):
        """
        Replace modules in the model by matching their names or types, using either regex
        for string input or exact string matching for a list of strings.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_cls (nn.Module):
                The new module class to replace the matched module(s).
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.
            **init_args:
                Arguments to initialize the new module.

        Raises:
            ValueError:
                If no matching modules are found, or if the new module's forward method
                has incompatible parameter names with the original module.
        """
        ...

    @overload
    def replace_module(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        new_module_instances: Union[nn.Module, List[nn.Module]],
        *,
        strict: bool = True,
    ):
        """
        Replace specific instances of modules in the model by matching their names or types.
        If `module_name_or_type` is a string, regex matching is used. If it is a list of strings,
        exact string matching is performed.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_instances (Union[nn.Module, List[nn.Module]]):
                New module instance(s) to replace the matched modules.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.

        Raises:
            ValueError:
                If the number of matched modules doesn't match the number of provided instances,
                or if the new module's forward method has incompatible parameter names with the original module.
        """
        ...

    def replace_module(
        self,
        module_name_or_type,
        new_module_cls_or_instances,
        *,
        strict=True,
        **init_args,
    ):
        """
        Replace modules in the model based on names or types, with either new classes or specific instances.
        Supports matching by regex when `module_name_or_type` is a string, or exact string matching when
        it is a list of strings. It checks whether the `forward` method of the new module has compatible
        parameter names with the original module's `forward` method.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_cls_or_instances (nn.Module or List[nn.Module]):
                A new module class or instance(s) to replace the matched modules.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.
            **init_args:
                Additional arguments to initialize the new module if providing a class.

        Raises:
            ValueError:
                If the number of matched modules does not match the number of new instances provided,
                or if the new module's forward method has incompatible parameter names with the original module.
        """

        def replace_module_by_name(name, orig_module, new_module):
            if strict:
                orig_params = list(
                    inspect.signature(orig_module.forward).parameters.keys()
                )
                new_params = list(
                    inspect.signature(new_module.forward).parameters.keys()
                )

                if orig_params != new_params[: len(orig_params)]:
                    missing_params = set(orig_params) - set(
                        new_params[: len(orig_params)]
                    )
                    unexpected_params = set(new_params[: len(orig_params)]) - set(
                        orig_params
                    )
                    raise ValueError(
                        "The first few parameters of the new module's forward method do not match "
                        "the original module's, which may lead unexpected behaviors."
                        f"{', '.join(missing_params)} are missing, {', '.join(unexpected_params)} are unexpected."
                        "If you want to add new parameters, they should be at the end of the parameter list,"
                        "or set strict=False to avoid this error."
                    )

            *parent_module_names, last_name = name.split(".")
            parent_module = self.model
            for pname in parent_module_names:
                parent_module = getattr(parent_module, pname)
            setattr(parent_module, last_name, new_module)

        if isinstance(module_name_or_type, str):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if re.search(module_name_or_type, name)
            }
        elif isinstance(module_name_or_type, list):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if name in module_name_or_type
            }
        elif isinstance(module_name_or_type, type):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if isinstance(module, module_name_or_type)
            }
        else:
            matched_modules = None

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name_or_type}")

        if isinstance(new_module_cls_or_instances, type):
            for name, module in matched_modules.items():
                replace_module_by_name(
                    name, module, new_module_cls_or_instances(**init_args)
                )
        else:
            if isinstance(new_module_cls_or_instances, list):
                if len(matched_modules) != len(new_module_cls_or_instances):
                    raise ValueError(
                        f"Number of matched modules ({len(matched_modules)}) does not match the number of provided instances ({len(new_module_cls_or_instances)})."
                    )
                for (name, module), new_instance in zip(
                    matched_modules.items(), new_module_cls_or_instances
                ):
                    replace_module_by_name(name, module, new_instance)
            else:
                if len(matched_modules) != 1:
                    raise ValueError(
                        "When replacing with a single instance, only one module should be matched."
                    )
                replace_module_by_name(
                    next(iter(matched_modules)),
                    matched_modules[next(iter(matched_modules))],
                    new_module_cls_or_instances,
                )

    def replace_module_method(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        method_name: str,
        new_method: Callable,
        *,
        strict: bool = True,
    ):
        """
        Replace a method of modules in the model based on names or types, with a new function.
        Optionally checks whether the new function's signature is compatible with the old method's signature.
        The new function will have `module_name` and `old_method` injected as keyword arguments, if possible.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module whose method is to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            method_name (str):
                The name of the method to replace (e.g., 'forward').
            new_method (Callable):
                A new function to replace the matched method.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.

        Raises:
            ValueError:
                If no module matches the given name or type, or if the new method's signature is incompatible
                with the old method's signature when strict is True.
        """
        if isinstance(module_name_or_type, str):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if re.search(module_name_or_type, name)
            }
        elif isinstance(module_name_or_type, list):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if name in module_name_or_type
            }
        elif isinstance(module_name_or_type, type):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if isinstance(module, module_name_or_type)
            }
        else:
            matched_modules = None

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name_or_type}")

        for name, module in matched_modules.items():
            old_method = getattr(module, method_name, None)
            if old_method is None:
                raise ValueError(f"Module '{name}' has no method '{method_name}'")

            if strict:
                orig_params = list(inspect.signature(old_method).parameters.keys())
                new_params = list(inspect.signature(new_method).parameters.keys())

                if "self" not in orig_params and isinstance(old_method, MethodType):
                    orig_params.insert(0, "self")

                if orig_params != new_params[: len(orig_params)]:
                    missing_params = set(orig_params) - set(
                        new_params[: len(orig_params)]
                    )
                    unexpected_params = set(new_params[: len(orig_params)]) - set(
                        orig_params
                    )
                    raise ValueError(
                        "The first few parameters of the new function do not match "
                        f"the original method '{method_name}' of module '{name}', which may lead unexpected behaviors."
                        f"{', '.join(missing_params)} are missing, {', '.join(unexpected_params)} are unexpected."
                        "If you want to add new parameters, they should be at the end of the parameter list,"
                        "or set strict=False to avoid this error."
                    )

            setattr(
                module,
                method_name,
                MethodType(
                    try_inject_params(
                        new_method, module_name=name, old_method=old_method
                    ),
                    module,
                ),
            )
