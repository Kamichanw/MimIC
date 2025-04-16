import hashlib
from typing import Any, Callable, List, Optional, Union, Dict

import xxhash
import dill
import torch
import torch_npu


class Hasher:
    """Hasher that accepts python objects as inputs."""

    # fmt: off
    algorithms_available = (
        set(hashlib.algorithms_available) 
        | set(xxhash.algorithms_available)
    )
    # fmt: on

    def __init__(self, method):
        self.method = method
        self.m = self.new(method)

    @classmethod
    def new(cls, method):
        def create_xxh_object():
            algorithms = {
                "xxh32": xxhash.xxh32,
                "xxh64": xxhash.xxh64,
                "xxh3_64": xxhash.xxh3_64,
                "xxh128": xxhash.xxh128,
                "xxh3_128": xxhash.xxh3_128,
            }

            return algorithms[method]()

        if method not in cls.algorithms_available:
            raise ValueError(
                f"Unsupported hash algorithm {method}, should be one of {cls.algorithms_available}."
            )

        return (
            hashlib.new(method, usedforsecurity=False)
            if method in hashlib.algorithms_available
            else create_xxh_object()
        )

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]], method) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = cls.new(method)
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any, method: str) -> str:
        return cls.hash_bytes(dill.dumps(value), method)

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value, self.method)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


class Fingerprint:
    def __init__(self, encode_fn: Callable[[Any], Any]) -> None:
        """
        Initialize the Fingerprint object, which is used to manage and validate fingerprints
        (hashes) of encoded samples.

        This class allows you to encode samples, generate fingerprints using a
        hashing algorithm, and store these fingerprints along with the original samples for
        later validation.

        Args:
            encode_fn (Callable[[Any], Any]):
                A function that encodes the input sample into a feature.
                The function accepts any type of input and must return a serializable object.
        """
        self.hash_values: Dict[str, Any] = {}
        self.hash_method = "xxh3_64"
        self.encode_fn = encode_fn

    def __getitem__(self, key: str) -> str:
        return self.hash_values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.hash_values

    def __delitem__(self, key: str) -> None:
        del self.hash_values[key]

    def update(self, key: str, sample: Any) -> None:
        """
        Update the fingerprint for a given key with the provided sample.

        This method encodes the given sample using the specified `encode_fn`, generates a fingerprint,
        and stores the resulting fingerprint along with the original sample under the provided key.

        Args:
            key (str):
                The key under which the fingerprint and sample will be stored.
            sample (Any):
                The sample to be encoded and hashed. The sample can be of any type.
        """
        self.hash_values[key] = {
            "fingerprint": Hasher.hash(self.encode_fn(sample), self.hash_method),
            "sample": sample,
        }

    def validate(self, key: str, sample: Optional[Any] = None) -> bool:
        """
        Validate the fingerprint for a given key against a provided sample or the stored sample.

        If a sample is provided, this method checks whether the fingerprint generated from the provided
        sample (using the stored `encode_fn`) matches the stored fingerprint for the given key.
        If no sample is provided, the method checks whether the stored fingerprint is valid when
        the current `encode_fn` is applied to the originally stored sample.

        Args:
            key (str):
                The key whose associated fingerprint is to be validated.
            sample (Optional[Any]):
                The sample to validate against. If not provided, the stored sample
                associated with the key will be used.

        Returns:
            bool: True if the fingerprint is valid, False otherwise.
        """
        if sample is None:
            sample = self.hash_values[key]["sample"]
        return self.hash_values[key]["fingerprint"] == Hasher.hash(
            self.encode_fn(sample), self.hash_method
        )

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return self.hash_values

    def load_from_dict(self, dct: Dict[str, Dict[str, str]]):
        if not isinstance(dct, dict):
            raise TypeError(
                "Fingerprint should be assigned with a `Dict[str, Dict[str, str]]`"
            )

        required_keys = {"fingerprint", "sample"}

        for key, value in dct.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Expected key of type 'str', but got {type(key).__name__}"
                ).__name__
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected value of type 'dict' for key '{key}', but got {type(value).__name__}"
                )
            if set(value.keys()) != required_keys:
                raise ValueError(
                    f"Dictionary for key '{key}' must be both 'fingerprint' and 'sample' keys"
                )

        self.hash_values = dct


if __name__ == "__main__":

    def encode_fn(sample):
        # may extract features, meta-data ... or encode to anything
        # takes str as an example
        return sample.encode()

    fingerprint = Fingerprint(encode_fn)
    sample = "hello, ICLTestbed"
    fingerprint.update(type(sample).__name__, sample)
    print(fingerprint.to_dict())
