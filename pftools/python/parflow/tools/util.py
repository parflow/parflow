import numpy as np
from typing import Iterable, Union


def _check_key_is_empty(key: slice) -> bool:
    """
    Checks if an accessor key is empty.

    :param key:
        An accessor key in the form of a slice object

    :returns:
        A boolean specifying whether the key is empty
    """
    for k in key:
        all_none = np.all([k.start is None, k.stop is None, k.step is None])
        if all_none:
            return True
    return False


def read_start_stop_n(dic: dict, dim: str, default_start, default_stop) -> tuple[int, int, int]:
    dim_dict = dic.get(dim, {}) # Little trick here
    start = dim_dict.get("start", default_start)
    stop = dim_dict.get("stop", default_stop)
    return start, stop, max(stop - start, 1)


def _key_to_explicit_accessor(key: Union[slice, int, Iterable]) -> dict:
    """
    Unifies key accessor types for simpler indexing on xarray datasets.

    :param key:
        The accessor key specifying the elements to select from an array

    :return:
        A dictionary representation of the key with entries for
        'start', 'stop', and 'indices' which is compatible with
        xarray selectors.
    """
    if isinstance(key, slice):
        needs_squeeze = False
        if key.stop is not None and key.start is not None:
            needs_squeeze = (key.stop - key.start) == 1
        accessor = {
            "start": key.start,
            "stop": key.stop,
            "indices": slice(None, None, key.step),
            "squeeze": needs_squeeze,
        }
    elif isinstance(key, int):
        accessor = {
            "start": key,
            "stop": key + 1,
            "indices": slice(0, 1),
            "squeeze": True,
        }
    elif isinstance(key, Iterable):
        accessor = {
            "start": int(np.min(key)),
            "stop": int(np.max(key) + 1),
            "indices": np.array(key) - np.min(key),
            "squeeze": False,
        }
    else:
        raise ValueError(f"Unknown type of key: {key} (type={type(key)})")
    return accessor