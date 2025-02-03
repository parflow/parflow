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
        start = key.start if key.start is not None else 0
        stop = key.stop + 1 if key.stop is not None else -1
        needs_squeeze = (stop - start) == 1
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
        sl = int(key - np.min(key))
        accessor = {
            "start": int(np.min(key)),
            "stop": int(np.max(key) + 1),
            "indices": slice(sl, sl + 1),
            "squeeze": False,
        }
    return accessor
