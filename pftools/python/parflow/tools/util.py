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
        all_none = np.all([k.start is None,
                           k.stop is None,
                           k.step is None])
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
        return {
            'start': key.start,
            'stop': key.stop,
            'indices': slice(None, None, key.step)
        }
    elif isinstance(key, int):
        return {
            'start': key,
            'stop': key+1,
            'indices': [0]
        }
    elif isinstance(key, Iterable):
        return {
            'start': np.min(key),
            'stop': np.max(key)+1,
            'indices': key - np.min(key)
        }

