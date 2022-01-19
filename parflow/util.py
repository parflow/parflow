import numpy as np
from typing import Iterable

def _check_key_is_empty(key):
    for k in key:
        all_none = np.all([k.start is None,
                           k.stop is None,
                           k.step is None])
        if all_none:
            return True
    return False


def _key_to_explicit_accessor(key):
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


