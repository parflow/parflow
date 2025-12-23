from functools import wraps
import os
import re

from .fs import get_absolute_path

# -----------------------------------------------------------------------------
# Map function Helper functions
# -----------------------------------------------------------------------------


def map_to_parent(pfdb_obj):
    """Helper function to extract the parent of a pfdb_obj"""
    return pfdb_obj._parent_


# -----------------------------------------------------------------------------


def map_to_self(pfdb_obj):
    """Helper function to extract self of self (noop)"""
    return pfdb_obj


# -----------------------------------------------------------------------------


def map_to_child(name):
    """Helper function that return a function for extracting a field name"""
    return lambda pfdb_obj: pfdb_obj.__getitem__(name)


# -----------------------------------------------------------------------------


def map_to_children_of_type(class_name):
    """Helper function that return a function for extracting children
    of a given type (class_name).
    """
    return lambda pfdb_obj: pfdb_obj.get_children_of_type(class_name)


# -----------------------------------------------------------------------------
# Filter helpers
# -----------------------------------------------------------------------------


def filter_none(x):
    return x is not None


# -----------------------------------------------------------------------------
# Key dictionary helpers
# -----------------------------------------------------------------------------


def get_key_priority(key_name):
    """Return number that can be used to sort keys in term of priority"""
    priority_value = 0
    path_token = key_name.split(".")
    if "Name" in key_name:
        priority_value -= 100

    for token in path_token:
        if token[0].isupper():
            priority_value += 1
        else:
            priority_value += 10

    priority_value *= 100
    priority_value += len(key_name)

    return priority_value


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Sort helpers
# -----------------------------------------------------------------------------


def sort_dict(d, key=None):
    """Create a key sorted dict"""
    return {k: d[k] for k in sorted(d, key=key)}


# -----------------------------------------------------------------------------


def sort_dict_by_priority(d):
    """Create a key sorted dict"""
    return sort_dict(d, key=get_key_priority)


# -----------------------------------------------------------------------------
# Dictionary helpers
# -----------------------------------------------------------------------------


def get_or_create_dict(root, key_path, overriden_keys):
    """Helper function to get/create a container dict for a given key path"""
    current_container = root
    for i, key in enumerate(key_path):
        if key not in current_container:
            current_container[key] = {}
        elif not isinstance(current_container[key], dict):
            overriden_keys[".".join(key_path[: i + 1])] = current_container[key]
            current_container[key] = {}
        current_container = current_container[key]

    return current_container


# -----------------------------------------------------------------------------
# String helpers
# -----------------------------------------------------------------------------


def remove_prefix(s, prefix):
    if not s or not prefix or not s.startswith(prefix):
        return s

    return s[len(prefix) :]


# First column is the regex. Second column is the replacement.
INVALID_DOT_REGEX_SUBS = [
    (re.compile(r"^\.([a-zA-Z]+)"), "\\1"),
    (re.compile(r"([a-zA-Z]+)\."), "\\1/"),
]


def _normalize_location(s):
    s = os.path.normpath(s)
    for regex, sub in INVALID_DOT_REGEX_SUBS:
        s = re.sub(regex, sub, s)
    return s


# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------


def normalize_location(func):
    """Assume the first string argument is location and normalize it.

    Normalizing it replaces dot notation with slash notation. For instance:

        .Geom.Perm => Geom/Perm

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                args[i] = _normalize_location(arg)
                break

        return func(*args, **kwargs)

    return wrapper


# -----------------------------------------------------------------------------


def with_absolute_path(func):
    """Assume the first string argument is a path and resolve it."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                args[i] = get_absolute_path(arg)
                break

        return func(*args, **kwargs)

    return wrapper
