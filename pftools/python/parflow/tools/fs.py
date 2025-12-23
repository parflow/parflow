# -*- coding: utf-8 -*-
"""file system module

This module provide helper function to deal with the working directory of a run
"""

import os
from pathlib import Path
import shutil

from . import settings


# -----------------------------------------------------------------------------


def get_absolute_path(file_path):
    """Helper function to resolve a file path while using the proper
    working directory.

    Return: Absolute file path
    """
    file_path = Path(os.path.expandvars(file_path))
    if file_path.is_absolute():
        return str(file_path)
    return str((settings.WORKING_DIRECTORY / file_path).resolve())


# -----------------------------------------------------------------------------


def cp(source, target_path="."):
    """Copying file/directory within python script"""
    full_source_path = get_absolute_path(source)
    full_target_path = get_absolute_path(target_path)
    try:
        if Path(full_source_path).is_dir():
            shutil.copytree(full_source_path, full_target_path)
        else:
            shutil.copy(full_source_path, full_target_path)
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")
    # For other errors
    except Exception:
        print(f"Error occurred while copying {full_source_path}.")


# -----------------------------------------------------------------------------


def rm(path):
    """Deleting file/directory within python script"""
    full_path = Path(get_absolute_path(path))
    if full_path.exists():
        if full_path.is_dir():
            shutil.rmtree(full_path)
        if full_path.is_file():
            os.remove(full_path)


# -----------------------------------------------------------------------------


def mkdir(dir_name):
    """mkdir within python script"""
    full_path = Path(get_absolute_path(dir_name))
    if not full_path.exists():
        full_path.mkdir(parents=True)


# -----------------------------------------------------------------------------


def get_text_file_content(file_path):
    full_path = Path(get_absolute_path(file_path))
    if not full_path.exists():
        raise Exception(f"{str(full_path)} does not exist!")

    return full_path.read_text()


# -----------------------------------------------------------------------------


def exists(file_path):
    return Path(get_absolute_path(file_path)).exists()


# -----------------------------------------------------------------------------


def chdir(directory_path):
    full_path = get_absolute_path(directory_path)
    os.chdir(full_path)
