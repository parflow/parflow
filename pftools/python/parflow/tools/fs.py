# -*- coding: utf-8 -*-
"""file system module

This module provide helper function to deal with the working directory of a run
"""

import os
import shutil

from . import settings

# -----------------------------------------------------------------------------

def __map_env_variables(name):
    if name and name[0] == '$':
      value = os.getenv(name[1:])
      if value is not None:
        return value
    return name

# -----------------------------------------------------------------------------

def get_absolute_path(file_path):
    """Helper function to resolve a file path while using the proper
    working directory.

    Return: Absolute file path
    """
    file_path = '/'.join(map(__map_env_variables, file_path.split('/')))
    if os.path.isabs(file_path):
        return file_path
    return os.path.abspath(os.path.join(settings.WORKING_DIRECTORY, file_path))

# -----------------------------------------------------------------------------

def cp(source, target_path='.'):
    """Copying file/directory within python script
    """
    full_source_path = get_absolute_path(source)
    full_target_path = get_absolute_path(target_path)
    try:
        if os.path.isdir(full_source_path):
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
    except:
        print("Error occurred while copying file.")

# -----------------------------------------------------------------------------

def rm(path):
    """Deleting file/directory within python script
    """
    full_path = get_absolute_path(path)
    if os.path.exists(full_path):
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        if os.path.isfile(full_path):
            os.remove(full_path)

# -----------------------------------------------------------------------------

def mkdir(dir_name):
    """mkdir within python script
    """
    full_path = get_absolute_path(dir_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

# -----------------------------------------------------------------------------

def get_text_file_content(file_path):
  full_path = get_absolute_path(file_path)
  file_content = ''
  if os.path.exists(full_path):
    with open(full_path, 'r') as txt_file:
      file_content = txt_file.read()

  return file_content

# -----------------------------------------------------------------------------

def exists(file_path):
  full_path = get_absolute_path(file_path)
  return os.path.exists(full_path)

# -----------------------------------------------------------------------------

def chdir(directory_path):
  full_path = get_absolute_path(directory_path)
  os.chdir(full_path)
