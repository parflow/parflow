# -*- coding: utf-8 -*-
"""io module

Helper functions to load or write files
"""

import os
import json
import yaml

from .helper import sort_dict, get_or_create_dict

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper

# -----------------------------------------------------------------------------

def write_dict_as_pfidb(dict_obj, file_name):
    """Write a Python dict in a pfidb format inside the provided file_name
    """
    with open(file_name, 'w') as out:
        out.write(f'{len(dict_obj)}\n')
        for key in dict_obj:
            out.write(f'{len(key)}\n')
            out.write(f'{key}\n')
            value = dict_obj[key]
            out.write(f'{len(str(value))}\n')
            out.write(f'{str(value)}\n')

# -----------------------------------------------------------------------------

def write_dict_as_yaml(dict_obj, file_name):
    """Write a Python dict in a pfidb format inside the provided file_name
    """
    yamlObj = {}
    overriden_keys = {}
    for key, value in dict_obj.items():
        keys_path = key.split('.')
        get_or_create_dict(
            yamlObj, keys_path[:-1], overriden_keys)[keys_path[-1]] = value

    # Push value back to yaml
    for key, value in overriden_keys.items():
      keys_path = key.split('.')
      valueObj = get_or_create_dict(yamlObj, keys_path, {})
      valueObj['$_'] = value

    with open(file_name, 'w') as out:
        out.write(yaml.dump(sort_dict(yamlObj), Dumper=YAMLDumper))

# -----------------------------------------------------------------------------

def write_dict_as_json(dict_obj, file_name):
    """Write a Python dict in a json format inside the provided file_name
    """
    with open(file_name, 'w') as out:
        out.write(json.dumps(dict_obj, indent=2))

# -----------------------------------------------------------------------------

def write_dict(dict_obj, file_name):
    """Write a Python dict into a file_name using the extension to
    determine its format.
    """
    # Always write a sorted dictionary
    sorted_dict = sort_dict(dict_obj)

    ext = file_name.split('.').pop().lower()
    if ext in ['yaml', 'yml']:
        write_dict_as_yaml(sorted_dict, file_name)
    elif ext == 'pfidb':
        write_dict_as_pfidb(sorted_dict, file_name)
    elif ext == 'json':
        write_dict_as_json(sorted_dict, file_name)
    else:
        print(f'Could not find writer for {file_name}')

# -----------------------------------------------------------------------------

def read_pfidb(file_path):
    """Load pfidb file into a Python dict
    """
    result_dict = {}
    action = 'nb_lines'  # nbLines, size, string
    size = 0
    key = ''
    value = ''
    string_type_count = 0

    with open(file_path, 'r') as input_file:
        for line in input_file:
            if action == 'string':
                if string_type_count % 2 == 0:
                    key = line[:size]
                else:
                    value = line[:size]
                    result_dict[key] = value
                string_type_count += 1
                action = 'size'

            elif action == 'size':
                size = int(line)
                action = 'string'

            elif action == 'nb_lines':
                action = 'size'

    return result_dict

# -----------------------------------------------------------------------------

def read_yaml(file_path):
  """Load yaml file into a Python dict
  """
  if os.path.exists(file_path):
    with open(file_path, 'r') as txt_file:
      return yaml.safe_load(txt_file.read())

  return {}
