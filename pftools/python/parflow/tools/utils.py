# -*- coding: utf-8 -*-
"""Utility module

This module provide generic and multi-purpose methods

"""
import json
import yaml

from .database.generated import PFDBObj

# -----------------------------------------------------------------------------


def convert_value_for_string_dict(value):
    """Ensure that the output value is a valid string
    """
    if isinstance(value, str):
        return value

    if hasattr(value, '__iter__'):
        return ' '.join([str(v) for v in value])

    return value

# -----------------------------------------------------------------------------


def extract_keys_from_object(dict_to_fill, instance, parent_namespace=''):
    """Method that walk PFDBObj object and record their key and value
    inside a Python dict.
    """
    for key in instance.get_key_names(skip_default=True):

        value = instance.__dict__[key]
        if value is None:
            continue

        full_qualified_key = instance.get_parflow_key(parent_namespace, key)
        if isinstance(value, PFDBObj):
            if hasattr(value, '_value'):
                dict_to_fill[full_qualified_key] = convert_value_for_string_dict(
                    value._value)
            extract_keys_from_object(dict_to_fill, value, full_qualified_key)
        else:
            dict_to_fill[full_qualified_key] = convert_value_for_string_dict(value)

# -----------------------------------------------------------------------------

# TODO: add feature to read external files
# def external_file_to_dict(file_name, fileFormat):
#     externalFileDict = {}
#     externalFileDict['GeomInput.Names.domain_input.InputType'] = 'Box'
#     print(externalFileDict)
#     return externalFileDict

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
        # out.write(yaml.dump(sort_dict(overriden_keys)))
        out.write(yaml.dump(sort_dict(yamlObj)))

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
    ext = file_name.split('.').pop().lower()
    if ext in ['yaml', 'yml']:
        write_dict_as_yaml(dict_obj, file_name)
    elif ext == 'pfidb':
        write_dict_as_pfidb(dict_obj, file_name)
    elif ext == 'json':
        write_dict_as_json(dict_obj, file_name)
    else:
        print(f'Could not find writer for {file_name}')

# -----------------------------------------------------------------------------


def load_pfidb(file_path):
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

def sort_dict(input):
    """Create a key sorted dict
    """
    output = {}
    keys = list(input.keys())
    keys.sort()
    for key in keys:
        output[key] = input[key]

    return output

# -----------------------------------------------------------------------------

def get_or_create_dict(root, keyPath, overriden_keys):
  currentContainer = root
  for i in range(len(keyPath)):
    if keyPath[i] not in currentContainer:
      currentContainer[keyPath[i]] = {}
    elif not isinstance(currentContainer[keyPath[i]], dict):
      overriden_keys['.'.join(keyPath[:i+1])] = currentContainer[keyPath[i]]
      currentContainer[keyPath[i]] = {}
    currentContainer = currentContainer[keyPath[i]]

  return currentContainer
