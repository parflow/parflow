# -*- coding: utf-8 -*-
"""io module

Helper functions to load or write files
"""

import os
import json
import yaml
import numpy as np

from .helper import sort_dict, get_or_create_dict

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper

# -----------------------------------------------------------------------------

def load_patch_matrix_from_pfb_file(file_name, layer=None):
    from parflowio.pyParflowio import PFData
    data = PFData(file_name)
    data.loadHeader()
    data.loadData()
    data_array = data.getDataAsArray()

    if data_array.ndim == 3:
        nlayer, nrows, ncols = data_array.shape
        if layer:
            nlayer = layer
        return data_array[nlayer-1, :, :]

    elif data_array.ndim == 2:
        return data_array

    else:
        print(f'invalid PFB file: {file_name}')

    return

# -----------------------------------------------------------------------------

def load_patch_matrix_from_image_file(file_name, color_to_patch=None, fall_back_id=0):
    import imageio
    im = imageio.imread(file_name)
    height, width, color = im.shape
    matrix = np.zeros((height, width), dtype=np.int16)
    if color_to_patch is None:
        for j in range(height):
            for i in range(width):
                if im[j, i, 0] != 255:
                    matrix[j, i] = 1
    else:
        size1 = set()
        size2 = set()
        size3 = set()
        colors = []
        to_key_1 = lambda c: f'{c[0]}'
        to_key_2 = lambda c: f'{c[0]},{c[1]}'
        to_key_3 = lambda c: f'{c[0]},{c[1]},{c[2]}'

        for key, value in color_to_patch.items():
            hexColor = key.lstrip('#')
            color = tuple(int(hexColor[i:i+2], 16) for i in (0, 2, 4))
            colors.append((color, value))
            size1.add(to_key_1(color))
            size2.add(to_key_2(color))
            size3.add(to_key_3(color))

        to_key = None
        if len(colors) == len(size3):
            to_key = to_key_3
        if len(colors) == len(size2):
            to_key = to_key_2
        if len(colors) == len(size1):
            to_key = to_key_1

        print(f'Sizes: colors({len(colors)}), 1({len(size1)}), 2({len(size2)}), 3({len(size3)})')

        if to_key is None:
            raise Exception('You have duplicate colors')

        fast_map = {}
        for color_patch in colors:
            fast_map[to_key(color_patch[0])] = color_patch[1]

        for j in range(height):
            for i in range(width):
                key = to_key(im[j, i])
                try:
                    matrix[j, i] = fast_map[key]
                except:
                    matrix[j, i] = fall_back_id

    return np.flip(matrix, 0)

# -----------------------------------------------------------------------------

def load_patch_matrix_from_asc_file(file_name):
    ncols = -1
    nrows = -1
    in_header = True
    nb_line_to_skip = 0
    with open(file_name) as f:
        while in_header:
            line = f.readline()
            try:
                int(line)
                in_header = False
            except:
                key, value = line.split()
                if key == 'ncols':
                    ncols = int(value)
                if key == 'nrows':
                    nrows = int(value)
                nb_line_to_skip += 1

    matrix = np.loadtxt(file_name, skiprows=nb_line_to_skip, dtype=np.int16)
    matrix.shape = (nrows, ncols)

    return np.flip(matrix, 0)

# -----------------------------------------------------------------------------

def load_patch_matrix_from_sa_file(file_name):
    iSize = -1
    jSize = -1
    kSize = -1
    with open(file_name) as f:
        iSize, jSize, kSize = map(int, f.readline().split())

    matrix = np.loadtxt(file_name, skiprows=1, dtype=np.int16)
    matrix.shape = (jSize, iSize)
    return matrix

# -----------------------------------------------------------------------------

def write_patch_matrix_as_asc(matrix, file_name, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, NODATA_value=0, **kwargs):
    """Write asc for pfsol"""
    height, width = matrix.shape
    with open(file_name, 'w') as out:
        out.write(f'ncols          {width}\n')
        out.write(f'nrows          {height}\n')
        out.write(f'xllcorner      {xllcorner}\n')
        out.write(f'yllcorner      {yllcorner}\n')
        out.write(f'cellsize       {cellsize}\n')
        out.write(f'NODATA_value   {NODATA_value}\n')
        # asc are vertically flipped
        for j in range(height):
            for i in range(width):
                out.write(f'{matrix[height - j - 1, i]}\n')

# -----------------------------------------------------------------------------

def write_patch_matrix_as_sa(matrix, file_name, **kwargs):
    """Write asc for pfsol"""
    nrows, ncols = matrix.shape
    with open(file_name, 'w') as out:
        out.write(f'{ncols} {nrows} 1\n')
        it = np.nditer(matrix)
        for value in it:
            out.write(f'{value}\n')

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
