# -*- coding: utf-8 -*-
"""io module

Helper functions to load or write files
"""

from functools import partial
import json
from pathlib import Path
import yaml
import numpy as np

from .fs import get_absolute_path
from .helper import sort_dict, get_or_create_dict

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper


# -----------------------------------------------------------------------------

def read_array(file_name):
    ext = Path(file_name).suffix[1:]
    funcs = {
        'pfb': read_array_pfb,
    }

    if ext not in funcs:
        raise Exception(f'Unknown extension: {file_name}')

    return funcs[ext](file_name)


# -----------------------------------------------------------------------------

def write_array(file_name, array):
    ext = Path(file_name).suffix[1:]
    funcs = {
        'pfb': write_array_pfb,
    }

    if ext not in funcs:
        raise Exception(f'Unknown extension: {file_name}')

    return funcs[ext](file_name, array)


# -----------------------------------------------------------------------------

def read_array_pfb(file_name):
    from parflowio.pyParflowio import PFData
    data = PFData(file_name)
    data.loadHeader()
    data.loadData()
    return data.moveDataArray()


# -----------------------------------------------------------------------------

def write_array_pfb(file_name, array):
    # Ensure this is 3 dimensions, since parflowio requires 3 dimensions.
    while array.ndim < 3:
        array = array[np.newaxis, :]

    if array.ndim > 3:
        raise Exception(f'Too many dimensions: {array.ndim}')

    from parflowio.pyParflowio import PFData
    data = PFData()
    data.setDataArray(array)
    return data.writeFile(file_name)


# -----------------------------------------------------------------------------

def load_patch_matrix_from_pfb_file(file_name, layer=None):
    data_array = read_array_pfb(file_name)
    if data_array.ndim == 3:
        nlayer, nrows, ncols = data_array.shape
        if layer:
            nlayer = layer
        return data_array[nlayer - 1, :, :]
    elif data_array.ndim == 2:
        return data_array
    else:
        raise Exception(f'invalid PFB file: {file_name}')


# -----------------------------------------------------------------------------

def load_patch_matrix_from_image_file(file_name, color_to_patch=None,
                                      fall_back_id=0):
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

        def _to_key(c, num):
            return ','.join([f'{c[i]}' for i in range(num)])

        to_key_1 = partial(_to_key, num=1)
        to_key_2 = partial(_to_key, num=2)
        to_key_3 = partial(_to_key, num=3)

        for key, value in color_to_patch.items():
            hex_color = key.lstrip('#')
            color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
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

        print(f'Sizes: colors({len(colors)}), 1({len(size1)}), '
              f'2({len(size2)}), 3({len(size3)})')

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
                except Exception:
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
            except Exception:
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
    i_size = -1
    j_size = -1
    k_size = -1
    with open(file_name) as f:
        i_size, j_size, k_size = map(int, f.readline().split())

    matrix = np.loadtxt(file_name, skiprows=1, dtype=np.int16)
    matrix.shape = (j_size, i_size)
    return matrix


# -----------------------------------------------------------------------------

def write_patch_matrix_as_asc(matrix, file_name, xllcorner=0.0, yllcorner=0.0,
                              cellsize=1.0, NODATA_value=0, **kwargs):
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
    yaml_obj = {}
    overriden_keys = {}
    for key, value in dict_obj.items():
        keys_path = key.split('.')
        get_or_create_dict(
            yaml_obj, keys_path[:-1], overriden_keys)[keys_path[-1]] = value

    # Push value back to yaml
    for key, value in overriden_keys.items():
        keys_path = key.split('.')
        value_obj = get_or_create_dict(yaml_obj, keys_path, {})
        value_obj['_value_'] = value

    output = yaml.dump(sort_dict(yaml_obj), Dumper=YAMLDumper)
    Path(file_name).write_text(output)


# -----------------------------------------------------------------------------

def write_dict_as_json(dict_obj, file_name):
    """Write a Python dict in a json format inside the provided file_name
    """
    Path(file_name).write_text(json.dumps(dict_obj, indent=2))


# -----------------------------------------------------------------------------

def write_dict(dict_obj, file_name):
    """Write a Python dict into a file_name using the extension to
    determine its format.
    """
    # Always write a sorted dictionary
    sorted_dict = sort_dict(dict_obj)

    ext = Path(file_name).suffix[1:].lower()
    if ext in ['yaml', 'yml']:
        write_dict_as_yaml(sorted_dict, file_name)
    elif ext == 'pfidb':
        write_dict_as_pfidb(sorted_dict, file_name)
    elif ext == 'json':
        write_dict_as_json(sorted_dict, file_name)
    else:
        raise Exception(f'Could not find writer for {file_name}')


# -----------------------------------------------------------------------------

def to_native_type(string):
    """Converting a string to a value in native format.
    Used for converting .pfidb files
    """
    types_to_try = [int, float]
    for t in types_to_try:
        try:
            return t(string)
        except ValueError:
            pass
    return string


# -----------------------------------------------------------------------------

def read_pfidb(file_path):
    """Load pfidb file into a Python dict
    """
    result_dict = {}
    action = 'nb_lines'  # nb_lines, size, string
    size = 0
    key = ''
    value = ''
    string_type_count = 0
    full_path = get_absolute_path(file_path)

    with open(full_path, 'r') as input_file:
        for line in input_file:
            if action == 'string':
                if string_type_count % 2 == 0:
                    key = line[:size]
                else:
                    value = line[:size]
                    result_dict[key] = to_native_type(value)
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
    path = Path(file_path)
    if not path.exists():
        return {}

    return yaml.safe_load(path.read_text())


# -----------------------------------------------------------------------------

def _read_clmin(file_name):
    """function to load in drv_clmin.dat files

       Args:
           - file_name: name of drv_clmin.dat file

       Returns:
           dictionary of key/value pairs of variables in file
    """
    clm_vars = {}
    with open(file_name, 'r') as rf:
        for line in rf:
            # skip if first 15 are empty or exclamation
            if line and line[0].islower():
                first_word = line.split()[0]
                if len(first_word) > 15:
                    clm_vars[first_word[:14]] = first_word[15:]
                else:
                    clm_vars[first_word] = line.split()[1]

    return clm_vars


# -----------------------------------------------------------------------------

def _read_vegm(file_name):
    """function to load in drv_vegm.dat files

       Args:
           - file_name: name of drv_vegm.dat file

       Returns:
           3D numpy array for domain, with 3rd dimension defining each column
           in the vegm.dat file except for x/y
    """
    with open(file_name, 'r') as rf:
        lines = rf.readlines()

    last_line_split = lines[-1].split()
    x_dim = int(last_line_split[0])
    y_dim = int(last_line_split[1])
    z_dim = len(last_line_split) - 2
    vegm_array = np.zeros((x_dim, y_dim, z_dim))
    # Assume first two lines are comments
    for line in lines[2:]:
        elements = line.split()
        x = int(elements[0])
        y = int(elements[1])
        for i in range(z_dim):
            vegm_array[x-1, y-1, i] = elements[i + 2]

    return vegm_array


# -----------------------------------------------------------------------------

def _read_vegp(file_name):
    """function to load in drv_vegp.dat files

       Args:
           - file_name: name of drv_vegp.dat file

       Returns:
           Dictionary with keys as variables and values as lists of parameter
           values for each of the 18 land cover types
    """
    vegp_data = {}
    current_var = None
    with open(file_name, 'r') as rf:
        for line in rf:
            if not line or line[0] == '!':
                continue

            split = line.split()
            if current_var is not None:
                vegp_data[current_var] = [to_native_type(i) for i in split]
                current_var = None
            elif line[0].islower():
                current_var = split[0]

    return vegp_data


# -----------------------------------------------------------------------------

def read_clm(file_name, type='clmin'):
    type_map = {
        'clmin': _read_clmin,
        'vegm': _read_vegm,
        'vegp': _read_vegp
    }

    if type not in type_map:
        raise Exception(f'Unknown clm type: {type}')

    return type_map[type](get_absolute_path(file_name))
