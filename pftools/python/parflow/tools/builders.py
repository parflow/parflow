# builders.py
# functions for helping build ParFlow scripts
import os
import yaml
import sys
import numpy as np

from .helper import sort_dict

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper

from parflow.tools.io import write_patch_matrix_as_asc, write_patch_matrix_as_sa
from parflow.tools.fs import get_absolute_path

# addressing alias printing when applying database properties to multiple keys
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

class SolidFileBuilder:

    def __init__(self, top=1, bottom=2, side=3):
        self.name = None
        self.mask_array = None
        self.patch_ids_top = None
        self.patch_ids_bottom = None
        self.patch_ids_side = None
        self.top_id = top
        self.bottom_id = bottom
        self.side_id = side

    def mask(self, mask_array):
        self.mask_array = mask_array
        return self

    def top(self, patch_id):
        self.top_id = patch_id
        self.patch_ids_top = None
        return self

    def bottom(self, patch_id):
        self.bottom_id = patch_id
        self.patch_ids_top = None
        return self

    def side(self, patch_id):
        self.side_id = patch_id
        self.patch_ids_side = None
        return self

    def top_ids(self, top_patch_ids):
        self.patch_ids_top = top_patch_ids
        return self

    def bottom_ids(self, bottom_patch_ids):
        self.patch_ids_bottom = bottom_patch_ids
        return self

    def side_ids(self, side_patch_ids):
        self.patch_ids_side = side_patch_ids
        return self

    def write(self, name, xllcorner=0, yllcorner=0, cellsize=0, vtk=False):
        self.name = name
        output_file_path = get_absolute_path(name)
        if self.mask_array is None:
            raise Exception('No mask were define')

        jSize, iSize = self.mask_array.shape
        leftMask = np.zeros((jSize, iSize), dtype=np.int16)
        rightMask = np.zeros((jSize, iSize), dtype=np.int16)
        backMask = np.zeros((jSize, iSize), dtype=np.int16)
        frontMask = np.zeros((jSize, iSize), dtype=np.int16)
        bottomMask = np.zeros((jSize, iSize), dtype=np.int16)
        topMask = np.zeros((jSize, iSize), dtype=np.int16)

        for j in range(jSize):
            for i in range(iSize):
                if self.mask_array[j, i] != 0:
                    patch_value = 0 if self.patch_ids_side is None else self.patch_ids_side[j, i]
                    # Left (-x)
                    if i == 0 or self.mask_array[j, i-1] == 0:
                        leftMask[j, i] = patch_value if patch_value else self.side_id

                    # Right (+x)
                    if i + 1 == iSize or self.mask_array[j, i+1] == 0:
                        rightMask[j, i] = patch_value if patch_value else self.side_id

                    # Back (-y) (y flipped)
                    if j + 1 == jSize or self.mask_array[j+1, i] == 0:
                        backMask[j, i] = patch_value if patch_value else self.side_id

                    # Front (+y) (y flipped)
                    if j == 0 or self.mask_array[j-1, i] == 0:
                        frontMask[j, i] = patch_value if patch_value else self.side_id

                    # Bottom (-z)
                    patch_value = 0 if self.patch_ids_bottom is None else self.patch_ids_bottom[j, i]
                    bottomMask[j, i] = patch_value if patch_value else self.bottom_id

                    # Top (+z)
                    patch_value = 0 if self.patch_ids_top is None else self.patch_ids_top[j, i]
                    topMask[j, i] = patch_value if patch_value else self.top_id

        # Generate asc / sa files
        writeFn = write_patch_matrix_as_asc
        settings = {
            'xllcorner': xllcorner,
            'yllcorner': yllcorner,
            'cellsize': cellsize,
            'NODATA_value': 0,
        }
        short_name = name[:-6]

        left_file_path = get_absolute_path(f'{short_name}_left.asc')
        writeFn(leftMask, left_file_path, **settings)

        right_file_path = get_absolute_path(f'{short_name}_right.asc')
        writeFn(rightMask, right_file_path, **settings)

        front_file_path = get_absolute_path(f'{short_name}_front.asc')
        writeFn(frontMask, front_file_path, **settings)

        back_file_path = get_absolute_path(f'{short_name}_back.asc')
        writeFn(backMask, back_file_path, **settings)

        top_file_path = get_absolute_path(f'{short_name}_top.asc')
        writeFn(topMask, top_file_path, **settings)

        bottom_file_path = get_absolute_path(f'{short_name}_bottom.asc')
        writeFn(bottomMask, bottom_file_path, **settings)

        # Trigger conversion
        print('=== pfmask-to-pfsol ===: BEGIN')
        extra = []
        if vtk:
            extra.append('--vtk')
            extra.append(f'{output_file_path[:-6]}.vtk')
        os.system(f'$PARFLOW_DIR/bin/pfmask-to-pfsol --mask-top {top_file_path} --mask-bottom {bottom_file_path} --mask-left {left_file_path} --mask-right {right_file_path} --mask-front {front_file_path} --mask-back {back_file_path} --pfsol {output_file_path} {" ".join(extra)}')
        print('=== pfmask-to-pfsol ===: END')
        return self

    def for_key(self, geomItem):
        geomItem.InputType = 'SolidFile'
        geomItem.FileName = self.name
        return self

# -----------------------------------------------------------------------------
# Subsurface hydraulic property input helper
# -----------------------------------------------------------------------------

# splitting csv and txt lines into tokens
def _csv_line_tokenizer(line):
    return [token.strip() for token in line.split(',')]

def _txt_line_tokenizer(line):
    return line.split()

class SubsurfacePropertiesBuilder:

    def __init__(self, run=None):
        if run is not None:
            self.run = run
        self.output = {}
        self.name_registration = {}
        self.column_index = {}
        self.props_in_row_header = True
        yaml_key_def = os.path.join(
            os.path.dirname(__file__), 'ref/table_keys.yaml')
        with open(yaml_key_def, 'r') as file:
            self.definition = yaml.safe_load(file)

        # Extract prop column names
        self.prop_names = []
        self.alias_to_pfkey = {}
        self.pfkey_to_alias = {}
        self.alias_duplicates = set()
        for key, value in self.definition.items():
            self.pfkey_to_alias[key] = value['alias'][0]
            for alias in value['alias']:
                # checking for duplicate aliases
                if alias in self.prop_names:
                    self.alias_duplicates.add(alias)

                self.prop_names.append(alias)
                self.alias_to_pfkey[alias] = key

        # crashes if there are duplicate aliases
        if self.alias_duplicates:
            raise Exception(f'Warning - duplicate alias name(s): {self.alias_duplicates}')

    def _process_data_line(self, tokens):
        # Skip new lines or comments
        if len(tokens) == 0 or tokens[0] == '#':
            return

        if self.props_in_row_header:
            # Key column contains geom_name
            data = {}
            registrations = []
            for alias, col_idx in self.column_index.items():
                str_value = tokens[col_idx]
                if str_value == '-':
                    continue

                key = self.alias_to_pfkey[alias]
                key_def = self.definition[key]
                value_type = key_def.get('type', 'float')
                value = __builtins__[value_type](str_value)
                data[key] = value

                # setting related addon keys
                if 'addon' in key_def.keys():
                    for key, value in key_def['addon'].items():
                        # local keys (appending to geom item)
                        if key[0] == '.':
                            data.update({key[1:]: value})
                        # global keys
                        elif key not in self.output:
                            self.output.update({key: value})

                # appending geom name to list for setting geom name keys
                if 'register' in key_def.keys():
                    registrations.append(key_def['register'])

            # Extract geom_name
            geom_name = data['key']
            del data['key']
            self.output[geom_name] = data

            if not hasattr(self.name_registration, geom_name):
                self.name_registration[geom_name] = set()

            self.name_registration[geom_name].update(registrations)

        else:
            # Key column contains property values
            data = {}
            registrations = []

            main_key = 'key'
            for key_alias in self.column_index:
                if key_alias in self.definition['key']['alias']:
                    main_key = key_alias

            prop_alias = tokens[self.column_index[main_key]]
            key = self.alias_to_pfkey[prop_alias]
            key_def = self.definition[key]
            value_type = key_def.get('type', 'float')
            value_convert = __builtins__[value_type]
            # setting related addon keys
            if 'addon' in key_def.keys():
                for addon_key, addon_value in key_def['addon'].items():
                    # local keys (appending to geom item)
                    if addon_key[0] == '.':
                        data.update({addon_key[1:]: addon_value})
                    # global keys
                    elif addon_key not in self.output:
                        self.output.update({addon_key: addon_value})

            # appending geom name to list for setting geom name keys
            if 'register' in key_def.keys():
                registrations.append(key_def['register'])

            for geom_name in self.column_index:
                if geom_name == main_key:
                    continue

                container = self.output[geom_name]
                value_str = tokens[self.column_index[geom_name]]
                if value_str == '-':
                    continue

                value = value_convert(value_str)
                container[key] = value
                container.update(data)
                if registrations:
                    self.name_registration[geom_name].update(registrations)


    def _process_first_line(self, first_line_tokens):
        # Skip new lines or comments
        if len(first_line_tokens) == 0 or first_line_tokens[0] == '#':
            return False

        self.props_in_row_header = None
        found = []
        not_found = []
        index = 0
        for token in first_line_tokens:
            self.column_index[token] = index
            index += 1

            if token in self.alias_to_pfkey:
                found.append(token)
            else:
                not_found.append(token)

        if len(not_found) == 0:
            self.props_in_row_header = True
        elif len(found) > 1 and len(not_found) > 1:
            print('Error while processing input table:')
            print(f' - Properties found: {found}')
            print(f' - Properties not found: {not_found}')
        elif len(found) == 1:
            self.props_in_row_header = False
            # Prefill geo_name containers
            for geom_name in self.column_index:
                if geom_name not in self.definition['key']['alias']:
                    self.output[geom_name] = {}
                    self.name_registration[geom_name] = set()

        if self.props_in_row_header is None:
            raise Exception('Invalid table format')

        return True

    def load_csv_file(self, tableFile, encoding='utf-8-sig'):
        with open(get_absolute_path(tableFile), 'r', encoding=encoding) as csv_file:
            data_line = False
            for line in csv_file.readlines():
                tokens = _csv_line_tokenizer(line)
                if data_line:
                    self._process_data_line(tokens)
                else:
                    data_line = self._process_first_line(tokens)
        return self

    def load_txt_file(self, tableFile, encoding='utf-8-sig'):
        with open(get_absolute_path(tableFile), 'r', encoding=encoding) as txt_file:
            data_line = False
            for line in txt_file.readlines():
                tokens = _txt_line_tokenizer(line)
                if data_line:
                    self._process_data_line(tokens)
                else:
                    data_line = self._process_first_line(tokens)

        return self

    def load_txt_content(self, txt_content):
        data_line = False
        for line in txt_content.splitlines():
            tokens = _txt_line_tokenizer(line)
            if data_line:
                self._process_data_line(tokens)
            else:
                data_line = self._process_first_line(tokens)

        return self

    def assign(self, old=None, new=None, mapping=None):
        if old != new:
            data = self.output[old]
            self.output[new] = data

        if mapping is not None:
            for old, new in mapping.items():
                if isinstance(new, list):
                    for item in new:
                        self.assign(old, item)
                else:
                    self.assign(old, new)

        return self

    def load_default_properties(self):
        default_prop_file = os.path.join(
            os.path.dirname(__file__), 'ref/default_subsurface.txt')
        self.load_txt_file(default_prop_file)

        return self

    def apply(self, run=None, name_registration=True):
        if run is None:
            if self.run is None:
                print('No run object assigned')
                sys.exit(1)
        else:
            self.run = run

        valid_geom_names = []
        addon_keys = {}
        for name in self.output:
            if name in self.run.Geom.__dict__.keys():
                valid_geom_names.append(name)
            elif name_registration and type(self.output[name]) is not dict:
                addon_keys[name] = self.output[name]


        # Run pfset on all geom sections
        for geom_name in valid_geom_names:
            self.run.Geom[geom_name].pfset(flat_map=self.output[geom_name])

        # Handle names
        if name_registration:
            names_to_set = addon_keys
            for geom_name in valid_geom_names:
                if geom_name in self.name_registration.keys():
                    for prop_name in self.name_registration[geom_name]:
                        if prop_name not in names_to_set.keys():
                            names_to_set[prop_name] = []
                        names_to_set[prop_name].append(geom_name)
            self.run.pfset(flat_map=names_to_set)

        return self

    def print(self):
        # printing in hierarchical format
        output_to_print = {'Geom': {}}
        valid_geom_names = []
        for geom_name in self.output:
            if hasattr(self.run.Geom, geom_name):
                valid_geom_names.append(geom_name)

        for geom_name in valid_geom_names:
            if hasattr(self.name_registration, geom_name):
                for prop_name in self.name_registration[geom_name]:
                    if not hasattr(output_to_print, prop_name):
                        output_to_print[prop_name] = []
                    output_to_print[prop_name].append(geom_name)

        for geom_name in valid_geom_names:
            output_to_print['Geom'][geom_name] = self.output[geom_name]

        print(yaml.dump(sort_dict(output_to_print), Dumper=NoAliasDumper))
        return self

    def get_table(self, props_in_header=True, column_separator='  '):
        entries = []
        prop_set = set()
        prop_sizes = {'key': 0}
        geom_sizes = {'key': 0}

        # Fill entries headers
        for geo_name, props in self.output.items():
            if not isinstance(props, dict):
                continue
            elif not hasattr(self.run.Geom, geo_name):
                continue

            entry = {
                'key': geo_name,
            }
            prop_sizes['key'] = max(prop_sizes['key'], len(geo_name))
            geom_sizes[geo_name] = len(geo_name)
            for prop in props:
                if prop in self.pfkey_to_alias:
                    alias = self.pfkey_to_alias[prop]
                    value = str(props[prop])
                    size = len(value)
                    entry[alias] = value
                    prop_set.add(alias)

                    # Find bigger size for geom
                    geom_sizes[geo_name] = max(geom_sizes[geo_name], size)

                    # Find bigger size for props
                    if alias not in prop_sizes:
                        prop_sizes[alias] = max(size, len(alias))
                    else:
                        prop_sizes[alias] = max(prop_sizes[alias], size)

                    geom_sizes['key'] = max(geom_sizes['key'], len(alias))

            entries.append(entry)

        # Figure out orientation
        prop_header_width = 0
        for alias in prop_sizes:
            prop_header_width += prop_sizes[alias] + 2

        geom_header_width = 0
        for geom_name in geom_sizes:
            geom_header_width += geom_sizes[geom_name] + 2

        # Build table
        table_lines = []
        header_keys = []
        if props_in_header:
            sizes = prop_sizes
            # Create table using props as header
            line = []
            for prop in sizes:
                header_keys.append(prop)
                width = sizes[prop]
                line.append(prop.ljust(width))

            # Add header
            table_lines.append(column_separator.join(line))

            # Add content
            for entry in entries:
                line = []
                for key in header_keys:
                    value = entry[key] if key in entry else '-'
                    width = sizes[key]
                    line.append(value.ljust(width))
                table_lines.append(column_separator.join(line))

        else:
            sizes = geom_sizes
            # Create table using geom name as header
            line = []
            for geom in sizes:
                header_keys.append(geom)
                width = sizes[geom]
                line.append(geom.ljust(width))

            # Add header
            table_lines.append(column_separator.join(line))

            # Add content
            for prop in prop_set:
                line = []
                for key in header_keys:
                    if key == 'key':
                        width = sizes[key]
                        line.append(prop.ljust(width))
                        continue

                    for entry in entries:
                        if entry['key'] != key:
                            continue
                        value = entry[prop] if prop in entry else '-'
                        width = sizes[key]
                        line.append(value.ljust(width))

                table_lines.append(column_separator.join(line))

        return '\n'.join(table_lines)

    def print_as_table(self, props_in_header=True, column_separator='  '):
        print(self.get_table(props_in_header, column_separator))
        return self
