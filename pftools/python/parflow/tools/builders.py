# builders.py
# functions for helping build ParFlow scripts
import os
from pathlib import Path
import sys
import yaml

import numpy as np

from .helper import sort_dict
from .fs import exists

from parflow.tools.io import write_patch_matrix_as_asc
from parflow.tools.fs import get_absolute_path


class NoAliasDumper(yaml.SafeDumper):

    def ignore_aliases(self, data):
        """addressing alias printing when applying database properties
        to multiple keys
        """
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
        """Setting mask array to SolidFileBuilder object

        Args:
            mask_array (array): Array of values to define the mask.
        """
        self.mask_array = mask_array
        return self

    def top(self, patch_id):
        """Overwriting top patch ID

        Args:
            patch_id (int): ID of top patch in mask array
        """
        self.top_id = patch_id
        self.patch_ids_top = None
        return self

    def bottom(self, patch_id):
        """Overwriting bottom patch ID

        Args:
            patch_id (int): ID of bottom patch in mask array
        """
        self.bottom_id = patch_id
        self.patch_ids_top = None
        return self

    def side(self, patch_id):
        """Overwriting side patch ID

        Args:
            patch_id (int): ID of side patch in mask array
        """
        self.side_id = patch_id
        self.patch_ids_side = None
        return self

    def top_ids(self, top_patch_ids):
        """Overwriting top patch ID with multiple IDs

        Args:
            top_patch_ids (array): array of top patch ids
        """
        self.patch_ids_top = top_patch_ids
        return self

    def bottom_ids(self, bottom_patch_ids):
        """Overwriting bottom patch ID with multiple IDs

        Args:
            bottom_patch_ids (array): array of bottom patch ids
        """
        self.patch_ids_bottom = bottom_patch_ids
        return self

    def side_ids(self, side_patch_ids):
        """Overwriting side patch ID with multiple IDs

        Args:
            side_patch_ids (array): array of side patch ids
        """
        self.patch_ids_side = side_patch_ids
        return self

    def write(self, name, xllcorner=0, yllcorner=0, cellsize=0, vtk=False):
        """Writing out pfsol file with optional output to vtk

        Args:
            name (str): Name of solid file to write
            xllcorner (int, float): coordinate of lower-left corner of x-axis
            yllcorner (int, float): coordinate of lower-left corner of y-axis
            cellsize (int): size of horizontal grid cell for solid file
        """
        self.name = name
        output_file_path = get_absolute_path(name)
        if self.mask_array is None:
            raise Exception('No mask was defined')

        shape = self.mask_array.shape
        dtype = np.int16
        left_mask = np.zeros(shape, dtype=dtype)
        right_mask = np.zeros(shape, dtype=dtype)
        back_mask = np.zeros(shape, dtype=dtype)
        front_mask = np.zeros(shape, dtype=dtype)
        bottom_mask = np.zeros(shape, dtype=dtype)
        top_mask = np.zeros(shape, dtype=dtype)

        j_size, i_size = shape
        for j in range(j_size):
            for i in range(i_size):
                if self.mask_array[j, i] != 0:
                    patch_value = 0 if self.patch_ids_side is None \
                        else self.patch_ids_side[j, i]
                    # Left (-x)
                    if i == 0 or self.mask_array[j, i - 1] == 0:
                        left_mask[j, i] = patch_value if patch_value \
                            else self.side_id

                    # Right (+x)
                    if i + 1 == i_size or self.mask_array[j, i + 1] == 0:
                        right_mask[j, i] = patch_value if patch_value \
                            else self.side_id

                    # Back (-y) (y flipped)
                    if j + 1 == j_size or self.mask_array[j + 1, i] == 0:
                        back_mask[j, i] = patch_value if patch_value \
                            else self.side_id

                    # Front (+y) (y flipped)
                    if j == 0 or self.mask_array[j - 1, i] == 0:
                        front_mask[j, i] = patch_value if patch_value \
                            else self.side_id

                    # Bottom (-z)
                    patch_value = 0 if self.patch_ids_bottom is None \
                        else self.patch_ids_bottom[j, i]
                    bottom_mask[j, i] = patch_value if patch_value \
                        else self.bottom_id

                    # Top (+z)
                    patch_value = 0 if self.patch_ids_top is None \
                        else self.patch_ids_top[j, i]
                    top_mask[j, i] = patch_value if patch_value \
                        else self.top_id

        # Generate asc / sa files
        write_func = write_patch_matrix_as_asc
        settings = {
            'xllcorner': xllcorner,
            'yllcorner': yllcorner,
            'cellsize': cellsize,
            'NODATA_value': 0,
        }
        short_name = name[:-6]

        left_file_path = get_absolute_path(f'{short_name}_left.asc')
        write_func(left_mask, left_file_path, **settings)

        right_file_path = get_absolute_path(f'{short_name}_right.asc')
        write_func(right_mask, right_file_path, **settings)

        front_file_path = get_absolute_path(f'{short_name}_front.asc')
        write_func(front_mask, front_file_path, **settings)

        back_file_path = get_absolute_path(f'{short_name}_back.asc')
        write_func(back_mask, back_file_path, **settings)

        top_file_path = get_absolute_path(f'{short_name}_top.asc')
        write_func(top_mask, top_file_path, **settings)

        bottom_file_path = get_absolute_path(f'{short_name}_bottom.asc')
        write_func(bottom_mask, bottom_file_path, **settings)

        # Trigger conversion
        print('=== pfmask-to-pfsol ===: BEGIN')
        extra = []
        if vtk:
            extra.append('--vtk')
            extra.append(f'{output_file_path[:-6]}.vtk')

        exe_path = get_absolute_path('$PARFLOW_DIR/bin/pfmask-to-pfsol')
        args = [
            f'--mask-top {top_file_path}',
            f'--mask-bottom {bottom_file_path}',
            f'--mask-left {left_file_path}',
            f'--mask-right {right_file_path}',
            f'--mask-front {front_file_path}',
            f'--mask-back {back_file_path}',
            f'--pfsol {output_file_path}'
        ] + extra
        os.system(f'{exe_path} ' + ' '.join(args))

        print('=== pfmask-to-pfsol ===: END')
        return self

    def for_key(self, geom_item):
        """Setting ParFlow keys associated with solid file

        Args:
            geom_item (str): Name of geometric unit in ParFlow run that will
            bet used as a token for the ParFlow key.
        """
        geom_item.InputType = 'SolidFile'
        geom_item.FileName = self.name
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
        self.run = run
        self.output = {}
        self.name_registration = {}
        self.column_index = {}
        self.props_in_row_header = True
        self.table_comments = []
        yaml_key_def = Path(__file__).parent / 'ref/table_keys.yaml'
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
            raise Exception(f'Warning - duplicate alias name(s):'
                            f' {self.alias_duplicates}')

    def _process_data_line(self, tokens):
        """Method to process lines of data in a table
        """
        # Skip new lines or comments
        if not tokens or tokens[0] == '#':
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
                if 'addon' in key_def:
                    for key, value in key_def['addon'].items():
                        # local keys (appending to geom item)
                        if key.startswith('.'):
                            data[key[1:]] = value
                        # global keys
                        elif key not in self.output:
                            self.output[key] = value

                # appending geom name to list for setting geom name keys
                if 'register' in key_def:
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
            if 'addon' in key_def:
                for addon_key, addon_value in key_def['addon'].items():
                    # local keys (appending to geom item)
                    if addon_key.startswith('.'):
                        data[addon_key[1:]] = addon_value
                    # global keys
                    elif addon_key not in self.output:
                        self.output[addon_key] = addon_value

            # appending geom name to list for setting geom name keys
            if 'register' in key_def:
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
        """Method to process first line in a table
        """
        # Skip new lines or comments
        if not first_line_tokens:
            return False
        if first_line_tokens[0] == '#':
            self.table_comments.append(' '.join(first_line_tokens))
            return False

        self.props_in_row_header = None
        found = []
        not_found = []
        for i, token in enumerate(first_line_tokens):
            self.column_index[token] = i

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

    def load_csv_file(self, table_file, encoding='utf-8-sig'):
        """Method to load a .csv file of a table of subsurface parameters

        Args:
            table_file (str): Path to the input .csv file.
            encoding='utf-8-sig': encoding of input file.
        """
        with open(get_absolute_path(table_file), 'r',
                  encoding=encoding) as csv_file:
            data_line = False
            for line in csv_file:
                tokens = _csv_line_tokenizer(line)
                if data_line:
                    self._process_data_line(tokens)
                else:
                    data_line = self._process_first_line(tokens)
        return self

    def load_txt_file(self, table_file, encoding='utf-8-sig'):
        """Method to load a .txt file of a table of subsurface parameters

        Args:
            table_file (str): Path to the input .txt file.
            encoding='utf-8-sig': encoding of input file.
        """
        with open(get_absolute_path(table_file), 'r',
                  encoding=encoding) as txt_file:
            data_line = False
            for line in txt_file:
                tokens = _txt_line_tokenizer(line)
                if data_line:
                    self._process_data_line(tokens)
                else:
                    data_line = self._process_first_line(tokens)

        return self

    def load_txt_content(self, txt_content):
        """Method to load an in-line text table of subsurface parameters

        Args:
            txt_content (str): In-line text string.
        """
        data_line = False
        for line in txt_content.splitlines():
            tokens = _txt_line_tokenizer(line)
            if data_line:
                self._process_data_line(tokens)
            else:
                data_line = self._process_first_line(tokens)

        return self

    def assign(self, old=None, new=None, mapping=None):
        """Method to assigning subsurface properties of one unit to another

        Args:
            old=None (str): Source unit with existing parameters
            new=None (str): Target unit to which the parameters
                            from old will be mapped
            mapping=None (dict): Dictionary that includes the old units as keys
                and new units as values.
        """
        # assigning subsurface properties of one unit to another
        if isinstance(new, list):
            for item in new:
                self.assign(old, item)

        elif old != new:
            data = self.output[old]
            self.output[new] = data

        if mapping is not None:
            for old, new in mapping.items():
                self.assign(old, new)

        return self

    def load_default_properties(self, database='conus_1'):
        """Method to load one of several default property databases.

        Args:
           database='conus_1': default database - options are:
           'conus_1': soil/rock properties from Maxwell and Condon (2016)
           'washita': soil/rock properties from Little Washita script
           'freeze_cherry': soil/rock properties from Freeze and Cherry (1979)
           Note: Freeze and Cherry only has permeability and porosity
        """
        database_file = f'ref/subsurface_{database}.txt'
        default_prop_file = str(Path(__file__).parent / database_file)

        if exists(default_prop_file):
            self.load_txt_file(default_prop_file)
            print('#' * 80)
            print('# Loaded database:')
            for item in self.table_comments:
                print(item)
            print('#' * 80)
        else:
            print('#' * 80)
            print(f'# {database} database not found. Available databases '
                  f'include:')
            for root, dirs, files in os.walk(Path(__file__).parent / 'ref'):
                for name in files:
                    if name.startswith('subsurface'):
                        print(f'# - {name} (use argument '
                              f'"{name[len("subsurface_"):-len(".txt")]}")')
            print('#' * 80)

        return self

    def apply(self, run=None, name_registration=True):
        """Method to apply the loaded subsurface properties to a given
           run object.

        Args:
            run=None (Run object): Run object to which the loaded subsurface
                parameters will be applied. If run=None, then the run object
                must be passed in as an argument when the
                SubsurfacePropertiesBuilder is instantiated.
            name_registration=True (bool): sets the auxiliary keys
                (e.g., GeomNames) related to the loaded subsurface properties
        """
        # applying subsurface properties to run keys
        if run is None:
            if self.run is None:
                print('No run object assigned')
                sys.exit(1)
        else:
            self.run = run

        valid_geom_names = []
        addon_keys = {}
        for name in self.output:
            if name in self.run.Geom.__dict__:
                valid_geom_names.append(name)
            elif name_registration and not isinstance(self.output[name], dict):
                addon_keys[name] = self.output[name]

        # Run pfset on all geom sections
        for geom_name in valid_geom_names:
            self.run.Geom[geom_name].pfset(flat_map=self.output[geom_name])

        # Handle names
        if name_registration:
            names_to_set = addon_keys
            for geom_name in valid_geom_names:
                if geom_name in self.name_registration:
                    for prop_name in self.name_registration[geom_name]:
                        if prop_name not in names_to_set:
                            names_to_set[prop_name] = []
                        names_to_set[prop_name].append(geom_name)
            self.run.pfset(flat_map=names_to_set)

        return self

    def print(self):
        """Method to print subsurface properties in hierarchical format
        """
        output_to_print = {'Geom': {}}
        valid_geom_names = []
        for geom_name in self.output:
            if hasattr(self.run.Geom, geom_name):
                valid_geom_names.append(geom_name)

        for geom_name in valid_geom_names:
            if hasattr(self.name_registration, geom_name):
                for prop_name in self.name_registration[geom_name]:
                    output_to_print.setdefault(prop_name, []).append(geom_name)

        for geom_name in valid_geom_names:
            output_to_print['Geom'][geom_name] = self.output[geom_name]

        print(yaml.dump(sort_dict(output_to_print), Dumper=NoAliasDumper))
        return self

    def get_table(self, props_in_header=True, column_separator='  '):
        """Method to convert loaded subsurface properties into a table

        Args:
            props_in_header=True (bool): Defaults to returning a table with
                property values at the top of each column
            column_separator='  ' (str): Defaults to returning a table that
                is space-delimited.

        Returns:
            text block of table of subsurface units and parameter values
        """
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
        """Method to print the table returned from the get_table method
        """
        print(self.get_table(props_in_header, column_separator))
        return self

# -----------------------------------------------------------------------------
# Domain input builder - setting keys for various common problem definitions
# -----------------------------------------------------------------------------


class DomainBuilder:

    def __init__(self, run, name='domain'):
        self.run = run
        self.run.Domain.GeomName = name

    def __file_check(self, file_name, key_path):
        """Checking files and setting keys for FileName keys
        """
        container, = self.run.select(key_path)
        container.FileName = file_name
        ext = Path(file_name).suffix
        if ext == '.pfb':
            container.Type = 'PFBFile'
        elif ext == '.nc':
            container.Type = 'NCFile'
        else:
            raise Exception(f'File extension {ext} for {file_name} is invalid')

        return self

    def water(self, geom_name=None):
        """Setting keys for water properties and gravity
        """
        self.run.Gravity = 1.0
        self.run.Phase.Names = 'water'
        self.run.Phase.water.Density.Type = 'Constant'
        self.run.Phase.water.Density.Value = 1.0
        self.run.Phase.water.Viscosity.Type = 'Constant'
        self.run.Phase.water.Viscosity.Value = 1.0
        self.run.Phase.water.Mobility.Type = 'Constant'
        self.run.Phase.water.Mobility.Value = 1.0
        self.run.PhaseSources.water.Type = 'Constant'

        if geom_name:
            self.run.PhaseSources.water.GeomNames = geom_name
            self.run.PhaseSources.water.Geom[geom_name].Value = 0.0

        return self

    def no_wells(self):
        """Setting key with no wells
        """
        self.run.Wells.Names = ''

        return self

    def no_contaminants(self):
        """Setting key with no contaminants
        """
        self.run.Contaminants.Names = ''

        return self

    def variably_saturated(self):
        """Setting keys for variably saturated domain.
        Solver settings taken from default_richards test case
        """
        self.run.Solver = 'Richards'
        self.run.Solver.MaxIter = 5
        self.run.Solver.Nonlinear.MaxIter = 10
        self.run.Solver.Nonlinear.ResidualTol = 1e-9
        self.run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
        self.run.Solver.Nonlinear.EtaValue = 1e-5
        self.run.Solver.Nonlinear.UseJacobian = True
        self.run.Solver.Nonlinear.DerivativeEpsilon = 1e-2
        self.run.Solver.Linear.KrylovDimension = 10
        self.run.Solver.Linear.Preconditioner = 'PFMG'

        return self

    def fully_saturated(self):
        """Fully saturated solver settings (other than solver ='Impes')
        taken from default_richards test case - need to change
        """
        self.run.Solver = 'Impes'
        self.run.Solver.MaxIter = 5
        self.run.Solver.Nonlinear.MaxIter = 10
        self.run.Solver.Nonlinear.ResidualTol = 1e-9
        self.run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
        self.run.Solver.Nonlinear.EtaValue = 1e-5
        self.run.Solver.Nonlinear.UseJacobian = True
        self.run.Solver.Nonlinear.DerivativeEpsilon = 1e-2
        self.run.Solver.Linear.KrylovDimension = 10
        self.run.Solver.Linear.Preconditioner = 'PFMG'

        return self

    def homogeneous_subsurface(self, domain_name, perm=None, porosity=None,
                               specific_storage=None, rel_perm=None,
                               saturation=None, isotropic=False):
        """Setting constant parameters for homogeneous subsurface
        """
        if perm is not None:
            if not self.run.Geom.Perm.Names:
                self.run.Geom.Perm.Names = []

            self.run.Geom.Perm._details_['Names']['history'] = []
            self.run.Geom.Perm.Names += [domain_name]

            # checking for Perm file
            if isinstance(perm, str):
                self.__file_check(perm, f'Geom/{domain_name}/Perm')
            else:
                self.run.Geom[domain_name].Perm.Type = 'Constant'
                self.run.Geom[domain_name].Perm.Value = perm

        if porosity is not None:
            if not self.run.Geom.Porosity.GeomNames:
                self.run.Geom.Porosity.GeomNames = []

            self.run.Geom.Porosity._details_['GeomNames']['history'] = []
            self.run.Geom.Porosity.GeomNames += [domain_name]

            # checking for Porosity file
            if isinstance(porosity, str):
                self.__file_check(porosity, f'Geom/{domain_name}/Porosity')
            else:
                self.run.Geom[domain_name].Porosity.Type = 'Constant'
                self.run.Geom[domain_name].Porosity.Value = porosity

        if specific_storage is not None:
            if not self.run.SpecificStorage.GeomNames:
                self.run.SpecificStorage.GeomNames = []

            self.run.SpecificStorage._details_['GeomNames']['history'] = []
            self.run.SpecificStorage.GeomNames += [domain_name]
            self.run.SpecificStorage.Type = 'Constant'
            self.run.Geom[domain_name].SpecificStorage.Value = specific_storage

        if rel_perm is not None:
            if not self.run.Phase.RelPerm.GeomNames:
                self.run.Phase.RelPerm.GeomNames = []

            self.run.Phase.RelPerm.Type = rel_perm['Type']
            self.run.Phase.RelPerm._details_['GeomNames']['history'] = []
            self.run.Phase.RelPerm.GeomNames += [domain_name]
            if rel_perm['Type'] == 'VanGenuchten':
                self.run.Geom[domain_name].RelPerm.Alpha = rel_perm['Alpha']
                self.run.Geom[domain_name].RelPerm.N = rel_perm['N']

        if saturation is not None:
            if not self.run.Phase.Saturation.GeomNames:
                self.run.Phase.Saturation.GeomNames = []

            self.run.Phase.Saturation.Type = saturation['Type']
            self.run.Phase.Saturation._details_['GeomNames']['history'] = []
            self.run.Phase.Saturation.GeomNames += [domain_name]
            if saturation['Type'] == 'VanGenuchten':
                # defaulting to RelPerm not working
                self.run.Geom[domain_name].Saturation.Alpha = (
                    saturation['Alpha'] if saturation['Alpha']
                    else rel_perm['Alpha'])
                self.run.Geom[domain_name].Saturation.N = (
                    saturation['N'] if saturation['N'] else rel_perm['N'])
                self.run.Geom[domain_name].Saturation.SRes = saturation['SRes']
                self.run.Geom[domain_name].Saturation.SSat = saturation['SSat']

        if isotropic:
            self.run.Perm.TensorType = 'TensorByGeom'

            if not self.run.Geom.Perm.TensorByGeom.Names:
                self.run.Geom.Perm.TensorByGeom.Names = []

            self.run.Geom.Perm.TensorByGeom._details_['Names']['history'] = []
            self.run.Geom.Perm.TensorByGeom.Names += [domain_name]
            self.run.Geom[domain_name].Perm.TensorValX = 1.0
            self.run.Geom[domain_name].Perm.TensorValY = 1.0
            self.run.Geom[domain_name].Perm.TensorValZ = 1.0

        return self

    def box_domain(self, box_input, domain_geom_name,
                   bounds=None, patches=None):
        """Defining box domain and extents
        """

        if not self.run.GeomInput.Names:
            self.run.GeomInput.Names = []

        if box_input not in self.run.GeomInput.Names:
            self.run.GeomInput._details_['Names']['history'] = []
            self.run.GeomInput.Names += [box_input]

        box_input_obj = self.run.GeomInput[box_input]
        if not box_input_obj.InputType:
            box_input_obj.InputType = 'Box'

        if not box_input_obj.GeomName:
            box_input_obj.GeomName = []

        if domain_geom_name not in box_input_obj.GeomName:
            box_input_obj._details_['GeomName']['history'] = []
            box_input_obj.GeomName += [domain_geom_name]

        domain_geom = self.run.Geom[domain_geom_name]
        if bounds is None:
            domain_geom.Lower.X = 0.0
            domain_geom.Lower.Y = 0.0
            domain_geom.Lower.Z = 0.0
            grid = self.run.ComputationalGrid
            domain_geom.Upper.X = grid.DX * grid.NX
            domain_geom.Upper.Y = grid.DY * grid.NY
            domain_geom.Upper.Z = grid.DZ * grid.NZ
        else:
            domain_geom.Lower.X = bounds[0]
            domain_geom.Upper.X = bounds[1]
            domain_geom.Lower.Y = bounds[2]
            domain_geom.Upper.Y = bounds[3]
            domain_geom.Lower.Z = bounds[4]
            domain_geom.Upper.Z = bounds[5]

        if patches:
            domain_geom.Patches = patches

        return self

    def slopes_mannings(self, domain_geom_name, slope_x=None,
                        slope_y=None, mannings=None):
        """Setting slopes and mannings coefficients as constant value
        or from an external file
        """
        if slope_x is not None:
            self.run.TopoSlopesX.GeomNames = domain_geom_name
            if isinstance(slope_x, str):
                self.__file_check(slope_x, 'TopoSlopesX')
            else:
                self.run.TopoSlopesX.Type = 'Constant'
                self.run.TopoSlopesX.Geom[domain_geom_name].Value = slope_x
        if slope_y is not None:
            self.run.TopoSlopesY.GeomNames = domain_geom_name
            if isinstance(slope_y, str):
                self.__file_check(slope_y, 'TopoSlopesY')
            else:
                self.run.TopoSlopesY.Type = 'Constant'
                self.run.TopoSlopesY.Geom[domain_geom_name].Value = slope_y
        if mannings is not None:
            self.run.Mannings.GeomNames = domain_geom_name
            if isinstance(mannings, str):
                self.__file_check(mannings, 'Mannings')
            else:
                self.run.Mannings.Type = 'Constant'
                self.run.Mannings.Geom[domain_geom_name].Value = mannings

        return self

    def zero_flux(self, patches, cycle_name, interval_name):
        """Setting zero-flux boundary condition for patch or patches
        """
        if not self.run.BCPressure.PatchNames:
            self.run.BCPressure.PatchNames = []

        for patch in patches.split():
            self.run.BCPressure._details_['PatchNames']['history'] = []
            self.run.BCPressure.PatchNames += [patch]
            self.run.Patch[patch].BCPressure.Type = 'FluxConst'
            self.run.Patch[patch].BCPressure.Cycle = cycle_name
            self.run.Patch[patch].BCPressure[interval_name].Value = 0.0

        return self

    def ic_pressure(self, domain_geom_name, patch, pressure):
        """Setting initial condition pressure from file or to constant value
        """
        self.run.ICPressure.GeomNames = domain_geom_name
        self.run.Geom[domain_geom_name].ICPressure.RefPatch = patch

        if isinstance(pressure, str) and Path(pressure).suffix == '.pfb':
            self.run.ICPressure.Type = 'PFBFile'
            self.run.Geom.domain.ICPressure.FileName = pressure
        elif isinstance(pressure, (float, int)):
            self.run.ICPressure.Type = 'HydroStaticPatch'
            self.run.Geom.domain.ICPressure.Value = pressure
        else:
            raise Exception(f'Incompatible type or file of {pressure}')

        return self

    def clm(self, met_file_name, top_patch, cycle_name, interval_name):
        """Setting keys associated with CLM
        """
        # ensure time step is hourly
        self.run.TimeStep.Type = 'Constant'
        self.run.TimeStep.Value = 1.0
        # ensure OverlandFlow is the top boundary condition
        self.run.Patch[top_patch].BCPressure.Type = 'OverlandFlow'
        self.run.Patch[top_patch].BCPressure.Cycle = cycle_name
        self.run.Patch[top_patch].BCPressure[interval_name].Value = 0.0
        # set CLM keys
        self.run.Solver.LSM = 'CLM'
        self.run.Solver.CLM.CLMFileDir = "."
        self.run.Solver.PrintCLM = True
        self.run.Solver.CLM.Print1dOut = False
        self.run.Solver.BinaryOutDir = False
        self.run.Solver.CLM.DailyRST = True
        self.run.Solver.CLM.SingleFile = True
        self.run.Solver.CLM.CLMDumpInterval = 24
        self.run.Solver.CLM.WriteLogs = False
        self.run.Solver.CLM.WriteLastRST = True
        self.run.Solver.CLM.MetForcing = '1D'
        self.run.Solver.CLM.MetFileName = met_file_name
        self.run.Solver.CLM.MetFilePath = "."
        self.run.Solver.CLM.MetFileNT = 24
        self.run.Solver.CLM.IstepStart = 1.0
        self.run.Solver.CLM.EvapBeta = 'Linear'
        self.run.Solver.CLM.VegWaterStress = 'Saturation'
        self.run.Solver.CLM.ResSat = 0.1
        self.run.Solver.CLM.WiltingPoint = 0.12
        self.run.Solver.CLM.FieldCapacity = 0.98
        self.run.Solver.CLM.IrrigationType = 'none'

        return self

    def well(self, name, type, x, y, z_upper, z_lower,
             cycle_name, interval_name, action='Extraction',
             saturation=1.0, phase='water', hydrostatic_pressure=None,
             value=None):
        """Setting keys necessary to define a simple well
        """

        if not self.run.Wells.Names:
            self.run.Wells.Names = []

        self.run.Wells.Names += [name]
        well = self.run.Wells[name]
        well.InputType = 'Vertical'
        well.Action = 'Extraction'
        well.Type = type
        well.X = x
        well.Y = y
        well.ZUpper = z_upper
        well.ZLower = z_lower
        well.Method = 'Standard'
        well.Cycle = cycle_name
        well[interval_name].Saturation[phase].Value = saturation

        if action == 'Extraction':
            well.Action = 'Extraction'
            if type == 'Pressure':
                well[interval_name].Pressure.Value = hydrostatic_pressure
                if value is not None:
                    well[interval_name].Extraction.Pressure.Value = value
            elif type == 'Flux' and value is not None:
                well[interval_name].Extraction.Flux[phase].Value = value

        if action == 'Injection':
            well.Action = 'Injection'
            if type == 'Pressure':
                well[interval_name].Pressure.Value = hydrostatic_pressure
                if value is not None:
                    well[interval_name].Injection.Pressure.Value = value
            elif type == 'Flux' and value is not None:
                well[interval_name].Injection.Flux[phase].Value = value

        return self

    def spinup_timing(self, initial_step, dump_interval):
        """Setting keys to assist a spinup run
        """

        self.run.TimingInfo.BaseUnit = 1
        self.run.TimingInfo.StartCount = 0
        self.run.TimingInfo.StartTime = 0.0
        self.run.TimingInfo.StopTime = 10000000
        self.run.TimingInfo.DumpInterval = dump_interval
        self.run.TimeStep.Type = 'Growth'
        self.run.TimeStep.InitialStep = initial_step
        self.run.TimeStep.GrowthFactor = 1.1
        self.run.TimeStep.MaxStep = 1000000
        self.run.TimeStep.MinStep = 0.1

        return self
