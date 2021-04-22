# -*- coding: utf-8 -*-
"""io module

Helper functions to load or write files
"""

from functools import partial
import json
from pathlib import Path
import yaml
import numpy as np

from .hydrology import calculate_evapotranspiration, calculate_overland_flow, calculate_overland_flow_grid, \
    calculate_subsurface_storage, calculate_surface_storage, calculate_water_table_depth
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

def write_array(file_name, array, *args, **kwargs):
    ext = Path(file_name).suffix[1:]
    funcs = {
        'pfb': write_array_pfb,
    }

    if ext not in funcs:
        raise Exception(f'Unknown extension: {file_name}')

    return funcs[ext](file_name, array, *args, **kwargs)


# -----------------------------------------------------------------------------

def read_array_pfb(file_name):
    from parflowio.pyParflowio import PFData
    data = PFData(file_name)
    data.loadHeader()
    data.loadData()
    return data.moveDataArray()


# -----------------------------------------------------------------------------

def write_array_pfb(file_name, array, dx=1, dy=1, dz=1):
    # Ensure this is 3 dimensions, since parflowio requires 3 dimensions.
    while array.ndim < 3:
        array = array[np.newaxis, :]

    if array.ndim > 3:
        raise Exception(f'Too many dimensions: {array.ndim}')

    from parflowio.pyParflowio import PFData
    data = PFData()
    data.setDataArray(array)
    data.setDX(dx)
    data.setDY(dy)
    data.setDZ(dz)
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

    # Handle boolean type
    lower_str = string.lower()
    if lower_str in ['true', 'false']:
        return lower_str[0] == 't'

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
            vegm_array[x - 1, y - 1, i] = elements[i + 2]

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


# -----------------------------------------------------------------------------


class DataAccessor:
    """Helper for extracting numpy array from a given run"""

    def __init__(self, run, selector=None):
        """Create DataAccessor from a Run instance"""
        self._run = run
        self._name = run.get_name()
        self._selector = selector
        self._t_padding = 5
        self._time = None
        self._ts = None
        # CLM
        self._forcing_time = 0
        self._process_id = 0
        # Initialize time
        self.time = 0

    # ---------------------------------------------------------------------------

    def _pfb_to_array(self, file_path):
        from parflowio.pyParflowio import PFData

        array = None
        if file_path:
            full_path = get_absolute_path(file_path)
            # FIXME do something with selector inside parflow-io
            pfb_data = PFData(full_path)
            pfb_data.loadHeader()
            pfb_data.loadData()
            array = pfb_data.moveDataArray()

        return array

    # ---------------------------------------------------------------------------
    # time
    # ---------------------------------------------------------------------------

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = int(t)
        self._ts = f'{self._time:0>{self._t_padding}}'

    @property
    def times(self):
        t0 = self._run.TimingInfo.StartCount
        t_start = self._run.TimingInfo.StartTime
        t_end = self._run.TimingInfo.StopTime
        t_step = self._run.TimeStep.Value
        t = t0 + t_start
        time_values = []
        while t <= t_end:
            time_values.append(int(t))
            t += t_step

        return time_values

    # ---------------------------------------------------------------------------
    # forcing time
    # ---------------------------------------------------------------------------

    @property
    def forcing_time(self):
        return self._forcing_time

    @forcing_time.setter
    def forcing_time(self, t):
        self._forcing_time = int(t)

    # ---------------------------------------------------------------------------
    # Process id
    # ---------------------------------------------------------------------------

    @property
    def process_id(self):
        return self._process_id

    @process_id.setter
    def process_id(self, t):
        self._process_id = int(t)

    # ---------------------------------------------------------------------------
    # Region selector
    # ---------------------------------------------------------------------------

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, selector):
        self._selector = selector

    # ---------------------------------------------------------------------------
    # Grid information
    # ---------------------------------------------------------------------------

    @property
    def shape(self):
        # FIXME do something with selector
        return (
            self._run.ComputationalGrid.NZ,
            self._run.ComputationalGrid.NY,
            self._run.ComputationalGrid.NX
        )

    @property
    def dx(self):
        return self._run.ComputationalGrid.DX

    @property
    def dy(self):
        return self._run.ComputationalGrid.DY

    @property
    def dz(self):
        if self._run.Solver.Nonlinear.VariableDz:
            assert self._run.dzScale.Type == 'nzList'
            dz_scale = []
            for i in range(self._run.dzScale.nzListNumber):
                dz_scale.append(self._run.Cell[str(i)]['dzScale']['Value'])
            dz_scale = np.array(dz_scale)
        else:
            dz_scale = np.ones((self._run.ComputationalGrid.NZ,))

        dz_values = dz_scale * self._run.ComputationalGrid.DZ
        return dz_values

    # ---------------------------------------------------------------------------
    # Mannings Roughness Coef
    # ---------------------------------------------------------------------------

    @property
    def mannings(self):
        return self._pfb_to_array(f'{self._name}.out.mannings.pfb')

    # ---------------------------------------------------------------------------
    # Mask
    # ---------------------------------------------------------------------------

    @property
    def mask(self):
        return self._pfb_to_array(f'{self._name}.out.mask.pfb')

    # ---------------------------------------------------------------------------
    # Slopes X Y
    # ---------------------------------------------------------------------------

    @property
    def slope_x(self):
        if self._run.TopoSlopesX.FileName is None:
            return self._pfb_to_array(f'{self._name}.out.slope_x.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopesX.FileName)

    @property
    def slope_y(self):
        if self._run.TopoSlopesY.FileName is None:
            return self._pfb_to_array(f'{self._name}.out.slope_y.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopesY.FileName)

    # ---------------------------------------------------------------------------
    # Elevation
    # ---------------------------------------------------------------------------

    @property
    def elevation(self):
        if self._run.TopoSlopes.Elevation.FileName is None:
            return self._pfb_to_array(f'{self._name}.DEM.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopes.Elevation.FileName)

    # ---------------------------------------------------------------------------
    # Computed Porosity
    # ---------------------------------------------------------------------------

    @property
    def computed_porosity(self):
        return self._pfb_to_array(f'{self._name}.out.porosity.pfb')

    # ---------------------------------------------------------------------------
    # Computed Permeability
    # ---------------------------------------------------------------------------

    @property
    def computed_permeability_x(self):
        return self._pfb_to_array(f'{self._name}.out.perm_x.pfb')

    @property
    def computed_permeability_y(self):
        return self._pfb_to_array(f'{self._name}.out.perm_y.pfb')

    @property
    def computed_permeability_z(self):
        return self._pfb_to_array(f'{self._name}.out.perm_z.pfb')

    # ---------------------------------------------------------------------------
    # Pressures
    # ---------------------------------------------------------------------------

    @property
    def pressure_initial_condition(self):
        press_type = self._run.ICPressure.Type
        if press_type == 'PFBFile':
            geom_name = self._run.ICPressure.GeomNames
            if len(geom_name) > 1:
                msg = f'ICPressure.GeomNames are set to {geom_name}'
                raise Exception(msg)
            file_name = self._run.Geom[geom_name[0]].ICPressure.FileName
            return self._pfb_to_array(file_name)
        else:
            # HydroStaticPatch, ... ?
            msg = f'Initial pressure of type {press_type} is not supported'
            raise Exception(msg)

    # ---------------------------------------------------------------------------

    @property
    def pressure_boundary_conditions(self):
        # Extract all BC names (bc[{patch_name}__{cycle_name}] = value)
        bc = {}
        patch_names = []

        # Handle patch names
        main_name = self._run.Domain.GeomName
        all_names = self._run.Geom[main_name].Patches
        patch_names.extend(all_names)

        # Extract cycle names for each patch
        for p_name in patch_names:
            cycle_name = self._run.Patch[p_name].BCPressure.Cycle
            cycle_names = self._run.Cycle[cycle_name].Names
            for c_name in cycle_names:
                key = f'{p_name}__{c_name}'
                bc[key] = self._run.Patch[p_name].BCPressure[c_name].Value

        return bc

    # ---------------------------------------------------------------------------

    @property
    def pressure(self):
        file_name = get_absolute_path(f'{self._name}.out.press.{self._ts}.pfb')
        return self._pfb_to_array(file_name)

    # ---------------------------------------------------------------------------
    # Saturations
    # ---------------------------------------------------------------------------

    @property
    def saturation(self):
        file_name = get_absolute_path(f'{self._name}.out.satur.{self._ts}.pfb')
        return self._pfb_to_array(file_name)

    # ---------------------------------------------------------------------------
    # Specific storage
    # ---------------------------------------------------------------------------

    @property
    def specific_storage(self):
        return self._pfb_to_array(f'{self._name}.out.specific_storage.pfb')

    # ---------------------------------------------------------------------------
    # Evapotranspiration
    # ---------------------------------------------------------------------------

    @property
    def et(self):
        if self._run.Solver.PrintCLM:
            # Read ET from CLM output
            return self.clm_output('qflx_evap_tot')
        else:
            # Assert that one and only one of Solver.EvapTransFile or Solver.EvapTransFileTransient is set
            assert self._run.Solver.EvapTransFile != self._run.Solver.EvapTransFileTransient, \
                'Only one of Solver.EvapTrans.FileName, Solver.EvapTransFileTransient can be set in order to ' \
                'calculate evapotranspiration'

            if self._run.Solver.EvapTransFile:
                # Read steady-state flux file
                et_data = self._pfb_to_array(self._run.Solver.EvapTrans.FileName)
            else:
                # Read current timestep from series of flux PFB files
                et_data = self._pfb_to_array(f'{self._run.Solver.EvapTrans.FileName}.{self._ts}.pfb')

        return calculate_evapotranspiration(et_data, self.dx, self.dy, self.dz)

    # ---------------------------------------------------------------------------
    # Overland Flow
    # ---------------------------------------------------------------------------

    def overland_flow(self, flow_method='OverlandKinematic', epsilon=1e-5):
        return calculate_overland_flow(self.pressure, self.slope_x, self.slope_y, self.mannings,
                                       self.dx, self.dy, flow_method=flow_method, epsilon=epsilon, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Overland Flow Grid
    # ---------------------------------------------------------------------------

    def overland_flow_grid(self, flow_method='OverlandKinematic', epsilon=1e-5):
        return calculate_overland_flow_grid(self.pressure, self.slope_x, self.slope_y, self.mannings,
                                            self.dx, self.dy, flow_method=flow_method, epsilon=epsilon, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Subsurface Storage
    # ---------------------------------------------------------------------------

    @property
    def subsurface_storage(self):
        return calculate_subsurface_storage(self.computed_porosity, self.pressure, self.saturation,
                                            self.specific_storage, self.dx, self.dy, self.dz, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Surface Storage
    # ---------------------------------------------------------------------------

    @property
    def surface_storage(self):
        return calculate_surface_storage(self.pressure, self.dx, self.dy, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Water Table Depth
    # ---------------------------------------------------------------------------

    @property
    def wtd(self):
        return calculate_water_table_depth(self.pressure, self.saturation, self.dz)

    # ---------------------------------------------------------------------------
    # CLM
    # ---------------------------------------------------------------------------

    def _clm_output_filepath(self, directory, prefix, ext):
        file_name = f'{prefix}.{self._ts}.{ext}.{self._process_id}'
        base_path = f'{self._run.Solver.CLM.CLMFileDir}/{directory}'
        return get_absolute_path(f'{base_path}/{file_name}')

    def _clm_output_bin(self, field, dtype):
        fp = self._clm_output_filepath(field, field, 'bin')
        return np.fromfile(fp, dtype=dtype, count=-1, sep='', offset=0)

    def clm_output(self, field, layer=-1):
        assert self._run.Solver.PrintCLM, 'CLM output must be enabled'
        assert field in self.clm_output_variables, f'Unrecognized variable {field}'

        if self._run.Solver.CLM.SingleFile:
            file_name = f'{self._name}.out.clm_output.{self._ts}.C.pfb'
            arr = self._pfb_to_array(f'{file_name}')

            nz = arr.shape[0]
            nz_expected = len(self.clm_output_variables) + self._run.Solver.CLM.RootZoneNZ - 1
            assert nz == nz_expected, f'Unexpected shape of CLM output, expected {nz_expected}, got {nz}'

            i = self.clm_output_variables.index(field)
            if field == 't_soil':
                if layer < 0:
                    i = layer
                else:
                    i += layer

            arr = arr[i, :, :]
        else:
            file_name = f'{self._name}.out.{field}.{self._ts}.pfb'
            arr = self._pfb_to_array(f'{file_name}')

            if field == 't_soil':
                nz = arr.shape[0]
                assert nz == self._run.Solver.CLM.RootZoneNZ, f'Unexpected shape of CLM output, expected ' \
                                                              f'{self._run.Solver.CLM.RootZoneNZ}, got {nz}'
                arr = arr[layer, :, :]

        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        return arr

    @property
    def clm_output_variables(self):
        return ('eflx_lh_tot',
                'eflx_lwrad_out',
                'eflx_sh_tot',
                'eflx_soil_grnd',
                'qflx_evap_tot',
                'qflx_evap_grnd',
                'qflx_evap_soi',
                'qflx_evap_veg',
                'qflx_tran_veg',
                'qflx_infl',
                'swe_out',
                't_grnd',
                'qflx_qirr',
                't_soil')

    @property
    def clm_output_diagnostics(self):
        return self._clm_output_filepath('diag_out', 'diagnostics', 'dat')

    @property
    def clm_output_eflx_lh_tot(self):
        return self._clm_output_bin('eflx_lh_tot', float)

    @property
    def clm_output_eflx_lwrad_out(self):
        return self._clm_output_bin('eflx_lwrad_out', float)

    @property
    def clm_output_eflx_sh_tot(self):
        return self._clm_output_bin('eflx_sh_tot', float)

    @property
    def clm_output_eflx_soil_grnd(self):
        return self._clm_output_bin('eflx_soil_grnd', float)

    @property
    def clm_output_qflx_evap_grnd(self):
        return self._clm_output_bin('qflx_evap_grnd', float)

    @property
    def clm_output_qflx_evap_soi(self):
        return self._clm_output_bin('qflx_evap_soi', float)

    @property
    def clm_output_qflx_evap_tot(self):
        return self._clm_output_bin('qflx_evap_tot', float)

    @property
    def clm_output_qflx_evap_veg(self):
        return self._clm_output_bin('qflx_evap_veg', float)

    @property
    def clm_output_qflx_infl(self):
        return self._clm_output_bin('qflx_infl', float)

    @property
    def clm_output_qflx_top_soil(self):
        return self._clm_output_bin('qflx_top_soil', float)

    @property
    def clm_output_qflx_tran_veg(self):
        return self._clm_output_bin('qflx_tran_veg', float)

    @property
    def clm_output_swe_out(self):
        return self._clm_output_bin('swe_out', float)

    @property
    def clm_output_t_grnd(self):
        return self._clm_output_bin('t_grnd', float)

    def clm_forcing(self, name):
        time_slice = self._run.Solver.CLM.MetFileNT
        prefix = self._run.Solver.CLM.MetFileName
        directory = self._run.Solver.CLM.MetFilePath
        file_index = int(self._forcing_time / time_slice)
        t0 = f'{file_index * time_slice + 1:0>6}'
        t1 = f'{(file_index + 1) * time_slice:0>6}'
        file_name = get_absolute_path(
            f'{directory}/{prefix}.{name}.{t0}_to_{t1}.pfb')

        return self._pfb_to_array(file_name)[self._forcing_time % time_slice]

    @property
    def clm_forcing_dswr(self):
        """Downward Visible or Short-Wave radiation [W/m2]"""
        return self.clm_forcing('DSWR')

    @property
    def clm_forcing_dlwr(self):
        """Downward Infa-Red or Long-Wave radiation [W/m2]"""
        return self.clm_forcing('DLWR')

    @property
    def clm_forcing_apcp(self):
        """Precipitation rate [mm/s]"""
        return self.clm_forcing('APCP')

    @property
    def clm_forcing_temp(self):
        """Air temperature [K]"""
        return self.clm_forcing('Temp')

    @property
    def clm_forcing_ugrd(self):
        """West-to-East or U-component of wind [m/s]"""
        return self.clm_forcing('UGRD')

    @property
    def clm_forcing_vgrd(self):
        """South-to-North or V-component of wind [m/s]"""
        return self.clm_forcing('VGRD')

    @property
    def clm_forcing_press(self):
        """Atmospheric Pressure [pa]"""
        return self.clm_forcing('Press')

    @property
    def clm_forcing_spfh(self):
        """Water-vapor specific humidity [kg/kg]"""
        return self.clm_forcing('SPFH')

    def _clm_map(self, root):
        if root.Type == 'Constant':
            return root.Value

        if root.Type == 'Linear':
            return (root.Min, root.Max)

        if root.Type == 'PFBFile':
            return self._pfb_to_array(root.FileName)

        return None

    def clm_map_land_fraction(self, name):
        root = self._run.Solver.CLM.Vegetation.Map.LandFrac[name]
        return self._clm_map(root)

    @property
    def clm_map_latitude(self):
        root = self._run.Solver.CLM.Vegetation.Map.Latitude
        return self._clm_map(root)

    @property
    def clm_map_longitude(self):
        root = self._run.Solver.CLM.Vegetation.Map.Longitude
        return self._clm_map(root)

    @property
    def clm_map_sand(self):
        root = self._run.Solver.CLM.Vegetation.Map.Sand
        return self._clm_map(root)

    @property
    def clm_map_clay(self):
        root = self._run.Solver.CLM.Vegetation.Map.Clay
        return self._clm_map(root)

    @property
    def clm_map_color(self):
        root = self._run.Solver.CLM.Vegetation.Map.Color
        return self._clm_map(root)
