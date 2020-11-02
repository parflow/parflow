# -*- coding: utf-8 -*-
"""export module

This module capture all core ParFlow exporters.
"""
import os
import yaml
import numpy as np
from .fs import cp, get_absolute_path

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper


class SubsurfacePropertiesExporter:

    def __init__(self, run):
        self.run = run
        self.props_found = set()
        self.entries = []
        yaml_key_def = os.path.join(
            os.path.dirname(__file__), 'ref/table_keys.yaml')
        with open(yaml_key_def, 'r') as file:
            self.definition = yaml.safe_load(file)

        self.pfkey_to_alias = {}
        self.alias_to_priority = {}
        priority = 0
        for key, value in self.definition.items():
            priority += 1
            self.pfkey_to_alias[key] = value['alias'][0]
            self.alias_to_priority[value['alias'][0]] = priority

        self._process()

    def _extract_sub_surface_props(self, geomItem):
        name = geomItem.full_name().split('.')[-1]
        entry = {'key': name}
        has_data = False
        for key in self.pfkey_to_alias:
            value = geomItem.get_value(key, skip_default=True)
            if value is not None:
                has_data = True
                alias = self.pfkey_to_alias[key]
                self.props_found.add(alias)
                entry[alias] = str(value)

        if has_data:
            return entry
        return None

    def _process(self):
        self.entries = []
        self.props_found.clear()
        geomItems = self.run.Geom.get_selection_from_location('{GeomItem}')
        for item in geomItems:
            entry = self._extract_sub_surface_props(item)
            if entry is not None:
                self.entries.append(entry)

    def get_table_as_txt(self, column_separator='  ', columns_justify=True):
        header = ['key'] + list(self.props_found)
        header.sort(key=lambda alias: self.alias_to_priority[alias])
        lines = []

        # Extract column size
        sizes = {}
        for key in header:
            if columns_justify:
                sizes[key] = len(key)
                for entry in self.entries:
                    value = entry[key] if key in entry else '-'
                    sizes[key] = max(sizes[key], len(value))
            else:
                sizes[key] = 0

        # Header
        line = []
        for key in header:
            line.append(key.ljust(sizes[key]))
        lines.append(column_separator.join(line))

        # Content
        for entry in self.entries:
            line = []
            for key in header:
                value = entry[key] if key in entry else '-'
                line.append(value.ljust(sizes[key]))
            lines.append(column_separator.join(line))

        return '\n'.join(lines)

    def write_csv(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(self.get_table_as_txt(column_separator=',',
                                             columns_justify=False))

    def write_txt(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(self.get_table_as_txt())


class CLMExporter:

    def __init__(self, run):
        self.run = run

    def export_drv_clmin(self, working_directory='.'):
        """Method to export drv_clmin.dat file based on metadata

        Args:
            - working_directory='.': specifies where drv_climin.dat
              file will be written
        """
        clm_drv_keys = {}
        header_doc = ''
        clm_dict = self.run.Solver.CLM.Input.to_dict()
        drv_clmin_file = os.path.join(get_absolute_path(working_directory),
                                      'drv_clmin.dat')

        for key in clm_dict.keys():
            old_header_doc = header_doc
            container_key = '.'.join([str(elem)
                                      for elem in key.split('.')[:-1]])
            header_doc = self.run.Solver.CLM.Input.get_help(container_key)
            clm_key = self.run.Solver.CLM.Input.get_detail(key, 'clm_key')
            clm_key_value = self.run.Solver.CLM.Input.get_value(key)
            clm_key_help = self.run.Solver.CLM.Input.get_help(key)
            if header_doc == old_header_doc:
                clm_drv_keys[container_key].update({clm_key: [clm_key_value,
                                                              clm_key_help]})
            else:
                clm_drv_keys.update({container_key: {
                                        'doc': header_doc,
                                         clm_key: [clm_key_value,
                                                   clm_key_help]}})

        with open(drv_clmin_file, 'w') as fout:
            fout.write(f'! CLM input file for {self.run.get_name()} '
                       f'ParFlow run' + '\n')
            for key, value in clm_drv_keys.items():
                fout.write('!' + '\n')
                fout.write('! ' + str(clm_drv_keys[key]["doc"])
                           .strip(' \n\t') + '\n')
                fout.write('!' + '\n')
                for sub_key, sub_value in value.items():
                    if sub_key != 'doc':
                        line = sub_key.ljust(15, ' ')
                        line += str(sub_value[0]).ljust(40, ' ')
                        line += str(sub_value[1])
                        fout.write(line)

        return self

    def _process_vegm(self, token, x, y, axis=None):
        """Method to convert veg mapping keys to array for drv_vegm.dat file

        Args:
            - token: string or list to get keys to process
            - x: x-dimension of array
            - y: y-dimension of array
            - axis=None: axis direction of linear increase.
              can be 'x' or 'y'
        """
        if isinstance(token, list) and len(token) > 1:
            vegm_root_key = self.run.Solver.CLM.Vegetation.Map[token[0]]
            for item in token[1:]:
                vegm_root_key = vegm_root_key[item]
        else:
            vegm_root_key = self.run.Solver.CLM.Vegetation.Map[token]
        array = np.zeros((x, y))
        if vegm_root_key.Type == 'Constant':
            array = np.full((x, y), vegm_root_key.Value)
        if vegm_root_key.Type == 'Linear':
            min_par = vegm_root_key.Min
            max_par = vegm_root_key.Max
            length = y if axis == 'y' else x
            inc = (max_par - min_par) / (length - 1)
            list_par = list(np.arange(min_par, max_par + inc, inc))
            for i in range(len(list_par)):
                if axis == 'y':
                    array[:, i] = list_par[i]
                elif axis == 'x':
                    array[i, :] = list_par[i]
                else:
                    print('Axis specification error')
        if vegm_root_key.Type == 'Matrix':
            # used only for veg mapping for land use
            array = vegm_root_key.Matrix
        if vegm_root_key.Type == 'PFBFile':
            # TODO
            pass

        return array

    def _process_vegm_loc(self, vegm_array, latitude=True, lat_axis='y',
                          longitude=True, long_axis='x'):

        # Need to better expose the options of which axis to use - maybe have it as an extra key?
        z, y, x = vegm_array.shape
        if latitude is True:
            vegm_array[:, :, 0] = self._process_vegm('Latitude', x, y, lat_axis)

        if longitude is True:
            vegm_array[:, :, 1] = self._process_vegm('Longitude', x, y, long_axis)

        return

    def _process_vegm_soil(self, vegm_array, sand=True, sand_axis='y', clay=True,
                           clay_axis='y', color=True, color_axis='y'):

        # Need to better expose the options of which axis to use - maybe have it as an extra key?
        z, y, x = vegm_array.shape
        if sand is True:
            vegm_array[:, :, 2] = self._process_vegm('Sand', x, y, sand_axis)

        if clay is True:
            vegm_array[:, :, 3] = self._process_vegm('Clay', x, y, clay_axis)

        if color is True:
            vegm_array[:, :, 4] = self._process_vegm('Color', x, y, color_axis).astype(int)

        return

    def export_drv_vegm(self, from_keys=True, vegm_array=None,
                        out_file='drv_vegm.dat', working_directory='.', dec_round=3):
        """Method to export drv_vegm.dat file based on keys or a 3D array of data

        Args:
            - from_keys=True: will generate the vegetation parameters from
              the keys set in the ParFlow run if set to True.
            - vegm_array=None: optional full array with gridded properties
              that needs to be passed in if from_keys is False.
            - out_file='drv_vegm.dat': Name of the output vegetation mapping file.
            - working_directory='.': specifies where drv_vegm.dat
              file will be written
            - dec_round=3: sets the maximum decimal rounding for the lat, long,
              sand, and clay parameters.
        """

        drv_vegm_file = os.path.join(get_absolute_path(working_directory), str(out_file))
        first_line = ' x  y  lat    lon    sand clay color  fractional coverage' \
                     ' of grid by vegetation class (Must/Should Add to 1.0)'
        second_line = '       (Deg)	 (Deg)  (%/100)   index'

        # if building table from keys
        if from_keys is True:
            land_col_map = {'column': 'land cover type'}
            x = self.run.ComputationalGrid.NX
            y = self.run.ComputationalGrid.NY
            vegm_array = np.zeros((x, y, 5))
            self._process_vegm_loc(vegm_array)
            self._process_vegm_soil(vegm_array)
            land_covers = self.run.Solver.CLM.Vegetation.Parameters.LandNames
            # CLM handles exactly 18 land cover types as a default
            if len(land_covers) > 18:
                print(f'WARNING: CLM must be recompiled to accommodate '
                      f'{len(land_covers)} land cover types.')
            for name in land_covers:
                vegm_array = np.dstack((vegm_array,
                          self._process_vegm([name, 'LandFrac'], x, y)))

        with open(drv_vegm_file, 'w') as fout:
            fout.write(first_line + '\n')
            if vegm_array.shape[2] < 23:
                print(f'{len(land_covers)} land cover types specified. '
                      f'Filling in zeros for {23 - vegm_array.shape[2]} '
                      f'land cover types.')
            fout.write(second_line + '\n')
            for i in range(vegm_array.shape[0]):
                for j in range(vegm_array.shape[1]):
                    line_elements = [str(i+1), str(j+1)]
                    for k in range(max(vegm_array.shape[2], 23)):
                        if k == 4:
                            # dealing with color (needs to be int)
                            line_elements.append(str(int(vegm_array[i, j, k])).ljust(7))
                        elif k < vegm_array.shape[2]:
                            line_elements.append(str(round(vegm_array[i, j, k], dec_round)).ljust(7))
                            if k > 4:
                                land_col_map.update({k-4: land_covers[k-5]})
                        else:
                            line_elements.append('0.0    ')
                    fout.write('   ' + ' '.join(line_elements[:]) + '\n')

            print('Land cover column mapping')
            for key, value in land_col_map.items():
                print(f'{str(key).ljust(6)}: {value}')

        return self

    def export_drv_vegp(self, vegp_data, working_directory='.'):
        """Method to export drv_vegp.dat file based on dictionary of data

        Args:
            - working_directory='.': specifies where drv_vegp.dat
              file will be written
        """
        # TODO: add ability to convert from keys, remove use of reference file
        drv_vegp_ref = os.path.join(
            os.path.dirname(__file__), 'ref/drv_vegp.dat')

        drv_vegp_file = os.path.join(get_absolute_path(working_directory), 'drv_vegp.dat')
        var_name = None

        with open(drv_vegp_ref, 'r') as fin:
            with open(drv_vegp_file, 'w') as fout:
                for line in fin.readlines():
                    if line[0].islower():
                        var_name = line.split()[0]
                    elif var_name is not None and line[0] != '!':
                        line = ' '.join(list(map(str, vegp_data[var_name]))) + '\n'
                        var_name = None
                    fout.write(line)

        return self
