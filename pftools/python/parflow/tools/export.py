# -*- coding: utf-8 -*-
"""export module

This module capture all core ParFlow exporters.
"""
import os
import yaml
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
        name = geomItem.get_full_key_name().split('.')[-1]
        entry = {'key': name}
        has_data = False
        for key in self.pfkey_to_alias:
            value = geomItem.get(key, skip_default=True)
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

def time_helper(name, time):
    td_dict = {}
    time_split = time.split('-')
    if name == 'StartDate':
        td_dict['syr'] = int(time_split[0])
        td_dict['smo'] = int(time_split[1])
        td_dict['sda'] = int(time_split[2])
    if name == 'StartTime':
        td_dict['shr'] = int(time_split[0])
        td_dict['smn'] = int(time_split[1])
        td_dict['sss'] = int(time_split[2])
    if name == 'StopDate':
        td_dict['eyr'] = int(time_split[0])
        td_dict['emo'] = int(time_split[1])
        td_dict['eda'] = int(time_split[2])
    if name == 'StopTime':
        td_dict['ehr'] = int(time_split[0])
        td_dict['emn'] = int(time_split[1])
        td_dict['ess'] = int(time_split[2])

    return td_dict


class CLMDriverExporter:

    def __init__(self, run):
        self.run = run

    def export_drv_clmin(self, working_directory='.'):
        clm_drv_keys = {}
        drv_clmin_ref = os.path.join(
            os.path.dirname(__file__), 'ref/drv_clmin.dat')
        drv_key_dict = self.run.get_key_dict()

        for key, value in drv_key_dict.items():
            if key.startswith('Metadata.CLM'):
                if key.split('.')[-1][0].isupper():
                    clm_drv_keys.update(time_helper(key.split('.')[-1], value))
                else:
                    clm_drv_keys.update({key.split('.')[-1]: value})

        cp(drv_clmin_ref, working_directory)
        drv_clmin_file = os.path.join(get_absolute_path(working_directory), 'drv_clmin.dat')

        with open(drv_clmin_ref, 'r') as fin:
            with open(drv_clmin_file, 'w') as fout:
                file_lines = fin.readlines()
                for line in file_lines:
                    if line[0].islower():
                        clm_var_name = line.split()[0]
                        extra_space = len(line.split()[1]) - len(str(clm_drv_keys[clm_var_name]))
                        if extra_space > 0:
                            fout.write(line.replace(f'{line.split()[1]}',
                                                    f'{clm_drv_keys[clm_var_name]}'+' '*abs(extra_space)))
                        else:
                            fout.write(line.replace(f'{line.split()[1]}'+' '*abs(extra_space),
                                                    f'{clm_drv_keys[clm_var_name]}'))
                    else:
                        fout.write(line)
