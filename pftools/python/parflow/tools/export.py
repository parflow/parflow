# -*- coding: utf-8 -*-
"""export module

This module capture all core ParFlow exporters.
"""
from pathlib import Path
import yaml


class SubsurfacePropertiesExporter:

    def __init__(self, run):
        self.run = run
        self.props_found = set()
        self.entries = []
        yaml_key_def = Path(__file__).parent / 'ref/table_keys.yaml'
        with open(yaml_key_def, 'r') as file:
            self.definition = yaml.safe_load(file)

        self.pfkey_to_alias = {}
        self.alias_to_priority = {}
        for i, (key, value) in enumerate(self.definition.items()):
            self.pfkey_to_alias[key] = value['alias'][0]
            self.alias_to_priority[value['alias'][0]] = i

        self._process()

    def _extract_sub_surface_props(self, geom_item):
        name = Path(geom_item.full_name()).suffix[1:]
        entry = {'key': name}
        has_data = False
        for key in self.pfkey_to_alias:
            value = geom_item.value(key, skip_default=True)
            if value is not None:
                has_data = True
                alias = self.pfkey_to_alias[key]
                self.props_found.add(alias)
                entry[alias] = str(value)

        return entry if has_data else None

    def _process(self):
        self.entries = []
        self.props_found.clear()
        geom_items = self.run.Geom.select('{GeomItem}')
        for item in geom_items:
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
        data = self.get_table_as_txt(column_separator=',',
                                     columns_justify=False)
        Path(file_path).write_text(data, encoding='utf-8')

    def write_txt(self, file_path):
        data = self.get_table_as_txt()
        Path(file_path).write_text(data, encoding='utf-8')
