# -*- coding: utf-8 -*-
"""export module

This module capture all core ParFlow exporters.
"""
import copy
import json
from pathlib import Path
import re

import numpy as np
import yaml

from .fs import get_absolute_path

from parflow.tools.io import read_pfb
from parflow.tools.database.core import PFDBObj
from parflow.tools.database.generated import LandCoverParamItem, CLM_KEY_DICT


# For instance: [Type: double]
TYPE_INFO_RE = re.compile(r"(^\[\w+: \w+\] )")


class SubsurfacePropertiesExporter:

    def __init__(self, run):
        self.run = run
        self.props_found = set()
        self.entries = []
        yaml_key_def = Path(__file__).parent / "ref/table_keys.yaml"
        with open(yaml_key_def, "r") as file:
            self.definition = yaml.safe_load(file)

        self.pfkey_to_alias = {}
        self.alias_to_priority = {}
        for i, (key, value) in enumerate(self.definition.items()):
            self.pfkey_to_alias[key] = value["alias"][0]
            self.alias_to_priority[value["alias"][0]] = i

        self._process()

    def _extract_sub_surface_props(self, geom_item):
        name = Path(geom_item.full_name()).suffix[1:]
        entry = {"key": name}
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
        geom_items = self.run.Geom.select("{GeomItem}")
        for item in geom_items:
            entry = self._extract_sub_surface_props(item)
            if entry is not None:
                self.entries.append(entry)

    def get_table_as_txt(self, column_separator="  ", columns_justify=True):
        header = ["key"] + list(self.props_found)
        header.sort(key=lambda alias: self.alias_to_priority[alias])
        lines = []

        # Extract column size
        sizes = {}
        for key in header:
            if columns_justify:
                sizes[key] = len(key)
                for entry in self.entries:
                    value = entry[key] if key in entry else "-"
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
                value = entry[key] if key in entry else "-"
                line.append(value.ljust(sizes[key]))
            lines.append(column_separator.join(line))

        return "\n".join(lines)

    def write_csv(self, file_path):
        data = self.get_table_as_txt(column_separator=",", columns_justify=False)
        Path(file_path).write_text(data, encoding="utf-8")

    def write_txt(self, file_path):
        data = self.get_table_as_txt()
        Path(file_path).write_text(data, encoding="utf-8")


class CLMExporter:

    def __init__(self, run):
        self.run = run

    def write(self, working_directory=".", **kwargs):
        """Method to export all clm files based on metadata

        Args:
            - working_directory='.': specifies where the files
              will be written
        """
        kwargs["working_directory"] = working_directory
        self.write_input(**kwargs)
        self.write_map(**kwargs)
        self.write_parameters(**kwargs)

    def write_allowed(self, working_directory=".", **kwargs):
        """Export files we are allowed to overwrite

        Args:
            - working_directory='.': specifies where the files
              will be written
        """
        kwargs["working_directory"] = working_directory
        to_write = [
            "input",
            "map",
            "parameters",
        ]

        for name in to_write:
            func = getattr(self, f"write_{name}")
            try:
                func(**kwargs)
            except NotOverwritableException:
                if self._changes_detected(working_directory, name):
                    # print a warning that changes will not be reflected
                    # in the calculation
                    self._print_not_written_warning(working_directory, name)

    def write_input(self, working_directory="."):
        """Method to export drv_clmin.dat file based on metadata

        Args:
            - working_directory='.': specifies where drv_climin.dat
              file will be written
        """
        output_file = self._file(working_directory, "input")

        # Make sure we don't overwrite pre-existing data
        self._ensure_writable(output_file, "input")

        clm_drv_keys = {}
        header_doc = ""
        clm_dict = self._clm_solver.Input.to_dict()

        required_input_keys = [
            "File.VegTileSpecification",
            "File.VegTypeParameter",
        ]

        # Ensure required input keys are set
        for key in required_input_keys:
            if key in clm_dict:
                continue

            # Set the default for this key
            default = self._clm_solver.Input.details(key).get("default")
            clm_dict[key] = default

        # Gather doc information to print out
        for key in clm_dict:
            old_header_doc = header_doc
            container_key = ".".join(map(str, key.split(".")[:-1]))
            header_doc = self._clm_solver.Input.doc(container_key)
            clm_key = self._clm_solver.Input.details(key).get("clm_key")
            clm_key_value = self._clm_solver.Input.value(key)
            clm_key_help = self._clm_solver.Input.doc(key)
            item = {"value": clm_key_value, "help": clm_key_help}

            clm_drv_keys.setdefault(container_key, {})[clm_key] = item
            if header_doc != old_header_doc:
                clm_drv_keys[container_key]["doc"] = header_doc

        with open(output_file, "w") as wf:
            wf.write(f"! {self._auto_generated_file_string}\n")
            wf.write(
                f"! CLM input file for {self.run.get_name()} " f"ParFlow run" + "\n"
            )
            for key, value in clm_drv_keys.items():
                doc = str(clm_drv_keys[key]["doc"]).strip().replace("\n", " ")
                wf.write("!\n")
                wf.write(f"! {doc}\n")
                wf.write("!\n")
                for sub_key, sub_value in value.items():
                    if sub_key == "doc":
                        continue

                    value = sub_value["value"]
                    help = sub_value["help"].replace("\n", " ")
                    line = f"{sub_key:<15}"
                    line += f"{value:<40}"
                    line += f"{help}\n\n"
                    wf.write(line)

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
            vegm_root_key = self._clm_solver.Vegetation.Map[token[0]]
            for item in token[1:]:
                vegm_root_key = vegm_root_key[item]
        else:
            vegm_root_key = self._clm_solver.Vegetation.Map[token]
        array = np.zeros((x, y))
        if vegm_root_key.Type == "Constant":
            array = np.full((x, y), vegm_root_key.Value)
        elif vegm_root_key.Type == "Linear":
            min_par = vegm_root_key.Min
            max_par = vegm_root_key.Max
            length = y if axis == "y" else x
            inc = (max_par - min_par) / (length - 1)
            list_par = list(np.arange(min_par, max_par + inc, inc))
            for i, v in enumerate(list_par):
                if axis == "y":
                    array[:, i] = v
                elif axis == "x":
                    array[i, :] = v
                else:
                    raise Exception(f"Axis specification error: {axis}")
        elif vegm_root_key.Type == "PFBFile":
            array = read_pfb(get_absolute_path(vegm_root_key.FileName))
            while array.ndim > 2:
                # PFB files return 3D arrays, but the data is actually 2D
                array = array[0]
        else:
            raise Exception(f"Unknown vegm type: {vegm_root_key.Type}")

        return array

    def _process_vegm_loc(
        self, vegm_array, latitude=True, lat_axis="y", longitude=True, long_axis="x"
    ):

        # Need to better expose the options of which axis to use
        # Maybe have it as an extra key?
        x, y, z = vegm_array.shape
        if latitude:
            vegm_array[:, :, 0] = self._process_vegm("Latitude", x, y, lat_axis)

        if longitude:
            vegm_array[:, :, 1] = self._process_vegm("Longitude", x, y, long_axis)

    def _process_vegm_soil(
        self,
        vegm_array,
        sand=True,
        sand_axis="y",
        clay=True,
        clay_axis="y",
        color=True,
        color_axis="y",
    ):

        # Need to better expose the options of which axis to use
        # Maybe have it as an extra key?
        x, y, z = vegm_array.shape
        if sand:
            vegm_array[:, :, 2] = self._process_vegm("Sand", x, y, sand_axis)

        if clay:
            vegm_array[:, :, 3] = self._process_vegm("Clay", x, y, clay_axis)

        if color:
            vegm_array[:, :, 4] = self._process_vegm("Color", x, y, color_axis).astype(
                int
            )

    def write_map(self, vegm_array=None, working_directory=".", dec_round=3):
        """Method to export drv_vegm.dat file based on keys or a 3D array of
           data

        Args:
            - vegm_array=None: optional full array with gridded properties
              to export. If None, the array will be generated automatically
              from the run object.
            - working_directory='.': specifies where drv_vegm.dat
              file will be written
            - dec_round=3: sets the maximum decimal rounding for the lat, long,
              sand, and clay parameters.
        """
        output_file = self._file(working_directory, "map")

        # Make sure we don't overwrite pre-existing data
        self._ensure_writable(output_file, "map")

        first_line = (
            " x  y  lat    lon    sand clay color  fractional "
            "coverage of grid by vegetation class (Must/Should Add "
            f"to 1.0) -- {self._auto_generated_file_string}"
        )
        second_line = "       (Deg)	 (Deg)  (%/100)   index"

        land_col_map = {"column": "land cover type"}
        land_covers = self._veg_params.LandNames
        if vegm_array is None:
            x = self.run.ComputationalGrid.NX
            y = self.run.ComputationalGrid.NY
            if x is None or y is None:
                raise Exception("Computational grid has not been set")

            vegm_array = np.zeros((x, y, 5))
            self._process_vegm_loc(vegm_array)
            self._process_vegm_soil(vegm_array)
            # CLM handles exactly 18 land cover types as a default
            if len(land_covers) > 18:
                print(
                    f"WARNING: CLM must be recompiled to accommodate "
                    f"{len(land_covers)} land cover types."
                )
            for name in land_covers:
                vegm_array = np.dstack(
                    (vegm_array, self._process_vegm(["LandFrac", name], x, y))
                )

        with open(output_file, "w") as wf:
            wf.write(first_line + "\n")
            if vegm_array.shape[2] < 23:
                print(
                    f"{len(land_covers)} land cover types specified. "
                    f"Filling in zeros for {23 - vegm_array.shape[2]} "
                    f"land cover types."
                )
            wf.write(second_line + "\n")
            for i in range(vegm_array.shape[0]):
                for j in range(vegm_array.shape[1]):
                    elements = [str(i + 1), str(j + 1)]
                    for k in range(max(vegm_array.shape[2], 23)):
                        if k == 4:
                            # dealing with color (needs to be int)
                            elements.append(f"{int(vegm_array[i, j, k]):<7}")
                        elif k < vegm_array.shape[2]:
                            s = f"{round(vegm_array[i, j, k], dec_round):<7}"
                            elements.append(s)
                            if k > 4:
                                land_col_map.update({k - 4: land_covers[k - 5]})
                        else:
                            elements.append("0.0    ")
                    wf.write("   " + " ".join(elements[:]) + "\n")

            print("Land cover column mapping")
            for key, value in land_col_map.items():
                print(f"{key:<6}: {value}")

        return self

    def _land_cover_docs(self, clm_keys):
        # Get the help for the land cover without type info
        item = LandCoverParamItem()
        path_to_item = self._veg_params_path + [".{land_cover_name}"]
        relative_start_ind = len(path_to_item)
        doc_strings = {}
        for key in clm_keys:
            relative_path = ".".join(CLM_KEY_DICT[key][relative_start_ind:])
            doc = item.details(relative_path).get("help", "").strip()
            # Remove the type information from the front
            doc = re.sub(TYPE_INFO_RE, "", doc)
            doc_strings[key] = doc

        return doc_strings

    @property
    def _land_cover_descriptions(self):
        path = "/".join(self._veg_params_path)
        ret = {}
        for name in self._land_names:
            desc = self.run.value(f"{path}/{name}/Description")
            ret[name] = desc if desc else ""
        return ret

    def write_parameters(self, vegp_data=None, working_directory="."):
        """Method to export drv_vegp.dat file based on dictionary of data

        Args:
            - vegp_data=None: optional full dict to export. If None, it will
              automatically be generated from the run object.
            - working_directory='.': specifies where drv_vegp.dat
              file will be written
        """
        output_file = self._file(working_directory, "parameters")

        # Make sure we don't overwrite pre-existing data
        self._ensure_writable(output_file, "parameters")

        if vegp_data is None:
            # Extract the vegp data from the run object
            vegp_data = self._generate_vegp_data()

        doc_strings = self._land_cover_docs(vegp_data.keys())
        descriptions = self._land_cover_descriptions

        # This will make sure there are at least 18 values for
        # everything.
        vegp_data = copy.deepcopy(vegp_data)
        self._resize_vegp_data(vegp_data)

        output = f"! {self._auto_generated_file_string}\n"
        output += self._vegp_header

        # Make the header based upon the land names
        for i, name in enumerate(self._land_names, 1):
            description = descriptions[name]
            output += f"! {i:>2} {name} {description}\n"

        # Fill in not set values
        for i in range(len(self._land_names), self._min_num_land_covers):
            output += f"! {i + 1:>2} not_set\n"

        output += "!" + "=" * 72 + "\n"

        for key, val in vegp_data.items():
            doc = doc_strings[key]
            output += "!\n"
            output += f"{key:<15}{doc}\n"
            output += f'{" ".join(map(str, val))}\n'

        Path(output_file).write_text(output)
        return self

    def _resize_vegp_data(self, data):
        defaults = self._default_vegp_values
        for key, val in data.items():
            num_to_fill = self._min_num_land_covers - len(val)
            val.extend([defaults[key]] * num_to_fill)

    @property
    def _min_num_land_covers(self):
        return 18

    @property
    def _veg_params_path(self):
        return ["Solver", "CLM", "Vegetation", "Parameters"]

    @property
    def _clm_solver(self):
        return self.run.Solver.CLM

    @property
    def _veg_params(self):
        return self._clm_solver.Vegetation.Parameters

    @property
    def _veg_map(self):
        return self._clm_solver.Vegetation.Map

    @property
    def _land_names(self):
        names = self._veg_params.LandNames
        return names if isinstance(names, list) else names.split()

    @property
    def _vegp_header_file(self):
        return Path(__file__).parent / "ref/vegp_header.txt"

    @property
    def _default_vegp_values_file(self):
        return Path(__file__).parent / "ref/default_vegp_values.json"

    @property
    def _vegp_header(self):
        return self._vegp_header_file.read_text()

    @property
    def _default_vegp_values(self):
        if not hasattr(self, "_default_vegp_values_data"):
            with open(self._default_vegp_values_file, "r") as rf:
                self._default_vegp_values_data = json.load(rf)
        return copy.deepcopy(self._default_vegp_values_data)

    def _generate_vegp_data(self):
        # Get the keys from the default values
        keys = list(self._default_vegp_values.keys())
        land_items = self._veg_params.select("{LandCoverParamItem}")

        result = {}
        for key in keys:
            name = CLM_KEY_DICT[key][-1]
            result[key] = [item[name] for item in land_items]
        return result

    @property
    def _auto_generated_file_string(self):
        # This will be placed in the first line of every file that
        # was automatically generated by us.
        return "Automatically generated by pftools python"

    def _contains_auto_generated_string(self, file_name):
        # Check if the file was automatically generated by us
        with open(file_name, "r") as rf:
            return self._auto_generated_file_string in next(rf)

    def _writable(self, file_name, type):
        # Make sure we are allowed to overwrite this file
        if not Path(file_name).exists():
            # We don't need to worry about overwriting anything...
            return True

        # If the file contains the auto generated string, it is okay to
        # write over it.
        if self._contains_auto_generated_string(file_name):
            return True

        # Otherwise, check the settings
        key = self._overwrite_settings_map[type]
        return getattr(self._clm_solver, key)

    @property
    def _overwrite_settings_map(self):
        return {
            "input": "OverwriteDrvClmin",
            "parameters": "OverwriteDrvVegp",
            "map": "OverwriteDrvVegm",
        }

    def _ensure_writable(self, file_name, type):
        # Make sure we can write to this file.
        if not self._writable(file_name, type):
            key = self._overwrite_settings_map[type]
            msg = (
                f"{file_name} already exists, and {key} is false. " "Cannot overwrite."
            )
            raise NotOverwritableException(msg)

    def _changes_detected(self, working_directory, type):
        # Check the history, and if there are changes, return True.
        items_to_check = {
            "input": self._clm_solver.Input,
            "map": self._clm_solver.Vegetation.Map,
            "parameters": self._clm_solver.Vegetation.Parameters,
        }
        if self._has_changes(items_to_check[type]):
            return True

        # Check the import path vs the write path. If they are different,
        # return True.
        import_path = self._import_paths.get(type)
        write_path = str(self._file(working_directory, type))
        return import_path != write_path

    def _file(self, working_directory, type):
        # Get a full path to the file of the specified type
        file_map = {
            "input": "drv_clmin.dat",
            "parameters": self._clm_solver.Input.File.VegTypeParameter,
            "map": self._clm_solver.Input.File.VegTileSpecification,
        }
        working_directory = get_absolute_path(working_directory)
        return Path(working_directory) / file_map[type]

    def _print_not_written_warning(self, working_directory, type):
        # Print a warning that the file was not written.
        write_path = self._file(working_directory, type)
        key_name = self._overwrite_settings_map[type]
        print(
            f"Warning: {write_path} will not be overwritten unless "
            f"{key_name} is set to True. Changes to the file may not "
            "be reflected in the calculation"
        )

    @property
    def _using_clm(self):
        # We can add other checks here if needed
        return self.run.Solver.LSM == "CLM"

    @property
    def can_export(self):
        if self._using_clm:
            land_param_items = self._veg_params.select("{LandCoverParamItem}")
            land_map_items = self._veg_map.select("LandFrac/{LandFracCoverMapItem}")
            return (
                land_param_items
                and land_map_items
                and all(x is not None for x in land_param_items + land_map_items)
            )

        return False

    @property
    def _import_paths(self):
        return self.run.__dict__.setdefault("_import_paths_", {})

    @staticmethod
    def _has_changes(root):
        if not isinstance(root, PFDBObj):
            return False

        for key in root.keys():
            item = root[key]

            if not isinstance(item, PFDBObj):
                continue

            if hasattr(item, "_details_"):
                for details in item._details_.values():
                    if details.get("history"):
                        return True

            if CLMExporter._has_changes(item):
                return True

        return False


class NotOverwritableException(Exception):
    pass
