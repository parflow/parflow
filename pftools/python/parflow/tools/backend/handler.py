from abc import abstractmethod
from typing import Optional, Iterable

import xarray as xr
import numpy as np
from xarray.structure.alignment import AlignmentError
from xarray.core import indexing

from parflow.tools.backend.backend_array import ParflowBackendArray
from parflow.tools.backend.generic import (
    PfbInfo,
    Run,
    TimeInfo,
    VariableSelector,
    LoadingOption,
)


def verbose_merge(objects: Iterable[xr.DataArray | xr.Dataset | xr.DataTree]):
    ds_final = xr.Dataset()
    for obj in objects:
        try:
            xr.align(ds_final, obj, join="exact", exclude=["time"])
        except AlignmentError:
            print(
                f"Error while adding: {obj} \nto: {ds_final}. This may be due to data from different simulations or "
                f"to files from the same simulation with different headers (in particular x0, y0, z0, for example)."
            )
            raise
        ds_final = xr.merge([ds_final, obj], join="outer")
    return ds_final


def get_z_values(
    var: xr.Variable, run: Optional[Run], z_first: bool
) -> tuple[str, np.array, dict]:
    """Return z values associated with the given variable and run"""
    if z_first:
        if len(var.shape) == 4:  # time, z, y, x
            len_z = var.shape[1]
        else:  # z, y, x
            len_z = var.shape[0]
    else:  # time, y, x, z or y, x, z
        len_z = var.shape[-1]

    if (
        run is None or len(run.data_accessor.dz) != len_z
    ):  # pfb only or clm (z = clm_variables)
        z = np.arange(len_z)
        return "z", z, dict(units="layer")

    return "z", np.cumsum(run.data_accessor.dz[::-1])[::-1], dict(units="meters")


def variable_from_files(
    list_pfb_info: PfbInfo | list[PfbInfo],
    loading_option: LoadingOption,
    z_is: str = "z",
) -> xr.Variable:
    """
    Create a xr.variable from pfb files.
    """
    backend_array = ParflowBackendArray(
        list_pfb_info,
        z_first=loading_option.z_first,
        z_is=z_is,
        replace_fill_value=loading_option.replace_fill_value,
    )

    chunks = {d: c for d, c in zip(backend_array.dims, backend_array.default_chunk)}

    data = indexing.LazilyIndexedArray(backend_array)

    return xr.Variable(backend_array.dims, data).chunk(chunks)


def load_single_file(pfb_info: PfbInfo, loading_option: LoadingOption) -> xr.DataArray:
    """
    Load a `pfb` file (stored in a pfb_info) in a xr.DataArray
    """
    var = variable_from_files(pfb_info, loading_option, z_is="z")

    header = pfb_info.get_header()
    coords = dict()

    y = header.get_centered_y()
    coords["y"] = ("y", y, dict(units="meters"))

    x = header.get_centered_x()
    coords["x"] = ("x", x, dict(units="meters"))

    if len(var.shape) > 2:
        coords["z"] = get_z_values(var, pfb_info.run, loading_option.z_first)

    return xr.DataArray(var, coords=coords)


def load_sequence_of_pfb(
    list_pfb_info: list[PfbInfo], loading_option: LoadingOption
) -> xr.DataArray:
    """
    Load a sequence of `pfb` files (stored in pfb_info) in a xr.DataArray
    """
    # We remove duplicate time_delta by keeping only the last item encountered.
    dict_pfb_info = {pfb_info.time: pfb_info for pfb_info in list_pfb_info}

    # We recreate the list and take the opportunity to sort it.
    list_pfb_info = sorted(dict_pfb_info.values(), key=lambda x: x.time)

    var = variable_from_files(list_pfb_info, loading_option, z_is="z")

    header = list_pfb_info[0].get_header()

    coords = dict()

    y = header.get_centered_y()
    coords["y"] = ("y", y, dict(units="meters"))

    x = header.get_centered_x()
    coords["x"] = ("x", x, dict(units="meters"))

    if "z" in var.dims:
        coords["z"] = get_z_values(
            var, run=list_pfb_info[0].run, z_first=loading_option.z_first
        )

    if loading_option.compute_time:
        times = np.array([pfb_info.time for pfb_info in list_pfb_info])
    else:
        times = np.array([pfb_info.timestep for pfb_info in list_pfb_info])

    coords["time"] = ("time", times, {})

    return xr.DataArray(var, coords=coords)


def load_sequence_of_forcing(
    list_pfb_info: list[PfbInfo], loading_option: LoadingOption
) -> xr.DataArray:
    """
    Load a sequence of `pfb` forcing files, ie z_is == "time" (stored in pfb_info) in a xr.DataArray
    """
    # We recreate the list and take the opportunity to sort it.
    dict_pfb_info = {pfb_info.time[0]: pfb_info for pfb_info in list_pfb_info}

    list_pfb_info = sorted(dict_pfb_info.values(), key=lambda x: x.time[0])

    var = variable_from_files(list_pfb_info, loading_option, z_is="time")

    coords = dict()
    header = list_pfb_info[0].get_header()

    y = header.get_centered_y()
    coords["y"] = ("y", y, dict(units="meters"))

    x = header.get_centered_x()
    coords["x"] = ("x", x, dict(units="meters"))

    if loading_option.compute_time:
        times = np.concatenate([pfb_info.time for pfb_info in list_pfb_info])
    else:
        times = np.concatenate(
            [
                list(range(pfb_info.timestep[0], pfb_info.timestep[1] + 1))
                for pfb_info in list_pfb_info
            ]
        )

    coords["time"] = ("time", times, {})

    return xr.DataArray(var, coords=coords)


class FileHandler:
    """
    Abstract class of Handler. A Handler must implement several methods:
    - A method "try_to_add" to identify the files it will manage.
    - A method “load” to prepare variables and return a dictionary name -> xr.DataArray:
    - An intern method "_update_time" to update pfb_info time
    - An intern method "_valid_file" to test if a file is accepted by this Handler
    - An intern method "_add_to_files" to update pfb_info values from filename (name, timestep, simulation ...)
    and save it internally.
    """

    def __init__(self):
        self.file_count = 0

    @abstractmethod
    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        """Test whether the file given as input is accepted by this Handler.

        :param pfb_info: PfbInfo to test
        :return: True if accepted
        """
        pass

    @abstractmethod
    def _add_to_files(self, pfb_info: PfbInfo):
        """Update the PfbInfo attributes from the pfb_info.basename_split and store it internally
        (usually in self.content).

        :param pfb_info: PfbInfo to update and store
        :return: None
        """
        pass

    def _update_time(self, pfb_info: PfbInfo, time_info: TimeInfo):
        """Update the PfbInfo time attribute from the pfb_info.timestep, pfb_info.run and time_info.
        Only implemented in temporal Handler.

        :param pfb_info: PfbInfo to update
        :param time_info: TimeInfo given by the open_dataset
        :return: None
        """
        pass

    def try_to_add(
        self, pfb_info: PfbInfo, runs: dict[str, Run], time_info: TimeInfo
    ) -> bool:
        """The first of two methods called by open_dataset. It tests whether the Handler accepts the given file.
        If so, it updates and saves it.

        :param pfb_info: PfbInfo to test
        :param runs: dictionary pfidb_filename -> run, used to update pfb_info.run
        :param time_info: TimeInfo given by the open_dataset
        :return: True if the file is accepted
        """
        if self._valid_file(pfb_info):
            # if pfb_info.pfidb is not None and pfb_info.sim_name is not None:
            #     pfidb_sim_name = os.path.basename(pfb_info.pfidb).split(".")[0]
            #     if pfidb_sim_name != pfb_info.sim_name:
            #         return True
            self._add_to_files(pfb_info)
            pfb_info.set_run(runs)
            self._update_time(pfb_info, time_info)
            self.file_count += 1
            return True
        return False

    @abstractmethod
    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        """The last of the two methods called directly by open_dataset.
        It initiates the creation of xr.DataArray for the variables it contains.

        :param variable_selector: VariableSelector used to select the variables to used.
        :param loading_option: Data loading import option
        :return: Dictionary var_name -> xr.DataArray
        """
        pass


class InputHandler(FileHandler):
    """Input variables: Static input variables, such as Porosity, Ksat..."""

    def __init__(self):
        super().__init__()
        self.content: list[PfbInfo] = []

    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        return len(pfb_info.basename_split) == 2

    def _add_to_files(self, pfb_info: PfbInfo):
        pfb_info.name = pfb_info.basename_split[0]
        self.content.append(pfb_info)

    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        # Select Variables
        self.content = [
            pfb_info
            for pfb_info in self.content
            if variable_selector.is_accepted(pfb_info)
        ]

        # Remove duplicates files
        dic = {}
        for pfb_info in self.content:
            l = dic.get(pfb_info.name, [])
            l.append(pfb_info)
            dic[pfb_info.name] = l

        new_content = []
        for name, l in dic.items():
            if len(l) > 1:
                print(
                    f"Removing {len(l[:-1])} duplicate files for the variable {name}."
                )
            new_content.append(l[-1])

        self.content = new_content

        # Generate dict name -> DataArray
        da_dict = {
            f"input_{pfb_info.name}": load_single_file(pfb_info, loading_option)
            for pfb_info in self.content
        }
        return verbose_merge([da.rename(name) for name, da in da_dict.items()])


class OutputHandler(FileHandler):
    """Output variables: Static output variables, exported by Parflow, such as Porosity, Permeability, Slope..."""

    def __init__(self):
        super().__init__()
        self.content: list[PfbInfo] = []

    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        return len(pfb_info.basename_split) == 4 and pfb_info.basename_split[1] == "out"

    def _add_to_files(self, pfb_info: PfbInfo):
        pfb_info.sim_name = pfb_info.basename_split[0]
        pfb_info.name = pfb_info.basename_split[2]
        self.content.append(pfb_info)

    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        # Select variables
        self.content = [
            pfb_info
            for pfb_info in self.content
            if variable_selector.is_accepted(pfb_info)
        ]

        # Remove duplicates files
        dic = {}
        for pfb_info in self.content:
            l = dic.get(pfb_info.name, [])
            l.append(pfb_info)
            dic[pfb_info.name] = l

        new_content = []
        for name, l in dic.items():
            if len(l) > 1:
                print(
                    f"Removing {len(l[:-1])} duplicate files for the variable {name}."
                )
            new_content.append(l[-1])

        self.content = new_content

        # Generate dict name -> DataArray
        da_dict = {
            f"{pfb_info.name}": load_single_file(pfb_info, loading_option)
            for pfb_info in self.content
        }
        return verbose_merge([da.rename(name) for name, da in da_dict.items()])


class TemporalOutputHandler(FileHandler):
    """Temporal output: Temporal output variables, exported by Parflow, such as press, evaptrans, satur..."""

    def __init__(self):
        super().__init__()
        self.content: dict[str, list[PfbInfo]] = {}

    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        return len(pfb_info.basename_split) == 5 and pfb_info.basename_split[1] == "out"

    def _add_to_files(self, pfb_info: PfbInfo):
        pfb_info.name = pfb_info.basename_split[2]
        pfb_info.sim_name = pfb_info.basename_split[0]
        pfb_info.timestep = int(pfb_info.basename_split[3])

        file_list = self.content.get(pfb_info.name, [])
        file_list.append(pfb_info)
        self.content[pfb_info.name] = file_list

    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        new_content = {}
        for var_name, pfbs in self.content.items():
            pfbs = [pfb for pfb in pfbs if variable_selector.is_accepted(pfb)]
            if pfbs:
                new_content[var_name] = pfbs

        self.content = new_content

        da_dict = {
            var_name: load_sequence_of_pfb(pfbs, loading_option)
            for var_name, pfbs in self.content.items()
        }
        return verbose_merge([da.rename(name) for name, da in da_dict.items()])

    def _update_time(self, pfb_info: PfbInfo, time_info: TimeInfo):
        timestep = pfb_info.timestep
        if pfb_info.run is None:
            s = int((timestep + time_info.time_shift) * time_info.default_output_ts)
        else:
            dump_interval = pfb_info.run.TimingInfo.DumpInterval
            if dump_interval < 0:
                dump_interval = abs(dump_interval * pfb_info.run.TimingInfo.BaseUnit)

            if (
                pfb_info.run.Solver.LSM == "CLM"
                and dump_interval > pfb_info.run.Solver.CLM.CLMDumpInterval
            ):
                s = int(timestep * 3600 * pfb_info.run.Solver.CLM.CLMDumpInterval)
            else:
                s = int(timestep * 3600 * dump_interval)
            s += time_info.time_shift * dump_interval * 3600
        pfb_info.time = time_info.start_date + np.timedelta64(s, "s")


class ClmOutputHandler(FileHandler):
    """CLM output: Temporal CLM output variables, exported by Parflow, such as clm and RST_clm"""

    def __init__(self):
        super().__init__()
        self.content: list[PfbInfo] = []

    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        return (
            len(pfb_info.basename_split) == 6
            and pfb_info.basename_split[1] == "out"
            and pfb_info.basename_split[2] == "clm_output"
            and pfb_info.basename_split[4] == "C"
        )

    def _add_to_files(self, pfb_info: PfbInfo):
        pfb_info.name = "clm_output"
        pfb_info.sim_name = pfb_info.basename_split[0]
        pfb_info.timestep = int(pfb_info.basename_split[3])

        self.content.append(pfb_info)

    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        self.content = [
            pfb for pfb in self.content if variable_selector.is_accepted(pfb)
        ]

        if not self.content:
            return xr.Dataset()

        da = load_sequence_of_pfb(self.content, loading_option).rename(
            {"z": "clm_z", "time": "clm_time"}
        )
        return da.rename("clm").to_dataset()

    def _update_time(self, pfb_info: PfbInfo, time_info: TimeInfo):
        timestep = pfb_info.timestep + time_info.time_shift
        if pfb_info.run is None:
            s = int(timestep * time_info.default_clm_ts)
        else:
            s = int(timestep * 3600 * pfb_info.run.Solver.CLM.CLMDumpInterval)
        pfb_info.time = time_info.start_date + np.timedelta64(s, "s")


class ForcingHandler(FileHandler):
    """Forcing variables: Temporal Forcing variables, such as APCP, Temp, LAI..."""

    def __init__(self):
        super().__init__()
        self.content: dict[str, list[PfbInfo]] = {}

    def _valid_file(self, pfb_info: PfbInfo) -> bool:
        return (
            len(pfb_info.basename_split) == 4 and "_to_" in pfb_info.basename_split[2]
        )

    def _add_to_files(self, pfb_info: PfbInfo):
        pfb_info.name = "forc_" + pfb_info.basename_split[1]
        time_end, _, time_start = pfb_info.basename_split[2].split("_")
        pfb_info.timestep = [int(time_end), int(time_start)]

        files = self.content.get(pfb_info.name, [])
        files.append(pfb_info)
        self.content[pfb_info.name] = files

    def load(
        self, variable_selector: VariableSelector, loading_option: LoadingOption
    ) -> xr.Dataset:
        da_dict = {
            var_name: load_sequence_of_forcing(pfbs, loading_option).rename(
                {"time": "forc_time"}
            )
            for var_name, pfbs in self.content.items()
            if variable_selector.is_accepted(pfbs[0])
        }
        return verbose_merge([da.rename(name) for name, da in da_dict.items()])

    def _update_time(self, pfb_info: PfbInfo, time_info: TimeInfo):
        timesteps = [
            np.timedelta64(
                int((i + time_info.time_shift) * time_info.default_forcing_ts), "s"
            )
            for i in range(pfb_info.timestep[0], pfb_info.timestep[1] + 1)
        ]
        pfb_info.time = time_info.start_date + np.array(timesteps)
