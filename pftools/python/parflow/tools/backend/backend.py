import os
import glob
from typing import Optional, Iterable, Callable, Literal

import xarray as xr
import numpy as np
from xarray.backends import BackendEntrypoint

from parflow.tools.backend.generic import (
    PfbInfo,
    TimeInfo,
    Run,
    VariableSelector,
    LoadingOption,
)
from parflow.tools.backend.handler import (
    FileHandler,
    InputHandler,
    OutputHandler,
    ClmOutputHandler,
    TemporalOutputHandler,
    ForcingHandler,
    verbose_merge,
)


def is_pfb(s: str) -> bool:
    return os.path.isfile(s) and s.endswith(".pfb")


def is_pfidb(s: str) -> bool:
    return os.path.isfile(s) and s.endswith(".pfidb")


def is_seq(seq: list[str], func: Callable[[str], bool]) -> bool:
    return all(func(element) for element in seq)


class ParflowBackendEntrypoint(BackendEntrypoint):
    # Main class of the project: BackendEntryPoint for Xarray open_dataset
    # See https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html for more information
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "start_date",
        "default_forcing_ts",
        "default_clm_ts",
        "default_output_ts",
        "time_label",
        "select_types",
        "select_variables",
        "select_sim",
        "z_first",
        "replace_fill_value",
        "compute_date",
    ]

    handler_classes: dict[str, type[FileHandler]] = {
        "input": InputHandler,
        "output": OutputHandler,
        "clm": ClmOutputHandler,
        "temporal_output": TemporalOutputHandler,
        "forcing": ForcingHandler,
    }

    def open_dataset(
        self,
        filename_or_obj: str | list[str],
        *,
        drop_variables=None,
        start_date: str | np.datetime64 = "2000-01-01",
        default_forcing_ts: int = 1800,
        default_clm_ts: int = 10800,
        default_output_ts: int = 10800,
        time_label: Literal["right", "left", "center"] = "right",
        select_types: Optional[str | list[str]] = None,
        select_variables: Optional[list[str]] = None,
        select_sim: Optional[str] = None,
        z_first: bool = True,
        replace_fill_value: bool = True,
        compute_time: bool = True,
    ) -> xr.Dataset:
        """
        :param filename_or_obj: PFB file, PFIDB file, or directory containing PFB (and possibly PFIDB) files or a list of one of these items.
        :param drop_variables: A variable or list of variables to exclude from being parsed from the dataset. This may be useful to drop variables with problems or inconsistent values.
        :param start_date: Start date of data (timestep=0).
        :param default_forcing_ts: Default value for forcing time step in s.
        :param default_clm_ts: Default value for clm timestep in s. The value used will be the one found in the pfidb file, if available.
        :param default_output_ts: Default value for output timestep in s. The value used will be the one found in the pfidb file, if available.
        :param time_label: Side of each interval to use for labeling for time axes.
        :param select_types: Allows you to choose the type of variables to be selected (see ParflowBackendEntrypoint.handler_classes.keys()).
        :param select_variables: Allows you to choose the variables to be selected. The engine will check if the given str is a substring of a given variable.
        :param select_sim: Allows you to choose the simulation to be selected.
        :param z_first: Whether the z axis is first. If not, it will be last.
        :param replace_fill_value: Automatically detects the fill value associated with each variable and replaces it with np.nan.
        :param compute_time: Use the datetime computed from the time step as the label for the time axis. If not, use the time step.
        :return: xr.Dataset containing the different variables found.
        """

        if isinstance(start_date, str):
            start_date = np.datetime64(start_date).astype("datetime64[s]")

        time_shift = dict(right=0, center=-0.5, left=-1)[time_label]

        time_info = TimeInfo(
            start_date=start_date,
            time_shift=time_shift,
            default_forcing_ts=default_forcing_ts,
            default_clm_ts=default_clm_ts,
            default_output_ts=default_output_ts,
        )

        loading_option = LoadingOption(
            z_first=z_first,
            replace_fill_value=replace_fill_value,
            compute_time=compute_time,
        )

        if isinstance(filename_or_obj, str):
            filename_or_obj = [filename_or_obj]

        assert isinstance(filename_or_obj, Iterable)

        if select_sim is None:
            print("No simulation specified (select_sim=None)")
        # Search for pfb files
        pfbs = []
        if is_seq(filename_or_obj, is_pfb):
            pfbs = [PfbInfo(pfb) for pfb in filename_or_obj]
        elif is_seq(filename_or_obj, is_pfidb):
            for pfidb in filename_or_obj:
                glob_pfbs = glob.glob(
                    os.path.join(os.path.dirname(pfidb), "**/*.pfb"), recursive=True
                )
                pfbs += [PfbInfo(pfb, pfidb) for pfb in glob_pfbs]
        elif is_seq(filename_or_obj, os.path.isdir):
            for path in filename_or_obj:
                glob_pfbs = glob.glob(os.path.join(path, "**/*.pfb"), recursive=True)
                pfbs += [PfbInfo(pfb) for pfb in glob_pfbs]
        else:
            raise ValueError(f"Unknown type of input: {filename_or_obj}")

        print(f"{len(pfbs)} file found!")

        # Register pfb files in corresponding handlers
        handlers: dict[str, FileHandler] = {
            name: handler() for name, handler in self.handler_classes.items()
        }
        runs: dict[str, Run] = {}

        pfbs = sorted(pfbs, key=lambda x: x.filename)

        for pfb_info in pfbs:
            for handler in handlers.values():
                if handler.try_to_add(pfb_info, runs, time_info):
                    break
            else:
                print(f"No handler found for: {pfb_info.filename}")

        print(
            "Files per handler:",
            {name: handler.file_count for name, handler in handlers.items()},
        )

        if select_types is not None:
            if isinstance(select_types, str):
                select_types = [select_types]
            handlers = {
                name: handler
                for name, handler in handlers.items()
                if name in select_types
            }
            print(f"Selected handlers (ie select_types): {select_types}")

        variable_selector = VariableSelector(
            select_variables, select_sim, drop_variables
        )

        handlers_ds = [
            handler.load(
                variable_selector=variable_selector, loading_option=loading_option
            )
            for handler in handlers.values()
        ]

        variable_selector.print_stats()

        # for key, value in da_dict.items():
        #     print(key, value)

        return verbose_merge(handlers_ds)

    def guess_can_open(self, filename_or_obj: str | list[str]) -> bool:
        """Registers the backend to recognize *.pfb and *.pfidb files"""
        if isinstance(filename_or_obj, str):
            filename_or_obj = [filename_or_obj]
        if is_seq(filename_or_obj, is_pfb):
            return True
        if is_seq(filename_or_obj, is_pfidb):
            return True
        if is_seq(filename_or_obj, os.path.isdir):
            for path in filename_or_obj:
                glob_pfbs = glob.glob(os.path.join(path, "**/*.pfb"), recursive=True)
                if len(glob_pfbs) > 0:
                    return True
        return False
