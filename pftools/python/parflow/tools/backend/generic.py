import os
from dataclasses import dataclass
from typing import Optional
import glob
from pathlib import Path

import numpy as np
from parflow.tools import settings
from parflow.tools.fs import get_absolute_path
from parflow import Run as BaseRun

from parflow.tools.backend.header import Header


class Run(BaseRun):
    """Run from parflow but without loading CLM files"""

    @classmethod
    def from_definition(cls, file_path):
        """Method to generate a Run object from a file.

        Currently accepts the following input file types:
            yaml
            pfidb

        Args:
            file_path (str): path to the file to read in

        Returns:
            A new Run object
        """
        # Reset working directory to cwd for the file path
        settings.set_working_directory()
        file_path = Path(get_absolute_path(file_path))
        name, ext = file_path.stem, file_path.suffix[1:]

        ext_map = {"yaml": "yaml_file", "yml": "yaml_file", "pfidb": "pfidb_file"}

        if ext not in ext_map:
            raise Exception(f"Unknown extension: {ext}")

        new_run = cls(name, file_path)
        kwargs = {ext_map[ext]: file_path}
        new_run.pfset(silence_if_undefined=True, **kwargs)

        # Try to solve order sensitive property settings
        while "_pfstore_" in new_run.__dict__:
            invalid_props = new_run.__dict__.pop("_pfstore_")
            previous_size = len(invalid_props)
            for key, value in invalid_props.items():
                new_run.pfset(key, value, silence_if_undefined=True)

            # Break if no key was able to be mapped outside pfstore
            if "_pfstore_" in new_run.__dict__ and previous_size == len(
                new_run.__dict__["_pfstore_"]
            ):
                break

        # Print any remaining key with no mapping
        if "_pfstore_" in new_run.__dict__:
            invalid_props = new_run.__dict__.pop("_pfstore_")
            for key, value in invalid_props.items():
                new_run.pfset(key, value)

        # if ext == 'pfidb':
        #     # Import CLM files if we need to
        #     try:
        #         CLMImporter(new_run).import_if_needed()
        #     except Exception:
        #         print(' => Error during CLM import - '
        #               'CLM specific key have been skipped')

        return new_run


@dataclass
class TimeInfo:
    """Used by the backend to transmit temporal information to handlers."""

    start_date: np.datetime64
    time_shift: float
    default_forcing_ts: int
    default_clm_ts: int
    default_output_ts: int


class PfbInfo:
    """
    Class associated with a pfb file that allows various information (name, simulation, etc.) to be associated with it.
    """

    def __init__(self, filename: str, pfidb: Optional[str] = None):
        self.filename = filename
        self.name: Optional[str] = None
        self.sim_name: Optional[str] = None
        self.timestep: Optional[int | list[int]] = None
        self.time: Optional[np.datetime64 | list[np.datetime64]] = None
        self.basename_split = os.path.basename(filename).split(".")

        self.pfidb = pfidb
        self.run: Optional[Run] = None

    def set_run(self, runs):
        if self.pfidb is None:
            self.pfidb = self._get_pfidb()

        if self.pfidb == "":
            return

        if self.pfidb not in runs:
            runs[self.pfidb] = Run.from_definition(self.pfidb)
        self.run = runs[self.pfidb]

    def _get_pfidb(self) -> str:
        path = os.path.dirname(self.filename)
        if self.sim_name is not None:
            pfidb_name = os.path.join(path, self.sim_name + ".pfidb")
            if os.path.isfile(pfidb_name):
                return pfidb_name
        pfidbs = glob.glob(os.path.join(path, "*.pfidb"))
        if len(pfidbs) == 0:
            return ""
        # warning if len > 1 ?
        return pfidbs[0]

    def get_header(self) -> Header:
        return Header(self.filename)

    def __str__(self):
        dic = vars(self)
        real_dic = {k: v for k, v in dic.items() if v is not None}
        return str(real_dic)


@dataclass
class LoadingOption:
    z_first: bool = True
    replace_fill_value: bool = True
    compute_time: bool = True


def _prepare_argument(arg: Optional[str | list[str]]):
    if arg is None:
        return None
    if isinstance(arg, str):
        arg = [arg]
    return [s.lower() for s in arg]


class AbstractSelector:
    def __init__(self, selection: Optional[str | list[str]]):
        self.selection = _prepare_argument(selection)
        self.count_tested = 0
        self.count_accepted = 0

    def is_accepted(self, pfb_info: PfbInfo) -> bool:
        res = self._is_accepted(pfb_info)
        self.count_tested += 1
        if res:
            self.count_accepted += 1
        return res

    def _is_accepted(self, pfb_info: PfbInfo) -> bool:
        pass

    def print_stats(self):
        if self.selection is None:
            return

        print(
            f'{self.__class__.__name__}: {self.count_accepted}/{self.count_tested} files accepted (arg="{self.selection})"'
        )


class DropVariables(AbstractSelector):
    def _is_accepted(self, pfb_info: PfbInfo) -> bool:
        if self.selection is None:
            return True

        return not any(name.lower() in pfb_info.name for name in self.selection)


class SelectVariables(AbstractSelector):
    def _is_accepted(self, pfb_info: PfbInfo) -> bool:
        if self.selection is None:
            return True

        return any(name.lower() in pfb_info.name for name in self.selection)


class SelectSimulation(AbstractSelector):
    def __init__(self, selection: Optional[str | list[str]]):
        super().__init__(selection)
        self.simulations = set()

    def _is_accepted(self, pfb_info: PfbInfo) -> bool:
        if pfb_info.sim_name is None:
            return True

        self.simulations.add(pfb_info.sim_name.lower())

        if self.selection is None:
            return True

        return any(sim_name == pfb_info.sim_name.lower() for sim_name in self.selection)

    def print_stats(self):
        print(f"Found files for simulations: {self.simulations}")
        super().print_stats()


class VariableSelector:
    """Filter for selecting variables compatible with the dataset parameters."""

    def __init__(
        self,
        select_variables: Optional[str | list[str]],
        select_sim: Optional[str],
        drop_variables: Optional[str | list[str]],
    ):
        self.selectors: list[AbstractSelector] = [
            SelectVariables(select_variables),
            SelectSimulation(select_sim),
            DropVariables(drop_variables),
        ]

    def is_accepted(self, pfb_info: PfbInfo) -> bool:
        return all(selector.is_accepted(pfb_info) for selector in self.selectors)

    def print_stats(self):
        for selector in self.selectors:
            selector.print_stats()
