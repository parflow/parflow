# -*- coding: utf-8 -*-
"""core module

This module provide the core objects for controlling ParFlow.

- A `Run()` object for configuring and running ParFlow simulations.
- A `check_parflow_execution(out_file)` function to parse output file
- A `get_current_parflow_version()` function to extract current ParFlow version

"""
import os
from pathlib import Path
import sys
import argparse

import numpy as np

from . import settings
from .fs import get_absolute_path
from .io import (
    DataAccessor,
    ParflowBinaryReader,
    write_dict,
    read_pfb,
    write_pfb,
    write_dist,
)

from .database.generated import BaseRun
from .export import CLMExporter, SubsurfacePropertiesExporter
from .builders import CLMImporter


def check_parflow_execution(out_file):
    """Helper function that can be used to parse ParFlow output file

    Args:
        out_file (str): Path to the output run file.

    Returns:
        bool: True for success, False otherwise.

    """
    print(f'# {"=" * 78}')
    execute_success = False
    if Path(out_file).exists():
        with open(out_file, "r") as f:
            contents = f.read()
            if "Problem solved" in contents:
                print(f"# ParFlow ran successfully")
                execute_success = True
            else:
                print(f"# ParFlow run failed. " f"Contents of error output file:")
                print("-" * 80)
                print(contents)
                print("-" * 80)
    else:
        print(f"# Cannot find {out_file} in {os.getcwd()}")
    print(f'# {"=" * 78}')
    return execute_success


# -----------------------------------------------------------------------------


def get_current_parflow_version():
    """Helper function to extract ParFlow version

    That method rely on PARFLOW_DIR environment variable to parse and
    extract the version of your installed version of ParFlow.

    TODO: This not a very good way to get the version.   Should get
    from cmake processed file.

    Returns:
        str: Return ParFlow version like '3.6.0'

    """
    version = "3.6.0"
    version_file = f'{os.getenv("PARFLOW_DIR")}/config/pf-cmake-env.sh'
    if Path(version_file).resolve().exists():
        with open(version_file, "r") as f:
            for line in f:
                if "PARFLOW_VERSION=" in line:
                    version = line[17:-2]
            if not version:
                print(f"Cannot find version in {version_file}")
    else:
        print(
            f"Cannot find environment file in " f"{str(Path(version_file).resolve())}."
        )
    return version


# -----------------------------------------------------------------------------


def get_process_args():
    """
    General processing of script arguments
    """
    parser = argparse.ArgumentParser(description="Parflow run arguments")

    # ++++++++++++++++
    group = parser.add_argument_group("Parflow settings")
    group.add_argument(
        "--parflow-directory",
        default=None,
        dest="parflow_directory",
        help="Path to use for PARFLOW_DIR",
    )
    group.add_argument(
        "--parflow-version",
        default=None,
        dest="parflow_version",
        help="Override detected Parflow version",
    )
    # ++++++++++++++++
    group = parser.add_argument_group("Execution settings")
    group.add_argument(
        "--working-directory",
        default=None,
        dest="working_directory",
        help="Path to execution working directory",
    )

    group.add_argument(
        "--skip-validation",
        default=False,
        dest="skip_validation",
        action="store_true",
        help="Disable validation pass",
    )

    group.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        dest="dry_run",
        help="Prevent execution",
    )
    # ++++++++++++++++
    group = parser.add_argument_group("Error handling settings")
    group.add_argument(
        "--show-line-error",
        default=False,
        dest="show_line_error",
        action="store_true",
        help="Show line error",
    )

    group.add_argument(
        "--exit-on-error",
        default=False,
        dest="exit_on_error",
        action="store_true",
        help="Exit at error",
    )
    # ++++++++++++++++
    group = parser.add_argument_group("Additional output")
    group.add_argument(
        "--write-yaml",
        default=False,
        dest="write_yaml",
        action="store_true",
        help="Enable config to be written as YAML file",
    )

    group.add_argument(
        "--validation-verbose",
        default=False,
        dest="validation_verbose",
        action="store_true",
        help="Only print validation results for " "key/value pairs with errors",
    )

    group.add_argument(
        "--overwrite-clm-driver-files",
        default=False,
        dest="overwrite_clm_driver_files",
        action="store_true",
        help="Allow old clm driver files to be overwritten",
    )

    # ++++++++++++++++
    group = parser.add_argument_group("Parallel execution")
    group.add_argument(
        "-p",
        type=int,
        default=0,
        dest="p",
        help="P allocates the number of processes " "to the grid-cells in x",
    )
    group.add_argument(
        "-q",
        type=int,
        default=0,
        dest="q",
        help="Q allocates the number of processes " "to the grid-cells in y",
    )
    group.add_argument(
        "-r",
        type=int,
        default=0,
        dest="r",
        help="R allocates the number of processes " "to the grid-cells in z",
    )

    args, unknown = parser.parse_known_args()
    return args


# -----------------------------------------------------------------------------


def update_run_from_args(run, args):
    """
    Push processed args onto run properties.
    """
    if args.parflow_directory:
        os.environ["PARFLOW_DIR"] = str(args.parflow_directory)
        settings.set_parflow_version(get_current_parflow_version())

    if args.working_directory:
        settings.set_working_directory(args.working_directory)

    if args.parflow_version:
        settings.set_parflow_version(args.parflow_version)

    if args.show_line_error:
        settings.enable_line_error()

    if args.exit_on_error:
        settings.enable_exit_error()

    if args.p and run.Process.Topology.P != args.p:
        run.Process.Topology.P = args.p

    if args.q and run.Process.Topology.Q != args.q:
        run.Process.Topology.Q = args.q

    if args.r and run.Process.Topology.R != args.r:
        run.Process.Topology.R = args.r

    if args.overwrite_clm_driver_files:
        clm_solver = run.Solver.CLM
        keys = [
            "OverwriteDrvClmin",
            "OverwriteDrvVegp",
            "OverwriteDrvVegm",
        ]
        for key in keys:
            if getattr(clm_solver, key) is False:
                setattr(clm_solver, key, True)


# -----------------------------------------------------------------------------


class Run(BaseRun):
    """Main object that can be used to define a ParFlow simulation

    Args:
        name (str): Name for the given run.
        basescript (str): Path to current file so the simulation
            execution and relative path will be assumed to have
            the script directory as their working directory.
            If not provided, the working directory will be the
            directory where the python executable was run from.

    """

    def __init__(self, name, basescript=None):
        super().__init__(None)
        self._accessor_ = None
        self._process_args_ = get_process_args()
        self.set_name(name)
        if basescript is not None:
            full_path = get_absolute_path(basescript)
            if Path(full_path).is_file():
                settings.set_working_directory(str(Path(full_path).parent))
            else:
                settings.set_working_directory(full_path)
        else:
            settings.set_working_directory()

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

        if ext == "pfidb":
            # Import CLM files if we need to
            try:
                CLMImporter(new_run).import_if_needed()
            except Exception:
                print(
                    " => Error during CLM import - "
                    "CLM specific key have been skipped"
                )

        return new_run

    def get_name(self):
        """Returns name of run"""
        return self._name_

    def set_name(self, new_name):
        """Setting new name for a run

        Args:
            new_name (str): New name for run
        """
        self._name_ = new_name

    @property
    def data_accessor(self):
        """Return a DataAccessor to streamline access to
        numpy array of the various field this run is linked to.
        """
        if self._accessor_ is None:
            self._accessor_ = DataAccessor(self)
        return self._accessor_

    def write(self, file_name=None, file_format="pfidb", working_directory=None):
        """Method to write database file to disk

        Args:
          file_name (str): Name of the file to write.
              If not provided, the name of the run will be used.
          file_format (str): File extension which also represent
              the output format to use. (pfidb, yaml, json)
              The default is pfidb.

        Returns:
          (str): The full path to the written file.
          (str): The full path of the run which is what should be
              given to ParFlow executable.

        """
        # overwrite current working directory
        prev_dir = settings.WORKING_DIRECTORY
        if working_directory:
            settings.set_working_directory(working_directory)

        f_name = os.path.join(
            settings.WORKING_DIRECTORY, f"{self._name_}.{file_format}"
        )
        if file_name:
            f_name = os.path.join(
                settings.WORKING_DIRECTORY, f"{file_name}.{file_format}"
            )
        full_file_path = os.path.abspath(f_name)
        write_dict(self.to_dict(), full_file_path)

        if CLMExporter(self).can_export:
            # If we are using CLM, write out any other files we need
            CLMExporter(self).write_allowed(settings.WORKING_DIRECTORY)

        # revert working directory to original directory
        settings.set_working_directory(prev_dir)

        return full_file_path, full_file_path[: -(len(file_format) + 1)]

    def write_subsurface_table(self, file_name=None, working_directory=None):
        # overwrite current working directory
        prev_dir = settings.WORKING_DIRECTORY
        if working_directory:
            settings.set_working_directory(working_directory)

        if file_name is None:
            file_name = f"{self._name_}_subsurface.csv"
        full_path = get_absolute_path(file_name)
        exporter = SubsurfacePropertiesExporter(self)
        if file_name.lower().endswith(".csv"):
            exporter.write_csv(full_path)
        else:
            exporter.write_txt(full_path)

        # revert working directory to original directory
        settings.set_working_directory(prev_dir)

    def clone(self, name):
        """Method to generate a clone of a run (for generating
        run ensembles, etc.)

        This will return an identical run with the given name.
        See parflow/test/python/new_features/serial_runs/serial_runs.py
        for an example.

        Args:
          name (str): Name of the new run.

        """
        new_run = Run(name)
        new_run.pfset(flat_map=self.to_dict())
        return new_run

    def run(self, working_directory=None, skip_validation=False):
        """Method to run simulation

        That method will automatically run the database write,
        validation and trigger the ParFlow execution while
        checking ParFlow output for any error.

        If an error occurred a sys.exit(1) will be issue.

        Args:
          working_directory (str): Path to write output files.
              If not provided, the default run working directory will
              be used. This also affect the relative FileNames.
          skip_validation (bool): Allow user to skip validation before
              running the simulation.

        """
        # overwrite current working directory
        prev_dir = settings.WORKING_DIRECTORY
        if working_directory:
            settings.set_working_directory(working_directory)

        settings.set_parflow_version(get_current_parflow_version())

        # Any provided args should override the scripts ones
        update_run_from_args(self, self._process_args_)

        file_name, run_file = self.write()

        print()
        print(f'# {"=" * 78}')
        print("# ParFlow directory")
        print(f'#  - {os.getenv("PARFLOW_DIR")}')
        print("# ParFlow version")
        print(f"#  - {settings.PARFLOW_VERSION}")
        print("# Working directory")
        print(f"#  - {os.path.dirname(file_name)}")
        print("# ParFlow database")
        print(f"#  - {os.path.basename(file_name)}")
        print(f'# {"=" * 78}')

        # Only write YAML in run()
        if self._process_args_.write_yaml:
            full_path, no_extension = self.write(file_format="yaml")
            print(f'YAML output: "{full_path}"')

        print()
        error_count = 0
        if not (skip_validation or self._process_args_.skip_validation):
            verbose = self._process_args_.validation_verbose
            error_count += self.validate(verbose=verbose)
            print()

        p = self.Process.Topology.P
        q = self.Process.Topology.Q
        r = self.Process.Topology.R
        num_procs = p * q * r

        success = True
        if not self._process_args_.dry_run:
            prev_dir = os.getcwd()
            try:
                os.chdir(settings.WORKING_DIRECTORY)
                os.system(f"sh $PARFLOW_DIR/bin/run {run_file} {num_procs}")
                success = check_parflow_execution(f"{run_file}.out.txt")
            finally:
                os.chdir(prev_dir)

        print()

        # revert working directory to original directory
        settings.set_working_directory(prev_dir)

        if not success or error_count > 0:
            sys.exit(1)

    def check_nans(self, working_directory, include_forcing=True):
        """Check the input files for NaNs.

        Args:
            working_directory: The working directory of the ParFlow run.
            include_forcing (bool): If set to True, forcing files are going
                to be checked.

        Raises:
            ValueError: If an input file contains a NaN.
        """

        runscript_path, _ = self.write(file_format="yaml")
        all_files = []
        with open(runscript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("FileName: ") and line.endswith(".pfb"):
                    path = os.path.join(working_directory, line.split()[1])
                    all_files.append(path)

        if include_forcing:
            run = self.__class__.from_definition(runscript_path)
            forcing_dir = run.Solver.CLM.MetFilePath
            for file in os.listdir(forcing_dir):
                if file.endswith(".pfb"):
                    all_files.append(os.path.join(forcing_dir, file))

        for file in all_files:
            data = read_pfb(file)
            if np.any(np.isnan(data)):
                raise ValueError(f"{file} contains NaN values.")

    def dist(self, pfb_file, **kwargs):
        """Distribute a PFB file using the P/Q/R settings from the run
        or override them with the provided arguments.

        We can also use the kwargs to set other properties such as:
          - NX, NY, NZ...
        """
        # Any provided args should override the scripts ones

        update_run_from_args(self, self._process_args_)

        pfb_file_full_path = get_absolute_path(pfb_file)
        p = kwargs.get("P", self.Process.Topology.P)
        q = kwargs.get("Q", self.Process.Topology.Q)
        r = kwargs.get("R", self.Process.Topology.R)

        with ParflowBinaryReader(pfb_file_full_path, read_sg_info=True) as pfb:
            array = pfb.read_all_subgrids()
            header = pfb.header

        dx, dy, dz = header["dx"], header["dy"], header["dz"]

        write_pfb(
            pfb_file_full_path, array, p=p, q=q, r=r, dx=dx, dy=dy, dz=dz, dist=True
        )
