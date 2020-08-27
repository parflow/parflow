# -*- coding: utf-8 -*-
"""core module

This module provide the core objects for controlling ParFlow.

- A `Run()` object for configuring and running ParFlow simulations.
- A `check_parflow_execution(out_file)` function to parse output file

"""
import os
import sys
import argparse

from .database.generated import BaseRun, PFDBObj, PFDBObjListNumber
from .database.core import extract_keys_from_object, write_dict, resolve_path
from .terminal import Symbols as termSymbol

def check_parflow_execution(out_file):
    """Helper function that can be used to parse ParFlow output file

    Args:
        out_file (str): Path to the output run file.

    Returns:
        bool: True for success, False otherwise.

    """
    print(f'# {"="*78}')
    execute_success = False
    if os.path.exists(out_file):
        with open(out_file, "rt") as f:
            contents = f.read()
            if 'Problem solved' in contents:
                emoji = f'{termSymbol.splash} '
                print(
                    f'# ParFlow ran successfully {emoji*3}')
                execute_success = True
            else:
                emoji = f'{termSymbol.x} '
                print(
                    f'# ParFlow run failed. {emoji*3} Contents of error output file:')
                print("-"*80)
                print(contents)
                print("-"*80)
    else:
        print(f'# Cannot find {out_file} in {os.getcwd()}')
    print(f'# {"=" * 78}')
    return execute_success

# -----------------------------------------------------------------------------

def get_current_parflow_version():
    """Helper function to extract ParFlow version

    That method rely on PARFLOW_DIR environment variable to parse and
    extract the version of your installed version of ParFlow.

    Returns:
        str: Return ParFlow version like '3.6.0'

    """
    version = '3.6.0'
    version_file = f'{os.getenv("PARFLOW_DIR")}/config/pf-cmake-env.sh'
    if os.path.exists(os.path.abspath(version_file)):
        with open(version_file, "rt") as f:
            for line in f.readlines():
                if 'PARFLOW_VERSION=' in line:
                    version = line[17:22]
            if not version:
                print(f'Cannot find version in {version_file}')
    else:
        print(
            f'Cannot find environment file in {os.path.abspath(version_file)}.')
    return version

# -----------------------------------------------------------------------------

def get_process_args():
    '''
    General processing of script arguments
    '''
    parser = argparse.ArgumentParser(description="Parflow run arguments")

    # ++++++++++++++++
    group = parser.add_argument_group('Parflow settings')
    group.add_argument("--parflow-directory",
                       default=None,
                       dest="parflow_directory",
                       help="Path to use for PARFLOW_DIR")
    group.add_argument("--parflow-version",
                       default=None,
                       dest="parflow_version",
                       help="Override detected Parflow version")
    # ++++++++++++++++
    group = parser.add_argument_group('Execution settings')
    group.add_argument("--working-directory",
                        default=None,
                        dest="working_directory",
                        help="Path to execution working directory")

    group.add_argument("--skip-validation",
                        default=False,
                        dest="skipValidation",
                        action='store_true',
                        help="Disable validation pass")
    # ++++++++++++++++
    group = parser.add_argument_group('Error handling settings')
    group.add_argument("--show-line-error",
                        default=False,
                        dest="show_line_error",
                        action='store_true',
                        help="Show line error")

    group.add_argument("--exit-on-error",
                        default=False,
                        dest="exit_on_error",
                        action='store_true',
                        help="Exit at error")
    # ++++++++++++++++
    group = parser.add_argument_group('Additional output')
    group.add_argument("--write-yaml",
                        default=False,
                        dest="writeYAML",
                        action='store_true',
                        help="Enable config to be written as YAML file")
    # ++++++++++++++++
    group = parser.add_argument_group('Parallel execution')
    group.add_argument("-p", type=int, default=0,
                        dest="p",
                        help="P allocates the number of processes to the grid-cells in x")
    group.add_argument("-q", type=int, default=0,
                        dest="q",
                        help="Q allocates the number of processes to the grid-cells in y")
    group.add_argument("-r", type=int, default=0,
                        dest="r",
                        help="R allocates the number of processes to the grid-cells in z")

    return parser.parse_args()

# -----------------------------------------------------------------------------

def update_run_from_args(run, args):
    if args.parflow_directory:
        os.environ["PARFLOW_DIR"] = str(args.parflow_directory)
        PFDBObj.set_parflow_version(get_current_parflow_version())

    if args.working_directory:
        PFDBObj.set_working_directory(args.working_directory)

    if args.parflow_version:
        PFDBObj.set_parflow_version(args.parflow_version)

    if args.show_line_error:
        PFDBObj.enable_line_error()

    if args.exit_on_error:
        PFDBObj.enable_exit_error()

    if args.p:
        run.Process.Topology.P = args.p

    if args.q:
        run.Process.Topology.Q = args.q

    if args.r:
        run.Process.Topology.R = args.r

    if args.writeYAML:
        run.write(file_format='yaml')

# -----------------------------------------------------------------------------

class Run(BaseRun):
    """Main object that can be used to define a ParFlow simulation

    Args:
        name (str): Name for the given run.
        basescript (str): Path to current file so the simulation
            execution will put output files next to it.
            If not provided, the working directory will be used
            as base path.

    """
    def __init__(self, name, basescript=None):
        super().__init__(None)
        self._process_args_ = get_process_args()
        self._name_ = name
        if basescript:
            PFDBObj.set_working_directory(os.path.dirname(basescript))

    def get_key_dict(self):
        """Method that will return a flat map of all the ParFlow keys.

        Returns:
          dict: Return Python dict with all the key set listed without
              any hierarchy.
        """
        key_dict = {}
        extract_keys_from_object(key_dict, self)
        return key_dict

    def write(self, file_name=None, file_format='pfidb'):
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
        f_name = os.path.join(PFDBObj.working_directory,
                              f'{self._name_}.{file_format}')
        if file_name:
            f_name = os.path.join(PFDBObj.working_directory,
                                  f'{file_name}.{file_format}')
        full_file_path = os.path.abspath(f_name)
        write_dict(self.get_key_dict(), full_file_path)
        return full_file_path, full_file_path[:-(len(file_format)+1)]

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
        new_run.pfset(flat_map=self.get_key_dict())
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
              be used.
          skip_validation (bool): Allow user to skip validation before
              running the simulation.

        """
        if working_directory:
            PFDBObj.set_working_directory(working_directory)
        PFDBObj.set_parflow_version(get_current_parflow_version())

        # Any provided args should override the scripts ones
        update_run_from_args(self, self._process_args_)

        file_name, run_file = self.write()

        print()
        print(f'# {"="*78}')
        print(f'# ParFlow directory')
        print(f'#  - {os.getenv("PARFLOW_DIR")}')
        print(f'# ParFlow version')
        print(f'#  - {PFDBObj.pf_version}')
        print(f'# Working directory')
        print(f'#  - {os.path.dirname(file_name)}')
        print(f'# ParFlow database')
        print(f'#  - {os.path.basename(file_name)}')
        print(f'# {"="*78}')
        print()

        error_count = 0
        if not (skip_validation or self._process_args_.skipValidation):
            error_count += self.validate()
            print()

        p = self.Process.Topology.P
        q = self.Process.Topology.Q
        r = self.Process.Topology.R
        num_procs = p * q * r

        os.chdir(PFDBObj.working_directory)
        os.system('sh $PARFLOW_DIR/bin/run ' + run_file + ' ' + str(num_procs))

        success = check_parflow_execution(f'{run_file}.out.txt')
        print()
        if not success or error_count > 0:
            sys.exit(1)

    def dist(pfbFile, p=-1, q=-1, r=-1):
        inputFile = resolve_path(pfbFile)
        outputFile = f'{inputFile}.dist'
        if p < 1:
            p = self.Process.Topology.P
        if q < 1:
            q = self.Process.Topology.Q
        if r < 1:
            r = self.Process.Topology.R

        from parflowio.pyParflowio import PFData
        pfb_data = PFData(inputFile)
        pfb_data.distFile(p, q, r, outputFile)
