# -*- coding: utf-8 -*-
"""core module

This module provide the core objects for controlling ParFlow.

- A `Run()` object for configuring and running ParFlow simulations.
- A `check_parflow_execution(out_file)` function to parse output file

"""
import os
import sys

from .database.generated import BaseRun, PFDBObj, PFDBObjListNumber
from .utils import extract_keys_from_object, write_dict
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
                print(f'# ParFlow ran successfully {termSymbol.splash*3}')
                execute_success = True
            else:
                print(
                    f'# ParFlow run failed. {termSymbol.x} {termSymbol.x} {termSymbol.x} Contents of error output file:')
                print("-"*80)
                print(contents)
                print("-"*80)
    else:
        print(f'# Cannot find {out_file} in {os.getcwd()}')
    print(f'# {"=" * 78}')
    return execute_success


def get_current_parflow_version():
    """Helper function to extract ParFlow version

    That method rely on PARFLOW_DIR environment variable to parse and
    extract the version of your installed version of ParFlow.

    Returns:
        str: Return ParFlow version like '3.6.0'

    """
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
        self._name = name
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

    # TODO: add feature to read external files
    # def readExternalFile(self, file_name=None, fileFormat='yaml'):
    #     inputDict = externalFileToDict(file_name, fileFormat)
    #     for key, value in inputDict.items():
    #         if not value == 'NA':
    #             # extKeyList =
    #             extKey = '.'.join(self, key)
    #             # print(extKey)

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
                              f'{self._name}.{file_format}')
        if file_name:
            f_name = os.path.join(PFDBObj.working_directory,
                                  f'{file_name}.{file_format}')
        full_file_path = os.path.abspath(f_name)
        write_dict(self.get_key_dict(), full_file_path)
        return full_file_path, full_file_path[:-(len(file_format)+1)]

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

        file_name, run_file = self.write()

        PFDBObj.set_parflow_version(get_current_parflow_version())

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
        if not skip_validation:
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
