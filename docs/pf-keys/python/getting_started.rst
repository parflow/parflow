********************************************************************************
PFTools
********************************************************************************

Welcome to Python PFTools. This is a Python package that creates a user-friendly
Python interface for ParFlow. This package allows users to run ParFlow directly
from a Python script, leveraging the power and accessibility of Python. More
documentation for this package can be found here: https://pypi.org/project/pftools/

================================================================================
Installation
================================================================================

``pftools`` can be installed with the following command:

.. code-block:: language

    pip install pftools[all]

The ``[all]`` argument will download the dependencies necessary for running ParFlow
and fully employing the other tools within this package. ``[all]`` encompasses the
subsets of dependencies, including:

- ``[pfb]``: installs the ``parflowio`` package for handling ParFlow binary (.pfb) files.
- ``[pfsol]``: installs the ``imageio`` package for handling image processing to assist some workflows to build ParFlow solid (.pfsol) files.

If you would like to set up a virtual environment to install ``pftools``, execute the following commands:

.. code-block:: language

    python3 -m venv py-env
    source py-env/bin/activate
    pip install pftools[all]


================================================================================
Execution
================================================================================

Command: ``python3 /path/to/script/run_script.py [args]``

----

Usage:

.. code-block:: language

    run_script.py [-h] [--parflow-directory PARFLOW_DIRECTORY] [--parflow-version PARFLOW_VERSION]
    [--working-directory WORKING_DIRECTORY] [--skip-validation] [--dry-run] [--show-line-error]
    [--exit-on-error] [--write-yaml] [--validation-verbose] [-p P] [-q Q] [-r R]

Parflow run arguments:

- Optional arguments:

  .. code-block:: language

        -h, --help            show help message and exit


- Parflow settings:

  .. code-block:: language

        --parflow-directory PARFLOW_DIRECTORY
                        Path to use for PARFLOW_DIR

        --parflow-version PARFLOW_VERSION
                        Override detected Parflow version

- Execution settings:

  .. code-block:: language

        --working-directory WORKING_DIRECTORY
                        Path to execution working directory

        --skip-validation     Disable validation pass

        --dry-run             Prevent execution

- Error handling settings:

  .. code-block:: language

      --show-line-error     Show line error

      --exit-on-error       Exit at error

- Additional output:

  .. code-block:: language

      --write-yaml          Enable config to be written as YAML file

      --validation-verbose    Prints validation results for all key/value pairs

- Parallel execution:

  .. code-block:: language

      -p P
           P allocates the number of processes to the grid-cells in x (overrides Process.Topology.P)
      -q Q
           Q allocates the number of processes to the grid-cells in y (overrides Process.Topology.Q)
      -r R
           R allocates the number of processes to the grid-cells in z (overrides Process.Topology.R)


----

Output:

When executing ParFlow via the Python script using ``run()``, you will get the following message if the ParFlow run succeeds:

.. image:: PF_success.png
   :width: 696

Or if it fails:

.. image:: PF_fail.png
   :width: 809

This will be followed by the contents of the *runname.out.txt* file.
