********************************************************************************
From TCL to Python
********************************************************************************

Welcome to the tutorial for the Python pftools. You will need the following to
fully follow this tutorial:

- Python >= 3.6
- ParFlow installed and running, with the correct ``$PARFLOW_DIR`` environment variable established
  (You can check this by running ``echo $PARFLOW_DIR`` in your terminal)

The commands in the tutorial assume that you are running a bash shell in Linux or MacOS.

================================================================================
Virtual environment setup
================================================================================
In this first tutorial, we will set up a virtual environment with pftools and its dependencies before importing a TCL file, converting it to Python, and running ParFlow.

----

First, let's set an environment variable for the newly cloned repo:

.. code-block:: language

    export PARFLOW_SOURCE=/path/to/new/parflow/

Now, set up a virtual environment and install pftools:

.. code-block:: language

    python3 -m venv tutorial-env
    source tutorial-env/bin/activate
    pip install pftools[all]

Test your pftools installation:

.. code-block:: language

    python3 $PARFLOW_SOURCE/test/python/base_3d/default_richards/default_richards.py

The run should execute successfully, printing the message ``ParFlow ran successfully``.

================================================================================
From TCL to Python file
================================================================================

Great, now you have a working ParFlow interface! Next, create a new directory and import a TCL file (example here drawn from the ParFlow TCL tests):

.. code-block:: language

    mkdir -p pftools_tutorial/tcl_to_py
    cd pftools_tutorial/tcl_to_py
    cp $PARFLOW_SOURCE/test/default_richards.tcl .

You can use our ``tcl2py`` tool to convert the TCL script to a Python script using the following command:

.. code-block:: language

   python3 -m parflow.cli.tcl2py -i default_richards.tcl

The converter gets you most of the way there, but there are a few things you'll have to change by hand. Open and edit the new ``.py`` file that you have generated and change the lines that need to be changed. If you are following this example, you will need to edit the ``Process.Topology`` values, the ``GeomInput.Names`` values, and comment out the two ``Solver.Linear.Preconditioner.MGSemi`` keys, as shown here:

.. code-block:: python3

   default_richards.Process.Topology.P = 1
   default_richards.Process.Topology.Q = 1
   default_richards.Process.Topology.R = 1
   ...

   default_richards.GeomInput.Names = 'domain_input background_input source_region_input \
            concen_region_input'
   ...

   # default_richards.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
   # default_richards.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100


Once you have edited your Python script, you can run it like you would any other Python script:

.. code-block:: language

   python3 default_richards.py

Voilà! You have now successfully converted your first ParFlow TCL script to Python. In the next tutorial, we'll get more advanced to leverage the many other features in the Python PFTools. Onward!

=====================================================
Troubleshooting when converting TCL script to Python
=====================================================

Although the tutorial above (hopefully) went without a hitch, you may not always be so lucky. For those instances, Python PFTools has a tool that allows you to sort two *.pfidb* files to determine any discrepancies between two files.
This is especially useful when comparing an existing TCL script's generated file to its Python-generated equivalent. First, you must sort each of the
*.pfidb* files, using the following command:

.. code-block:: bash

    python3 -m parflow.cli.pfdist_sort -i /path/to/file.pfidb -o /tmp/sorted.pfidb

``/path/to/file.pfidb`` is the path to the existing (input, denoted by the ``-i``) *.pfidb* file, and ``/tmp/sorted.pfidb`` is the file path where you want the sorted output (denoted by the ``-o``) *.pfidb* file to be written.

Once you have the newly sorted files, you can compare them using one of many methods of file comparison, such as ``diff``:

.. code-block:: bash

    diff /path/to/from_tcl_sorted.pfidb /path/to/from_py_sorted.pfidb

You'll likely see some subtle format differences between the TCL- and Python-generated files (decimal printing, etc.). Most of these do not affect the execution of ParFlow.
