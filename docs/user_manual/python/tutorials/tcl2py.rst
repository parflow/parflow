.. _tcl2py:
    
From TCL to Python
===================

Welcome to the tutorial for the Python pftools. You will need the following to
fully follow this tutorial:

- Python >= 3.7
- ParFlow installed and running, with the correct ``$PARFLOW_DIR`` environment variable established
  (You can check this by running ``echo $PARFLOW_DIR`` in your terminal)

The commands in the tutorial assume that you are running a bash shell in Linux or MacOS.

.. _tcl2py_virtual_env:

Virtual environment setup
--------------------------

In this first tutorial, we will set up a virtual environment with pftools and its dependencies before converting a TCL runscript to Python and running ParFlow.

----

Set up a virtual environment and install pftools:

.. code-block::

    python3 -m venv tutorial-env
    source tutorial-env/bin/activate
    pip install pftools

Test your pftools installation:

.. code-block::

    python3 $PARFLOW_DIR/test/python/default_richards.py

The run should execute successfully, printing the message ``ParFlow ran successfully``.

.. _tcl2py_example:

From TCL to Python file
------------------------

Great, now you have a working ParFlow interface! Next, create a new directory and import a TCL file (example here drawn from the ParFlow TCL tests):

.. code-block::

    mkdir -p pftools_tutorial/tcl_to_py
    cd pftools_tutorial/tcl_to_py
    cp $PARFLOW_DIR/test/tcl/default_richards.tcl .

TCL ``pfset`` keys and Python ``run.Key =`` keys are the same ParFlow database entries written in two syntaxes. While you are converting a script — and afterward, when you need to look up a key — both forms are shown side by side in the :ref:`ParFlow Input Keys` chapter. For example:

::

   run.Process.Topology.P = 2  # Python syntax
   pfset Process.Topology.P 2  # TCL syntax

To convert the TCL runscript, use an AI coding assistant (for example Cursor, Claude, Copilot, or a similar tool). Give the assistant the TCL file and the ParFlow tcl-to-python skill, then ask it to convert the runscript to Python PFTools.

Download the skill here: :download:`tcl-to-python SKILL.md <tcl-to-python/SKILL.md>`.
In the ParFlow source tree it lives at ``docs/user_manual/python/tutorials/tcl-to-python/SKILL.md``.
Attach or add that file as a skill in your AI tool so the conversion follows ParFlow Python conventions (``Run`` object, hyphenated patch names, ``dist()`` / ``run()``, and so on).

A converted key assignment looks like this:

.. code-block::

   # TCL
   pfset ComputationalGrid.NX                      18

   # Python
   default_richards.ComputationalGrid.NX = 18

Review the generated ``.py`` file, then run it like any other Python script:

.. code-block::

   python3 default_richards.py

Voilà! You have now successfully converted your first ParFlow TCL script to Python. In the next tutorial, we'll get more advanced to leverage the many other features in the Python PFTools. Onward!