Introduction
============

Python PFTools (``pftools``) is a Python package that provides a user-friendly
interface to ParFlow. Instead of writing a TCL input script, you define a
ParFlow problem in Python: keys are set on a ``Run`` object, which builds the
ParFlow database (``.pfidb``) that the simulator reads as input. You can then
validate that database, write it to disk, and execute ParFlow from the same
script.

The package is published on PyPI: https://pypi.org/project/pftools/

Earlier versions of ParFlow also exposed data-manipulation tools through TCL
(``PFTCL``). This chapter documents the **Python** PFTools package.

What you can do with PFTools
----------------------------

- Define and run simulations using the ``Run`` object and ParFlow keys in
  ordinary Python syntax, including validation and writing of input databases.
- Read and write ParFlow binary (.pfb) data, including workflows that use ``xarray``.
- Post-process results with hydrology helpers, data accessors, and related
  utilities.

A minimal example
-----------------

The usual pattern is to create a ``Run``, assign keys, then execute:

.. code-block:: python

    from parflow import Run

    run = Run("my_run", __file__)
    run.FileVersion = 4
    # ... set additional ParFlow keys ...
    run.run()

Prerequisites
-------------

- A working ParFlow installation, with the environment variable ``PARFLOW_DIR`` set so that PFTools can
  locate the ParFlow executable and related tools when you call ``run()``.
- Python 3 (see the package metadata for the supported version range).

Installing ``pftools`` from pip provides the Python interface; it does not
replace building and installing ParFlow itself. See :ref:`pftools_installation`
for install options, optional extras, and virtual environments.

In this chapter
---------------

The remaining sections of this chapter cover:

- **Installation** — how to install ``pftools`` and optional extras.
- **PFTools API Reference** — generated reference for the main modules and the
  ``Run`` class.
- **Contributing New Keys** — how ParFlow keys are defined in YAML and instructions for adding new keys.

For the full catalog of ParFlow keys and their meanings, see
:ref:`ParFlow Input Keys`. For how a run is structured and annotated example
problems, see :ref:`The ParFlow System`.
