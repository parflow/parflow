.. _fs:

Filesystem
===========

.. _fs_intro:

Introduction
-------------

Files in ParFlow are like sand at the beach: everywhere.
Python's native modules offer plenty of methods to handle files and directories,
but they can be inconvenient when dealing with a ParFlow runs.
Fortunately, Python PFTools has some helpful functions to deal with the ParFlow
run working directory.

For example, let's pretend you want to automatically create a sub-directory for
your run while copying some data files into it at run time based on where your
run script lives.
You can simply do the following to achieve that while using environment
variable interpolation to dynamically adapt your run script at runtime without
having to continuously edit your script:

.. code-block:: python3

   from parflow.tools.fs import mkdir, cp

   mkdir('input-data')
   cp('$PF_SRC/test/input/*.pfb', './input-data/')


The working directory used to resolve your relative path gets automatically set
when you initialize your run instance by doing ``test_run = Run("demo", __file__)``.
This means that you should only use the ``fs`` methods after that initialization line.

The ``parflow.tools.fs`` module offers the following set of methods which all allow usage
of environment variables and relative paths within your run script:

.. code-block:: python3

   from parflow import Run
   from parflow.tools.fs import get_absolute_path, exists, chdir
   from parflow.tools.fs import mkdir, cp, rm
   from parflow.tools.fs import get_text_file_content
   # Initialize Run object and set working directory
   test_run = Run("demo", __file__)

   # Initialize Run object
   test_run = Run("demo", __file__)

   # Create directory in your current run script directory
   mkdir('input')
   mkdir('tmp')

   # Copy if file missing
   if not exists('data.pfb'):
       # Use environment variable to resolve location of PF_DATA
       cp('$PF_DATA/data.pfb')

   # Read data using Python tools
   full_path = get_absolute_path('data.csv')
   with open(full_path) as file:
       pass

   # Or use python working directory
   chdir('.')
   with open('data.csv') as file:
       pass

   # Or use the text file content helper
   txt = get_text_file_content('data.csv')

   # Clean behind yourself
   rm('tmp')


.. _fs_example:

Example
--------

If you want more examples on how to leverage those helper functions,
you can look at `$PARFLOW_SOURCE/test/python/clm/clm.py <https://github.com/parflow/parflow/blob/master/test/python/clm/clm.py>`_

The syntax and usage is more compact than the ``os`` and ``shutil`` methods commonly used in Python.
If you don't provide an absolute path to the file name, these functions will use ``get_absolute_path``
to find the absolute path based on your working directory, which defaults to the directory where your
Python script lives.
