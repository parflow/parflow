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


.. _fs_api:

Full API
---------

1. ``get_absolute_path(file_path)``
    Returns the absolute file path of the relative file location.

    :param ``file_path``: The file path of the relative file location.
    :return: The absolute file path.

2. ``exists(file_path)``
    Determines whether a file path exists.

    :param ``file_path``: The file path to check existence of.
    :return: ``True`` or ``False`` based on whether the file at ``file_path`` exists.

3. ``mkdir(dir_name)``
    Makes a new directory ``dir_name``. This works recursively, so it will also create intermediate directories 
    if they do not exist.

    :param ``dir_name``: The name of the directory to create.

4. ``chdir(directory_path)``
    Changes the working directory to ``directory_path``.

    :param ``directory_path``: The path of the directory to use as the current working directory.

5. ``cp(source, target_path='.')``
    Copies the file specified in the ``source`` argument to the location and/or file name specified in the
    ``target_path`` argument.

    :param ``source``: The path to the file to be copied.
    :param ``target_path``: The path to the directory/file name to copy ``source`` to.

6. ``rm(path)``
    Removes the file or directory located at ``path``.

    :param ``path``: The file or directory to remove.

7. ``get_text_file_content(file_path)``
    Reads a text file located at ``file_path`` and returns its content.

    :param ``file_path``: The path to the text file.
    :return: The content of the text file.


.. _fs_example:

Example
--------

If you want more examples on how to leverage those helper functions,
you can look at `$PARFLOW_SOURCE/test/python/clm/clm/clm.py <https://github.com/parflow/parflow/blob/master/test/python/clm/clm/clm.py#L32-L38>`_

The syntax and usage is more compact than the ``os`` and ``shutil`` methods commonly used in Python.
If you don't provide an absolute path to the file name, these functions will use ``get_absolute_path``
to find the absolute path based on your working directory, which defaults to the directory where your
Python script lives.
