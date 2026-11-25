.. _solid_files:

Solid Files
============

Generating solid (.pfsol) files for a ParFlow run can be somewhat of a pain. PFTools has a few features that can help with this process.

.. _solid_files_examples:

Example
--------

To see the how Python can help generate solid files, navigate to `$PARFLOW_SOURCE/test/python/new_features/simple-mask.py <https://github.com/parflow/parflow/blob/master/test/python/new_features/simple-mask.py>`_. Here, you'll see the following lines at the top of the script:

.. code-block:: python3

    from parflow import Run
    from parflow.tools.fs import get_absolute_path
    from parflow.tools.io import load_patch_matrix_from_sa_file, load_patch_matrix_from_asc_file, load_patch_matrix_from_image_file
    from parflow.tools.builders import SolidFileBuilder

By now, you should be familiar with the first two modules and functions. The ``load_patch_matrix...`` functions handle different file
types to generate solid files. The ``SolidFileBuilder`` class imported from the ``parflow.tools.builders`` module handles the matrices of
patches, converting them to ASCII files and passing those to the ``pfmask-to-pfsol`` converter in ParFlow. This way, the user doesn't
have to deal with the more complicated steps.

----

Lines 56 through 61 show examples of how the ``patch_matrix`` functions are used for different types of files:

.. code-block:: python3

    sabino_mask = load_patch_matrix_from_sa_file(get_absolute_path('Sabino_Mask.sa'))
    # sabino_mask = load_patch_matrix_from_asc_file(get_absolute_path('Sabino_Mask.asc'))
    # sabino_mask = load_patch_matrix_from_image_file(get_absolute_path('Sabino_Mask.png'))
    # sabino_mask = load_patch_matrix_from_image_file(get_absolute_path('Sabino_Mask.tiff'))

Note that only one is used at a time, but all four will work. These functions return a matrix, which is assigned to ``sabino_mask``.
The input files are located in the same directory as the example, so feel free to reference them.

----

Next, we'll show some examples of the ``SolidFileBuilder`` class to demonstrate the arguments and methods that can be called on the object:

.. code-block:: python3

    # Example of using unique ids for each surface [top/bottom/side]
    SolidFileBuilder(top=1, bottom=2, side=3) \ # Initializing the SolidFileBuilder
        .mask(sabino_mask) \                      # Setting the 2D mask
        .write('sabino_domain.pfsol', cellsize=90) \  # Write pfsol file
        .for_key(sabino.GeomInput.domaininput)  # Setting keys to "sabino" Run object that relate to the solid file

    # Example using an id mask for the top patches
    SolidFileBuilder(bottom=2, side=3) \ # Initializing the SolidFileBuilder
        .mask(sabino_mask) \               # Setting the 2D mask
        .top_ids(id_array) \                  # Using a 2D numpy array to provide patch ids
        .write('sabino_domain.pfsol', cellsize=90) # Write pfsol file

    # Example using the same matrix to write multiple solid files
    SolidFileBuilder(top=1, bottom=2, side=3) \
        .mask(sabino_mask) \                      # Setting the 2D mask
        .write('sabino_domain.pfsol', cellsize=90) \  # Write first pfsol file
        .mask(sabino_mask_2) \                      # Setting another 2D mask
        .side_ids(id_array) \              # Using a 2D numpy array to provide new patch ids (possibly to change boundary conditions)
        .write('sabino_domain_2.pfsol', cellsize=90)   # Write second pfsol file

.. _solid_files_more_examples:

More examples
--------------

Other example scripts showing how to use the ``SolidFileBuilder`` can be found in `$PARFLOW_SOURCE/test/python/new_features/ <https://github.com/parflow/parflow/blob/master/test/python/new_features>`_. If you have an idea for a new feature or
improvement to the functionality, please let us know, or better yet, become a contributor!
