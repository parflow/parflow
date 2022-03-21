********************************************************************************
Solid Files
********************************************************************************
Generating solid (.pfsol) files for a ParFlow run can be somewhat of a pain. PFTools has a few features that can help with this process.

================================================================================
Example
================================================================================

To see the how Python can help generate solid files, navigate to *$PARFLOW_SOURCE/test/python/pfsol/simple-mask/* and open the Python script
*simple-mask.py*. Here, you'll see the following lines at the top of the script:

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

Lines 52 through 55 show examples of how the ``patch_matrix`` functions are used for different types of files:

.. code-block:: python3

    sabino_mask = load_patch_matrix_from_sa_file(get_absolute_path('Sabino_Mask.sa'))
    # sabino_mask = load_patch_matrix_from_asc_file(get_absolute_path('Sabino_Mask.asc'))
    # sabino_mask = load_patch_matrix_from_image_file(get_absolute_path('Sabino_Mask.png'))
    # sabino_mask = load_patch_matrix_from_image_file(get_absolute_path('Sabino_Mask.tiff'))

Note that only one is used at a time, but all four will work. These functions return a matrix, which is assigned to ``sabino_mask``.
The input files are located in the same directory as the example, so feel free to reference them. Several of these functions have
extra optional arguments, which are described in the full API below.

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

================================================================================
Full API: IO tools (from ``parflow.tools.io``)
================================================================================

1. ``load_patch_matrix_from_pfb_file(file_name, layer=None)`` - reads in a 2D or 3D ParFlow binary (PFB) file ``file_name`` and converts it to a patch matrix. If it is a 3D PFB file, the user can specify a vertical layer of the file to convert to the matrix. If no layer is specified, the function will use the top layer of the file.

2. ``load_patch_matrix_from_image_file(file_name, color_to_patch=None, fall_back_id=0)`` - reads in an image file ``file_name`` and converts it to a patch matrix. The ``color_to_patch`` argument is a dictionary with hexidecimal colors as keys with their corresponding ID numbers as values. See *$PARFLOW_SOURCE/test/python/pfsol/image-as-mask/image-as-mask.py* for an example. If ``color_to_patch`` is not provided, it will default to assume that everything in white is not part of the mask and everything else is part of the mask. The ``fall_back_id`` is the ID number for colors that are found in the image but are not specified in the ``color_to_patch`` dictionary. Its default value is zero.

3. ``load_patch_matrix_from_asc_file(file_name)`` - reads in an ASCII file ``file_name`` and converts it to a patch matrix.

4. ``load_patch_matrix_from_sa_file(file_name)`` - reads in a simple ASCII file ``file_name`` and converts it to a patch matrix.

5. ``write_patch_matrix_as_asc(matrix, file_name, xllcorner=0.0, yllcorner=0.0, cellsize=1.0, NODATA_value=0)`` - writes an ASCII file to ``file_name`` from the patch matrix ``matrix``. The arguments ``xllcorner``, ``yllcorner``, ``cellsize``, and ``NODATA_value`` are necessary for the header of the ASCII file.

6. ``write_patch_matrix_as_sa(matrix, file_name)`` - writes a simple ASCII file to ``file_name`` from the patch matrix ``matrix``.

================================================================================
Full API: SolidFileBuilder
================================================================================

1. ``SolidFileBuilder(top=1, bottom=2, side=3)`` - initializes a SolidFileBuilder object with default values for the top, bottom, and sides of a domain, respectively.

2. ``mask(mask_array)`` - applies the matrix array ``mask_array`` to the SolidFileBuilder object.

3. ``write(self, name, xllcorner=0, yllcorner=0, cellsize=0, vtk=False, extra=None, generate_asc_files=False)`` - writes the ``SolidFileBuilder`` object data to the *.pfsol* file ``name``. The arguments ``xllcorner``, ``yllcorner``, and ``cellsize=0`` help define the size of the solid file domain. If ``vtk`` is set to ``True``, it will write a VTK file ``name.vtk`` that you can view in ParaView or another VTK viewer to check that the solid file is correct. If there are any extra arguments you want to pass to the ``pfmask-to-pfsol`` converter in Parflow, specify them using the ``extra`` parameter, as a list of strings. When ``generate_asc_files`` is set to ``True``, this method generates .asc files for top/bottom/sides, with filenames ``<name>_top.asc, <name>_bottom.asc, <name>_front.asc, <name>_back.asc, <name>_left.asc, <name>_right.asc``, and calls ``pfmask-to-pfsol`` with individual ``mask-*`` flags with these files.

4. ``for_key(self, geomItem)`` - sets two keys on the ``Run`` object passed in as the ``geomItem`` argument: 1) ``geomItem.InputType = 'SolidFile'`` 2) ``geomItem.FileName = 'name.pfsol'``. ``'name.pfsol'`` is implicitly referenced from the ``name`` argument of the ``write`` method.

5. ``top(patch_id)`` - sets the ID number of the top of the solid file domain to the integer ``patch_id``. This will override the ``top`` argument passed to the ``SolidFileBuilder`` object.

6. ``bottom(patch_id)`` - sets the ID number of the bottom of the solid file domain to the integer ``patch_id``. This will override the ``bottom`` argument passed to the ``SolidFileBuilder`` object.

7. ``side(patch_id)`` - sets the ID number of the side of the solid file domain to the integer ``patch_id``. This will override the ``side`` argument passed to the ``SolidFileBuilder`` object.

8. ``top_ids(top_patch_ids)`` - sets the ID numbers of the top of the solid file domain to the values in the numpy array ``top_patch_ids``.

9. ``bottom_ids(bottom_patch_ids)`` - sets the ID numbers of the bottom of the solid file domain to the values in the numpy array ``bottom_patch_ids``.

10. ``side_ids(side_patch_ids)`` - sets the ID numbers of the side of the solid file domain to the values in the numpy array ``side_patch_ids``.

================================================================================
More examples
================================================================================

Other example scripts showing how to use the ``SolidFileBuilder`` can be found in *$PARFLOW_SOURCE/test/python/pfsol/*. If you have an idea for a new feature or
improvement to the functionality, please let us know, or better yet, become a contributor!
