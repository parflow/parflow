********************************************************************************
PFB
********************************************************************************


================================================================================
Introduction
================================================================================

ParFlow Binary (PFB) files are an integral part of ParFlow, and we need an easy way to handle them. 
Fortunately, we have a several functions within ``pftools`` that help us with this.  The ``parflow.tools.io`` 
module allows the user to work with numpy arrays, which are easy to visualize and manipulate 
in Python. We'll walk through some examples working with PFB files in Python to see just how powerful 
this is.

================================================================================
Distributing
================================================================================

Let's say you have mastered the conversion of a TCL script to Python, and you have a few PFB 
files that you need to distribute to convert your workflow to Python. Here, you can use the 
``dist()`` method on your ``Run`` object that you created, as mentioned in the first tutorial:

.. code-block:: python3

    LWvdz.dist('lw.1km.slope_x.10x.pfb', 'P'=2, 'Q'=2)

This will distribute the PFB file with the distribution assigned to the ``Process.Topology`` 
keys on the ``Run`` object (``LWvdz`` in this example). However, this can be overwritten for 
a particular file, as shown above.

================================================================================
Creating PFB from Python
================================================================================
Let's copy another test Python script into our tutorial directory:

.. code-block::

    mkdir -p ~/path/pftools_tutorial/pfb_test
    cd ~/path/pftools_tutorial/pfb_test
    cp $PARFLOW_SOURCE/test/python/base/richards_FBx/richards_FBx.py .

This test is a use case where an internal flow boundary is defined as a numpy array, written 
to a PFB file, and distributed for use in the run. Open the file, and add the following 
modules at the top:

.. code-block::

    from parflow import Run
    from parflow.tools.fs import get_absolute_path
    from parflow.tools.io import write_pfb, read_pfb
    import numpy as np

We have already covered the first two in prior tutorials. The third line imports the ``read_pfb`` and
``write_pfb`` functions from the ``parflow.tools.io`` module, and the fourth line imports the ``numpy`` module. 
We convert a numpy array to a PFB file with the ``write_pfb()`` function:

.. code-block:: python3

    # Create numpy array
    FBx_data = np.ones((20, 20, 20))

    # Reduction of 1E-3
    FBx_data[:, :, 9] = 0.001

    # Write flow boundary file as PFB with write_pfb() function
    write_pfb(get_absolute_path('Flow_Barrier_X.pfb'), FBx_data)


This creates a 3D numpy array that covers the entire domain and changes the values in the array 
where X = 10 to 0.001. Note that the numpy array translation to a PFB file reads the dimensions 
as (Z, Y, X). ``write_pfb(get_absolute_path('Flow_Barrier_X.pfb'), FBx_data)`` writes the data 
from the ``FBx_data`` numpy array to a file called ``'Flow_Barrier_X.pfb'``, which will be located in 
the current working directory, and distributes it.

----

Now, try running the file. It should execute successfully. Check out the files you now have in 
your directory - among the other output files is the *'Flow_Barrier_X.pfb'* that you created! 
If you have a PFB reader tool (such as ParaView), you can see what the file looks like: a 
20 x 20 x 20 unit cube with a low-conductivity slice through the middle. Nice!

================================================================================
Loading PFB from Python
================================================================================
Now that we understand how to write a PFB file, how about reading one? This can be useful to do 
inside a Python script so you can visualize or manipulate existing data. Visualizing output data 
within the same script as a run can be very helpful!

----

Let's say you want to visualize some of your output data from the model you just ran, ``richards_FBx.py``. 
In the script, add the following lines to the bottom:

.. code-block:: python3

    FBx_press_out_data = read_pfb(get_absolute_path('richards_FBx.out.press.00010.pfb'))

    print(f'Dimensions of output file: {FBx_press_out_data.shape}')
    print(FBx_press_out_data)

The first line reads the PFB file of the output pressure field at time step = 10 and converts the data to a 
numpy array. The ``print`` statements print the dimensions of the array and the data from the file. Save and 
run this script again to see the printed output. If you're savvy with ``matplotlib`` or other visualization 
packages in Python, feel free to visualize to your heart's content!

================================================================================
Full API
================================================================================

1. ``read_pfb(file: str, keys: dict=None, mode: str='full', z_first: bool=True)``
    Write a single pfb file. The data must be a 3D numpy array with ``float64``
    values. The number of subgrids in the saved file will be ``p`` * ``q`` * ``r``. This
    is regardless of the number of subgrids in the PFB file loaded by the
    ParflowBinaryReader into the numpy array. Therefore, loading a file with
    ParflowBinaryReader and saving it with this method may restructure the
    file into a different number of subgrids if you change these values.
    
    If dist is True then also write a file with the .dist extension added to
    the file_name. The ``.dist`` file will contain one line per subgrid with the
    offset of the subgrid in the ``.pfb`` file.

    :param ``file``: The name of the file to write the array to.
    :param ``array``: The array to write.
    :param ``p``: Number of subgrids in the x direction.
    :param ``q``: Number of subgrids in the y direction.
    :param ``r``: Number of subgrids in the z direction.
    :param ``x``: The length of the x-axis
    :param ``y``: The length of the y-axis
    :param ``z``: The length of the z-axis
    :param ``dx``: The spacing between cells in the x direction
    :param ``dy``: The spacing between cells in the y direction
    :param ``dz``: The spacing between cells in the z direction
    :param ``z_first``: Whether the z-axis should be first or last.
    :param ``dist``: Whether to write the distfile in addition to the pfb.
    :param ``kwargs``: Extra keyword arguments, primarily to eat unnecessary args by passing in a dictionary with ``**dict``.

2. ``write_pfb(file, array, p=1, q=1, r=1, x=0.0, y=0.0, z=0.0, dx=1.0, dy=1.0, dz=1.0, z_first=True, dist=True, **kwargs)``
    Write a single pfb file. The data must be a 3D numpy array with ``float64``
    values. The number of subgrids in the saved file will be ``p`` * ``q`` * ``r``. This
    is regardless of the number of subgrids in the PFB file loaded by the
    ParflowBinaryReader into the numpy array. Therefore, loading a file with
    ParflowBinaryReader and saving it with this method may restructure the
    file into a different number of subgrids if you change these values.

    If dist is True then also write a file with the ``.dist`` extension added to
    the file_name. The .dist file will contain one line per subgrid with the
    offset of the subgrid in the ``.pfb`` file.

    :param ``file``: The name of the file to write the array to.
    :param ``array``: The array to write.
    :param ``p``: Number of subgrids in the x direction.
    :param ``q``: Number of subgrids in the y direction.
    :param ``r``: Number of subgrids in the z direction.
    :param ``x``: The length of the x-axis
    :param ``y``: The length of the y-axis
    :param ``z``: The length of the z-axis
    :param ``dx``: The spacing between cells in the x direction
    :param ``dy``: The spacing between cells in the y direction
    :param ``dz``: The spacing between cells in the z direction
    :param ``z_first``: Whether the z-axis should be first or last.
    :param ``dist``: Whether to write the distfile in addition to the pfb.
    :param ``kwargs``: Extra keyword arguments, primarily to eat unnecessary args by passing in a dictionary with ``**dict``.

3. ``write_dist(file, sg_offs)``
    Write a distfile.

    :param ``file``: The path of the file to be written.
    :param ``sg_offs``: The subgrid offsets.

4. ``read_pfb_sequence(file_seq: Iterable[str], keys=None, z_first: bool=True, z_is: str='z')``
    An efficient wrapper to read a sequence of pfb files. This
    approach is faster than looping over the ``read_pfb`` function
    because it caches the subgrid information from the first
    pfb file and then uses that to initialize all other readers.

    :param ``file_seq``: An iterable sequence of file names to be read.
    :param ``keys``: A set of keys for indexing subarrays of the full pfb. Optional. This is mainly a trick for interfacing with 
        xarray, but the format of the keys is:

        ::

            {'x': {'start': start_x, 'stop': end_x},
            'y': {'start': start_y, 'stop': end_y},
            'z': {'start': start_z, 'stop': end_z}}

    :param ``z_first``: Whether the z dimension should be first. If true returned arrays have dimensions ``('z', 'y', 'x')`` else ``('x', 'y', 'z')``
    :param ``z_is``: A descriptor of what the z axis represents. Can be one of ``'z'``, ``'time'``, ``'variable'``. Default is ``'z'``.
    :return: An ``ndarray`` containing the data from the files.
