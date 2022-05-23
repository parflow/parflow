********************************************************************************
PFB
********************************************************************************


================================================================================
Introduction
================================================================================

ParFlow Binary (PFB) files are an integral part of ParFlow, and we need an easy way to handle them. Fortunately, we have a supporting Python package, ``parflowio``, that helps us with this. The ``parflowio`` package is included when you install ``pftools`` using either ``pip install pftools[all]`` or ``pip install pftools[pfb]``. ``parflowio`` allows the user to work with numpy arrays, which are easy to visualize and manipulate in Python. We'll walk through some examples working with PFB files in Python to see just how powerful this is.

================================================================================
Distributing
================================================================================

Let's say you have mastered the conversion of a TCL script to Python, and you have a few PFB files that you need to distribute to convert your workflow to Python. Here, you can use the ``dist()`` method on your ``Run`` object that you created, as mentioned in the first tutorial:

.. code-block:: python3

    LWvdz.dist('lw.1km.slope_x.10x.pfb', 'P'=2, 'Q'=2)

This will distribute the PFB file with the distribution assigned to the ``Process.Topology`` keys on the ``Run`` object (``LWvdz`` in this example). However, this can be overwritten for a particular file, as shown above. Since ``dist()`` is a method on the ``Run`` object, you do not need to add any commands to your script to load the ``parflowio`` module if this is all you want to do. However, if you plan to work more with PFB files inside your script, you will need to load this module, as you'll see in the next example.

================================================================================
Creating PFB from Python
================================================================================
Let's copy another test Python script into our tutorial directory:

.. code-block:: language

    mkdir -p ~/path/pftools_tutorial/pfb_test
    cd ~/path/pftools_tutorial/pfb_test
    cp $PARFLOW_SOURCE/test/python/base/richards_FBx/richards_FBx.py .

This test is a use case where an internal flow boundary is defined as a numpy array, written to a PFB file, and distributed for use in the run. Open the file, and you'll see the following modules at the top:

.. code-block:: language

    from parflow import Run
    from parflow.tools.fs import get_absolute_path
    from parflowio.pyParflowio import PFData
    import numpy as np

We have already covered the first two in prior tutorials. The third line imports the ``PFData`` class from the ``parflowio.pyParflowio`` module, and the fourth line imports the ``numpy`` module. We convert a numpy array to a PFB file by instantiating it as a ``PFData`` object. Head down to lines 172 through 184 to see how this is used:

.. code-block:: python3

    ## write flow boundary file
    FBx_data = np.full((20, 20, 20), 1.0)
    for i in range(20):
        for j in range(20):
            # from cell 10 (index 9) to cell 11
            # reduction of 1E-3
            FBx_data[i, j, 9] = 0.001

    FBx_data_pfb = PFData(FBx_data)
    FBx_data_pfb.writeFile(get_absolute_path('Flow_Barrier_X.pfb'))
    FBx_data_pfb.close()

    rich_fbx.dist('Flow_Barrier_X.pfb')

This creates a 3D numpy array that covers the entire domain and changes the values in the array where X = 10 to 0.001. Note that the numpy array translation to a PFB file reads the dimensions as (Z, Y, X). ``FBx_data_pfb = PFData(FBx_data)`` takes the numpy array and instantiates it as a ``PFData`` object. ``FBx_data_pfb.writeFile(get_absolute_path('Flow_Barrier_X.pfb'))`` writes the data from the ``PFData`` object to a file ``'Flow_Barrier_X.pfb'``, which will be located in the current working directory. ``FBx_data_pfb.close()`` closes the file. ``rich_fbx.dist('Flow_Barrier_X.pfb')`` connects the new PFB file to the ``Run`` object and distributes it.

----

Now, try running the file. It should execute successfully. Check out the files you now have in your directory - among the other output files is the *'Flow_Barrier_X.pfb'* that you created! If you have a PFB reader tool (such as ParaView), you can see what the file looks like: a 20 x 20 x 20 unit cube with a low-conductivity slice through the middle. Nice!

================================================================================
Loading PFB from Python
================================================================================
Now that we understand how to write a PFB file, how about reading one? This can be useful to do inside a Python script so you can visualize or manipulate existing data. Visualizing output data within the same script as a run can be very helpful!

----

Let's say you want to visualize some of your output data from the model you just ran, ``richards_FBx.py``. In the script, add the following lines to the bottom:

.. code-block:: python3

    FBx_press_out = PFData(get_absolute_path('richards_FBx.out.press.00010.pfb'))
    FBx_press_out.loadHeader()
    FBx_press_out.loadData()
    FBx_press_out_data = FBx_press_out.viewDataArray()

    print(f'Dimensions of output file: {FBx_press_out_data.shape}')
    print(FBx_press_out_data)

The first line reads the PFB file of the output pressure field at time step = 10 and instantiates it as a ``PFData`` object. ``loadHeader()`` and ``loadData()`` load the header of the binary file (to figure out the dimensions of the file) and loads the data in the file, respectively. ``FBx_press_out_data = FBx_press_out.viewDataArray()`` converts the data to a numpy array and sets it to ``FBx_press_out_data``. The ``print`` statements print the dimensions of the array and the data from the file. Save and run this script again to see the printed output. If you're savvy with ``matplotlib`` or other visualization packages in Python, feel free to visualize to your heart's content!
