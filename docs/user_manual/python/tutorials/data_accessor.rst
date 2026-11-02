.. _data_accessor:

Data Accessor
==============

.. _data_accessor_intro:

Introduction
-------------

The ``DataAccessor`` class is a helper class for extracting numpy arrays from a given
ParFlow run.

.. _data_accessor_usage:

Usage of ``DataAccessor``
-------------------------

First, we’ll show some examples of using the DataAccessor class within a ParFlow Python script:

.. code-block:: python3

    from parflow import Run

    # Create a Run object from a .pfidb file
    run = Run.from_definition('/path/to/pfidb/file')

    # Get the DataAccessor object corresponding to the Run object
    data = run.data_accessor

    # Iterate through the timesteps of the DataAccessor object
    # i goes from 0 to n_timesteps - 1
    for i in data.times:

        #----------------------------- Evapotranspiration -------------------------------

        # nz-by-ny-by-nx array of ET values (bottom to top layer)
        print(data.et)

        #------------------------------- Overland Flow ----------------------------------

        # ny-by-nx array of overland flow values - 'OverlandKinematic' flow method
        print(data.overland_flow_grid())

        # ny-by-nx array of overland flow values - 'OverlandFlow' flow method
        print(data.overland_flow_grid(flow_method='OverlandFlow'))

        # Total outflow for the domain (scalar value) - 'OverlandKinematic' flow method
        print(data.overland_flow())
        
        # Total outflow for the domain (scalar value) - 'OverlandFlow' flow method
        print(data.overland_flow(flow_method='OverlandFlow'))

        #-------------------------- Subsurface/Surface Storage --------------------------

        # nz-by-ny-by-nx array of subsurface storage values (bottom to top layer)
        print(data.subsurface_storage)

        # ny-by-nx array of surface storage values
        print(data.surface_storage)

        #----------------------------- Water Table Depth --------------------------------

        # ny-by-nx array of water table depth values
        print(data.wtd)

        data.time += 1
