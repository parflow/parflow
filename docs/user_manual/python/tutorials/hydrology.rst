.. _hydrology:

Hydrology Module
=================

.. _hydrology_intro:

Introduction
-------------

The Python PFTools Hydrology module provides standalone functions for common hydrologic calculations.

.. _hydrology_usage:

Usage of ``Hydrology``
----------------------

First, we’ll show some examples of using the Hydrology class within a ParFlow Python script:

.. code-block:: python3

    import numpy as np
    from parflow import Run
    from parflow.tools.hydrology import calculate_surface_storage, calculate_subsurface_storage, \
        calculate_water_table_depth, calculate_evapotranspiration, calculate_overland_flow_grid

    # Create a Run object from the .pfidb file
    run = Run.from_definition('/path/to/pfidb/file')

    # Get the DataAccessor object corresponding to the Run object
    data = run.data_accessor

    # ----------------------------------------------
    # Get relevant information from the DataAccessor
    # ----------------------------------------------

    # Resolution
    dx = data.dx
    dy = data.dy
    # Thickness of each layer, bottom to top
    dz = data.dz

    # Extent
    nx = data.shape[2]
    ny = data.shape[1]
    nz = data.shape[0]

    # ------------------------------------------
    # Time-invariant values
    # ------------------------------------------

    porosity = data.computed_porosity
    specific_storage = data.specific_storage
    mask = data.mask
    et = data.et                        # shape (nz, ny, nx) - units 1/T.
    slopex = data.slope_x               # shape (ny, nx)
    slopey = data.slope_y               # shape (ny, nx)
    mannings = data.mannings            # scalar value

    # ------------------------------------------
    # Time-variant values
    # ------------------------------------------

    # no. of time steps
    nt = len(data.times)

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Arrays for total values (across all layers), with time as the first axis
    subsurface_storage = np.zeros(nt)
    surface_storage = np.zeros(nt)
    wtd = np.zeros((nt, ny, nx))
    et = np.zeros(nt)
    overland_flow = np.zeros((nt, ny, nx))

    # ------------------------------------------
    # Loop through time steps
    # i goes from 0 to n_timesteps - 1
    # ------------------------------------------
    for i in data.times:
        pressure = data.pressure
        saturation = data.saturation

        # Total subsurface storage for this time step is the summation of substorage surface across all x/y/z slices
        subsurface_storage[i, ...] = np.sum(
            calculate_subsurface_storage(porosity, pressure, saturation, specific_storage, dx, dy, dz, mask=mask),
            axis=(0, 1, 2)
        )

        # Total surface storage for this time step is the summation of substorage surface across all x/y slices
        surface_storage[i, ...] = np.sum(
            calculate_surface_storage(pressure, dx, dy, mask=mask),
            axis=(0, 1)
        )

        wtd[i, ...] = calculate_water_table_depth(pressure, saturation, dz)

        if et is not None:
            # Total ET for this time step is the summation of ET values across all x/y/z slices
            et[i, ...] = np.sum(
                calculate_evapotranspiration(et_flux_values, dx, dy, dz, mask=mask),
                axis=(0, 1, 2)
            )

        overland_flow[i, ...] = calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, mask=mask)

        data.time += 1
