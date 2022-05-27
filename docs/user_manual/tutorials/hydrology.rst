********************************************************************************
Hydrology Module
********************************************************************************

================================================================================
Introduction
================================================================================

The Python PFTools Hydrology module provides standalone functions for common hydrologic calculations.

================================================================================
Usage of ``Hydrology``
================================================================================

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

================================================================================
Full API
================================================================================

1. ``calculate_water_table_depth(pressure, saturation, dz)``
    Calculate water table depth from the land surface.

    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``saturation``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` ndarray of saturation values (bottom layer to top layer)
    :param ``dz``: An ``ndarray`` of shape ``(nz,)`` of thickness values (bottom layer to top layer)
    :return: A ``ny`` by ``nx`` ``ndarray`` of water table depth values (measured from the top)

2. ``calculate_subsurface_storage(porosity, pressure, saturation, specific_storage, dx, dy, dz, mask=None)``
    Calculate gridded subsurface storage across several layers. For each layer in the subsurface, storage consists of two parts:

        1) Incompressible subsurface storage (``porosity`` * ``saturation`` * depth of this layer) * ``dx`` * ``dy``
        2) Compressible subsurface storage (``pressure`` * ``saturation`` * ``specific storage`` * depth of this layer) * ``dx`` * ``dy``

    :param ``porosity``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of porosity values (bottom layer to top layer)
    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``saturation``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of saturation values (bottom layer to top layer)
    :param ``specific_storage``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of specific storage values (bottom layer to top layer)
    :param ``dx``: Length of a grid element in the ``x`` direction
    :param ``dy``: Length of a grid element in the ``y`` direction
    :param ``dz``: Thickness of a grid element in the ``z`` direction (bottom layer to top layer)
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If ``None``, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of subsurface storage values, spanning all layers (bottom to top)

3. ``calculate_surface_storage(pressure, dx, dy, mask=None)``
    Calculate gridded surface storage on the top layer. Surface storage is given by: Pressure at the top layer * ``dx`` * ``dy`` (for pressure values > 0)

    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``dx``: Length of a grid element in the x direction
    :param ``dy``: Length of a grid element in the y direction
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If ``None``, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: An ``ny`` by ``nx`` ``ndarray`` of surface storage values

4. ``calculate_evapotranspiration(et, dx, dy, dz, mask=None)``
    Calculate gridded evapotranspiration across several layers.

    :param ``et``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of evapotranspiration flux values with units 1/T (bottom layer to top layer)
    :param ``dx``: Length of a grid element in the ``x`` direction
    :param ``dy``: Length of a grid element in the ``y`` direction
    :param ``dz``: Thickness of a grid element in the ``z`` direction (bottom layer to top layer)
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If ``None``, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of evapotranspiration values (units L^3/T), spanning all layers (bottom to top)

5. ``calculate_overland_fluxes(pressure, slopex, slopey, mannings, dx, dy, flow_method='OverlandKinematic', epsilon=1e-5, mask=None)``
    Calculate overland fluxes across grid faces.

    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``slopex``: ``ny`` by ``nx``
    :param ``slopey``: ``ny`` by ``nx``
    :param ``mannings``: a scalar value, or a ``ny`` by ``nx`` ``ndarray``
    :param ``dx``: Length of a grid element in the ``x`` direction
    :param ``dy``: Length of a grid element in the ``y`` direction
    :param ``flow_method``: Either 'OverlandFlow' or 'OverlandKinematic'. 'OverlandKinematic' by default.
    :param ``epsilon``: Minimum slope magnitude for solver. Only applicable if ``flow_method='OverlandKinematic'``. This is set using the ``Solver.OverlandKinematic.Epsilon`` key in Parflow.
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If ``None``, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: A 2-tuple: 

        (``qeast``: A ``ny`` by ``(nx+1)`` ``ndarray`` of overland flux values,  

        ``qnorth``: A ``(ny+1)`` by ``nx`` ``ndarray`` of overland flux values)

    ::

        Numpy array origin is at the top left.
        The cardinal direction along axis 0 (rows) is North (going down!!).
        The cardinal direction along axis 1 (columns) is East (going right).
        qnorth ``(ny+1,nx)`` and qeast ``(ny,nx+1)`` values are to be interpreted as follows.

        +-------------------------------------> (East)
        |
        |                           qnorth_i,j (outflow if negative)
        |                                  +-----+------+
        |                                  |     |      |
        |                                  |     |      |
        |  qeast_i,j (outflow if negative) |-->  v      |---> qeast_i,j+1 (outflow if positive)
        |                                  |            |
        |                                  | Cell  i,j  |
        |                                  +-----+------+
        |                                        |
        |                                        |
        |                                        v
        |                           qnorth_i+1,j (outflow if positive)
        v
        (North)


6. ``calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, flow_method='OverlandKinematic', epsilon=1e-5, mask=None)``
    Calculate overland outflow per grid cell of a domain.

    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``slopex``: ``ny`` by ``nx``
    :param ``slopey``: ``ny`` by ``nx``
    :param ``mannings``: a scalar value, or a ``ny`` by ``nx`` ``ndarray``
    :param ``dx``: Length of a grid element in the ``x`` direction
    :param ``dy``: Length of a grid element in the ``y`` direction
    :param ``flow_method``: Either 'OverlandFlow' or 'OverlandKinematic'. 'OverlandKinematic' by default.
    :param ``epsilon``: Minimum slope magnitude for solver. Only applicable if ``kinematic=True``. This is set using the ``Solver.OverlandKinematic.Epsilon`` key in Parflow.
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If ``None``, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: An ``ny`` by ``nx`` ``ndarray`` of overland flow values

7. ``calculate_overland_flow(pressure, slopex, slopey, mannings, dx, dy, flow_method='OverlandKinematic', epsilon=1e-5, mask=None)``

    :param ``pressure``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of pressure values (bottom layer to top layer)
    :param ``slopex``: ``ny`` by ``nx``
    :param ``slopey``: ``ny`` by ``nx``
    :param ``mannings``: a scalar value, or a ``ny`` by ``nx`` ``ndarray``
    :param ``dx``: Length of a grid element in the ``x`` direction
    :param ``dy``: Length of a grid element in the ``y`` direction
    :param ``flow_method``: Either 'OverlandFlow' or 'OverlandKinematic'. 'OverlandKinematic' by default.
    :param ``epsilon``: Minimum slope magnitude for solver. Only applicable if ``flow_method='OverlandKinematic'``. This is set using the ``Solver.OverlandKinematic.Epsilon`` key in Parflow.
    :param ``mask``: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of mask values (bottom layer to top layer). If None, assumed to be an ``nz`` by ``ny`` by ``nx`` ``ndarray`` of 1s.
    :return: A ``ny`` by ``nx`` ``ndarray`` of overland flow values
