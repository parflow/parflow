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

Here is an example of using Hydrology

.. code-block:: python3

    from parflow.tools.hydrology import calculate_surface_storage, calculate_subsurface_storage, \
        calculate_water_table_depth, calculate_evapotranspiration, calculate_overland_flow_grid

    


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
