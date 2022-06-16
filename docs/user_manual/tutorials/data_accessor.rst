********************************************************************************
Data Accessor
********************************************************************************

================================================================================
Introduction
================================================================================

The ``DataAccessor`` class is a helper class for extracting numpy arrays from a given
ParFlow run.

================================================================================
Usage of ``DataAccessor``
================================================================================

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


================================================================================
Full API
================================================================================

#. ``run.data_accessor`` 
    Accesses the ``DataAccessor`` object on the current ``Run`` object ``run``. 

    :return: An instance of the ``DataAccessor`` class.

#. ``data.time``
    Get current timestep set on the current ``DataAccessor`` object ``data``. This timestep will be used to determine the file index for accessing timeseries data.
    
    :return: An integer value representing the current timestep.

#. ``data.times``
    :return: An array of timesteps on the current ``DataAccessor`` object ``data``, from 0 to n_timesteps - 1.

#. ``data.forcing_time``
    Get current forcing timestep set on the current ``DataAccessor`` object ``data``. This timestep will be used to determine the file index for accessing forcing timeseries data.

    :return: An integer value representing the current forcing timestep.

#. ``data.shape``
    :return: Tuple containing ``(ComputationalGrid.NZ, ComputationalGrid.NY, ComputationalGrid.NX)`` set on the current ``DataAccessor`` object ``data``.

#. ``data.dx``
    :return: Value of ``ComputationalGrid.DX`` on the current ``DataAccessor`` object ``data``.

#. ``data.dy``
    :return: Value of ``ComputationalGrid.DY`` on the current ``DataAccessor`` object ``data``.

#. ``data.dz``
    :return: Array of size ``ComputationalGrid.NZ`` containing either ``dz_scale`` values or the value of ``ComputationalGrid.DZ`` set on the current ``DataAccessor`` object ``data``.

#. ``data.mannings``
    :return: An ``ndarray`` containing mannings roughness coefficient data for the current ``DataAccessor`` object ``data``.

#. ``data.mask``
    :return: An ``ndarray`` containing the mask of your domain for the current ``DataAccessor`` object ``data``.

#. ``data.slope_x``
    :return: An ``ndarray`` containing the ``x`` topographic slope values for the current ``DataAccessor`` object ``data``.

#. ``data.slope_y``
    :return: An ``ndarray`` containing the ``y`` topographic slope values for the current ``DataAccessor`` object ``data``.

#. ``data.elevation``
    :return: An ``ndarray`` containing the elevation topographic slope values for the current ``DataAccessor`` object ``data``.

#. ``data.computed_porosity``
    :return: An ``ndarray`` containing computed porosity values on the current ``DataAccessor`` object ``data``.

#. ``data.computed_permeability_x``
    :return: An ``ndarray`` containing computed permeability ``x`` values on the current ``DataAccessor`` object ``data``.

#. ``data.computed_permeability_y``
    :return: An ``ndarray`` containing computed permeability ``y`` values on the current ``DataAccessor`` object ``data``.

#. ``data.computed_permeability_z``
    :return: An ``ndarray`` containing computed permeability ``z`` values on the current ``DataAccessor`` object ``data``.

#. ``data.pressure_initial_condition``
    :return: An ``ndarray`` containing initial condition pressure values on the current ``DataAccessor`` object ``data``.

#. ``data.pressure_boundary_conditions``
    :return: A dictionary containing ``key=value`` pairs of the form ``{patch_name}__{cycle_name} = value`` for pressure boundary conditions on the current ``DataAccessor`` object ``data``.

#. ``data.pressure``
    :return: An ``ndarray`` containing pressure values for the current timestep set on the ``DataAccessor`` object ``data``.

#. ``data.saturation``
    :return: An ``ndarray`` containing saturation values for the current timestep set on the ``DataAccessor`` object ``data``.

#. ``data.specific_storage``
    :return: An ``ndarray`` containing specific storage values for the current ``DataAccessor`` object ``data``.

#. ``data.et``
    :return: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of evapotranspiration values (units L^3/T), spanning all layers (bottom to top) for the current ``DataAccessor`` object ``data``.

#. ``data.overland_flow(flow_method='OverlandKinematic', epsilon=1e-5)``
    :param ``flow_method``: Either ``'OverlandFlow'`` or ``'OverlandKinematic'``. ``'OverlandKinematic'`` by default.
    :param ``epsilon``: Minimum slope magnitude for solver. Only applicable if ``flow_method='OverlandKinematic'``. This is set using the ``Solver.OverlandKinematic.Epsilon`` key in Parflow.

    :return: An ``ny`` by ``nx`` ``ndarray`` of overland flow values for the current ``DataAccessor`` object ``data``.

#. ``data.overland_flow_grid(flow_method='OverlandKinematic', epsilon=1e-5)``
    :param ``flow_method``: Either ``'OverlandFlow'`` or ``'OverlandKinematic'``. ``'OverlandKinematic'`` by default.
    :param ``epsilon``: Minimum slope magnitude for solver. Only applicable if ``kinematic=True``. This is set using the ``Solver.OverlandKinematic.Epsilon`` key in Parflow.

    :return: An ``ny`` by ``nx`` ``ndarray`` of overland flow values for the current ``DataAccessor`` object ``data``.

#. ``data.subsurface_storage``
    :return: An ``nz`` by ``ny`` by ``nx`` ``ndarray`` of subsurface storage values, spanning all layers (bottom to top), for the current ``DataAccessor`` object ``data``.

#. ``data.surface_storage``
    :return: An ``ny`` by ``nx`` ``ndarray`` of surface storage values for the current ``DataAccessor`` object ``data``.

#. ``data.wtd``
    :return: An ``ny`` by ``nx`` ``ndarray`` of water table depth values (measured from the top) for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output(field, layer=-1)``
    :param field: CLM field, one of: ``'eflx_lh_tot', 'eflx_lwrad_out', 'eflx_sh_tot', 'eflx_soil_grnd', 'qflx_evap_tot', 'qflx_evap_grnd', 'qflx_evap_soi', 'qflx_evap_veg', 'qflx_tran_veg', 'qflx_infl',
        'swe_out', 't_grnd', 'qflx_qirr', 't_soil'``
    :param layer: Layer of data
    :return: An ``ndarray`` of CLM output for the given ``field`` and ``layer`` on the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_variables``
    :return: Tuple containing names of all CLM output variables: ``('eflx_lh_tot', 'eflx_lwrad_out', 'eflx_sh_tot', 
        'eflx_soil_grnd', 'qflx_evap_tot', 'qflx_evap_grnd', 'qflx_evap_soi', 'qflx_evap_veg', 'qflx_tran_veg', 'qflx_infl',
        'swe_out', 't_grnd', 'qflx_qirr', 't_soil')``

#. ``data.clm_output_diagnostics``
    :return: String filepath to CLM output diagnostics file for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_eflx_lh_tot``
    :return: An ``ndarray`` containing CLM ``eflx_lh_tot`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_eflx_lwrad_out``
    :return: An ``ndarray`` containing CLM ``eflx_lwrad_out`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_eflx_sh_tot``
    :return: An ``ndarray`` containing CLM ``eflx_sh_tot`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_eflx_soil_grnd``
    :return: An ``ndarray`` containing CLM ``eflx_soil_grnd`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_evap_grnd``
    :return: An ``ndarray`` containing CLM ``qflx_evap_grnd`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_evap_soi``
    :return: An ``ndarray`` containing CLM ``qflx_evap_soi`` data for the current ``DataAccessor`` object ``data``.

#. ``data.lm_output_qflx_evap_tot``
    :return: An ``ndarray`` containing CLM ``qflx_evap_tot`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_evap_veg``
    :return: An ``ndarray`` containing CLM ``qflx_evap_veg`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_infl``
    :return: An ``ndarray`` containing CLM ``qflx_infl`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_top_soil``
    :return: An ``ndarray`` containing CLM ``qflx_top_soil`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_qflx_tran_veg``
    :return: An ``ndarray`` containing CLM ``qflx_tran_veg`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_swe_out``
    :return: An ``ndarray`` containing CLM ``swe_out`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_output_t_grnd``
    :return: An ``ndarray`` containing CLM ``t_grnd`` data for the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing(name)``
    :param ``name``: Forcing type you're interested in
    :return: An ``ndarray`` containing CLM forcing data for the given ``name`` and forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_dswr``
    :return: An ``ndarray`` containing CLM forcing data for Downward Visible or Short-Wave radiation [W/m2] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_dlwr``
    :return: An ``ndarray`` containing CLM forcing data for Downward Infa-Red or Long-Wave radiation [W/m2] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_apcp``
    :return: An ``ndarray`` containing CLM forcing data for Precipitation rate [mm/s] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_temp``
    :return: An ``ndarray`` containing CLM forcing data for Air temperature [K] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_ugrd``
    :return: An ``ndarray`` containing CLM forcing data for West-to-East or U-component of wind [m/s] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_vgrd``
    :return: An ``ndarray`` containing CLM forcing data for South-to-North or V-component of wind [m/s] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_press``
    :return: An ``ndarray`` containing CLM forcing data for Atmospheric Pressure [pa] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_forcing_spfh``
    :return: An ``ndarray`` containing CLM forcing data for Water-vapor specific humidity [kg/kg] for the forcing timestep set on the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_land_fraction(name)``
    :param name: Type of land frac data you're interested in
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.LandFrac[name]`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_latitude``
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.Latitude`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_longitude``
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.Longitude`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_sand``
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.Sand`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_clay``
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.Clay`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.

#. ``data.clm_map_color``
    :return: Data corresponding to ``Solver.CLM.Vegetation.Map.Color`` key set on the ParFlow run for the current ``DataAccessor`` object ``data``.
