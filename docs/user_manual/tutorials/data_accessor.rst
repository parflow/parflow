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

#. ``data.time``
    Get current timestep set within the DataAccessor class. This timestep will be used to access timeseries data.
    
    :return: An integer value representing the current timestep

#. ``data.time(time)``
    Set current timestep.
    
    :param ``time``: Integer representing current timestep

#. ``data.times``
    :return: An array of timesteps, from 0 to n_timesteps - 1

#. ``data.forcing_time``
    :return: Return value

#. ``data.forcing_time(time)``
    Set current forcing timestep.
    
    :param ``time``: Integer representing current forcing timestep

#. ``data.shape``
    :return: Tuple containing ``(ComputationalGrid.NZ, ComputationalGrid.NY, ComputationalGrid.NX)``

#. ``data.dx``
    :return: Value of ``ComputationalGrid.DX``

#. ``data.dy``
    :return: Value of ``ComputationalGrid.DY``

#. ``data.dz``
    :return: array of size ``ComputationalGrid.NZ``

#. ``data.mannings``
    :return: Array of mannings roughness coefficient data

#. ``data.mask``
    :return: Array representing the mask of the current ParFlow run

#. ``data.slope_x``
    :return: Return value

#. ``data.slope_y``
    :return: Return value

#. ``data.elevation``
    :return: Return value

#. ``data.computed_porosity``
    :return: Return value

#. ``data.computed_permeability_x``
    :return: Return value

#. ``data.computed_permeability_y``
    :return: Return value

#. ``data.computed_permeability_z``
    :return: Return value

#. ``data.pressure_initial_condition``
    :return: Return value

#. ``data.pressure_boundary_conditions``
    :return: Return value

#. ``data.pressure``
    :return: Return value

#. ``data.saturation``
    :return: Return value

#. ``data.specific_storage``
    :return: Return value

#. ``data.et``
    :return: Return value

#. ``data.overland_flow()``
    :return: Return value

#. ``data.overland_flow_grid()``
    :return: Return value

#. ``data.subsurface_storage``
    :return: Return value

#. ``data.surface_storage``
    :return: Return value

#. ``data.wtd``
    :return: Return value

#. ``data.clm_output``
    :return: Return value

#. ``data.clm_output_variables``
    :return: Tuple containing names of all CLM output variables: ``('eflx_lh_tot', 'eflx_lwrad_out', 'eflx_sh_tot', 
        'eflx_soil_grnd', 'qflx_evap_tot', 'qflx_evap_grnd', 'qflx_evap_soi', 'qflx_evap_veg', 'qflx_tran_veg', 'qflx_infl',
        'swe_out', 't_grnd', 'qflx_qirr', 't_soil')``


#. ``data.clm_output_diagnostics``
    :return: Return value

#. ``data.clm_output_eflx_lh_tot``
    :return: Return value

#. ``clm_output_eflx_lwrad_out``
    :return: Return value

#. ``clm_output_eflx_sh_tot``
    :return: Return value

#. ``clm_output_eflx_soil_grnd``
    :return: Return value

#. ``clm_output_qflx_evap_grnd``
    :return: Return value

#. ``clm_output_qflx_evap_soi``
    :return: Return value

#. ``lm_output_qflx_evap_tot``
    :return: Return value

#. ``clm_output_qflx_evap_veg``
    :return: Return value

#. ``clm_output_qflx_infl``
    :return: Return value

#. ``clm_output_qflx_top_soil``
    :return: Return value

#. ``clm_output_qflx_tran_veg``
    :return: Return value

#. ``clm_output_swe_out``
    :return: Return value

#. ``clm_output_t_grnd``
    :return: Return value

#. ``clm_forcing(name)``
    :param name: Type of forcing you're interested in
    :return: Array containing 

#. ``clm_forcing_dswr``
    :return: Return value

#. ``clm_forcing_dlwr``
    :return: Return value

#. ``clm_forcing_apcp``
    :return: Return value

#. ``clm_forcing_temp``
    :return: Return value

#. ``clm_forcing_ugrd``
    :return: Return value

#. ``clm_forcing_vgrd``
    :return: Return value

#. ``clm_forcing_press``
    :return: Return value

#. ``clm_forcing_spfh``
    :return: Return value

#. ``clm_map_land_fraction(name)``
    :param name: jfdlks
    :return: Return value

#. ``clm_map_latitude``
    :return: Value of ``Solver.CLM.Vegetation.Map.Latitude`` key

#. ``clm_map_longitude``
    :return: Value of ``Solver.CLM.Vegetation.Map.Longitude`` key

#. ``clm_map_sand``
    :return: Value of ``Solver.CLM.Vegetation.Map.Sand`` key

#. ``clm_map_clay``
    :return: Value of ``Solver.CLM.Vegetation.Map.Clay`` key

#. ``clm_map_color``
    :return: Value of ``Solver.CLM.Vegetation.Map.Color`` key
