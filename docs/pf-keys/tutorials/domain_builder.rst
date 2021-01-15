********************************************************************************
Domain definition helpers
********************************************************************************


================================================================================
Introduction
================================================================================

One of ParFlow's strengths is its customizability; you can practically define any type of hydrologic problem with it.
One of the downsides of that, however, is that setting all the keys can be cumbersome, especially when starting a run from scratch.
With the new ``DomainBuilder``, Python-PFTools helps condense the setting of keys for many common problem definitions.

================================================================================
Usage of ``DomainBuilder``
================================================================================

First, we'll show some usage examples of loading tables of parameters within a ParFlow Python script:

.. code-block:: python3

    from parflow import Run
    from parflow.tools.builders import DomainBuilder

    LW_Test = Run("LW_Test", __file__)

    # ----------------------------------------------------------------------------

    bounds = [
        0.0, 41000.0,
        0.0, 41000.0,
        0.0, 100.0
    ]

    domain_patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    zero_flux_patches = 'x_lower x_upper y_lower y_upper z_lower'

    DomainBuilder(LW_Test) \
        .no_wells() \
        .no_contaminants() \
        .water('domain') \
        .variably_saturated() \
        .box_domain('box_input', 'domain', bounds, domain_patches) \
        .homogeneous_subsurface('domain', specific_storage=1.0e-5, isotropic=True) \
        .zero_flux(zero_flux_patches, 'constant', 'alltime') \
        .slopes_mannings('domain', slope_x='LW.slopex.pfb', slope_y='LW.slopey.pfb', mannings=5.52e-6) \
        .ic_pressure('domain', patch='z_upper', pressure='press.init.pfb')

In this example, the 10 lines associated with the instantiation of the ``DomainBuilder`` class generate about 70 keys!
As is possible with any other key setting, you can always overwrite the keys as necessary; the ``DomainBuilder`` is designed to help you get started.
Once you instantitate the ``DomainBuilder`` object on a ``Run`` object, each method will set various keys with the given arguments, which are described below.

================================================================================
Full API
================================================================================

1. ``DomainBuilder(run, name='domain')``: Instantiates the ``DomainBuilder`` object on the ``Run`` object ``run``. The ``name`` argument is used to define this key:

.. code-block:: python3

      run.Domain.GeomName = name

The following examples of the method usage assume that the name of the ``Run`` object is ``run``. All arguments for methods that are passed in as tokens in keys are denoted by ``{argument}``.

2. ``no_wells()``: Sets the key ``run.Wells.Names = ''``

3. ``no_contaminants()``: Sets the key ``run.Contaminants.Names = ''``

4. ``water(self, geom_name=None)``: Sets the following keys:

.. code-block:: python3

      run.Gravity = 1.0
      run.Phase.Names = 'water'
      run.Phase.water.Density.Type = 'Constant'
      run.Phase.water.Density.Value = 1.0
      run.Phase.water.Viscosity.Type = 'Constant'
      run.Phase.water.Viscosity.Value = 1.0
      run.Phase.water.Mobility.Type = 'Constant'
      run.Phase.water.Mobility.Value = 1.0
      run.PhaseSources.water.Type = 'Constant'

      # if geom_name is provided, it will set these keys:
      run.PhaseSources.water.GeomNames = geom_name
      run.PhaseSources.water.Geom.{geom_name}.Value = 0.0

5. ``variably_saturated()``: Sets the following keys:

.. code-block:: python3

      run.Solver = 'Richards'
      run.Solver.Nonlinear.MaxIter = 10
      run.Solver.Nonlinear.ResidualTol = 1e-5
      run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
      run.Solver.Nonlinear.EtaValue = 1e-5
      run.Solver.Nonlinear.UseJacobian = True
      run.Solver.Nonlinear.DerivativeEpsilon = 1e-2
      run.Solver.Linear.Preconditioner = 'PFMG'

6. ``fully_saturated()``: Sets the following keys:

.. code-block:: python3

      run.Solver = 'Impes'

7. ``homogeneous_subsurface(domain_name, perm=None, porosity=None, specific_storage=None, rel_perm=None, saturation=None, isotropic=False)``: Sets the following keys:

.. code-block:: python3

      # if perm is a value, it will set these keys:
      # appending domain_name to the list of Geom.Perm.Names
      run.Geom.Perm.Names = domain_name
      run.Geom.{domain_name}.Perm.Type = 'Constant'
      run.Geom.{domain_name}.Perm.Value = perm
      # if perm is a file name, it will set these keys:
      run.Geom.{domain_name}.Perm.FileName = perm
      # if the file name is a PFB file:
      run.Geom.{domain_name}.Perm.Type = 'PFBFile'
      # if the file name is a NetCDF file:
      run.Geom.{domain_name}.Perm.Type = 'NCFile'

      # if porosity is a value, it will set these keys:
      # appending domain_name to the list of Geom.Porosity.Names
      run.Geom.Porosity.GeomNames = domain_name
      run.Geom.{domain_name}.Porosity.Type = 'Constant'
      run.Geom.{domain_name}.Porosity.Value = porosity
      # if porosity is a file name, it will set these keys:
      run.Geom.{domain_name}.Porosity.FileName = porosity
      # if the file name is a PFB file:
      run.Geom.{domain_name}.Porosity.Type = 'PFBFile'
      # if the file name is a NetCDF file:
      run.Geom.{domain_name}.Porosity.Type = 'NCFile'

      # if specific_storage is provided, it will set these keys:
      # appending domain_name to the list of SpecificStorage.GeomNames
      run.SpecificStorage.GeomNames = domain_name
      run.SpecificStorage.Type = 'Constant'
      run.Geom.{domain_name}.SpecificStorage.Value = specific_storage

      # if rel_perm is provided, it must be a dictionary with the following key/value pairs:
      # {'Type': 'VanGenuchten', 'Alpha': 3.5, 'N': 2.0}
      # using this dictionary, it will set the following keys:
      # appending domain_name to the list of Phase.RelPerm.GeomNames
      run.Phase.RelPerm.GeomNames = domain_name
      # if Type = VanGenuchten, it will set the following keys:
      self.run.Geom.{domain_name}.RelPerm.Alpha = rel_perm['Alpha']
      self.run.Geom.{domain_name}.RelPerm.N = rel_perm['N']

      # if saturation is provided, it must be a dictionary with the following key/value pairs:
      # {'Type': 'VanGenuchten', 'Alpha': 3.5, 'N': 2.0, 'SRes': 0.1, 'SSat': 1.0}
      # Alpha and N are optional, and can default to the value of the corresponding properties in rel_perm
      # using this dictionary, it will set the following keys:
      # appending domain_name to the list of Phase.Saturation.GeomNames
      run.Phase.Saturation.GeomNames = domain_name
      # if Type = VanGenuchten, it will set the following keys:
      run.Geom.{domain_name}.Saturation.Alpha = saturation['Alpha']
      run.Geom.{domain_name}.Saturation.N = saturation['N']
      run.Geom.{domain_name}.Saturation.SRes = saturation['SRes']
      run.Geom.{domain_name}.Saturation.SSat = saturation['SSat']

      # if isotropic is True, it will set these keys:
      run.Perm.TensorType = 'TensorByGeom'
      # appending domain_name to the list of Geom.Perm.TensorByGeom.Names
      run.Geom.Perm.TensorByGeom.Names = domain_name
      run.Geom.{domain_name}.Perm.TensorValX = 1.0
      run.Geom.{domain_name}.Perm.TensorValY = 1.0
      run.Geom.{domain_name}.Perm.TensorValZ = 1.0

8. ``box_domain(box_input, domain_geom_name, bounds=None, patches=None)``: Sets the following keys:

.. code-block:: python3

      # append box_input to the GeomInput.Names
      run.GeomInput.Names = box_input
      run.GeomInput.{box_input}.InputType = 'Box'
      run.GeomInput.{box_input}.GeomName = domain_geom_name

      # if bounds is not provided, it will default to using the ComputationalGrid keys to define the boundaries:
      run.Geom.{domain_geom_name}.Lower.X = 0.0
      run.Geom.{domain_geom_name}.Lower.Y = 0.0
      run.Geom.{domain_geom_name}.Lower.Z = 0.0
      run.Geom.{domain_geom_name}.Upper.X = run.ComputationalGrid.DX * run.ComputationalGrid.NX
      run.Geom.{domain_geom_name}.Upper.Y = run.ComputationalGrid.DY * run.ComputationalGrid.NY
      run.Geom.{domain_geom_name}.Upper.Z = run.ComputationalGrid.DZ * run.ComputationalGrid.NZ

      # bounds should be provided as a list of coordinates in this order:
      # [lower_x, upper_x, lower_y, upper_y, lower_z, upper_z]
      run.Geom.{domain_geom_name}.Lower.X = bounds[0]
      run.Geom.{domain_geom_name}.Upper.X = bounds[1]
      run.Geom.{domain_geom_name}.Lower.Y = bounds[2]
      run.Geom.{domain_geom_name}.Upper.Y = bounds[3]
      run.Geom.{domain_geom_name}.Lower.Z = bounds[4]
      run.Geom.{domain_geom_name}.Upper.Z = bounds[5]

      # if patches is provided as a single string of the box domain patches (e.g., 'left right ...'), it will set this key:
      run.Geom.{domain_geom_name}.Patches = patches


9. ``slopes_mannings(self, domain_geom_name, slope_x=None, slope_y=None, mannings=None)``: Sets the following keys:

.. code-block:: python3

      # if slope_x is provided, it will set these keys:
      # appending domain_name to the list of TopoSlopesX.GeomNames
      run.TopoSlopesX.GeomNames = domain_geom_name
      # if slope_x is a number, it will set these keys:
      run.TopoSlopesX.Type = 'Constant'
      run.TopoSlopesX.Geom.{domain_geom_name}.Value = slope_x
      # if slope_x is a file name, it will set these keys:
      run.TopoSlopesX.FileName = slope_x
      # if the file name is a PFB file:
      run.TopoSlopesX.Type = 'PFBFile'
      # if the file name is a NetCDF file:
      run.TopoSlopesX.Type = 'NCFile'

      # if slope_y is provided, it will set these keys:
      # appending domain_name to the list of TopoSlopesY.GeomNames
      run.TopoSlopesY.GeomNames = domain_geom_name
      # if slope_y is a number, it will set these keys:
      run.TopoSlopesY.Type = 'Constant'
      run.TopoSlopesY.Geom.{domain_geom_name}.Value = slope_y
      # if slope_y is a file name, it will set these keys:
      run.TopoSlopesY.FileName = slope_y
      # if the file name is a PFB file:
      run.TopoSlopesY.Type = 'PFBFile'
      # if the file name is a NetCDF file:
      run.TopoSlopesY.Type = 'NCFile'

      # if mannings is provided, it will set these keys:
      # appending domain_name to the list of Mannings.GeomNames
      run.Mannings.GeomNames = domain_geom_name
      # if mannings is a number, it will set these keys:
      run.Mannings.Type = 'Constant'
      run.Mannings.Geom.{domain_geom_name}.Value = mannings
      # if mannings is a file name, it will set these keys:
      run.Mannings.FileName = mannings
      # if the file name is a PFB file:
      run.Mannings.Type = 'PFBFile'
      # if the file name is a NetCDF file:
      run.Mannings.Type = 'NCFile'

10. ``zero_flux(self, patches, cycle_name, interval_name)``: Sets the following keys:

.. code-block:: python3

      run.BCPressure.PatchNames += [patch]
      run.Patch[patch].BCPressure.Type = 'FluxConst'
      run.Patch[patch].BCPressure.Cycle = cycle_name
      run.Patch[patch].BCPressure[interval_name].Value = 0.0

11. ``ic_pressure(self, domain_geom_name, patch, pressure)``: Sets the following keys:

.. code-block:: python3

      run.ICPressure.GeomNames = domain_geom_name
      run.Geom.{domain_geom_name}.ICPressure.RefPatch = patch

      # if pressure is a PFB file, it will set the following keys:
      run.ICPressure.Type = 'PFBFile'
      run.Geom.domain.ICPressure.FileName = pressure

12. ``clm(met_file_name, top_patch, cycle_name, interval_name)``: Sets the following keys:

.. code-block:: python3

      # ensure time step is hourly
      run.TimeStep.Type = 'Constant'
      run.TimeStep.Value = 1.0
      # ensure OverlandFlow is the top boundary condition
      run.Patch.{top_patch}.BCPressure.Type = 'OverlandFlow'
      run.Patch.{top_patch}.BCPressure.Cycle = cycle_name
      run.Patch.{top_patch}.BCPressure.{interval_name}.Value = 0.0
      # set CLM keys
      run.Solver.LSM = 'CLM'
      run.Solver.CLM.CLMFileDir = "."
      run.Solver.PrintCLM = True
      run.Solver.CLM.Print1dOut = False
      run.Solver.BinaryOutDir = False
      run.Solver.CLM.DailyRST = True
      run.Solver.CLM.SingleFile = True
      run.Solver.CLM.CLMDumpInterval = 24
      run.Solver.CLM.WriteLogs = False
      run.Solver.CLM.WriteLastRST = True
      run.Solver.CLM.MetForcing = '1D'
      run.Solver.CLM.MetFileName = met_file_name
      run.Solver.CLM.MetFilePath = "."
      run.Solver.CLM.MetFileNT = 24
      run.Solver.CLM.IstepStart = 1.0
      run.Solver.CLM.EvapBeta = 'Linear'
      run.Solver.CLM.VegWaterStress = 'Saturation'
      run.Solver.CLM.ResSat = 0.1
      run.Solver.CLM.WiltingPoint = 0.12
      run.Solver.CLM.FieldCapacity = 0.98
      run.Solver.CLM.IrrigationType = 'none'

13. ``well(name, type, x, y, z_upper, z_lower, cycle_name, interval_name, action='Extraction', saturation=1.0, phase='water', hydrostatic_pressure=None, value=None)``: Sets the following keys:

.. code-block:: python3

      # append name to Wells.Names
      run.Wells.Names += [name]

      run.Wells.{name}.InputType = 'Vertical'
      run.Wells.{name}.Action = action
      run.Wells.{name}.Type = type
      run.Wells.{name}.X = x
      run.Wells.{name}.Y = y
      run.Wells.{name}.ZUpper = z_upper
      run.Wells.{name}.ZLower = z_lower
      run.Wells.{name}.Method = 'Standard'
      run.Wells.{name}.Cycle = cycle_name
      run.Wells.{name}.{interval_name}.Saturation.{phase}.Value = saturation

      # if type is set to 'Pressure', set Pressure.Value
      run.Wells.{name}.{interval_name}.Pressure.Value = hydrostatic_pressure

      # For extraction wells (run.Wells.{name}.Action = 'Extraction'), set these keys:
      # if type is set to 'Pressure' and value is provided, set Extraction.Pressure.Value
      run.Wells.{name}.{interval_name}.Extraction.Pressure.Value = value
      # if type is set to 'Flux' and value is provided, set Extraction.Flux.{phase}.Value
      run.Wells.{name}.{interval_name}.Extraction.Flux.{phase}.Value = value

      # For injection wells (run.Wells.{name}.Action = 'Injection'), set these keys:
      # if type is set to 'Pressure' and value is provided, set Injection.Pressure.Value
      run.Wells.{name}.{interval_name}.Injection.Pressure.Value = value
      # if type is set to 'Flux' and value is provided, set Injection.Flux.{phase}.Value
      run.Wells.{name}.{interval_name}.Injection.Flux.{phase}.Value = value

14. ``spinup_timing(self, initial_step, dump_interval)``:
Sets the following keys:

.. code-block:: python3

      run.TimingInfo.BaseUnit = 1
      run.TimingInfo.StartCount = 0
      run.TimingInfo.StartTime = 0.0
      run.TimingInfo.StopTime = 10000000
      run.TimingInfo.DumpInterval = dump_interval
      run.TimeStep.Type = 'Growth'
      run.TimeStep.InitialStep = initial_step
      run.TimeStep.GrowthFactor = 1.1
      run.TimeStep.MaxStep = 1000000
      run.TimeStep.MinStep = 0.1
