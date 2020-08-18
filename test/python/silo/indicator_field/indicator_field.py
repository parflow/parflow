#
# Problem to test use of indicator field.
#

#
# Import the ParFlow TCL package
#
from parflow import Run
indicator_field = Run("indicator_field", __file__)

indicator_field.FileVersion = 4

name = "indicator_field"

#
# Control use of indicator field: 
# 0 = use domain
# 1 = use indicator field
#
useIndicatorField = 1

indicator_field.Process.Topology.P = 1
indicator_field.Process.Topology.Q = 1
indicator_field.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
indicator_field.ComputationalGrid.Lower.X = 0.0
indicator_field.ComputationalGrid.Lower.Y = 0.0
indicator_field.ComputationalGrid.Lower.Z = 0.0

indicator_field.ComputationalGrid.NX = 12
indicator_field.ComputationalGrid.NY = 12
indicator_field.ComputationalGrid.NZ = 12

UpperX = 440
UpperY = 120
UpperZ = 220

LowerX = indicator_field.ComputationalGrid.Lower.X
LowerY = indicator_field.ComputationalGrid.Lower.Y
LowerZ = indicator_field.ComputationalGrid.Lower.Z

NX = indicator_field.ComputationalGrid.NX
NY = indicator_field.ComputationalGrid.NY
NZ = indicator_field.ComputationalGrid.NZ

indicator_field.ComputationalGrid.DX = (UpperX - LowerX)/indicator_field.ComputationalGrid.NX
indicator_field.ComputationalGrid.DY = (UpperY - LowerY)/indicator_field.ComputationalGrid.NY
indicator_field.ComputationalGrid.DZ = (UpperZ - LowerZ)/indicator_field.ComputationalGrid.NZ

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

indicator_field.GeomInput.Names = 'solid_input indicator_input'

indicator_field.GeomInput.solid_input.InputType = 'SolidFile'
indicator_field.GeomInput.solid_input.GeomNames = 'domain'
indicator_field.GeomInput.solid_input.FileName = 'small_domain.pfsol'

indicator_field.Geom.domain.Patches = 'infiltration z_upper x_lower y_lower x_upper y_upper z_lower'

indicator_field.GeomInput.indicator_input.InputType = 'IndicatorField'
indicator_field.GeomInput.indicator_input.GeomNames = 'indicator'
indicator_field.Geom.indicator_input.FileName = 'small_domain_indicator_field.pfb'

indicator_field.GeomInput.indicator.Value = 1

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
indicator_field.Geom.Perm.Names = 'domain'

indicator_field.Geom.domain.Perm.Type = 'Constant'
indicator_field.Geom.domain.Perm.Value = 1.0

indicator_field.Perm.TensorType = 'TensorByGeom'

indicator_field.Geom.Perm.TensorByGeom.Names = 'domain'

indicator_field.Geom.domain.Perm.TensorValX = 1.0
indicator_field.Geom.domain.Perm.TensorValY = 1.0
indicator_field.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

indicator_field.SpecificStorage.Type = 'Constant'
indicator_field.SpecificStorage.GeomNames = 'domain'
indicator_field.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

indicator_field.Phase.Names = 'water'

indicator_field.Phase.water.Density.Type = 'Constant'
indicator_field.Phase.water.Density.Value = 1.0

indicator_field.Phase.water.Viscosity.Type = 'Constant'
indicator_field.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

indicator_field.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

indicator_field.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

indicator_field.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

indicator_field.TimingInfo.BaseUnit = 1.0
indicator_field.TimingInfo.StartCount = 0
indicator_field.TimingInfo.StartTime = 0.0
indicator_field.TimingInfo.StopTime = 30.0*1
indicator_field.TimingInfo.DumpInterval = 0
indicator_field.TimeStep.Type = 'Constant'
indicator_field.TimeStep.Value = 10.0
indicator_field.TimingInfo.DumpAtEnd = True

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

indicator_field.Geom.Porosity.GeomNames = 'domain'

indicator_field.Geom.domain.Porosity.Type = 'Constant'
indicator_field.Geom.domain.Porosity.Value = 0.3680

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

indicator_field.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

indicator_field.Phase.RelPerm.Type = 'VanGenuchten'

if useIndicatorField == 1:
  indicator_field.Phase.RelPerm.GeomNames = 'indicator'
  indicator_field.Geom.indicator.RelPerm.Alpha = 3.34
  indicator_field.Geom.indicator.RelPerm.N = 1.982
else:
  indicator_field.Phase.RelPerm.GeomNames = 'domain'
  indicator_field.Geom.domain.RelPerm.Alpha = 3.34
  indicator_field.Geom.domain.RelPerm.N = 1.982


#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

indicator_field.Phase.Saturation.Type = 'VanGenuchten'
indicator_field.Phase.Saturation.GeomNames = 'domain'

indicator_field.Geom.domain.Saturation.Alpha = 3.34
indicator_field.Geom.domain.Saturation.N = 1.982
indicator_field.Geom.domain.Saturation.SRes = 0.2771
indicator_field.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
indicator_field.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
indicator_field.Cycle.Names = 'constant onoff'
indicator_field.Cycle.constant.Names = 'alltime'
indicator_field.Cycle.constant.alltime.Length = 1
indicator_field.Cycle.constant.Repeat = -1

indicator_field.Cycle.onoff.Names = 'on off'
indicator_field.Cycle.onoff.on.Length = 10
indicator_field.Cycle.onoff.off.Length = 90
indicator_field.Cycle.onoff.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
indicator_field.BCPressure.PatchNames = indicator_field.Geom.domain.Patches

indicator_field.Patch.infiltration.BCPressure.Type = 'FluxConst'
indicator_field.Patch.infiltration.BCPressure.Cycle = 'constant'
indicator_field.Patch.infiltration.BCPressure.alltime.Value = -0.10
indicator_field.Patch.infiltration.BCPressure.off.Value = 0.0

indicator_field.Patch.x_lower.BCPressure.Type = 'FluxConst'
indicator_field.Patch.x_lower.BCPressure.Cycle = 'constant'
indicator_field.Patch.x_lower.BCPressure.alltime.Value = 0.0

indicator_field.Patch.y_lower.BCPressure.Type = 'FluxConst'
indicator_field.Patch.y_lower.BCPressure.Cycle = 'constant'
indicator_field.Patch.y_lower.BCPressure.alltime.Value = 0.0

indicator_field.Patch.z_lower.BCPressure.Type = 'FluxConst'
indicator_field.Patch.z_lower.BCPressure.Cycle = 'constant'
indicator_field.Patch.z_lower.BCPressure.alltime.Value = 0.0

indicator_field.Patch.x_upper.BCPressure.Type = 'FluxConst'
indicator_field.Patch.x_upper.BCPressure.Cycle = 'constant'
indicator_field.Patch.x_upper.BCPressure.alltime.Value = 0.0

indicator_field.Patch.y_upper.BCPressure.Type = 'FluxConst'
indicator_field.Patch.y_upper.BCPressure.Cycle = 'constant'
indicator_field.Patch.y_upper.BCPressure.alltime.Value = 0.0

indicator_field.Patch.z_upper.BCPressure.Type = 'FluxConst'
indicator_field.Patch.z_upper.BCPressure.Cycle = 'constant'
indicator_field.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

indicator_field.TopoSlopesX.Type = 'Constant'
indicator_field.TopoSlopesX.GeomNames = 'domain'

indicator_field.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

indicator_field.TopoSlopesY.Type = 'Constant'
indicator_field.TopoSlopesY.GeomNames = 'domain'

indicator_field.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

indicator_field.Mannings.Type = 'Constant'
indicator_field.Mannings.GeomNames = 'domain'
indicator_field.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

indicator_field.ICPressure.Type = 'HydroStaticPatch'
indicator_field.ICPressure.GeomNames = 'domain'

indicator_field.Geom.domain.ICPressure.Value = 1.0
indicator_field.Geom.domain.ICPressure.RefPatch = 'z_lower'
indicator_field.Geom.domain.ICPressure.RefGeom = 'domain'

indicator_field.Geom.infiltration.ICPressure.Value = 10.0
indicator_field.Geom.infiltration.ICPressure.RefPatch = 'infiltration'
indicator_field.Geom.infiltration.ICPressure.RefGeom = 'domain'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

indicator_field.PhaseSources.water.Type = 'Constant'
indicator_field.PhaseSources.water.GeomNames = 'domain'
indicator_field.PhaseSources.water.Geom.domain.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

indicator_field.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
indicator_field.Solver = 'Richards'
indicator_field.Solver.MaxIter = 1

indicator_field.Solver.Nonlinear.MaxIter = 15
indicator_field.Solver.Nonlinear.ResidualTol = 1e-9
indicator_field.Solver.Nonlinear.StepTol = 1e-9
indicator_field.Solver.Nonlinear.EtaValue = 1e-5
indicator_field.Solver.Nonlinear.UseJacobian = True
indicator_field.Solver.Nonlinear.DerivativeEpsilon = 1e-7

indicator_field.Solver.Linear.KrylovDimension = 25
indicator_field.Solver.Linear.MaxRestarts = 2

indicator_field.Solver.Linear.Preconditioner = 'MGSemi'
indicator_field.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
indicator_field.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

indicator_field.Solver.PrintSubsurfData = False
indicator_field.Solver.PrintPressure = False
indicator_field.Solver.PrintSaturation = False
indicator_field.Solver.PrintConcentration = False

indicator_field.Solver.WriteSiloSubsurfData = True
indicator_field.Solver.WriteSiloPressure = True
indicator_field.Solver.WriteSiloSaturation = True
indicator_field.Solver.WriteSiloConcentration = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

indicator_field.run()
