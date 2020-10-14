#---------------------------------------------------------
# This runs a test case with the Richards' solver
# in hydrostatic equilibrium.  As such the solution
# should not change over time and should not
# take any solver iterations.
#---------------------------------------------------------

from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path

rich = Run("richards_hydrostatic_equilibrium", __file__)

#---------------------------------------------------------

rich.FileVersion = 4

rich.Process.Topology.P = 1
rich.Process.Topology.Q = 1
rich.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

rich.ComputationalGrid.Lower.X = 0.0
rich.ComputationalGrid.Lower.Y = 0.0
rich.ComputationalGrid.Lower.Z = 0.0

rich.ComputationalGrid.DX = 1
rich.ComputationalGrid.DY = 1
rich.ComputationalGrid.DZ = 0.3

rich.ComputationalGrid.NX = 15
rich.ComputationalGrid.NY = 20
rich.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

rich.GeomInput.Names = 'domain_input background_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

rich.GeomInput.domain_input.InputType = 'Box'
rich.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

rich.Geom.domain.Lower.X = 0.0
rich.Geom.domain.Lower.Y = 0.0
rich.Geom.domain.Lower.Z = 0.0

rich.Geom.domain.Upper.X = 15.0
rich.Geom.domain.Upper.Y = 19.0
rich.Geom.domain.Upper.Z = 3.0

rich.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------

rich.GeomInput.background_input.InputType = 'Box'
rich.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------

rich.Geom.background.Lower.X = -99999999.0
rich.Geom.background.Lower.Y = -99999999.0
rich.Geom.background.Lower.Z = -99999999.0

rich.Geom.background.Upper.X = 99999999.0
rich.Geom.background.Upper.Y = 99999999.0
rich.Geom.background.Upper.Z = 99999999.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

rich.Geom.Perm.Names = 'background'
rich.Geom.background.Perm.Type = 'Constant'
rich.Geom.background.Perm.Value = 4.0

rich.Perm.TensorType = 'TensorByGeom'

rich.Geom.Perm.TensorByGeom.Names = 'background'

rich.Geom.background.Perm.TensorValX = 1.0
rich.Geom.background.Perm.TensorValY = 1.0
rich.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

rich.SpecificStorage.Type = 'Constant'
rich.SpecificStorage.GeomNames = 'background'
rich.Geom.background.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

rich.Phase.Names = 'water'

rich.Phase.water.Density.Type = 'Constant'
rich.Phase.water.Density.Value = 1.0

rich.Phase.water.Viscosity.Type = 'Constant'
rich.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

rich.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

rich.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

rich.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

rich.TimingInfo.BaseUnit = 0.5
rich.TimingInfo.StartCount = 0
rich.TimingInfo.StartTime = 0.0
rich.TimingInfo.StopTime = 1.5
rich.TimingInfo.DumpInterval = -1
rich.TimeStep.Type = 'Constant'
rich.TimeStep.Value = 0.5

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

rich.Geom.Porosity.GeomNames = 'background'
rich.Geom.background.Porosity.Type = 'Constant'
rich.Geom.background.Porosity.Value = 0.15

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

rich.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

rich.Phase.RelPerm.Type = 'VanGenuchten'
rich.Phase.RelPerm.GeomNames = 'background'
rich.Geom.background.RelPerm.Alpha = 2.0
rich.Geom.background.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

rich.Phase.Saturation.Type = 'VanGenuchten'
rich.Phase.Saturation.GeomNames = 'background'
rich.Geom.background.Saturation.Alpha = 2.0
rich.Geom.background.Saturation.N = 2.0
rich.Geom.background.Saturation.SRes = 0.0
rich.Geom.background.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

rich.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

rich.Cycle.Names = 'constant'
rich.Cycle.constant.Names = 'alltime'
rich.Cycle.constant.alltime.Length = 1
rich.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

rich.BCPressure.PatchNames = 'left right front back bottom top'

rich.Patch.front.BCPressure.Type = 'DirEquilRefPatch'
rich.Patch.front.BCPressure.Cycle = 'constant'
rich.Patch.front.BCPressure.RefGeom = 'domain'
rich.Patch.front.BCPressure.RefPatch = 'bottom'
rich.Patch.front.BCPressure.alltime.Value = 1.0

rich.Patch.back.BCPressure.Type = 'DirEquilRefPatch'
rich.Patch.back.BCPressure.Cycle = 'constant'
rich.Patch.back.BCPressure.RefGeom = 'domain'
rich.Patch.back.BCPressure.RefPatch = 'bottom'
rich.Patch.back.BCPressure.alltime.Value = 1.0

rich.Patch.left.BCPressure.Type = 'FluxConst'
rich.Patch.left.BCPressure.Cycle = 'constant'
rich.Patch.left.BCPressure.alltime.Value = 0.0

rich.Patch.right.BCPressure.Type = 'FluxConst'
rich.Patch.right.BCPressure.Cycle = 'constant'
rich.Patch.right.BCPressure.alltime.Value = 0.0

rich.Patch.bottom.BCPressure.Type = 'FluxConst'
rich.Patch.bottom.BCPressure.Cycle = 'constant'
rich.Patch.bottom.BCPressure.alltime.Value = 0.0

rich.Patch.top.BCPressure.Type = 'FluxConst'
rich.Patch.top.BCPressure.Cycle = 'constant'
rich.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

rich.TopoSlopesX.Type = 'Constant'
rich.TopoSlopesX.GeomNames = 'domain'
rich.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

rich.TopoSlopesY.Type = 'Constant'
rich.TopoSlopesY.GeomNames = 'domain'
rich.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

rich.Mannings.Type = 'Constant'
rich.Mannings.GeomNames = 'domain'
rich.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

rich.ICPressure.Type = 'HydroStaticPatch'
rich.ICPressure.GeomNames = 'domain'
rich.Geom.domain.ICPressure.Value = 1.0
rich.Geom.domain.ICPressure.RefGeom = 'domain'
rich.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

rich.PhaseSources.water.Type = 'Constant'
rich.PhaseSources.water.GeomNames = 'background'
rich.PhaseSources.water.Geom.background.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

rich.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

rich.Solver = 'Richards'
rich.Solver.MaxIter = 50000

rich.Solver.Nonlinear.MaxIter = 100
rich.Solver.Nonlinear.ResidualTol = 1e-9
rich.Solver.Nonlinear.EtaChoice = 'EtaConstant'
rich.Solver.Nonlinear.EtaValue = 1e-2
rich.Solver.Nonlinear.UseJacobian = True
rich.Solver.Nonlinear.DerivativeEpsilon = 1e-9

rich.Solver.Linear.KrylovDimension = 10

rich.Solver.Linear.Preconditioner = 'MGSemi'
rich.Solver.Linear.Preconditioner.MGSemi.MaxIter = 10
rich.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

dir_name = get_absolute_path('test_output/rich_heq')
mkdir(dir_name)
rich.run(working_directory=dir_name)
