# This runs a test case with the Richards' solver 
# in hydrostatic equalibrium.  As such the solution
# should not change over time and should not
# take any solver iteratorions.

# set runname richards_hydrostatic_equalibrium

# Import the ParFlow TCL package
#
from parflow import Run
richards_hydrostatic_equalibrium = Run("richards_hydrostatic_equalibrium", __file__)

richards_hydrostatic_equalibrium.FileVersion = 4

richards_hydrostatic_equalibrium.Process.Topology.P = 1
richards_hydrostatic_equalibrium.Process.Topology.Q = 1
richards_hydrostatic_equalibrium.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
richards_hydrostatic_equalibrium.ComputationalGrid.Lower.X = 0.0
richards_hydrostatic_equalibrium.ComputationalGrid.Lower.Y = 0.0
richards_hydrostatic_equalibrium.ComputationalGrid.Lower.Z = 0.0

richards_hydrostatic_equalibrium.ComputationalGrid.DX = 1
richards_hydrostatic_equalibrium.ComputationalGrid.DY = 1
richards_hydrostatic_equalibrium.ComputationalGrid.DZ = 0.3

richards_hydrostatic_equalibrium.ComputationalGrid.NX = 15
richards_hydrostatic_equalibrium.ComputationalGrid.NY = 20
richards_hydrostatic_equalibrium.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
richards_hydrostatic_equalibrium.GeomInput.Names = 'domain_input background_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
richards_hydrostatic_equalibrium.GeomInput.domain_input.InputType = 'Box'
richards_hydrostatic_equalibrium.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
richards_hydrostatic_equalibrium.Geom.domain.Lower.X = 0.0
richards_hydrostatic_equalibrium.Geom.domain.Lower.Y = 0.0
richards_hydrostatic_equalibrium.Geom.domain.Lower.Z = 0.0

richards_hydrostatic_equalibrium.Geom.domain.Upper.X = 15.0
richards_hydrostatic_equalibrium.Geom.domain.Upper.Y = 19.0
richards_hydrostatic_equalibrium.Geom.domain.Upper.Z = 3.0

richards_hydrostatic_equalibrium.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
richards_hydrostatic_equalibrium.GeomInput.background_input.InputType = 'Box'
richards_hydrostatic_equalibrium.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
richards_hydrostatic_equalibrium.Geom.background.Lower.X = -99999999.0
richards_hydrostatic_equalibrium.Geom.background.Lower.Y = -99999999.0
richards_hydrostatic_equalibrium.Geom.background.Lower.Z = -99999999.0

richards_hydrostatic_equalibrium.Geom.background.Upper.X = 99999999.0
richards_hydrostatic_equalibrium.Geom.background.Upper.Y = 99999999.0
richards_hydrostatic_equalibrium.Geom.background.Upper.Z = 99999999.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.Geom.Perm.Names = 'background'
richards_hydrostatic_equalibrium.Geom.background.Perm.Type = 'Constant'
richards_hydrostatic_equalibrium.Geom.background.Perm.Value = 4.0

richards_hydrostatic_equalibrium.Perm.TensorType = 'TensorByGeom'

richards_hydrostatic_equalibrium.Geom.Perm.TensorByGeom.Names = 'background'

richards_hydrostatic_equalibrium.Geom.background.Perm.TensorValX = 1.0
richards_hydrostatic_equalibrium.Geom.background.Perm.TensorValY = 1.0
richards_hydrostatic_equalibrium.Geom.background.Perm.TensorValZ = 1.0

# kf-Zone

# is the kfzone defined or used?
# richards_hydrostatic_equalibrium.Geom.kfzone.Perm.Type = 'Constant'
# richards_hydrostatic_equalibrium.Geom.kfzone.Perm.Value = 40


#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.SpecificStorage.Type = 'Constant'
richards_hydrostatic_equalibrium.SpecificStorage.GeomNames = 'background'
richards_hydrostatic_equalibrium.Geom.background.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.Phase.Names = 'water'

richards_hydrostatic_equalibrium.Phase.water.Density.Type = 'Constant'
richards_hydrostatic_equalibrium.Phase.water.Density.Value = 1.0

richards_hydrostatic_equalibrium.Phase.water.Viscosity.Type = 'Constant'
richards_hydrostatic_equalibrium.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.TimingInfo.BaseUnit = 0.5
richards_hydrostatic_equalibrium.TimingInfo.StartCount = 0
richards_hydrostatic_equalibrium.TimingInfo.StartTime = 0.0
richards_hydrostatic_equalibrium.TimingInfo.StopTime = 1.5
richards_hydrostatic_equalibrium.TimingInfo.DumpInterval = -1
richards_hydrostatic_equalibrium.TimeStep.Type = 'Constant'
richards_hydrostatic_equalibrium.TimeStep.Value = 0.5

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.Geom.Porosity.GeomNames = 'background'

richards_hydrostatic_equalibrium.Geom.background.Porosity.Type = 'Constant'
richards_hydrostatic_equalibrium.Geom.background.Porosity.Value = 0.15

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.Phase.RelPerm.Type = 'VanGenuchten'
richards_hydrostatic_equalibrium.Phase.RelPerm.GeomNames = 'background'
richards_hydrostatic_equalibrium.Geom.background.RelPerm.Alpha = 2.0
richards_hydrostatic_equalibrium.Geom.background.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_hydrostatic_equalibrium.Phase.Saturation.Type = 'VanGenuchten'
richards_hydrostatic_equalibrium.Phase.Saturation.GeomNames = 'background'
richards_hydrostatic_equalibrium.Geom.background.Saturation.Alpha = 2.0
richards_hydrostatic_equalibrium.Geom.background.Saturation.N = 2.0
richards_hydrostatic_equalibrium.Geom.background.Saturation.SRes = 0.0
richards_hydrostatic_equalibrium.Geom.background.Saturation.SSat = 1.0


#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Cycle.Names = 'constant'
richards_hydrostatic_equalibrium.Cycle.constant.Names = 'alltime'
richards_hydrostatic_equalibrium.Cycle.constant.alltime.Length = 1
richards_hydrostatic_equalibrium.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.BCPressure.PatchNames = 'left right front back bottom top'

richards_hydrostatic_equalibrium.Patch.front.BCPressure.Type = 'DirEquilRefPatch'
richards_hydrostatic_equalibrium.Patch.front.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.front.BCPressure.RefGeom = 'domain'
richards_hydrostatic_equalibrium.Patch.front.BCPressure.RefPatch = 'bottom'
richards_hydrostatic_equalibrium.Patch.front.BCPressure.alltime.Value = 1.0

richards_hydrostatic_equalibrium.Patch.back.BCPressure.Type = 'DirEquilRefPatch'
richards_hydrostatic_equalibrium.Patch.back.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.back.BCPressure.RefGeom = 'domain'
richards_hydrostatic_equalibrium.Patch.back.BCPressure.RefPatch = 'bottom'
richards_hydrostatic_equalibrium.Patch.back.BCPressure.alltime.Value = 1.0

richards_hydrostatic_equalibrium.Patch.left.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equalibrium.Patch.left.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.left.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equalibrium.Patch.right.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equalibrium.Patch.right.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.right.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equalibrium.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equalibrium.Patch.bottom.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.bottom.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equalibrium.Patch.top.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equalibrium.Patch.top.BCPressure.Cycle = 'constant'
richards_hydrostatic_equalibrium.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_hydrostatic_equalibrium.TopoSlopesX.Type = 'Constant'
richards_hydrostatic_equalibrium.TopoSlopesX.GeomNames = 'domain'

richards_hydrostatic_equalibrium.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_hydrostatic_equalibrium.TopoSlopesY.Type = 'Constant'
richards_hydrostatic_equalibrium.TopoSlopesY.GeomNames = 'domain'

richards_hydrostatic_equalibrium.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

richards_hydrostatic_equalibrium.Mannings.Type = 'Constant'
richards_hydrostatic_equalibrium.Mannings.GeomNames = 'domain'
richards_hydrostatic_equalibrium.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_hydrostatic_equalibrium.ICPressure.Type = 'HydroStaticPatch'
richards_hydrostatic_equalibrium.ICPressure.GeomNames = 'domain'
richards_hydrostatic_equalibrium.Geom.domain.ICPressure.Value = 1.0
richards_hydrostatic_equalibrium.Geom.domain.ICPressure.RefGeom = 'domain'
richards_hydrostatic_equalibrium.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.PhaseSources.water.Type = 'Constant'
richards_hydrostatic_equalibrium.PhaseSources.water.GeomNames = 'background'
richards_hydrostatic_equalibrium.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
richards_hydrostatic_equalibrium.Solver = 'Richards'
richards_hydrostatic_equalibrium.Solver.MaxIter = 50000

richards_hydrostatic_equalibrium.Solver.Nonlinear.MaxIter = 100
richards_hydrostatic_equalibrium.Solver.Nonlinear.ResidualTol = 1e-9
richards_hydrostatic_equalibrium.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_hydrostatic_equalibrium.Solver.Nonlinear.EtaValue = 1e-2
richards_hydrostatic_equalibrium.Solver.Nonlinear.UseJacobian = True
richards_hydrostatic_equalibrium.Solver.Nonlinear.DerivativeEpsilon = 1e-9

richards_hydrostatic_equalibrium.Solver.Linear.KrylovDimension = 10

richards_hydrostatic_equalibrium.Solver.Linear.Preconditioner = 'MGSemi'
richards_hydrostatic_equalibrium.Solver.Linear.Preconditioner.MGSemi.MaxIter = 10
richards_hydrostatic_equalibrium.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

richards_hydrostatic_equalibrium.Solver.WriteSiloSubsurfData = True
richards_hydrostatic_equalibrium.Solver.WriteSiloPressure = True
richards_hydrostatic_equalibrium.Solver.WriteSiloSaturation = True
richards_hydrostatic_equalibrium.Solver.WriteSiloConcentration = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_hydrostatic_equalibrium.run()
