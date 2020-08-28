# This runs a test case with the Richards' solver 
# in hydrostatic equilibrium.  As such the solution
# should not change over time and should not
# take any solver iterations.

from parflow import Run
richards_hydrostatic_equilibrium = Run("richards_hydrostatic_equilibrium", __file__)

richards_hydrostatic_equilibrium.FileVersion = 4

richards_hydrostatic_equilibrium.Process.Topology.P = 1
richards_hydrostatic_equilibrium.Process.Topology.Q = 1
richards_hydrostatic_equilibrium.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
richards_hydrostatic_equilibrium.ComputationalGrid.Lower.X = 0.0
richards_hydrostatic_equilibrium.ComputationalGrid.Lower.Y = 0.0
richards_hydrostatic_equilibrium.ComputationalGrid.Lower.Z = 0.0

richards_hydrostatic_equilibrium.ComputationalGrid.DX = 1
richards_hydrostatic_equilibrium.ComputationalGrid.DY = 1
richards_hydrostatic_equilibrium.ComputationalGrid.DZ = 0.3

richards_hydrostatic_equilibrium.ComputationalGrid.NX = 15
richards_hydrostatic_equilibrium.ComputationalGrid.NY = 20
richards_hydrostatic_equilibrium.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
richards_hydrostatic_equilibrium.GeomInput.Names = 'domain_input background_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
richards_hydrostatic_equilibrium.GeomInput.domain_input.InputType = 'Box'
richards_hydrostatic_equilibrium.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
richards_hydrostatic_equilibrium.Geom.domain.Lower.X = 0.0
richards_hydrostatic_equilibrium.Geom.domain.Lower.Y = 0.0
richards_hydrostatic_equilibrium.Geom.domain.Lower.Z = 0.0

richards_hydrostatic_equilibrium.Geom.domain.Upper.X = 15.0
richards_hydrostatic_equilibrium.Geom.domain.Upper.Y = 19.0
richards_hydrostatic_equilibrium.Geom.domain.Upper.Z = 3.0

richards_hydrostatic_equilibrium.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
richards_hydrostatic_equilibrium.GeomInput.background_input.InputType = 'Box'
richards_hydrostatic_equilibrium.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
richards_hydrostatic_equilibrium.Geom.background.Lower.X = -99999999.0
richards_hydrostatic_equilibrium.Geom.background.Lower.Y = -99999999.0
richards_hydrostatic_equilibrium.Geom.background.Lower.Z = -99999999.0

richards_hydrostatic_equilibrium.Geom.background.Upper.X = 99999999.0
richards_hydrostatic_equilibrium.Geom.background.Upper.Y = 99999999.0
richards_hydrostatic_equilibrium.Geom.background.Upper.Z = 99999999.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.Geom.Perm.Names = 'background'
richards_hydrostatic_equilibrium.Geom.background.Perm.Type = 'Constant'
richards_hydrostatic_equilibrium.Geom.background.Perm.Value = 4.0

richards_hydrostatic_equilibrium.Perm.TensorType = 'TensorByGeom'

richards_hydrostatic_equilibrium.Geom.Perm.TensorByGeom.Names = 'background'

richards_hydrostatic_equilibrium.Geom.background.Perm.TensorValX = 1.0
richards_hydrostatic_equilibrium.Geom.background.Perm.TensorValY = 1.0
richards_hydrostatic_equilibrium.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.SpecificStorage.Type = 'Constant'
richards_hydrostatic_equilibrium.SpecificStorage.GeomNames = 'background'
richards_hydrostatic_equilibrium.Geom.background.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.Phase.Names = 'water'

richards_hydrostatic_equilibrium.Phase.water.Density.Type = 'Constant'
richards_hydrostatic_equilibrium.Phase.water.Density.Value = 1.0

richards_hydrostatic_equilibrium.Phase.water.Viscosity.Type = 'Constant'
richards_hydrostatic_equilibrium.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.TimingInfo.BaseUnit = 0.5
richards_hydrostatic_equilibrium.TimingInfo.StartCount = 0
richards_hydrostatic_equilibrium.TimingInfo.StartTime = 0.0
richards_hydrostatic_equilibrium.TimingInfo.StopTime = 1.5
richards_hydrostatic_equilibrium.TimingInfo.DumpInterval = -1
richards_hydrostatic_equilibrium.TimeStep.Type = 'Constant'
richards_hydrostatic_equilibrium.TimeStep.Value = 0.5

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.Geom.Porosity.GeomNames = 'background'

richards_hydrostatic_equilibrium.Geom.background.Porosity.Type = 'Constant'
richards_hydrostatic_equilibrium.Geom.background.Porosity.Value = 0.15

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.Phase.RelPerm.Type = 'VanGenuchten'
richards_hydrostatic_equilibrium.Phase.RelPerm.GeomNames = 'background'
richards_hydrostatic_equilibrium.Geom.background.RelPerm.Alpha = 2.0
richards_hydrostatic_equilibrium.Geom.background.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_hydrostatic_equilibrium.Phase.Saturation.Type = 'VanGenuchten'
richards_hydrostatic_equilibrium.Phase.Saturation.GeomNames = 'background'
richards_hydrostatic_equilibrium.Geom.background.Saturation.Alpha = 2.0
richards_hydrostatic_equilibrium.Geom.background.Saturation.N = 2.0
richards_hydrostatic_equilibrium.Geom.background.Saturation.SRes = 0.0
richards_hydrostatic_equilibrium.Geom.background.Saturation.SSat = 1.0


#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Cycle.Names = 'constant'
richards_hydrostatic_equilibrium.Cycle.constant.Names = 'alltime'
richards_hydrostatic_equilibrium.Cycle.constant.alltime.Length = 1
richards_hydrostatic_equilibrium.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.BCPressure.PatchNames = 'left right front back bottom top'

richards_hydrostatic_equilibrium.Patch.front.BCPressure.Type = 'DirEquilRefPatch'
richards_hydrostatic_equilibrium.Patch.front.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.front.BCPressure.RefGeom = 'domain'
richards_hydrostatic_equilibrium.Patch.front.BCPressure.RefPatch = 'bottom'
richards_hydrostatic_equilibrium.Patch.front.BCPressure.alltime.Value = 1.0

richards_hydrostatic_equilibrium.Patch.back.BCPressure.Type = 'DirEquilRefPatch'
richards_hydrostatic_equilibrium.Patch.back.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.back.BCPressure.RefGeom = 'domain'
richards_hydrostatic_equilibrium.Patch.back.BCPressure.RefPatch = 'bottom'
richards_hydrostatic_equilibrium.Patch.back.BCPressure.alltime.Value = 1.0

richards_hydrostatic_equilibrium.Patch.left.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equilibrium.Patch.left.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.left.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equilibrium.Patch.right.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equilibrium.Patch.right.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.right.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equilibrium.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equilibrium.Patch.bottom.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.bottom.BCPressure.alltime.Value = 0.0

richards_hydrostatic_equilibrium.Patch.top.BCPressure.Type = 'FluxConst'
richards_hydrostatic_equilibrium.Patch.top.BCPressure.Cycle = 'constant'
richards_hydrostatic_equilibrium.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_hydrostatic_equilibrium.TopoSlopesX.Type = 'Constant'
richards_hydrostatic_equilibrium.TopoSlopesX.GeomNames = 'domain'

richards_hydrostatic_equilibrium.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_hydrostatic_equilibrium.TopoSlopesY.Type = 'Constant'
richards_hydrostatic_equilibrium.TopoSlopesY.GeomNames = 'domain'

richards_hydrostatic_equilibrium.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

richards_hydrostatic_equilibrium.Mannings.Type = 'Constant'
richards_hydrostatic_equilibrium.Mannings.GeomNames = 'domain'
richards_hydrostatic_equilibrium.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_hydrostatic_equilibrium.ICPressure.Type = 'HydroStaticPatch'
richards_hydrostatic_equilibrium.ICPressure.GeomNames = 'domain'
richards_hydrostatic_equilibrium.Geom.domain.ICPressure.Value = 1.0
richards_hydrostatic_equilibrium.Geom.domain.ICPressure.RefGeom = 'domain'
richards_hydrostatic_equilibrium.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.PhaseSources.water.Type = 'Constant'
richards_hydrostatic_equilibrium.PhaseSources.water.GeomNames = 'background'
richards_hydrostatic_equilibrium.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
richards_hydrostatic_equilibrium.Solver = 'Richards'
richards_hydrostatic_equilibrium.Solver.MaxIter = 50000

richards_hydrostatic_equilibrium.Solver.Nonlinear.MaxIter = 100
richards_hydrostatic_equilibrium.Solver.Nonlinear.ResidualTol = 1e-9
richards_hydrostatic_equilibrium.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_hydrostatic_equilibrium.Solver.Nonlinear.EtaValue = 1e-2
richards_hydrostatic_equilibrium.Solver.Nonlinear.UseJacobian = True
richards_hydrostatic_equilibrium.Solver.Nonlinear.DerivativeEpsilon = 1e-9

richards_hydrostatic_equilibrium.Solver.Linear.KrylovDimension = 10

richards_hydrostatic_equilibrium.Solver.Linear.Preconditioner = 'MGSemi'
richards_hydrostatic_equilibrium.Solver.Linear.Preconditioner.MGSemi.MaxIter = 10
richards_hydrostatic_equilibrium.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

richards_hydrostatic_equilibrium.Solver.WriteSiloSubsurfData = True
richards_hydrostatic_equilibrium.Solver.WriteSiloPressure = True
richards_hydrostatic_equilibrium.Solver.WriteSiloSaturation = True
richards_hydrostatic_equilibrium.Solver.WriteSiloConcentration = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_hydrostatic_equilibrium.run()
