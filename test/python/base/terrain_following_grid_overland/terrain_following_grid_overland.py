#  This runs a simple 2D, terrain following problem with a 5% slope
#  R. Maxwell 1-11

from parflow import Run
terrain_following_grid_overland = Run("terrain_following_grid_overland", __file__)

terrain_following_grid_overland.FileVersion = 4

terrain_following_grid_overland.Process.Topology.P = 1
terrain_following_grid_overland.Process.Topology.Q = 1
terrain_following_grid_overland.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
terrain_following_grid_overland.ComputationalGrid.Lower.X = 0.0
terrain_following_grid_overland.ComputationalGrid.Lower.Y = 0.0
terrain_following_grid_overland.ComputationalGrid.Lower.Z = 0.0

terrain_following_grid_overland.ComputationalGrid.NX = 20
terrain_following_grid_overland.ComputationalGrid.NY = 1
terrain_following_grid_overland.ComputationalGrid.NZ = 30

terrain_following_grid_overland.ComputationalGrid.DX = 5.0
terrain_following_grid_overland.ComputationalGrid.DY = 1.0
terrain_following_grid_overland.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
terrain_following_grid_overland.GeomInput.Names = 'boxinput'

terrain_following_grid_overland.GeomInput.boxinput.InputType = 'Box'
terrain_following_grid_overland.GeomInput.boxinput.GeomName = 'domain'


#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
terrain_following_grid_overland.Geom.domain.Lower.X = 0.0
terrain_following_grid_overland.Geom.domain.Lower.Y = 0.0
terrain_following_grid_overland.Geom.domain.Lower.Z = 0.0

terrain_following_grid_overland.Geom.domain.Upper.X = 100.0
terrain_following_grid_overland.Geom.domain.Upper.Y = 1.0
terrain_following_grid_overland.Geom.domain.Upper.Z = 1.5

terrain_following_grid_overland.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Geom.Perm.Names = 'domain'



terrain_following_grid_overland.Geom.domain.Perm.Type = 'Constant'
terrain_following_grid_overland.Geom.domain.Perm.Value = 10.

terrain_following_grid_overland.Perm.TensorType = 'TensorByGeom'

terrain_following_grid_overland.Geom.Perm.TensorByGeom.Names = 'domain'

terrain_following_grid_overland.Geom.domain.Perm.TensorValX = 1.0
terrain_following_grid_overland.Geom.domain.Perm.TensorValY = 1.0
terrain_following_grid_overland.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

terrain_following_grid_overland.SpecificStorage.Type = 'Constant'
terrain_following_grid_overland.SpecificStorage.GeomNames = 'domain'
terrain_following_grid_overland.Geom.domain.SpecificStorage.Value = 1.0e-5

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Phase.Names = 'water'

terrain_following_grid_overland.Phase.water.Density.Type = 'Constant'
terrain_following_grid_overland.Phase.water.Density.Value = 1.0

terrain_following_grid_overland.Phase.water.Viscosity.Type = 'Constant'
terrain_following_grid_overland.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

# run for 2 hours @ 6min timesteps
# 
terrain_following_grid_overland.TimingInfo.BaseUnit = 1.0
terrain_following_grid_overland.TimingInfo.StartCount = 0
terrain_following_grid_overland.TimingInfo.StartTime = 0.0
terrain_following_grid_overland.TimingInfo.StopTime = 2.0
terrain_following_grid_overland.TimingInfo.DumpInterval = -1
terrain_following_grid_overland.TimeStep.Type = 'Constant'
terrain_following_grid_overland.TimeStep.Value = 0.1
#  
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Geom.Porosity.GeomNames = 'domain'


terrain_following_grid_overland.Geom.domain.Porosity.Type = 'Constant'
terrain_following_grid_overland.Geom.domain.Porosity.Value = 0.1

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Phase.RelPerm.Type = 'VanGenuchten'
terrain_following_grid_overland.Phase.RelPerm.GeomNames = 'domain'

terrain_following_grid_overland.Geom.domain.RelPerm.Alpha = 6.0
terrain_following_grid_overland.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

terrain_following_grid_overland.Phase.Saturation.Type = 'VanGenuchten'
terrain_following_grid_overland.Phase.Saturation.GeomNames = 'domain'

terrain_following_grid_overland.Geom.domain.Saturation.Alpha = 6.0
terrain_following_grid_overland.Geom.domain.Saturation.N = 2.
terrain_following_grid_overland.Geom.domain.Saturation.SRes = 0.2
terrain_following_grid_overland.Geom.domain.Saturation.SSat = 1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
terrain_following_grid_overland.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
terrain_following_grid_overland.Cycle.Names = 'constant rainrec'
terrain_following_grid_overland.Cycle.constant.Names = 'alltime'
terrain_following_grid_overland.Cycle.constant.alltime.Length = 1
terrain_following_grid_overland.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

terrain_following_grid_overland.Cycle.rainrec.Names = 'rain rec'
terrain_following_grid_overland.Cycle.rainrec.rain.Length = 2
terrain_following_grid_overland.Cycle.rainrec.rec.Length = 2
terrain_following_grid_overland.Cycle.rainrec.Repeat = -1
#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
terrain_following_grid_overland.BCPressure.PatchNames = terrain_following_grid_overland.Geom.domain.Patches

terrain_following_grid_overland.Patch.x_lower.BCPressure.Type = 'FluxConst'
terrain_following_grid_overland.Patch.x_lower.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

terrain_following_grid_overland.Patch.y_lower.BCPressure.Type = 'FluxConst'
terrain_following_grid_overland.Patch.y_lower.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

terrain_following_grid_overland.Patch.z_lower.BCPressure.Type = 'FluxConst'
terrain_following_grid_overland.Patch.z_lower.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

terrain_following_grid_overland.Patch.x_upper.BCPressure.Type = 'FluxConst'
terrain_following_grid_overland.Patch.x_upper.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

terrain_following_grid_overland.Patch.y_upper.BCPressure.Type = 'FluxConst'
terrain_following_grid_overland.Patch.y_upper.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

terrain_following_grid_overland.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
##pfset Patch.z-upper.BCPressure.Type		      FluxConst

terrain_following_grid_overland.Patch.z_upper.BCPressure.Cycle = 'constant'
terrain_following_grid_overland.Patch.z_upper.BCPressure.alltime.Value = 0.00

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

terrain_following_grid_overland.TopoSlopesX.Type = 'Constant'
terrain_following_grid_overland.TopoSlopesX.GeomNames = 'domain'
terrain_following_grid_overland.TopoSlopesX.Geom.domain.Value = 0.05

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------


terrain_following_grid_overland.TopoSlopesY.Type = 'Constant'
terrain_following_grid_overland.TopoSlopesY.GeomNames = 'domain'
terrain_following_grid_overland.TopoSlopesY.Geom.domain.Value = 0.00

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

terrain_following_grid_overland.Mannings.Type = 'Constant'
terrain_following_grid_overland.Mannings.GeomNames = 'domain'
terrain_following_grid_overland.Mannings.Geom.domain.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

terrain_following_grid_overland.PhaseSources.water.Type = 'Constant'
terrain_following_grid_overland.PhaseSources.water.GeomNames = 'domain'
terrain_following_grid_overland.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

terrain_following_grid_overland.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

terrain_following_grid_overland.Solver = 'Richards'
# setting this to True sets a subsurface slope that is the same as the topo slopes
#
terrain_following_grid_overland.Solver.TerrainFollowingGrid = True

terrain_following_grid_overland.Solver.MaxIter = 2500

terrain_following_grid_overland.Solver.Nonlinear.MaxIter = 300
terrain_following_grid_overland.Solver.Nonlinear.ResidualTol = 1e-6
terrain_following_grid_overland.Solver.Nonlinear.EtaChoice = 'Walker1'
terrain_following_grid_overland.Solver.Nonlinear.EtaValue = 0.001
terrain_following_grid_overland.Solver.Nonlinear.UseJacobian = False
terrain_following_grid_overland.Solver.Nonlinear.DerivativeEpsilon = 1e-12
terrain_following_grid_overland.Solver.Nonlinear.StepTol = 1e-20
terrain_following_grid_overland.Solver.Nonlinear.Globalization = 'LineSearch'
terrain_following_grid_overland.Solver.Linear.KrylovDimension = 20
terrain_following_grid_overland.Solver.Linear.MaxRestart = 2

terrain_following_grid_overland.Solver.Linear.Preconditioner = 'MGSemi'
terrain_following_grid_overland.Solver.Linear.Preconditioner = 'PFMG'
terrain_following_grid_overland.Solver.Linear.Preconditioner.SymmetricMat = 'Symmetric'
terrain_following_grid_overland.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
terrain_following_grid_overland.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10
terrain_following_grid_overland.Solver.PrintSubsurf = False
terrain_following_grid_overland.Solver.Drop = 1E-20
terrain_following_grid_overland.Solver.AbsTol = 1E-12

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be 1m from the bottom of the domain, the top layer is initially dry
terrain_following_grid_overland.ICPressure.Type = 'HydroStaticPatch'
terrain_following_grid_overland.ICPressure.GeomNames = 'domain'
terrain_following_grid_overland.Geom.domain.ICPressure.Value = 1.0

terrain_following_grid_overland.Geom.domain.ICPressure.RefGeom = 'domain'
terrain_following_grid_overland.Geom.domain.ICPressure.RefPatch = 'z_lower'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

terrain_following_grid_overland.run()
