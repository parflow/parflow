#---------------------------------------------------------
#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR
#---------------------------------------------------------

#---------------------------------------------------------
# Some controls for the test
#---------------------------------------------------------

#---------------------------------------------------------
# Control slopes 
#-1 = slope to lower-y
# 0 = flat top (no overland flow)
# 1 = slope to upper-y 
#---------------------------------------------------------

use_slopes = 1

#---------------------------------------------------------
# Flux on the top surface
#---------------------------------------------------------

rain_flux = -0.05
rec_flux = 0.0

#---------------------------------------------------------
# Import ParFlow
#---------------------------------------------------------

from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path

wby = Run("water_balance_y", __file__)

#---------------------------------------------------------

wby.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------

wby.Process.Topology.P = 1
wby.Process.Topology.Q = 1
wby.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

wby.ComputationalGrid.Lower.X = 0.0
wby.ComputationalGrid.Lower.Y = 0.0
wby.ComputationalGrid.Lower.Z = 0.0

wby.ComputationalGrid.NX = 30
wby.ComputationalGrid.NY = 30
wby.ComputationalGrid.NZ = 30

wby.ComputationalGrid.DX = 10.0
wby.ComputationalGrid.DY = 10.0
wby.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

wby.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

wby.GeomInput.domaininput.GeomName = 'domain'
wby.GeomInput.leftinput.GeomName = 'left'
wby.GeomInput.rightinput.GeomName = 'right'
wby.GeomInput.channelinput.GeomName = 'channel'

wby.GeomInput.domaininput.InputType = 'Box'
wby.GeomInput.leftinput.InputType = 'Box'
wby.GeomInput.rightinput.InputType = 'Box'
wby.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------

wby.Geom.domain.Lower.X = 0.0
wby.Geom.domain.Lower.Y = 0.0
wby.Geom.domain.Lower.Z = 0.0

wby.Geom.domain.Upper.X = 300.0
wby.Geom.domain.Upper.Y = 300.0
wby.Geom.domain.Upper.Z = 1.5
wby.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------

wby.Geom.left.Lower.X = 0.0
wby.Geom.left.Lower.Y = 0.0
wby.Geom.left.Lower.Z = 0.0

wby.Geom.left.Upper.X = 140.0
wby.Geom.left.Upper.Y = 300.0
wby.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------

wby.Geom.right.Lower.X = 160.0
wby.Geom.right.Lower.Y = 0.0
wby.Geom.right.Lower.Z = 0.0

wby.Geom.right.Upper.X = 300.0
wby.Geom.right.Upper.Y = 300.0
wby.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------

wby.Geom.channel.Lower.X = 140.0
wby.Geom.channel.Lower.Y = 0.0
wby.Geom.channel.Lower.Z = 0.0

wby.Geom.channel.Upper.X = 160.0
wby.Geom.channel.Upper.Y = 300.0
wby.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

wby.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

wby.Geom.left.Perm.Type = 'TurnBands'
wby.Geom.left.Perm.LambdaX = 50.
wby.Geom.left.Perm.LambdaY = 50.
wby.Geom.left.Perm.LambdaZ = 0.5
wby.Geom.left.Perm.GeomMean = 0.01

wby.Geom.left.Perm.Sigma = 0.5
wby.Geom.left.Perm.NumLines = 40
wby.Geom.left.Perm.RZeta = 5.0
wby.Geom.left.Perm.KMax = 100.0
wby.Geom.left.Perm.DelK = 0.2
wby.Geom.left.Perm.Seed = 33333
wby.Geom.left.Perm.LogNormal = 'Log'
wby.Geom.left.Perm.StratType = 'Bottom'

wby.Geom.right.Perm.Type = 'TurnBands'
wby.Geom.right.Perm.LambdaX = 50.
wby.Geom.right.Perm.LambdaY = 50.
wby.Geom.right.Perm.LambdaZ = 0.5
wby.Geom.right.Perm.GeomMean = 0.05

wby.Geom.right.Perm.Sigma = 0.5
wby.Geom.right.Perm.NumLines = 40
wby.Geom.right.Perm.RZeta = 5.0
wby.Geom.right.Perm.KMax = 100.0
wby.Geom.right.Perm.DelK = 0.2
wby.Geom.right.Perm.Seed = 13333
wby.Geom.right.Perm.LogNormal = 'Log'
wby.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface

wby.Geom.left.Perm.Type = 'Constant'
wby.Geom.left.Perm.Value = 0.001

wby.Geom.right.Perm.Type = 'Constant'
wby.Geom.right.Perm.Value = 0.01

wby.Geom.channel.Perm.Type = 'Constant'
wby.Geom.channel.Perm.Value = 0.00001

wby.Perm.TensorType = 'TensorByGeom'

wby.Geom.Perm.TensorByGeom.Names = 'domain'

wby.Geom.domain.Perm.TensorValX = 1.0
wby.Geom.domain.Perm.TensorValY = 1.0
wby.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

wby.SpecificStorage.Type = 'Constant'
wby.SpecificStorage.GeomNames = 'domain'
wby.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

wby.Phase.Names = 'water'

wby.Phase.water.Density.Type = 'Constant'
wby.Phase.water.Density.Value = 1.0

wby.Phase.water.Viscosity.Type = 'Constant'
wby.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

wby.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

wby.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

wby.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

wby.TimingInfo.BaseUnit = 0.1
wby.TimingInfo.StartCount = 0
wby.TimingInfo.StartTime = 0.0
wby.TimingInfo.StopTime = 2.0
wby.TimingInfo.DumpInterval = 0.1
wby.TimeStep.Type = 'Constant'
wby.TimeStep.Value = 0.1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

wby.Geom.Porosity.GeomNames = 'left right channel'

wby.Geom.left.Porosity.Type = 'Constant'
wby.Geom.left.Porosity.Value = 0.25

wby.Geom.right.Porosity.Type = 'Constant'
wby.Geom.right.Porosity.Value = 0.25

wby.Geom.channel.Porosity.Type = 'Constant'
wby.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

wby.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

wby.Phase.RelPerm.Type = 'VanGenuchten'
wby.Phase.RelPerm.GeomNames = 'domain'

wby.Geom.domain.RelPerm.Alpha = 6.0
wby.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

wby.Phase.Saturation.Type = 'VanGenuchten'
wby.Phase.Saturation.GeomNames = 'domain'

wby.Geom.domain.Saturation.Alpha = 6.0
wby.Geom.domain.Saturation.N = 2.
wby.Geom.domain.Saturation.SRes = 0.2
wby.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

wby.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

wby.Cycle.Names = 'constant rainrec'
wby.Cycle.constant.Names = 'alltime'
wby.Cycle.constant.alltime.Length = 1
wby.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

wby.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
wby.Cycle.rainrec.r0.Length = 1
wby.Cycle.rainrec.r1.Length = 1
wby.Cycle.rainrec.r2.Length = 1
wby.Cycle.rainrec.r3.Length = 1
wby.Cycle.rainrec.r4.Length = 1
wby.Cycle.rainrec.r5.Length = 1
wby.Cycle.rainrec.r6.Length = 1

wby.Cycle.rainrec.Repeat = 1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

wby.BCPressure.PatchNames = wby.Geom.domain.Patches

wby.Patch.x_lower.BCPressure.Type = 'FluxConst'
wby.Patch.x_lower.BCPressure.Cycle = 'constant'
wby.Patch.x_lower.BCPressure.alltime.Value = 0.0

wby.Patch.y_lower.BCPressure.Type = 'FluxConst'
wby.Patch.y_lower.BCPressure.Cycle = 'constant'
wby.Patch.y_lower.BCPressure.alltime.Value = 0.0

wby.Patch.z_lower.BCPressure.Type = 'FluxConst'
wby.Patch.z_lower.BCPressure.Cycle = 'constant'
wby.Patch.z_lower.BCPressure.alltime.Value = 0.0

wby.Patch.x_upper.BCPressure.Type = 'FluxConst'
wby.Patch.x_upper.BCPressure.Cycle = 'constant'
wby.Patch.x_upper.BCPressure.alltime.Value = 0.0

wby.Patch.y_upper.BCPressure.Type = 'FluxConst'
wby.Patch.y_upper.BCPressure.Cycle = 'constant'
wby.Patch.y_upper.BCPressure.alltime.Value = 0.0

wby.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
wby.Patch.z_upper.BCPressure.Cycle = 'rainrec'
wby.Patch.z_upper.BCPressure.r0.Value = rec_flux
wby.Patch.z_upper.BCPressure.r1.Value = rec_flux
wby.Patch.z_upper.BCPressure.r2.Value = rain_flux
wby.Patch.z_upper.BCPressure.r3.Value = rain_flux
wby.Patch.z_upper.BCPressure.r4.Value = rec_flux
wby.Patch.z_upper.BCPressure.r5.Value = rec_flux
wby.Patch.z_upper.BCPressure.r6.Value = rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

wby.TopoSlopesX.Type = 'Constant'
wby.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  wby.TopoSlopesX.Geom.left.Value = -0.005
  wby.TopoSlopesX.Geom.right.Value = 0.005
  wby.TopoSlopesX.Geom.channel.Value = 0.00
else:
  wby.TopoSlopesX.Geom.left.Value = 0.00
  wby.TopoSlopesX.Geom.right.Value = 0.00
  wby.TopoSlopesX.Geom.channel.Value = 0.00

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

wby.TopoSlopesY.Type = 'Constant'
wby.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  wby.TopoSlopesY.Geom.left.Value = 0.000
  wby.TopoSlopesY.Geom.right.Value = 0.000
  wby.TopoSlopesY.Geom.channel.Value = 0.001*use_slopes
else:
  wby.TopoSlopesY.Geom.left.Value = 0.000
  wby.TopoSlopesY.Geom.right.Value = 0.000
  wby.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

wby.Mannings.Type = 'Constant'
wby.Mannings.GeomNames = 'left right channel'
wby.Mannings.Geom.left.Value = 5.e-6
wby.Mannings.Geom.right.Value = 5.e-6
wby.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

wby.PhaseSources.water.Type = 'Constant'
wby.PhaseSources.water.GeomNames = 'domain'
wby.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

wby.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

wby.Solver = 'Richards'
wby.Solver.MaxIter = 2500

wby.Solver.AbsTol = 1E-12
wby.Solver.Nonlinear.MaxIter = 300
wby.Solver.Nonlinear.ResidualTol = 1e-12
wby.Solver.Nonlinear.EtaChoice = 'Walker1'
wby.Solver.Nonlinear.EtaChoice = 'EtaConstant'
wby.Solver.Nonlinear.EtaValue = 0.001
wby.Solver.Nonlinear.UseJacobian = False
wby.Solver.Nonlinear.DerivativeEpsilon = 1e-16
wby.Solver.Nonlinear.StepTol = 1e-30
wby.Solver.Nonlinear.Globalization = 'LineSearch'
wby.Solver.Linear.KrylovDimension = 20
wby.Solver.Linear.MaxRestart = 2

wby.Solver.Linear.Preconditioner = 'PFMG'
wby.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
wby.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
wby.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
wby.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1

wby.Solver.WriteSiloSubsurfData = True
wby.Solver.WriteSiloPressure = True
wby.Solver.WriteSiloSaturation = True
wby.Solver.WriteSiloConcentration = True
wby.Solver.WriteSiloSlopes = True
wby.Solver.WriteSiloMask = True
wby.Solver.WriteSiloEvapTrans = True
wby.Solver.WriteSiloEvapTransSum = True
wby.Solver.WriteSiloOverlandSum = True
wby.Solver.WriteSiloMannings = True
wby.Solver.WriteSiloSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
wby.ICPressure.Type = 'HydroStaticPatch'
wby.ICPressure.GeomNames = 'domain'

wby.Geom.domain.ICPressure.Value = -3.0

wby.Geom.domain.ICPressure.RefGeom = 'domain'
wby.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

dir_name = get_absolute_path('test_output/wby')
mkdir(dir_name)
wby.run(working_directory=dir_name)
