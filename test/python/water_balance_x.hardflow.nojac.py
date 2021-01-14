#---------------------------------------------------------
#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR
#---------------------------------------------------------

#---------------------------------------------------------
# Some controls for the test
#---------------------------------------------------------

#---------------------------------------------------------
# Control slopes 
# -1 = slope to lower-x
#  0 = flat top (no overland flow)
#  1 = slope to upper-x
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

#---------------------------------------------------------
# Name of the run
#---------------------------------------------------------

wbx = Run("water_balance_x", __file__)

#---------------------------------------------------------

wbx.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------

wbx.Process.Topology.P = 1
wbx.Process.Topology.Q = 1
wbx.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

wbx.ComputationalGrid.Lower.X = 0.0
wbx.ComputationalGrid.Lower.Y = 0.0
wbx.ComputationalGrid.Lower.Z = 0.0

wbx.ComputationalGrid.NX = 30
wbx.ComputationalGrid.NY = 30
wbx.ComputationalGrid.NZ = 30

wbx.ComputationalGrid.DX = 10.0
wbx.ComputationalGrid.DY = 10.0
wbx.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

wbx.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

wbx.GeomInput.domaininput.GeomName = 'domain'
wbx.GeomInput.leftinput.GeomName = 'left'
wbx.GeomInput.rightinput.GeomName = 'right'
wbx.GeomInput.channelinput.GeomName = 'channel'

wbx.GeomInput.domaininput.InputType = 'Box'
wbx.GeomInput.leftinput.InputType = 'Box'
wbx.GeomInput.rightinput.InputType = 'Box'
wbx.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------

wbx.Geom.domain.Lower.X = 0.0
wbx.Geom.domain.Lower.Y = 0.0
wbx.Geom.domain.Lower.Z = 0.0

wbx.Geom.domain.Upper.X = 300.0
wbx.Geom.domain.Upper.Y = 300.0
wbx.Geom.domain.Upper.Z = 1.5
wbx.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------

wbx.Geom.left.Lower.X = 0.0
wbx.Geom.left.Lower.Y = 0.0
wbx.Geom.left.Lower.Z = 0.0

wbx.Geom.left.Upper.X = 300.0
wbx.Geom.left.Upper.Y = 140.0
wbx.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------

wbx.Geom.right.Lower.X = 0.0
wbx.Geom.right.Lower.Y = 160.0
wbx.Geom.right.Lower.Z = 0.0

wbx.Geom.right.Upper.X = 300.0
wbx.Geom.right.Upper.Y = 300.0
wbx.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------

wbx.Geom.channel.Lower.X = 0.0
wbx.Geom.channel.Lower.Y = 140.0
wbx.Geom.channel.Lower.Z = 0.0

wbx.Geom.channel.Upper.X = 300.0
wbx.Geom.channel.Upper.Y = 160.0
wbx.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

wbx.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

wbx.Geom.left.Perm.Type = 'TurnBands'
wbx.Geom.left.Perm.LambdaX = 50.
wbx.Geom.left.Perm.LambdaY = 50.
wbx.Geom.left.Perm.LambdaZ = 0.5
wbx.Geom.left.Perm.GeomMean = 0.01

wbx.Geom.left.Perm.Sigma = 0.5
wbx.Geom.left.Perm.NumLines = 40
wbx.Geom.left.Perm.RZeta = 5.0
wbx.Geom.left.Perm.KMax = 100.0
wbx.Geom.left.Perm.DelK = 0.2
wbx.Geom.left.Perm.Seed = 33333
wbx.Geom.left.Perm.LogNormal = 'Log'
wbx.Geom.left.Perm.StratType = 'Bottom'

wbx.Geom.right.Perm.Type = 'TurnBands'
wbx.Geom.right.Perm.LambdaX = 50.
wbx.Geom.right.Perm.LambdaY = 50.
wbx.Geom.right.Perm.LambdaZ = 0.5
wbx.Geom.right.Perm.GeomMean = 0.05

wbx.Geom.right.Perm.Sigma = 0.5
wbx.Geom.right.Perm.NumLines = 40
wbx.Geom.right.Perm.RZeta = 5.0
wbx.Geom.right.Perm.KMax = 100.0
wbx.Geom.right.Perm.DelK = 0.2
wbx.Geom.right.Perm.Seed = 13333
wbx.Geom.right.Perm.LogNormal = 'Log'
wbx.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface

wbx.Geom.left.Perm.Type = 'Constant'
wbx.Geom.left.Perm.Value = 0.001

wbx.Geom.right.Perm.Type = 'Constant'
wbx.Geom.right.Perm.Value = 0.01

wbx.Geom.channel.Perm.Type = 'Constant'
wbx.Geom.channel.Perm.Value = 0.00001

wbx.Perm.TensorType = 'TensorByGeom'

wbx.Geom.Perm.TensorByGeom.Names = 'domain'

wbx.Geom.domain.Perm.TensorValX = 1.0
wbx.Geom.domain.Perm.TensorValY = 1.0
wbx.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

wbx.SpecificStorage.Type = 'Constant'
wbx.SpecificStorage.GeomNames = 'domain'
wbx.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

wbx.Phase.Names = 'water'

wbx.Phase.water.Density.Type = 'Constant'
wbx.Phase.water.Density.Value = 1.0

wbx.Phase.water.Viscosity.Type = 'Constant'
wbx.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

wbx.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

wbx.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

wbx.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

wbx.TimingInfo.BaseUnit = 0.1
wbx.TimingInfo.StartCount = 0
wbx.TimingInfo.StartTime = 0.0
wbx.TimingInfo.StopTime = 2.0
wbx.TimingInfo.DumpInterval = 0.1
wbx.TimeStep.Type = 'Constant'
wbx.TimeStep.Value = 0.1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

wbx.Geom.Porosity.GeomNames = 'left right channel'

wbx.Geom.left.Porosity.Type = 'Constant'
wbx.Geom.left.Porosity.Value = 0.25

wbx.Geom.right.Porosity.Type = 'Constant'
wbx.Geom.right.Porosity.Value = 0.25

wbx.Geom.channel.Porosity.Type = 'Constant'
wbx.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

wbx.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

wbx.Phase.RelPerm.Type = 'VanGenuchten'
wbx.Phase.RelPerm.GeomNames = 'domain'

wbx.Geom.domain.RelPerm.Alpha = 0.5
wbx.Geom.domain.RelPerm.N = 3.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

wbx.Phase.Saturation.Type = 'VanGenuchten'
wbx.Phase.Saturation.GeomNames = 'domain'

wbx.Geom.domain.Saturation.Alpha = 0.5
wbx.Geom.domain.Saturation.N = 3.
wbx.Geom.domain.Saturation.SRes = 0.2
wbx.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

wbx.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

wbx.Cycle.Names = 'constant rainrec'
wbx.Cycle.constant.Names = 'alltime'
wbx.Cycle.constant.alltime.Length = 1
wbx.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

wbx.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
wbx.Cycle.rainrec.r0.Length = 1
wbx.Cycle.rainrec.r1.Length = 1
wbx.Cycle.rainrec.r2.Length = 1
wbx.Cycle.rainrec.r3.Length = 1
wbx.Cycle.rainrec.r4.Length = 1
wbx.Cycle.rainrec.r5.Length = 1
wbx.Cycle.rainrec.r6.Length = 1

wbx.Cycle.rainrec.Repeat = 1
#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

wbx.BCPressure.PatchNames = wbx.Geom.domain.Patches

wbx.Patch.x_lower.BCPressure.Type = 'FluxConst'
wbx.Patch.x_lower.BCPressure.Cycle = 'constant'
wbx.Patch.x_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.y_lower.BCPressure.Type = 'FluxConst'
wbx.Patch.y_lower.BCPressure.Cycle = 'constant'
wbx.Patch.y_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.z_lower.BCPressure.Type = 'FluxConst'
wbx.Patch.z_lower.BCPressure.Cycle = 'constant'
wbx.Patch.z_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.x_upper.BCPressure.Type = 'FluxConst'
wbx.Patch.x_upper.BCPressure.Cycle = 'constant'
wbx.Patch.x_upper.BCPressure.alltime.Value = 0.0

wbx.Patch.y_upper.BCPressure.Type = 'FluxConst'
wbx.Patch.y_upper.BCPressure.Cycle = 'constant'
wbx.Patch.y_upper.BCPressure.alltime.Value = 0.0

wbx.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
wbx.Patch.z_upper.BCPressure.Cycle = 'rainrec'
wbx.Patch.z_upper.BCPressure.r0.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r1.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r2.Value = rain_flux
wbx.Patch.z_upper.BCPressure.r3.Value = rain_flux
wbx.Patch.z_upper.BCPressure.r4.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r5.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r6.Value = rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

wbx.TopoSlopesX.Type = 'Constant'
wbx.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  wbx.TopoSlopesX.Geom.left.Value = 0.000
  wbx.TopoSlopesX.Geom.right.Value = 0.000
  wbx.TopoSlopesX.Geom.channel.Value = 0.001*use_slopes
else:
  wbx.TopoSlopesX.Geom.left.Value = 0.000
  wbx.TopoSlopesX.Geom.right.Value = 0.000
  wbx.TopoSlopesX.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

wbx.TopoSlopesY.Type = 'Constant'
wbx.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  wbx.TopoSlopesY.Geom.left.Value = -0.005
  wbx.TopoSlopesY.Geom.right.Value = 0.005
  wbx.TopoSlopesY.Geom.channel.Value = 0.000
else:
  wbx.TopoSlopesY.Geom.left.Value = 0.000
  wbx.TopoSlopesY.Geom.right.Value = 0.000
  wbx.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

wbx.Mannings.Type = 'Constant'
wbx.Mannings.GeomNames = 'left right channel'
wbx.Mannings.Geom.left.Value = 5.e-6
wbx.Mannings.Geom.right.Value = 5.e-6
wbx.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

wbx.PhaseSources.water.Type = 'Constant'
wbx.PhaseSources.water.GeomNames = 'domain'
wbx.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

wbx.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

wbx.Solver = 'Richards'
wbx.Solver.MaxIter = 2500

wbx.Solver.AbsTol = 1E-10
wbx.Solver.Nonlinear.MaxIter = 20
wbx.Solver.Nonlinear.ResidualTol = 1e-9
wbx.Solver.Nonlinear.EtaChoice = 'Walker1'
wbx.Solver.Nonlinear.EtaChoice = 'EtaConstant'
wbx.Solver.Nonlinear.EtaValue = 0.01
wbx.Solver.Nonlinear.UseJacobian = False
wbx.Solver.Nonlinear.DerivativeEpsilon = 1e-8
wbx.Solver.Nonlinear.StepTol = 1e-30
wbx.Solver.Nonlinear.Globalization = 'LineSearch'
wbx.Solver.Linear.KrylovDimension = 20
wbx.Solver.Linear.MaxRestart = 2

wbx.Solver.Linear.Preconditioner = 'PFMG'
wbx.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
wbx.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
wbx.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
wbx.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1

wbx.Solver.WriteSiloSubsurfData = True
wbx.Solver.WriteSiloPressure = True
wbx.Solver.WriteSiloSaturation = True
wbx.Solver.WriteSiloConcentration = True
wbx.Solver.WriteSiloSlopes = True
wbx.Solver.WriteSiloMask = True
wbx.Solver.WriteSiloEvapTrans = True
wbx.Solver.WriteSiloEvapTransSum = True
wbx.Solver.WriteSiloOverlandSum = True
wbx.Solver.WriteSiloMannings = True
wbx.Solver.WriteSiloSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
wbx.ICPressure.Type = 'HydroStaticPatch'
wbx.ICPressure.GeomNames = 'domain'

wbx.Geom.domain.ICPressure.Value = -3.0

wbx.Geom.domain.ICPressure.RefGeom = 'domain'
wbx.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

dir_name = get_absolute_path('test_output/wbx_hnj')
mkdir(dir_name)
wbx.run(working_directory=dir_name)
