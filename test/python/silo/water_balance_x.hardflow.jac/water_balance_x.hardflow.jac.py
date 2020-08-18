#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR

tcl_precision = 16

verbose = 0

#---------------------------------------------------------
# Some controls for the test
#---------------------------------------------------------

#---------------------------------------------------------
# Control slopes 
#-1 = slope to lower-x
# 0 = flat top (no overland flow)
# 1 = slope to upper-x
#---------------------------------------------------------
use_slopes = -1

#---------------------------------------------------------
# Flux on the top surface
#---------------------------------------------------------
rain_flux = -0.05
rec_flux = 0.0

#---------------------------------------------------------
# Import the ParFlow TCL package
#---------------------------------------------------------
from parflow import Run
water_balance_x = Run("water_balance_x", __file__)

#---------------------------------------------------------
# Name of the run
#---------------------------------------------------------
# set runname water_balance

water_balance_x.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------
water_balance_x.Process.Topology.P = 1
water_balance_x.Process.Topology.Q = 1
water_balance_x.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
water_balance_x.ComputationalGrid.Lower.X = 0.0
water_balance_x.ComputationalGrid.Lower.Y = 0.0
water_balance_x.ComputationalGrid.Lower.Z = 0.0

water_balance_x.ComputationalGrid.NX = 30
water_balance_x.ComputationalGrid.NY = 30
water_balance_x.ComputationalGrid.NZ = 30

water_balance_x.ComputationalGrid.DX = 10.0
water_balance_x.ComputationalGrid.DY = 10.0
water_balance_x.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
water_balance_x.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

water_balance_x.GeomInput.domaininput.GeomName = 'domain'
water_balance_x.GeomInput.leftinput.GeomName = 'left'
water_balance_x.GeomInput.rightinput.GeomName = 'right'
water_balance_x.GeomInput.channelinput.GeomName = 'channel'

water_balance_x.GeomInput.domaininput.InputType = 'Box'
water_balance_x.GeomInput.leftinput.InputType = 'Box'
water_balance_x.GeomInput.rightinput.InputType = 'Box'
water_balance_x.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
water_balance_x.Geom.domain.Lower.X = 0.0
water_balance_x.Geom.domain.Lower.Y = 0.0
water_balance_x.Geom.domain.Lower.Z = 0.0
#  
water_balance_x.Geom.domain.Upper.X = 300.0
water_balance_x.Geom.domain.Upper.Y = 300.0
water_balance_x.Geom.domain.Upper.Z = 1.5
water_balance_x.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------
water_balance_x.Geom.left.Lower.X = 0.0
water_balance_x.Geom.left.Lower.Y = 0.0
water_balance_x.Geom.left.Lower.Z = 0.0
#  
water_balance_x.Geom.left.Upper.X = 300.0
water_balance_x.Geom.left.Upper.Y = 140.0
water_balance_x.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------
water_balance_x.Geom.right.Lower.X = 0.0
water_balance_x.Geom.right.Lower.Y = 160.0
water_balance_x.Geom.right.Lower.Z = 0.0
#  
water_balance_x.Geom.right.Upper.X = 300.0
water_balance_x.Geom.right.Upper.Y = 300.0
water_balance_x.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------
water_balance_x.Geom.channel.Lower.X = 0.0
water_balance_x.Geom.channel.Lower.Y = 140.0
water_balance_x.Geom.channel.Lower.Z = 0.0
#  
water_balance_x.Geom.channel.Upper.X = 300.0
water_balance_x.Geom.channel.Upper.Y = 160.0
water_balance_x.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

water_balance_x.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

water_balance_x.Geom.left.Perm.Type = 'TurnBands'
water_balance_x.Geom.left.Perm.LambdaX = 50.
water_balance_x.Geom.left.Perm.LambdaY = 50.
water_balance_x.Geom.left.Perm.LambdaZ = 0.5
water_balance_x.Geom.left.Perm.GeomMean = 0.01

water_balance_x.Geom.left.Perm.Sigma = 0.5
water_balance_x.Geom.left.Perm.NumLines = 40
water_balance_x.Geom.left.Perm.RZeta = 5.0
water_balance_x.Geom.left.Perm.KMax = 100.0
water_balance_x.Geom.left.Perm.DelK = 0.2
water_balance_x.Geom.left.Perm.Seed = 33333
water_balance_x.Geom.left.Perm.LogNormal = 'Log'
water_balance_x.Geom.left.Perm.StratType = 'Bottom'


water_balance_x.Geom.right.Perm.Type = 'TurnBands'
water_balance_x.Geom.right.Perm.LambdaX = 50.
water_balance_x.Geom.right.Perm.LambdaY = 50.
water_balance_x.Geom.right.Perm.LambdaZ = 0.5
water_balance_x.Geom.right.Perm.GeomMean = 0.05

water_balance_x.Geom.right.Perm.Sigma = 0.5
water_balance_x.Geom.right.Perm.NumLines = 40
water_balance_x.Geom.right.Perm.RZeta = 5.0
water_balance_x.Geom.right.Perm.KMax = 100.0
water_balance_x.Geom.right.Perm.DelK = 0.2
water_balance_x.Geom.right.Perm.Seed = 13333
water_balance_x.Geom.right.Perm.LogNormal = 'Log'
water_balance_x.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface
#

water_balance_x.Geom.left.Perm.Type = 'Constant'
water_balance_x.Geom.left.Perm.Value = 0.001

water_balance_x.Geom.right.Perm.Type = 'Constant'
water_balance_x.Geom.right.Perm.Value = 0.01

water_balance_x.Geom.channel.Perm.Type = 'Constant'
water_balance_x.Geom.channel.Perm.Value = 0.00001

water_balance_x.Perm.TensorType = 'TensorByGeom'

water_balance_x.Geom.Perm.TensorByGeom.Names = 'domain'

water_balance_x.Geom.domain.Perm.TensorValX = 1.0
water_balance_x.Geom.domain.Perm.TensorValY = 1.0
water_balance_x.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

water_balance_x.SpecificStorage.Type = 'Constant'
water_balance_x.SpecificStorage.GeomNames = 'domain'
water_balance_x.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

water_balance_x.Phase.Names = 'water'

water_balance_x.Phase.water.Density.Type = 'Constant'
water_balance_x.Phase.water.Density.Value = 1.0

water_balance_x.Phase.water.Viscosity.Type = 'Constant'
water_balance_x.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

water_balance_x.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

water_balance_x.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

water_balance_x.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

water_balance_x.TimingInfo.BaseUnit = 0.1
water_balance_x.TimingInfo.StartCount = 0
water_balance_x.TimingInfo.StartTime = 0.0
water_balance_x.TimingInfo.StopTime = 2.0
water_balance_x.TimingInfo.DumpInterval = 0.1
water_balance_x.TimeStep.Type = 'Constant'
water_balance_x.TimeStep.Value = 0.1
#  
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

water_balance_x.Geom.Porosity.GeomNames = 'left right channel'

water_balance_x.Geom.left.Porosity.Type = 'Constant'
water_balance_x.Geom.left.Porosity.Value = 0.25

water_balance_x.Geom.right.Porosity.Type = 'Constant'
water_balance_x.Geom.right.Porosity.Value = 0.25

water_balance_x.Geom.channel.Porosity.Type = 'Constant'
water_balance_x.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

water_balance_x.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

water_balance_x.Phase.RelPerm.Type = 'VanGenuchten'
water_balance_x.Phase.RelPerm.GeomNames = 'domain'

water_balance_x.Geom.domain.RelPerm.Alpha = 0.5
water_balance_x.Geom.domain.RelPerm.N = 3.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

water_balance_x.Phase.Saturation.Type = 'VanGenuchten'
water_balance_x.Phase.Saturation.GeomNames = 'domain'

water_balance_x.Geom.domain.Saturation.Alpha = 0.5
water_balance_x.Geom.domain.Saturation.N = 3.
water_balance_x.Geom.domain.Saturation.SRes = 0.2
water_balance_x.Geom.domain.Saturation.SSat = 1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
water_balance_x.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
water_balance_x.Cycle.Names = 'constant rainrec'
water_balance_x.Cycle.constant.Names = 'alltime'
water_balance_x.Cycle.constant.alltime.Length = 1
water_balance_x.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

water_balance_x.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
water_balance_x.Cycle.rainrec.r0.Length = 1
water_balance_x.Cycle.rainrec.r1.Length = 1
water_balance_x.Cycle.rainrec.r2.Length = 1
water_balance_x.Cycle.rainrec.r3.Length = 1
water_balance_x.Cycle.rainrec.r4.Length = 1
water_balance_x.Cycle.rainrec.r5.Length = 1
water_balance_x.Cycle.rainrec.r6.Length = 1

water_balance_x.Cycle.rainrec.Repeat = 1
#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
water_balance_x.BCPressure.PatchNames = water_balance_x.Geom.domain.Patches

water_balance_x.Patch.x_lower.BCPressure.Type = 'FluxConst'
water_balance_x.Patch.x_lower.BCPressure.Cycle = 'constant'
water_balance_x.Patch.x_lower.BCPressure.alltime.Value = 0.0

water_balance_x.Patch.y_lower.BCPressure.Type = 'FluxConst'
water_balance_x.Patch.y_lower.BCPressure.Cycle = 'constant'
water_balance_x.Patch.y_lower.BCPressure.alltime.Value = 0.0

water_balance_x.Patch.z_lower.BCPressure.Type = 'FluxConst'
water_balance_x.Patch.z_lower.BCPressure.Cycle = 'constant'
water_balance_x.Patch.z_lower.BCPressure.alltime.Value = 0.0

water_balance_x.Patch.x_upper.BCPressure.Type = 'FluxConst'
water_balance_x.Patch.x_upper.BCPressure.Cycle = 'constant'
water_balance_x.Patch.x_upper.BCPressure.alltime.Value = 0.0

water_balance_x.Patch.y_upper.BCPressure.Type = 'FluxConst'
water_balance_x.Patch.y_upper.BCPressure.Cycle = 'constant'
water_balance_x.Patch.y_upper.BCPressure.alltime.Value = 0.0


water_balance_x.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
water_balance_x.Patch.z_upper.BCPressure.Cycle = 'rainrec'
water_balance_x.Patch.z_upper.BCPressure.r0.Value = rec_flux
water_balance_x.Patch.z_upper.BCPressure.r1.Value = rec_flux
water_balance_x.Patch.z_upper.BCPressure.r2.Value = rain_flux
water_balance_x.Patch.z_upper.BCPressure.r3.Value = rain_flux
water_balance_x.Patch.z_upper.BCPressure.r4.Value = rec_flux
water_balance_x.Patch.z_upper.BCPressure.r5.Value = rec_flux
water_balance_x.Patch.z_upper.BCPressure.r6.Value = rec_flux


#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

water_balance_x.TopoSlopesX.Type = 'Constant'
water_balance_x.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  water_balance_x.TopoSlopesX.Geom.left.Value = -0.005
  water_balance_x.TopoSlopesX.Geom.right.Value = 0.005
  water_balance_x.TopoSlopesX.Geom.channel.Value = 0.00
else:
  water_balance_x.TopoSlopesX.Geom.left.Value = 0.000
  water_balance_x.TopoSlopesX.Geom.right.Value = 0.000
  water_balance_x.TopoSlopesX.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
water_balance_x.TopoSlopesY.Type = 'Constant'
water_balance_x.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  water_balance_x.TopoSlopesY.Geom.left.Value = 0.000
  water_balance_x.TopoSlopesY.Geom.right.Value = 0.000
  water_balance_x.TopoSlopesY.Geom.channel.Value = 0.001*use_slopes
else:
  water_balance_x.TopoSlopesY.Geom.left.Value = 0.000
  water_balance_x.TopoSlopesY.Geom.right.Value = 0.000
  water_balance_x.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

water_balance_x.Mannings.Type = 'Constant'
water_balance_x.Mannings.GeomNames = 'left right channel'
water_balance_x.Mannings.Geom.left.Value = 5.e-6
water_balance_x.Mannings.Geom.right.Value = 5.e-6
water_balance_x.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

water_balance_x.PhaseSources.water.Type = 'Constant'
water_balance_x.PhaseSources.water.GeomNames = 'domain'
water_balance_x.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

water_balance_x.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

water_balance_x.Solver = 'Richards'
water_balance_x.Solver.MaxIter = 2500

water_balance_x.Solver.AbsTol = 1E-10
water_balance_x.Solver.Nonlinear.MaxIter = 20
water_balance_x.Solver.Nonlinear.ResidualTol = 1e-9
water_balance_x.Solver.Nonlinear.EtaChoice = 'Walker1'
water_balance_x.Solver.Nonlinear.EtaChoice = 'EtaConstant'
water_balance_x.Solver.Nonlinear.EtaValue = 0.01
water_balance_x.Solver.Nonlinear.UseJacobian = True
water_balance_x.Solver.Nonlinear.DerivativeEpsilon = 1e-8
water_balance_x.Solver.Nonlinear.StepTol = 1e-30
water_balance_x.Solver.Nonlinear.Globalization = 'LineSearch'
water_balance_x.Solver.Linear.KrylovDimension = 20
water_balance_x.Solver.Linear.MaxRestart = 2

water_balance_x.Solver.Linear.Preconditioner = 'PFMG'
water_balance_x.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
water_balance_x.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
water_balance_x.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
water_balance_x.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1


water_balance_x.Solver.WriteSiloSubsurfData = True
water_balance_x.Solver.WriteSiloPressure = True
water_balance_x.Solver.WriteSiloSaturation = True
water_balance_x.Solver.WriteSiloConcentration = True
water_balance_x.Solver.WriteSiloSlopes = True
water_balance_x.Solver.WriteSiloMask = True
water_balance_x.Solver.WriteSiloEvapTrans = True
water_balance_x.Solver.WriteSiloEvapTransSum = True
water_balance_x.Solver.WriteSiloOverlandSum = True
water_balance_x.Solver.WriteSiloMannings = True
water_balance_x.Solver.WriteSiloSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
water_balance_x.ICPressure.Type = 'HydroStaticPatch'
water_balance_x.ICPressure.GeomNames = 'domain'

water_balance_x.Geom.domain.ICPressure.Value = -3.0

water_balance_x.Geom.domain.ICPressure.RefGeom = 'domain'
water_balance_x.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

water_balance_x.run()
