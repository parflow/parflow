#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR

tcl_precision = 16

verbose = 0

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
# Import the ParFlow TCL package
#---------------------------------------------------------
from parflow import Run
water_balance_y = Run("water_balance_y", __file__)

#---------------------------------------------------------
# Name of the run
#---------------------------------------------------------
# set runname water_balance

water_balance_y.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------
water_balance_y.Process.Topology.P = 1
water_balance_y.Process.Topology.Q = 1
water_balance_y.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
water_balance_y.ComputationalGrid.Lower.X = 0.0
water_balance_y.ComputationalGrid.Lower.Y = 0.0
water_balance_y.ComputationalGrid.Lower.Z = 0.0

water_balance_y.ComputationalGrid.NX = 30
water_balance_y.ComputationalGrid.NY = 30
water_balance_y.ComputationalGrid.NZ = 30

water_balance_y.ComputationalGrid.DX = 10.0
water_balance_y.ComputationalGrid.DY = 10.0
water_balance_y.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
water_balance_y.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

water_balance_y.GeomInput.domaininput.GeomName = 'domain'
water_balance_y.GeomInput.leftinput.GeomName = 'left'
water_balance_y.GeomInput.rightinput.GeomName = 'right'
water_balance_y.GeomInput.channelinput.GeomName = 'channel'

water_balance_y.GeomInput.domaininput.InputType = 'Box'
water_balance_y.GeomInput.leftinput.InputType = 'Box'
water_balance_y.GeomInput.rightinput.InputType = 'Box'
water_balance_y.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
water_balance_y.Geom.domain.Lower.X = 0.0
water_balance_y.Geom.domain.Lower.Y = 0.0
water_balance_y.Geom.domain.Lower.Z = 0.0
#  
water_balance_y.Geom.domain.Upper.X = 300.0
water_balance_y.Geom.domain.Upper.Y = 300.0
water_balance_y.Geom.domain.Upper.Z = 1.5
water_balance_y.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------
water_balance_y.Geom.left.Lower.X = 0.0
water_balance_y.Geom.left.Lower.Y = 0.0
water_balance_y.Geom.left.Lower.Z = 0.0
#  
water_balance_y.Geom.left.Upper.X = 140.0
water_balance_y.Geom.left.Upper.Y = 300.0
water_balance_y.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------
water_balance_y.Geom.right.Lower.X = 160.0
water_balance_y.Geom.right.Lower.Y = 0.0
water_balance_y.Geom.right.Lower.Z = 0.0
#  
water_balance_y.Geom.right.Upper.X = 300.0
water_balance_y.Geom.right.Upper.Y = 300.0
water_balance_y.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------
water_balance_y.Geom.channel.Lower.X = 140.0
water_balance_y.Geom.channel.Lower.Y = 0.0
water_balance_y.Geom.channel.Lower.Z = 0.0
#  
water_balance_y.Geom.channel.Upper.X = 160.0
water_balance_y.Geom.channel.Upper.Y = 300.0
water_balance_y.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

water_balance_y.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

water_balance_y.Geom.left.Perm.Type = 'TurnBands'
water_balance_y.Geom.left.Perm.LambdaX = 50.
water_balance_y.Geom.left.Perm.LambdaY = 50.
water_balance_y.Geom.left.Perm.LambdaZ = 0.5
water_balance_y.Geom.left.Perm.GeomMean = 0.01

water_balance_y.Geom.left.Perm.Sigma = 0.5
water_balance_y.Geom.left.Perm.NumLines = 40
water_balance_y.Geom.left.Perm.RZeta = 5.0
water_balance_y.Geom.left.Perm.KMax = 100.0
water_balance_y.Geom.left.Perm.DelK = 0.2
water_balance_y.Geom.left.Perm.Seed = 33333
water_balance_y.Geom.left.Perm.LogNormal = 'Log'
water_balance_y.Geom.left.Perm.StratType = 'Bottom'


water_balance_y.Geom.right.Perm.Type = 'TurnBands'
water_balance_y.Geom.right.Perm.LambdaX = 50.
water_balance_y.Geom.right.Perm.LambdaY = 50.
water_balance_y.Geom.right.Perm.LambdaZ = 0.5
water_balance_y.Geom.right.Perm.GeomMean = 0.05

water_balance_y.Geom.right.Perm.Sigma = 0.5
water_balance_y.Geom.right.Perm.NumLines = 40
water_balance_y.Geom.right.Perm.RZeta = 5.0
water_balance_y.Geom.right.Perm.KMax = 100.0
water_balance_y.Geom.right.Perm.DelK = 0.2
water_balance_y.Geom.right.Perm.Seed = 13333
water_balance_y.Geom.right.Perm.LogNormal = 'Log'
water_balance_y.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface
#

water_balance_y.Geom.left.Perm.Type = 'Constant'
water_balance_y.Geom.left.Perm.Value = 0.001

water_balance_y.Geom.right.Perm.Type = 'Constant'
water_balance_y.Geom.right.Perm.Value = 0.01

water_balance_y.Geom.channel.Perm.Type = 'Constant'
water_balance_y.Geom.channel.Perm.Value = 0.00001

water_balance_y.Perm.TensorType = 'TensorByGeom'

water_balance_y.Geom.Perm.TensorByGeom.Names = 'domain'

water_balance_y.Geom.domain.Perm.TensorValX = 1.0
water_balance_y.Geom.domain.Perm.TensorValY = 1.0
water_balance_y.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

water_balance_y.SpecificStorage.Type = 'Constant'
water_balance_y.SpecificStorage.GeomNames = 'domain'
water_balance_y.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

water_balance_y.Phase.Names = 'water'

water_balance_y.Phase.water.Density.Type = 'Constant'
water_balance_y.Phase.water.Density.Value = 1.0

water_balance_y.Phase.water.Viscosity.Type = 'Constant'
water_balance_y.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

water_balance_y.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

water_balance_y.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

water_balance_y.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

water_balance_y.TimingInfo.BaseUnit = 0.1
water_balance_y.TimingInfo.StartCount = 0
water_balance_y.TimingInfo.StartTime = 0.0
water_balance_y.TimingInfo.StopTime = 2.0
water_balance_y.TimingInfo.DumpInterval = 0.1
water_balance_y.TimeStep.Type = 'Constant'
water_balance_y.TimeStep.Value = 0.1
#  
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

water_balance_y.Geom.Porosity.GeomNames = 'left right channel'

water_balance_y.Geom.left.Porosity.Type = 'Constant'
water_balance_y.Geom.left.Porosity.Value = 0.25

water_balance_y.Geom.right.Porosity.Type = 'Constant'
water_balance_y.Geom.right.Porosity.Value = 0.25

water_balance_y.Geom.channel.Porosity.Type = 'Constant'
water_balance_y.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

water_balance_y.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

water_balance_y.Phase.RelPerm.Type = 'VanGenuchten'
water_balance_y.Phase.RelPerm.GeomNames = 'domain'

water_balance_y.Geom.domain.RelPerm.Alpha = 6.0
water_balance_y.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

water_balance_y.Phase.Saturation.Type = 'VanGenuchten'
water_balance_y.Phase.Saturation.GeomNames = 'domain'

water_balance_y.Geom.domain.Saturation.Alpha = 6.0
water_balance_y.Geom.domain.Saturation.N = 2.
water_balance_y.Geom.domain.Saturation.SRes = 0.2
water_balance_y.Geom.domain.Saturation.SSat = 1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
water_balance_y.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
water_balance_y.Cycle.Names = 'constant rainrec'
water_balance_y.Cycle.constant.Names = 'alltime'
water_balance_y.Cycle.constant.alltime.Length = 1
water_balance_y.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

water_balance_y.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
water_balance_y.Cycle.rainrec.r0.Length = 1
water_balance_y.Cycle.rainrec.r1.Length = 1
water_balance_y.Cycle.rainrec.r2.Length = 1
water_balance_y.Cycle.rainrec.r3.Length = 1
water_balance_y.Cycle.rainrec.r4.Length = 1
water_balance_y.Cycle.rainrec.r5.Length = 1
water_balance_y.Cycle.rainrec.r6.Length = 1

water_balance_y.Cycle.rainrec.Repeat = 1
#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
water_balance_y.BCPressure.PatchNames = water_balance_y.Geom.domain.Patches

water_balance_y.Patch.x_lower.BCPressure.Type = 'FluxConst'
water_balance_y.Patch.x_lower.BCPressure.Cycle = 'constant'
water_balance_y.Patch.x_lower.BCPressure.alltime.Value = 0.0

water_balance_y.Patch.y_lower.BCPressure.Type = 'FluxConst'
water_balance_y.Patch.y_lower.BCPressure.Cycle = 'constant'
water_balance_y.Patch.y_lower.BCPressure.alltime.Value = 0.0

water_balance_y.Patch.z_lower.BCPressure.Type = 'FluxConst'
water_balance_y.Patch.z_lower.BCPressure.Cycle = 'constant'
water_balance_y.Patch.z_lower.BCPressure.alltime.Value = 0.0

water_balance_y.Patch.x_upper.BCPressure.Type = 'FluxConst'
water_balance_y.Patch.x_upper.BCPressure.Cycle = 'constant'
water_balance_y.Patch.x_upper.BCPressure.alltime.Value = 0.0

water_balance_y.Patch.y_upper.BCPressure.Type = 'FluxConst'
water_balance_y.Patch.y_upper.BCPressure.Cycle = 'constant'
water_balance_y.Patch.y_upper.BCPressure.alltime.Value = 0.0


water_balance_y.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
water_balance_y.Patch.z_upper.BCPressure.Cycle = 'rainrec'
water_balance_y.Patch.z_upper.BCPressure.r0.Value = rec_flux
water_balance_y.Patch.z_upper.BCPressure.r1.Value = rec_flux
water_balance_y.Patch.z_upper.BCPressure.r2.Value = rain_flux
water_balance_y.Patch.z_upper.BCPressure.r3.Value = rain_flux
water_balance_y.Patch.z_upper.BCPressure.r4.Value = rec_flux
water_balance_y.Patch.z_upper.BCPressure.r5.Value = rec_flux
water_balance_y.Patch.z_upper.BCPressure.r6.Value = rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

water_balance_y.TopoSlopesX.Type = 'Constant'
water_balance_y.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  water_balance_y.TopoSlopesX.Geom.left.Value = -0.005
  water_balance_y.TopoSlopesX.Geom.right.Value = 0.005
  water_balance_y.TopoSlopesX.Geom.channel.Value = 0.00
else:
  water_balance_y.TopoSlopesX.Geom.left.Value = 0.00
  water_balance_y.TopoSlopesX.Geom.right.Value = 0.00
  water_balance_y.TopoSlopesX.Geom.channel.Value = 0.00


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------


water_balance_y.TopoSlopesY.Type = 'Constant'
water_balance_y.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  water_balance_y.TopoSlopesY.Geom.left.Value = 0.000
  water_balance_y.TopoSlopesY.Geom.right.Value = 0.000
  water_balance_y.TopoSlopesY.Geom.channel.Value = 0.001*use_slopes
else:
  water_balance_y.TopoSlopesY.Geom.left.Value = 0.000
  water_balance_y.TopoSlopesY.Geom.right.Value = 0.000
  water_balance_y.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

water_balance_y.Mannings.Type = 'Constant'
water_balance_y.Mannings.GeomNames = 'left right channel'
water_balance_y.Mannings.Geom.left.Value = 5.e-6
water_balance_y.Mannings.Geom.right.Value = 5.e-6
water_balance_y.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

water_balance_y.PhaseSources.water.Type = 'Constant'
water_balance_y.PhaseSources.water.GeomNames = 'domain'
water_balance_y.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

water_balance_y.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

water_balance_y.Solver = 'Richards'
water_balance_y.Solver.MaxIter = 2500

water_balance_y.Solver.AbsTol = 1E-12
water_balance_y.Solver.Nonlinear.MaxIter = 300
water_balance_y.Solver.Nonlinear.ResidualTol = 1e-12
water_balance_y.Solver.Nonlinear.EtaChoice = 'Walker1'
water_balance_y.Solver.Nonlinear.EtaChoice = 'EtaConstant'
water_balance_y.Solver.Nonlinear.EtaValue = 0.001
water_balance_y.Solver.Nonlinear.UseJacobian = False
water_balance_y.Solver.Nonlinear.DerivativeEpsilon = 1e-16
water_balance_y.Solver.Nonlinear.StepTol = 1e-30
water_balance_y.Solver.Nonlinear.Globalization = 'LineSearch'
water_balance_y.Solver.Linear.KrylovDimension = 20
water_balance_y.Solver.Linear.MaxRestart = 2

water_balance_y.Solver.Linear.Preconditioner = 'PFMG'
water_balance_y.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
water_balance_y.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
water_balance_y.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
water_balance_y.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1


water_balance_y.Solver.WriteSiloSubsurfData = True
water_balance_y.Solver.WriteSiloPressure = True
water_balance_y.Solver.WriteSiloSaturation = True
water_balance_y.Solver.WriteSiloConcentration = True
water_balance_y.Solver.WriteSiloSlopes = True
water_balance_y.Solver.WriteSiloMask = True
water_balance_y.Solver.WriteSiloEvapTrans = True
water_balance_y.Solver.WriteSiloEvapTransSum = True
water_balance_y.Solver.WriteSiloOverlandSum = True
water_balance_y.Solver.WriteSiloMannings = True
water_balance_y.Solver.WriteSiloSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
water_balance_y.ICPressure.Type = 'HydroStaticPatch'
water_balance_y.ICPressure.GeomNames = 'domain'

water_balance_y.Geom.domain.ICPressure.Value = -3.0

water_balance_y.Geom.domain.ICPressure.RefGeom = 'domain'
water_balance_y.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

water_balance_y.run()
