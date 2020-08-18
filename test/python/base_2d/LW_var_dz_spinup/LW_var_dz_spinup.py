#  This runs a Little Washita test problem with variable dz
#  in spinup mode with overland dampening turned on

tcl_precision = 17

# set runname LW_var_dz_spinup

#
# Import the ParFlow TCL package
#
from parflow import Run
LW_var_dz_spinup = Run("LW_var_dz_spinup", __file__)

LW_var_dz_spinup.FileVersion = 4

LW_var_dz_spinup.Process.Topology.P = 1
LW_var_dz_spinup.Process.Topology.Q = 1
LW_var_dz_spinup.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
LW_var_dz_spinup.ComputationalGrid.Lower.X = 0.0
LW_var_dz_spinup.ComputationalGrid.Lower.Y = 0.0
LW_var_dz_spinup.ComputationalGrid.Lower.Z = 0.0

LW_var_dz_spinup.ComputationalGrid.NX = 45
LW_var_dz_spinup.ComputationalGrid.NY = 32
LW_var_dz_spinup.ComputationalGrid.NZ = 25
LW_var_dz_spinup.ComputationalGrid.NZ = 10
LW_var_dz_spinup.ComputationalGrid.NZ = 6

LW_var_dz_spinup.ComputationalGrid.DX = 1000.0
LW_var_dz_spinup.ComputationalGrid.DY = 1000.0
#"native" grid resolution is 2m everywhere X NZ=25 for 50m 
#computational domain.
LW_var_dz_spinup.ComputationalGrid.DZ = 2.0

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
LW_var_dz_spinup.GeomInput.Names = 'domaininput'

LW_var_dz_spinup.GeomInput.domaininput.GeomName = 'domain'
LW_var_dz_spinup.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
LW_var_dz_spinup.Geom.domain.Lower.X = 0.0
LW_var_dz_spinup.Geom.domain.Lower.Y = 0.0
LW_var_dz_spinup.Geom.domain.Lower.Z = 0.0
#  
LW_var_dz_spinup.Geom.domain.Upper.X = 45000.0
LW_var_dz_spinup.Geom.domain.Upper.Y = 32000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
LW_var_dz_spinup.Geom.domain.Upper.Z = 12.0
LW_var_dz_spinup.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#--------------------------------------------
# variable dz assignments
#------------------------------------------
LW_var_dz_spinup.Solver.Nonlinear.VariableDz = True
LW_var_dz_spinup.dzScale.GeomNames = 'domain'
LW_var_dz_spinup.dzScale.Type = 'nzList'
LW_var_dz_spinup.dzScale.nzListNumber = 6

#pfset dzScale.Type            nzList
#pfset dzScale.nzListNumber       3
LW_var_dz_spinup.Cell.l0.dzScale.Value = 1.0
LW_var_dz_spinup.Cell.l1.dzScale.Value = 1.00
LW_var_dz_spinup.Cell.l2.dzScale.Value = 1.000
LW_var_dz_spinup.Cell.l3.dzScale.Value = 1.000
LW_var_dz_spinup.Cell.l4.dzScale.Value = 1.000
LW_var_dz_spinup.Cell.l5.dzScale.Value = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Geom.Perm.Names = 'domain'

# Values in m/hour


LW_var_dz_spinup.Geom.domain.Perm.Type = 'Constant'
LW_var_dz_spinup.Geom.domain.Perm.Value = 0.1

#pfset Geom.domain.Perm.Type "TurnBands"
LW_var_dz_spinup.Geom.domain.Perm.LambdaX = 5000.0
LW_var_dz_spinup.Geom.domain.Perm.LambdaY = 5000.0
LW_var_dz_spinup.Geom.domain.Perm.LambdaZ = 50.0
LW_var_dz_spinup.Geom.domain.Perm.GeomMean = 0.0001427686
#pfset Geom.domain.Perm.GeomMean  0.001427686

LW_var_dz_spinup.Geom.domain.Perm.Sigma = 0.20
LW_var_dz_spinup.Geom.domain.Perm.Sigma = 1.20
#pfset Geom.domain.Perm.Sigma   0.48989794
LW_var_dz_spinup.Geom.domain.Perm.NumLines = 150
LW_var_dz_spinup.Geom.domain.Perm.RZeta = 10.0
LW_var_dz_spinup.Geom.domain.Perm.KMax = 100.0000001
LW_var_dz_spinup.Geom.domain.Perm.DelK = 0.2
LW_var_dz_spinup.Geom.domain.Perm.Seed = 33333
LW_var_dz_spinup.Geom.domain.Perm.LogNormal = 'Log'
LW_var_dz_spinup.Geom.domain.Perm.StratType = 'Bottom'


LW_var_dz_spinup.Perm.TensorType = 'TensorByGeom'

LW_var_dz_spinup.Geom.Perm.TensorByGeom.Names = 'domain'

LW_var_dz_spinup.Geom.domain.Perm.TensorValX = 1.0
LW_var_dz_spinup.Geom.domain.Perm.TensorValY = 1.0
LW_var_dz_spinup.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

LW_var_dz_spinup.SpecificStorage.Type = 'Constant'
LW_var_dz_spinup.SpecificStorage.GeomNames = 'domain'
LW_var_dz_spinup.Geom.domain.SpecificStorage.Value = 1.0e-5
#pfset Geom.domain.SpecificStorage.Value 0.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Phase.Names = 'water'

LW_var_dz_spinup.Phase.water.Density.Type = 'Constant'
LW_var_dz_spinup.Phase.water.Density.Value = 1.0

LW_var_dz_spinup.Phase.water.Viscosity.Type = 'Constant'
LW_var_dz_spinup.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
LW_var_dz_spinup.TimingInfo.BaseUnit = 10.0
LW_var_dz_spinup.TimingInfo.StartCount = 0
LW_var_dz_spinup.TimingInfo.StartTime = 0.0
LW_var_dz_spinup.TimingInfo.StopTime = 7000.0
LW_var_dz_spinup.TimingInfo.DumpInterval = 1000.0
LW_var_dz_spinup.TimeStep.Type = 'Constant'
LW_var_dz_spinup.TimeStep.Value = 1000.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Geom.Porosity.GeomNames = 'domain'

LW_var_dz_spinup.Geom.domain.Porosity.Type = 'Constant'
LW_var_dz_spinup.Geom.domain.Porosity.Value = 0.25
#pfset Geom.domain.Porosity.Value         0.


#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Phase.RelPerm.Type = 'VanGenuchten'
LW_var_dz_spinup.Phase.RelPerm.GeomNames = 'domain'

LW_var_dz_spinup.Geom.domain.RelPerm.Alpha = 1.
LW_var_dz_spinup.Geom.domain.RelPerm.Alpha = 1.0
LW_var_dz_spinup.Geom.domain.RelPerm.N = 3.
#pfset Geom.domain.RelPerm.NumSamplePoints   10000
#pfset Geom.domain.RelPerm.MinPressureHead   -200
#pfset Geom.domain.RelPerm.InterpolationMethod   "Linear"
#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

LW_var_dz_spinup.Phase.Saturation.Type = 'VanGenuchten'
LW_var_dz_spinup.Phase.Saturation.GeomNames = 'domain'

LW_var_dz_spinup.Geom.domain.Saturation.Alpha = 1.0
LW_var_dz_spinup.Geom.domain.Saturation.Alpha = 1.0
LW_var_dz_spinup.Geom.domain.Saturation.N = 3.
LW_var_dz_spinup.Geom.domain.Saturation.SRes = 0.1
LW_var_dz_spinup.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
LW_var_dz_spinup.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
LW_var_dz_spinup.Cycle.Names = 'constant rainrec'
LW_var_dz_spinup.Cycle.Names = 'constant'
LW_var_dz_spinup.Cycle.constant.Names = 'alltime'
LW_var_dz_spinup.Cycle.constant.alltime.Length = 10000000
LW_var_dz_spinup.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

LW_var_dz_spinup.Cycle.rainrec.Names = 'rain rec'
LW_var_dz_spinup.Cycle.rainrec.rain.Length = 10
LW_var_dz_spinup.Cycle.rainrec.rec.Length = 20
LW_var_dz_spinup.Cycle.rainrec.Repeat = 14
#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
LW_var_dz_spinup.BCPressure.PatchNames = LW_var_dz_spinup.Geom.domain.Patches

LW_var_dz_spinup.Patch.x_lower.BCPressure.Type = 'FluxConst'
LW_var_dz_spinup.Patch.x_lower.BCPressure.Cycle = 'constant'
LW_var_dz_spinup.Patch.x_lower.BCPressure.alltime.Value = 0.0

LW_var_dz_spinup.Patch.y_lower.BCPressure.Type = 'FluxConst'
LW_var_dz_spinup.Patch.y_lower.BCPressure.Cycle = 'constant'
LW_var_dz_spinup.Patch.y_lower.BCPressure.alltime.Value = 0.0

LW_var_dz_spinup.Patch.z_lower.BCPressure.Type = 'FluxConst'
LW_var_dz_spinup.Patch.z_lower.BCPressure.Cycle = 'constant'
LW_var_dz_spinup.Patch.z_lower.BCPressure.alltime.Value = 0.0

LW_var_dz_spinup.Patch.x_upper.BCPressure.Type = 'FluxConst'
LW_var_dz_spinup.Patch.x_upper.BCPressure.Cycle = 'constant'
LW_var_dz_spinup.Patch.x_upper.BCPressure.alltime.Value = 0.0

LW_var_dz_spinup.Patch.y_upper.BCPressure.Type = 'FluxConst'
LW_var_dz_spinup.Patch.y_upper.BCPressure.Cycle = 'constant'
LW_var_dz_spinup.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
LW_var_dz_spinup.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
LW_var_dz_spinup.Patch.z_upper.BCPressure.Cycle = 'constant'
# constant recharge at 100 mm / y
LW_var_dz_spinup.Patch.z_upper.BCPressure.alltime.Value = -0.005
LW_var_dz_spinup.Patch.z_upper.BCPressure.alltime.Value = -0.0001

#---------------
# Copy slopes to working dir
#----------------

# file copy -force input/lw.1km.slope_x.10x.pfb .
# file copy -force input/lw.1km.slope_y.10x.pfb .

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

LW_var_dz_spinup.TopoSlopesX.Type = 'PFBFile'
LW_var_dz_spinup.TopoSlopesX.GeomNames = 'domain'

LW_var_dz_spinup.TopoSlopesX.FileName = 'lw.1km.slope_x.10x.pfb'


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

LW_var_dz_spinup.TopoSlopesY.Type = 'PFBFile'
LW_var_dz_spinup.TopoSlopesY.GeomNames = 'domain'

LW_var_dz_spinup.TopoSlopesY.FileName = 'lw.1km.slope_y.10x.pfb'

#---------
##  Distribute slopes
#---------

LW_var_dz_spinup.ComputationalGrid.NX = 45
LW_var_dz_spinup.ComputationalGrid.NY = 32
LW_var_dz_spinup.ComputationalGrid.NZ = 6

# Slope files 1D files so distribute with -nz 1
# pfdist -nz 1 lw.1km.slope_x.10x.pfb
# pfdist -nz 1 lw.1km.slope_y.10x.pfb

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

LW_var_dz_spinup.Mannings.Type = 'Constant'
LW_var_dz_spinup.Mannings.GeomNames = 'domain'
LW_var_dz_spinup.Mannings.Geom.domain.Value = 0.00005


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

LW_var_dz_spinup.PhaseSources.water.Type = 'Constant'
LW_var_dz_spinup.PhaseSources.water.GeomNames = 'domain'
LW_var_dz_spinup.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

LW_var_dz_spinup.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

LW_var_dz_spinup.Solver = 'Richards'
LW_var_dz_spinup.Solver.MaxIter = 2500

LW_var_dz_spinup.Solver.TerrainFollowingGrid = True


LW_var_dz_spinup.Solver.Nonlinear.MaxIter = 80
LW_var_dz_spinup.Solver.Nonlinear.ResidualTol = 1e-5
LW_var_dz_spinup.Solver.Nonlinear.EtaValue = 0.001


LW_var_dz_spinup.Solver.PrintSubsurf = False
LW_var_dz_spinup.Solver.Drop = 1E-20
LW_var_dz_spinup.Solver.AbsTol = 1E-10


LW_var_dz_spinup.Solver.Nonlinear.EtaChoice = 'EtaConstant'
LW_var_dz_spinup.Solver.Nonlinear.EtaValue = 0.001
LW_var_dz_spinup.Solver.Nonlinear.UseJacobian = True
LW_var_dz_spinup.Solver.Nonlinear.StepTol = 1e-25
LW_var_dz_spinup.Solver.Nonlinear.Globalization = 'LineSearch'
LW_var_dz_spinup.Solver.Linear.KrylovDimension = 80
LW_var_dz_spinup.Solver.Linear.MaxRestarts = 2

LW_var_dz_spinup.Solver.Linear.Preconditioner = 'MGSemi'
LW_var_dz_spinup.Solver.Linear.Preconditioner = 'PFMG'
LW_var_dz_spinup.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

LW_var_dz_spinup.Solver.WriteSiloPressure = True
LW_var_dz_spinup.Solver.WriteSiloSaturation = True

##---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
LW_var_dz_spinup.ICPressure.Type = 'HydroStaticPatch'
LW_var_dz_spinup.ICPressure.GeomNames = 'domain'
LW_var_dz_spinup.Geom.domain.ICPressure.Value = 0.0

LW_var_dz_spinup.Geom.domain.ICPressure.RefGeom = 'domain'
LW_var_dz_spinup.Geom.domain.ICPressure.RefPatch = 'z_upper'


#spinup key

LW_var_dz_spinup.OverlandFlowSpinUp = 1
LW_var_dz_spinup.OverlandSpinupDampP1 = 1.0
LW_var_dz_spinup.OverlandSpinupDampP2 = 0.00001

#-----------------------------------------------------------------------------
# Run and do tests
#-----------------------------------------------------------------------------


LW_var_dz_spinup.run()
