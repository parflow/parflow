# -----------------------------------------------------------------------------
#  Testing serial runs within same script
# -----------------------------------------------------------------------------

from parflow import Run

dover_1 = Run("dover_1", __file__)

dover_1.FileVersion = 4

dover_1.Process.Topology.P = 1
dover_1.Process.Topology.Q = 1
dover_1.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
dover_1.ComputationalGrid.Lower.X = 0.0
dover_1.ComputationalGrid.Lower.Y = 0.0
dover_1.ComputationalGrid.Lower.Z = 0.0

dover_1.ComputationalGrid.NX = 30
dover_1.ComputationalGrid.NY = 30
dover_1.ComputationalGrid.NZ = 30

dover_1.ComputationalGrid.DX = 10.0
dover_1.ComputationalGrid.DY = 10.0
dover_1.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
dover_1.GeomInput.Names = "domaininput leftinput rightinput channelinput"

dover_1.GeomInput.domaininput.GeomName = "domain"
dover_1.GeomInput.leftinput.GeomName = "left"
dover_1.GeomInput.rightinput.GeomName = "right"
dover_1.GeomInput.channelinput.GeomName = "channel"

dover_1.GeomInput.domaininput.InputType = "Box"
dover_1.GeomInput.leftinput.InputType = "Box"
dover_1.GeomInput.rightinput.InputType = "Box"
dover_1.GeomInput.channelinput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------
dover_1.Geom.domain.Lower.X = 0.0
dover_1.Geom.domain.Lower.Y = 0.0
dover_1.Geom.domain.Lower.Z = 0.0
#
dover_1.Geom.domain.Upper.X = 300.0
dover_1.Geom.domain.Upper.Y = 300.0
dover_1.Geom.domain.Upper.Z = 1.5
dover_1.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# ---------------------------------------------------------
# Left Slope Geometry
# ---------------------------------------------------------
dover_1.Geom.left.Lower.X = 0.0
dover_1.Geom.left.Lower.Y = 0.0
dover_1.Geom.left.Lower.Z = 0.0
#
dover_1.Geom.left.Upper.X = 140.0
dover_1.Geom.left.Upper.Y = 300.0
dover_1.Geom.left.Upper.Z = 1.5

# ---------------------------------------------------------
# Right Slope Geometry
# ---------------------------------------------------------
dover_1.Geom.right.Lower.X = 160.0
dover_1.Geom.right.Lower.Y = 0.0
dover_1.Geom.right.Lower.Z = 0.0
#
dover_1.Geom.right.Upper.X = 300.0
dover_1.Geom.right.Upper.Y = 300.0
dover_1.Geom.right.Upper.Z = 1.5

# ---------------------------------------------------------
# Channel Geometry
# ---------------------------------------------------------
dover_1.Geom.channel.Lower.X = 140.0
dover_1.Geom.channel.Lower.Y = 0.0
dover_1.Geom.channel.Lower.Z = 0.0
#
dover_1.Geom.channel.Upper.X = 160.0
dover_1.Geom.channel.Upper.Y = 300.0
dover_1.Geom.channel.Upper.Z = 1.5

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

dover_1.Geom.Perm.Names = "left right channel"

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

dover_1.Geom.left.Perm.Type = "TurnBands"
dover_1.Geom.left.Perm.LambdaX = 50.0
dover_1.Geom.left.Perm.LambdaY = 50.0
dover_1.Geom.left.Perm.LambdaZ = 0.5
dover_1.Geom.left.Perm.GeomMean = 0.01

dover_1.Geom.left.Perm.Sigma = 0.5
dover_1.Geom.left.Perm.NumLines = 40
dover_1.Geom.left.Perm.RZeta = 5.0
dover_1.Geom.left.Perm.KMax = 100.0
dover_1.Geom.left.Perm.DelK = 0.2
dover_1.Geom.left.Perm.Seed = 33333
dover_1.Geom.left.Perm.LogNormal = "Log"
dover_1.Geom.left.Perm.StratType = "Bottom"


dover_1.Geom.right.Perm.Type = "TurnBands"
dover_1.Geom.right.Perm.LambdaX = 50.0
dover_1.Geom.right.Perm.LambdaY = 50.0
dover_1.Geom.right.Perm.LambdaZ = 0.5
dover_1.Geom.right.Perm.GeomMean = 0.05

dover_1.Geom.right.Perm.Sigma = 0.5
dover_1.Geom.right.Perm.NumLines = 40
dover_1.Geom.right.Perm.RZeta = 5.0
dover_1.Geom.right.Perm.KMax = 100.0
dover_1.Geom.right.Perm.DelK = 0.2
dover_1.Geom.right.Perm.Seed = 13333
dover_1.Geom.right.Perm.LogNormal = "Log"
dover_1.Geom.right.Perm.StratType = "Bottom"

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface
#

dover_1.Geom.left.Perm.Type = "Constant"
dover_1.Geom.left.Perm.Value = 0.001

dover_1.Geom.right.Perm.Type = "Constant"
dover_1.Geom.right.Perm.Value = 0.01

dover_1.Geom.channel.Perm.Type = "Constant"
dover_1.Geom.channel.Perm.Value = 0.00001

dover_1.Perm.TensorType = "TensorByGeom"

dover_1.Geom.Perm.TensorByGeom.Names = "domain"

dover_1.Geom.domain.Perm.TensorValX = 1.0
dover_1.Geom.domain.Perm.TensorValY = 1.0
dover_1.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

dover_1.SpecificStorage.Type = "Constant"
dover_1.SpecificStorage.GeomNames = "domain"
dover_1.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

dover_1.Phase.Names = "water"

dover_1.Phase.water.Density.Type = "Constant"
dover_1.Phase.water.Density.Value = 1.0

dover_1.Phase.water.Viscosity.Type = "Constant"
dover_1.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

dover_1.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

dover_1.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

dover_1.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

#
dover_1.TimingInfo.BaseUnit = 0.1
dover_1.TimingInfo.StartCount = 0
dover_1.TimingInfo.StartTime = 0.0
dover_1.TimingInfo.StopTime = 0.4
dover_1.TimingInfo.DumpInterval = 0
dover_1.TimeStep.Type = "Constant"
dover_1.TimeStep.Value = 0.1
#
# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

dover_1.Geom.Porosity.GeomNames = "left right channel"

dover_1.Geom.left.Porosity.Type = "Constant"
dover_1.Geom.left.Porosity.Value = 0.25

dover_1.Geom.right.Porosity.Type = "Constant"
dover_1.Geom.right.Porosity.Value = 0.25

dover_1.Geom.channel.Porosity.Type = "Constant"
dover_1.Geom.channel.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

dover_1.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

dover_1.Phase.RelPerm.Type = "VanGenuchten"
dover_1.Phase.RelPerm.GeomNames = "domain"

dover_1.Geom.domain.RelPerm.Alpha = 6.0
dover_1.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

dover_1.Phase.Saturation.Type = "VanGenuchten"
dover_1.Phase.Saturation.GeomNames = "domain"

dover_1.Geom.domain.Saturation.Alpha = 6.0
dover_1.Geom.domain.Saturation.N = 2.0
dover_1.Geom.domain.Saturation.SRes = 0.2
dover_1.Geom.domain.Saturation.SSat = 1.0


# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
dover_1.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
dover_1.Cycle.Names = "constant rainrec"
dover_1.Cycle.constant.Names = "alltime"
dover_1.Cycle.constant.alltime.Length = 1
dover_1.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

dover_1.Cycle.rainrec.Names = "rain rec"
dover_1.Cycle.rainrec.rain.Length = 1
dover_1.Cycle.rainrec.rec.Length = 2
dover_1.Cycle.rainrec.Repeat = -1
#
# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
dover_1.BCPressure.PatchNames = dover_1.Geom.domain.Patches

dover_1.Patch.x_lower.BCPressure.Type = "FluxConst"
dover_1.Patch.x_lower.BCPressure.Cycle = "constant"
dover_1.Patch.x_lower.BCPressure.alltime.Value = 0.0

dover_1.Patch.y_lower.BCPressure.Type = "FluxConst"
dover_1.Patch.y_lower.BCPressure.Cycle = "constant"
dover_1.Patch.y_lower.BCPressure.alltime.Value = 0.0

dover_1.Patch.z_lower.BCPressure.Type = "FluxConst"
dover_1.Patch.z_lower.BCPressure.Cycle = "constant"
dover_1.Patch.z_lower.BCPressure.alltime.Value = 0.0

dover_1.Patch.x_upper.BCPressure.Type = "FluxConst"
dover_1.Patch.x_upper.BCPressure.Cycle = "constant"
dover_1.Patch.x_upper.BCPressure.alltime.Value = 0.0

dover_1.Patch.y_upper.BCPressure.Type = "FluxConst"
dover_1.Patch.y_upper.BCPressure.Cycle = "constant"
dover_1.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
dover_1.Patch.z_upper.BCPressure.Type = "OverlandFlow"
dover_1.Patch.z_upper.BCPressure.Cycle = "rainrec"
dover_1.Patch.z_upper.BCPressure.rain.Value = -0.05
dover_1.Patch.z_upper.BCPressure.rec.Value = 0.000001

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

dover_1.TopoSlopesX.Type = "Constant"
dover_1.TopoSlopesX.GeomNames = "left right channel"
dover_1.TopoSlopesX.Geom.left.Value = -0.005
dover_1.TopoSlopesX.Geom.right.Value = 0.005
dover_1.TopoSlopesX.Geom.channel.Value = 0.00

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------


dover_1.TopoSlopesY.Type = "Constant"
dover_1.TopoSlopesY.GeomNames = "left right channel"
dover_1.TopoSlopesY.Geom.left.Value = 0.001
dover_1.TopoSlopesY.Geom.right.Value = 0.001
dover_1.TopoSlopesY.Geom.channel.Value = 0.001

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

dover_1.Mannings.Type = "Constant"
dover_1.Mannings.GeomNames = "left right channel"
dover_1.Mannings.Geom.left.Value = 5.0e-6
dover_1.Mannings.Geom.right.Value = 5.0e-6
dover_1.Mannings.Geom.channel.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

dover_1.PhaseSources.water.Type = "Constant"
dover_1.PhaseSources.water.GeomNames = "domain"
dover_1.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

dover_1.KnownSolution = "NoKnownSolution"


# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

dover_1.Solver = "Richards"
dover_1.Solver.MaxIter = 2500

dover_1.Solver.Nonlinear.MaxIter = 20
dover_1.Solver.Nonlinear.ResidualTol = 1e-9
dover_1.Solver.Nonlinear.EtaChoice = "EtaConstant"
dover_1.Solver.Nonlinear.EtaValue = 0.01
dover_1.Solver.Nonlinear.UseJacobian = False
dover_1.Solver.Nonlinear.DerivativeEpsilon = 1e-8
dover_1.Solver.Nonlinear.StepTol = 1e-20
dover_1.Solver.Nonlinear.Globalization = "LineSearch"
dover_1.Solver.Linear.KrylovDimension = 20
dover_1.Solver.Linear.MaxRestart = 2

dover_1.Solver.Linear.Preconditioner = "PFMGOctree"
dover_1.Solver.PrintSubsurf = False
dover_1.Solver.Drop = 1e-20
dover_1.Solver.AbsTol = 1e-9

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
dover_1.ICPressure.Type = "HydroStaticPatch"
dover_1.ICPressure.GeomNames = "domain"
dover_1.Geom.domain.ICPressure.Value = -3.0

dover_1.Geom.domain.ICPressure.RefGeom = "domain"
dover_1.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

dover_1.run()

# -----------------------------------------------------------------------------
# Testing serial run
# -----------------------------------------------------------------------------


dover_2 = dover_1.clone("dover_2")

dover_2.TopoSlopesX.Type = "Constant"
dover_2.TopoSlopesX.GeomNames = "left right channel"
dover_2.TopoSlopesX.Geom.left.Value = -0.002
dover_2.TopoSlopesX.Geom.right.Value = 0.002
dover_2.TopoSlopesX.Geom.channel.Value = 0.00

dover_2.run()
