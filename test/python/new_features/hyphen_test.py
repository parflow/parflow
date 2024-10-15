# ---------------------------------------------------------
# Testing hyphenated user-defined tokens
# (from default_overland)
# ---------------------------------------------------------

from parflow import Run

dover = Run("dover", __file__)

dover.FileVersion = 4

dover.Process.Topology.P = 1
dover.Process.Topology.Q = 1
dover.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
dover.ComputationalGrid.Lower.X = 0.0
dover.ComputationalGrid.Lower.Y = 0.0
dover.ComputationalGrid.Lower.Z = 0.0

dover.ComputationalGrid.NX = 30
dover.ComputationalGrid.NY = 30
dover.ComputationalGrid.NZ = 30

dover.ComputationalGrid.DX = 10.0
dover.ComputationalGrid.DY = 10.0
dover.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
dover.GeomInput.Names = "domaininput leftinput rightinput channelinput"

dover.GeomInput.domaininput.GeomName = "domain"
dover.GeomInput.leftinput.GeomName = "left"
dover.GeomInput.rightinput.GeomName = "right"
dover.GeomInput.channelinput.GeomName = "channel"

dover.GeomInput.domaininput.InputType = "Box"
dover.GeomInput.leftinput.InputType = "Box"
dover.GeomInput.rightinput.InputType = "Box"
dover.GeomInput.channelinput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------
dover.Geom.domain.Lower.X = 0.0
dover.Geom.domain.Lower.Y = 0.0
dover.Geom.domain.Lower.Z = 0.0
#
dover.Geom.domain.Upper.X = 300.0
dover.Geom.domain.Upper.Y = 300.0
dover.Geom.domain.Upper.Z = 1.5
dover.Geom.domain.Patches = "x-lower x-upper y-lower y-upper z-lower z-upper"

# ---------------------------------------------------------
# Left Slope Geometry
# ---------------------------------------------------------
dover.Geom.left.Lower.X = 0.0
dover.Geom.left.Lower.Y = 0.0
dover.Geom.left.Lower.Z = 0.0
#
dover.Geom.left.Upper.X = 140.0
dover.Geom.left.Upper.Y = 300.0
dover.Geom.left.Upper.Z = 1.5

# ---------------------------------------------------------
# Right Slope Geometry
# ---------------------------------------------------------
dover.Geom.right.Lower.X = 160.0
dover.Geom.right.Lower.Y = 0.0
dover.Geom.right.Lower.Z = 0.0
#
dover.Geom.right.Upper.X = 300.0
dover.Geom.right.Upper.Y = 300.0
dover.Geom.right.Upper.Z = 1.5

# ---------------------------------------------------------
# Channel Geometry
# ---------------------------------------------------------
dover.Geom.channel.Lower.X = 140.0
dover.Geom.channel.Lower.Y = 0.0
dover.Geom.channel.Lower.Z = 0.0
#
dover.Geom.channel.Upper.X = 160.0
dover.Geom.channel.Upper.Y = 300.0
dover.Geom.channel.Upper.Z = 1.5

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

dover.Geom.Perm.Names = "left right channel"

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

dover.Geom.left.Perm.Type = "TurnBands"
dover.Geom.left.Perm.LambdaX = 50.0
dover.Geom.left.Perm.LambdaY = 50.0
dover.Geom.left.Perm.LambdaZ = 0.5
dover.Geom.left.Perm.GeomMean = 0.01

dover.Geom.left.Perm.Sigma = 0.5
dover.Geom.left.Perm.NumLines = 40
dover.Geom.left.Perm.RZeta = 5.0
dover.Geom.left.Perm.KMax = 100.0
dover.Geom.left.Perm.DelK = 0.2
dover.Geom.left.Perm.Seed = 33333
dover.Geom.left.Perm.LogNormal = "Log"
dover.Geom.left.Perm.StratType = "Bottom"


dover.Geom.right.Perm.Type = "TurnBands"
dover.Geom.right.Perm.LambdaX = 50.0
dover.Geom.right.Perm.LambdaY = 50.0
dover.Geom.right.Perm.LambdaZ = 0.5
dover.Geom.right.Perm.GeomMean = 0.05

dover.Geom.right.Perm.Sigma = 0.5
dover.Geom.right.Perm.NumLines = 40
dover.Geom.right.Perm.RZeta = 5.0
dover.Geom.right.Perm.KMax = 100.0
dover.Geom.right.Perm.DelK = 0.2
dover.Geom.right.Perm.Seed = 13333
dover.Geom.right.Perm.LogNormal = "Log"
dover.Geom.right.Perm.StratType = "Bottom"

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface
#

dover.Geom.left.Perm.Type = "Constant"
dover.Geom.left.Perm.Value = 0.001

dover.Geom.right.Perm.Type = "Constant"
dover.Geom.right.Perm.Value = 0.01

dover.Geom.channel.Perm.Type = "Constant"
dover.Geom.channel.Perm.Value = 0.00001

dover.Perm.TensorType = "TensorByGeom"

dover.Geom.Perm.TensorByGeom.Names = "domain"

dover.Geom.domain.Perm.TensorValX = 1.0
dover.Geom.domain.Perm.TensorValY = 1.0
dover.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

dover.SpecificStorage.Type = "Constant"
dover.SpecificStorage.GeomNames = "domain"
dover.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

dover.Phase.Names = "water"

dover.Phase.water.Density.Type = "Constant"
dover.Phase.water.Density.Value = 1.0

dover.Phase.water.Viscosity.Type = "Constant"
dover.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

dover.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

dover.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

dover.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

#
dover.TimingInfo.BaseUnit = 0.1
dover.TimingInfo.StartCount = 0
dover.TimingInfo.StartTime = 0.0
dover.TimingInfo.StopTime = 0.4
dover.TimingInfo.DumpInterval = -1
dover.TimeStep.Type = "Constant"
dover.TimeStep.Value = 0.1
#
# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

dover.Geom.Porosity.GeomNames = "left right channel"

dover.Geom.left.Porosity.Type = "Constant"
dover.Geom.left.Porosity.Value = 0.25

dover.Geom.right.Porosity.Type = "Constant"
dover.Geom.right.Porosity.Value = 0.25

dover.Geom.channel.Porosity.Type = "Constant"
dover.Geom.channel.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

dover.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

dover.Phase.RelPerm.Type = "VanGenuchten"
dover.Phase.RelPerm.GeomNames = "domain"

dover.Geom.domain.RelPerm.Alpha = 6.0
dover.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

dover.Phase.Saturation.Type = "VanGenuchten"
dover.Phase.Saturation.GeomNames = "domain"

dover.Geom.domain.Saturation.Alpha = 6.0
dover.Geom.domain.Saturation.N = 2.0
dover.Geom.domain.Saturation.SRes = 0.2
dover.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

dover.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

dover.Cycle.Names = "constant rainrec"
dover.Cycle.constant.Names = "alltime"
dover.Cycle.constant.alltime.Length = 1
dover.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

dover.Cycle.rainrec.Names = "rain rec"
dover.Cycle.rainrec.rain.Length = 1
dover.Cycle.rainrec.rec.Length = 2
dover.Cycle.rainrec.Repeat = -1
#
# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
dover.BCPressure.PatchNames = dover.Geom.domain.Patches

dover.Patch["x-lower"].BCPressure.Type = "FluxConst"
dover.Patch["x-lower"].BCPressure.Cycle = "constant"
dover.Patch["x-lower"].BCPressure.alltime.Value = 0.0

dover.Patch["y-lower"].BCPressure.Type = "FluxConst"
dover.Patch["y-lower"].BCPressure.Cycle = "constant"
dover.Patch["y-lower"].BCPressure.alltime.Value = 0.0

dover.Patch["z-lower"].BCPressure.Type = "FluxConst"
dover.Patch["z-lower"].BCPressure.Cycle = "constant"
dover.Patch["z-lower"].BCPressure.alltime.Value = 0.0

dover.Patch["x-upper"].BCPressure.Type = "FluxConst"
dover.Patch["x-upper"].BCPressure.Cycle = "constant"
dover.Patch["x-upper"].BCPressure.alltime.Value = 0.0

dover.Patch["y-upper"].BCPressure.Type = "FluxConst"
dover.Patch["y-upper"].BCPressure.Cycle = "constant"
dover.Patch["y-upper"].BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
dover.Patch["z-upper"].BCPressure.Type = "OverlandFlow"
dover.Patch["z-upper"].BCPressure.Cycle = "rainrec"
dover.Patch["z-upper"].BCPressure.rain.Value = -0.05
dover.Patch["z-upper"].BCPressure.rec.Value = 0.000001

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

dover.TopoSlopesX.Type = "Constant"
dover.TopoSlopesX.GeomNames = "left right channel"
dover.TopoSlopesX.Geom.left.Value = -0.005
dover.TopoSlopesX.Geom.right.Value = 0.005
dover.TopoSlopesX.Geom.channel.Value = 0.00

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------


dover.TopoSlopesY.Type = "Constant"
dover.TopoSlopesY.GeomNames = "left right channel"
dover.TopoSlopesY.Geom.left.Value = 0.001
dover.TopoSlopesY.Geom.right.Value = 0.001
dover.TopoSlopesY.Geom.channel.Value = 0.001

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

dover.Mannings.Type = "Constant"
dover.Mannings.GeomNames = "left right channel"
dover.Mannings.Geom.left.Value = 5.0e-6
dover.Mannings.Geom.right.Value = 5.0e-6
dover.Mannings.Geom.channel.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

dover.PhaseSources.water.Type = "Constant"
dover.PhaseSources.water.GeomNames = "domain"
dover.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

dover.KnownSolution = "NoKnownSolution"


# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

dover.Solver = "Richards"
dover.Solver.MaxIter = 2500

dover.Solver.Nonlinear.MaxIter = 20
dover.Solver.Nonlinear.ResidualTol = 1e-9
dover.Solver.Nonlinear.EtaChoice = "EtaConstant"
dover.Solver.Nonlinear.EtaValue = 0.01
dover.Solver.Nonlinear.UseJacobian = False
dover.Solver.Nonlinear.DerivativeEpsilon = 1e-8
dover.Solver.Nonlinear.StepTol = 1e-20
dover.Solver.Nonlinear.Globalization = "LineSearch"
dover.Solver.Linear.KrylovDimension = 20
dover.Solver.Linear.MaxRestart = 2

dover.Solver.Linear.Preconditioner = "PFMGOctree"
dover.Solver.PrintSubsurf = False
dover.Solver.Drop = 1e-20
dover.Solver.AbsTol = 1e-9

dover.Solver.WriteSiloSubsurfData = True
dover.Solver.WriteSiloPressure = True
dover.Solver.WriteSiloSaturation = True
dover.Solver.WriteSiloConcentration = True

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
dover.ICPressure.Type = "HydroStaticPatch"
dover.ICPressure.GeomNames = "domain"
dover.Geom.domain.ICPressure.Value = -3.0

dover.Geom.domain.ICPressure.RefGeom = "domain"
dover.Geom.domain.ICPressure.RefPatch = "z-upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

dover.validate()
