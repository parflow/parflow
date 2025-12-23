# -----------------------------------------------------------------------------
#  This is an adaptation of the LW_var_dz.py test
#  This test demonstrates the different ways to set integers within a key name
#  See dzScale and Coeff keys
# -----------------------------------------------------------------------------

from parflow import Run

prefix = Run("prefix", __file__)

prefix.FileVersion = 4

prefix.Process.Topology.P = 1
prefix.Process.Topology.Q = 1
prefix.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
prefix.ComputationalGrid.Lower.X = 0.0
prefix.ComputationalGrid.Lower.Y = 0.0
prefix.ComputationalGrid.Lower.Z = 0.0

prefix.ComputationalGrid.NX = 45
prefix.ComputationalGrid.NY = 32
prefix.ComputationalGrid.NZ = 25
prefix.ComputationalGrid.NZ = 10
prefix.ComputationalGrid.NZ = 6

prefix.ComputationalGrid.DX = 1000.0
prefix.ComputationalGrid.DY = 1000.0
# "native" grid resolution is 2m everywhere X NZ=25 for 50m
# computational domain.
prefix.ComputationalGrid.DZ = 2.0

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
prefix.GeomInput.Names = "domaininput"

prefix.GeomInput.domaininput.GeomName = "domain"
prefix.GeomInput.domaininput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------
prefix.Geom.domain.Lower.X = 0.0
prefix.Geom.domain.Lower.Y = 0.0
prefix.Geom.domain.Lower.Z = 0.0

prefix.Geom.domain.Upper.X = 45000.0
prefix.Geom.domain.Upper.Y = 32000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
prefix.Geom.domain.Upper.Z = 12.0
prefix.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# --------------------------------------------
# variable dz assignments
# --------------------------------------------
prefix.Solver.Nonlinear.VariableDz = True
prefix.dzScale.GeomNames = "domain"
prefix.dzScale.Type = "nzList"
prefix.dzScale.nzListNumber = 6

# Here are four different ways to set integer values as part of a key name:
# 1) bracket, quotes, underscore
prefix.Cell["_0"].dzScale.Value = 1.0
# 2) bracket, quotes, no underscore
prefix.Cell["1"].dzScale.Value = 1.00
# 3) bracket, no quotes, no underscore
prefix.Cell[2].dzScale.Value = 1.000
# 4) no bracket, no quotes, underscore
prefix.Cell._3.dzScale.Value = 1.000
prefix.Cell._4.dzScale.Value = 1.000
prefix.Cell._5.dzScale.Value = 0.05

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

prefix.Geom.Perm.Names = "domain"

# Values in m/hour


prefix.Geom.domain.Perm.Type = "Constant"

prefix.Geom.domain.Perm.Type = "TurnBands"
prefix.Geom.domain.Perm.LambdaX = 5000.0
prefix.Geom.domain.Perm.LambdaY = 5000.0
prefix.Geom.domain.Perm.LambdaZ = 50.0
prefix.Geom.domain.Perm.GeomMean = 0.0001427686

prefix.Geom.domain.Perm.Sigma = 0.20
prefix.Geom.domain.Perm.Sigma = 1.20
# pfset Geom.domain.Perm.Sigma   0.48989794
prefix.Geom.domain.Perm.NumLines = 150
prefix.Geom.domain.Perm.RZeta = 10.0
prefix.Geom.domain.Perm.KMax = 100.0000001
prefix.Geom.domain.Perm.DelK = 0.2
prefix.Geom.domain.Perm.Seed = 33333
prefix.Geom.domain.Perm.LogNormal = "Log"
prefix.Geom.domain.Perm.StratType = "Bottom"


prefix.Perm.TensorType = "TensorByGeom"

prefix.Geom.Perm.TensorByGeom.Names = "domain"

prefix.Geom.domain.Perm.TensorValX = 1.0
prefix.Geom.domain.Perm.TensorValY = 1.0
prefix.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

prefix.SpecificStorage.Type = "Constant"
prefix.SpecificStorage.GeomNames = "domain"
prefix.Geom.domain.SpecificStorage.Value = 1.0e-5
# pfset Geom.domain.SpecificStorage.Value 0.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

prefix.Phase.Names = "water"

prefix.Phase.water.Density.Type = "Constant"
prefix.Phase.water.Density.Value = 1.0

prefix.Phase.water.Viscosity.Type = "Constant"
prefix.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

prefix.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

prefix.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

prefix.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
prefix.TimingInfo.BaseUnit = 10.0
prefix.TimingInfo.StartCount = 0
prefix.TimingInfo.StartTime = 0.0
prefix.TimingInfo.StopTime = 200.0
prefix.TimingInfo.DumpInterval = 20.0
prefix.TimeStep.Type = "Constant"
prefix.TimeStep.Value = 10.0
# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

prefix.Geom.Porosity.GeomNames = "domain"

prefix.Geom.domain.Porosity.Type = "Constant"
prefix.Geom.domain.Porosity.Value = 0.25


# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

prefix.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

prefix.Phase.RelPerm.Type = "VanGenuchten"
prefix.Phase.RelPerm.GeomNames = "domain"

prefix.Geom.domain.RelPerm.Alpha = 1.0
prefix.Geom.domain.RelPerm.N = 3.0

# Another example
prefix.Geom.domain.RelPerm.Coeff._0 = 0.5
prefix.Geom.domain.RelPerm.Coeff["_1"] = 1.0
prefix.Geom.domain.RelPerm.Coeff["2"] = 1.0
prefix.Geom.domain.RelPerm.Coeff[3] = 1.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

prefix.Phase.Saturation.Type = "VanGenuchten"
prefix.Phase.Saturation.GeomNames = "domain"

prefix.Geom.domain.Saturation.Alpha = 1.0
prefix.Geom.domain.Saturation.Alpha = 1.0
prefix.Geom.domain.Saturation.N = 3.0
prefix.Geom.domain.Saturation.SRes = 0.1
prefix.Geom.domain.Saturation.SSat = 1.0


# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
prefix.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
prefix.Cycle.Names = "constant rainrec"
prefix.Cycle.constant.Names = "alltime"
prefix.Cycle.constant.alltime.Length = 10000000
prefix.Cycle.constant.Repeat = -1

prefix.Cycle.rainrec.Names = "rain rec"
prefix.Cycle.rainrec.rain.Length = 10
prefix.Cycle.rainrec.rec.Length = 20
prefix.Cycle.rainrec.Repeat = 14

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
prefix.BCPressure.PatchNames = prefix.Geom.domain.Patches

prefix.Patch.x_lower.BCPressure.Type = "DirEquilPLinear"
prefix.Patch.x_lower.BCPressure.Cycle = "constant"
prefix.Patch.x_lower.BCPressure.alltime.NumPoints = 2

# Another example
prefix.Patch.x_lower.BCPressure.alltime._0.Location = 0.0
prefix.Patch.x_lower.BCPressure.alltime["0"].Value = 0.0
prefix.Patch.x_lower.BCPressure.alltime["1"].Location = 0.0
prefix.Patch.x_lower.BCPressure.alltime[1].Value = 14.0
prefix.Patch.x_lower.BCPressure.alltime.XLower = 0.0
prefix.Patch.x_lower.BCPressure.alltime.YLower = 0.0
prefix.Patch.x_lower.BCPressure.alltime.XUpper = 1.0
prefix.Patch.x_lower.BCPressure.alltime.YUpper = 1.0


prefix.Patch.y_lower.BCPressure.Type = "FluxConst"
prefix.Patch.y_lower.BCPressure.Cycle = "constant"
prefix.Patch.y_lower.BCPressure.alltime.Value = 0.0

prefix.Patch.z_lower.BCPressure.Type = "FluxConst"
prefix.Patch.z_lower.BCPressure.Cycle = "constant"
prefix.Patch.z_lower.BCPressure.alltime.Value = 0.0

prefix.Patch.x_upper.BCPressure.Type = "FluxConst"
prefix.Patch.x_upper.BCPressure.Cycle = "constant"
prefix.Patch.x_upper.BCPressure.alltime.Value = 0.0

prefix.Patch.y_upper.BCPressure.Type = "FluxConst"
prefix.Patch.y_upper.BCPressure.Cycle = "constant"
prefix.Patch.y_upper.BCPressure.alltime.Value = 0.0

prefix.Patch.z_upper.BCPressure.Type = "OverlandFlow"
prefix.Patch.z_upper.BCPressure.Cycle = "constant"
prefix.Patch.z_upper.BCPressure.alltime.Value = -0.005

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

prefix.TopoSlopesX.Type = "Constant"
prefix.TopoSlopesX.GeomNames = "domain"
prefix.TopoSlopesX.Geom.domain.Value = 0.05


# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

prefix.TopoSlopesY.Type = "Constant"
prefix.TopoSlopesY.GeomNames = "domain"
prefix.TopoSlopesY.Geom.domain.Value = -0.05

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

prefix.Mannings.Type = "Constant"
prefix.Mannings.GeomNames = "domain"
prefix.Mannings.Geom.domain.Value = 0.00005

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

prefix.PhaseSources.water.Type = "Constant"
prefix.PhaseSources.water.GeomNames = "domain"
prefix.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

prefix.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

prefix.Solver = "Richards"
prefix.Solver.MaxIter = 2500

prefix.Solver.TerrainFollowingGrid = True


prefix.Solver.Nonlinear.MaxIter = 80
prefix.Solver.Nonlinear.ResidualTol = 1e-5
prefix.Solver.Nonlinear.EtaValue = 0.001


prefix.Solver.PrintSubsurf = False
prefix.Solver.Drop = 1e-20
prefix.Solver.AbsTol = 1e-10


prefix.Solver.Nonlinear.EtaChoice = "EtaConstant"
prefix.Solver.Nonlinear.EtaValue = 0.001
prefix.Solver.Nonlinear.UseJacobian = True
prefix.Solver.Nonlinear.DerivativeEpsilon = 1e-14
prefix.Solver.Nonlinear.StepTol = 1e-25
prefix.Solver.Nonlinear.Globalization = "LineSearch"
prefix.Solver.Linear.KrylovDimension = 80
prefix.Solver.Linear.MaxRestarts = 2

prefix.Solver.Linear.Preconditioner = "MGSemi"
prefix.Solver.Linear.Preconditioner = "PFMG"
prefix.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
prefix.ICPressure.Type = "HydroStaticPatch"
prefix.ICPressure.GeomNames = "domain"
prefix.Geom.domain.ICPressure.Value = -10.0

prefix.Geom.domain.ICPressure.RefGeom = "domain"
prefix.Geom.domain.ICPressure.RefPatch = "z_upper"

prefix.Solver.Spinup = False

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

prefix.run()
