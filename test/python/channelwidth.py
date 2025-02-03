# Import the ParFlow package
import os
import sys
from parflow import Run, write_pfb
import shutil
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm, exists
from parflow.tools.io import read_pfb
import numpy as np

# Set our Run Name
run_name = "channel_width_example"
wc_test = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
wc_test.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
wc_test.Process.Topology.P = 1
wc_test.Process.Topology.Q = 1
wc_test.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
wc_test.ComputationalGrid.Lower.X = 0.0
wc_test.ComputationalGrid.Lower.Y = 0.0
wc_test.ComputationalGrid.Lower.Z = 0.0

wc_test.ComputationalGrid.DX = 100.0
wc_test.ComputationalGrid.DY = 2.0
wc_test.ComputationalGrid.DZ = 1.0

wc_test.ComputationalGrid.NX = 20
wc_test.ComputationalGrid.NY = 1
wc_test.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
wc_test.GeomInput.Names = "domain_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
wc_test.GeomInput.domain_input.InputType = "Box"
wc_test.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
wc_test.Geom.domain.Lower.X = 0.0
wc_test.Geom.domain.Lower.Y = 0.0
wc_test.Geom.domain.Lower.Z = 0.0

wc_test.Geom.domain.Upper.X = 2000.0
wc_test.Geom.domain.Upper.Y = 2.0
wc_test.Geom.domain.Upper.Z = 10.0

wc_test.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"


# --------------------------------------------
# Variable dz Assignments
# ------------------------------------------
wc_test.Solver.Nonlinear.VariableDz = False
wc_test.dzScale.GeomNames = "domain"
wc_test.dzScale.Type = "nzList"
wc_test.dzScale.nzListNumber = 10

# cells start at the bottom (0) and moves up to the top
# domain is 49 m thick, root zone is down to 4 cells
# so the root zone is 2 m thick
wc_test.Cell._0.dzScale.Value = 10.0  # 10* 1.0 = 10  m  layer
wc_test.Cell._1.dzScale.Value = 10.0
wc_test.Cell._2.dzScale.Value = 10.0
wc_test.Cell._3.dzScale.Value = 10.0
wc_test.Cell._4.dzScale.Value = 5.0
wc_test.Cell._5.dzScale.Value = 1.0
wc_test.Cell._6.dzScale.Value = 1.0
wc_test.Cell._7.dzScale.Value = 0.6  # 0.6* 1.0 = 0.6  60 cm 3rd layer
wc_test.Cell._8.dzScale.Value = 0.3  # 0.3* 1.0 = 0.3  30 cm 2nd layer
wc_test.Cell._9.dzScale.Value = 0.1  # 0.1* 1.0 = 0.1  10 cm top layer

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
wc_test.Geom.Perm.Names = "domain"
wc_test.Geom.domain.Perm.Type = "Constant"
wc_test.Geom.domain.Perm.Value = 0.01465  # m/h

wc_test.Perm.TensorType = "TensorByGeom"
wc_test.Geom.Perm.TensorByGeom.Names = "domain"
wc_test.Geom.domain.Perm.TensorValX = 1.0
wc_test.Geom.domain.Perm.TensorValY = 1.0
wc_test.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
wc_test.SpecificStorage.Type = "Constant"
wc_test.SpecificStorage.GeomNames = "domain"
wc_test.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
wc_test.Phase.Names = "water"

wc_test.Phase.water.Density.Type = "Constant"
wc_test.Phase.water.Density.Value = 1.0

wc_test.Phase.water.Viscosity.Type = "Constant"
wc_test.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
wc_test.Contaminants.Names = ""


# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
wc_test.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup Timing
# -----------------------------------------------------------------------------
wc_test.TimingInfo.BaseUnit = 1.0
wc_test.TimingInfo.StartCount = 0
wc_test.TimingInfo.StartTime = 0.0
wc_test.TimingInfo.StopTime = 5.0
wc_test.TimingInfo.DumpInterval = 1.0
wc_test.TimeStep.Type = "Constant"
wc_test.TimeStep.Value = 1.0


# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
wc_test.Geom.Porosity.GeomNames = "domain"

wc_test.Geom.domain.Porosity.Type = "Constant"
wc_test.Geom.domain.Porosity.Value = 0.25

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
wc_test.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
wc_test.Phase.water.Mobility.Type = "Constant"
wc_test.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
wc_test.Phase.RelPerm.Type = "VanGenuchten"
wc_test.Phase.RelPerm.GeomNames = "domain"

wc_test.Geom.domain.RelPerm.Alpha = 1.0
wc_test.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------
wc_test.Phase.Saturation.Type = "VanGenuchten"
wc_test.Phase.Saturation.GeomNames = "domain"

wc_test.Geom.domain.Saturation.Alpha = 1.0
wc_test.Geom.domain.Saturation.N = 2.0
wc_test.Geom.domain.Saturation.SRes = 0.2
wc_test.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
wc_test.Wells.Names = ""


# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
wc_test.Cycle.Names = "constant"
wc_test.Cycle.constant.Names = "alltime"
wc_test.Cycle.constant.alltime.Length = 1
wc_test.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
wc_test.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"

wc_test.Patch.y_lower.BCPressure.Type = "FluxConst"
wc_test.Patch.y_lower.BCPressure.Cycle = "constant"
wc_test.Patch.y_lower.BCPressure.alltime.Value = 0.0

wc_test.Patch.z_lower.BCPressure.Type = "FluxConst"
wc_test.Patch.z_lower.BCPressure.Cycle = "constant"
wc_test.Patch.z_lower.BCPressure.alltime.Value = 0.0

wc_test.Patch.x_lower.BCPressure.Type = "FluxConst"
wc_test.Patch.x_lower.BCPressure.Cycle = "constant"
wc_test.Patch.x_lower.BCPressure.alltime.Value = 0.0

wc_test.Patch.x_upper.BCPressure.Type = "DirEquilRefPatch"
wc_test.Patch.x_upper.BCPressure.RefGeom = "domain"
wc_test.Patch.x_upper.BCPressure.RefPatch = "z_upper"
wc_test.Patch.x_upper.BCPressure.Cycle = "constant"
wc_test.Patch.x_upper.BCPressure.alltime.Value = (
    -1.0
)  # ocean boundary is 1m below land surface

wc_test.Patch.y_upper.BCPressure.Type = "FluxConst"
wc_test.Patch.y_upper.BCPressure.Cycle = "constant"
wc_test.Patch.y_upper.BCPressure.alltime.Value = 0.0

wc_test.Patch.z_upper.BCPressure.Type = "OverlandFlow"
wc_test.Patch.z_upper.BCPressure.Cycle = "constant"
wc_test.Patch.z_upper.BCPressure.alltime.Value = -0.01

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
wc_test.TopoSlopesX.Type = "Constant"
wc_test.TopoSlopesX.GeomNames = "domain"
wc_test.TopoSlopesX.Geom.domain.Value = (
    -0.1
)  # slope in X-direction to allow ponded water to run off

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
wc_test.TopoSlopesY.Type = "Constant"
wc_test.TopoSlopesY.GeomNames = "domain"
wc_test.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
wc_test.Mannings.Type = "Constant"
wc_test.Mannings.GeomNames = "domain"
wc_test.Mannings.Geom.domain.Value = 2.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
wc_test.PhaseSources.water.Type = "Constant"
wc_test.PhaseSources.water.GeomNames = "domain"
wc_test.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
wc_test.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
wc_test.Solver = "Richards"
wc_test.Solver.MaxIter = 9000

wc_test.Solver.Nonlinear.MaxIter = 100
wc_test.Solver.Nonlinear.ResidualTol = 1e-5
wc_test.Solver.Nonlinear.EtaChoice = "Walker1"
wc_test.Solver.Nonlinear.EtaValue = 0.01
wc_test.Solver.Nonlinear.UseJacobian = True
wc_test.Solver.Nonlinear.DerivativeEpsilon = 1e-12
wc_test.Solver.Nonlinear.StepTol = 1e-30
wc_test.Solver.Nonlinear.Globalization = "LineSearch"
wc_test.Solver.Linear.KrylovDimension = 100
wc_test.Solver.Linear.MaxRestarts = 5
wc_test.Solver.Linear.Preconditioner = "PFMG"
wc_test.Solver.PrintSubsurf = True
wc_test.Solver.Drop = 1e-20
wc_test.Solver.AbsTol = 1e-9

# Writing output options for ParFlow
should_write = True  # only PFB output for water balance example
#  PFB  no SILO
wc_test.Solver.PrintSubsurfData = should_write
wc_test.Solver.PrintPressure = should_write
wc_test.Solver.PrintSaturation = should_write
wc_test.Solver.PrintCLM = should_write
wc_test.Solver.PrintMask = should_write
wc_test.Solver.PrintSpecificStorage = should_write
wc_test.Solver.PrintEvapTrans = should_write
wc_test.Solver.PrintVelocities = True
wc_test.Solver.PrintChannelWidth = True


# ---------------------------------------------------
# LSM / CLM options
# ---------------------------------------------------

# Writing output options for CLM
# no native CLM logs
wc_test.Solver.PrintLSMSink = False
wc_test.Solver.CLM.CLMDumpInterval = 1
wc_test.Solver.CLM.CLMFileDir = "output/"
wc_test.Solver.CLM.BinaryOutDir = False
wc_test.Solver.CLM.IstepStart = 1
wc_test.Solver.WriteCLMBinary = False
wc_test.Solver.CLM.WriteLogs = False
wc_test.Solver.CLM.WriteLastRST = True
wc_test.Solver.CLM.DailyRST = False
wc_test.Solver.CLM.SingleFile = True


# ---------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------
wc_test.ICPressure.Type = "HydroStaticPatch"
wc_test.ICPressure.GeomNames = "domain"
wc_test.Geom.domain.ICPressure.Value = -10.00
wc_test.Geom.domain.ICPressure.RefGeom = "domain"
wc_test.Geom.domain.ICPressure.RefPatch = "z_upper"

# ---------------------------------------------------------
# Test channel width with constant input
# ---------------------------------------------------------

# ---------------------------------------------------------
# Channel width in x-direction
# ---------------------------------------------------------
wc_test.Solver.Nonlinear.ChannelWidthExistX = True
wc_test.ChannelWidthX.Type = "Constant"
wc_test.ChannelWidthX.GeomNames = "domain"
wc_test.ChannelWidthX.Geom.domain.Value = 1.0

# ---------------------------------------------------------
# Channel width in y-direction
# ---------------------------------------------------------
wc_test.Solver.Nonlinear.ChannelWidthExistY = True
wc_test.ChannelWidthY.Type = "Constant"
wc_test.ChannelWidthY.GeomNames = "domain"
wc_test.ChannelWidthY.Geom.domain.Value = 1.0

# -----------------------------------------------------------------------------
# Run ParFlow
# -----------------------------------------------------------------------------
base = os.path.join(os.getcwd(), "output/constant")
mkdir(base)
print(f"base: {base}")
wc_test.run(working_directory=base)

# -----------------------------------------------------------------------------
# Check if values are as expected
# -----------------------------------------------------------------------------

data_wcx = read_pfb(os.path.join(base, "channel_width_example.out.wc_x.pfb"))
data_wcy = read_pfb(os.path.join(base, "channel_width_example.out.wc_y.pfb"))

ones_array = np.ones((1, 1, 20))

if not np.array_equal(data_wcx, ones_array) or not np.array_equal(data_wcy, ones_array):
    print(f"{run_name} : FAILED")
    sys.exit(1)

# ---------------------------------------------------------
# Test channel width with PFB input
# ---------------------------------------------------------

base = os.path.join(os.getcwd(), "output/pfb")
mkdir(base)
print(f"base: {base}")
write_pfb(get_absolute_path(base + "/Channel_Width_X.pfb"), ones_array)
write_pfb(get_absolute_path(base + "/Channel_Width_Y.pfb"), ones_array)

# ---------------------------------------------------------
# Channel width in x-direction
# ---------------------------------------------------------
wc_test.Solver.Nonlinear.ChannelWidthExistX = True
wc_test.ChannelWidthX.Type = "PFBFile"
wc_test.ChannelWidthX.GeomNames = "domain"
wc_test.ChannelWidthX.FileName = "Channel_Width_X.pfb"

# ---------------------------------------------------------
# Channel width in y-direction
# ---------------------------------------------------------
wc_test.Solver.Nonlinear.ChannelWidthExistY = True
wc_test.ChannelWidthY.Type = "PFBFile"
wc_test.ChannelWidthY.GeomNames = "domain"
wc_test.ChannelWidthY.FileName = "Channel_Width_Y.pfb"

# -----------------------------------------------------------------------------
# Run ParFlow
# -----------------------------------------------------------------------------

wc_test.run(working_directory=base)

# -----------------------------------------------------------------------------
# Check if values are as expected
# -----------------------------------------------------------------------------

data_wcx = read_pfb(os.path.join(base, "channel_width_example.out.wc_x.pfb"))
data_wcy = read_pfb(os.path.join(base, "channel_width_example.out.wc_y.pfb"))

if not np.array_equal(data_wcx, ones_array) or not np.array_equal(data_wcy, ones_array):
    print(f"{run_name} : FAILED")
    sys.exit(1)

# Do same checks as in constant case (read wc_y & y and check they are right)
