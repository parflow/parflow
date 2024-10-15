# ---------------------------------------------------------
#  This runs a Little Washita test problem with variable dz
#  and adjusts surface pressures
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "LW_surface_press"

LW_surface_press = Run(run_name, __file__)

# ---------------------------------------------------------
# Copying slope files
# ---------------------------------------------------------
new_output_dir_name = get_absolute_path("test_output/LW_surface_press")
mkdir(new_output_dir_name)

cp("../../test/input/lw.1km.slope_x.10x.pfb", new_output_dir_name)
cp("../../test/input/lw.1km.slope_y.10x.pfb", new_output_dir_name)

# ---------------------------------------------------------

LW_surface_press.FileVersion = 4

LW_surface_press.Process.Topology.P = 1
LW_surface_press.Process.Topology.Q = 1
LW_surface_press.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

LW_surface_press.ComputationalGrid.Lower.X = 0.0
LW_surface_press.ComputationalGrid.Lower.Y = 0.0
LW_surface_press.ComputationalGrid.Lower.Z = 0.0

LW_surface_press.ComputationalGrid.NX = 45
LW_surface_press.ComputationalGrid.NY = 32
LW_surface_press.ComputationalGrid.NZ = 25
LW_surface_press.ComputationalGrid.NZ = 10
LW_surface_press.ComputationalGrid.NZ = 6

LW_surface_press.ComputationalGrid.DX = 1000.0
LW_surface_press.ComputationalGrid.DY = 1000.0
# "native" grid resolution is 2m everywhere X NZ=25 for 50m
# computational domain.
LW_surface_press.ComputationalGrid.DZ = 2.0

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

LW_surface_press.GeomInput.Names = "domaininput"

LW_surface_press.GeomInput.domaininput.GeomName = "domain"
LW_surface_press.GeomInput.domaininput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

LW_surface_press.Geom.domain.Lower.X = 0.0
LW_surface_press.Geom.domain.Lower.Y = 0.0
LW_surface_press.Geom.domain.Lower.Z = 0.0
#
LW_surface_press.Geom.domain.Upper.X = 45000.0
LW_surface_press.Geom.domain.Upper.Y = 32000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
LW_surface_press.Geom.domain.Upper.Z = 12.0
LW_surface_press.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# --------------------------------------------
# variable dz assignments
# --------------------------------------------

LW_surface_press.Solver.Nonlinear.VariableDz = True
LW_surface_press.dzScale.GeomNames = "domain"
LW_surface_press.dzScale.Type = "nzList"
LW_surface_press.dzScale.nzListNumber = 6

LW_surface_press.Cell._0.dzScale.Value = 1.0
LW_surface_press.Cell._1.dzScale.Value = 1.00
LW_surface_press.Cell._2.dzScale.Value = 1.000
LW_surface_press.Cell._3.dzScale.Value = 1.000
LW_surface_press.Cell._4.dzScale.Value = 1.000
LW_surface_press.Cell._5.dzScale.Value = 0.05

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

LW_surface_press.Geom.Perm.Names = "domain"

# Values in m/hour

LW_surface_press.Geom.domain.Perm.Type = "Constant"
LW_surface_press.Geom.domain.Perm.Value = 0.1

LW_surface_press.Geom.domain.Perm.LambdaX = 5000.0
LW_surface_press.Geom.domain.Perm.LambdaY = 5000.0
LW_surface_press.Geom.domain.Perm.LambdaZ = 50.0
LW_surface_press.Geom.domain.Perm.GeomMean = 0.0001427686

LW_surface_press.Geom.domain.Perm.Sigma = 0.20
LW_surface_press.Geom.domain.Perm.Sigma = 1.20
LW_surface_press.Geom.domain.Perm.NumLines = 150
LW_surface_press.Geom.domain.Perm.RZeta = 10.0
LW_surface_press.Geom.domain.Perm.KMax = 100.0000001
LW_surface_press.Geom.domain.Perm.DelK = 0.2
LW_surface_press.Geom.domain.Perm.Seed = 33333
LW_surface_press.Geom.domain.Perm.LogNormal = "Log"
LW_surface_press.Geom.domain.Perm.StratType = "Bottom"

LW_surface_press.Perm.TensorType = "TensorByGeom"

LW_surface_press.Geom.Perm.TensorByGeom.Names = "domain"

LW_surface_press.Geom.domain.Perm.TensorValX = 1.0
LW_surface_press.Geom.domain.Perm.TensorValY = 1.0
LW_surface_press.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

LW_surface_press.SpecificStorage.Type = "Constant"
LW_surface_press.SpecificStorage.GeomNames = "domain"
LW_surface_press.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

LW_surface_press.Phase.Names = "water"

LW_surface_press.Phase.water.Density.Type = "Constant"
LW_surface_press.Phase.water.Density.Value = 1.0

LW_surface_press.Phase.water.Viscosity.Type = "Constant"
LW_surface_press.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

LW_surface_press.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

LW_surface_press.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

LW_surface_press.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

LW_surface_press.TimingInfo.BaseUnit = 10.0
LW_surface_press.TimingInfo.StartCount = 0
LW_surface_press.TimingInfo.StartTime = 0.0
LW_surface_press.TimingInfo.StopTime = 1000.0
LW_surface_press.TimingInfo.DumpInterval = 100.0
LW_surface_press.TimeStep.Type = "Constant"
LW_surface_press.TimeStep.Value = 100.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

LW_surface_press.Geom.Porosity.GeomNames = "domain"
LW_surface_press.Geom.domain.Porosity.Type = "Constant"
LW_surface_press.Geom.domain.Porosity.Value = 0.25

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

LW_surface_press.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

LW_surface_press.Phase.RelPerm.Type = "VanGenuchten"
LW_surface_press.Phase.RelPerm.GeomNames = "domain"

LW_surface_press.Geom.domain.RelPerm.Alpha = 1.0
LW_surface_press.Geom.domain.RelPerm.Alpha = 1.0
LW_surface_press.Geom.domain.RelPerm.N = 3.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

LW_surface_press.Phase.Saturation.Type = "VanGenuchten"
LW_surface_press.Phase.Saturation.GeomNames = "domain"

LW_surface_press.Geom.domain.Saturation.Alpha = 1.0
LW_surface_press.Geom.domain.Saturation.Alpha = 1.0
LW_surface_press.Geom.domain.Saturation.N = 3.0
LW_surface_press.Geom.domain.Saturation.SRes = 0.1
LW_surface_press.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

LW_surface_press.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

LW_surface_press.Cycle.Names = "constant rainrec"
LW_surface_press.Cycle.Names = "constant"
LW_surface_press.Cycle.constant.Names = "alltime"
LW_surface_press.Cycle.constant.alltime.Length = 10000000
LW_surface_press.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

LW_surface_press.Cycle.rainrec.Names = "rain rec"
LW_surface_press.Cycle.rainrec.rain.Length = 10
LW_surface_press.Cycle.rainrec.rec.Length = 20
LW_surface_press.Cycle.rainrec.Repeat = 14
#
# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

LW_surface_press.BCPressure.PatchNames = LW_surface_press.Geom.domain.Patches

LW_surface_press.Patch.x_lower.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.x_lower.BCPressure.Cycle = "constant"
LW_surface_press.Patch.x_lower.BCPressure.alltime.Value = 0.0

LW_surface_press.Patch.y_lower.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.y_lower.BCPressure.Cycle = "constant"
LW_surface_press.Patch.y_lower.BCPressure.alltime.Value = 0.0

LW_surface_press.Patch.z_lower.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.z_lower.BCPressure.Cycle = "constant"
LW_surface_press.Patch.z_lower.BCPressure.alltime.Value = 0.0

LW_surface_press.Patch.x_upper.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.x_upper.BCPressure.Cycle = "constant"
LW_surface_press.Patch.x_upper.BCPressure.alltime.Value = 0.0

LW_surface_press.Patch.y_upper.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.y_upper.BCPressure.Cycle = "constant"
LW_surface_press.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
LW_surface_press.Patch.z_upper.BCPressure.Type = "FluxConst"
LW_surface_press.Patch.z_upper.BCPressure.Cycle = "constant"
# constant recharge at 100 mm / y
LW_surface_press.Patch.z_upper.BCPressure.alltime.Value = -0.005
LW_surface_press.Patch.z_upper.BCPressure.alltime.Value = -0.0001

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

LW_surface_press.TopoSlopesX.Type = "PFBFile"
LW_surface_press.TopoSlopesX.GeomNames = "domain"
LW_surface_press.TopoSlopesX.FileName = "lw.1km.slope_x.10x.pfb"

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

LW_surface_press.TopoSlopesY.Type = "PFBFile"
LW_surface_press.TopoSlopesY.GeomNames = "domain"
LW_surface_press.TopoSlopesY.FileName = "lw.1km.slope_y.10x.pfb"

# ---------------------------------------------------------
#  Distribute slopes
# ---------------------------------------------------------

LW_surface_press.dist(new_output_dir_name + "/lw.1km.slope_x.10x.pfb")
LW_surface_press.dist(new_output_dir_name + "/lw.1km.slope_y.10x.pfb")

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

LW_surface_press.Mannings.Type = "Constant"
LW_surface_press.Mannings.GeomNames = "domain"
LW_surface_press.Mannings.Geom.domain.Value = 0.00005

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

LW_surface_press.PhaseSources.water.Type = "Constant"
LW_surface_press.PhaseSources.water.GeomNames = "domain"
LW_surface_press.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

LW_surface_press.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

LW_surface_press.Solver = "Richards"
LW_surface_press.Solver.MaxIter = 2500

LW_surface_press.Solver.TerrainFollowingGrid = True

LW_surface_press.Solver.Nonlinear.MaxIter = 80
LW_surface_press.Solver.Nonlinear.ResidualTol = 1e-5
LW_surface_press.Solver.Nonlinear.EtaValue = 0.001

LW_surface_press.Solver.PrintSubsurf = False
LW_surface_press.Solver.Drop = 1e-20
LW_surface_press.Solver.AbsTol = 1e-10

LW_surface_press.Solver.Nonlinear.EtaChoice = "EtaConstant"
LW_surface_press.Solver.Nonlinear.EtaValue = 0.001
LW_surface_press.Solver.Nonlinear.UseJacobian = True
LW_surface_press.Solver.Nonlinear.StepTol = 1e-25
LW_surface_press.Solver.Nonlinear.Globalization = "LineSearch"
LW_surface_press.Solver.Linear.KrylovDimension = 80
LW_surface_press.Solver.Linear.MaxRestarts = 2

LW_surface_press.Solver.Linear.Preconditioner = "MGSemi"


##---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
LW_surface_press.ICPressure.Type = "HydroStaticPatch"
LW_surface_press.ICPressure.GeomNames = "domain"
LW_surface_press.Geom.domain.ICPressure.Value = 0.0

LW_surface_press.Geom.domain.ICPressure.RefGeom = "domain"
LW_surface_press.Geom.domain.ICPressure.RefPatch = "z_upper"

LW_surface_press.Solver.ResetSurfacePressure = True
LW_surface_press.Solver.ResetSurfacePressure.ThresholdPressure = 10.0
LW_surface_press.Solver.ResetSurfacePressure.ResetPressure = -0.00001

# -----------------------------------------------------------------------------
# Run and do tests
# -----------------------------------------------------------------------------
correct_output_dir_name = get_absolute_path("../correct_output")
LW_surface_press.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in {test_file}",
    ):
        passed = False

for i in range(0, 11):
    timestep = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timestep {timestep}",
    ):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timestep {timestep}",
    ):
        passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
