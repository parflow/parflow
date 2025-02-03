# ---------------------------------------------------------
# Problem to test use of indicator field.
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "indicator_field"
ifield = Run(run_name, __file__)

# ---------------------------------------------------------
# Copying files
# ---------------------------------------------------------

dir_name = get_absolute_path("test_output/ifield")
mkdir(dir_name)

cp("$PF_SRC/test/input/small_domain.pfsol", dir_name)
cp("$PF_SRC/test/input/small_domain_indicator_field.pfb", dir_name)

# ---------------------------------------------------------

ifield.FileVersion = 4

# Control use of indicator field:
# 0 = use domain
# 1 = use indicator field
useIndicatorField = 1

ifield.Process.Topology.P = 1
ifield.Process.Topology.Q = 1
ifield.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

ifield.ComputationalGrid.Lower.X = 0.0
ifield.ComputationalGrid.Lower.Y = 0.0
ifield.ComputationalGrid.Lower.Z = 0.0

ifield.ComputationalGrid.NX = 12
ifield.ComputationalGrid.NY = 12
ifield.ComputationalGrid.NZ = 12

UpperX = 440
UpperY = 120
UpperZ = 220

LowerX = ifield.ComputationalGrid.Lower.X
LowerY = ifield.ComputationalGrid.Lower.Y
LowerZ = ifield.ComputationalGrid.Lower.Z

NX = ifield.ComputationalGrid.NX
NY = ifield.ComputationalGrid.NY
NZ = ifield.ComputationalGrid.NZ

ifield.ComputationalGrid.DX = (UpperX - LowerX) / NX
ifield.ComputationalGrid.DY = (UpperY - LowerY) / NY
ifield.ComputationalGrid.DZ = (UpperZ - LowerZ) / NZ

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

ifield.GeomInput.Names = "solid_input indicator_input"

ifield.GeomInput.solid_input.InputType = "SolidFile"
ifield.GeomInput.solid_input.GeomNames = "domain"
ifield.GeomInput.solid_input.FileName = "small_domain.pfsol"

ifield.Geom.domain.Patches = (
    "infiltration z_upper x_lower y_lower x_upper y_upper z_lower"
)

ifield.GeomInput.indicator_input.InputType = "IndicatorField"
ifield.GeomInput.indicator_input.GeomNames = "indicator"
ifield.Geom.indicator_input.FileName = "small_domain_indicator_field.pfb"

ifield.GeomInput.indicator.Value = 1

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

ifield.Geom.Perm.Names = "domain"

ifield.Geom.domain.Perm.Type = "Constant"
ifield.Geom.domain.Perm.Value = 1.0

ifield.Perm.TensorType = "TensorByGeom"

ifield.Geom.Perm.TensorByGeom.Names = "domain"

ifield.Geom.domain.Perm.TensorValX = 1.0
ifield.Geom.domain.Perm.TensorValY = 1.0
ifield.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

ifield.SpecificStorage.Type = "Constant"
ifield.SpecificStorage.GeomNames = "domain"
ifield.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

ifield.Phase.Names = "water"

ifield.Phase.water.Density.Type = "Constant"
ifield.Phase.water.Density.Value = 1.0

ifield.Phase.water.Viscosity.Type = "Constant"
ifield.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

ifield.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

ifield.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

ifield.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

ifield.TimingInfo.BaseUnit = 1.0
ifield.TimingInfo.StartCount = 0
ifield.TimingInfo.StartTime = 0.0
ifield.TimingInfo.StopTime = 30.0 * 1
ifield.TimingInfo.DumpInterval = 0
ifield.TimeStep.Type = "Constant"
ifield.TimeStep.Value = 10.0
ifield.TimingInfo.DumpAtEnd = True

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

ifield.Geom.Porosity.GeomNames = "domain"
ifield.Geom.domain.Porosity.Type = "Constant"
ifield.Geom.domain.Porosity.Value = 0.3680

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

ifield.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

ifield.Phase.RelPerm.Type = "VanGenuchten"

if useIndicatorField == 1:
    ifield.Phase.RelPerm.GeomNames = "indicator"
    ifield.Geom.indicator.RelPerm.Alpha = 3.34
    ifield.Geom.indicator.RelPerm.N = 1.982
else:
    ifield.Phase.RelPerm.GeomNames = "domain"
    ifield.Geom.domain.RelPerm.Alpha = 3.34
    ifield.Geom.domain.RelPerm.N = 1.982


# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

ifield.Phase.Saturation.Type = "VanGenuchten"
ifield.Phase.Saturation.GeomNames = "domain"

ifield.Geom.domain.Saturation.Alpha = 3.34
ifield.Geom.domain.Saturation.N = 1.982
ifield.Geom.domain.Saturation.SRes = 0.2771
ifield.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

ifield.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

ifield.Cycle.Names = "constant onoff"
ifield.Cycle.constant.Names = "alltime"
ifield.Cycle.constant.alltime.Length = 1
ifield.Cycle.constant.Repeat = -1

ifield.Cycle.onoff.Names = "on off"
ifield.Cycle.onoff.on.Length = 10
ifield.Cycle.onoff.off.Length = 90
ifield.Cycle.onoff.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

ifield.BCPressure.PatchNames = ifield.Geom.domain.Patches

ifield.Patch.infiltration.BCPressure.Type = "FluxConst"
ifield.Patch.infiltration.BCPressure.Cycle = "constant"
ifield.Patch.infiltration.BCPressure.alltime.Value = -0.10
ifield.Patch.infiltration.BCPressure.off.Value = 0.0

ifield.Patch.x_lower.BCPressure.Type = "FluxConst"
ifield.Patch.x_lower.BCPressure.Cycle = "constant"
ifield.Patch.x_lower.BCPressure.alltime.Value = 0.0

ifield.Patch.y_lower.BCPressure.Type = "FluxConst"
ifield.Patch.y_lower.BCPressure.Cycle = "constant"
ifield.Patch.y_lower.BCPressure.alltime.Value = 0.0

ifield.Patch.z_lower.BCPressure.Type = "FluxConst"
ifield.Patch.z_lower.BCPressure.Cycle = "constant"
ifield.Patch.z_lower.BCPressure.alltime.Value = 0.0

ifield.Patch.x_upper.BCPressure.Type = "FluxConst"
ifield.Patch.x_upper.BCPressure.Cycle = "constant"
ifield.Patch.x_upper.BCPressure.alltime.Value = 0.0

ifield.Patch.y_upper.BCPressure.Type = "FluxConst"
ifield.Patch.y_upper.BCPressure.Cycle = "constant"
ifield.Patch.y_upper.BCPressure.alltime.Value = 0.0

ifield.Patch.z_upper.BCPressure.Type = "FluxConst"
ifield.Patch.z_upper.BCPressure.Cycle = "constant"
ifield.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

ifield.TopoSlopesX.Type = "Constant"
ifield.TopoSlopesX.GeomNames = ""
ifield.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

ifield.TopoSlopesY.Type = "Constant"
ifield.TopoSlopesY.GeomNames = ""
ifield.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

ifield.Mannings.Type = "Constant"
ifield.Mannings.GeomNames = ""
ifield.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

ifield.ICPressure.Type = "HydroStaticPatch"
ifield.ICPressure.GeomNames = "domain"

ifield.Geom.domain.ICPressure.Value = 1.0
ifield.Geom.domain.ICPressure.RefPatch = "z_lower"
ifield.Geom.domain.ICPressure.RefGeom = "domain"

ifield.Geom.infiltration.ICPressure.Value = 10.0
ifield.Geom.infiltration.ICPressure.RefPatch = "infiltration"
ifield.Geom.infiltration.ICPressure.RefGeom = "domain"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

ifield.PhaseSources.water.Type = "Constant"
ifield.PhaseSources.water.GeomNames = "domain"
ifield.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

ifield.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

ifield.Solver = "Richards"
ifield.Solver.MaxIter = 1

ifield.Solver.Nonlinear.MaxIter = 15
ifield.Solver.Nonlinear.ResidualTol = 1e-9
ifield.Solver.Nonlinear.StepTol = 1e-9
ifield.Solver.Nonlinear.EtaValue = 1e-5
ifield.Solver.Nonlinear.UseJacobian = True
ifield.Solver.Nonlinear.DerivativeEpsilon = 1e-7

ifield.Solver.Linear.KrylovDimension = 25
ifield.Solver.Linear.MaxRestarts = 2

ifield.Solver.Linear.Preconditioner = "MGSemi"
ifield.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
ifield.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

ifield.dist(dir_name + "/small_domain_indicator_field.pfb")
correct_output_dir_name = get_absolute_path("../correct_output")
ifield.run(working_directory=dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z", "porosity"]


for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(
        dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in {test_file}",
    ):
        passed = False

for i in range(2):
    timestep = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(
        dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timestep {timestep}",
    ):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(
        dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timestep {timestep}",
    ):
        passed = False


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
