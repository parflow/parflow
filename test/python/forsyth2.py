# -----------------------------------------------------------------------------
#  This runs Problem 2 in the paper
#     "Robust Numerical Methods for Saturated-Unsaturated Flow with
#      Dry Initial Conditions", Forsyth, Wu and Pruess,
#      Advances in Water Resources, 1995.
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path
from parflow.tools import settings
from parflow.tools.compare import pf_test_file

run_name = "forsyth2"
forsyth2 = Run(run_name, __file__)

# ---------------------------------------------------------
# Copy solid file
# ---------------------------------------------------------

dir_name = get_absolute_path("test_output/forsyth2")
mkdir(dir_name)

cp("$PF_SRC/test/input/fors2_hf.pfsol", dir_name)

# ---------------------------------------------------------

forsyth2.FileVersion = 4

forsyth2.Process.Topology.P = 1
forsyth2.Process.Topology.Q = 1
forsyth2.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

forsyth2.ComputationalGrid.Lower.X = 0.0
forsyth2.ComputationalGrid.Lower.Y = 0.0
forsyth2.ComputationalGrid.Lower.Z = 0.0

forsyth2.ComputationalGrid.NX = 96
forsyth2.ComputationalGrid.NY = 1
forsyth2.ComputationalGrid.NZ = 67

UpperX = 800.0
UpperY = 1.0
UpperZ = 650.0

LowerX = forsyth2.ComputationalGrid.Lower.X
LowerY = forsyth2.ComputationalGrid.Lower.Y
LowerZ = forsyth2.ComputationalGrid.Lower.Z

NX = forsyth2.ComputationalGrid.NX
NY = forsyth2.ComputationalGrid.NY
NZ = forsyth2.ComputationalGrid.NZ

forsyth2.ComputationalGrid.DX = (UpperX - LowerX) / NX
forsyth2.ComputationalGrid.DY = (UpperY - LowerY) / NY
forsyth2.ComputationalGrid.DZ = (UpperZ - LowerZ) / NZ

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

Zones = "zone1 zone2 zone3above4 zone3left4 zone3right4 zone3below4 zone4"

forsyth2.GeomInput.Names = f"solidinput {Zones} background"

forsyth2.GeomInput.solidinput.InputType = "SolidFile"
forsyth2.GeomInput.solidinput.GeomNames = "domain"
forsyth2.GeomInput.solidinput.FileName = "fors2_hf.pfsol"

forsyth2.GeomInput.zone1.InputType = "Box"
forsyth2.GeomInput.zone1.GeomName = "zone1"

forsyth2.Geom.zone1.Lower.X = 0.0
forsyth2.Geom.zone1.Lower.Y = 0.0
forsyth2.Geom.zone1.Lower.Z = 610.0
forsyth2.Geom.zone1.Upper.X = 800.0
forsyth2.Geom.zone1.Upper.Y = 1.0
forsyth2.Geom.zone1.Upper.Z = 650.0

forsyth2.GeomInput.zone2.InputType = "Box"
forsyth2.GeomInput.zone2.GeomName = "zone2"

forsyth2.Geom.zone2.Lower.X = 0.0
forsyth2.Geom.zone2.Lower.Y = 0.0
forsyth2.Geom.zone2.Lower.Z = 560.0
forsyth2.Geom.zone2.Upper.X = 800.0
forsyth2.Geom.zone2.Upper.Y = 1.0
forsyth2.Geom.zone2.Upper.Z = 610.0

forsyth2.GeomInput.zone3above4.InputType = "Box"
forsyth2.GeomInput.zone3above4.GeomName = "zone3above4"

forsyth2.Geom.zone3above4.Lower.X = 0.0
forsyth2.Geom.zone3above4.Lower.Y = 0.0
forsyth2.Geom.zone3above4.Lower.Z = 500.0
forsyth2.Geom.zone3above4.Upper.X = 800.0
forsyth2.Geom.zone3above4.Upper.Y = 1.0
forsyth2.Geom.zone3above4.Upper.Z = 560.0

forsyth2.GeomInput.zone3left4.InputType = "Box"
forsyth2.GeomInput.zone3left4.GeomName = "zone3left4"

forsyth2.Geom.zone3left4.Lower.X = 0.0
forsyth2.Geom.zone3left4.Lower.Y = 0.0
forsyth2.Geom.zone3left4.Lower.Z = 400.0
forsyth2.Geom.zone3left4.Upper.X = 100.0
forsyth2.Geom.zone3left4.Upper.Y = 1.0
forsyth2.Geom.zone3left4.Upper.Z = 500.0

forsyth2.GeomInput.zone3right4.InputType = "Box"
forsyth2.GeomInput.zone3right4.GeomName = "zone3right4"

forsyth2.Geom.zone3right4.Lower.X = 300.0
forsyth2.Geom.zone3right4.Lower.Y = 0.0
forsyth2.Geom.zone3right4.Lower.Z = 400.0
forsyth2.Geom.zone3right4.Upper.X = 800.0
forsyth2.Geom.zone3right4.Upper.Y = 1.0
forsyth2.Geom.zone3right4.Upper.Z = 500.0

forsyth2.GeomInput.zone3below4.InputType = "Box"
forsyth2.GeomInput.zone3below4.GeomName = "zone3below4"

forsyth2.Geom.zone3below4.Lower.X = 0.0
forsyth2.Geom.zone3below4.Lower.Y = 0.0
forsyth2.Geom.zone3below4.Lower.Z = 0.0
forsyth2.Geom.zone3below4.Upper.X = 800.0
forsyth2.Geom.zone3below4.Upper.Y = 1.0
forsyth2.Geom.zone3below4.Upper.Z = 400.0

forsyth2.GeomInput.zone4.InputType = "Box"
forsyth2.GeomInput.zone4.GeomName = "zone4"

forsyth2.Geom.zone4.Lower.X = 100.0
forsyth2.Geom.zone4.Lower.Y = 0.0
forsyth2.Geom.zone4.Lower.Z = 400.0
forsyth2.Geom.zone4.Upper.X = 300.0
forsyth2.Geom.zone4.Upper.Y = 1.0
forsyth2.Geom.zone4.Upper.Z = 500.0

forsyth2.GeomInput.background.InputType = "Box"
forsyth2.GeomInput.background.GeomName = "background"

forsyth2.Geom.background.Lower.X = -99999999.0
forsyth2.Geom.background.Lower.Y = -99999999.0
forsyth2.Geom.background.Lower.Z = -99999999.0
forsyth2.Geom.background.Upper.X = 99999999.0
forsyth2.Geom.background.Upper.Y = 99999999.0
forsyth2.Geom.background.Upper.Z = 99999999.0

forsyth2.Geom.domain.Patches = (
    "infiltration z_upper x_lower y_lower x_upper y_upper z_lower"
)

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

forsyth2.Geom.Perm.Names = Zones

# Values in cm^2

forsyth2.Geom.zone1.Perm.Type = "Constant"
forsyth2.Geom.zone1.Perm.Value = 9.1496e-5

forsyth2.Geom.zone2.Perm.Type = "Constant"
forsyth2.Geom.zone2.Perm.Value = 5.4427e-5

forsyth2.Geom.zone3above4.Perm.Type = "Constant"
forsyth2.Geom.zone3above4.Perm.Value = 4.8033e-5

forsyth2.Geom.zone3left4.Perm.Type = "Constant"
forsyth2.Geom.zone3left4.Perm.Value = 4.8033e-5

forsyth2.Geom.zone3right4.Perm.Type = "Constant"
forsyth2.Geom.zone3right4.Perm.Value = 4.8033e-5

forsyth2.Geom.zone3below4.Perm.Type = "Constant"
forsyth2.Geom.zone3below4.Perm.Value = 4.8033e-5

forsyth2.Geom.zone4.Perm.Type = "Constant"
forsyth2.Geom.zone4.Perm.Value = 4.8033e-4

forsyth2.Perm.TensorType = "TensorByGeom"

forsyth2.Geom.Perm.TensorByGeom.Names = "background"

forsyth2.Geom.background.Perm.TensorValX = 1.0
forsyth2.Geom.background.Perm.TensorValY = 1.0
forsyth2.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

forsyth2.SpecificStorage.Type = "Constant"
forsyth2.SpecificStorage.GeomNames = "domain"
forsyth2.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

forsyth2.Phase.Names = "water"

forsyth2.Phase.water.Density.Type = "Constant"
forsyth2.Phase.water.Density.Value = 1.0

forsyth2.Phase.water.Viscosity.Type = "Constant"
forsyth2.Phase.water.Viscosity.Value = 1.124e-2

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

forsyth2.Contaminants.Names = "tce"
forsyth2.Contaminants.tce.Degradation.Value = 0.0

forsyth2.PhaseConcen.water.tce.Type = "Constant"
forsyth2.PhaseConcen.water.tce.GeomNames = "domain"
forsyth2.PhaseConcen.water.tce.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

forsyth2.Geom.Retardation.GeomNames = "background"
forsyth2.Geom.background.tce.Retardation.Type = "Linear"
forsyth2.Geom.background.tce.Retardation.Rate = 0.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

forsyth2.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

forsyth2.TimingInfo.BaseUnit = 1.0
forsyth2.TimingInfo.StartCount = 0
forsyth2.TimingInfo.StartTime = 0.0
forsyth2.TimingInfo.StopTime = 2592000.0
forsyth2.TimingInfo.StopTime = 8640.0

forsyth2.TimingInfo.DumpInterval = -1
forsyth2.TimeStep.Type = "Constant"
forsyth2.TimeStep.Value = 8640.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

forsyth2.Geom.Porosity.GeomNames = Zones

forsyth2.Geom.zone1.Porosity.Type = "Constant"
forsyth2.Geom.zone1.Porosity.Value = 0.3680

forsyth2.Geom.zone2.Porosity.Type = "Constant"
forsyth2.Geom.zone2.Porosity.Value = 0.3510

forsyth2.Geom.zone3above4.Porosity.Type = "Constant"
forsyth2.Geom.zone3above4.Porosity.Value = 0.3250

forsyth2.Geom.zone3left4.Porosity.Type = "Constant"
forsyth2.Geom.zone3left4.Porosity.Value = 0.3250

forsyth2.Geom.zone3right4.Porosity.Type = "Constant"
forsyth2.Geom.zone3right4.Porosity.Value = 0.3250

forsyth2.Geom.zone3below4.Porosity.Type = "Constant"
forsyth2.Geom.zone3below4.Porosity.Value = 0.3250

forsyth2.Geom.zone4.Porosity.Type = "Constant"
forsyth2.Geom.zone4.Porosity.Value = 0.3250

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

forsyth2.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

forsyth2.Phase.RelPerm.Type = "VanGenuchten"
forsyth2.Phase.RelPerm.GeomNames = Zones

forsyth2.Geom.zone1.RelPerm.Alpha = 0.0334
forsyth2.Geom.zone1.RelPerm.N = 1.982

forsyth2.Geom.zone2.RelPerm.Alpha = 0.0363
forsyth2.Geom.zone2.RelPerm.N = 1.632

forsyth2.Geom.zone3above4.RelPerm.Alpha = 0.0345
forsyth2.Geom.zone3above4.RelPerm.N = 1.573

forsyth2.Geom.zone3left4.RelPerm.Alpha = 0.0345
forsyth2.Geom.zone3left4.RelPerm.N = 1.573

forsyth2.Geom.zone3right4.RelPerm.Alpha = 0.0345
forsyth2.Geom.zone3right4.RelPerm.N = 1.573

forsyth2.Geom.zone3below4.RelPerm.Alpha = 0.0345
forsyth2.Geom.zone3below4.RelPerm.N = 1.573

forsyth2.Geom.zone4.RelPerm.Alpha = 0.0345
forsyth2.Geom.zone4.RelPerm.N = 1.573

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

forsyth2.Phase.Saturation.Type = "VanGenuchten"
forsyth2.Phase.Saturation.GeomNames = Zones

forsyth2.Geom.zone1.Saturation.Alpha = 0.0334
forsyth2.Geom.zone1.Saturation.N = 1.982
forsyth2.Geom.zone1.Saturation.SRes = 0.2771
forsyth2.Geom.zone1.Saturation.SSat = 1.0

forsyth2.Geom.zone2.Saturation.Alpha = 0.0363
forsyth2.Geom.zone2.Saturation.N = 1.632
forsyth2.Geom.zone2.Saturation.SRes = 0.2806
forsyth2.Geom.zone2.Saturation.SSat = 1.0

forsyth2.Geom.zone3above4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone3above4.Saturation.N = 1.573
forsyth2.Geom.zone3above4.Saturation.SRes = 0.2643
forsyth2.Geom.zone3above4.Saturation.SSat = 1.0

forsyth2.Geom.zone3left4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone3left4.Saturation.N = 1.573
forsyth2.Geom.zone3left4.Saturation.SRes = 0.2643
forsyth2.Geom.zone3left4.Saturation.SSat = 1.0

forsyth2.Geom.zone3right4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone3right4.Saturation.N = 1.573
forsyth2.Geom.zone3right4.Saturation.SRes = 0.2643
forsyth2.Geom.zone3right4.Saturation.SSat = 1.0

forsyth2.Geom.zone3below4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone3below4.Saturation.N = 1.573
forsyth2.Geom.zone3below4.Saturation.SRes = 0.2643
forsyth2.Geom.zone3below4.Saturation.SSat = 1.0

forsyth2.Geom.zone3below4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone3below4.Saturation.N = 1.573
forsyth2.Geom.zone3below4.Saturation.SRes = 0.2643
forsyth2.Geom.zone3below4.Saturation.SSat = 1.0

forsyth2.Geom.zone4.Saturation.Alpha = 0.0345
forsyth2.Geom.zone4.Saturation.N = 1.573
forsyth2.Geom.zone4.Saturation.SRes = 0.2643
forsyth2.Geom.zone4.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

forsyth2.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

forsyth2.Cycle.Names = "constant"
forsyth2.Cycle.constant.Names = "alltime"
forsyth2.Cycle.constant.alltime.Length = 1
forsyth2.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

forsyth2.BCPressure.PatchNames = forsyth2.Geom.domain.Patches

forsyth2.Patch.infiltration.BCPressure.Type = "FluxConst"
forsyth2.Patch.infiltration.BCPressure.Cycle = "constant"
forsyth2.Patch.infiltration.BCPressure.alltime.Value = -2.3148e-5

forsyth2.Patch.x_lower.BCPressure.Type = "FluxConst"
forsyth2.Patch.x_lower.BCPressure.Cycle = "constant"
forsyth2.Patch.x_lower.BCPressure.alltime.Value = 0.0

forsyth2.Patch.y_lower.BCPressure.Type = "FluxConst"
forsyth2.Patch.y_lower.BCPressure.Cycle = "constant"
forsyth2.Patch.y_lower.BCPressure.alltime.Value = 0.0

forsyth2.Patch.z_lower.BCPressure.Type = "FluxConst"
forsyth2.Patch.z_lower.BCPressure.Cycle = "constant"
forsyth2.Patch.z_lower.BCPressure.alltime.Value = 0.0

forsyth2.Patch.x_upper.BCPressure.Type = "FluxConst"
forsyth2.Patch.x_upper.BCPressure.Cycle = "constant"
forsyth2.Patch.x_upper.BCPressure.alltime.Value = 0.0

forsyth2.Patch.y_upper.BCPressure.Type = "FluxConst"
forsyth2.Patch.y_upper.BCPressure.Cycle = "constant"
forsyth2.Patch.y_upper.BCPressure.alltime.Value = 0.0

forsyth2.Patch.z_upper.BCPressure.Type = "FluxConst"
forsyth2.Patch.z_upper.BCPressure.Cycle = "constant"
forsyth2.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

forsyth2.TopoSlopesX.Type = "Constant"
forsyth2.TopoSlopesX.GeomNames = "domain"

forsyth2.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

forsyth2.TopoSlopesY.Type = "Constant"
forsyth2.TopoSlopesY.GeomNames = "domain"

forsyth2.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

forsyth2.Mannings.Type = "Constant"
forsyth2.Mannings.GeomNames = "domain"
forsyth2.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

forsyth2.ICPressure.Type = "Constant"
forsyth2.ICPressure.GeomNames = "domain"
forsyth2.Geom.domain.ICPressure.Value = -734.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

forsyth2.PhaseSources.water.Type = "Constant"
forsyth2.PhaseSources.water.GeomNames = "background"
forsyth2.PhaseSources.water.Geom.background.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

forsyth2.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

forsyth2.Solver = "Richards"
forsyth2.Solver.MaxIter = 10000

forsyth2.Solver.Nonlinear.MaxIter = 15
forsyth2.Solver.Nonlinear.ResidualTol = 1e-9
forsyth2.Solver.Nonlinear.StepTol = 1e-9
forsyth2.Solver.Nonlinear.UseJacobian = True
forsyth2.Solver.Nonlinear.DerivativeEpsilon = 1e-7

forsyth2.Solver.Linear.KrylovDimension = 25
forsyth2.Solver.Linear.MaxRestarts = 2

forsyth2.Solver.Linear.Preconditioner = "MGSemi"
forsyth2.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
forsyth2.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = dir_name
correct_output_dir_name = get_absolute_path("../correct_output")
forsyth2.run(working_directory=dir_name)

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

for i in range(2):
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


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
