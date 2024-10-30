import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs

run_name = "Dirichlet"
dirichlet = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
dirichlet.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

dirichlet.Process.Topology.P = 1
dirichlet.Process.Topology.Q = 1
dirichlet.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
dirichlet.ComputationalGrid.Lower.X = 0.0
dirichlet.ComputationalGrid.Lower.Y = 0.0
dirichlet.ComputationalGrid.Lower.Z = 0.0

dirichlet.ComputationalGrid.DX = 1.0
dirichlet.ComputationalGrid.DY = 1.0
dirichlet.ComputationalGrid.DZ = 0.1

dirichlet.ComputationalGrid.NX = 1
dirichlet.ComputationalGrid.NY = 1
dirichlet.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
dirichlet.GeomInput.Names = "domain_input"


# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
dirichlet.GeomInput.domain_input.InputType = "Box"
dirichlet.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
dirichlet.Geom.domain.Lower.X = 0.0
dirichlet.Geom.domain.Lower.Y = 0.0
dirichlet.Geom.domain.Lower.Z = 0.0

dirichlet.Geom.domain.Upper.X = 1.0
dirichlet.Geom.domain.Upper.Y = 1.0
dirichlet.Geom.domain.Upper.Z = 1.0

dirichlet.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
dirichlet.Geom.Perm.Names = "domain"

dirichlet.Geom.domain.Perm.Type = "Constant"
dirichlet.Geom.domain.Perm.Value = 4.0

dirichlet.Perm.TensorType = "TensorByGeom"

dirichlet.Geom.Perm.TensorByGeom.Names = "domain"

dirichlet.Geom.domain.Perm.TensorValX = 1.0
dirichlet.Geom.domain.Perm.TensorValY = 1.0
dirichlet.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

dirichlet.SpecificStorage.Type = "Constant"
dirichlet.SpecificStorage.GeomNames = ""
dirichlet.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

dirichlet.Phase.Names = "water"

dirichlet.Phase.water.Density.Type = "Constant"
dirichlet.Phase.water.Density.Value = 1.0

dirichlet.Phase.water.Viscosity.Type = "Constant"
dirichlet.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
dirichlet.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
dirichlet.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

dirichlet.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

dirichlet.TimingInfo.BaseUnit = 1.0
dirichlet.TimingInfo.StartCount = 0
dirichlet.TimingInfo.StartTime = 0.0
dirichlet.TimingInfo.StopTime = 1.0
dirichlet.TimingInfo.DumpInterval = -1
dirichlet.TimeStep.Type = "Constant"
dirichlet.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

dirichlet.Geom.Porosity.GeomNames = "domain"

dirichlet.Geom.domain.Porosity.Type = "Constant"
dirichlet.Geom.domain.Porosity.Value = 0.5

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
dirichlet.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

dirichlet.Phase.RelPerm.Type = "VanGenuchten"
dirichlet.Phase.RelPerm.GeomNames = "domain"
dirichlet.Geom.domain.RelPerm.Alpha = 1.0
dirichlet.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

dirichlet.Phase.Saturation.Type = "VanGenuchten"
dirichlet.Phase.Saturation.GeomNames = "domain"
dirichlet.Geom.domain.Saturation.Alpha = 1.0
dirichlet.Geom.domain.Saturation.N = 2.0
dirichlet.Geom.domain.Saturation.SRes = 0.2
dirichlet.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
dirichlet.Wells.Names = ""


# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
dirichlet.Cycle.Names = "constant"
dirichlet.Cycle.constant.Names = "alltime"
dirichlet.Cycle.constant.alltime.Length = 1
dirichlet.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
dirichlet.BCPressure.PatchNames = "left right front back bottom top"

dirichlet.Patch.left.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.left.BCPressure.Cycle = "constant"
dirichlet.Patch.left.BCPressure.RefGeom = "domain"
dirichlet.Patch.left.BCPressure.RefPatch = "bottom"
dirichlet.Patch.left.BCPressure.alltime.Value = 0.5

dirichlet.Patch.right.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.right.BCPressure.Cycle = "constant"
dirichlet.Patch.right.BCPressure.RefGeom = "domain"
dirichlet.Patch.right.BCPressure.RefPatch = "bottom"
dirichlet.Patch.right.BCPressure.alltime.Value = 0.5

dirichlet.Patch.front.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.front.BCPressure.Cycle = "constant"
dirichlet.Patch.front.BCPressure.RefGeom = "domain"
dirichlet.Patch.front.BCPressure.RefPatch = "bottom"
dirichlet.Patch.front.BCPressure.alltime.Value = 0.5

dirichlet.Patch.back.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.back.BCPressure.Cycle = "constant"
dirichlet.Patch.back.BCPressure.RefGeom = "domain"
dirichlet.Patch.back.BCPressure.RefPatch = "bottom"
dirichlet.Patch.back.BCPressure.alltime.Value = 0.5

dirichlet.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.bottom.BCPressure.Cycle = "constant"
dirichlet.Patch.bottom.BCPressure.RefGeom = "domain"
dirichlet.Patch.bottom.BCPressure.RefPatch = "bottom"
dirichlet.Patch.bottom.BCPressure.alltime.Value = 0.5

dirichlet.Patch.top.BCPressure.Type = "DirEquilRefPatch"
dirichlet.Patch.top.BCPressure.Cycle = "constant"
dirichlet.Patch.top.BCPressure.RefGeom = "domain"
dirichlet.Patch.top.BCPressure.RefPatch = "top"
dirichlet.Patch.top.BCPressure.alltime.Value = -0.5

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

dirichlet.TopoSlopesX.Type = "Constant"
dirichlet.TopoSlopesX.GeomNames = ""

dirichlet.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

dirichlet.TopoSlopesY.Type = "Constant"
dirichlet.TopoSlopesY.GeomNames = ""

dirichlet.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

dirichlet.Mannings.Type = "Constant"
dirichlet.Mannings.GeomNames = ""
dirichlet.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

dirichlet.ICPressure.Type = "HydroStaticPatch"
dirichlet.ICPressure.GeomNames = "domain"
dirichlet.Geom.domain.ICPressure.Value = 0.5
dirichlet.Geom.domain.ICPressure.RefGeom = "domain"
dirichlet.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

dirichlet.PhaseSources.water.Type = "Constant"
dirichlet.PhaseSources.water.GeomNames = "domain"
dirichlet.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

dirichlet.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
dirichlet.Solver = "Richards"
dirichlet.Solver.MaxIter = 5

dirichlet.Solver.Nonlinear.MaxIter = 10
dirichlet.Solver.Nonlinear.ResidualTol = 1e-9
dirichlet.Solver.Nonlinear.EtaChoice = "EtaConstant"
dirichlet.Solver.Nonlinear.EtaValue = 1e-5
dirichlet.Solver.Nonlinear.UseJacobian = True
dirichlet.Solver.Nonlinear.DerivativeEpsilon = 1e-2

dirichlet.Solver.Linear.KrylovDimension = 10

dirichlet.Solver.Linear.Preconditioner = "PFMG"

dirichlet.Solver.PrintVelocities = True


# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------
new_output_dir_name = get_absolute_path("test_output/dirichlet")
correct_output_dir_name = get_absolute_path("../correct_output")
mkdir(new_output_dir_name)
dirichlet.run(working_directory=new_output_dir_name)

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

sig_digits = 6
abs_value = 1e-12
for i in range(0, 2):
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
    filename = f"/{run_name}.out.velx.{timestep}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in x-velocity for timestep {timestep}",
        abs_value,
        sig_digits,
    ):
        passed = False
    filename = f"/{run_name}.out.vely.{timestep}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in y-velocity for timestep {timestep}",
        abs_value,
        sig_digits,
    ):
        passed = False
    filename = f"/{run_name}.out.vely.{timestep}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in z-velocity for timestep {timestep}",
        abs_value,
        sig_digits,
    ):
        passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
