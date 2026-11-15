# -----------------------------------------------------------------------------
# MGSemi preconditioner test for overland flow with tilted V
# running different configurations of tilted V, OverlandKinematic, OverlandFlow and OverlandDiffusive
# with and without analytical jacobian and with MGSemi preconditioner
# The results should be almost identical for the different configurations
# The test is to check the correctness of the MGSemi preconditioner
# The analytical jacobian cases also run with PCMatrixType PFSymmetric, which
# checks the symmetric MGSemi matrix with overland flow (it previously failed)
# RMM, 2024-07-02
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
import os


def check_output(run_name, correct_root, correct_output_dir_name):

    passed = True
    sig_digits = 4
    abs_value = 1e-12
    test_files = ["press"]
    for timestep in range(20):
        for test_file in test_files:
            filename = f"/{run_name}.out.{test_file}.{timestep:05d}.pfb"
            correct_filename = f"/{correct_root}.out.{test_file}.{timestep:05d}.pfb"
            result = pf_test_file_with_abs(
                new_output_dir_name + filename,
                correct_output_dir_name + correct_filename,
                f"Max difference in {new_output_dir_name + filename}",
                abs_value,
                sig_digits,
            )
            if not result:
                passed = False

    return passed


overland = Run("overland_tiltedV_KWE", __file__)

# -----------------------------------------------------------------------------

overland.FileVersion = 4

overland.Process.Topology.P = 1
overland.Process.Topology.Q = 1
overland.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = 5
overland.ComputationalGrid.NY = 5
overland.ComputationalGrid.NZ = 1

overland.ComputationalGrid.DX = 10.0
overland.ComputationalGrid.DY = 10.0
overland.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

overland.GeomInput.Names = "domaininput leftinput rightinput channelinput"

overland.GeomInput.domaininput.GeomName = "domain"
overland.GeomInput.leftinput.GeomName = "left"
overland.GeomInput.rightinput.GeomName = "right"
overland.GeomInput.channelinput.GeomName = "channel"

overland.GeomInput.domaininput.InputType = "Box"
overland.GeomInput.leftinput.InputType = "Box"
overland.GeomInput.rightinput.InputType = "Box"
overland.GeomInput.channelinput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

overland.Geom.domain.Lower.X = 0.0
overland.Geom.domain.Lower.Y = 0.0
overland.Geom.domain.Lower.Z = 0.0

overland.Geom.domain.Upper.X = 50.0
overland.Geom.domain.Upper.Y = 50.0
overland.Geom.domain.Upper.Z = 0.05
overland.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# ---------------------------------------------------------
# Left Slope Geometry
# ---------------------------------------------------------

overland.Geom.left.Lower.X = 0.0
overland.Geom.left.Lower.Y = 0.0
overland.Geom.left.Lower.Z = 0.0

overland.Geom.left.Upper.X = 20.0
overland.Geom.left.Upper.Y = 50.0
overland.Geom.left.Upper.Z = 0.05

# ---------------------------------------------------------
# Right Slope Geometry
# ---------------------------------------------------------

overland.Geom.right.Lower.X = 30.0
overland.Geom.right.Lower.Y = 0.0
overland.Geom.right.Lower.Z = 0.0

overland.Geom.right.Upper.X = 50.0
overland.Geom.right.Upper.Y = 50.0
overland.Geom.right.Upper.Z = 0.05

# ---------------------------------------------------------
# Channel Geometry
# ---------------------------------------------------------

overland.Geom.channel.Lower.X = 20.0
overland.Geom.channel.Lower.Y = 0.0
overland.Geom.channel.Lower.Z = 0.0

overland.Geom.channel.Upper.X = 30.0
overland.Geom.channel.Upper.Y = 50.0
overland.Geom.channel.Upper.Z = 0.05

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

overland.Geom.Perm.Names = "domain"
overland.Geom.domain.Perm.Type = "Constant"
overland.Geom.domain.Perm.Value = 0.0000001

overland.Perm.TensorType = "TensorByGeom"

overland.Geom.Perm.TensorByGeom.Names = "domain"

overland.Geom.domain.Perm.TensorValX = 1.0
overland.Geom.domain.Perm.TensorValY = 1.0
overland.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

overland.SpecificStorage.Type = "Constant"
overland.SpecificStorage.GeomNames = "domain"
overland.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

overland.Phase.Names = "water"

overland.Phase.water.Density.Type = "Constant"
overland.Phase.water.Density.Value = 1.0

overland.Phase.water.Viscosity.Type = "Constant"
overland.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

overland.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

overland.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

overland.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
overland.TimingInfo.BaseUnit = 0.05
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = 2.0
overland.TimingInfo.DumpInterval = -2
overland.TimeStep.Type = "Constant"
overland.TimeStep.Value = 0.05

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

overland.Geom.Porosity.GeomNames = "domain"
overland.Geom.domain.Porosity.Type = "Constant"
overland.Geom.domain.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

overland.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

overland.Phase.RelPerm.Type = "VanGenuchten"
overland.Phase.RelPerm.GeomNames = "domain"

overland.Geom.domain.RelPerm.Alpha = 6.0
overland.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

overland.Phase.Saturation.Type = "VanGenuchten"
overland.Phase.Saturation.GeomNames = "domain"

overland.Geom.domain.Saturation.Alpha = 6.0
overland.Geom.domain.Saturation.N = 2.0
overland.Geom.domain.Saturation.SRes = 0.2
overland.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

overland.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

overland.Cycle.Names = "constant rainrec"
overland.Cycle.constant.Names = "alltime"
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland.Cycle.rainrec.Names = "rain rec"
overland.Cycle.rainrec.rain.Length = 2
overland.Cycle.rainrec.rec.Length = 300
overland.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

overland.BCPressure.PatchNames = overland.Geom.domain.Patches

overland.Patch.x_lower.BCPressure.Type = "FluxConst"
overland.Patch.x_lower.BCPressure.Cycle = "constant"
overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland.Patch.y_lower.BCPressure.Type = "FluxConst"
overland.Patch.y_lower.BCPressure.Cycle = "constant"
overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland.Patch.z_lower.BCPressure.Type = "FluxConst"
overland.Patch.z_lower.BCPressure.Cycle = "constant"
overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland.Patch.x_upper.BCPressure.Type = "FluxConst"
overland.Patch.x_upper.BCPressure.Cycle = "constant"
overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland.Patch.y_upper.BCPressure.Type = "FluxConst"
overland.Patch.y_upper.BCPressure.Cycle = "constant"
overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall
overland.Patch.z_upper.BCPressure.Type = "OverlandFlow"
overland.Patch.z_upper.BCPressure.Cycle = "rainrec"
overland.Patch.z_upper.BCPressure.rain.Value = -0.01
overland.Patch.z_upper.BCPressure.rec.Value = 0.0000

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

overland.Mannings.Type = "Constant"
overland.Mannings.GeomNames = "domain"
overland.Mannings.Geom.domain.Value = 3.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

overland.PhaseSources.water.Type = "Constant"
overland.PhaseSources.water.GeomNames = "domain"
overland.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

overland.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

overland.Solver = "Richards"
overland.Solver.MaxIter = 2500

overland.Solver.Nonlinear.MaxIter = 100
overland.Solver.Nonlinear.ResidualTol = 1e-9
overland.Solver.Nonlinear.EtaChoice = "EtaConstant"
overland.Solver.Nonlinear.EtaValue = 0.01
overland.Solver.Nonlinear.UseJacobian = False
overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland.Solver.Nonlinear.StepTol = 1e-20
overland.Solver.Nonlinear.Globalization = "LineSearch"
overland.Solver.Linear.KrylovDimension = 50
overland.Solver.Linear.MaxRestart = 2
overland.Solver.OverlandKinematic.Epsilon = 1e-5

overland.Solver.Linear.Preconditioner = "MGSemi"
overland.Solver.PrintSubsurf = False
overland.Solver.Drop = 1e-20
overland.Solver.AbsTol = 1e-10

overland.Solver.WriteSiloSubsurfData = False
overland.Solver.WriteSiloPressure = False
overland.Solver.WriteSiloSlopes = False

overland.Solver.WriteSiloSaturation = False
overland.Solver.WriteSiloConcentration = False

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland.ICPressure.Type = "HydroStaticPatch"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.Value = -3.0

overland.Geom.domain.ICPressure.RefGeom = "domain"
overland.Geom.domain.ICPressure.RefPatch = "z_upper"


# -----------------------------------------------------------------------------
# New kinematic formulations without the zero channel
# Note: The difference in configuration here is to be consistent with the way
# the upwinding is handled for the new and original formulations.
# These two results should be almost identical for the new and old formulations
# -----------------------------------------------------------------------------
overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "left right channel"
overland.TopoSlopesX.Geom.left.Value = -0.01
overland.TopoSlopesX.Geom.right.Value = 0.01
overland.TopoSlopesX.Geom.channel.Value = 0.01

overland.TopoSlopesY.Type = "Constant"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.Geom.domain.Value = 0.01

# run with KWE upwinding
overland.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland.Solver.Nonlinear.UseJacobian = False
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

passed = True

correct_output_dir_name = get_absolute_path(
    "../correct_output/TiltedV_OverlandKin_MGSemi"
)
correct_root = "TiltedV_OverlandKin"

run_name = "TiltedV_OverlandKin_JacFalse_MGSemi"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
test_dir_name = get_absolute_path("test_output/")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

run_name = "TiltedV_OverlandKin_JacTrue_MGSemi"
overland.set_name(run_name)
# run with KWE upwinding and analytical jacobian
overland.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland.Solver.Nonlinear.UseJacobian = True
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

print("##########")
print(f"Running {run_name} Jacobian True")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

# run with KWE upwinding, analytical jacobian and the symmetric preconditioner
# matrix (PFSymmetric, the default PCMatrixType); should match the same result
run_name = "TiltedV_OverlandKin_JacTrue_MGSemi_PFSymmetric"
overland.set_name(run_name)
overland.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

print("##########")
print(f"Running {run_name} Jacobian True PFSymmetric")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"


# -----------------------------------------------------------------------------
# Original formulation with a zero value channel
# -----------------------------------------------------------------------------

correct_output_dir_name = get_absolute_path(
    f"{correct_output_dir_name}/../TiltedV_OverlandFlow_MGSemi"
)
correct_root = "TiltedV_OverlandFlow"
## need to change the channel slopes for Overland Flow which is cell-centered
overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "left right channel"
overland.TopoSlopesX.Geom.left.Value = -0.01
overland.TopoSlopesX.Geom.right.Value = 0.01
overland.TopoSlopesX.Geom.channel.Value = 0.00
overland.TopoSlopesY.Type = "Constant"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.Geom.domain.Value = 0.01
overland.Patch.z_upper.BCPressure.Type = "OverlandFlow"
run_name = "TiltedV_OverlandFlow_JacTrue_MGSemi"
overland.set_name(run_name)
overland.Solver.Linear.Preconditioner = "MGSemi"
print("##########")
print(f"Running {run_name} Jacobian True MGSemi")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

run_name = "TiltedV_OverlandFlow_JacTrue_MGSemi_PFSymmetric"
overland.set_name(run_name)
overland.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"
print("##########")
print(f"Running {run_name} Jacobian True MGSemi PFSymmetric")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

run_name = "TiltedV_OverlandFlow_JacFalse_MGSemi"
overland.set_name(run_name)
overland.Solver.Nonlinear.UseJacobian = False
print("##########")
print(f"Running {run_name} Jacobian False MGSemi")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False


# -----------------------------------------------------------------------------
# Diffusive wave (DWE) formulation without the zero channel
# Note: The difference in configuration here is to be consistent with the way
# the upwinding is handled for the new and original formulations.
# Tests with Jacobian True and False and run for both MGSemi preconditioner
# Results should be almost identical for the different configurations
# -----------------------------------------------------------------------------
overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "left right channel"
overland.TopoSlopesX.Geom.left.Value = -0.01
overland.TopoSlopesX.Geom.right.Value = 0.01
overland.TopoSlopesX.Geom.channel.Value = 0.01

overland.TopoSlopesY.Type = "Constant"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.Geom.domain.Value = 0.01

# run with DWE Finite Difference Jacobian
overland.Patch.z_upper.BCPressure.Type = "OverlandDiffusive"
overland.Solver.Nonlinear.UseJacobian = False

correct_output_dir_name = get_absolute_path(
    f"{correct_output_dir_name}/../TiltedV_OverlandDiff_MGSemi"
)
correct_root = "TiltedV_OverlandDiff"

run_name = "TiltedV_OverlandDiff_JacFalse_MGSemi"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

# run with DWE and analytical jacobian
run_name = "TiltedV_OverlandDiff_JacTrue_MGSemi"
overland.set_name(run_name)
overland.Solver.Nonlinear.UseJacobian = True
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

print("##########")
print(f"Running {run_name} Jacobian True")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

# run with DWE, analytical jacobian and the symmetric preconditioner matrix
run_name = "TiltedV_OverlandDiff_JacTrue_MGSemi_PFSymmetric"
overland.set_name(run_name)
overland.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

print("##########")
print(f"Running {run_name} Jacobian True PFSymmetric")
new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
mkdir(new_output_dir_name)
overland.run(working_directory=new_output_dir_name)
if not check_output(run_name, correct_root, correct_output_dir_name):
    passed = False

# -----------------------------------------------------------------------------
# Symmetric preconditioner matrix on a larger grid
# On the 5 x 5 x 1 grid above GMRES converges even with a poor preconditioner.
# Here MGSemi with PCMatrixType PFSymmetric must match the FullJacobian result
# and keep GMRES below its iteration limit (KrylovDimension).  Before the fix,
# the symmetric MGSemi matrix took the overland west/south terms into the
# neighbours' east/north coefficients, and about 20 of 47 Newton steps hit
# the limit here; with the fix none do.
# -----------------------------------------------------------------------------
NX_big = 30
NZ_big = 10
dz_big = overland.ComputationalGrid.DZ
overland.ComputationalGrid.NX = NX_big
overland.ComputationalGrid.NY = NX_big
overland.ComputationalGrid.NZ = NZ_big
L_big = NX_big * overland.ComputationalGrid.DX
x_left = (NX_big // 2 - 1) * overland.ComputationalGrid.DX
x_right = (NX_big // 2 + 1) * overland.ComputationalGrid.DX
for geom_name, (x_lower, x_upper) in {
    "domain": (0.0, L_big),
    "left": (0.0, x_left),
    "channel": (x_left, x_right),
    "right": (x_right, L_big),
}.items():
    overland.Geom[geom_name].Lower.X = x_lower
    overland.Geom[geom_name].Upper.X = x_upper
    overland.Geom[geom_name].Upper.Y = L_big
    overland.Geom[geom_name].Upper.Z = NZ_big * dz_big

overland.TopoSlopesX.Geom.channel.Value = 0.01
overland.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland.Solver.Nonlinear.UseJacobian = True
overland.Solver.Nonlinear.PrintFlag = "HighVerbosity"
overland.TimingInfo.BaseUnit = 0.25
overland.TimeStep.Value = 0.25
overland.TimingInfo.StopTime = 2.0

big_runs = {}
for matrix_type in ["FullJacobian", "PFSymmetric"]:
    run_name = f"TiltedV_OverlandKin_JacTrue_MGSemi_{matrix_type}_30x30x10"
    overland.set_name(run_name)
    overland.Solver.Linear.Preconditioner.PCMatrixType = matrix_type
    print("##########")
    print(f"Running {run_name}")
    new_output_dir_name = os.path.join(test_dir_name, f"{run_name}")
    mkdir(new_output_dir_name)
    overland.run(working_directory=new_output_dir_name)
    big_runs[matrix_type] = (run_name, new_output_dir_name)

full_name, full_dir = big_runs["FullJacobian"]
sym_name, sym_dir = big_runs["PFSymmetric"]
for timestep in range(5):
    if not pf_test_file_with_abs(
        f"{sym_dir}/{sym_name}.out.press.{timestep:05d}.pfb",
        f"{full_dir}/{full_name}.out.press.{timestep:05d}.pfb",
        f"Max difference in {sym_name} press {timestep:05d}",
        1e-12,
        4,
    ):
        passed = False

krylov_limit = int(overland.Solver.Linear.KrylovDimension)
with open(f"{sym_dir}/{sym_name}.out.kinsol.log") as kinsol_log:
    gmres_its = [
        int(line.split("=")[1])
        for line in kinsol_log
        if line.startswith("KINSpgmrSolve: nli_inc=")
    ]
at_limit = sum(its >= krylov_limit for its in gmres_its)
print(
    f"{sym_name}: {len(gmres_its)} Newton steps, {at_limit} reached the "
    f"GMRES limit of {krylov_limit}"
)
if len(gmres_its) == 0 or at_limit > 2:
    print(f"{sym_name}: too many Newton steps reached the GMRES limit")
    passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
