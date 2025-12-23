#  Testing restart using PFB and NetCDF files

import sys, argparse, shutil, os, glob
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs

run_name = "default_richards"

default_richards = Run(run_name, __file__)


def remove_restart_files():
    """
    Remove restart check files between the restart tests.
    """
    for path in glob.glob("default_richards_restart.out.press.*") + glob.glob(
        "default_richards_restart.out.satur.*"
    ):
        os.remove(path)


default_richards.FileVersion = 4

# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

default_richards.Process.Topology.P = args.p
default_richards.Process.Topology.Q = args.q
default_richards.Process.Topology.R = args.r

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
default_richards.ComputationalGrid.Lower.X = -10.0
default_richards.ComputationalGrid.Lower.Y = 10.0
default_richards.ComputationalGrid.Lower.Z = 1.0

default_richards.ComputationalGrid.DX = 8.8888888888888893
default_richards.ComputationalGrid.DY = 10.666666666666666
default_richards.ComputationalGrid.DZ = 1.0

default_richards.ComputationalGrid.NX = 18
default_richards.ComputationalGrid.NY = 15
default_richards.ComputationalGrid.NZ = 8

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
default_richards.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------
default_richards.GeomInput.domain_input.InputType = "Box"
default_richards.GeomInput.domain_input.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------
default_richards.Geom.domain.Lower.X = -10.0
default_richards.Geom.domain.Lower.Y = 10.0
default_richards.Geom.domain.Lower.Z = 1.0

default_richards.Geom.domain.Upper.X = 150.0
default_richards.Geom.domain.Upper.Y = 170.0
default_richards.Geom.domain.Upper.Z = 9.0

default_richards.Geom.domain.Patches = "left right front back bottom top"

# ---------------------------------------------------------
# Background Geometry Input
# ---------------------------------------------------------
default_richards.GeomInput.background_input.InputType = "Box"
default_richards.GeomInput.background_input.GeomName = "background"

# ---------------------------------------------------------
# Background Geometry
# ---------------------------------------------------------
default_richards.Geom.background.Lower.X = -99999999.0
default_richards.Geom.background.Lower.Y = -99999999.0
default_richards.Geom.background.Lower.Z = -99999999.0

default_richards.Geom.background.Upper.X = 99999999.0
default_richards.Geom.background.Upper.Y = 99999999.0
default_richards.Geom.background.Upper.Z = 99999999.0


# ---------------------------------------------------------
# Source_Region Geometry Input
# ---------------------------------------------------------
default_richards.GeomInput.source_region_input.InputType = "Box"
default_richards.GeomInput.source_region_input.GeomName = "source_region"

# ---------------------------------------------------------
# Source_Region Geometry
# ---------------------------------------------------------
default_richards.Geom.source_region.Lower.X = 65.56
default_richards.Geom.source_region.Lower.Y = 79.34
default_richards.Geom.source_region.Lower.Z = 4.5

default_richards.Geom.source_region.Upper.X = 74.44
default_richards.Geom.source_region.Upper.Y = 89.99
default_richards.Geom.source_region.Upper.Z = 5.5


# ---------------------------------------------------------
# Concen_Region Geometry Input
# ---------------------------------------------------------
default_richards.GeomInput.concen_region_input.InputType = "Box"
default_richards.GeomInput.concen_region_input.GeomName = "concen_region"

# ---------------------------------------------------------
# Concen_Region Geometry
# ---------------------------------------------------------
default_richards.Geom.concen_region.Lower.X = 60.0
default_richards.Geom.concen_region.Lower.Y = 80.0
default_richards.Geom.concen_region.Lower.Z = 4.0

default_richards.Geom.concen_region.Upper.X = 80.0
default_richards.Geom.concen_region.Upper.Y = 100.0
default_richards.Geom.concen_region.Upper.Z = 6.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
default_richards.Geom.Perm.Names = "background"

default_richards.Geom.background.Perm.Type = "Constant"
default_richards.Geom.background.Perm.Value = 4.0

default_richards.Perm.TensorType = "TensorByGeom"

default_richards.Geom.Perm.TensorByGeom.Names = "background"

default_richards.Geom.background.Perm.TensorValX = 1.0
default_richards.Geom.background.Perm.TensorValY = 1.0
default_richards.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

default_richards.SpecificStorage.Type = "Constant"
default_richards.SpecificStorage.GeomNames = "domain"
default_richards.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

default_richards.Phase.Names = "water"

default_richards.Phase.water.Density.Type = "Constant"
default_richards.Phase.water.Density.Value = 1.0

default_richards.Phase.water.Viscosity.Type = "Constant"
default_richards.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
default_richards.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
default_richards.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

default_richards.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

timestep = 0.001

default_richards.TimingInfo.BaseUnit = 1.0
default_richards.TimingInfo.StartCount = 0
default_richards.TimingInfo.StartTime = 0.0
default_richards.TimingInfo.StopTime = 0.05
default_richards.TimingInfo.DumpInterval = -1
default_richards.TimeStep.Type = "Constant"
default_richards.TimeStep.Value = timestep

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

default_richards.Geom.Porosity.GeomNames = "background"

default_richards.Geom.background.Porosity.Type = "Constant"
default_richards.Geom.background.Porosity.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
default_richards.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

default_richards.Phase.RelPerm.Type = "VanGenuchten"
default_richards.Phase.RelPerm.GeomNames = "domain"
default_richards.Geom.domain.RelPerm.Alpha = 0.005
default_richards.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

default_richards.Phase.Saturation.Type = "VanGenuchten"
default_richards.Phase.Saturation.GeomNames = "domain"
default_richards.Geom.domain.Saturation.Alpha = 0.005
default_richards.Geom.domain.Saturation.N = 2.0
default_richards.Geom.domain.Saturation.SRes = 0.2
default_richards.Geom.domain.Saturation.SSat = 0.99

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
default_richards.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
default_richards.Cycle.Names = "constant"
default_richards.Cycle.constant.Names = "alltime"
default_richards.Cycle.constant.alltime.Length = 1
default_richards.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
default_richards.BCPressure.PatchNames = "left right front back bottom top"

default_richards.Patch.left.BCPressure.Type = "DirEquilRefPatch"
default_richards.Patch.left.BCPressure.Cycle = "constant"
default_richards.Patch.left.BCPressure.RefGeom = "domain"
default_richards.Patch.left.BCPressure.RefPatch = "bottom"
default_richards.Patch.left.BCPressure.alltime.Value = 5.0

default_richards.Patch.right.BCPressure.Type = "DirEquilRefPatch"
default_richards.Patch.right.BCPressure.Cycle = "constant"
default_richards.Patch.right.BCPressure.RefGeom = "domain"
default_richards.Patch.right.BCPressure.RefPatch = "bottom"
default_richards.Patch.right.BCPressure.alltime.Value = 3.0

default_richards.Patch.front.BCPressure.Type = "FluxConst"
default_richards.Patch.front.BCPressure.Cycle = "constant"
default_richards.Patch.front.BCPressure.alltime.Value = 0.0

default_richards.Patch.back.BCPressure.Type = "FluxConst"
default_richards.Patch.back.BCPressure.Cycle = "constant"
default_richards.Patch.back.BCPressure.alltime.Value = 0.0

default_richards.Patch.bottom.BCPressure.Type = "FluxConst"
default_richards.Patch.bottom.BCPressure.Cycle = "constant"
default_richards.Patch.bottom.BCPressure.alltime.Value = 0.0

default_richards.Patch.top.BCPressure.Type = "FluxConst"
default_richards.Patch.top.BCPressure.Cycle = "constant"
default_richards.Patch.top.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

default_richards.TopoSlopesX.Type = "Constant"
default_richards.TopoSlopesX.GeomNames = ""

default_richards.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

default_richards.TopoSlopesY.Type = "Constant"
default_richards.TopoSlopesY.GeomNames = ""

default_richards.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

default_richards.Mannings.Type = "Constant"
default_richards.Mannings.GeomNames = ""
default_richards.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

default_richards.ICPressure.Type = "HydroStaticPatch"
default_richards.ICPressure.GeomNames = "domain"
default_richards.Geom.domain.ICPressure.Value = 3.0
default_richards.Geom.domain.ICPressure.RefGeom = "domain"
default_richards.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

default_richards.PhaseSources.water.Type = "Constant"
default_richards.PhaseSources.water.GeomNames = "background"
default_richards.PhaseSources.water.Geom.background.Value = 0.0


# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

default_richards.KnownSolution = "NoKnownSolution"


# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
default_richards.Solver = "Richards"
default_richards.Solver.MaxIter = 50

default_richards.Solver.Nonlinear.MaxIter = 10
default_richards.Solver.Nonlinear.ResidualTol = 1e-9
default_richards.Solver.Nonlinear.EtaChoice = "EtaConstant"
default_richards.Solver.Nonlinear.EtaValue = 1e-5
default_richards.Solver.Nonlinear.UseJacobian = True
default_richards.Solver.Nonlinear.DerivativeEpsilon = 1e-2

default_richards.Solver.Linear.KrylovDimension = 10

default_richards.Solver.Linear.Preconditioner = "PFMG"

default_richards.Solver.PrintVelocities = True

# -----------------------------------------------------------------------------
# Write pressure as NetCDF for testing restart
# -----------------------------------------------------------------------------
default_richards.NetCDF.WritePressure = True
default_richards.NetCDF.NumStepsPerFile = 25

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/default_richards")
mkdir(new_output_dir_name)
correct_output_dir_name = get_absolute_path("../correct_output")
default_richards.run(working_directory=new_output_dir_name)

#
# Tests
#
# source pftest.tcl
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
for i in range(51):
    timeindex = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timeindex {timeindex}",
    ):
        passed = False

    filename = f"/{run_name}.out.satur.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timeindex {timeindex}",
    ):
        passed = False

    filename = f"/{run_name}.out.vely.{timeindex}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in y-velocity for timeindex {timeindex}",
        abs_value,
        sig_digits,
    ):
        passed = False

    filename = f"/{run_name}.out.vely.{timeindex}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in z-velocity for timeindex {timeindex}",
        abs_value,
        sig_digits,
    ):
        passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)


passed = 1
remove_restart_files()

# Restart run from step 10
istep = 10
initial_pressure = "initial_pressure.pfb"
shutil.copy(
    os.path.join(new_output_dir_name, "default_richards.out.press.00010.pfb"),
    os.path.join(new_output_dir_name, initial_pressure),
)

default_richards.dist(os.path.join(new_output_dir_name, initial_pressure))

default_richards.Solver.PrintInitialConditions = False

default_richards.ICPressure.Type = "PFBFile"
default_richards.ICPressure.GeomNames = "domain"
default_richards.Geom.domain.ICPressure.FileName = initial_pressure
default_richards.Geom.domain.ICPressure.RefGeom = "domain"
default_richards.Geom.domain.ICPressure.RefPatch = "bottom"

default_richards.TimingInfo.StartCount = istep
default_richards.TimingInfo.StartTime = istep * timestep

run_name = "default_richards_restart"

default_richards.set_name(run_name)

default_richards.run(working_directory=new_output_dir_name)

for i in range(11, 51):
    timeindex = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timeindex {timeindex}",
    ):
        passed = False

    filename = f"/{run_name}.out.satur.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timeindex {timeindex}",
    ):
        passed = False

# Remove the temporary initial pressure for the restart
# file delete -force $initial_pressure
if passed:
    print("default_richards_restart PFB : PASSED")
else:
    print("default_richards_restart PFB : FAILED")
    sys.exit(1)

# Restart run from step 10

remove_restart_files()

passed = True

istep = 10
initial_pressure = "default_richards.out.00001.nc"

default_richards.Solver.PrintInitialConditions = False

default_richards.ICPressure.Type = "NCFile"
default_richards.ICPressure.GeomNames = "domain"
default_richards.Geom.domain.ICPressure.FileName = initial_pressure
default_richards.Geom.domain.ICPressure.TimeStep = 9
default_richards.Geom.domain.ICPressure.RefGeom = "domain"
default_richards.Geom.domain.ICPressure.RefPatch = "bottom"


default_richards.TimingInfo.StartCount = istep
default_richards.TimingInfo.StartTime = istep * timestep

default_richards.run(working_directory=new_output_dir_name)

for i in range(11, 51):
    timeindex = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timeindex {timeindex}",
    ):
        passed = False

    filename = f"/{run_name}.out.satur.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timeindex {timeindex}",
    ):
        passed = False

# Remove the temporary initial pressure for the restart
# file delete -force $initial_pressure
if passed:
    print("default_richards_restart NetCDF : PASSED")
else:
    print("default_richards_restart NetCDF : FAILED")
    sys.exit(1)

# # Restart run from step last timestep in a file.  Indices in the
# # NetCDF files may be negative values to index from last
# # timestep.  This is similar to Python vector indexing.
# # Note we still have to know the initial timestep (istep here)
# # index but the negative index feature makes restartng
# # from NetCDF files a bit easier.

remove_restart_files()

passed = True

istep = 25
initial_pressure = "default_richards.out.00001.nc"

default_richards.Solver.PrintInitialConditions = False

default_richards.ICPressure.Type = "NCFile"
default_richards.ICPressure.GeomNames = "domain"
default_richards.Geom.domain.ICPressure.FileName = initial_pressure
default_richards.Geom.domain.ICPressure.TimeStep = -1
default_richards.Geom.domain.ICPressure.RefGeom = "domain"
default_richards.Geom.domain.ICPressure.RefPatch = "bottom"


default_richards.TimingInfo.StartCount = istep
default_richards.TimingInfo.StartTime = istep * timestep

default_richards.run(working_directory=new_output_dir_name)

for i in range(25, 51):
    timeindex = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timeindex {timeindex}",
    ):
        passed = False

    filename = f"/{run_name}.out.satur.{timeindex}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timeindex {timeindex}",
    ):
        passed = False

if passed:
    print("default_richards_restart NetCDF negative index: PASSED")
else:
    print("default_richards_restart NetCDF negative index: FAILED")
    sys.exit(1)
