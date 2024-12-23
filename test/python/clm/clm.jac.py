# -----------------------------------------------------------------------------
# this runs CLM test case
# -----------------------------------------------------------------------------

import sys, argparse
from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path, rm
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.compare import pf_test_file
from parflow.tools.top import compute_top, extract_top

# -----------------------------------------------------------------------------
# Making output directories and copying input files
# -----------------------------------------------------------------------------

run_name = "clm"
clm_jac = Run(run_name, __file__)

new_output_dir_name = get_absolute_path("test_output/clm_jac")
mkdir(new_output_dir_name)

directories = [
    "qflx_evap_grnd",
    "eflx_lh_tot",
    "qflx_evap_tot",
    "qflx_tran_veg",
    "correct_output",
    "qflx_infl",
    "swe_out",
    "eflx_lwrad_out",
    "t_grnd",
    "diag_out",
    "qflx_evap_soi",
    "eflx_soil_grnd",
    "eflx_sh_tot",
    "qflx_evap_veg",
    "qflx_top_soil",
]

for directory in directories:
    mkdir(new_output_dir_name + "/" + directory)

cp("$PF_SRC/test/tcl/clm/drv_clmin.dat", new_output_dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegm.dat", new_output_dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegp.dat", new_output_dir_name)
cp("$PF_SRC/test/tcl/clm/narr_1hr.sc3.txt.0", new_output_dir_name)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------

clm_jac.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

clm_jac.Process.Topology.P = args.p
clm_jac.Process.Topology.Q = args.q
clm_jac.Process.Topology.R = args.r

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

clm_jac.ComputationalGrid.Lower.X = 0.0
clm_jac.ComputationalGrid.Lower.Y = 0.0
clm_jac.ComputationalGrid.Lower.Z = 0.0

clm_jac.ComputationalGrid.DX = 1000.0
clm_jac.ComputationalGrid.DY = 1000.0
clm_jac.ComputationalGrid.DZ = 0.5

clm_jac.ComputationalGrid.NX = 5
clm_jac.ComputationalGrid.NY = 5
clm_jac.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------

clm_jac.GeomInput.Names = "domain_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

clm_jac.GeomInput.domain_input.InputType = "Box"
clm_jac.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

clm_jac.Geom.domain.Lower.X = 0.0
clm_jac.Geom.domain.Lower.Y = 0.0
clm_jac.Geom.domain.Lower.Z = 0.0

clm_jac.Geom.domain.Upper.X = 5000.0
clm_jac.Geom.domain.Upper.Y = 5000.0
clm_jac.Geom.domain.Upper.Z = 5.0

clm_jac.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

clm_jac.Geom.Perm.Names = "domain"

clm_jac.Geom.domain.Perm.Type = "Constant"
clm_jac.Geom.domain.Perm.Value = 0.2

clm_jac.Perm.TensorType = "TensorByGeom"

clm_jac.Geom.Perm.TensorByGeom.Names = "domain"

clm_jac.Geom.domain.Perm.TensorValX = 1.0
clm_jac.Geom.domain.Perm.TensorValY = 1.0
clm_jac.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_jac.SpecificStorage.Type = "Constant"
clm_jac.SpecificStorage.GeomNames = "domain"
clm_jac.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm_jac.Phase.Names = "water"

clm_jac.Phase.water.Density.Type = "Constant"
clm_jac.Phase.water.Density.Value = 1.0

clm_jac.Phase.water.Viscosity.Type = "Constant"
clm_jac.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

clm_jac.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm_jac.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

clm_jac.TimingInfo.BaseUnit = 1.0
clm_jac.TimingInfo.StartCount = 0
clm_jac.TimingInfo.StartTime = 0.0
clm_jac.TimingInfo.StopTime = 5
clm_jac.TimingInfo.DumpInterval = -1
clm_jac.TimeStep.Type = "Constant"
clm_jac.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm_jac.Geom.Porosity.GeomNames = "domain"
clm_jac.Geom.domain.Porosity.Type = "Constant"
clm_jac.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

clm_jac.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------

clm_jac.Phase.water.Mobility.Type = "Constant"
clm_jac.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

clm_jac.Phase.RelPerm.Type = "VanGenuchten"
clm_jac.Phase.RelPerm.GeomNames = "domain"

clm_jac.Geom.domain.RelPerm.Alpha = 3.5
clm_jac.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm_jac.Phase.Saturation.Type = "VanGenuchten"
clm_jac.Phase.Saturation.GeomNames = "domain"

clm_jac.Geom.domain.Saturation.Alpha = 3.5
clm_jac.Geom.domain.Saturation.N = 2.0
clm_jac.Geom.domain.Saturation.SRes = 0.01
clm_jac.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

clm_jac.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

clm_jac.Cycle.Names = "constant"
clm_jac.Cycle.constant.Names = "alltime"
clm_jac.Cycle.constant.alltime.Length = 1
clm_jac.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

clm_jac.BCPressure.PatchNames = clm_jac.Geom.domain.Patches

clm_jac.Patch.x_lower.BCPressure.Type = "FluxConst"
clm_jac.Patch.x_lower.BCPressure.Cycle = "constant"
clm_jac.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.y_lower.BCPressure.Type = "FluxConst"
clm_jac.Patch.y_lower.BCPressure.Cycle = "constant"
clm_jac.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.z_lower.BCPressure.Type = "FluxConst"
clm_jac.Patch.z_lower.BCPressure.Cycle = "constant"
clm_jac.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.x_upper.BCPressure.Type = "FluxConst"
clm_jac.Patch.x_upper.BCPressure.Cycle = "constant"
clm_jac.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_jac.Patch.y_upper.BCPressure.Type = "FluxConst"
clm_jac.Patch.y_upper.BCPressure.Cycle = "constant"
clm_jac.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_jac.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm_jac.Patch.z_upper.BCPressure.Cycle = "constant"
clm_jac.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

clm_jac.TopoSlopesX.Type = "Constant"
clm_jac.TopoSlopesX.GeomNames = "domain"
clm_jac.TopoSlopesX.Geom.domain.Value = -0.001

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

clm_jac.TopoSlopesY.Type = "Constant"
clm_jac.TopoSlopesY.GeomNames = "domain"
clm_jac.TopoSlopesY.Geom.domain.Value = 0.001

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

clm_jac.Mannings.Type = "Constant"
clm_jac.Mannings.GeomNames = "domain"
clm_jac.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm_jac.PhaseSources.water.Type = "Constant"
clm_jac.PhaseSources.water.GeomNames = "domain"
clm_jac.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

clm_jac.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

clm_jac.Solver = "Richards"
clm_jac.Solver.MaxIter = 500

clm_jac.Solver.Nonlinear.MaxIter = 15
clm_jac.Solver.Nonlinear.ResidualTol = 1e-9
clm_jac.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_jac.Solver.Nonlinear.EtaValue = 0.01
clm_jac.Solver.Nonlinear.UseJacobian = True
clm_jac.Solver.Nonlinear.DerivativeEpsilon = 1e-12
clm_jac.Solver.Nonlinear.StepTol = 1e-20
clm_jac.Solver.Nonlinear.Globalization = "LineSearch"
clm_jac.Solver.Linear.KrylovDimension = 15
clm_jac.Solver.Linear.MaxRestart = 2

clm_jac.Solver.Linear.Preconditioner = "PFMG"
clm_jac.Solver.PrintSubsurf = False
clm_jac.Solver.Drop = 1e-20
clm_jac.Solver.AbsTol = 1e-9

clm_jac.Solver.LSM = "CLM"
clm_jac.Solver.CLM.MetForcing = "1D"
clm_jac.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_jac.Solver.CLM.MetFilePath = "."


# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

clm_jac.ICPressure.Type = "HydroStaticPatch"
clm_jac.ICPressure.GeomNames = "domain"
clm_jac.Geom.domain.ICPressure.Value = -2.0

clm_jac.Geom.domain.ICPressure.RefGeom = "domain"
clm_jac.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

correct_output_dir_name = get_absolute_path("../../correct_output/clm_output")
clm_jac.run(working_directory=new_output_dir_name)

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

for i in range(6):
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

mask = read_pfb(new_output_dir_name + "/clm.out.mask.pfb")
top = compute_top(mask)
write_pfb(new_output_dir_name + "/clm.out.top_index.pfb", top)

data = read_pfb(new_output_dir_name + "/clm.out.press.00000.pfb")
top_data = extract_top(data, top)
write_pfb(new_output_dir_name + "/clm.out.top.press.00000.pfb", top_data)


filename = "/clm.out.top_index.pfb"
if not pf_test_file(
    new_output_dir_name + filename,
    correct_output_dir_name + filename,
    f"Max difference in top_index",
):
    passed = False

filename = "/clm.out.top.press.00000.pfb"
if not pf_test_file(
    new_output_dir_name + filename,
    correct_output_dir_name + filename,
    f"Max difference in top_clm.out.press.00000.pfb",
):
    passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
