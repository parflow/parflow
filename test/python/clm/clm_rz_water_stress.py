# this runs the CLM test case
# with RZ water stress distributed over the RZ
# as a function of moisture limitation as discussed in
# Ferguson, Jefferson, et al ESS 2016
#
# this also represents some CLM best practices from the experience
# of the Maxwell group -- limited output, no CLM logs, solver settings
# to maximize runtime especially on large, parallel runs
# @R Maxwell 24-Nov-27

#
# Import the ParFlow TCL package
#
import sys, argparse

from parflow import Run
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.compare import pf_test_file
from parflow.tools.top import compute_top, extract_top

run_name = "clm_rz_water_stress"
clm_rz_water_stress = Run(run_name, __file__)

dir_name = get_absolute_path("test_output/" + run_name)
mkdir(dir_name)

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
    mkdir(dir_name + "/" + directory)

cp("$PF_SRC/test/tcl/clm/drv_clmin.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegm.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegp.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/lai.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/sai.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/z0m.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/displa.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/narr_1hr.sc3.txt.0", dir_name)
cp("$PF_SRC/test/tcl/clm/veg_map.cpfb", dir_name + "/veg_map.pfb")

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
clm_rz_water_stress.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

clm_rz_water_stress.Process.Topology.P = args.p
clm_rz_water_stress.Process.Topology.Q = args.q
clm_rz_water_stress.Process.Topology.R = args.r

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
clm_rz_water_stress.ComputationalGrid.Lower.X = 0.0
clm_rz_water_stress.ComputationalGrid.Lower.Y = 0.0
clm_rz_water_stress.ComputationalGrid.Lower.Z = 0.0

clm_rz_water_stress.ComputationalGrid.DX = 1000.0
clm_rz_water_stress.ComputationalGrid.DY = 1000.0
clm_rz_water_stress.ComputationalGrid.DZ = 0.5

clm_rz_water_stress.ComputationalGrid.NX = 5
clm_rz_water_stress.ComputationalGrid.NY = 5
clm_rz_water_stress.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
clm_rz_water_stress.GeomInput.Names = "domain_input"


# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
clm_rz_water_stress.GeomInput.domain_input.InputType = "Box"
clm_rz_water_stress.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
clm_rz_water_stress.Geom.domain.Lower.X = 0.0
clm_rz_water_stress.Geom.domain.Lower.Y = 0.0
clm_rz_water_stress.Geom.domain.Lower.Z = 0.0

clm_rz_water_stress.Geom.domain.Upper.X = 5000.0
clm_rz_water_stress.Geom.domain.Upper.Y = 5000.0
clm_rz_water_stress.Geom.domain.Upper.Z = 5.0

clm_rz_water_stress.Geom.domain.Patches = (
    "x_lower x_upper y_lower y_upper z_lower z_upper"
)

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
clm_rz_water_stress.Geom.Perm.Names = "domain"

clm_rz_water_stress.Geom.domain.Perm.Type = "Constant"
clm_rz_water_stress.Geom.domain.Perm.Value = 0.2


clm_rz_water_stress.Perm.TensorType = "TensorByGeom"

clm_rz_water_stress.Geom.Perm.TensorByGeom.Names = "domain"

clm_rz_water_stress.Geom.domain.Perm.TensorValX = 1.0
clm_rz_water_stress.Geom.domain.Perm.TensorValY = 1.0
clm_rz_water_stress.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

clm_rz_water_stress.SpecificStorage.Type = "Constant"
clm_rz_water_stress.SpecificStorage.GeomNames = "domain"
clm_rz_water_stress.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm_rz_water_stress.Phase.Names = "water"

clm_rz_water_stress.Phase.water.Density.Type = "Constant"
clm_rz_water_stress.Phase.water.Density.Value = 1.0

clm_rz_water_stress.Phase.water.Viscosity.Type = "Constant"
clm_rz_water_stress.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
clm_rz_water_stress.Contaminants.Names = ""


# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm_rz_water_stress.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
#
clm_rz_water_stress.TimingInfo.BaseUnit = 1.0
clm_rz_water_stress.TimingInfo.StartCount = 0
clm_rz_water_stress.TimingInfo.StartTime = 0.0
clm_rz_water_stress.TimingInfo.StopTime = 5
clm_rz_water_stress.TimingInfo.DumpInterval = -1
clm_rz_water_stress.TimeStep.Type = "Constant"
clm_rz_water_stress.TimeStep.Value = 1.0
#

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm_rz_water_stress.Geom.Porosity.GeomNames = "domain"

clm_rz_water_stress.Geom.domain.Porosity.Type = "Constant"
clm_rz_water_stress.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
clm_rz_water_stress.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
clm_rz_water_stress.Phase.water.Mobility.Type = "Constant"
clm_rz_water_stress.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
#
clm_rz_water_stress.Phase.RelPerm.Type = "VanGenuchten"
clm_rz_water_stress.Phase.RelPerm.GeomNames = "domain"
#
clm_rz_water_stress.Geom.domain.RelPerm.Alpha = 3.5
clm_rz_water_stress.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm_rz_water_stress.Phase.Saturation.Type = "VanGenuchten"
clm_rz_water_stress.Phase.Saturation.GeomNames = "domain"
#
clm_rz_water_stress.Geom.domain.Saturation.Alpha = 3.5
clm_rz_water_stress.Geom.domain.Saturation.N = 2.0
clm_rz_water_stress.Geom.domain.Saturation.SRes = 0.01
clm_rz_water_stress.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
clm_rz_water_stress.Wells.Names = ""


# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
clm_rz_water_stress.Cycle.Names = "constant"
clm_rz_water_stress.Cycle.constant.Names = "alltime"
clm_rz_water_stress.Cycle.constant.alltime.Length = 1
clm_rz_water_stress.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
clm_rz_water_stress.BCPressure.PatchNames = (
    "x_lower x_upper y_lower y_upper z_lower z_upper"
)
#
clm_rz_water_stress.Patch.x_lower.BCPressure.Type = "FluxConst"
clm_rz_water_stress.Patch.x_lower.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.x_lower.BCPressure.alltime.Value = 0.0
#
clm_rz_water_stress.Patch.y_lower.BCPressure.Type = "FluxConst"
clm_rz_water_stress.Patch.y_lower.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.y_lower.BCPressure.alltime.Value = 0.0
#
clm_rz_water_stress.Patch.z_lower.BCPressure.Type = "FluxConst"
clm_rz_water_stress.Patch.z_lower.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.z_lower.BCPressure.alltime.Value = 0.0
#
clm_rz_water_stress.Patch.x_upper.BCPressure.Type = "FluxConst"
clm_rz_water_stress.Patch.x_upper.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.x_upper.BCPressure.alltime.Value = 0.0
#
clm_rz_water_stress.Patch.y_upper.BCPressure.Type = "FluxConst"
clm_rz_water_stress.Patch.y_upper.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.y_upper.BCPressure.alltime.Value = 0.0
#
clm_rz_water_stress.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm_rz_water_stress.Patch.z_upper.BCPressure.Cycle = "constant"
clm_rz_water_stress.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
#
clm_rz_water_stress.TopoSlopesX.Type = "Constant"
clm_rz_water_stress.TopoSlopesX.GeomNames = "domain"
clm_rz_water_stress.TopoSlopesX.Geom.domain.Value = -0.001
#
# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
#
clm_rz_water_stress.TopoSlopesY.Type = "Constant"
clm_rz_water_stress.TopoSlopesY.GeomNames = "domain"
clm_rz_water_stress.TopoSlopesY.Geom.domain.Value = 0.001
#
# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
#
clm_rz_water_stress.Mannings.Type = "Constant"
clm_rz_water_stress.Mannings.GeomNames = "domain"
clm_rz_water_stress.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm_rz_water_stress.PhaseSources.water.Type = "Constant"
clm_rz_water_stress.PhaseSources.water.GeomNames = "domain"
clm_rz_water_stress.PhaseSources.water.Geom.domain.Value = 0.0
#
# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
#
clm_rz_water_stress.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
#
clm_rz_water_stress.Solver = "Richards"
# Max iter limits total timesteps, this is important as PF-CLM will not run
# past this number of steps even if end time set longer
clm_rz_water_stress.Solver.MaxIter = 500
#
clm_rz_water_stress.Solver.Nonlinear.MaxIter = 15
clm_rz_water_stress.Solver.Nonlinear.ResidualTol = 1e-9
clm_rz_water_stress.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_rz_water_stress.Solver.Nonlinear.EtaValue = 0.01
clm_rz_water_stress.Solver.Nonlinear.UseJacobian = True
clm_rz_water_stress.Solver.Nonlinear.StepTol = 1e-20
clm_rz_water_stress.Solver.Nonlinear.Globalization = "LineSearch"
clm_rz_water_stress.Solver.Linear.KrylovDimension = 15
clm_rz_water_stress.Solver.Linear.MaxRestart = 2
#
clm_rz_water_stress.Solver.Linear.Preconditioner = "PFMG"
clm_rz_water_stress.Solver.PrintSubsurf = False
clm_rz_water_stress.Solver.Drop = 1e-20
clm_rz_water_stress.Solver.AbsTol = 1e-9
#
# This key turns on CLM LSM
clm_rz_water_stress.Solver.LSM = "CLM"

clm_rz_water_stress.Solver.CLM.MetForcing = "1D"
clm_rz_water_stress.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_rz_water_stress.Solver.CLM.MetFilePath = "."

#  We are NOT writing CLM files as SILO but setting this to True
#  will write both SILO and PFB output for CLM (in a single file as
#  specified below)
clm_rz_water_stress.Solver.WriteSiloCLM = False
clm_rz_water_stress.Solver.WriteSiloEvapTrans = False
clm_rz_water_stress.Solver.WriteSiloOverlandBCFlux = False
#  We are writing CLM files as PFB
clm_rz_water_stress.Solver.PrintCLM = True

# Limit native CLM output and logs
clm_rz_water_stress.Solver.CLM.Print1dOut = False
clm_rz_water_stress.Solver.BinaryOutDir = False
clm_rz_water_stress.Solver.WriteCLMBinary = False
clm_rz_water_stress.Solver.CLM.CLMDumpInterval = 1
clm_rz_water_stress.Solver.CLM.WriteLogs = False


# Set evaporation Beta (resistance) function to Linear
clm_rz_water_stress.Solver.CLM.EvapBeta = "Linear"
# Set plant water stress to be a function of Saturation
clm_rz_water_stress.Solver.CLM.VegWaterStress = "Saturation"
# Set residual Sat for soil moisture resistance
clm_rz_water_stress.Solver.CLM.ResSat = 0.2
# Set wilting point limit and field capacity (values are for Saturation, not pressure)
clm_rz_water_stress.Solver.CLM.WiltingPoint = 0.2
clm_rz_water_stress.Solver.CLM.FieldCapacity = 1.00
## this key sets the option described in Ferguson, Jefferson, et al ESS 2016
# a setting of 0 (default) will use standard water stress distribution
clm_rz_water_stress.Solver.CLM.RZWaterStress = 1
# No irrigation
clm_rz_water_stress.Solver.CLM.IrrigationType = "none"


## writing only last daily restarts.  This will be at Midnight GMT and
## starts at timestep 18, then intervals of 24 thereafter
clm_rz_water_stress.Solver.CLM.WriteLastRST = True
clm_rz_water_stress.Solver.CLM.DailyRST = True
# we write a single CLM file for all output at each timestep (one file / timestep
# for all 17 CLM output variables) as described in PF manual
clm_rz_water_stress.Solver.CLM.SingleFile = True


# Initial conditions: water pressure
# ---------------------------------------------------------
#
clm_rz_water_stress.ICPressure.Type = "HydroStaticPatch"
clm_rz_water_stress.ICPressure.GeomNames = "domain"
clm_rz_water_stress.Geom.domain.ICPressure.Value = -2.0
#
clm_rz_water_stress.Geom.domain.ICPressure.RefGeom = "domain"
clm_rz_water_stress.Geom.domain.ICPressure.RefPatch = "z_upper"


# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------
correct_output_dir_name = get_absolute_path("../../correct_output/clm_output")
print(f"dir={correct_output_dir_name}")

clm_rz_water_stress.run(working_directory=dir_name)

# pfrun clm.rz_stress
# pfundist clm.rz_stress

#
# Tests
#
passed = 1


# we compare pressure, saturation and CLM output

for i in range(6):
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

    if i > 0:
        filename = f"/{run_name}.out.clm_output.{timestep}.C.pfb"
        if not pf_test_file(
            dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in CLM for timestep {timestep}",
        ):
            passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)


# for {set i 0} { $i <= 5 } {incr i} {
#     set i_string [format "%05d" $i]
#     if ![pftestFile clm.rz_stress.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits $correct_output_dir] {
#     set passed 0
#     }
#     if ![pftestFile clm.rz_stress.out.satur.$i_string.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
#     set passed 0
#     }
#     if {$i > 0} {
#     if ![pftestFile clm.rz_stress.out.clm_output.$i_string.C.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
#     set passed 0
#     }
#     }

# }


# if $passed {
#     puts "clm.rz_stress : PASSED"
# } {
#     puts "clm.rz_stress : FAILED"
# }
