# -----------------------------------------------------------------------------
# CLM Snow Age Parameters Test
#
# This test verifies the new VIS/NIR-separated snow age parameterization
# using optimal values from Abolafia-Rosenzweig et al. (2022).
#
# New keys tested:
#   Solver.CLM.SnowAgeTau0Vis = 3.05e6      (AR2022: 3.05e6, default: 1e6)
#   Solver.CLM.SnowAgeTau0Nir = 5.29e5      (AR2022: 5.29e5, default: 1e6)
#   Solver.CLM.SnowAgeGrainGrowthVis = 9287 (AR2022: 9287, default: 5000)
#   Solver.CLM.SnowAgeGrainGrowthNir = 7715 (AR2022: 7715, default: 5000)
#   Solver.CLM.SnowAgeDirtSootVis = 0.25    (AR2022: 0.25, default: 0.3)
#   Solver.CLM.SnowAgeDirtSootNir = 0.11    (AR2022: 0.11, default: 0.3)
#   Solver.CLM.SnowAgeResetFactor = 0.1     (default: 0.1)
#
# Reference:
#   Abolafia-Rosenzweig et al. (2022) J. Hydrometeorol. - Calibration study
#   showing VIS and NIR bands require different aging parameters for
#   Western US snowpacks.
# -----------------------------------------------------------------------------

import sys
import argparse

from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "clm_snow_age_params"
clm = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# Making output directories and copying input files
# -----------------------------------------------------------------------------

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
cp("$PF_SRC/test/tcl/clm/snow_forcing.1hr.txt", dir_name)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------

clm.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

clm.Process.Topology.P = args.p
clm.Process.Topology.Q = args.q
clm.Process.Topology.R = args.r

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 1000.0
clm.ComputationalGrid.DY = 1000.0
clm.ComputationalGrid.DZ = 0.5

clm.ComputationalGrid.NX = 5
clm.ComputationalGrid.NY = 5
clm.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------

clm.GeomInput.Names = "domain_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

clm.GeomInput.domain_input.InputType = "Box"
clm.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

clm.Geom.domain.Lower.X = 0.0
clm.Geom.domain.Lower.Y = 0.0
clm.Geom.domain.Lower.Z = 0.0

clm.Geom.domain.Upper.X = 5000.0
clm.Geom.domain.Upper.Y = 5000.0
clm.Geom.domain.Upper.Z = 5.0

clm.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

clm.Geom.Perm.Names = "domain"
clm.Geom.domain.Perm.Type = "Constant"
clm.Geom.domain.Perm.Value = 0.2

clm.Perm.TensorType = "TensorByGeom"
clm.Geom.Perm.TensorByGeom.Names = "domain"

clm.Geom.domain.Perm.TensorValX = 1.0
clm.Geom.domain.Perm.TensorValY = 1.0
clm.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

clm.SpecificStorage.Type = "Constant"
clm.SpecificStorage.GeomNames = "domain"
clm.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm.Phase.Names = "water"
clm.Phase.water.Density.Type = "Constant"
clm.Phase.water.Density.Value = 1.0
clm.Phase.water.Viscosity.Type = "Constant"
clm.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

clm.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

clm.TimingInfo.BaseUnit = 1.0
clm.TimingInfo.StartCount = 0
clm.TimingInfo.StartTime = 0.0
clm.TimingInfo.StopTime = 5
clm.TimingInfo.DumpInterval = -1
clm.TimeStep.Type = "Constant"
clm.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm.Geom.Porosity.GeomNames = "domain"
clm.Geom.domain.Porosity.Type = "Constant"
clm.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

clm.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------

clm.Phase.water.Mobility.Type = "Constant"
clm.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

clm.Phase.RelPerm.Type = "VanGenuchten"
clm.Phase.RelPerm.GeomNames = "domain"
clm.Geom.domain.RelPerm.Alpha = 3.5
clm.Geom.domain.RelPerm.N = 2.0

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

clm.Phase.Saturation.Type = "VanGenuchten"
clm.Phase.Saturation.GeomNames = "domain"
clm.Geom.domain.Saturation.Alpha = 3.5
clm.Geom.domain.Saturation.N = 2.0
clm.Geom.domain.Saturation.SRes = 0.01
clm.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

clm.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

clm.Cycle.Names = "constant"
clm.Cycle.constant.Names = "alltime"
clm.Cycle.constant.alltime.Length = 1
clm.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

clm.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"

clm.Patch.x_lower.BCPressure.Type = "FluxConst"
clm.Patch.x_lower.BCPressure.Cycle = "constant"
clm.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm.Patch.y_lower.BCPressure.Type = "FluxConst"
clm.Patch.y_lower.BCPressure.Cycle = "constant"
clm.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm.Patch.z_lower.BCPressure.Type = "FluxConst"
clm.Patch.z_lower.BCPressure.Cycle = "constant"
clm.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm.Patch.x_upper.BCPressure.Type = "FluxConst"
clm.Patch.x_upper.BCPressure.Cycle = "constant"
clm.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm.Patch.y_upper.BCPressure.Type = "FluxConst"
clm.Patch.y_upper.BCPressure.Cycle = "constant"
clm.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm.Patch.z_upper.BCPressure.Cycle = "constant"
clm.Patch.z_upper.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

clm.TopoSlopesX.Type = "Constant"
clm.TopoSlopesX.GeomNames = "domain"
clm.TopoSlopesX.Geom.domain.Value = -0.001

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

clm.TopoSlopesY.Type = "Constant"
clm.TopoSlopesY.GeomNames = "domain"
clm.TopoSlopesY.Geom.domain.Value = 0.001

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

clm.Mannings.Type = "Constant"
clm.Mannings.GeomNames = "domain"
clm.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources
# -----------------------------------------------------------------------------

clm.PhaseSources.water.Type = "Constant"
clm.PhaseSources.water.GeomNames = "domain"
clm.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

clm.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

clm.Solver = "Richards"
clm.Solver.MaxIter = 500

clm.Solver.Nonlinear.MaxIter = 15
clm.Solver.Nonlinear.ResidualTol = 1e-9
clm.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm.Solver.Nonlinear.EtaValue = 0.01
clm.Solver.Nonlinear.UseJacobian = True
clm.Solver.Nonlinear.StepTol = 1e-20
clm.Solver.Nonlinear.Globalization = "LineSearch"
clm.Solver.Linear.KrylovDimension = 15
clm.Solver.Linear.MaxRestart = 2

clm.Solver.Linear.Preconditioner = "PFMG"
clm.Solver.PrintSubsurf = False
clm.Solver.Drop = 1e-20
clm.Solver.AbsTol = 1e-9

# -----------------------------------------------------------------------------
# CLM Settings
# -----------------------------------------------------------------------------

clm.Solver.LSM = "CLM"
clm.Solver.CLM.MetForcing = "1D"
clm.Solver.CLM.MetFileName = "snow_forcing.1hr.txt"
clm.Solver.CLM.MetFilePath = "."

clm.Solver.WriteSiloCLM = False
clm.Solver.WriteSiloEvapTrans = False
clm.Solver.WriteSiloOverlandBCFlux = False
clm.Solver.PrintCLM = True

clm.Solver.CLM.Print1dOut = False
clm.Solver.BinaryOutDir = False
clm.Solver.WriteCLMBinary = False
clm.Solver.CLM.CLMDumpInterval = 1
clm.Solver.CLM.WriteLogs = False
clm.Solver.CLM.WriteLastRST = True
clm.Solver.CLM.DailyRST = True
clm.Solver.CLM.SingleFile = True

# -----------------------------------------------------------------------------
# Snow Parameterization Keys - Base settings (same as defaults test)
# -----------------------------------------------------------------------------

clm.Solver.CLM.SnowPartition = "CLM"
clm.Solver.CLM.ThinSnowDamping = 1.0
clm.Solver.CLM.AlbedoScheme = "CLM"

# -----------------------------------------------------------------------------
# Snow Age VIS/NIR Parameters - AR2022 Optimal Values
# These differ from defaults and should produce different albedo evolution
# -----------------------------------------------------------------------------

# VIS band ages slower (longer e-folding time)
clm.Solver.CLM.SnowAgeTau0Vis = 3.05e6       # AR2022 optimal (default: 1e6)
# NIR band ages faster (shorter e-folding time)
clm.Solver.CLM.SnowAgeTau0Nir = 5.29e5       # AR2022 optimal (default: 1e6)

# Grain growth temperature sensitivity
clm.Solver.CLM.SnowAgeGrainGrowthVis = 9287  # AR2022 optimal (default: 5000)
clm.Solver.CLM.SnowAgeGrainGrowthNir = 7715  # AR2022 optimal (default: 5000)

# Dirt/soot contamination effect
clm.Solver.CLM.SnowAgeDirtSootVis = 0.25     # AR2022 optimal (default: 0.3)
clm.Solver.CLM.SnowAgeDirtSootNir = 0.11     # AR2022 optimal (default: 0.3)

# Fresh snow reset factor (keep at default)
clm.Solver.CLM.SnowAgeResetFactor = 0.1      # default

# -----------------------------------------------------------------------------
# Initial conditions: water pressure
# -----------------------------------------------------------------------------

clm.ICPressure.Type = "HydroStaticPatch"
clm.ICPressure.GeomNames = "domain"
clm.Geom.domain.ICPressure.Value = -2.0
clm.Geom.domain.ICPressure.RefGeom = "domain"
clm.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

correct_output_dir_name = get_absolute_path("../../correct_output/" + run_name)

clm.run(working_directory=dir_name)

# -----------------------------------------------------------------------------
# Tests - Compare pressure, saturation, and CLM output
# -----------------------------------------------------------------------------

passed = True

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
            f"Max difference in CLM output for timestep {timestep}",
        ):
            passed = False

rm(dir_name)

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
