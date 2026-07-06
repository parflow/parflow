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

run_name = "clm_sms"
clm_sms = Run(run_name, __file__)

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

def setup_dir(suffix):
    d = get_absolute_path("test_output/" + run_name + "_" + suffix)
    mkdir(d)
    for directory in directories:
        mkdir(d + "/" + directory)
    for f in ["drv_clmin.dat", "drv_vegm.dat", "drv_vegp.dat", "lai.dat",
              "sai.dat", "z0m.dat", "displa.dat", "narr_1hr.sc3.txt.0"]:
        cp("$PF_SRC/test/tcl/clm/" + f, d)
    cp("$PF_SRC/test/tcl/clm/veg_map.cpfb", d + "/veg_map.pfb")
    return d

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
clm_sms.FileVersion = 4

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

clm_sms.Process.Topology.P = args.p
clm_sms.Process.Topology.Q = args.q
clm_sms.Process.Topology.R = args.r

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
clm_sms.ComputationalGrid.Lower.X = 0.0
clm_sms.ComputationalGrid.Lower.Y = 0.0
clm_sms.ComputationalGrid.Lower.Z = 0.0

clm_sms.ComputationalGrid.DX = 1000.0
clm_sms.ComputationalGrid.DY = 1000.0
clm_sms.ComputationalGrid.DZ = 0.5

clm_sms.ComputationalGrid.NX = 5
clm_sms.ComputationalGrid.NY = 5
clm_sms.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
clm_sms.GeomInput.Names = "domain_input"


# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
clm_sms.GeomInput.domain_input.InputType = "Box"
clm_sms.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
clm_sms.Geom.domain.Lower.X = 0.0
clm_sms.Geom.domain.Lower.Y = 0.0
clm_sms.Geom.domain.Lower.Z = 0.0

clm_sms.Geom.domain.Upper.X = 5000.0
clm_sms.Geom.domain.Upper.Y = 5000.0
clm_sms.Geom.domain.Upper.Z = 5.0

clm_sms.Geom.domain.Patches = (
    "x_lower x_upper y_lower y_upper z_lower z_upper"
)

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
clm_sms.Geom.Perm.Names = "domain"

clm_sms.Geom.domain.Perm.Type = "Constant"
clm_sms.Geom.domain.Perm.Value = 0.2


clm_sms.Perm.TensorType = "TensorByGeom"

clm_sms.Geom.Perm.TensorByGeom.Names = "domain"

clm_sms.Geom.domain.Perm.TensorValX = 1.0
clm_sms.Geom.domain.Perm.TensorValY = 1.0
clm_sms.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

clm_sms.SpecificStorage.Type = "Constant"
clm_sms.SpecificStorage.GeomNames = "domain"
clm_sms.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm_sms.Phase.Names = "water"

clm_sms.Phase.water.Density.Type = "Constant"
clm_sms.Phase.water.Density.Value = 1.0

clm_sms.Phase.water.Viscosity.Type = "Constant"
clm_sms.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
clm_sms.Contaminants.Names = ""


# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm_sms.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
#
clm_sms.TimingInfo.BaseUnit = 1.0
clm_sms.TimingInfo.StartCount = 0
clm_sms.TimingInfo.StartTime = 0.0
clm_sms.TimingInfo.StopTime = 5
clm_sms.TimingInfo.DumpInterval = -1
clm_sms.TimeStep.Type = "Constant"
clm_sms.TimeStep.Value = 1.0
#

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm_sms.Geom.Porosity.GeomNames = "domain"

clm_sms.Geom.domain.Porosity.Type = "Constant"
clm_sms.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
clm_sms.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
clm_sms.Phase.water.Mobility.Type = "Constant"
clm_sms.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
#
clm_sms.Phase.RelPerm.Type = "VanGenuchten"
clm_sms.Phase.RelPerm.GeomNames = "domain"
#
clm_sms.Geom.domain.RelPerm.Alpha = 3.5
clm_sms.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm_sms.Phase.Saturation.Type = "VanGenuchten"
clm_sms.Phase.Saturation.GeomNames = "domain"
#
clm_sms.Geom.domain.Saturation.Alpha = 3.5
clm_sms.Geom.domain.Saturation.N = 2.0
clm_sms.Geom.domain.Saturation.SRes = 0.01
clm_sms.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
clm_sms.Wells.Names = ""


# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
clm_sms.Cycle.Names = "constant"
clm_sms.Cycle.constant.Names = "alltime"
clm_sms.Cycle.constant.alltime.Length = 1
clm_sms.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
clm_sms.BCPressure.PatchNames = (
    "x_lower x_upper y_lower y_upper z_lower z_upper"
)
#
clm_sms.Patch.x_lower.BCPressure.Type = "FluxConst"
clm_sms.Patch.x_lower.BCPressure.Cycle = "constant"
clm_sms.Patch.x_lower.BCPressure.alltime.Value = 0.0
#
clm_sms.Patch.y_lower.BCPressure.Type = "FluxConst"
clm_sms.Patch.y_lower.BCPressure.Cycle = "constant"
clm_sms.Patch.y_lower.BCPressure.alltime.Value = 0.0
#
clm_sms.Patch.z_lower.BCPressure.Type = "FluxConst"
clm_sms.Patch.z_lower.BCPressure.Cycle = "constant"
clm_sms.Patch.z_lower.BCPressure.alltime.Value = 0.0
#
clm_sms.Patch.x_upper.BCPressure.Type = "FluxConst"
clm_sms.Patch.x_upper.BCPressure.Cycle = "constant"
clm_sms.Patch.x_upper.BCPressure.alltime.Value = 0.0
#
clm_sms.Patch.y_upper.BCPressure.Type = "FluxConst"
clm_sms.Patch.y_upper.BCPressure.Cycle = "constant"
clm_sms.Patch.y_upper.BCPressure.alltime.Value = 0.0
#
clm_sms.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm_sms.Patch.z_upper.BCPressure.Cycle = "constant"
clm_sms.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
#
clm_sms.TopoSlopesX.Type = "Constant"
clm_sms.TopoSlopesX.GeomNames = "domain"
clm_sms.TopoSlopesX.Geom.domain.Value = -0.001
#
# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
#
clm_sms.TopoSlopesY.Type = "Constant"
clm_sms.TopoSlopesY.GeomNames = "domain"
clm_sms.TopoSlopesY.Geom.domain.Value = 0.001
#
# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
#
clm_sms.Mannings.Type = "Constant"
clm_sms.Mannings.GeomNames = "domain"
clm_sms.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm_sms.PhaseSources.water.Type = "Constant"
clm_sms.PhaseSources.water.GeomNames = "domain"
clm_sms.PhaseSources.water.Geom.domain.Value = 0.0
#
# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
#
clm_sms.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
#
clm_sms.Solver = "Richards"
# Max iter limits total timesteps, this is important as PF-CLM will not run
# past this number of steps even if end time set longer
clm_sms.Solver.MaxIter = 500
#
clm_sms.Solver.Nonlinear.MaxIter = 15
clm_sms.Solver.Nonlinear.ResidualTol = 1e-9
clm_sms.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_sms.Solver.Nonlinear.EtaValue = 0.01
clm_sms.Solver.Nonlinear.UseJacobian = True
clm_sms.Solver.Nonlinear.StepTol = 1e-20
clm_sms.Solver.Nonlinear.Globalization = "LineSearch"
clm_sms.Solver.Linear.KrylovDimension = 15
clm_sms.Solver.Linear.MaxRestart = 2
#
clm_sms.Solver.Linear.Preconditioner = "PFMG"
clm_sms.Solver.PrintSubsurf = False
clm_sms.Solver.Drop = 1e-20
clm_sms.Solver.AbsTol = 1e-9
#
# This key turns on CLM LSM
clm_sms.Solver.LSM = "CLM"

clm_sms.Solver.CLM.MetForcing = "1D"
clm_sms.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_sms.Solver.CLM.MetFilePath = "."

#  We are NOT writing CLM files as SILO but setting this to True
#  will write both SILO and PFB output for CLM (in a single file as
#  specified below)
clm_sms.Solver.WriteSiloCLM = False
clm_sms.Solver.WriteSiloEvapTrans = False
clm_sms.Solver.WriteSiloOverlandBCFlux = False
#  We are writing CLM files as PFB
clm_sms.Solver.PrintCLM = True

# Limit native CLM output and logs
clm_sms.Solver.CLM.Print1dOut = False
clm_sms.Solver.BinaryOutDir = False
clm_sms.Solver.WriteCLMBinary = False
clm_sms.Solver.CLM.CLMDumpInterval = 1
clm_sms.Solver.CLM.WriteLogs = False


# Set evaporation Beta (resistance) function to Linear
clm_sms.Solver.CLM.EvapBeta = "Linear"
# Set plant water stress to be a function of Saturation
clm_sms.Solver.CLM.VegWaterStress = "Saturation"
# Set residual Sat for soil moisture resistance
clm_sms.Solver.CLM.ResSat = 0.2
# Set wilting point limit and field capacity (values are for Saturation, not pressure)
clm_sms.Solver.CLM.WiltingPoint = 0.2
clm_sms.Solver.CLM.FieldCapacity = 1.00
## this key sets the option described in Ferguson, Jefferson, et al ESS 2016
# a setting of 0 (default) will use standard water stress distribution
clm_sms.Solver.CLM.RZWaterStress = 1
# No irrigation
clm_sms.Solver.CLM.IrrigationType = "none"


## writing only last daily restarts.  This will be at Midnight GMT and
## starts at timestep 18, then intervals of 24 thereafter
clm_sms.Solver.CLM.WriteLastRST = True
clm_sms.Solver.CLM.DailyRST = True
# we write a single CLM file for all output at each timestep (one file / timestep
# for all 17 CLM output variables) as described in PF manual
clm_sms.Solver.CLM.SingleFile = True


# Initial conditions: water pressure
# ---------------------------------------------------------
#
clm_sms.ICPressure.Type = "HydroStaticPatch"
clm_sms.ICPressure.GeomNames = "domain"
clm_sms.Geom.domain.ICPressure.Value = -2.0
#
clm_sms.Geom.domain.ICPressure.RefGeom = "domain"
clm_sms.Geom.domain.ICPressure.RefPatch = "z_upper"


# -----------------------------------------------------------------------------
# Self-checking guard test.  van Genuchten S_res = 0.01, margin = 0.02, so the
# residual limit raises the effective wilting point to 0.03.
#   B: low wp (0.005) with the residual limit ON  -> wp_eff = 0.03
#   C: wp set safely at 0.03, guard OFF           -> wp_eff = 0.03   (B must match C)
#   A: low wp (0.005), guard OFF                  -> wp_eff = 0.005  (B must differ)
# -----------------------------------------------------------------------------
import sys
import numpy as np
from parflow.tools.io import read_pfb

MARGIN = 0.02
SAFE_WP = 0.03   # Saturation.SRes (0.01) + MARGIN


def final_press(d):
    return read_pfb(d + "/" + run_name + ".out.press.00005.pfb")


# B: low wilting point, residual limit ON
clm_sms.Solver.CLM.WiltingPoint = 0.005
clm_sms.Solver.CLM.SoilMoistureStress = True
clm_sms.Solver.CLM.SoilMoistureStress.ResidualLimit = True
clm_sms.Solver.CLM.SoilMoistureStress.ResidualLimit.Margin = MARGIN
_dir = setup_dir("guard_on")
clm_sms.run(working_directory=_dir)
pB = final_press(_dir)

# C: wilting point set safely, guard OFF (reference)
clm_sms.Solver.CLM.WiltingPoint = SAFE_WP
clm_sms.Solver.CLM.SoilMoistureStress = False
_dir = setup_dir("safe_wp")
clm_sms.run(working_directory=_dir)
pC = final_press(_dir)

# A: low wilting point, guard OFF (control)
clm_sms.Solver.CLM.WiltingPoint = 0.005
clm_sms.Solver.CLM.SoilMoistureStress = False
_dir = setup_dir("unguarded")
clm_sms.run(working_directory=_dir)
pA = final_press(_dir)

d_match = float(np.max(np.abs(pB - pC)))
d_diff = float(np.max(np.abs(pB - pA)))
ok_match = d_match < 1.0e-8
ok_engage = d_diff > 1.0e-8
print(f"guarded low-wp vs safe-wp (expect ~0): {d_match:.3e} -> {'PASS' if ok_match else 'FAIL'}")
print(f"guard on vs off at low wp (expect >0):  {d_diff:.3e} -> {'PASS' if ok_engage else 'FAIL'}")

if ok_match and ok_engage:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
