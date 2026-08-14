# Self-checking test for per-PFT wilting point / field capacity in PF-CLM
# (Solver.CLM.PerPFTWaterStress).
#
# CLM reads a single Solver.CLM.WiltingPoint / FieldCapacity scalar by default.
# With PerPFTWaterStress = True it reads per-PFT values from four rows in
# drv_vegp.dat: wp_press/fc_press for the Pressure formulation, wp_sat/fc_sat for
# the Saturation formulation.  Only the pair matching the active VegWaterStress
# mode is used; an absent row falls back to the scalar for that PFT.
#
# Four self-checks:
#   SAT-BC   Saturation, single PFT: rows present + switch OFF must be
#            bit-identical to the standard vegp (rows ignored when off).
#   SAT-ON   Saturation, single PFT: switch ON differs (per-PFT engages).
#   MULTI    Saturation, two PFTs (split veg map): with the switch ON the two
#            PFT regions get DIFFERENT wilting points, so the per-cell change
#            vs the scalar run differs between regions.  This is the true
#            per-PFT proof (SAT-ON alone only shows per-PFT != scalar).
#   PRESS    Pressure formulation: switch OFF bit-identical, switch ON differs.
#            Exercises the pressure-mode per-PFT ramp (the path East River uses).
#
# The rows are appended to the standard drv_vegp.dat at run time, so the OFF and
# base runs differ only by the four rows.  Pressure wilting/field values are
# chosen so the ramp actually engages from the -2 m initial condition (a
# physical -150 m wilting point never activates in a 5-step run); they exercise
# the mechanism, they are not physical parameters.  Modeled on
# clm_soil_moisture_stress.py.  @RMM 2026

import sys, argparse

from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path
from parflow.tools.io import read_pfb

import numpy as np

run_name = "clm_pft_ws"
clm_pft = Run(run_name, __file__)

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

# Four per-PFT rows, 18 IGBP classes.  fc is wetter than wp in every column
# (fc_sat > wp_sat; fc_press less negative than wp_press), so the clm.F90 startup
# sanity check passes.  wp_sat differs across classes (class 1 = 0.12, class 10 =
# 0.08) and from the Saturation scalar (0.2); wp_press (-5) differs from the
# Pressure scalar (-3).  Pressure values are tuned to engage from IC = -2 m.
PFT_ROWS = """!
wp_press       Wilting point, pressure formulation [m, negative head]
-5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5. -5.
!
fc_press       Field capacity, pressure formulation [m, negative head]
-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.
!
wp_sat         Wilting point, saturation formulation [-]
0.12 0.12 0.10 0.10 0.11 0.09 0.09 0.10 0.09 0.08 0.11 0.10 0.10 0.10 0.05 0.08 0.05 0.08
!
fc_sat         Field capacity, saturation formulation [-]
0.55 0.55 0.50 0.50 0.52 0.45 0.45 0.50 0.45 0.42 0.52 0.50 0.50 0.50 0.30 0.42 0.30 0.42
!
"""

# Two-PFT split: left columns (x <= 2) become IGBP class 1, right columns class
# 10.  Their wp_sat (0.12 vs 0.08) differ, so per-PFT introduces a region
# contrast that the scalar run cannot.
CLASS_LEFT = 1
CLASS_RIGHT = 10


def write_vegm(dst, split):
    src = get_absolute_path("$PF_SRC/test/tcl/clm/drv_vegm.dat")
    with open(src) as fh:
        lines = fh.readlines()
    out = []
    for i, line in enumerate(lines):
        parts = line.split()
        if i < 2 or len(parts) < 25:  # two header lines, or non-data
            out.append(line)
            continue
        x = int(parts[0])
        lead = parts[:7]  # x y lat lon sand clay colorindex
        cls = (CLASS_LEFT if x <= 2 else CLASS_RIGHT) if split else CLASS_RIGHT
        frac = ["0.0"] * 18
        frac[cls - 1] = "1.0"
        out.append("   " + "  ".join(lead + frac) + "\n")
    with open(dst, "w") as fh:
        fh.writelines(out)


def setup_dir(suffix, add_rows, split_vegm=False):
    d = get_absolute_path("test_output/" + run_name + "_" + suffix)
    mkdir(d)
    for directory in directories:
        mkdir(d + "/" + directory)
    for f in [
        "drv_clmin.dat",
        "lai.dat",
        "sai.dat",
        "z0m.dat",
        "displa.dat",
        "narr_1hr.sc3.txt.0",
    ]:
        cp("$PF_SRC/test/tcl/clm/" + f, d)
    cp("$PF_SRC/test/tcl/clm/veg_map.cpfb", d + "/veg_map.pfb")

    # drv_vegm.dat: uniform (single PFT) or a two-PFT left/right split
    write_vegm(d + "/drv_vegm.dat", split_vegm)

    # drv_vegp.dat: standard, optionally with the four per-PFT rows appended
    base = get_absolute_path("$PF_SRC/test/tcl/clm/drv_vegp.dat")
    with open(base, "r") as fh:
        vegp = fh.read()
    if add_rows:
        vegp = vegp + PFT_ROWS
    with open(d + "/drv_vegp.dat", "w") as fh:
        fh.write(vegp)
    return d


def final_press(d):
    return read_pfb(d + "/" + run_name + ".out.press.00005.pfb")


def _run_and_get(suffix, add_rows, split_vegm=False):
    d = setup_dir(suffix, add_rows, split_vegm)
    clm_pft.run(working_directory=d)
    return d


# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
clm_pft.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

clm_pft.Process.Topology.P = args.p
clm_pft.Process.Topology.Q = args.q
clm_pft.Process.Topology.R = args.r

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
clm_pft.ComputationalGrid.Lower.X = 0.0
clm_pft.ComputationalGrid.Lower.Y = 0.0
clm_pft.ComputationalGrid.Lower.Z = 0.0

clm_pft.ComputationalGrid.DX = 1000.0
clm_pft.ComputationalGrid.DY = 1000.0
clm_pft.ComputationalGrid.DZ = 0.5

clm_pft.ComputationalGrid.NX = 5
clm_pft.ComputationalGrid.NY = 5
clm_pft.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# GeomInput / Geometry
# -----------------------------------------------------------------------------
clm_pft.GeomInput.Names = "domain_input"
clm_pft.GeomInput.domain_input.InputType = "Box"
clm_pft.GeomInput.domain_input.GeomName = "domain"

clm_pft.Geom.domain.Lower.X = 0.0
clm_pft.Geom.domain.Lower.Y = 0.0
clm_pft.Geom.domain.Lower.Z = 0.0
clm_pft.Geom.domain.Upper.X = 5000.0
clm_pft.Geom.domain.Upper.Y = 5000.0
clm_pft.Geom.domain.Upper.Z = 5.0
clm_pft.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
clm_pft.Geom.Perm.Names = "domain"
clm_pft.Geom.domain.Perm.Type = "Constant"
clm_pft.Geom.domain.Perm.Value = 0.2
clm_pft.Perm.TensorType = "TensorByGeom"
clm_pft.Geom.Perm.TensorByGeom.Names = "domain"
clm_pft.Geom.domain.Perm.TensorValX = 1.0
clm_pft.Geom.domain.Perm.TensorValY = 1.0
clm_pft.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
clm_pft.SpecificStorage.Type = "Constant"
clm_pft.SpecificStorage.GeomNames = "domain"
clm_pft.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
clm_pft.Phase.Names = "water"
clm_pft.Phase.water.Density.Type = "Constant"
clm_pft.Phase.water.Density.Value = 1.0
clm_pft.Phase.water.Viscosity.Type = "Constant"
clm_pft.Phase.water.Viscosity.Value = 1.0

clm_pft.Contaminants.Names = ""
clm_pft.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------
clm_pft.TimingInfo.BaseUnit = 1.0
clm_pft.TimingInfo.StartCount = 0
clm_pft.TimingInfo.StartTime = 0.0
clm_pft.TimingInfo.StopTime = 5
clm_pft.TimingInfo.DumpInterval = -1
clm_pft.TimeStep.Type = "Constant"
clm_pft.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
clm_pft.Geom.Porosity.GeomNames = "domain"
clm_pft.Geom.domain.Porosity.Type = "Constant"
clm_pft.Geom.domain.Porosity.Value = 0.390

clm_pft.Domain.GeomName = "domain"

clm_pft.Phase.water.Mobility.Type = "Constant"
clm_pft.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability / Saturation
# -----------------------------------------------------------------------------
clm_pft.Phase.RelPerm.Type = "VanGenuchten"
clm_pft.Phase.RelPerm.GeomNames = "domain"
clm_pft.Geom.domain.RelPerm.Alpha = 3.5
clm_pft.Geom.domain.RelPerm.N = 2.0

clm_pft.Phase.Saturation.Type = "VanGenuchten"
clm_pft.Phase.Saturation.GeomNames = "domain"
clm_pft.Geom.domain.Saturation.Alpha = 3.5
clm_pft.Geom.domain.Saturation.N = 2.0
clm_pft.Geom.domain.Saturation.SRes = 0.01
clm_pft.Geom.domain.Saturation.SSat = 1.0

clm_pft.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
clm_pft.Cycle.Names = "constant"
clm_pft.Cycle.constant.Names = "alltime"
clm_pft.Cycle.constant.alltime.Length = 1
clm_pft.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------
clm_pft.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"
for p in ["x_lower", "y_lower", "z_lower", "x_upper", "y_upper"]:
    clm_pft.Patch[p].BCPressure.Type = "FluxConst"
    clm_pft.Patch[p].BCPressure.Cycle = "constant"
    clm_pft.Patch[p].BCPressure.alltime.Value = 0.0
clm_pft.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm_pft.Patch.z_upper.BCPressure.Cycle = "constant"
clm_pft.Patch.z_upper.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Topo slopes / Mannings
# -----------------------------------------------------------------------------
clm_pft.TopoSlopesX.Type = "Constant"
clm_pft.TopoSlopesX.GeomNames = "domain"
clm_pft.TopoSlopesX.Geom.domain.Value = -0.001
clm_pft.TopoSlopesY.Type = "Constant"
clm_pft.TopoSlopesY.GeomNames = "domain"
clm_pft.TopoSlopesY.Geom.domain.Value = 0.001
clm_pft.Mannings.Type = "Constant"
clm_pft.Mannings.GeomNames = "domain"
clm_pft.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources / solution
# -----------------------------------------------------------------------------
clm_pft.PhaseSources.water.Type = "Constant"
clm_pft.PhaseSources.water.GeomNames = "domain"
clm_pft.PhaseSources.water.Geom.domain.Value = 0.0
clm_pft.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
clm_pft.Solver = "Richards"
clm_pft.Solver.MaxIter = 500
clm_pft.Solver.Nonlinear.MaxIter = 15
clm_pft.Solver.Nonlinear.ResidualTol = 1e-9
clm_pft.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_pft.Solver.Nonlinear.EtaValue = 0.01
clm_pft.Solver.Nonlinear.UseJacobian = True
clm_pft.Solver.Nonlinear.StepTol = 1e-20
clm_pft.Solver.Nonlinear.Globalization = "LineSearch"
clm_pft.Solver.Linear.KrylovDimension = 15
clm_pft.Solver.Linear.MaxRestart = 2
clm_pft.Solver.Linear.Preconditioner = "PFMG"
clm_pft.Solver.PrintSubsurf = False
clm_pft.Solver.Drop = 1e-20
clm_pft.Solver.AbsTol = 1e-9

clm_pft.Solver.LSM = "CLM"
clm_pft.Solver.CLM.MetForcing = "1D"
clm_pft.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_pft.Solver.CLM.MetFilePath = "."

clm_pft.Solver.WriteSiloCLM = False
clm_pft.Solver.WriteSiloEvapTrans = False
clm_pft.Solver.WriteSiloOverlandBCFlux = False
clm_pft.Solver.PrintCLM = True
clm_pft.Solver.CLM.Print1dOut = False
clm_pft.Solver.BinaryOutDir = False
clm_pft.Solver.WriteCLMBinary = False
clm_pft.Solver.CLM.CLMDumpInterval = 1
clm_pft.Solver.CLM.WriteLogs = False

clm_pft.Solver.CLM.EvapBeta = "Linear"
clm_pft.Solver.CLM.ResSat = 0.2
clm_pft.Solver.CLM.RZWaterStress = 1
clm_pft.Solver.CLM.IrrigationType = "none"
clm_pft.Solver.CLM.WriteLastRST = True
clm_pft.Solver.CLM.DailyRST = True
clm_pft.Solver.CLM.SingleFile = True

# Initial conditions
clm_pft.ICPressure.Type = "HydroStaticPatch"
clm_pft.ICPressure.GeomNames = "domain"
clm_pft.Geom.domain.ICPressure.Value = -2.0
clm_pft.Geom.domain.ICPressure.RefGeom = "domain"
clm_pft.Geom.domain.ICPressure.RefPatch = "z_upper"

# =============================================================================
# Saturation formulation
# =============================================================================
clm_pft.Solver.CLM.VegWaterStress = "Saturation"
clm_pft.Solver.CLM.WiltingPoint = 0.2
clm_pft.Solver.CLM.FieldCapacity = 1.00

# SAT base: standard vegp, switch off
clm_pft.Solver.CLM.PerPFTWaterStress = False
pA = final_press(_run_and_get("sat_base", add_rows=False))

# SAT rows-off: vegp + rows, switch off -> must match base
clm_pft.Solver.CLM.PerPFTWaterStress = False
pB = final_press(_run_and_get("sat_rows_off", add_rows=True))

# SAT rows-on: vegp + rows, switch on -> per-PFT engages
clm_pft.Solver.CLM.PerPFTWaterStress = True
pC = final_press(_run_and_get("sat_rows_on", add_rows=True))

# MULTI: two-PFT split map, scalar (off) vs per-PFT (on)
clm_pft.Solver.CLM.PerPFTWaterStress = False
pM_off = final_press(_run_and_get("multi_off", add_rows=True, split_vegm=True))
clm_pft.Solver.CLM.PerPFTWaterStress = True
pM_on = final_press(_run_and_get("multi_on", add_rows=True, split_vegm=True))

# =============================================================================
# Pressure formulation (the path East River uses)
# =============================================================================
clm_pft.Solver.CLM.VegWaterStress = "Pressure"
clm_pft.Solver.CLM.WiltingPoint = -3.0
clm_pft.Solver.CLM.FieldCapacity = -1.0

clm_pft.Solver.CLM.PerPFTWaterStress = False
pP_base = final_press(_run_and_get("press_base", add_rows=False))
clm_pft.Solver.CLM.PerPFTWaterStress = False
pP_off = final_press(_run_and_get("press_rows_off", add_rows=True))
clm_pft.Solver.CLM.PerPFTWaterStress = True
pP_on = final_press(_run_and_get("press_rows_on", add_rows=True))

# =============================================================================
# Self-checks
# =============================================================================
# MULTI: per-cell change vs the scalar run, averaged over depth.  With one PFT
# this map is spatially uniform; with two PFTs (different wp/fc) the two regions
# change by different amounts, so the map has nonzero spatial spread.  Using the
# spread is axis-agnostic (drv_readvegtf transposes vegm columns to grid rows).
dM = np.abs(pM_on - pM_off).mean(axis=0)  # (ny, nx)
multi_spread = float(dM.std())

checks = [
    (
        "SAT-BC  rows off == base (expect ~0)",
        float(np.max(np.abs(pB - pA))),
        "lt",
        1e-12,
    ),
    (
        "SAT-ON  per-PFT on vs base (expect >0)",
        float(np.max(np.abs(pC - pA))),
        "gt",
        1e-8,
    ),
    ("MULTI   two PFT regions differ spatially (expect >0)", multi_spread, "gt", 1e-7),
    (
        "PRESS-BC pressure rows off == base (expect ~0)",
        float(np.max(np.abs(pP_off - pP_base))),
        "lt",
        1e-12,
    ),
    (
        "PRESS-ON pressure per-PFT on vs base (expect >0)",
        float(np.max(np.abs(pP_on - pP_base))),
        "gt",
        1e-8,
    ),
]

all_ok = True
for label, val, op, tol in checks:
    ok = (val < tol) if op == "lt" else (val > tol)
    all_ok = all_ok and ok
    print(f"{label:52s}: {val:.3e} -> {'PASS' if ok else 'FAIL'}")

if all_ok:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
