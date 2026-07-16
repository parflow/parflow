# Gate test for the prescribed evap_trans sink guard (Solver.EvapTransGuard).
#
# A closed 3x3x10 Richards box (no-flux everywhere, no CLM) is forced with a
# steady EvapTransFile whose top-layer cells carry a negative (ET-demand) flux
# that exceeds what the column can supply -- the pathology of prescribed P-ET
# spin-up forcing.  Three runs:
#
#   A: guard OFF  -> the unlimited sink drives top-layer pressure far below
#                    any physical suction (the failure mode the guard removes)
#   B: guard ON   -> the sink ramps off as S approaches S_res + Margin; the
#                    minimum pressure stays bounded, and the PrintLog CSV's
#                    applied-sink total closes the storage balance
#   C: guard keys absent -> byte-identical to A (default-off discipline)
#
# @R Maxwell + Claude, 16-Jul-2026

import sys
import argparse
import numpy as np

from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb

run_name = "etguard"
etg = Run(run_name, __file__)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1, type=int)
parser.add_argument("-q", "--q", default=1, type=int)
parser.add_argument("-r", "--r", default=1, type=int)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Grid: small closed box, 5 m deep, 0.5 m cells
# -----------------------------------------------------------------------------
etg.FileVersion = 4

etg.Process.Topology.P = args.p
etg.Process.Topology.Q = args.q
etg.Process.Topology.R = args.r

etg.ComputationalGrid.Lower.X = 0.0
etg.ComputationalGrid.Lower.Y = 0.0
etg.ComputationalGrid.Lower.Z = 0.0

etg.ComputationalGrid.DX = 100.0
etg.ComputationalGrid.DY = 100.0
etg.ComputationalGrid.DZ = 0.5

etg.ComputationalGrid.NX = 3
etg.ComputationalGrid.NY = 3
etg.ComputationalGrid.NZ = 10

etg.GeomInput.Names = "domain_input"
etg.GeomInput.domain_input.InputType = "Box"
etg.GeomInput.domain_input.GeomName = "domain"

etg.Geom.domain.Lower.X = 0.0
etg.Geom.domain.Lower.Y = 0.0
etg.Geom.domain.Lower.Z = 0.0
etg.Geom.domain.Upper.X = 300.0
etg.Geom.domain.Upper.Y = 300.0
etg.Geom.domain.Upper.Z = 5.0

etg.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Soil: single homogeneous unit, van Genuchten by region
# -----------------------------------------------------------------------------
etg.Geom.Perm.Names = "domain"
etg.Geom.domain.Perm.Type = "Constant"
etg.Geom.domain.Perm.Value = 0.01

etg.Perm.TensorType = "TensorByGeom"
etg.Geom.Perm.TensorByGeom.Names = "domain"
etg.Geom.domain.Perm.TensorValX = 1.0
etg.Geom.domain.Perm.TensorValY = 1.0
etg.Geom.domain.Perm.TensorValZ = 1.0

etg.SpecificStorage.Type = "Constant"
etg.SpecificStorage.GeomNames = "domain"
etg.Geom.domain.SpecificStorage.Value = 1.0e-5

etg.Phase.Names = "water"
etg.Phase.water.Density.Type = "Constant"
etg.Phase.water.Density.Value = 1.0
etg.Phase.water.Viscosity.Type = "Constant"
etg.Phase.water.Viscosity.Value = 1.0

etg.Contaminants.Names = ""
etg.Geom.Retardation.GeomNames = ""

etg.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing: 48 hours of constant 1-hour steps
# -----------------------------------------------------------------------------
etg.TimingInfo.BaseUnit = 1.0
etg.TimingInfo.StartCount = 0
etg.TimingInfo.StartTime = 0.0
etg.TimingInfo.StopTime = 48.0
etg.TimingInfo.DumpInterval = 12.0
etg.TimeStep.Type = "Constant"
etg.TimeStep.Value = 1.0

etg.Geom.Porosity.GeomNames = "domain"
etg.Geom.domain.Porosity.Type = "Constant"
etg.Geom.domain.Porosity.Value = 0.4

etg.Domain.GeomName = "domain"

SRES = 0.1
etg.Phase.RelPerm.Type = "VanGenuchten"
etg.Phase.RelPerm.GeomNames = "domain"
etg.Geom.domain.RelPerm.Alpha = 2.0
etg.Geom.domain.RelPerm.N = 2.0

etg.Phase.Saturation.Type = "VanGenuchten"
etg.Phase.Saturation.GeomNames = "domain"
etg.Geom.domain.Saturation.Alpha = 2.0
etg.Geom.domain.Saturation.N = 2.0
etg.Geom.domain.Saturation.SRes = SRES
etg.Geom.domain.Saturation.SSat = 1.0

etg.Wells.Names = ""

etg.Cycle.Names = "constant"
etg.Cycle.constant.Names = "alltime"
etg.Cycle.constant.alltime.Length = 1
etg.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Closed box: no-flux everywhere
# -----------------------------------------------------------------------------
etg.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"
for patch in ["x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"]:
    etg.pfset(f"Patch.{patch}.BCPressure.Type", "FluxConst")
    etg.pfset(f"Patch.{patch}.BCPressure.Cycle", "constant")
    etg.pfset(f"Patch.{patch}.BCPressure.alltime.Value", 0.0)

etg.TopoSlopesX.Type = "Constant"
etg.TopoSlopesX.GeomNames = "domain"
etg.TopoSlopesX.Geom.domain.Value = 0.0
etg.TopoSlopesY.Type = "Constant"
etg.TopoSlopesY.GeomNames = "domain"
etg.TopoSlopesY.Geom.domain.Value = 0.0
etg.Mannings.Type = "Constant"
etg.Mannings.GeomNames = "domain"
etg.Mannings.Geom.domain.Value = 5.52e-6

etg.PhaseSources.water.Type = "Constant"
etg.PhaseSources.water.GeomNames = "domain"
etg.PhaseSources.water.Geom.domain.Value = 0.0

etg.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
etg.Solver = "Richards"
etg.Solver.MaxIter = 100
etg.Solver.Nonlinear.MaxIter = 250
etg.Solver.Nonlinear.ResidualTol = 1e-9
etg.Solver.Nonlinear.EtaChoice = "EtaConstant"
etg.Solver.Nonlinear.EtaValue = 1e-5
etg.Solver.Nonlinear.UseJacobian = True
etg.Solver.Nonlinear.DerivativeEpsilon = 1e-2
etg.Solver.Nonlinear.StepTol = 1e-30
etg.Solver.Nonlinear.Globalization = "LineSearch"
etg.Solver.Linear.KrylovDimension = 25
etg.Solver.Linear.Preconditioner = "MGSemi"
etg.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
etg.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

etg.Solver.PrintSubsurfData = False
etg.Solver.PrintPressure = True
etg.Solver.PrintSaturation = True
etg.Solver.PrintVelocities = False
etg.Solver.PrintMask = False

# -----------------------------------------------------------------------------
# Initial condition: water table 1.5 m above the bottom
# -----------------------------------------------------------------------------
etg.ICPressure.Type = "HydroStaticPatch"
etg.ICPressure.GeomNames = "domain"
etg.Geom.domain.ICPressure.Value = 1.5
etg.Geom.domain.ICPressure.RefGeom = "domain"
etg.Geom.domain.ICPressure.RefPatch = "z_lower"

# -----------------------------------------------------------------------------
# Steady prescribed flux: strong ET demand in the top layer, nothing below.
# -0.02 1/h over a 0.5 m cell strips the top layer's available water in a few
# hours; resupply through the dry column cannot keep up.
# -----------------------------------------------------------------------------
NX, NY, NZ = 3, 3, 10
ET_TOP = -0.02

etg.Solver.EvapTransFile = True
etg.Solver.EvapTrans.FileName = "et_flux.pfb"


def setup_dir(suffix):
    d = get_absolute_path("test_output/" + run_name + "_" + suffix)
    mkdir(d)
    flux = np.zeros((NZ, NY, NX))
    flux[NZ - 1, :, :] = ET_TOP
    write_pfb(d + "/et_flux.pfb", flux, p=args.p, q=args.q, r=args.r,
              dx=100.0, dy=100.0, dz=0.5, dist=True)
    return d


FINAL_DUMP = 4   # DumpInterval 12 over 48 h: files 00000 (IC) .. 00004 (48 h)


def min_pressure(d, dump):
    p = read_pfb(d + "/" + run_name + ".out.press.%05d.pfb" % dump)
    return float(np.min(p))


def water_volume(d, dump):
    """Total water volume in the closed box [m^3] (porosity 0.4, 100x100x0.5
    cells; specific-storage contribution is negligible at this tolerance)."""
    s = read_pfb(d + "/" + run_name + ".out.satur.%05d.pfb" % dump)
    return float(np.sum(s) * 0.4 * 100.0 * 100.0 * 0.5)


# -----------------------------------------------------------------------------
# C first: guard keys entirely absent (default-off reference)
# -----------------------------------------------------------------------------
dir_c = setup_dir("absent")
etg.run(working_directory=dir_c)

# -----------------------------------------------------------------------------
# A: guard explicitly OFF -- unlimited sink, unphysical suction
# -----------------------------------------------------------------------------
etg.Solver.EvapTransGuard = False
dir_a = setup_dir("off")
etg.run(working_directory=dir_a)
min_p_off = min_pressure(dir_a, FINAL_DUMP)

# -----------------------------------------------------------------------------
# B: guard ON with the CSV log
# -----------------------------------------------------------------------------
etg.Solver.EvapTransGuard = True
etg.Solver.EvapTransGuard.Margin = 0.02
etg.Solver.EvapTransGuard.RampWidth = 0.05
etg.Solver.EvapTransGuard.PrintLog = True
dir_b = setup_dir("on")
etg.run(working_directory=dir_b)
min_p_on = min_pressure(dir_b, FINAL_DUMP)

s_final = read_pfb(dir_b + "/" + run_name + ".out.satur.%05d.pfb" % FINAL_DUMP)
min_s_on = float(np.min(s_final))

csv = np.genfromtxt(dir_b + "/" + run_name + ".out.etguard.csv",
                    delimiter=",", names=True)
applied_total = float(np.sum(csv["applied_sink"]))
withheld_cum = float(csv["withheld_cum"][-1])
prescribed_total = float(np.sum(csv["prescribed_sink"]))
n_limited_max = int(np.max(csv["n_limited"]))
storage_removed = water_volume(dir_b, 0) - water_volume(dir_b, FINAL_DUMP)

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
checks = []


def check(name, ok, detail):
    checks.append(ok)
    print(f"{name}: {detail} -> {'PASS' if ok else 'FAIL'}")


# The unguarded sink must have driven pressure to unphysical suction, and the
# guard must hold it in the physical range (S_stop = 0.12 is ~22 m suction
# for alpha = 2, n = 2).
check("unguarded suction", min_p_off < -100.0, f"min press {min_p_off:.3e} m (expect < -100)")
check("guarded bound", min_p_on > -30.0, f"min press {min_p_on:.3e} m (expect > -30)")
check("guarded saturation floor", min_s_on > SRES + 0.01,
      f"min S {min_s_on:.4f} (expect > {SRES + 0.01})")
check("guard engaged", n_limited_max > 0, f"max n_limited {n_limited_max} (expect > 0)")
check("withheld positive", withheld_cum > 0.0, f"withheld_cum {withheld_cum:.4e} m^3")
check("csv identity", abs(prescribed_total - applied_total - withheld_cum)
      < 1e-6 * prescribed_total,
      f"prescribed - applied - withheld = "
      f"{prescribed_total - applied_total - withheld_cum:.3e} m^3")
check("storage closure", abs(storage_removed - applied_total)
      < 0.02 * max(applied_total, 1.0),
      f"storage removed {storage_removed:.4e} vs applied sink {applied_total:.4e} m^3")

# -----------------------------------------------------------------------------
# Keys absent (C) must be byte-identical to guard False (A) -- default-off
# -----------------------------------------------------------------------------
p_off = read_pfb(dir_a + "/" + run_name + ".out.press.%05d.pfb" % FINAL_DUMP)
p_absent = read_pfb(dir_c + "/" + run_name + ".out.press.%05d.pfb" % FINAL_DUMP)
check("default-off bit-identity", bool(np.all(p_off == p_absent)),
      f"max |diff| {float(np.max(np.abs(p_off - p_absent))):.3e}")

if all(checks):
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
