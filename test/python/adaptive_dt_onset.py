# -----------------------------------------------------------------------------
# Gate 5 for adaptive-dt Layer 4 (Solver.AdaptiveDt.OnsetControl).
#
# A flat domain driven by transient evap_trans forcing: quiet for 4 forcing
# intervals, then a strong storm (rain >> Ksat onto dry soil) beginning exactly
# at a forcing-interval boundary, entered blind at a base dt equal to the
# forcing interval.  OnsetControl reads the incoming evap_trans before the
# solve, sees the capacity-exceedance jump, and caps that one step's dt.
#
# On this benign uniform case Newton+LineSearch converges the blind entry step
# right at the MaxIter cliff instead of failing (the Layer 1 gate's
# idealized-case caveat: failure avoidance shows on real domains), so the
# verifiable Layer 4 behaviors are detection, regulation, and headroom:
# the 'o' cap fires exactly once (at the storm boundary) then releases; the
# blind entry step sits at the failure edge (>= MaxIter - 1 iterations) while
# the capped entry step costs strictly fewer; the onset run has no convergence
# failures.  A third run with the extrapolated guess on verifies the
# onset-step guess suppression (the onset step converges without a retry) and
# that the run completes.
# -----------------------------------------------------------------------------

import sys
import glob
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import write_pfb


NX = NY = 8
NZ = 24
DX = DY = 2.0
DZ = 0.1
STOP = 8.0          # 8 forcing intervals of length 1.0 (constant base dt)
STORM_FILE = 4      # forcing file index where the storm begins (t = 4.0)
RAIN = 1.0          # [m/h] incoming rain rate, 200x Ksat


def parse_kinsol(log_path):
    """Return (total Newton its, attempts, per-attempt (time, its) list)."""
    nonlin = attempts = 0
    per_attempt = []
    start_t = None
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("KINSOL starting step"):
                attempts += 1
                start_t = float(line.split()[-1])
            elif line.startswith("Nonlin. Its.:"):
                its = int(line.split()[2])
                nonlin += its
                per_attempt.append((start_t, its))
    return nonlin, attempts, per_attempt


def parse_outlog(log_path):
    """Return list of (time, dt, dt_info) from the .out.log step table."""
    rows = []
    with open(log_path) as fh:
        in_table = False
        for line in fh:
            if line.startswith("Sequence #"):
                in_table = True
                continue
            if in_table:
                tok = line.split()
                if len(tok) >= 4 and tok[0].isdigit():
                    rows.append((float(tok[1]), float(tok[2]), tok[3]))
                elif tok and not tok[0].startswith("-"):
                    in_table = False
    return rows


def write_forcing(outdir):
    """Quiet evap_trans files, then storm files from STORM_FILE on.  Rain is
    applied as a source in the top layer: et = RAIN / (DZ * dz_mult) [1/h]."""
    quiet = np.zeros((NZ, NY, NX))
    storm = np.zeros((NZ, NY, NX))
    storm[NZ - 1, :, :] = RAIN / DZ
    for n in range(int(STOP) + 2):
        arr = storm if n >= STORM_FILE else quiet
        write_pfb(f"{outdir}/forcing.{n:05d}.pfb", arr, dx=DX, dy=DY, dz=DZ)


def build_run(name):
    r = Run(name, __file__)
    r.FileVersion = 4
    r.Process.Topology.P = 1
    r.Process.Topology.Q = 1
    r.Process.Topology.R = 1

    r.ComputationalGrid.Lower.X = 0.0
    r.ComputationalGrid.Lower.Y = 0.0
    r.ComputationalGrid.Lower.Z = 0.0
    r.ComputationalGrid.NX = NX
    r.ComputationalGrid.NY = NY
    r.ComputationalGrid.NZ = NZ
    r.ComputationalGrid.DX = DX
    r.ComputationalGrid.DY = DY
    r.ComputationalGrid.DZ = DZ

    r.GeomInput.Names = "boxinput"
    r.GeomInput.boxinput.InputType = "Box"
    r.GeomInput.boxinput.GeomName = "domain"
    r.Geom.domain.Lower.X = 0.0
    r.Geom.domain.Lower.Y = 0.0
    r.Geom.domain.Lower.Z = 0.0
    r.Geom.domain.Upper.X = NX * DX
    r.Geom.domain.Upper.Y = NY * DY
    r.Geom.domain.Upper.Z = NZ * DZ
    r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = 0.005
    r.Perm.TensorType = "TensorByGeom"
    r.Geom.Perm.TensorByGeom.Names = "domain"
    r.Geom.domain.Perm.TensorValX = 1.0
    r.Geom.domain.Perm.TensorValY = 1.0
    r.Geom.domain.Perm.TensorValZ = 1.0

    r.SpecificStorage.Type = "Constant"
    r.SpecificStorage.GeomNames = "domain"
    r.Geom.domain.SpecificStorage.Value = 1.0e-5

    r.Phase.Names = "water"
    r.Phase.water.Density.Type = "Constant"
    r.Phase.water.Density.Value = 1.0
    r.Phase.water.Viscosity.Type = "Constant"
    r.Phase.water.Viscosity.Value = 1.0
    r.Contaminants.Names = ""
    r.Geom.Retardation.GeomNames = ""
    r.Gravity = 1.0

    # Constant base dt equal to the forcing interval; the forcing clock (ct)
    # advances by this proposal, so all runs see the storm at the same time.
    r.TimingInfo.BaseUnit = 0.05
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = STOP
    r.TimingInfo.DumpInterval = 2.0
    r.TimeStep.Type = "Constant"
    r.TimeStep.Value = 1.0

    r.Geom.Porosity.GeomNames = "domain"
    r.Geom.domain.Porosity.Type = "Constant"
    r.Geom.domain.Porosity.Value = 0.25
    r.Domain.GeomName = "domain"

    r.Phase.RelPerm.Type = "VanGenuchten"
    r.Phase.RelPerm.GeomNames = "domain"
    r.Geom.domain.RelPerm.Alpha = 3.0
    r.Geom.domain.RelPerm.N = 3.0
    r.Phase.Saturation.Type = "VanGenuchten"
    r.Phase.Saturation.GeomNames = "domain"
    r.Geom.domain.Saturation.Alpha = 3.0
    r.Geom.domain.Saturation.N = 3.0
    r.Geom.domain.Saturation.SRes = 0.1
    r.Geom.domain.Saturation.SSat = 1.0

    r.Wells.Names = ""
    r.Cycle.Names = "constant"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1

    r.BCPressure.PatchNames = r.Geom.domain.Patches
    for p in ["x_lower", "y_lower", "x_upper", "y_upper", "z_lower"]:
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    r.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    r.Patch.z_upper.BCPressure.Cycle = "constant"
    r.Patch.z_upper.BCPressure.alltime.Value = 0.0

    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = 0.01
    r.TopoSlopesY.Type = "Constant"
    r.TopoSlopesY.GeomNames = "domain"
    r.TopoSlopesY.Geom.domain.Value = 0.0
    r.Mannings.Type = "Constant"
    r.Mannings.GeomNames = "domain"
    r.Mannings.Geom.domain.Value = 5.0e-6

    r.PhaseSources.water.Type = "Constant"
    r.PhaseSources.water.GeomNames = "domain"
    r.PhaseSources.water.Geom.domain.Value = 0.0
    r.KnownSolution = "NoKnownSolution"

    r.Solver = "Richards"
    # Storm forcing arrives through transient evap_trans files.
    r.Solver.EvapTransFileTransient = True
    r.Solver.EvapTrans.FileName = "forcing"
    r.Solver.MaxIter = 250000
    r.Solver.Nonlinear.MaxIter = 10
    r.Solver.Nonlinear.ResidualTol = 1e-6
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-3
    r.Solver.Nonlinear.UseJacobian = True
    r.Solver.Nonlinear.DerivativeEpsilon = 1e-12
    r.Solver.Nonlinear.StepTol = 1e-20
    r.Solver.Nonlinear.Globalization = "LineSearch"
    r.Solver.Linear.KrylovDimension = 20
    r.Solver.Linear.MaxRestart = 2
    r.Solver.Linear.Preconditioner = "PFMG"
    r.Solver.PrintSubsurf = False
    r.Solver.Drop = 1e-20
    r.Solver.AbsTol = 1e-12

    r.ICPressure.Type = "Constant"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.Value = -5.0
    return r


MAX_ITER = 10       # Solver.Nonlinear.MaxIter; blind entry must sit at its edge


def run_case(name, onset=False, extrap=False):
    outdir = get_absolute_path(f"test_output/{name}")
    mkdir(outdir)
    write_forcing(outdir)
    r = build_run(name)
    if onset or extrap:
        r.Solver.AdaptiveDt = True
    if onset:
        r.Solver.AdaptiveDt.OnsetControl = True
        r.Solver.AdaptiveDt.OnsetControl.FillHorizon = 1.0
    if extrap:
        r.Solver.AdaptiveDt.ExtrapolatedGuess = True
    r.run(working_directory=outdir)
    nonlin, attempts, per_att = parse_kinsol(f"{outdir}/{name}.out.kinsol.log")
    rows = parse_outlog(f"{outdir}/{name}.out.log")
    steps = [row for row in rows if row[2] != "i"]
    failures = attempts - len(steps)
    # storm entry = first solve attempt targeting a time past the boundary
    entry_its = next(its for (t, its) in per_att if t > STORM_FILE)
    reached_end = abs(rows[-1][0] - STOP) < 1e-6
    return nonlin, failures, entry_its, reached_end, rows


def main():
    b_nl, b_fail, b_entry, b_end, b_rows = run_case("adapt_onset_off")
    o_nl, o_fail, o_entry, o_end, o_rows = run_case("adapt_onset_on",
                                                    onset=True)
    g_nl, g_fail, g_entry, g_end, g_rows = run_case("adapt_onset_extrap",
                                                    onset=True, extrap=True)

    o_fires = [(t, dt) for (t, dt, c) in o_rows if c == "o"]
    print(f"blind   : {b_nl} Newton, {b_fail} failures, "
          f"entry step {b_entry}/{MAX_ITER} its")
    print(f"onset   : {o_nl} Newton, {o_fail} failures, "
          f"entry step {o_entry}/{MAX_ITER} its, 'o' fired at {o_fires}")
    print(f"+extrap : {g_nl} Newton, {g_fail} failures, "
          f"entry step {g_entry}/{MAX_ITER} its")

    # The 'o' row logs the post-step time: storm start (t=4) + capped dt.
    fires_once_at_storm = (len(o_fires) == 1 and
                           STORM_FILE < o_fires[0][0] <= STORM_FILE + 0.5)
    blind_at_cliff = b_entry >= MAX_ITER - 1
    regulated = o_entry < b_entry
    onset_clean = o_fail == 0 and o_end
    # suppression must get the onset step itself through without a retry:
    # exactly one attempt in (storm, storm + 0.5], i.e. no repeated target
    g_onset_attempts = len([1 for (t, dt, c) in g_rows
                            if STORM_FILE < t <= STORM_FILE + 0.5])
    extrap_ok = g_onset_attempts == 1 and g_end

    print(f"'o' fires exactly once, at storm boundary : {fires_once_at_storm}")
    print(f"blind entry at the MaxIter cliff          : {blind_at_cliff}")
    print(f"capped entry strictly easier              : {regulated}")
    print(f"onset run clean and completes             : {onset_clean}")
    print(f"guess suppressed at onset, run completes  : {extrap_ok}")
    failed = not (fires_once_at_storm and blind_at_cliff and regulated
                  and onset_clean and extrap_ok)

    if failed:
        print("adaptive_dt_onset : FAILED")
        sys.exit(1)
    print("adaptive_dt_onset : PASSED")


if __name__ == "__main__":
    main()
