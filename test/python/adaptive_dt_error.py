# -----------------------------------------------------------------------------
# Gate 3 for adaptive-dt Layer 2 (Solver.AdaptiveDt.ErrorControl and
# Solver.AdaptiveDt.ExtrapolatedGuess).
#
# A 1D infiltration column with a sharp wetting front, run three ways:
#   (a) fine constant-dt reference;
#   (b) geometric-growth dt with ErrorControl ON  -- must track the reference
#       at the final time while taking materially fewer steps, and must engage
#       (change the dt sequence vs the same growth run with ErrorControl OFF);
#   (c) matched constant-dt pair with ExtrapolatedGuess OFF/ON -- the
#       extrapolated Newton guess must not increase total Newton work.
# Plus a drainage column (pressures falling toward/through p=0) with the
# extrapolated guess ON: the p=0 clamp must keep the guess from spuriously
# ponding, so the run converges with no extra solver attempts and lands on
# the same solution as the unextrapolated run.
# -----------------------------------------------------------------------------

import sys
import glob
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb


def parse_kinsol(log_path):
    """Return (total nonlinear iterations, number of KINSOL solve attempts)."""
    nonlin = attempts = 0
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("Nonlin. Its.:"):
                nonlin += int(line.split()[2])
            elif line.startswith("KINSOL starting step"):
                attempts += 1
    return nonlin, attempts


def build_column(name, drainage=False):
    r = Run(name, __file__)
    r.FileVersion = 4
    r.Process.Topology.P = 1
    r.Process.Topology.Q = 1
    r.Process.Topology.R = 1

    r.ComputationalGrid.Lower.X = 0.0
    r.ComputationalGrid.Lower.Y = 0.0
    r.ComputationalGrid.Lower.Z = 0.0
    r.ComputationalGrid.NX = 1
    r.ComputationalGrid.NY = 1
    r.ComputationalGrid.NZ = 100
    r.ComputationalGrid.DX = 1.0
    r.ComputationalGrid.DY = 1.0
    r.ComputationalGrid.DZ = 0.05

    r.GeomInput.Names = "boxinput"
    r.GeomInput.boxinput.InputType = "Box"
    r.GeomInput.boxinput.GeomName = "domain"
    r.Geom.domain.Lower.X = 0.0
    r.Geom.domain.Lower.Y = 0.0
    r.Geom.domain.Lower.Z = 0.0
    r.Geom.domain.Upper.X = 1.0
    r.Geom.domain.Upper.Y = 1.0
    r.Geom.domain.Upper.Z = 5.0
    r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = 0.05
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

    r.TimingInfo.BaseUnit = 0.01
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = 10.0
    r.TimingInfo.DumpInterval = 10.0

    r.Geom.Porosity.GeomNames = "domain"
    r.Geom.domain.Porosity.Type = "Constant"
    r.Geom.domain.Porosity.Value = 0.25
    r.Domain.GeomName = "domain"

    # Sharp van Genuchten front.
    r.Phase.RelPerm.Type = "VanGenuchten"
    r.Phase.RelPerm.GeomNames = "domain"
    r.Geom.domain.RelPerm.Alpha = 2.0
    r.Geom.domain.RelPerm.N = 2.5
    r.Phase.Saturation.Type = "VanGenuchten"
    r.Phase.Saturation.GeomNames = "domain"
    r.Geom.domain.Saturation.Alpha = 2.0
    r.Geom.domain.Saturation.N = 2.5
    r.Geom.domain.Saturation.SRes = 0.1
    r.Geom.domain.Saturation.SSat = 1.0

    r.Wells.Names = ""
    r.Cycle.Names = "constant"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1

    r.BCPressure.PatchNames = r.Geom.domain.Patches
    for p in ["x_lower", "y_lower", "x_upper", "y_upper"]:
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0

    if drainage:
        # Initially wet column draining out the bottom; near-surface pressures
        # fall toward p = 0 from above -- exercises the predictor clamp.
        r.Patch.z_upper.BCPressure.Type = "FluxConst"
        r.Patch.z_upper.BCPressure.Cycle = "constant"
        r.Patch.z_upper.BCPressure.alltime.Value = 0.0
        r.Patch.z_lower.BCPressure.Type = "DirEquilRefPatch"
        r.Patch.z_lower.BCPressure.Cycle = "constant"
        r.Patch.z_lower.BCPressure.RefGeom = "domain"
        r.Patch.z_lower.BCPressure.RefPatch = "z_lower"
        r.Patch.z_lower.BCPressure.alltime.Value = 0.5
        r.ICPressure.Type = "HydroStaticPatch"
        r.ICPressure.GeomNames = "domain"
        r.Geom.domain.ICPressure.Value = 5.1
        r.Geom.domain.ICPressure.RefGeom = "domain"
        r.Geom.domain.ICPressure.RefPatch = "z_lower"
    else:
        # Dry column, constant infiltration at the surface: a sharp downward
        # wetting front whose speed sets the truncation error.
        r.Patch.z_upper.BCPressure.Type = "FluxConst"
        r.Patch.z_upper.BCPressure.Cycle = "constant"
        r.Patch.z_upper.BCPressure.alltime.Value = -0.02
        r.Patch.z_lower.BCPressure.Type = "FluxConst"
        r.Patch.z_lower.BCPressure.Cycle = "constant"
        r.Patch.z_lower.BCPressure.alltime.Value = 0.0
        r.ICPressure.Type = "Constant"
        r.ICPressure.GeomNames = "domain"
        r.Geom.domain.ICPressure.Value = -3.0

    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = 0.0
    r.TopoSlopesY.Type = "Constant"
    r.TopoSlopesY.GeomNames = "domain"
    r.TopoSlopesY.Geom.domain.Value = 0.0
    r.Mannings.Type = "Constant"
    r.Mannings.GeomNames = "domain"
    r.Mannings.Geom.domain.Value = 1.0e-6

    r.PhaseSources.water.Type = "Constant"
    r.PhaseSources.water.GeomNames = "domain"
    r.PhaseSources.water.Geom.domain.Value = 0.0
    r.KnownSolution = "NoKnownSolution"

    r.Solver = "Richards"
    r.Solver.MaxIter = 250000
    r.Solver.Nonlinear.MaxIter = 25
    r.Solver.Nonlinear.ResidualTol = 1e-8
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-4
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
    return r


def set_constant_dt(r, dt):
    r.TimeStep.Type = "Constant"
    r.TimeStep.Value = dt


def set_growth_dt(r):
    r.TimeStep.Type = "Growth"
    r.TimeStep.InitialStep = 0.01
    r.TimeStep.GrowthFactor = 1.5
    r.TimeStep.MaxStep = 2.0
    r.TimeStep.MinStep = 1.0e-5


def run_case(r, name):
    outdir = get_absolute_path(f"test_output/{name}")
    mkdir(outdir)
    r.run(working_directory=outdir)
    logs = glob.glob(f"{outdir}/{name}.out.kinsol.log")
    nonlin, attempts = parse_kinsol(logs[0]) if logs else (-1, -1)
    press_files = sorted(glob.glob(f"{outdir}/{name}.out.press.*.pfb"))
    final_press = read_pfb(press_files[-1]).flatten() if press_files else None
    return nonlin, attempts, final_press


def main():
    failed = False

    # ---- (a)+(b): error control tracks a fine-dt reference cheaply -------
    ref = build_column("adapt_err_ref")
    set_constant_dt(ref, 0.025)
    ref_nl, ref_at, ref_p = run_case(ref, "adapt_err_ref")

    blind = build_column("adapt_err_blind")
    set_growth_dt(blind)
    blind_nl, blind_at, blind_p = run_case(blind, "adapt_err_blind")

    errc = build_column("adapt_err_on")
    set_growth_dt(errc)
    errc.Solver.AdaptiveDt = True
    errc.Solver.AdaptiveDt.ErrorControl = True
    errc.Solver.AdaptiveDt.ErrorControl.RelTol = 1.0e-2
    errc.Solver.AdaptiveDt.ErrorControl.AbsTol = 1.0e-3
    errc_nl, errc_at, errc_p = run_case(errc, "adapt_err_on")

    def maxdiff(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    def rmsdiff(a, b):
        return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5

    # RMS is the right tracking metric: the controller regulates a weighted
    # RMS norm, and a max norm at a sharp moving front punishes phase lag.
    errc_err = rmsdiff(errc_p, ref_p)
    blind_err = rmsdiff(blind_p, ref_p)
    print(f"reference    : {ref_at} steps ({ref_nl} Newton)")
    print(f"blind growth : {blind_at} attempts ({blind_nl} Newton), "
          f"rms|p-ref| = {blind_err:.4e}")
    print(f"ErrorControl : {errc_at} attempts ({errc_nl} Newton), "
          f"rms|p-ref| = {errc_err:.4e}")

    engaged = errc_at != blind_at
    tracks = errc_err <= 0.05 and errc_err <= blind_err / 3.0
    cheaper = errc_at <= 0.5 * ref_at
    print(f"engages (dt sequence changed)      : {engaged}")
    print(f"tracks ref (rms <= 5 cm, 3x blind) : {tracks}")
    print(f"materially fewer steps             : {cheaper}")
    failed |= not (engaged and tracks and cheaper)

    # ---- (c): extrapolated guess does not increase Newton work -----------
    base = build_column("adapt_guess_off")
    set_constant_dt(base, 0.1)
    off_nl, off_at, off_p = run_case(base, "adapt_guess_off")

    extr = build_column("adapt_guess_on")
    set_constant_dt(extr, 0.1)
    extr.Solver.AdaptiveDt = True
    extr.Solver.AdaptiveDt.ExtrapolatedGuess = True
    on_nl, on_at, on_p = run_case(extr, "adapt_guess_on")

    guess_diff = maxdiff(on_p, off_p)
    print(f"guess OFF : {off_nl} Newton in {off_at} steps")
    print(f"guess ON  : {on_nl} Newton in {on_at} steps, "
          f"max|p_on-p_off| = {guess_diff:.4e}")
    no_extra_work = on_nl <= off_nl and on_at == off_at
    same_answer = guess_diff <= 1.0e-4
    print(f"no extra Newton work : {no_extra_work}")
    print(f"same solution        : {same_answer}")
    failed |= not (no_extra_work and same_answer)

    # ---- drainage: the p=0 clamp keeps the extrapolated guess safe -------
    doff = build_column("adapt_drain_off", drainage=True)
    set_constant_dt(doff, 0.1)
    doff_nl, doff_at, doff_p = run_case(doff, "adapt_drain_off")

    don = build_column("adapt_drain_on", drainage=True)
    set_constant_dt(don, 0.1)
    don.Solver.AdaptiveDt = True
    don.Solver.AdaptiveDt.ExtrapolatedGuess = True
    don_nl, don_at, don_p = run_case(don, "adapt_drain_on")

    drain_diff = maxdiff(don_p, doff_p)
    print(f"drainage OFF : {doff_nl} Newton in {doff_at} attempts")
    print(f"drainage ON  : {don_nl} Newton in {don_at} attempts, "
          f"max|p_on-p_off| = {drain_diff:.4e}")
    drain_safe = don_at == doff_at and don_nl <= doff_nl + 2
    drain_same = drain_diff <= 1.0e-4
    print(f"no retries / no extra work : {drain_safe}")
    print(f"same drainage solution     : {drain_same}")
    failed |= not (drain_safe and drain_same)

    if failed:
        print("adaptive_dt_error : FAILED")
        sys.exit(1)
    print("adaptive_dt_error : PASSED")


if __name__ == "__main__":
    main()
