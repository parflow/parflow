# -----------------------------------------------------------------------------
# OFFLINE performance comparison for the TFG smoothed-upwinding feature.
#
# This is NOT a pass/fail regression gate -- it is an offline benchmark that
# follows the accel_factorial_experiment_plan.md "S factor" strategy: run the
# same terrain-following-grid problem with the smoother OFF
# (Solver.TerrainFollowingGrid.UpwindEpsilon = 0.0) and ON (= eps*), across a
# slope sweep, and report solver-work metrics (nonlinear/linear iterations,
# beta-condition failures, backtracks) parsed from the KINSOL log.
#
# Expected signature (plan prediction): the smoother helps most at intermediate
# slopes, where the TFG upwind switch sits between cells in different saturation
# states and the Jacobian kink is largest; flat and very steep slopes change
# little.  Run manually:  python3 tfg_smooth_performance.py
#
# CAVEAT: this idealized case is laterally HOMOGENEOUS (uniform perm/slope), so
# adjacent cells across the switch never differ by orders of magnitude in
# mobility -- the regime that triggers the KINSOL beta-condition failures the
# smoother removes.  On such benign cases the smoother is (correctly) ~neutral:
# use this harness to confirm it does not regress and to sanity-check the metric
# parsing.  The real tail-percentile speedup is measured by the factorial study
# (accel_factorial_experiment_plan.md) on the heterogeneous UCRB/CONUS2 storm
# windows; point build_run() at a heterogeneous perm field or a storm-onset
# forcing to reproduce the pathology here.
# -----------------------------------------------------------------------------

import glob
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path

# Slope sweep and the "on" epsilon (eps*), per the plan's hillslope_sweep.
SLOPES = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
EPS_STAR = 1.0e-3


def parse_kinsol(log_path):
    """Sum per-step KINSOL work counters over the whole run."""
    nonlin = lin = beta = back = 0
    with open(log_path) as fh:
        for line in fh:
            tok = line.split()
            if line.startswith("Nonlin. Its.:"):
                nonlin += int(tok[2])
            elif line.startswith("Lin. Its.:"):
                lin += int(tok[2])
            elif line.startswith("Beta Cond. Fails:"):
                beta += int(tok[3])
            elif line.startswith("Backtracks:"):
                back += int(tok[1])
    return nonlin, lin, beta, back


def build_run(slope, eps, name):
    r = Run(name, __file__)
    r.FileVersion = 4
    r.Process.Topology.P = 1
    r.Process.Topology.Q = 1
    r.Process.Topology.R = 1

    r.ComputationalGrid.Lower.X = 0.0
    r.ComputationalGrid.Lower.Y = 0.0
    r.ComputationalGrid.Lower.Z = 0.0
    r.ComputationalGrid.NX = 20
    r.ComputationalGrid.NY = 1
    r.ComputationalGrid.NZ = 30
    r.ComputationalGrid.DX = 5.0
    r.ComputationalGrid.DY = 1.0
    r.ComputationalGrid.DZ = 0.05

    r.GeomInput.Names = "boxinput"
    r.GeomInput.boxinput.InputType = "Box"
    r.GeomInput.boxinput.GeomName = "domain"
    r.Geom.domain.Lower.X = 0.0
    r.Geom.domain.Lower.Y = 0.0
    r.Geom.domain.Lower.Z = 0.0
    r.Geom.domain.Upper.X = 100.0
    r.Geom.domain.Upper.Y = 1.0
    r.Geom.domain.Upper.Z = 1.5
    r.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    r.Geom.Perm.Names = "domain"
    r.Geom.domain.Perm.Type = "Constant"
    r.Geom.domain.Perm.Value = 10.0
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

    r.TimingInfo.BaseUnit = 1.0
    r.TimingInfo.StartCount = 0
    r.TimingInfo.StartTime = 0.0
    r.TimingInfo.StopTime = 2.0
    r.TimingInfo.DumpInterval = -1
    r.TimeStep.Type = "Constant"
    r.TimeStep.Value = 0.1

    r.Geom.Porosity.GeomNames = "domain"
    r.Geom.domain.Porosity.Type = "Constant"
    r.Geom.domain.Porosity.Value = 0.1
    r.Domain.GeomName = "domain"

    r.Phase.RelPerm.Type = "VanGenuchten"
    r.Phase.RelPerm.GeomNames = "domain"
    r.Geom.domain.RelPerm.Alpha = 6.0
    r.Geom.domain.RelPerm.N = 2.0
    r.Phase.Saturation.Type = "VanGenuchten"
    r.Phase.Saturation.GeomNames = "domain"
    r.Geom.domain.Saturation.Alpha = 6.0
    r.Geom.domain.Saturation.N = 2.0
    r.Geom.domain.Saturation.SRes = 0.2
    r.Geom.domain.Saturation.SSat = 1.0

    r.Wells.Names = ""

    r.Cycle.Names = "constant rainrec"
    r.Cycle.constant.Names = "alltime"
    r.Cycle.constant.alltime.Length = 1
    r.Cycle.constant.Repeat = -1
    r.Cycle.rainrec.Names = "rain rec"
    r.Cycle.rainrec.rain.Length = 1
    r.Cycle.rainrec.rec.Length = 1
    r.Cycle.rainrec.Repeat = -1

    r.BCPressure.PatchNames = r.Geom.domain.Patches
    for p in ["x_lower", "y_lower", "z_lower", "x_upper", "y_upper"]:
        r.Patch[p].BCPressure.Type = "FluxConst"
        r.Patch[p].BCPressure.Cycle = "constant"
        r.Patch[p].BCPressure.alltime.Value = 0.0
    # Rainfall pulse then recession on the top overland-flow face.
    r.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    r.Patch.z_upper.BCPressure.Cycle = "rainrec"
    r.Patch.z_upper.BCPressure.rain.Value = -0.005
    r.Patch.z_upper.BCPressure.rec.Value = 0.0

    # The swept slope.
    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = slope
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
    r.Solver.TerrainFollowingGrid = True
    # The feature under test: 0.0 = hard upwind (OFF), eps* = smoothed (ON).
    r.Solver.TerrainFollowingGrid.UpwindEpsilon = eps
    r.Solver.MaxIter = 2500
    r.Solver.Nonlinear.MaxIter = 300
    r.Solver.Nonlinear.ResidualTol = 1e-6
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-5
    r.Solver.Nonlinear.UseJacobian = True
    r.Solver.Nonlinear.DerivativeEpsilon = 1e-12
    r.Solver.Nonlinear.StepTol = 1e-20
    r.Solver.Nonlinear.Globalization = "LineSearch"
    r.Solver.Linear.KrylovDimension = 20
    r.Solver.Linear.MaxRestart = 2
    r.Solver.Linear.Preconditioner = "PFMG"
    r.Solver.Linear.Preconditioner.SymmetricMat = "Symmetric"
    r.Solver.PrintSubsurf = False
    r.Solver.Drop = 1e-20
    r.Solver.AbsTol = 1e-12

    r.ICPressure.Type = "HydroStaticPatch"
    r.ICPressure.GeomNames = "domain"
    r.Geom.domain.ICPressure.Value = 1.0
    r.Geom.domain.ICPressure.RefGeom = "domain"
    r.Geom.domain.ICPressure.RefPatch = "z_lower"
    return r


def main():
    print("TFG smoothed-upwinding offline performance sweep (analytic Jacobian)")
    print(f"eps* = {EPS_STAR:g}; slope sweep {SLOPES}\n")
    hdr = f"{'slope':>8} {'eps':>8} {'Newton':>8} {'Linear':>8} {'beta':>6} {'backtr':>7}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for slope in SLOPES:
        for eps in (0.0, EPS_STAR):
            tag = f"s{slope:g}_e{eps:g}".replace(".", "p").replace("-", "m")
            name = f"tfg_perf_{tag}"
            outdir = get_absolute_path(f"test_output/{name}")
            mkdir(outdir)
            r = build_run(slope, eps, name)
            r.run(working_directory=outdir)
            logs = glob.glob(f"{outdir}/{name}.out.kinsol.log")
            nn, li, bf, bt = parse_kinsol(logs[0]) if logs else (-1, -1, -1, -1)
            results[(slope, eps)] = (nn, li, bf, bt)
            print(f"{slope:>8g} {eps:>8g} {nn:>8} {li:>8} {bf:>6} {bt:>7}")

    print("\nWith/without comparison (Newton iterations, off -> on):")
    print(f"{'slope':>8} {'off':>8} {'on':>8} {'delta':>8} {'ratio':>8}")
    for slope in SLOPES:
        off = results[(slope, 0.0)][0]
        on = results[(slope, EPS_STAR)][0]
        ratio = (on / off) if off else float("nan")
        print(f"{slope:>8g} {off:>8} {on:>8} {on - off:>8} {ratio:>8.3f}")


if __name__ == "__main__":
    main()
