# -----------------------------------------------------------------------------
# Correctness gates for the TFG smoothed-upwinding feature
# (Solver.TerrainFollowingGrid.UpwindEpsilon).  Self-checking: asserts
# relationships between runs, so it needs no stored baselines.
#
#   Gate 2 (Jacobian consistency): at eps = 1e-3, the analytic-Jacobian solve
#     and the finite-difference-Jacobian solve must converge to the same
#     pressure field.  A wrong chain term in the Jacobian would make the analytic
#     run take a different path or fail; agreement to solver tolerance confirms
#     the blended Jacobian is exact.
#
#   Gate 3 (eps convergence): the smoothed solution must approach the eps = 0
#     (hard-upwind) solution as eps shrinks, and be below solver tolerance for
#     eps <= 1e-3.  This establishes the usable eps ceiling.
# -----------------------------------------------------------------------------

import sys
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb
from parflow.tools import hydrology


def build_run(eps, use_jacobian, name, closed=False):
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
    if closed:
        # Fully closed domain: every boundary no-flux, no rain/overland.  Total
        # subsurface storage must be exactly conserved as water redistributes
        # laterally under TFG -- this is the conservation guard (Gate 5).
        r.Patch.z_upper.BCPressure.Type = "FluxConst"
        r.Patch.z_upper.BCPressure.Cycle = "constant"
        r.Patch.z_upper.BCPressure.alltime.Value = 0.0
    else:
        r.Patch.z_upper.BCPressure.Type = "OverlandFlow"
        r.Patch.z_upper.BCPressure.Cycle = "rainrec"
        r.Patch.z_upper.BCPressure.rain.Value = -0.005
        r.Patch.z_upper.BCPressure.rec.Value = 0.0

    # Intermediate slope -- where the TFG upwind switch is most active.
    r.TopoSlopesX.Type = "Constant"
    r.TopoSlopesX.GeomNames = "domain"
    r.TopoSlopesX.Geom.domain.Value = 0.01
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
    r.Solver.TerrainFollowingGrid.UpwindEpsilon = eps
    r.Solver.MaxIter = 2500
    r.Solver.Nonlinear.MaxIter = 300
    r.Solver.Nonlinear.ResidualTol = 1e-6
    r.Solver.Nonlinear.EtaChoice = "EtaConstant"
    r.Solver.Nonlinear.EtaValue = 1e-5
    r.Solver.Nonlinear.UseJacobian = use_jacobian
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


def run_case(eps, use_jacobian, tag, closed=False):
    name = f"tfg_smooth_{tag}"
    outdir = get_absolute_path(f"test_output/{name}")
    mkdir(outdir)
    r = build_run(eps, use_jacobian, name, closed=closed)
    r.run(working_directory=outdir)
    return outdir, name


def final_pressure(eps, use_jacobian, tag):
    outdir, name = run_case(eps, use_jacobian, tag)
    # StopTime 2.0 / dt 0.1 -> 20 dumped steps.
    return read_pfb(f"{outdir}/{name}.out.press.00020.pfb")


def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def conservation_drift(eps, tag):
    """Max relative drift of total subsurface storage over a closed-domain run."""
    outdir, name = run_case(eps, True, tag, closed=True)
    nx, ny, nz = 20, 1, 30
    dx, dy, dz = 5.0, 1.0, 0.05
    porosity = np.full((nz, ny, nx), 0.1)
    specific_storage = np.full((nz, ny, nx), 1.0e-5)
    mask = np.ones((nz, ny, nx))
    totals = []
    for i in range(21):
        p = read_pfb(f"{outdir}/{name}.out.press.{i:05d}.pfb")
        s = read_pfb(f"{outdir}/{name}.out.satur.{i:05d}.pfb")
        storage = hydrology.calculate_subsurface_storage(
            porosity, p, s, specific_storage, dx, dy, np.array([dz] * nz), mask
        )
        totals.append(float(np.sum(storage)))
    totals = np.array(totals)
    return float(np.max(np.abs(totals - totals[0])) / totals[0])


def main():
    passed = True

    # Baseline: hard upwind (eps = 0).
    p0 = final_pressure(0.0, False, "eps0")

    # --- Gate 2: analytic vs finite-difference Jacobian at eps = 1e-3 ---
    p_analytic = final_pressure(1e-3, True, "e1e3_jac")
    p_fd = final_pressure(1e-3, False, "e1e3_fd")
    jac_diff = max_abs_diff(p_analytic, p_fd)
    gate2_tol = 1e-4
    ok2 = jac_diff < gate2_tol
    passed = passed and ok2
    print(f"Gate 2 (Jacobian consistency): max|analytic - FD| = {jac_diff:.3e} "
          f"(tol {gate2_tol:.0e}) -> {'PASS' if ok2 else 'FAIL'}")

    # --- Gate 3: eps convergence toward the eps = 0 solution ---
    d_1em3 = max_abs_diff(p_fd, p0)                       # reuse the eps=1e-3 FD run
    p_1em4 = final_pressure(1e-4, False, "e1e4")
    d_1em4 = max_abs_diff(p_1em4, p0)
    gate3_tol = 1e-3
    monotone = d_1em4 <= d_1em3
    small = d_1em3 < gate3_tol
    ok3 = monotone and small
    passed = passed and ok3
    print(f"Gate 3 (eps convergence): |p(1e-3)-p(0)| = {d_1em3:.3e}, "
          f"|p(1e-4)-p(0)| = {d_1em4:.3e}; monotone={monotone}, "
          f"below tol({gate3_tol:.0e})={small} -> {'PASS' if ok3 else 'FAIL'}")

    # --- Gate 5: conservation on a closed domain at eps = 1e-3 ---
    drift = conservation_drift(1e-3, "closed")
    gate5_tol = 1e-6
    ok5 = drift < gate5_tol
    passed = passed and ok5
    print(f"Gate 5 (conservation, closed domain): max relative storage drift "
          f"= {drift:.3e} (tol {gate5_tol:.0e}) -> {'PASS' if ok5 else 'FAIL'}")

    if passed:
        print("tfg_smooth_upwind : PASSED")
    else:
        print("tfg_smooth_upwind : FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
