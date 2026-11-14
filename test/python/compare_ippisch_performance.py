#!/usr/bin/env python3
"""
Compare solver performance and field accuracy across 4 configurations:
  1. Standard VG, direct eval (no table)
  2. Standard VG, spline table
  3. Ippisch VG (InverseAlpha), direct eval
  4. Ippisch VG (InverseAlpha), spline table

Reports:
  - KINSOL iteration counts per timestep
  - Total nonlinear/linear iterations
  - Max/mean pressure and saturation differences (table vs direct)
"""

import sys, os, re, argparse
import numpy as np
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path
from parflow.tools.io import read_pfb

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VG_alpha = 1.0
VG_N_standard = 2.0
VG_N_ippisch = 1.5
VG_points = 20000

s_res_vals = {
    "zone1": 0.2771,
    "zone2": 0.2806,
    "zone3above4": 0.2643,
    "zone3left4": 0.2643,
    "zone3right4": 0.2643,
    "zone3below4": 0.2643,
    "zone4": 0.2643,
}

Zones = "zone1 zone2 zone3above4 zone3left4 zone3right4 zone3below4 zone4"


def build_run(run_name, dir_name, vg_n, use_table, use_ippisch):
    """Build a crater2D ParFlow run with the given configuration.
    use_table: True (both), False (neither), "kr_only", "sat_only"
    """
    crater = Run(run_name, __file__)

    cp("$PF_SRC/test/input/crater2D.pfsol", dir_name)

    crater.FileVersion = 4
    crater.Process.Topology.P = 1
    crater.Process.Topology.Q = 1
    crater.Process.Topology.R = 1

    # Grid
    crater.ComputationalGrid.Lower.X = 0.0
    crater.ComputationalGrid.Lower.Y = 0.0
    crater.ComputationalGrid.Lower.Z = 0.0
    crater.ComputationalGrid.NX = 100
    crater.ComputationalGrid.NY = 1
    crater.ComputationalGrid.NZ = 100
    crater.ComputationalGrid.DX = 4.0
    crater.ComputationalGrid.DY = 1.0
    crater.ComputationalGrid.DZ = 2.0

    # Geometry
    crater.GeomInput.Names = f"solidinput {Zones} background"
    crater.GeomInput.solidinput.InputType = "SolidFile"
    crater.GeomInput.solidinput.GeomNames = "domain"
    crater.GeomInput.solidinput.FileName = "crater2D.pfsol"

    crater.GeomInput.zone1.InputType = "Box"
    crater.GeomInput.zone1.GeomName = "zone1"
    crater.Geom.zone1.Lower.X = 0.0
    crater.Geom.zone1.Lower.Y = 0.0
    crater.Geom.zone1.Lower.Z = 0.0
    crater.Geom.zone1.Upper.X = 400.0
    crater.Geom.zone1.Upper.Y = 1.0
    crater.Geom.zone1.Upper.Z = 200.0

    crater.GeomInput.zone2.InputType = "Box"
    crater.GeomInput.zone2.GeomName = "zone2"
    crater.Geom.zone2.Lower.X = 0.0
    crater.Geom.zone2.Lower.Y = 0.0
    crater.Geom.zone2.Lower.Z = 60.0
    crater.Geom.zone2.Upper.X = 200.0
    crater.Geom.zone2.Upper.Y = 1.0
    crater.Geom.zone2.Upper.Z = 80.0

    crater.GeomInput.zone3above4.InputType = "Box"
    crater.GeomInput.zone3above4.GeomName = "zone3above4"
    crater.Geom.zone3above4.Lower.X = 0.0
    crater.Geom.zone3above4.Lower.Y = 0.0
    crater.Geom.zone3above4.Lower.Z = 180.0
    crater.Geom.zone3above4.Upper.X = 200.0
    crater.Geom.zone3above4.Upper.Y = 1.0
    crater.Geom.zone3above4.Upper.Z = 200.0

    crater.GeomInput.zone3left4.InputType = "Box"
    crater.GeomInput.zone3left4.GeomName = "zone3left4"
    crater.Geom.zone3left4.Lower.X = 0.0
    crater.Geom.zone3left4.Lower.Y = 0.0
    crater.Geom.zone3left4.Lower.Z = 190.0
    crater.Geom.zone3left4.Upper.X = 100.0
    crater.Geom.zone3left4.Upper.Y = 1.0
    crater.Geom.zone3left4.Upper.Z = 200.0

    crater.GeomInput.zone3right4.InputType = "Box"
    crater.GeomInput.zone3right4.GeomName = "zone3right4"
    crater.Geom.zone3right4.Lower.X = 30.0
    crater.Geom.zone3right4.Lower.Y = 0.0
    crater.Geom.zone3right4.Lower.Z = 90.0
    crater.Geom.zone3right4.Upper.X = 80.0
    crater.Geom.zone3right4.Upper.Y = 1.0
    crater.Geom.zone3right4.Upper.Z = 100.0

    crater.GeomInput.zone3below4.InputType = "Box"
    crater.GeomInput.zone3below4.GeomName = "zone3below4"
    crater.Geom.zone3below4.Lower.X = 0.0
    crater.Geom.zone3below4.Lower.Y = 0.0
    crater.Geom.zone3below4.Lower.Z = 0.0
    crater.Geom.zone3below4.Upper.X = 400.0
    crater.Geom.zone3below4.Upper.Y = 1.0
    crater.Geom.zone3below4.Upper.Z = 20.0

    crater.GeomInput.zone4.InputType = "Box"
    crater.GeomInput.zone4.GeomName = "zone4"
    crater.Geom.zone4.Lower.X = 0.0
    crater.Geom.zone4.Lower.Y = 0.0
    crater.Geom.zone4.Lower.Z = 100.0
    crater.Geom.zone4.Upper.X = 300.0
    crater.Geom.zone4.Upper.Y = 1.0
    crater.Geom.zone4.Upper.Z = 150.0

    crater.GeomInput.background.InputType = "Box"
    crater.GeomInput.background.GeomName = "background"
    crater.Geom.background.Lower.X = -99999999.0
    crater.Geom.background.Lower.Y = -99999999.0
    crater.Geom.background.Lower.Z = -99999999.0
    crater.Geom.background.Upper.X = 99999999.0
    crater.Geom.background.Upper.Y = 99999999.0
    crater.Geom.background.Upper.Z = 99999999.0

    crater.Geom.domain.Patches = "infiltration z_upper x_lower y_lower \
        x_upper y_upper z_lower"

    # Perm
    crater.Geom.Perm.Names = Zones
    perm_vals = {
        "zone1": 9.1496,
        "zone2": 5.4427,
        "zone3above4": 4.8033,
        "zone3left4": 4.8033,
        "zone3right4": 4.8033,
        "zone3below4": 4.8033,
        "zone4": 0.48033,
    }
    for zone in Zones.split():
        setattr(crater.Geom, zone, getattr(crater.Geom, zone))
        getattr(crater.Geom, zone).Perm.Type = "Constant"
        getattr(crater.Geom, zone).Perm.Value = perm_vals[zone]

    crater.Perm.TensorType = "TensorByGeom"
    crater.Geom.Perm.TensorByGeom.Names = "background"
    crater.Geom.background.Perm.TensorValX = 1.0
    crater.Geom.background.Perm.TensorValY = 1.0
    crater.Geom.background.Perm.TensorValZ = 1.0

    # Storage, phases, etc.
    crater.SpecificStorage.Type = "Constant"
    crater.SpecificStorage.GeomNames = "domain"
    crater.Geom.domain.SpecificStorage.Value = 1.0e-4

    crater.Phase.Names = "water"
    crater.Phase.water.Density.Type = "Constant"
    crater.Phase.water.Density.Value = 1.0
    crater.Phase.water.Viscosity.Type = "Constant"
    crater.Phase.water.Viscosity.Value = 1.0

    crater.Contaminants.Names = ""
    crater.Geom.Retardation.GeomNames = ""
    crater.Gravity = 1.0

    # Timing
    crater.TimingInfo.BaseUnit = 1.0
    crater.TimingInfo.StartCount = 0
    crater.TimingInfo.StartTime = 0.0
    crater.TimingInfo.StopTime = 20.0
    crater.TimingInfo.DumpInterval = 10.0
    crater.TimeStep.Type = "Constant"
    crater.TimeStep.Value = 10.0

    # Porosity
    crater.Geom.Porosity.GeomNames = Zones
    poro_vals = {
        "zone1": 0.3680,
        "zone2": 0.3510,
        "zone3above4": 0.3250,
        "zone3left4": 0.3250,
        "zone3right4": 0.3250,
        "zone3below4": 0.3250,
        "zone4": 0.3250,
    }
    for zone in Zones.split():
        getattr(crater.Geom, zone).Porosity.Type = "Constant"
        getattr(crater.Geom, zone).Porosity.Value = poro_vals[zone]

    crater.Domain.GeomName = "domain"

    # Ippisch
    if use_ippisch:
        crater.Phase.Saturation.VanGenuchten.AirEntryMode = "InverseAlpha"

    # RelPerm
    use_kr_table = use_table in (True, "kr_only")
    use_sat_table = use_table in (True, "sat_only")

    crater.Phase.RelPerm.Type = "VanGenuchten"
    crater.Phase.RelPerm.GeomNames = Zones
    for zone in Zones.split():
        getattr(crater.Geom, zone).RelPerm.Alpha = VG_alpha
        getattr(crater.Geom, zone).RelPerm.N = vg_n
        if use_kr_table:
            getattr(crater.Geom, zone).RelPerm.NumSamplePoints = VG_points
            getattr(crater.Geom, zone).RelPerm.MinPressureHead = -300

    # Saturation
    crater.Phase.Saturation.Type = "VanGenuchten"
    crater.Phase.Saturation.GeomNames = Zones
    for zone in Zones.split():
        getattr(crater.Geom, zone).Saturation.Alpha = VG_alpha
        getattr(crater.Geom, zone).Saturation.N = vg_n
        getattr(crater.Geom, zone).Saturation.SRes = s_res_vals[zone]
        getattr(crater.Geom, zone).Saturation.SSat = 1.0
        if use_sat_table:
            getattr(crater.Geom, zone).Saturation.NumSamplePoints = VG_points
            getattr(crater.Geom, zone).Saturation.MinPressureHead = -300

    # BCs
    crater.Wells.Names = ""
    crater.Cycle.Names = "constant onoff"
    crater.Cycle.constant.Names = "alltime"
    crater.Cycle.constant.alltime.Length = 1
    crater.Cycle.constant.Repeat = -1
    crater.Cycle.onoff.Names = "on off"
    crater.Cycle.onoff.on.Length = 10
    crater.Cycle.onoff.off.Length = 90
    crater.Cycle.onoff.Repeat = -1

    crater.BCPressure.PatchNames = crater.Geom.domain.Patches
    crater.Patch.infiltration.BCPressure.Type = "FluxConst"
    crater.Patch.infiltration.BCPressure.Cycle = "onoff"
    crater.Patch.infiltration.BCPressure.on.Value = -0.10
    crater.Patch.infiltration.BCPressure.off.Value = 0.0
    for patch in ["x_lower", "y_lower", "z_lower", "x_upper", "y_upper", "z_upper"]:
        getattr(crater.Patch, patch).BCPressure.Type = "FluxConst"
        getattr(crater.Patch, patch).BCPressure.Cycle = "constant"
        getattr(crater.Patch, patch).BCPressure.alltime.Value = 0.0

    crater.TopoSlopesX.Type = "Constant"
    crater.TopoSlopesX.GeomNames = "domain"
    crater.TopoSlopesX.Geom.domain.Value = 0.0
    crater.TopoSlopesY.Type = "Constant"
    crater.TopoSlopesY.GeomNames = "domain"
    crater.TopoSlopesY.Geom.domain.Value = 0.0

    crater.Mannings.Type = "Constant"
    crater.Mannings.GeomNames = "domain"
    crater.Mannings.Geom.domain.Value = 0.0

    crater.ICPressure.Type = "HydroStaticPatch"
    crater.ICPressure.GeomNames = "domain"
    crater.Geom.domain.ICPressure.Value = 1.0
    crater.Geom.domain.ICPressure.RefPatch = "z_lower"
    crater.Geom.domain.ICPressure.RefGeom = "domain"
    crater.Geom.infiltration.ICPressure.Value = 10.0
    crater.Geom.infiltration.ICPressure.RefPatch = "infiltration"
    crater.Geom.infiltration.ICPressure.RefGeom = "domain"

    crater.PhaseSources.water.Type = "Constant"
    crater.PhaseSources.water.GeomNames = "background"
    crater.PhaseSources.water.Geom.background.Value = 0.0

    crater.KnownSolution = "NoKnownSolution"

    # Solver
    crater.Solver = "Richards"
    crater.Solver.MaxIter = 10000
    crater.Solver.Nonlinear.MaxIter = 15
    crater.Solver.Nonlinear.ResidualTol = 1e-9
    crater.Solver.Nonlinear.StepTol = 1e-9
    crater.Solver.Nonlinear.UseJacobian = True
    crater.Solver.Nonlinear.DerivativeEpsilon = 1e-7
    crater.Solver.Linear.KrylovDimension = 25
    crater.Solver.Linear.MaxRestarts = 10
    crater.Solver.Linear.Preconditioner = "MGSemi"
    crater.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
    crater.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

    return crater


def parse_kinsol_log(log_path):
    """Parse KINSOL log for iteration counts per timestep."""
    timesteps = []
    with open(log_path) as f:
        text = f.read()

    # Split by timestep blocks
    blocks = re.split(r"KINSOL starting step for time\s+([\d.]+)", text)
    for i in range(1, len(blocks), 2):
        time = float(blocks[i])
        block = blocks[i + 1]

        nl_match = re.search(r"Nonlin\. Its\.:\s+(\d+)\s+(\d+)", block)
        lin_match = re.search(r"Lin\. Its\.:\s+(\d+)\s+(\d+)", block)
        func_match = re.search(r"Func\. Evals\.:\s+(\d+)\s+(\d+)", block)

        if nl_match:
            timesteps.append(
                {
                    "time": time,
                    "nl_its": int(nl_match.group(1)),
                    "lin_its": int(lin_match.group(1)) if lin_match else 0,
                    "func_evals": int(func_match.group(1)) if func_match else 0,
                }
            )

    return timesteps


def run_config(name, vg_n, use_table, use_ippisch):
    """Run a single configuration and return results."""
    dir_name = get_absolute_path(f"test_output/compare_{name}")
    mkdir(dir_name)

    crater = build_run(name, dir_name, vg_n, use_table, use_ippisch)
    crater.run(working_directory=dir_name)

    # Read mask and output fields
    results = {"name": name, "dir": dir_name, "press": {}, "satur": {}}
    mask_file = f"{dir_name}/{name}.out.mask.pfb"
    if os.path.exists(mask_file):
        results["mask"] = read_pfb(mask_file)
    else:
        results["mask"] = None

    for i in range(3):
        ts = str(i).rjust(5, "0")
        results["press"][i] = read_pfb(f"{dir_name}/{name}.out.press.{ts}.pfb")
        results["satur"][i] = read_pfb(f"{dir_name}/{name}.out.satur.{ts}.pfb")

    # Parse solver performance
    kinsol_log = f"{dir_name}/{name}.out.kinsol.log"
    if os.path.exists(kinsol_log):
        results["kinsol"] = parse_kinsol_log(kinsol_log)
    else:
        results["kinsol"] = []

    return results


def compare_fields(res_a, res_b, label):
    """Compare pressure and saturation fields between two runs.
    Uses domain mask to exclude inactive cells."""
    print(f"\n{'='*70}")
    print(f"  Field comparison: {res_a['name']}  vs  {res_b['name']}")
    print(f"  ({label})")
    print(f"{'='*70}")

    for i in range(3):
        ts = str(i).rjust(5, "0")
        pa = res_a["press"][i]
        pb = res_b["press"][i]
        sa = res_a["satur"][i]
        sb = res_b["satur"][i]

        # Use mask to identify active cells (mask > 0 = active)
        mask_a = res_a.get("mask")
        mask_b = res_b.get("mask")
        if mask_a is not None and mask_b is not None:
            active = (mask_a > 0) & (mask_b > 0)
        else:
            active = (np.abs(pa) < 1e30) & (np.abs(pb) < 1e30)
        n_active = np.sum(active)

        if n_active == 0:
            print(f"  Timestep {ts}: no active cells")
            continue

        p_diff = np.abs(pa[active] - pb[active])
        s_diff = np.abs(sa[active] - sb[active])

        # Find where max differences occur
        p_max_idx = np.argmax(p_diff)
        s_max_idx = np.argmax(s_diff)

        print(f"  Timestep {ts} ({n_active} active cells):")
        print(
            f"    Pressure  - max diff: {np.max(p_diff):.2e}  "
            f"mean diff: {np.mean(p_diff):.2e}  "
            f"p95 diff: {np.percentile(p_diff, 95):.2e}  "
            f"p99 diff: {np.percentile(p_diff, 99):.2e}"
        )
        print(
            f"    Saturation - max diff: {np.max(s_diff):.2e}  "
            f"mean diff: {np.mean(s_diff):.2e}  "
            f"p95 diff: {np.percentile(s_diff, 95):.2e}  "
            f"p99 diff: {np.percentile(s_diff, 99):.2e}"
        )


def print_solver_summary(results):
    """Print solver performance summary for a run."""
    kinsol = results["kinsol"]
    if not kinsol:
        print(f"  {results['name']}: No KINSOL data found")
        return

    total_nl = sum(ts["nl_its"] for ts in kinsol)
    total_lin = sum(ts["lin_its"] for ts in kinsol)
    total_fe = sum(ts["func_evals"] for ts in kinsol)

    print(
        f"  {results['name']:40s}  "
        f"NL its: {total_nl:4d}  "
        f"Lin its: {total_lin:4d}  "
        f"Func evals: {total_fe:4d}"
    )

    for ts in kinsol:
        print(
            f"    t={ts['time']:6.1f}:  "
            f"NL={ts['nl_its']:3d}  "
            f"Lin={ts['lin_its']:3d}  "
            f"FE={ts['func_evals']:3d}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 8 configs including isolated table tests",
    )
    args = parser.parse_args()

    configs = [
        ("std_direct", VG_N_standard, False, False),
        ("std_spline", VG_N_standard, True, False),
        ("ipp_n15_direct", VG_N_ippisch, False, True),
        ("ipp_n15_spline", VG_N_ippisch, True, True),
        ("ipp_n20_direct", VG_N_standard, False, True),
        ("ipp_n20_spline", VG_N_standard, True, True),
    ]

    if args.full:
        # Add configs with only RelPerm or only Saturation table
        configs.extend(
            [
                ("ipp_n15_kr_only", VG_N_ippisch, "kr_only", True),
                ("ipp_n15_sat_only", VG_N_ippisch, "sat_only", True),
            ]
        )

    results = {}
    for name, vg_n, use_table, use_ippisch in configs:
        print(f"\n{'#'*70}")
        desc = f"n={vg_n}, table={'yes' if use_table else 'no'}, ippisch={'yes' if use_ippisch else 'no'}"
        print(f"# Running: {name} ({desc})")
        print(f"{'#'*70}")
        results[name] = run_config(name, vg_n, use_table, use_ippisch)

    all_names = [c[0] for c in configs]

    # Solver performance comparison
    print(f"\n\n{'='*70}")
    print(f"  SOLVER PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    for name in all_names:
        print_solver_summary(results[name])

    # Field comparisons: table vs direct within each mode
    compare_fields(
        results["std_direct"],
        results["std_spline"],
        "Standard VG n=2.0: table interpolation error",
    )
    compare_fields(
        results["ipp_n15_direct"],
        results["ipp_n15_spline"],
        "Ippisch VG n=1.5: table interpolation error",
    )
    compare_fields(
        results["ipp_n20_direct"],
        results["ipp_n20_spline"],
        "Ippisch VG n=2.0: table interpolation error",
    )

    # Cross-mode: standard vs ippisch at same n
    compare_fields(
        results["std_spline"],
        results["ipp_n20_spline"],
        "Effect of Ippisch at n=2.0 (both using tables)",
    )
    compare_fields(
        results["std_spline"],
        results["ipp_n15_spline"],
        "Effect of Ippisch + lower n (n=2.0 std vs n=1.5 ipp, tables)",
    )

    # Ippisch n=2 vs n=1.5
    compare_fields(
        results["ipp_n20_spline"],
        results["ipp_n15_spline"],
        "Effect of n on Ippisch (n=2.0 vs n=1.5, both ipp+tables)",
    )

    if args.full:
        compare_fields(
            results["ipp_n15_direct"],
            results["ipp_n15_kr_only"],
            "Ippisch n=1.5: Kr table only (isolate Kr table error)",
        )
        compare_fields(
            results["ipp_n15_direct"],
            results["ipp_n15_sat_only"],
            "Ippisch n=1.5: Sat table only (isolate Sat table error)",
        )

    print(f"\n{'='*70}")
    print(f"  All comparisons complete.")
    print(f"{'='*70}")
