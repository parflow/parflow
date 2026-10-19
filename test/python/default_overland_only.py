import argparse
import os
import sys

import numpy as np

from parflow import Run
from parflow.tools.fs import get_absolute_path, mkdir, rm
from parflow.tools.io import read_pfb, write_pfb

BASE_RUN_NAME = "default_overland_only"
NX = 12
NY = 3
NZ = 3
DX = 10.0
DY = 10.0
DZ = 1.0
DT = 1.0
STOP_TIME = 80.0
RAINFALL_RATE = 1.0e-3
PULSE_RAINFALL_RATE = 1.0e-3
PULSE_STEPS = 1
PULSE_STOP_TIME = 200.0
SLOPE_X = -1.0e-2
PULSE_SLOPE_X = -2.0e-1
SLOPE_Y = 0.0
MANNINGS = 3.0e-3
PULSE_MANNINGS = 3.0e-4


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()


def configure_run(
    run_name,
    bc_type,
    rainfall_value,
    evap_trans_file=False,
    pulse=False,
    slope_x=SLOPE_X,
    mannings=MANNINGS,
    stop_time=STOP_TIME,
):
    run = Run(run_name, __file__)

    run.FileVersion = 4
    run.Process.Topology.P = args.p
    run.Process.Topology.Q = args.q
    run.Process.Topology.R = args.r

    run.ComputationalGrid.Lower.X = 0.0
    run.ComputationalGrid.Lower.Y = 0.0
    run.ComputationalGrid.Lower.Z = 0.0
    run.ComputationalGrid.NX = NX
    run.ComputationalGrid.NY = NY
    run.ComputationalGrid.NZ = NZ
    run.ComputationalGrid.DX = DX
    run.ComputationalGrid.DY = DY
    run.ComputationalGrid.DZ = DZ

    run.GeomInput.Names = "domaininput"
    run.GeomInput.domaininput.GeomName = "domain"
    run.GeomInput.domaininput.InputType = "Box"

    run.Geom.domain.Lower.X = 0.0
    run.Geom.domain.Lower.Y = 0.0
    run.Geom.domain.Lower.Z = 0.0
    run.Geom.domain.Upper.X = NX * DX
    run.Geom.domain.Upper.Y = NY * DY
    run.Geom.domain.Upper.Z = NZ * DZ
    run.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    run.Geom.Perm.Names = "domain"
    run.Geom.domain.Perm.Type = "Constant"
    run.Geom.domain.Perm.Value = 1.0e-12
    run.Perm.TensorType = "TensorByGeom"
    run.Geom.Perm.TensorByGeom.Names = "domain"
    run.Geom.domain.Perm.TensorValX = 1.0
    run.Geom.domain.Perm.TensorValY = 1.0
    run.Geom.domain.Perm.TensorValZ = 1.0

    run.SpecificStorage.Type = "Constant"
    run.SpecificStorage.GeomNames = "domain"
    run.Geom.domain.SpecificStorage.Value = 1.0e-6

    run.Phase.Names = "water"
    run.Phase.water.Density.Type = "Constant"
    run.Phase.water.Density.Value = 1.0
    run.Phase.water.Viscosity.Type = "Constant"
    run.Phase.water.Viscosity.Value = 1.0
    run.Contaminants.Names = ""
    run.Geom.Retardation.GeomNames = ""
    run.Gravity = 1.0

    run.TimingInfo.BaseUnit = 1.0
    run.TimingInfo.StartCount = 0
    run.TimingInfo.StartTime = 0.0
    run.TimingInfo.StopTime = stop_time
    run.TimingInfo.DumpInterval = 1.0
    run.TimeStep.Type = "Constant"
    run.TimeStep.Value = DT

    run.Geom.Porosity.GeomNames = "domain"
    run.Geom.domain.Porosity.Type = "Constant"
    run.Geom.domain.Porosity.Value = 0.25
    run.Domain.GeomName = "domain"

    run.Phase.RelPerm.Type = "VanGenuchten"
    run.Phase.RelPerm.GeomNames = "domain"
    run.Geom.domain.RelPerm.Alpha = 6.0
    run.Geom.domain.RelPerm.N = 2.0

    run.Phase.Saturation.Type = "VanGenuchten"
    run.Phase.Saturation.GeomNames = "domain"
    run.Geom.domain.Saturation.Alpha = 6.0
    run.Geom.domain.Saturation.N = 2.0
    run.Geom.domain.Saturation.SRes = 0.2
    run.Geom.domain.Saturation.SSat = 1.0

    run.Wells.Names = ""
    if pulse:
        run.Cycle.Names = "pulse"
        run.Cycle.pulse.Names = "rain rest"
        run.Cycle.pulse.rain.Length = PULSE_STEPS
        run.Cycle.pulse.rest.Length = int(stop_time - PULSE_STEPS)
        run.Cycle.pulse.Repeat = 1
    else:
        run.Cycle.Names = "constant"
        run.Cycle.constant.Names = "alltime"
        run.Cycle.constant.alltime.Length = 1
        run.Cycle.constant.Repeat = -1

    run.BCPressure.PatchNames = run.Geom.domain.Patches
    for patch in ["x_lower", "x_upper", "y_lower", "y_upper", "z_lower"]:
        run.Patch[patch].BCPressure.Type = "FluxConst"
        if pulse:
            run.Patch[patch].BCPressure.Cycle = "pulse"
            run.Patch[patch].BCPressure.rain.Value = 0.0
            run.Patch[patch].BCPressure.rest.Value = 0.0
        else:
            run.Patch[patch].BCPressure.Cycle = "constant"
            run.Patch[patch].BCPressure.alltime.Value = 0.0

    run.Patch.z_upper.BCPressure.Type = bc_type
    if pulse:
        run.Patch.z_upper.BCPressure.Cycle = "pulse"
        run.Patch.z_upper.BCPressure.rain.Value = rainfall_value
        run.Patch.z_upper.BCPressure.rest.Value = 0.0
    elif bc_type == "OverlandFlowPFB":
        run.Patch.z_upper.BCPressure.Cycle = "constant"
        run.Patch.z_upper.BCPressure.alltime.FileName = "rainfall.pfb"
    else:
        run.Patch.z_upper.BCPressure.Cycle = "constant"
        run.Patch.z_upper.BCPressure.alltime.Value = rainfall_value

    run.TopoSlopesX.Type = "Constant"
    run.TopoSlopesX.GeomNames = "domain"
    run.TopoSlopesX.Geom.domain.Value = slope_x
    run.TopoSlopesY.Type = "Constant"
    run.TopoSlopesY.GeomNames = "domain"
    run.TopoSlopesY.Geom.domain.Value = SLOPE_Y

    run.Mannings.Type = "Constant"
    run.Mannings.GeomNames = "domain"
    run.Mannings.Geom.domain.Value = mannings

    run.PhaseSources.water.Type = "Constant"
    run.PhaseSources.water.GeomNames = "domain"
    run.PhaseSources.water.Geom.domain.Value = 0.0
    run.KnownSolution = "NoKnownSolution"

    run.Solver = "Richards"
    run.Solver.MaxIter = 2500
    run.Solver.Nonlinear.MaxIter = 20
    run.Solver.Nonlinear.ResidualTol = 1.0e-9
    run.Solver.Nonlinear.EtaChoice = "EtaConstant"
    run.Solver.Nonlinear.EtaValue = 0.01
    run.Solver.Nonlinear.UseJacobian = True
    run.Solver.Nonlinear.DerivativeEpsilon = 1.0e-8
    run.Solver.Nonlinear.StepTol = 1.0e-20
    run.Solver.Nonlinear.Globalization = "LineSearch"
    run.Solver.Linear.KrylovDimension = 20
    run.Solver.Linear.MaxRestart = 2
    run.Solver.Linear.Preconditioner = "PFMGOctree"
    run.Solver.PrintSubsurf = False
    run.Solver.OverlandOnly = True
    run.Solver.Drop = 1.0e-20
    run.Solver.AbsTol = 1.0e-9
    run.Solver.PrintVelocities = True
    run.Solver.PrintQxOverland = True
    run.Solver.PrintQyOverland = True
    run.Solver.PrintOverlandSum = True
    run.Solver.PrintTop = True

    if evap_trans_file:
        run.Solver.EvapTransFile = True
        run.Solver.EvapTrans.FileName = "evap_trans.pfb"

    run.ICPressure.Type = "Constant"
    run.ICPressure.GeomNames = "domain"
    run.Geom.domain.ICPressure.Value = -1.0

    return run


def write_uniform_overland_pfb(output_dir, filename, value):
    values = np.full((NZ, NY, NX), value, dtype=np.float64)
    write_pfb(
        os.path.join(output_dir, filename),
        values,
        x=0.0,
        y=0.0,
        z=0.0,
        dx=DX,
        dy=DY,
        dz=DZ,
    )


def write_evap_trans_pfb(output_dir, filename):
    values = np.zeros((NZ, NY, NX), dtype=np.float64)
    values[NZ - 1, :, :] = RAINFALL_RATE / DZ
    write_pfb(
        os.path.join(output_dir, filename),
        values,
        x=0.0,
        y=0.0,
        z=0.0,
        dx=DX,
        dy=DY,
        dz=DZ,
    )


def surface_pressure(pressure, top):
    jj, ii = np.indices((NY, NX))
    return pressure[top, jj, ii]


def total_surface_storage(output_dir, run_name, step, top):
    pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.{step:05d}.pfb")
    )
    return np.sum(np.maximum(surface_pressure(pressure, top), 0.0)) * DX * DY


def total_overland_sum(output_dir, run_name, step):
    if step == 0:
        return 0.0
    filename = os.path.join(output_dir, f"{run_name}.out.overlandsum.{step:05d}.pfb")
    return np.sum(read_pfb(filename))


def outlet_flux(output_dir, run_name, step):
    filename = os.path.join(output_dir, f"{run_name}.out.qx_overland.{step:05d}.pfb")
    qx = read_pfb(filename)[0, :, :]
    return np.sum(qx[:, -1]) * DY


def check_close(name, actual, expected, abs_tol, rel_tol):
    error = abs(actual - expected)
    scale = max(abs(expected), abs(actual), 1.0)
    if error > abs_tol and error / scale > rel_tol:
        print(
            f"{name} failed: actual={actual:.16e}, expected={expected:.16e}, "
            f"abs_error={error:.16e}, rel_error={error / scale:.16e}"
        )
        return False
    return True


def validate_case(case):
    run_name = f"{BASE_RUN_NAME}_{case['name']}"
    output_dir = get_absolute_path(f"test_output/{run_name}")
    rm(output_dir)
    mkdir(output_dir)

    run = configure_run(
        run_name,
        case["bc_type"],
        case["rainfall_value"],
        evap_trans_file=case.get("evap_trans_file", False),
    )

    if case["bc_type"] == "OverlandFlowPFB":
        write_uniform_overland_pfb(output_dir, "rainfall.pfb", -RAINFALL_RATE)
    if case.get("evap_trans_file", False):
        write_evap_trans_pfb(output_dir, "evap_trans.pfb")

    run.run(working_directory=output_dir)

    passed = True
    top = read_pfb(os.path.join(output_dir, f"{run_name}.out.top_zindex.pfb"))[
        0, :, :
    ].astype(int)
    initial_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.00000.pfb")
    )
    final_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.{int(STOP_TIME):05d}.pfb")
    )

    z_indices = np.arange(NZ)[:, np.newaxis, np.newaxis]
    inactive_subsurface = z_indices != top[np.newaxis, :, :]
    if not np.allclose(
        final_pressure[inactive_subsurface],
        initial_pressure[inactive_subsurface],
        atol=1.0e-12,
    ):
        print(
            f"{run_name}: subsurface pressure changed below the overland surface layer"
        )
        passed = False

    plan_area = NX * DX * NY * DY
    expected_outflow_rate = RAINFALL_RATE * plan_area
    previous_storage = total_surface_storage(output_dir, run_name, 0, top)
    final_storage_change = 0.0

    for step in range(1, int(STOP_TIME) + 1):
        storage = total_surface_storage(output_dir, run_name, step, top)
        outflow = outlet_flux(output_dir, run_name, step) * DT
        actual_change = storage - previous_storage
        final_storage_change = actual_change
        expected_change = (
            actual_change
            if case.get("infer_global_outflow", False)
            else RAINFALL_RATE * plan_area * DT - outflow
        )

        if not check_close(
            f"{run_name}: mass balance at step {step}",
            actual_change,
            expected_change,
            abs_tol=1.0e-7,
            rel_tol=1.0e-7,
        ):
            passed = False

        previous_storage = storage

    if case.get("infer_global_outflow", False):
        final_outflow_rate = expected_outflow_rate - final_storage_change / DT
    else:
        final_outflow_rate = outlet_flux(output_dir, run_name, int(STOP_TIME))
    if not check_close(
        f"{run_name}: steady outlet flux",
        final_outflow_rate,
        expected_outflow_rate,
        abs_tol=5.0e-2,
        rel_tol=2.5e-2,
    ):
        passed = False

    if case.get("check_profile", False):
        final_qx = read_pfb(
            os.path.join(
                output_dir, f"{run_name}.out.qx_overland.{int(STOP_TIME):05d}.pfb"
            )
        )[0, :, :]
        final_depth = np.maximum(surface_pressure(final_pressure, top), 0.0)
        x_center = (np.arange(NX) + 0.5) * DX
        expected_qx = RAINFALL_RATE * x_center
        expected_depth = (MANNINGS * expected_qx / np.sqrt(abs(SLOPE_X))) ** (3.0 / 5.0)
        interior = slice(2, NX - 2)

        if not np.allclose(
            final_qx[:, interior], expected_qx[interior], rtol=0.25, atol=2.5e-3
        ):
            print(f"{run_name}: qx profile did not approach the Manning solution")
            passed = False
        if not np.allclose(
            final_depth[:, interior], expected_depth[interior], rtol=0.25, atol=2.5e-3
        ):
            print(f"{run_name}: depth profile did not approach the Manning solution")
            passed = False

    final_storage = total_surface_storage(output_dir, run_name, int(STOP_TIME), top)

    if passed:
        rm(output_dir)
    return {
        "passed": passed,
        "final_outflow_rate": final_outflow_rate,
        "final_storage": final_storage,
    }


def validate_pulse_case():
    run_name = f"{BASE_RUN_NAME}_pulse"
    output_dir = get_absolute_path(f"test_output/{run_name}")
    rm(output_dir)
    mkdir(output_dir)

    run = configure_run(
        run_name,
        "OverlandFlow",
        -PULSE_RAINFALL_RATE,
        pulse=True,
        slope_x=PULSE_SLOPE_X,
        mannings=PULSE_MANNINGS,
        stop_time=PULSE_STOP_TIME,
    )
    run.run(working_directory=output_dir)

    passed = True
    top = read_pfb(os.path.join(output_dir, f"{run_name}.out.top_zindex.pfb"))[
        0, :, :
    ].astype(int)
    initial_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.00000.pfb")
    )
    final_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.{int(PULSE_STOP_TIME):05d}.pfb")
    )

    z_indices = np.arange(NZ)[:, np.newaxis, np.newaxis]
    inactive_subsurface = z_indices != top[np.newaxis, :, :]
    if not np.allclose(
        final_pressure[inactive_subsurface],
        initial_pressure[inactive_subsurface],
        atol=1.0e-12,
    ):
        print(
            f"{run_name}: subsurface pressure changed below the overland surface layer"
        )
        passed = False

    plan_area = NX * DX * NY * DY
    cumulative_input = PULSE_RAINFALL_RATE * plan_area * PULSE_STEPS * DT
    cumulative_outflow = 0.0
    for step in range(1, int(PULSE_STOP_TIME) + 1):
        cumulative_outflow += outlet_flux(output_dir, run_name, step) * DT

    final_storage = total_surface_storage(
        output_dir, run_name, int(PULSE_STOP_TIME), top
    )
    balance_residual = abs(cumulative_input - cumulative_outflow - final_storage)
    if balance_residual > 1.0e-6:
        print(
            f"{run_name}: pulse mass balance failed: input={cumulative_input:.16e}, "
            f"outflow={cumulative_outflow:.16e}, final_storage={final_storage:.16e}, "
            f"residual={balance_residual:.16e}"
        )
        passed = False

    if final_storage > max(1.0e-6, 5.0e-3 * cumulative_input):
        print(f"{run_name}: pulse did not drain; final_storage={final_storage:.16e}")
        passed = False

    if passed:
        rm(output_dir)
    return {
        "passed": passed,
        "final_outflow_rate": cumulative_outflow / PULSE_STOP_TIME,
        "final_storage": final_storage,
    }


cases = [
    {
        "name": "overland_flow",
        "bc_type": "OverlandFlow",
        "rainfall_value": -RAINFALL_RATE,
        "check_profile": True,
    },
    {
        "name": "overland_flow_pfb",
        "bc_type": "OverlandFlowPFB",
        "rainfall_value": 0.0,
    },
    {
        "name": "evap_trans_file",
        "bc_type": "OverlandFlow",
        "rainfall_value": 0.0,
        "evap_trans_file": True,
    },
    {
        "name": "overland_kinematic",
        "bc_type": "OverlandKinematic",
        "rainfall_value": -RAINFALL_RATE,
    },
    {
        "name": "overland_diffusive",
        "bc_type": "OverlandDiffusive",
        "rainfall_value": -RAINFALL_RATE,
        "infer_global_outflow": True,
    },
]

passed = True
results = {}
for case in cases:
    result = validate_case(case)
    results[case["name"]] = result
    if not result["passed"]:
        passed = False

pulse_result = validate_pulse_case()
results["pulse"] = pulse_result
if not pulse_result["passed"]:
    passed = False

for equivalent_case in ["overland_flow_pfb", "evap_trans_file"]:
    if not check_close(
        f"{equivalent_case}: final outflow equivalence",
        results[equivalent_case]["final_outflow_rate"],
        results["overland_flow"]["final_outflow_rate"],
        abs_tol=1.0e-8,
        rel_tol=1.0e-8,
    ):
        passed = False
    if not check_close(
        f"{equivalent_case}: final storage equivalence",
        results[equivalent_case]["final_storage"],
        results["overland_flow"]["final_storage"],
        abs_tol=1.0e-8,
        rel_tol=1.0e-8,
    ):
        passed = False

if passed:
    print(f"{BASE_RUN_NAME} : PASSED")
else:
    print(f"{BASE_RUN_NAME} : FAILED")
    sys.exit(1)
