import os
import subprocess
import sys

import numpy as np

from parflow import Run
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
from parflow.tools.fs import cp, get_absolute_path, mkdir, rm
from parflow.tools.io import read_pfb, write_pfb

RUN_NAME = "FJC_overland_only_mannings"
NX = 125
NY = 133
NZ = 2
DX = 50.0
DY = 50.0
DZ = 0.1
RAIN_RATE = 4.2447e-5
RAIN_DURATION = 2
STOP_TIME = 12
FINAL_STEP = 12
CHECK_STEPS = (2, 12)
MIN_NETWORK_DEPTH = 1.0e-3
MIN_DEPTH_TOP5_SHARE = 0.50
MIN_QMAG_TOP5_SHARE = 0.75
MIN_NONZERO_Q_CELLS = 1000

INPUT_FILES = {
    "slope_x": "FJC_SlopeX.pfb",
    "slope_y": "FJC_SlopeY.pfb",
    "mask": "FJC_Mask.pfb",
    "dem": "FJC_DEM.pfb",
    "mannings": "FJC_Spatial_Mannings.pfb",
}


def input_path(filename):
    return get_absolute_path(f"$PF_SRC/test/input/{filename}")


def validate_input_files():
    expected_shape = (1, NY, NX)
    arrays = {}
    passed = True

    for key, filename in INPUT_FILES.items():
        values = read_pfb(input_path(filename))
        arrays[key] = values
        if values.shape != expected_shape:
            print(f"{filename}: expected shape {expected_shape}, got {values.shape}")
            passed = False
        if not np.all(np.isfinite(values)):
            print(f"{filename}: contains non-finite values")
            passed = False

    mask_values = np.unique(arrays["mask"])
    if not np.array_equal(mask_values, np.array([0.0, 1.0])):
        print(f"{INPUT_FILES['mask']}: expected mask values [0, 1], got {mask_values}")
        passed = False

    if np.any(arrays["mannings"] <= 0.0):
        print(f"{INPUT_FILES['mannings']}: Manning's values must be positive")
        passed = False

    if not passed:
        sys.exit(1)

    return arrays


def configure_run(output_dir):
    run = Run(RUN_NAME, __file__)
    run.FileVersion = 4
    run.Process.Topology.P = 1
    run.Process.Topology.Q = 1
    run.Process.Topology.R = 1

    run.ComputationalGrid.Lower.X = 0.0
    run.ComputationalGrid.Lower.Y = 0.0
    run.ComputationalGrid.Lower.Z = -DZ
    run.ComputationalGrid.DX = DX
    run.ComputationalGrid.DY = DY
    run.ComputationalGrid.DZ = DZ
    run.ComputationalGrid.NX = NX
    run.ComputationalGrid.NY = NY
    run.ComputationalGrid.NZ = NZ

    run.GeomInput.Names = "domain_input"
    run.GeomInput.domain_input.InputType = "SolidFile"
    run.GeomInput.domain_input.GeomNames = "domain"
    run.GeomInput.domain_input.FileName = "FJC_domain.pfsol"

    mask_matrix = read_pfb(input_path(INPUT_FILES["mask"]))[0, :, :]
    mask_path = os.path.join(output_dir, "FJC_mask_for_pfsol.pfb")
    write_pfb(
        mask_path,
        mask_matrix[np.newaxis, :, :],
        x=0.0,
        y=0.0,
        z=0.0,
        dx=DX,
        dy=DY,
        dz=DZ,
    )
    subprocess.run(
        [
            get_absolute_path("$PARFLOW_DIR/bin/pfmask-to-pfsol"),
            "--mask",
            mask_path,
            "--side-patch-label",
            "3",
            "--bottom-patch-label",
            "2",
            "--pfsol",
            os.path.join(output_dir, "FJC_domain.pfsol"),
        ],
        check=True,
    )

    run.Geom.domain.Patches = "z_upper z_lower side"
    run.Geom.domain.Lower.X = 0.0
    run.Geom.domain.Lower.Y = 0.0
    run.Geom.domain.Lower.Z = -DZ
    run.Geom.domain.Upper.X = NX * DX
    run.Geom.domain.Upper.Y = NY * DY
    run.Geom.domain.Upper.Z = DZ

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
    run.Geom.domain.SpecificStorage.Value = 1.0e-5

    run.Phase.Names = "water"
    run.Phase.water.Density.Type = "Constant"
    run.Phase.water.Density.Value = 1.0
    run.Phase.water.Viscosity.Type = "Constant"
    run.Phase.water.Viscosity.Value = 1.0
    run.Phase.water.Mobility.Type = "Constant"
    run.Phase.water.Mobility.Value = 1.0
    run.Contaminants.Names = ""
    run.Geom.Retardation.GeomNames = ""
    run.Gravity = 1.0

    run.TimingInfo.BaseUnit = 1.0
    run.TimingInfo.StartCount = 0
    run.TimingInfo.StartTime = 0.0
    run.TimingInfo.StopTime = STOP_TIME
    run.TimingInfo.DumpInterval = 1.0
    run.TimeStep.Type = "Constant"
    run.TimeStep.Value = 1.0

    run.Geom.Porosity.GeomNames = "domain"
    run.Geom.domain.Porosity.Type = "Constant"
    run.Geom.domain.Porosity.Value = 0.25
    run.Domain.GeomName = "domain"

    run.Phase.RelPerm.Type = "VanGenuchten"
    run.Phase.RelPerm.GeomNames = "domain"
    run.Geom.domain.RelPerm.Alpha = 3.5
    run.Geom.domain.RelPerm.N = 2.0
    run.Phase.Saturation.Type = "VanGenuchten"
    run.Phase.Saturation.GeomNames = "domain"
    run.Geom.domain.Saturation.Alpha = 3.5
    run.Geom.domain.Saturation.N = 2.0
    run.Geom.domain.Saturation.SRes = 0.2
    run.Geom.domain.Saturation.SSat = 1.0

    run.Wells.Names = ""
    run.Reservoirs.Names = ""
    run.PhaseSources.water.Type = "Constant"
    run.PhaseSources.water.GeomNames = "domain"
    run.PhaseSources.water.Geom.domain.Value = 0.0

    run.Cycle.Names = "pulse"
    run.Cycle.pulse.Names = "rain recession"
    run.Cycle.pulse.rain.Length = RAIN_DURATION
    run.Cycle.pulse.recession.Length = STOP_TIME - RAIN_DURATION
    run.Cycle.pulse.Repeat = 1

    run.BCPressure.PatchNames = run.Geom.domain.Patches
    for patch in ["z_lower", "side"]:
        run.Patch[patch].BCPressure.Type = "FluxConst"
        run.Patch[patch].BCPressure.Cycle = "pulse"
        run.Patch[patch].BCPressure.rain.Value = 0.0
        run.Patch[patch].BCPressure.recession.Value = 0.0

    run.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    run.Patch.z_upper.BCPressure.Cycle = "pulse"
    run.Patch.z_upper.BCPressure.rain.Value = -RAIN_RATE
    run.Patch.z_upper.BCPressure.recession.Value = 0.0

    run.TopoSlopesX.Type = "PFBFile"
    run.TopoSlopesX.GeomNames = "domain"
    run.TopoSlopesX.FileName = INPUT_FILES["slope_x"]
    run.TopoSlopesY.Type = "PFBFile"
    run.TopoSlopesY.GeomNames = "domain"
    run.TopoSlopesY.FileName = INPUT_FILES["slope_y"]
    run.pfset(key="TopoSlopes.Elevation.FileName", value=INPUT_FILES["dem"])

    run.Mannings.Type = "PFBFile"
    run.Mannings.FileName = INPUT_FILES["mannings"]

    run.ICPressure.Type = "Constant"
    run.ICPressure.GeomNames = "domain"
    run.Geom.domain.ICPressure.Value = 0.0

    run.Solver = "Richards"
    run.Solver.TerrainFollowingGrid = True
    run.Solver.OverlandOnly = True
    run.Solver.MaxIter = 25000
    run.Solver.MaxConvergenceFailures = 8
    run.Solver.Nonlinear.MaxIter = 80
    run.Solver.Nonlinear.ResidualTol = 1.0e-6
    run.Solver.Nonlinear.EtaChoice = "EtaConstant"
    run.Solver.Nonlinear.EtaValue = 0.001
    run.Solver.Nonlinear.UseJacobian = True
    run.Solver.Nonlinear.DerivativeEpsilon = 1.0e-16
    run.Solver.Nonlinear.StepTol = 1.0e-30
    run.Solver.Nonlinear.Globalization = "LineSearch"
    run.Solver.Linear.KrylovDimension = 70
    run.Solver.Linear.MaxRestart = 2
    run.Solver.Linear.Preconditioner = "PFMG"
    run.Solver.Drop = 1.0e-20
    run.Solver.AbsTol = 1.0e-8
    run.Solver.PrintSubsurfData = False
    run.Solver.PrintPressure = True
    run.Solver.PrintSaturation = False
    run.Solver.PrintMask = True
    run.Solver.PrintMannings = True
    run.Solver.PrintTop = True
    run.Solver.PrintQxOverland = True
    run.Solver.PrintQyOverland = True
    run.Solver.PrintOverlandSum = False
    run.KnownSolution = "NoKnownSolution"
    return run


def prepare_run(output_dir):
    rm(output_dir)
    mkdir(output_dir)
    arrays = validate_input_files()
    for filename in INPUT_FILES.values():
        cp(input_path(filename), output_dir)
    return arrays


def surface_from_pressure(pressure, top):
    return np.take_along_axis(pressure, top[np.newaxis, :, :], axis=0)[0, :, :]


def top_fraction_share(values, fraction):
    total = np.sum(values)
    if total <= 0.0:
        return 0.0
    count = max(1, int(np.ceil(values.size * fraction)))
    return np.sum(np.sort(values)[-count:]) / total


def compare_known_output(output_dir, file_type, step, abs_tolerance=None, sig_digits=6):
    correct_file = get_absolute_path(
        f"$PF_SRC/test/correct_output/{RUN_NAME}.out.{file_type}.{step:05d}.pfb"
    )
    output_file = os.path.join(output_dir, f"{RUN_NAME}.out.{file_type}.{step:05d}.pfb")
    if not os.path.exists(correct_file):
        print(
            f"{RUN_NAME}: reference {file_type} PFB for step {step:05d} is not checked in yet"
        )
        return False
    if abs_tolerance is None:
        return pf_test_file(
            output_file,
            correct_file,
            f"{RUN_NAME}: {file_type} at step {step:05d}",
            sig_digits=sig_digits,
        )
    return pf_test_file_with_abs(
        output_file,
        correct_file,
        f"{RUN_NAME}: {file_type} at step {step:05d}",
        abs_tolerance,
        sig_digits,
    )


def validate_network_signature(surface, qx, qy, mask):
    passed = True
    active_depth = np.maximum(surface[mask], 0.0)
    if np.sum(active_depth) <= 0.0:
        print(f"{RUN_NAME}: no positive top-layer storage for stream signature check")
        passed = False
    elif np.max(active_depth) <= MIN_NETWORK_DEPTH:
        print(
            f"{RUN_NAME}: maximum top-layer depth {np.max(active_depth)} "
            f"does not exceed {MIN_NETWORK_DEPTH}"
        )
        passed = False
    elif top_fraction_share(active_depth, 0.05) < MIN_DEPTH_TOP5_SHARE:
        print(f"{RUN_NAME}: top-layer pressure is not concentrated enough")
        passed = False

    active_qx = qx[0, :, :][mask]
    active_qy = qy[0, :, :][mask]
    qmag = np.sqrt(active_qx * active_qx + active_qy * active_qy)
    if np.sum(qmag) <= 0.0:
        print(
            f"{RUN_NAME}: no positive overland flux magnitude for stream signature check"
        )
        passed = False
    elif top_fraction_share(qmag, 0.05) < MIN_QMAG_TOP5_SHARE:
        print(f"{RUN_NAME}: overland flux magnitude is not concentrated enough")
        passed = False

    if np.count_nonzero(np.abs(active_qx) > 1.0e-8) < MIN_NONZERO_Q_CELLS:
        print(f"{RUN_NAME}: qx_overland has too few nonzero active cells")
        passed = False
    if np.count_nonzero(np.abs(active_qy) > 1.0e-8) < MIN_NONZERO_Q_CELLS:
        print(f"{RUN_NAME}: qy_overland has too few nonzero active cells")
        passed = False

    return passed


def validate_outputs(output_dir, arrays):
    passed = True
    mask = arrays["mask"][0, :, :] == 1.0
    active_area = np.count_nonzero(mask) * DX * DY
    total_input = RAIN_RATE * active_area * RAIN_DURATION

    mannings_out = read_pfb(os.path.join(output_dir, f"{RUN_NAME}.out.mannings.pfb"))
    active_mask = arrays["mask"] == 1.0
    if not np.allclose(
        mannings_out[active_mask],
        arrays["mannings"][active_mask],
        rtol=0.0,
        atol=1.0e-15,
    ):
        print(f"{RUN_NAME}: printed Manning's field does not match input PFB")
        passed = False

    initial_pressure = read_pfb(
        os.path.join(output_dir, f"{RUN_NAME}.out.press.00000.pfb")
    )
    top = read_pfb(os.path.join(output_dir, f"{RUN_NAME}.out.top_zindex.pfb"))[
        0, :, :
    ].astype(int)
    if not np.all(np.isfinite(top)):
        print(f"{RUN_NAME}: top output contains non-finite values")
        passed = False

    checked = {}
    for step in CHECK_STEPS:
        pressure = read_pfb(
            os.path.join(output_dir, f"{RUN_NAME}.out.press.{step:05d}.pfb")
        )
        qx = read_pfb(
            os.path.join(output_dir, f"{RUN_NAME}.out.qx_overland.{step:05d}.pfb")
        )
        qy = read_pfb(
            os.path.join(output_dir, f"{RUN_NAME}.out.qy_overland.{step:05d}.pfb")
        )
        checked[step] = (pressure, qx, qy)

        for label, values in [("pressure", pressure), ("qx", qx), ("qy", qy)]:
            if not np.all(np.isfinite(values)):
                print(
                    f"{RUN_NAME}: {label} output at step {step:05d} contains non-finite values"
                )
                passed = False

        active_qx = qx[0, :, :][mask]
        active_qy = qy[0, :, :][mask]
        qmag = np.sqrt(active_qx * active_qx + active_qy * active_qy)
        if np.sum(np.abs(active_qx)) <= 0.0:
            print(
                f"{RUN_NAME}: qx_overland has no active-domain signal at step {step:05d}"
            )
            passed = False
        if np.sum(np.abs(active_qy)) <= 0.0:
            print(
                f"{RUN_NAME}: qy_overland has no active-domain signal at step {step:05d}"
            )
            passed = False
        if np.sum(qmag) <= 0.0:
            print(
                f"{RUN_NAME}: combined overland flux magnitude is zero at step {step:05d}"
            )
            passed = False

        if not compare_known_output(output_dir, "press", step):
            passed = False
        if not compare_known_output(
            output_dir, "qx_overland", step, abs_tolerance=1.0e-12, sig_digits=8
        ):
            passed = False
        if not compare_known_output(
            output_dir, "qy_overland", step, abs_tolerance=1.0e-12, sig_digits=8
        ):
            passed = False

    final_pressure, final_qx, final_qy = checked[FINAL_STEP]

    z_indices = np.arange(NZ)[:, np.newaxis, np.newaxis]
    below_surface = (top[np.newaxis, :, :] >= 0) & (z_indices < top[np.newaxis, :, :])
    if np.count_nonzero(below_surface) == 0:
        print(f"{RUN_NAME}: no below-surface cells were available for pinning check")
        passed = False
    elif not np.allclose(
        final_pressure[below_surface],
        initial_pressure[below_surface],
        rtol=0.0,
        atol=1.0e-12,
    ):
        print(f"{RUN_NAME}: subsurface pressure changed below the overland surface")
        passed = False

    surface = surface_from_pressure(final_pressure, top)
    storage = np.sum(np.maximum(surface[mask], 0.0)) * DX * DY
    if storage < -1.0e-12 or storage > total_input + 1.0e-8:
        print(
            f"{RUN_NAME}: final surface storage {storage} is outside "
            f"[0, total pulse input {total_input}]"
        )
        passed = False

    if not validate_network_signature(surface, final_qx, final_qy, mask):
        passed = False

    return passed


output_dir = get_absolute_path(f"test_output/{RUN_NAME}")
arrays = prepare_run(output_dir)
run = configure_run(output_dir)
for filename in INPUT_FILES.values():
    run.dist(os.path.join(output_dir, filename))
run.run(working_directory=output_dir)

if validate_outputs(output_dir, arrays):
    print(f"{RUN_NAME} : PASSED")
    if not os.environ.get("PARFLOW_KEEP_TEST_OUTPUTS"):
        rm(output_dir)
else:
    print(f"{RUN_NAME} : FAILED")
    sys.exit(1)
