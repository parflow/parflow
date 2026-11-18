import sys

from parflow import Run


def base_run(name):
    run = Run(name, __file__)
    run.Solver = "Richards"
    run.Solver.OverlandOnly = True
    run.Wells.Names = ""
    run.Reservoirs.Names = ""
    run.Cycle.Names = "constant"
    run.Cycle.constant.Names = "alltime"
    run.Cycle.constant.alltime.Length = 1
    run.Cycle.constant.Repeat = -1
    run.BCPressure.PatchNames = "z_upper"
    run.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    run.Patch.z_upper.BCPressure.Cycle = "constant"
    run.Patch.z_upper.BCPressure.alltime.Value = 0.0
    return run


def expect_error(name, mutate, expected_text):
    run = base_run(name)
    mutate(run)
    errors = run._validate_overland_only()
    if not any(expected_text in error for error in errors):
        print(
            f"{name}: expected OverlandOnly validation error containing {expected_text!r}"
        )
        print("Observed errors:")
        for error in errors:
            print(f"  {error}")
        return False
    if run.validate(enable_print=False) == 0:
        print(f"{name}: run.validate() unexpectedly passed")
        return False
    return True


cases = [
    (
        "overland_only_requires_richards",
        lambda run: setattr(run, "Solver", "Impes"),
        "Solver to be Richards",
    ),
    (
        "overland_only_rejects_clm",
        lambda run: setattr(run.Solver, "LSM", "CLM"),
        "Solver.LSM",
    ),
    (
        "overland_only_rejects_wells",
        lambda run: setattr(run.Wells, "Names", "well1"),
        "Wells.Names",
    ),
    (
        "overland_only_rejects_reservoirs",
        lambda run: setattr(run.Reservoirs, "Names", "res1"),
        "Reservoirs.Names",
    ),
    (
        "overland_only_rejects_surface_predictor",
        lambda run: setattr(run.Solver, "SurfacePredictor", True),
        "Solver.SurfacePredictor",
    ),
    (
        "overland_only_rejects_spinup",
        lambda run: setattr(run.Solver, "Spinup", True),
        "Solver.Spinup",
    ),
    (
        "overland_only_rejects_reset_surface_pressure",
        lambda run: setattr(run.Solver, "ResetSurfacePressure", True),
        "Solver.ResetSurfacePressure",
    ),
    (
        "overland_only_requires_overland_bc",
        lambda run: setattr(run.Patch.z_upper.BCPressure, "Type", "FluxConst"),
        "requires at least one BCPressure patch",
    ),
    (
        "overland_only_rejects_flux_file",
        lambda run: setattr(run.Patch.z_upper.BCPressure, "Type", "FluxFile"),
        "FluxFile",
    ),
    (
        "overland_only_rejects_kinematic_file_forcing",
        lambda run: (
            setattr(run.Patch.z_upper.BCPressure, "Type", "OverlandKinematic"),
            setattr(run.Patch.z_upper.BCPressure.alltime, "FileName", "rain.pfb"),
        ),
        "file-backed forcing for OverlandKinematic",
    ),
    (
        "overland_only_rejects_diffusive_file_forcing",
        lambda run: (
            setattr(run.Patch.z_upper.BCPressure, "Type", "OverlandDiffusive"),
            setattr(run.Patch.z_upper.BCPressure.alltime, "FileName", "rain.pfb"),
        ),
        "file-backed forcing for OverlandDiffusive",
    ),
]


passed = True
for name, mutate, expected_text in cases:
    if not expect_error(name, mutate, expected_text):
        passed = False

if passed:
    print("overland_only_validation : PASSED")
else:
    print("overland_only_validation : FAILED")
    sys.exit(1)
