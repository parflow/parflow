# -----------------------------------------------------------------------------
# CLM restart round-trip test
#
# Guards the write/read symmetry of the CLM restart file
# (pfsimulator/clm/drv_restart.F90). Every field written by the rw=2 block
# must be consumed by the rw=1 block at the same position in the file. A
# field written mid-record without a matching read shifts every later record,
# and the restarted run dies at step 0 with "Fortran runtime error: I/O past
# end of record". This happened when snowage_vis/snowage_nir were added:
# they were written after snowage but never read back.
#
# Three runs on the standard 5x5 CLM box with snow forcing:
#   cold : hours 0-5, cold start, writes a CLM restart file every step
#   cont : hours 0-10, cold start, reference trajectory
#   warm : hours 5-10, restarted from cold's step-5 restart file
#
# Checks:
#   1. Structure: the step-5 restart file is parsed record by record and must
#      match EXPECTED_RECORDS exactly (count, order, byte size).
#   2. Round trip: the warm run must complete (the desync bug crashed here)
#      and reproduce the continuous run's pressure, saturation, and CLM
#      outputs for hours 6-10.
#
# If you add a CLM restart variable: append it at the END of the file in
# drv_restart.F90 (write block and iostat-guarded read block), then add it
# at the end of EXPECTED_RECORDS below. See "Adding new CLM variables" in
# docs/user_manual/python/keys_contribution.rst.
# -----------------------------------------------------------------------------

import struct
import sys

import numpy as np

from parflow import Run
from parflow.tools.fs import cp, get_absolute_path, mkdir, rm
from parflow.tools.compare import pf_test_file

NX = NY = 5  # grid cells; one CLM tile per cell (maxt=1 in drv_clmin.dat)
NCH = NX * NY  # number of CLM tiles
NLEVSOI = 10  # soil layers, set by Solver.CLM.RootZoneNZ below
NLEVSNO = 5  # max snow layers, parameter in pfsimulator/clm/clm_varpar.F90

FORCING_FILE = "snow_forcing_restart.1hr.txt"
NUM_FORCING_HOURS = 12  # must cover the 10-hour continuous run

# drv%rstf ("washita.rst." in drv_clmin.dat) + istep (I5.5) + "." + rank
RST_STEP5 = "washita.rst.00005.0"

# -----------------------------------------------------------------------------
# Expected restart file layout (Fortran sequential unformatted records).
# Order and sizes must match the rw=2 write block in drv_restart.F90.
# -----------------------------------------------------------------------------

R8 = 8 * NCH  # one real(r8) tile record
I4 = 4 * NCH  # one integer tile record

EXPECTED_RECORDS = (
    [("header yr/mo/da/hr/mn/ss/vclass/nc/nr/nch", 40)]
    + [("col", I4), ("row", I4), ("fgrd", R8), ("vegt", I4)]
    + [
        (name, R8)
        for name in (
            "t_grnd",
            "t_veg",
            "h2osno",
            "snowage",
            "snowdp",
            "h2ocan",
            "frac_sno",
            "elai",
            "esai",
        )
    ]
    + [("snl", I4), ("acc_errh2o", R8), ("acc_errseb", R8), ("istep", 4)]
    + [(f"dz layer {l}", R8) for l in range(-NLEVSNO + 1, NLEVSOI + 1)]
    + [(f"z layer {l}", R8) for l in range(-NLEVSNO + 1, NLEVSOI + 1)]
    + [(f"zi layer {l}", R8) for l in range(-NLEVSNO, NLEVSOI + 1)]
    + [(f"t_soisno layer {l}", R8) for l in range(-NLEVSNO + 1, NLEVSOI + 1)]
    + [(f"h2osoi_liq layer {l}", R8) for l in range(-NLEVSNO + 1, NLEVSOI + 1)]
    + [(f"h2osoi_ice layer {l}", R8) for l in range(-NLEVSNO + 1, NLEVSOI + 1)]
    # Fields appended after the original format for backward compatibility.
    # Each one is read back with an iostat guard. Add new restart variables
    # here, at the end, never mid-file.
    + [("coszen_avg", R8), ("snowage_vis", R8), ("snowage_nir", R8)]
)


def read_fortran_records(path):
    """Parse a Fortran sequential unformatted file into raw record payloads."""
    records = []
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            (nbytes,) = struct.unpack("=i", head)
            payload = f.read(nbytes)
            tail = f.read(4)
            if len(payload) != nbytes or len(tail) != 4:
                raise ValueError(f"{path}: truncated record")
            if struct.unpack("=i", tail)[0] != nbytes:
                raise ValueError(f"{path}: corrupt record markers")
            records.append(payload)
    return records


def check_restart_structure(path):
    """Verify the restart file matches the documented record layout."""
    ok = True
    records = read_fortran_records(path)

    if len(records) != len(EXPECTED_RECORDS):
        print(
            f"FAILED: {path} has {len(records)} records, "
            f"expected {len(EXPECTED_RECORDS)}."
        )
        print(
            "A CLM restart variable was probably added or removed. Update "
            "the write AND read blocks in drv_restart.F90 (append at end of "
            "file, iostat-guarded read) and EXPECTED_RECORDS in this test. "
            "See 'Adding new CLM variables' in the ParFlow manual."
        )
        return False

    for i, ((name, size), rec) in enumerate(zip(EXPECTED_RECORDS, records)):
        if len(rec) != size:
            print(
                f"FAILED: restart record {i} ({name}) is {len(rec)} bytes, "
                f"expected {size}"
            )
            ok = False

    if not ok:
        return False

    # Header sanity: grid and tile counts
    header = struct.unpack("=10i", records[0])
    yr, mo, da, hr, mn, ss, vclass, nc, nr, nch = header
    if (nc, nr, nch) != (NX, NY, NCH):
        print(f"FAILED: restart header grid/tile mismatch: {header}")
        ok = False

    # The three appended fields must hold finite, non-negative values
    for name, rec in zip(("coszen_avg", "snowage_vis", "snowage_nir"), records[-3:]):
        values = np.frombuffer(rec, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            print(f"FAILED: restart field {name} has bad values: {values}")
            ok = False

    if ok:
        print(f"Restart file structure OK ({len(records)} records): {path}")
    return ok


# -----------------------------------------------------------------------------
# Run setup helpers
# -----------------------------------------------------------------------------


def stage_inputs(dir_name):
    mkdir(dir_name)
    cp("$PF_SRC/test/tcl/clm/drv_clmin.dat", dir_name)
    cp("$PF_SRC/test/tcl/clm/drv_vegm.dat", dir_name)
    cp("$PF_SRC/test/tcl/clm/drv_vegp.dat", dir_name)

    # Tile the 6-hour snow forcing to cover the 10-hour continuous run.
    # The 1D met reader sizes its arrays from the line count of this file.
    src = get_absolute_path("$PF_SRC/test/tcl/clm/snow_forcing.1hr.txt")
    with open(src) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    with open(dir_name + "/" + FORCING_FILE, "w") as f:
        for i in range(NUM_FORCING_HOURS):
            f.write(lines[i % len(lines)] + "\n")


def set_restart_startcode(dir_name):
    """Flip drv_clmin.dat to restart mode (startcode=1, clm_ic=1)."""
    path = dir_name + "/drv_clmin.dat"
    with open(path) as f:
        text = f.read()
    text = text.replace("startcode      2", "startcode      1")
    text = text.replace("clm_ic         2", "clm_ic         1")
    assert "startcode      1" in text and "clm_ic         1" in text
    with open(path, "w") as f:
        f.write(text)


def build_run(run_name):
    run = Run(run_name, __file__)

    run.FileVersion = 4

    run.Process.Topology.P = 1
    run.Process.Topology.Q = 1
    run.Process.Topology.R = 1

    run.ComputationalGrid.Lower.X = 0.0
    run.ComputationalGrid.Lower.Y = 0.0
    run.ComputationalGrid.Lower.Z = 0.0

    run.ComputationalGrid.DX = 1000.0
    run.ComputationalGrid.DY = 1000.0
    run.ComputationalGrid.DZ = 0.5

    run.ComputationalGrid.NX = NX
    run.ComputationalGrid.NY = NY
    run.ComputationalGrid.NZ = 10

    run.GeomInput.Names = "domain_input"
    run.GeomInput.domain_input.InputType = "Box"
    run.GeomInput.domain_input.GeomName = "domain"

    run.Geom.domain.Lower.X = 0.0
    run.Geom.domain.Lower.Y = 0.0
    run.Geom.domain.Lower.Z = 0.0
    run.Geom.domain.Upper.X = 5000.0
    run.Geom.domain.Upper.Y = 5000.0
    run.Geom.domain.Upper.Z = 5.0
    run.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    run.Geom.Perm.Names = "domain"
    run.Geom.domain.Perm.Type = "Constant"
    run.Geom.domain.Perm.Value = 0.2

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
    run.Gravity = 1.0

    run.TimingInfo.BaseUnit = 1.0
    run.TimingInfo.StartCount = 0
    run.TimingInfo.StartTime = 0.0
    run.TimingInfo.StopTime = 5
    run.TimingInfo.DumpInterval = -1
    run.TimeStep.Type = "Constant"
    run.TimeStep.Value = 1.0

    run.Geom.Porosity.GeomNames = "domain"
    run.Geom.domain.Porosity.Type = "Constant"
    run.Geom.domain.Porosity.Value = 0.390

    run.Domain.GeomName = "domain"

    run.Phase.water.Mobility.Type = "Constant"
    run.Phase.water.Mobility.Value = 1.0

    run.Phase.RelPerm.Type = "VanGenuchten"
    run.Phase.RelPerm.GeomNames = "domain"
    run.Geom.domain.RelPerm.Alpha = 3.5
    run.Geom.domain.RelPerm.N = 2.0

    run.Phase.Saturation.Type = "VanGenuchten"
    run.Phase.Saturation.GeomNames = "domain"
    run.Geom.domain.Saturation.Alpha = 3.5
    run.Geom.domain.Saturation.N = 2.0
    run.Geom.domain.Saturation.SRes = 0.01
    run.Geom.domain.Saturation.SSat = 1.0

    run.Wells.Names = ""

    run.Cycle.Names = "constant"
    run.Cycle.constant.Names = "alltime"
    run.Cycle.constant.alltime.Length = 1
    run.Cycle.constant.Repeat = -1

    run.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"

    run.Patch.x_lower.BCPressure.Type = "FluxConst"
    run.Patch.x_lower.BCPressure.Cycle = "constant"
    run.Patch.x_lower.BCPressure.alltime.Value = 0.0

    run.Patch.y_lower.BCPressure.Type = "FluxConst"
    run.Patch.y_lower.BCPressure.Cycle = "constant"
    run.Patch.y_lower.BCPressure.alltime.Value = 0.0

    run.Patch.z_lower.BCPressure.Type = "FluxConst"
    run.Patch.z_lower.BCPressure.Cycle = "constant"
    run.Patch.z_lower.BCPressure.alltime.Value = 0.0

    run.Patch.x_upper.BCPressure.Type = "FluxConst"
    run.Patch.x_upper.BCPressure.Cycle = "constant"
    run.Patch.x_upper.BCPressure.alltime.Value = 0.0

    run.Patch.y_upper.BCPressure.Type = "FluxConst"
    run.Patch.y_upper.BCPressure.Cycle = "constant"
    run.Patch.y_upper.BCPressure.alltime.Value = 0.0

    run.Patch.z_upper.BCPressure.Type = "OverlandFlow"
    run.Patch.z_upper.BCPressure.Cycle = "constant"
    run.Patch.z_upper.BCPressure.alltime.Value = 0.0

    run.TopoSlopesX.Type = "Constant"
    run.TopoSlopesX.GeomNames = "domain"
    run.TopoSlopesX.Geom.domain.Value = -0.001

    run.TopoSlopesY.Type = "Constant"
    run.TopoSlopesY.GeomNames = "domain"
    run.TopoSlopesY.Geom.domain.Value = 0.001

    run.Mannings.Type = "Constant"
    run.Mannings.GeomNames = "domain"
    run.Mannings.Geom.domain.Value = 5.52e-6

    run.PhaseSources.water.Type = "Constant"
    run.PhaseSources.water.GeomNames = "domain"
    run.PhaseSources.water.Geom.domain.Value = 0.0

    run.KnownSolution = "NoKnownSolution"

    run.Solver = "Richards"
    run.Solver.MaxIter = 500

    run.Solver.Nonlinear.MaxIter = 15
    run.Solver.Nonlinear.ResidualTol = 1e-9
    run.Solver.Nonlinear.EtaChoice = "EtaConstant"
    run.Solver.Nonlinear.EtaValue = 0.01
    run.Solver.Nonlinear.UseJacobian = True
    run.Solver.Nonlinear.StepTol = 1e-20
    run.Solver.Nonlinear.Globalization = "LineSearch"
    run.Solver.Linear.KrylovDimension = 15
    run.Solver.Linear.MaxRestart = 2

    run.Solver.Linear.Preconditioner = "PFMG"
    run.Solver.PrintSubsurf = False
    run.Solver.Drop = 1e-20
    run.Solver.AbsTol = 1e-9

    run.Solver.LSM = "CLM"
    run.Solver.CLM.MetForcing = "1D"
    run.Solver.CLM.MetFileName = FORCING_FILE
    run.Solver.CLM.MetFilePath = "."
    run.Solver.CLM.RootZoneNZ = NLEVSOI

    run.Solver.WriteSiloCLM = False
    run.Solver.WriteSiloEvapTrans = False
    run.Solver.WriteSiloOverlandBCFlux = False
    run.Solver.PrintCLM = True

    run.Solver.CLM.Print1dOut = False
    run.Solver.BinaryOutDir = False
    run.Solver.WriteCLMBinary = False
    run.Solver.CLM.CLMDumpInterval = 1
    run.Solver.CLM.WriteLogs = False
    run.Solver.CLM.SingleFile = True

    # Write a numbered CLM restart file at every step
    run.Solver.CLM.WriteLastRST = False
    run.Solver.CLM.DailyRST = False
    run.Solver.CLM.IstepStart = 1

    run.ICPressure.Type = "HydroStaticPatch"
    run.ICPressure.GeomNames = "domain"
    run.Geom.domain.ICPressure.Value = -2.0
    run.Geom.domain.ICPressure.RefGeom = "domain"
    run.Geom.domain.ICPressure.RefPatch = "z_upper"

    return run


# -----------------------------------------------------------------------------
# Run 1 "cold": hours 0-5, cold start, restart file written every step
# -----------------------------------------------------------------------------

cold_dir = get_absolute_path("test_output/clm_restart_cold")
stage_inputs(cold_dir)

cold = build_run("clm_restart_cold")
cold.run(working_directory=cold_dir)

passed = True

# -----------------------------------------------------------------------------
# Check 1: restart file structure ("write an rst file with all fields...")
# -----------------------------------------------------------------------------

if not check_restart_structure(cold_dir + "/" + RST_STEP5):
    passed = False

# -----------------------------------------------------------------------------
# Run 2 "cont": hours 0-10, cold start, reference trajectory
# -----------------------------------------------------------------------------

cont_dir = get_absolute_path("test_output/clm_restart_cont")
stage_inputs(cont_dir)

cont = build_run("clm_restart_cont")
cont.TimingInfo.StopTime = 10
cont.run(working_directory=cont_dir)

# The cold and continuous runs must share a trajectory for hours 0-5,
# otherwise the warm comparison below is meaningless.
for i in range(6):
    timestep = str(i).rjust(5, "0")
    if not pf_test_file(
        f"{cold_dir}/clm_restart_cold.out.press.{timestep}.pfb",
        f"{cont_dir}/clm_restart_cont.out.press.{timestep}.pfb",
        f"Max difference in cold vs continuous pressure, timestep {timestep}",
    ):
        passed = False

# -----------------------------------------------------------------------------
# Run 3 "warm": hours 5-10, restarted from cold's step-5 state
# ("...and read it back")
# -----------------------------------------------------------------------------

warm_dir = get_absolute_path("test_output/clm_restart_warm")
stage_inputs(warm_dir)
set_restart_startcode(warm_dir)

# CLM state comes from the restart file, ParFlow pressure from the step-5 dump
cp(cold_dir + "/" + RST_STEP5, warm_dir)
press_ic = "clm_restart_cold.out.press.00005.pfb"
cp(cold_dir + "/" + press_ic, warm_dir)

warm = build_run("clm_restart_warm")
warm.TimingInfo.StartCount = 5
warm.TimingInfo.StartTime = 5.0
warm.TimingInfo.StopTime = 10
# First step is istep 6; drv_restart reads washita.rst.<IstepStart-1>.<rank>
warm.Solver.CLM.IstepStart = 6
warm.ICPressure.Type = "PFBFile"
warm.ICPressure.GeomNames = "domain"
warm.Geom.domain.ICPressure.FileName = press_ic
warm.dist(warm_dir + "/" + press_ic)

# With the write/read desync bug, this run aborts at its first step with
# "Fortran runtime error: I/O past end of record".
warm.run(working_directory=warm_dir)

# -----------------------------------------------------------------------------
# Check 2: the restarted run reproduces the continuous trajectory
# -----------------------------------------------------------------------------

for i in range(6, 11):
    timestep = str(i).rjust(5, "0")
    for field, label in (("press", "Pressure"), ("satur", "Saturation")):
        if not pf_test_file(
            f"{warm_dir}/clm_restart_warm.out.{field}.{timestep}.pfb",
            f"{cont_dir}/clm_restart_cont.out.{field}.{timestep}.pfb",
            f"Max difference in warm vs continuous {label}, timestep {timestep}",
        ):
            passed = False
    if not pf_test_file(
        f"{warm_dir}/clm_restart_warm.out.clm_output.{timestep}.C.pfb",
        f"{cont_dir}/clm_restart_cont.out.clm_output.{timestep}.C.pfb",
        f"Max difference in warm vs continuous CLM output, timestep {timestep}",
    ):
        passed = False

# -----------------------------------------------------------------------------

if passed:
    rm(cold_dir)
    rm(cont_dir)
    rm(warm_dir)
    print("clm_restart : PASSED")
else:
    print("clm_restart : FAILED")
    sys.exit(1)
