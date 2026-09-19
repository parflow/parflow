import argparse
import os
import sys

import numpy as np

from parflow import Run
from parflow.tools.compare import pf_test_file
from parflow.tools.fs import cp, get_absolute_path, mkdir, rm
from parflow.tools.io import read_pfb

NX = 41
NY = 41
NZ = 50
DX = 1000.0
DY = 1000.0
DZ = 2.0
STOP_TIME = 12.0
FINAL_DUMP = 1
RAINFALL_RATE = 1.0e-5


def parse_topology_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--p", default=1, type=int)
    parser.add_argument("-q", "--q", default=1, type=int)
    parser.add_argument("-r", "--r", default=1, type=int)
    return parser.parse_args()


def configure_lw_overland_only(run_name, bc_type, args):
    run = Run(run_name, __file__)
    run.FileVersion = 4
    run.Process.Topology.P = args.p
    run.Process.Topology.Q = args.q
    run.Process.Topology.R = args.r

    run.ComputationalGrid.Lower.X = 0.0
    run.ComputationalGrid.Lower.Y = 0.0
    run.ComputationalGrid.Lower.Z = 0.0
    run.ComputationalGrid.DX = DX
    run.ComputationalGrid.DY = DY
    run.ComputationalGrid.DZ = DZ
    run.ComputationalGrid.NX = NX
    run.ComputationalGrid.NY = NY
    run.ComputationalGrid.NZ = NZ

    run.GeomInput.Names = "box_input indi_input"
    run.GeomInput.box_input.InputType = "Box"
    run.GeomInput.box_input.GeomName = "domain"
    run.GeomInput.indi_input.InputType = "IndicatorField"
    run.GeomInput.indi_input.GeomNames = (
        "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
    )
    run.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"
    for index in range(1, 14):
        setattr(getattr(run.GeomInput, f"s{index}"), "Value", index)
    for index in range(1, 9):
        setattr(getattr(run.GeomInput, f"g{index}"), "Value", index + 20)

    run.Geom.domain.Lower.X = 0.0
    run.Geom.domain.Lower.Y = 0.0
    run.Geom.domain.Lower.Z = 0.0
    run.Geom.domain.Upper.X = 41000.0
    run.Geom.domain.Upper.Y = 41000.0
    run.Geom.domain.Upper.Z = 100.0
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
    run.Geom.domain.SpecificStorage.Value = 1.0e-5

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
    run.TimingInfo.StopTime = STOP_TIME
    run.TimingInfo.DumpInterval = STOP_TIME
    run.TimeStep.Type = "Constant"
    run.TimeStep.Value = 1.0

    run.Geom.Porosity.GeomNames = "domain"
    run.Geom.domain.Porosity.Type = "Constant"
    run.Geom.domain.Porosity.Value = 0.4
    run.Domain.GeomName = "domain"
    run.Phase.water.Mobility.Type = "Constant"
    run.Phase.water.Mobility.Value = 1.0
    run.Wells.Names = ""
    run.Reservoirs.Names = ""

    run.Cycle.Names = "constant"
    run.Cycle.constant.Names = "alltime"
    run.Cycle.constant.alltime.Length = 1
    run.Cycle.constant.Repeat = -1

    run.BCPressure.PatchNames = run.Geom.domain.Patches
    for patch in ["x_lower", "x_upper", "y_lower", "y_upper", "z_lower"]:
        getattr(run.Patch, patch).BCPressure.Type = "FluxConst"
        getattr(run.Patch, patch).BCPressure.Cycle = "constant"
        getattr(run.Patch, patch).BCPressure.alltime.Value = 0.0
    run.Patch.z_upper.BCPressure.Type = bc_type
    run.Patch.z_upper.BCPressure.Cycle = "constant"
    run.Patch.z_upper.BCPressure.alltime.Value = -RAINFALL_RATE

    run.TopoSlopesX.Type = "PFBFile"
    run.TopoSlopesX.GeomNames = "domain"
    run.TopoSlopesX.FileName = "LW.slopex.pfb"
    run.TopoSlopesY.Type = "PFBFile"
    run.TopoSlopesY.GeomNames = "domain"
    run.TopoSlopesY.FileName = "LW.slopey.pfb"
    run.Mannings.Type = "Constant"
    run.Mannings.GeomNames = "domain"
    run.Mannings.Geom.domain.Value = 5.52e-6

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
    run.PhaseSources.water.Type = "Constant"
    run.PhaseSources.water.GeomNames = "domain"
    run.PhaseSources.water.Geom.domain.Value = 0.0

    run.ICPressure.Type = "PFBFile"
    run.ICPressure.GeomNames = "domain"
    run.Geom.domain.ICPressure.RefPatch = "z_upper"
    run.Geom.domain.ICPressure.FileName = "press.init.pfb"

    run.Solver = "Richards"
    run.Solver.TerrainFollowingGrid = True
    run.Solver.Nonlinear.VariableDz = False
    run.Solver.MaxIter = 25000
    run.Solver.Drop = 1.0e-20
    run.Solver.AbsTol = 1.0e-8
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
    run.Solver.Linear.MaxRestarts = 2
    run.Solver.Linear.Preconditioner = "PFMGOctree"
    run.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"
    run.Solver.PrintSubsurfData = False
    run.Solver.PrintPressure = True
    run.Solver.PrintSaturation = False
    run.Solver.PrintMask = True
    run.Solver.PrintTop = True
    run.Solver.PrintQxOverland = True
    run.Solver.PrintQyOverland = True
    run.Solver.PrintOverlandSum = True
    run.Solver.OverlandOnly = True
    run.KnownSolution = "NoKnownSolution"
    return run


def prepare_inputs(output_dir, run):
    rm(output_dir)
    mkdir(output_dir)
    for filename in [
        "LW.slopex.pfb",
        "LW.slopey.pfb",
        "IndicatorFile_Gleeson.50z.pfb",
        "press.init.pfb",
    ]:
        cp(f"$PF_SRC/test/tcl/washita/parflow_input/{filename}", output_dir)
        run.dist(os.path.join(output_dir, filename))


def surface_pressure(pressure, top):
    jj, ii = np.indices((NY, NX))
    return pressure[top, jj, ii]


def validate_outputs(run_name, output_dir, mode_name):
    passed = True
    final_step = FINAL_DUMP
    top = read_pfb(os.path.join(output_dir, f"{run_name}.out.top_zindex.pfb"))[
        0, :, :
    ].astype(int)
    initial_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.00000.pfb")
    )
    final_pressure = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.press.{final_step:05d}.pfb")
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

    surface = np.maximum(surface_pressure(final_pressure, top), 0.0)
    if not np.all(np.isfinite(surface)) or np.any(surface < 0.0):
        print(f"{run_name}: final surface storage is not finite and nonnegative")
        passed = False

    storage = np.sum(surface) * DX * DY
    qx = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.qx_overland.{final_step:05d}.pfb")
    )[0, :, :]
    qy = read_pfb(
        os.path.join(output_dir, f"{run_name}.out.qy_overland.{final_step:05d}.pfb")
    )[0, :, :]
    routing_signal = np.sum(np.abs(qx)) * DY + np.sum(np.abs(qy)) * DX
    if storage <= 0.0 and routing_signal <= 0.0:
        print(f"{run_name}: no finite overland storage or routing flux was produced")
        passed = False

    correct_file = get_absolute_path(
        f"$PF_SRC/test/correct_output/{run_name}.out.press.{final_step:05d}.pfb"
    )
    if os.path.exists(correct_file):
        output_file = os.path.join(
            output_dir, f"{run_name}.out.press.{final_step:05d}.pfb"
        )
        if not pf_test_file(
            output_file, correct_file, f"{run_name}: final pressure", sig_digits=6
        ):
            passed = False
    else:
        print(f"{run_name}: reference final pressure PFB is not checked in yet")

    if passed and not os.environ.get("PARFLOW_KEEP_TEST_OUTPUTS"):
        rm(output_dir)
    return passed


def run_lw_overland_only(mode_name, bc_type):
    run_name = f"LW_overland_only_{mode_name}"
    args = parse_topology_args()
    output_dir = get_absolute_path(f"test_output/{run_name}")
    run = configure_lw_overland_only(run_name, bc_type, args)
    prepare_inputs(output_dir, run)
    run.run(working_directory=output_dir)

    if validate_outputs(run_name, output_dir, mode_name):
        print(f"{run_name} : PASSED")
    else:
        print(f"{run_name} : FAILED")
        sys.exit(1)
