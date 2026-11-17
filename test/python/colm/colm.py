import shutil
import sys
from pathlib import Path

from parflow import Run
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
from parflow.tools.fs import mkdir, rm
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.settings import set_working_directory
from parflow.tools.top import compute_top, extract_top

# -----------------------------------------------------------------------------
# SET RUN NAME
# -----------------------------------------------------------------------------

runname = "PF_CoLM_CI"

# -----------------------------------------------------------------------------
# SET WORK DIRECTORY
# -----------------------------------------------------------------------------

example_dir = Path(__file__).resolve().parent
run_path = example_dir / "outputs" / runname
run_dir = str(run_path)
if run_path.exists():
    rm(run_dir)
mkdir(run_dir)
model = Run(runname, __file__)
model.FileVersion = 4

# -----------------------------------------------------------------------------
# GET MODEL INPUTS
# -----------------------------------------------------------------------------

input_dir = example_dir / "inputs"
for source in input_dir.iterdir():
    if source.is_file():
        shutil.copy(source, run_dir)

set_working_directory(run_dir)

# -----------------------------------------------------------------------------
# SET PROCESSOR TOPOLOGY
# -----------------------------------------------------------------------------

model.Process.Topology.P = 1
model.Process.Topology.Q = 1
model.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

model.ComputationalGrid.Lower.X = 0.0
model.ComputationalGrid.Lower.Y = 0.0
model.ComputationalGrid.Lower.Z = 0.0

model.ComputationalGrid.DX = 961.72
model.ComputationalGrid.DY = 961.72
model.ComputationalGrid.DZ = 200.0

model.ComputationalGrid.NX = 5
model.ComputationalGrid.NY = 5
model.ComputationalGrid.NZ = 11

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

model.GeomInput.Names = "domaininput indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

model.GeomInput.domaininput.InputType = "Box"
model.GeomInput.domaininput.GeomName = "domain"
model.Geom.domain.Lower.X = 0.0
model.Geom.domain.Lower.Y = 0.0
model.Geom.domain.Lower.Z = 0.0

model.Geom.domain.Upper.X = 5 * 961.72
model.Geom.domain.Upper.Y = 5 * 961.72
model.Geom.domain.Upper.Z = 2200.0
model.Domain.GeomName = "domain"
model.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

model.GeomInput.indi_input.InputType = "IndicatorField"
model.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10"
)
model.Geom.indi_input.FileName = "subsurface_11layer.pfb"

model.GeomInput.s1.Value = 1
model.GeomInput.s2.Value = 2
model.GeomInput.s3.Value = 3
model.GeomInput.s4.Value = 4
model.GeomInput.s5.Value = 5
model.GeomInput.s6.Value = 6
model.GeomInput.s7.Value = 7
model.GeomInput.s8.Value = 8
model.GeomInput.s9.Value = 9
model.GeomInput.s10.Value = 10
model.GeomInput.s11.Value = 11
model.GeomInput.s12.Value = 12
model.GeomInput.s13.Value = 13

model.GeomInput.g1.Value = 19
model.GeomInput.g2.Value = 20
model.GeomInput.g3.Value = 21
model.GeomInput.g4.Value = 22
model.GeomInput.g5.Value = 23
model.GeomInput.g6.Value = 24
model.GeomInput.g7.Value = 25
model.GeomInput.g8.Value = 26
model.GeomInput.g9.Value = 27
model.GeomInput.g10.Value = 28


# -----------------------------------------------------------------------------
# SET VARIABLE DZ
# -----------------------------------------------------------------------------

model.Solver.Nonlinear.VariableDz = True
model.dzScale.GeomNames = "domain"
model.dzScale.Type = "nzList"
model.dzScale.nzListNumber = 11

model.Cell._0.dzScale.Value = 100 / 200
model.Cell._1.dzScale.Value = (3.4331 - 2.2961) / 200
model.Cell._2.dzScale.Value = (2.2961 - 1.3828) / 200
model.Cell._3.dzScale.Value = (1.3828 - 0.8289) / 200
model.Cell._4.dzScale.Value = (0.8289 - 0.4929) / 200
model.Cell._5.dzScale.Value = (0.4929 - 0.2891) / 200
model.Cell._6.dzScale.Value = (0.2891 - 0.1655) / 200
model.Cell._7.dzScale.Value = (0.1655 - 0.0906) / 200
model.Cell._8.dzScale.Value = (0.0906 - 0.0451) / 200
model.Cell._9.dzScale.Value = (0.0451 - 0.0175) / 200
model.Cell._10.dzScale.Value = 0.0175 / 200

# -----------------------------------------------------------------------------
# SET SLOPEX AND SLOPEY
# -----------------------------------------------------------------------------
model.TopoSlopesX.Type = "PFBFile"
model.TopoSlopesX.GeomNames = "domain"
model.TopoSlopesX.FileName = "slope_x.pfb"

model.TopoSlopesY.Type = "PFBFile"
model.TopoSlopesY.GeomNames = "domain"
model.TopoSlopesY.FileName = "slope_y.pfb"

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

model.Geom.Perm.Names = (
    "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10"
)

model.Geom.domain.Perm.Type = "Constant"
model.Geom.domain.Perm.Value = 0.02

model.Geom.s1.Perm.Type = "Constant"
model.Geom.s1.Perm.Value = 0.269022595

model.Geom.s2.Perm.Type = "Constant"
model.Geom.s2.Perm.Value = 0.043630356

model.Geom.s3.Perm.Type = "Constant"
model.Geom.s3.Perm.Value = 0.015841225

model.Geom.s4.Perm.Type = "Constant"
model.Geom.s4.Perm.Value = 0.007582087

model.Geom.s5.Perm.Type = "Constant"
model.Geom.s5.Perm.Value = 0.01818816

model.Geom.s6.Perm.Type = "Constant"
model.Geom.s6.Perm.Value = 0.005009435

model.Geom.s7.Perm.Type = "Constant"
model.Geom.s7.Perm.Value = 0.005492736

model.Geom.s8.Perm.Type = "Constant"
model.Geom.s8.Perm.Value = 0.004675077

model.Geom.s9.Perm.Type = "Constant"
model.Geom.s9.Perm.Value = 0.003386794

model.Geom.s10.Perm.Type = "Constant"
model.Geom.s10.Perm.Value = 0.004783973

model.Geom.s11.Perm.Type = "Constant"
model.Geom.s11.Perm.Value = 0.003979136

model.Geom.s12.Perm.Type = "Constant"
model.Geom.s12.Perm.Value = 0.006162952

model.Geom.s13.Perm.Type = "Constant"
model.Geom.s13.Perm.Value = 0.005009435

model.Geom.g1.Perm.Type = "Constant"
model.Geom.g1.Perm.Value = 5e-3

model.Geom.g2.Perm.Type = "Constant"
model.Geom.g2.Perm.Value = 1e-2

model.Geom.g3.Perm.Type = "Constant"
model.Geom.g3.Perm.Value = 2e-2

model.Geom.g4.Perm.Type = "Constant"
model.Geom.g4.Perm.Value = 3e-2

model.Geom.g5.Perm.Type = "Constant"
model.Geom.g5.Perm.Value = 4e-2

model.Geom.g6.Perm.Type = "Constant"
model.Geom.g6.Perm.Value = 5e-2

model.Geom.g7.Perm.Type = "Constant"
model.Geom.g7.Perm.Value = 6e-2

model.Geom.g8.Perm.Type = "Constant"
model.Geom.g8.Perm.Value = 8e-2

model.Geom.g9.Perm.Type = "Constant"
model.Geom.g9.Perm.Value = 0.1

model.Geom.g10.Perm.Type = "Constant"
model.Geom.g10.Perm.Value = 0.2

model.Perm.TensorType = "TensorByGeom"
model.Geom.Perm.TensorByGeom.Names = "domain"
model.Geom.domain.Perm.TensorValX = 1.0
model.Geom.domain.Perm.TensorValY = 1.0
model.Geom.domain.Perm.TensorValZ = 1.0


# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

model.SpecificStorage.Type = "Constant"
model.SpecificStorage.GeomNames = "domain"
model.Geom.domain.SpecificStorage.Value = 0.0001

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

model.Geom.Porosity.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13"

model.Geom.domain.Porosity.Type = "Constant"
model.Geom.domain.Porosity.Value = 0.33

model.Geom.s1.Porosity.Type = "Constant"
model.Geom.s1.Porosity.Value = 0.375

model.Geom.s2.Porosity.Type = "Constant"
model.Geom.s2.Porosity.Value = 0.39

model.Geom.s3.Porosity.Type = "Constant"
model.Geom.s3.Porosity.Value = 0.387

model.Geom.s4.Porosity.Type = "Constant"
model.Geom.s4.Porosity.Value = 0.439

model.Geom.s5.Porosity.Type = "Constant"
model.Geom.s5.Porosity.Value = 0.489

model.Geom.s6.Porosity.Type = "Constant"
model.Geom.s6.Porosity.Value = 0.399

model.Geom.s7.Porosity.Type = "Constant"
model.Geom.s7.Porosity.Value = 0.384

model.Geom.s8.Porosity.Type = "Constant"
model.Geom.s8.Porosity.Value = 0.482

model.Geom.s9.Porosity.Type = "Constant"
model.Geom.s9.Porosity.Value = 0.442

model.Geom.s10.Porosity.Type = "Constant"
model.Geom.s10.Porosity.Value = 0.385

model.Geom.s11.Porosity.Type = "Constant"
model.Geom.s11.Porosity.Value = 0.481

model.Geom.s12.Porosity.Type = "Constant"
model.Geom.s12.Porosity.Value = 0.459

model.Geom.s13.Porosity.Type = "Constant"
model.Geom.s13.Porosity.Value = 0.399

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

model.Phase.RelPerm.Type = "VanGenuchten"
model.Phase.RelPerm.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13"

model.Geom.domain.RelPerm.Alpha = 1.0
model.Geom.domain.RelPerm.N = 3.0

model.Geom.s1.RelPerm.Alpha = 3.548
model.Geom.s1.RelPerm.N = 4.162

model.Geom.s2.RelPerm.Alpha = 3.467
model.Geom.s2.RelPerm.N = 2.738

model.Geom.s3.RelPerm.Alpha = 2.692
model.Geom.s3.RelPerm.N = 2.445

model.Geom.s4.RelPerm.Alpha = 0.501
model.Geom.s4.RelPerm.N = 2.659

model.Geom.s5.RelPerm.Alpha = 0.661
model.Geom.s5.RelPerm.N = 2.659

model.Geom.s6.RelPerm.Alpha = 1.122
model.Geom.s6.RelPerm.N = 2.479

model.Geom.s7.RelPerm.Alpha = 2.089
model.Geom.s7.RelPerm.N = 2.318

model.Geom.s8.RelPerm.Alpha = 0.832
model.Geom.s8.RelPerm.N = 2.514

model.Geom.s9.RelPerm.Alpha = 1.585
model.Geom.s9.RelPerm.N = 2.413

model.Geom.s10.RelPerm.Alpha = 3.311
model.Geom.s10.RelPerm.N = 2.202

model.Geom.s11.RelPerm.Alpha = 1.622
model.Geom.s11.RelPerm.N = 2.318

model.Geom.s12.RelPerm.Alpha = 1.514
model.Geom.s12.RelPerm.N = 2.259

model.Geom.s13.RelPerm.Alpha = 1.122
model.Geom.s13.RelPerm.N = 2.479


# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

model.Phase.Saturation.Type = "VanGenuchten"
model.Phase.Saturation.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13"

model.Geom.domain.Saturation.Alpha = 1.0
model.Geom.domain.Saturation.N = 3.0
model.Geom.domain.Saturation.SRes = 0.001
model.Geom.domain.Saturation.SSat = 1.0

model.Geom.s1.Saturation.Alpha = 3.548
model.Geom.s1.Saturation.N = 4.162
model.Geom.s1.Saturation.SRes = 0.0001
model.Geom.s1.Saturation.SSat = 1.0

model.Geom.s2.Saturation.Alpha = 3.467
model.Geom.s2.Saturation.N = 2.738
model.Geom.s2.Saturation.SRes = 0.0001
model.Geom.s2.Saturation.SSat = 1.0

model.Geom.s3.Saturation.Alpha = 2.692
model.Geom.s3.Saturation.N = 2.445
model.Geom.s3.Saturation.SRes = 0.0001
model.Geom.s3.Saturation.SSat = 1.0

model.Geom.s4.Saturation.Alpha = 0.501
model.Geom.s4.Saturation.N = 2.659
model.Geom.s4.Saturation.SRes = 0.1
model.Geom.s4.Saturation.SSat = 1.0

model.Geom.s5.Saturation.Alpha = 0.661
model.Geom.s5.Saturation.N = 2.659
model.Geom.s5.Saturation.SRes = 0.0001
model.Geom.s5.Saturation.SSat = 1.0

model.Geom.s6.Saturation.Alpha = 1.122
model.Geom.s6.Saturation.N = 2.479
model.Geom.s6.Saturation.SRes = 0.0001
model.Geom.s6.Saturation.SSat = 1.0

model.Geom.s7.Saturation.Alpha = 2.089
model.Geom.s7.Saturation.N = 2.318
model.Geom.s7.Saturation.SRes = 0.0001
model.Geom.s7.Saturation.SSat = 1.0

model.Geom.s8.Saturation.Alpha = 0.832
model.Geom.s8.Saturation.N = 2.514
model.Geom.s8.Saturation.SRes = 0.0001
model.Geom.s8.Saturation.SSat = 1.0

model.Geom.s9.Saturation.Alpha = 1.585
model.Geom.s9.Saturation.N = 2.413
model.Geom.s9.Saturation.SRes = 0.0001
model.Geom.s9.Saturation.SSat = 1.0

model.Geom.s10.Saturation.Alpha = 3.311
model.Geom.s10.Saturation.N = 2.202
model.Geom.s10.Saturation.SRes = 0.0001
model.Geom.s10.Saturation.SSat = 1.0

model.Geom.s11.Saturation.Alpha = 1.622
model.Geom.s11.Saturation.N = 2.318
model.Geom.s11.Saturation.SRes = 0.0001
model.Geom.s11.Saturation.SSat = 1.0

model.Geom.s12.Saturation.Alpha = 1.514
model.Geom.s12.Saturation.N = 2.259
model.Geom.s12.Saturation.SRes = 0.0001
model.Geom.s12.Saturation.SSat = 1.0

model.Geom.s13.Saturation.Alpha = 1.122
model.Geom.s13.Saturation.N = 2.479
model.Geom.s13.Saturation.SRes = 0.0001
model.Geom.s13.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

model.Mannings.Type = "Constant"
model.Mannings.GeomNames = "domain"
model.Mannings.Geom.domain.Value = 0.0000044

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

model.Phase.Names = "water"
model.Phase.water.Density.Type = "Constant"
model.Phase.water.Density.Value = 1.0
model.Phase.water.Viscosity.Type = "Constant"
model.Phase.water.Viscosity.Value = 1.0
model.Phase.water.Mobility.Type = "Constant"
model.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

model.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

model.Gravity = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

model.Wells.Names = ""
# -----------------------------------------------------------------------------
# Phase Sources
# -----------------------------------------------------------------------------

model.PhaseSources.water.Type = "Constant"
model.PhaseSources.water.GeomNames = "domain"
model.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Set time information
# -----------------------------------------------------------------------------

model.TimingInfo.BaseUnit = 1.0
model.TimingInfo.StartCount = 0  # restart
model.TimingInfo.StartTime = 0.0  # restart
model.TimingInfo.StopTime = 5.0
model.TimingInfo.DumpInterval = 1
model.TimeStep.Type = "Constant"
model.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Time cycle
# -----------------------------------------------------------------------------

model.Cycle.Names = "constant"
model.Cycle.constant.Names = "alltime"
model.Cycle.constant.alltime.Length = 1
model.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

model.BCPressure.PatchNames = "x_lower x_upper y_lower y_upper z_lower z_upper"

model.Patch.x_lower.BCPressure.Type = "FluxConst"
model.Patch.x_lower.BCPressure.Cycle = "constant"
model.Patch.x_lower.BCPressure.alltime.Value = 0.0

model.Patch.x_upper.BCPressure.Type = "FluxConst"
model.Patch.x_upper.BCPressure.Cycle = "constant"
model.Patch.x_upper.BCPressure.alltime.Value = 0.0

model.Patch.y_lower.BCPressure.Type = "FluxConst"
model.Patch.y_lower.BCPressure.Cycle = "constant"
model.Patch.y_lower.BCPressure.alltime.Value = 0.0

model.Patch.y_upper.BCPressure.Type = "FluxConst"
model.Patch.y_upper.BCPressure.Cycle = "constant"
model.Patch.y_upper.BCPressure.alltime.Value = 0.0

model.Patch.z_lower.BCPressure.Type = "FluxConst"
model.Patch.z_lower.BCPressure.Cycle = "constant"
model.Patch.z_lower.BCPressure.alltime.Value = 0.0

model.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
model.Patch.z_upper.BCPressure.Cycle = "constant"
model.Patch.z_upper.BCPressure.alltime.Value = 0

# -----------------------------------------------------------------------------
# Initial conditions: water pressure
# -----------------------------------------------------------------------------

model.ICPressure.Type = "PFBFile"
model.ICPressure.GeomNames = "domain"
model.Geom.domain.ICPressure.RefPatch = "top"
model.Geom.domain.ICPressure.FileName = "initial_press_11layer.pfb"  # restart

# -----------------------------------------------------------------------------
# CoLM Settings
# -----------------------------------------------------------------------------

model.Solver.LSM = "CoLM"

model.Solver.CLM.CLMFileDir = "clm_output/"
model.Solver.CLM.Print1dOut = False
model.Solver.CLM.DailyRST = False
model.Solver.CLM.CLMDumpInterval = 1

model.Solver.CLM.MetFileName = "E5L"
model.Solver.CLM.MetFilePath = str(example_dir / "forcing")
model.Solver.CLM.MetForcing = "3D"
model.Solver.CLM.MetFileNT = 24
model.Solver.CLM.IstepStart = 1

model.Solver.CLM.EvapBeta = "Linear"
model.Solver.CLM.VegWaterStress = "Saturation"
model.Solver.CLM.ResSat = 0.1
model.Solver.CLM.WiltingPoint = 0.12
model.Solver.CLM.FieldCapacity = 0.98
model.Solver.CLM.IrrigationType = "none"
model.Solver.CLM.RootZoneNZ = 10
model.Solver.CLM.SoiLayer = 10

# -----------------------------------------------------------------------------
# Output Settings
# -----------------------------------------------------------------------------

model.Solver.PrintSubsurfData = True
model.Solver.PrintPressure = True
model.Solver.PrintSaturation = True
model.Solver.PrintMask = True
model.Solver.PrintVelocities = False
model.Solver.PrintEvapTrans = False
model.Solver.CLM.SingleFile = True
model.Solver.PrintSlopes = False
model.Solver.PrintMannings = False

model.Solver.WriteCLMBinary = False
model.Solver.PrintCLM = False

# -----------------------------------------------------------------------------
# Solver Settings
# -----------------------------------------------------------------------------

model.Solver = "Richards"
model.Solver.TerrainFollowingGrid = True
model.Solver.TerrainFollowingGrid.SlopeUpwindFormulation = "Upwind"
model.Solver.Linear.Preconditioner = "PFMG"
# model.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

model.KnownSolution = "NoKnownSolution"

model.Solver.MaxIter = 25000
model.Solver.Drop = 1e-20
model.Solver.AbsTol = 1e-8
model.Solver.MaxConvergenceFailures = 8
model.Solver.Nonlinear.MaxIter = 1000
model.Solver.Nonlinear.ResidualTol = 1e-6
model.Solver.Nonlinear.EtaChoice = "EtaConstant"
model.Solver.Nonlinear.EtaValue = 0.001
model.Solver.Nonlinear.UseJacobian = True
model.Solver.Nonlinear.DerivativeEpsilon = 1e-16
model.Solver.Nonlinear.StepTol = 1e-15
model.Solver.Nonlinear.Globalization = "LineSearch"
model.Solver.Linear.KrylovDimension = 70
model.Solver.Linear.MaxRestarts = 2

# -----------------------------------------------------------------------------
# Distribute inputs
# -----------------------------------------------------------------------------

model.dist("slope_x.pfb")
model.dist("slope_y.pfb")
model.dist("subsurface_11layer.pfb")
model.dist("initial_press_11layer.pfb")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

model.write()
model.write(file_format="yaml")
model.write(file_format="json")

model.run(working_directory=run_dir)

FINAL_TIMESTEP = 5
PRESSURE_SATURATION_ABS_TOLERANCE = 1.0e-4

reference_path = example_dir.parents[1] / "correct_output" / "colm_output"
passed = True

for field in ("perm_x", "perm_y", "perm_z"):
    name = f"PF_CoLM_CI.out.{field}.pfb"
    if not pf_test_file(
        str(run_path / name),
        str(reference_path / name),
        f"Max difference in {name}",
    ):
        passed = False

for timestep in range(FINAL_TIMESTEP + 1):
    for field in ("press", "satur"):
        name = f"PF_CoLM_CI.out.{field}.{timestep:05d}.pfb"
        if not pf_test_file_with_abs(
            str(run_path / name),
            str(reference_path / name),
            f"Max difference in {name}",
            abs_value=PRESSURE_SATURATION_ABS_TOLERANCE,
        ):
            passed = False

mask = read_pfb(str(run_path / "PF_CoLM_CI.out.mask.pfb"))
top = compute_top(mask)
write_pfb(str(run_path / "PF_CoLM_CI.out.top_index.pfb"), top)

pressure = read_pfb(str(run_path / "PF_CoLM_CI.out.press.00000.pfb"))
top_pressure = extract_top(pressure, top)
write_pfb(str(run_path / "PF_CoLM_CI.out.top.press.00000.pfb"), top_pressure)

for name in (
    "PF_CoLM_CI.out.top_index.pfb",
    "PF_CoLM_CI.out.top.press.00000.pfb",
):
    if not pf_test_file(
        str(run_path / name),
        str(reference_path / name),
        f"Max difference in {name}",
    ):
        passed = False

rm(run_dir)
if passed:
    print(f"{runname} : PASSED")
else:
    print(f"{runname} : FAILED")
    sys.exit(1)
