# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID

from pathlib import Path
import sys

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.builders import SubsurfacePropertiesBuilder

LW_Test = Run("LW_Test", __file__)

LW_Test.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------

LW_Test.Process.Topology.P = 1
LW_Test.Process.Topology.Q = 1
LW_Test.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

LW_Test.ComputationalGrid.Lower.X = 0.0
LW_Test.ComputationalGrid.Lower.Y = 0.0
LW_Test.ComputationalGrid.Lower.Z = 0.0

LW_Test.ComputationalGrid.DX = 1000.0
LW_Test.ComputationalGrid.DY = 1000.0
LW_Test.ComputationalGrid.DZ = 2.0

LW_Test.ComputationalGrid.NX = 41
LW_Test.ComputationalGrid.NY = 41
LW_Test.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

LW_Test.GeomInput.Names = "box_input indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

LW_Test.GeomInput.box_input.InputType = "Box"
LW_Test.GeomInput.box_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

LW_Test.Geom.domain.Lower.X = 0.0
LW_Test.Geom.domain.Lower.Y = 0.0
LW_Test.Geom.domain.Lower.Z = 0.0

LW_Test.Geom.domain.Upper.X = 41000.0
LW_Test.Geom.domain.Upper.Y = 41000.0
LW_Test.Geom.domain.Upper.Z = 100.0
LW_Test.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

LW_Test.GeomInput.indi_input.InputType = "IndicatorField"
LW_Test.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
)
LW_Test.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"

LW_Test.GeomInput.s1.Value = 1
LW_Test.GeomInput.s2.Value = 2
LW_Test.GeomInput.s3.Value = 3
LW_Test.GeomInput.s4.Value = 4
LW_Test.GeomInput.s5.Value = 5
LW_Test.GeomInput.s6.Value = 6
LW_Test.GeomInput.s7.Value = 7
LW_Test.GeomInput.s8.Value = 8
LW_Test.GeomInput.s9.Value = 9
LW_Test.GeomInput.s10.Value = 10
LW_Test.GeomInput.s11.Value = 11
LW_Test.GeomInput.s12.Value = 12
LW_Test.GeomInput.s13.Value = 13
LW_Test.GeomInput.g1.Value = 21
LW_Test.GeomInput.g2.Value = 22
LW_Test.GeomInput.g3.Value = 23
LW_Test.GeomInput.g4.Value = 24
LW_Test.GeomInput.g5.Value = 25
LW_Test.GeomInput.g6.Value = 26
LW_Test.GeomInput.g7.Value = 27
LW_Test.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

LW_subsurface_properties = """
key     Perm          Porosity   RelPermAlpha  RelPermN  SatAlpha  SatN    SatSRes    SatSSat
domain  0.2           0.4        3.5           2         3.5       2       0.2        1
s1      0.269022595   0.375      3.548         4.162     3.548     4.162   0.000001   1
s2      0.043630356   0.39       3.467         2.738     3.467     2.738   0.000001   1
s3      0.015841225   0.387      2.692         2.445     2.692     2.445   0.000001   1
s4      0.007582087   0.439      0.501         2.659     0.501     2.659   0.000001   1
s5      0.01818816    0.489      0.661         2.659     0.661     2.659   0.000001   1
s6      0.005009435   0.399      1.122         2.479     1.122     2.479   0.000001   1
s7      0.005492736   0.384      2.089         2.318     2.089     2.318   0.000001   1
s8      0.004675077   0.482      0.832         2.514     0.832     2.514   0.000001   1
s9      0.003386794   0.442      1.585         2.413     1.585     2.413   0.000001   1
g2      0.025         -          -             -         -         -       -          -
g3      0.059         -          -             -         -         -       -          -
g6      0.2           -          -             -         -         -       -          -
g8      0.68          -          -             -         -         -       -          -
"""

# standard inline procedure calls in Python
SubsurfacePropertiesBuilder(LW_Test).load_txt_content(LW_subsurface_properties).apply()

# alternative option separating method calls
# spb = SubsurfacePropertiesBuilder(LW_Test)
# spb.load_txt_content(LW_subsurface_properties)
# spb.apply(name_registration=True)

LW_Test.Perm.TensorType = "TensorByGeom"
LW_Test.Geom.Perm.TensorByGeom.Names = "domain"
LW_Test.Geom.domain.Perm.TensorValX = 1.0
LW_Test.Geom.domain.Perm.TensorValY = 1.0
LW_Test.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

LW_Test.SpecificStorage.Type = "Constant"
LW_Test.SpecificStorage.GeomNames = "domain"
LW_Test.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

LW_Test.Phase.Names = "water"
LW_Test.Phase.water.Density.Type = "Constant"
LW_Test.Phase.water.Density.Value = 1.0
LW_Test.Phase.water.Viscosity.Type = "Constant"
LW_Test.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

LW_Test.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

LW_Test.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

LW_Test.TimingInfo.BaseUnit = 1.0
LW_Test.TimingInfo.StartCount = 0
LW_Test.TimingInfo.StartTime = 0.0
LW_Test.TimingInfo.StopTime = 1000.0
LW_Test.TimingInfo.DumpInterval = 1.0
LW_Test.TimeStep.Type = "Constant"
LW_Test.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

LW_Test.Domain.GeomName = "domain"

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

LW_Test.Phase.water.Mobility.Type = "Constant"
LW_Test.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

LW_Test.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

LW_Test.Cycle.Names = "constant rainrec"
LW_Test.Cycle.constant.Names = "alltime"
LW_Test.Cycle.constant.alltime.Length = 1
LW_Test.Cycle.constant.Repeat = -1

LW_Test.Cycle.rainrec.Names = "rain rec"
LW_Test.Cycle.rainrec.rain.Length = 10
LW_Test.Cycle.rainrec.rec.Length = 150
LW_Test.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

LW_Test.BCPressure.PatchNames = LW_Test.Geom.domain.Patches

LW_Test.Patch.x_lower.BCPressure.Type = "FluxConst"
LW_Test.Patch.x_lower.BCPressure.Cycle = "constant"
LW_Test.Patch.x_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.y_lower.BCPressure.Type = "FluxConst"
LW_Test.Patch.y_lower.BCPressure.Cycle = "constant"
LW_Test.Patch.y_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.z_lower.BCPressure.Type = "FluxConst"
LW_Test.Patch.z_lower.BCPressure.Cycle = "constant"
LW_Test.Patch.z_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.x_upper.BCPressure.Type = "FluxConst"
LW_Test.Patch.x_upper.BCPressure.Cycle = "constant"
LW_Test.Patch.x_upper.BCPressure.alltime.Value = 0.0

LW_Test.Patch.y_upper.BCPressure.Type = "FluxConst"
LW_Test.Patch.y_upper.BCPressure.Cycle = "constant"
LW_Test.Patch.y_upper.BCPressure.alltime.Value = 0.0

LW_Test.Patch.z_upper.BCPressure.Type = "OverlandFlow"
LW_Test.Patch.z_upper.BCPressure.Cycle = "rainrec"
LW_Test.Patch.z_upper.BCPressure.rain.Value = -0.1
LW_Test.Patch.z_upper.BCPressure.rec.Value = 0.0000

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

LW_Test.TopoSlopesX.Type = "PFBFile"
LW_Test.TopoSlopesX.GeomNames = "domain"
LW_Test.TopoSlopesX.FileName = "LW.slopex.pfb"

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

LW_Test.TopoSlopesY.Type = "PFBFile"
LW_Test.TopoSlopesY.GeomNames = "domain"
LW_Test.TopoSlopesY.FileName = "LW.slopey.pfb"

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

LW_Test.Mannings.Type = "Constant"
LW_Test.Mannings.GeomNames = "domain"
LW_Test.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

LW_Test.PhaseSources.water.Type = "Constant"
LW_Test.PhaseSources.water.GeomNames = "domain"
LW_Test.PhaseSources.water.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

LW_Test.ICPressure.Type = "PFBFile"
LW_Test.ICPressure.GeomNames = "domain"
LW_Test.Geom.domain.ICPressure.RefPatch = "z_upper"
LW_Test.Geom.domain.ICPressure.FileName = "press.init.pfb"

# ----------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------

LW_Test.Solver.PrintSubsurfData = False
LW_Test.Solver.PrintPressure = True
LW_Test.Solver.PrintSaturation = True
LW_Test.Solver.PrintMask = True

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

LW_Test.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

# ParFlow Solution
LW_Test.Solver = "Richards"
LW_Test.Solver.TerrainFollowingGrid = True
LW_Test.Solver.Nonlinear.VariableDz = False

LW_Test.Solver.MaxIter = 25000
LW_Test.Solver.Drop = 1e-20
LW_Test.Solver.AbsTol = 1e-8
LW_Test.Solver.MaxConvergenceFailures = 8
LW_Test.Solver.Nonlinear.MaxIter = 80
LW_Test.Solver.Nonlinear.ResidualTol = 1e-6

LW_Test.Solver.Nonlinear.EtaChoice = "EtaConstant"
LW_Test.Solver.Nonlinear.EtaValue = 0.001
LW_Test.Solver.Nonlinear.UseJacobian = True
LW_Test.Solver.Nonlinear.DerivativeEpsilon = 1e-16
LW_Test.Solver.Nonlinear.StepTol = 1e-30
LW_Test.Solver.Nonlinear.Globalization = "LineSearch"
LW_Test.Solver.Linear.KrylovDimension = 70
LW_Test.Solver.Linear.MaxRestarts = 2

LW_Test.Solver.Linear.Preconditioner = "PFMG"


def test_subsurface_table(file_name):
    path = "$PF_SRC/test/correct_output/tables_LW_subsurface.txt.ref"
    ref = Path(get_absolute_path(path)).read_text()
    new = Path(get_absolute_path(file_name)).read_text()
    assert ref == new


# -----------------------------------------------------------------------------
# Comparing written key/value pairs
# -----------------------------------------------------------------------------


def test_output(file_name):
    LW_Test.write(file_name, file_format="yaml")
    with open(get_absolute_path(f"{file_name}.yaml")) as new, open(
        get_absolute_path("$PF_SRC/test/correct_output/LW_test_ref.yaml.ref")
    ) as ref:
        if new.read() == ref.read():
            print("Success we have the same file")
            return True
        else:
            print("Files are different")
            return False


# testing inline text
print("+" * 40)
print("Comparing inline text to ref:")

LW_Test.write_subsurface_table("inline_input.txt")
test_subsurface_table("inline_input.txt")

if not test_output("inline_input_yaml"):
    sys.exit(1)
print("")

# resetting properties to zero
print("+" * 40)
print("Clearing:")
SubsurfacePropertiesBuilder(LW_Test).load_csv_file(
    "$PF_SRC/test/input/LW_test_data_clear.csv"
).apply()
if test_output("clear_yaml"):  # should fail
    sys.exit(1)
print("+" * 40)
print("")

# testing csv file
print("Loading csv file:")
SubsurfacePropertiesBuilder(LW_Test).load_csv_file(
    "$PF_SRC/test/input/LW_test_data.csv"
).apply().print_as_table(props_in_header=False)
print("+" * 40)
print("Comparing csv file to ref:")
if not test_output("csv_input_yaml"):
    sys.exit(1)
print("+" * 40)
print("")

# resetting properties to zero
print("Clearing:")
SubsurfacePropertiesBuilder(LW_Test).load_csv_file(
    "$PF_SRC/test/input/LW_test_data_clear.csv"
).apply()
if test_output("clear_yaml"):  # should fail
    sys.exit(1)
print("+" * 40)
print("")

# testing txt file
print("Loading txt file:")
SubsurfacePropertiesBuilder(LW_Test).load_txt_file(
    "$PF_SRC/test/input/LW_test_data.txt"
).apply().print_as_table()
print("+" * 40)
print("Comparing txt file to ref:")
if not test_output("txt_input_yaml"):
    sys.exit(1)
print("+" * 40)

# resetting properties to zero
print("Clearing:")
SubsurfacePropertiesBuilder(LW_Test).load_csv_file(
    "$PF_SRC/test/input/LW_test_data_clear.csv"
).apply()
if test_output("clear_yaml"):  # should fail
    sys.exit(1)
print("+" * 40)
print("")

# testing txt file
print("Loading transposed txt file:")
SubsurfacePropertiesBuilder(LW_Test).load_txt_file(
    "$PF_SRC/test/input/LW_test_data_transposed.txt"
).apply().print()
print("+" * 40)
print("Comparing transposed txt file to ref:")
if not test_output("trasnposed_txt_input_yaml"):
    sys.exit(1)
print("+" * 40)
