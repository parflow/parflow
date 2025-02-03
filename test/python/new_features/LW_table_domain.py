# ---------------------------------------------------------
# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID
# Using combination of tables and domain builder
# ---------------------------------------------------------

import sys

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.builders import SubsurfacePropertiesBuilder, DomainBuilder

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
# Using domain builder
# -----------------------------------------------------------------------------

bounds = [0.0, 41000.0, 0.0, 41000.0, 0.0, 100.0]

domain_patches = "x_lower x_upper y_lower y_upper z_lower z_upper"
zero_flux_patches = "x_lower x_upper y_lower y_upper z_lower"

DomainBuilder(LW_Test).no_wells().no_contaminants().water(
    "domain"
).variably_saturated().box_domain(
    "box_input", "domain", bounds, domain_patches
).homogeneous_subsurface(
    "domain", specific_storage=1.0e-5, isotropic=True
).zero_flux(
    zero_flux_patches, "constant", "alltime"
).slopes_mannings(
    "domain", slope_x="LW.slopex.pfb", slope_y="LW.slopey.pfb", mannings=5.52e-6
).ic_pressure(
    "domain", patch="z_upper", pressure="press.init.pfb"
)

# -----------------------------------------------------------------------------
# Adding indi_input to GeomInput.Names
# -----------------------------------------------------------------------------

LW_Test.GeomInput.Names = "box_input indi_input"

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

# Setting subsurface properties
SubsurfacePropertiesBuilder(LW_Test).load_txt_content(LW_subsurface_properties).apply()

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

LW_Test.BCPressure.PatchNames = LW_Test.Geom.domain.Patches

LW_Test.Patch.z_upper.BCPressure.Type = "OverlandFlow"
LW_Test.Patch.z_upper.BCPressure.Cycle = "rainrec"
LW_Test.Patch.z_upper.BCPressure.rain.Value = -0.1
LW_Test.Patch.z_upper.BCPressure.rec.Value = 0.0000

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

LW_Test.Solver.TerrainFollowingGrid = True
LW_Test.Solver.Nonlinear.VariableDz = False

LW_Test.Solver.MaxIter = 25000
LW_Test.Solver.Drop = 1e-20
LW_Test.Solver.AbsTol = 1e-8
LW_Test.Solver.MaxConvergenceFailures = 8
LW_Test.Solver.Nonlinear.MaxIter = 80
LW_Test.Solver.Nonlinear.ResidualTol = 1e-6

LW_Test.Solver.Nonlinear.EtaValue = 0.001
LW_Test.Solver.Nonlinear.DerivativeEpsilon = 1e-16
LW_Test.Solver.Nonlinear.StepTol = 1e-30
LW_Test.Solver.Nonlinear.Globalization = "LineSearch"
LW_Test.Solver.Linear.KrylovDimension = 70
LW_Test.Solver.Linear.MaxRestarts = 2

# -----------------------------------------------------------------------------
# Test validation
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


# testing output
print("+" * 40)
print("Comparing table and domain builder output to ref:")
print("...")
# LW_Test.write_subsurface_table('inline_input.txt')
if not test_output("LW_table_domain"):
    sys.exit(1)
print("")
