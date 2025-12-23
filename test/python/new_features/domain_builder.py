# ---------------------------------------------------------
#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.
# ---------------------------------------------------------

import sys

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.builders import DomainBuilder

drich = Run("drich", __file__)

# ---------------------------------------------------------

drich.Process.Topology.P = 1
drich.Process.Topology.Q = 1
drich.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

drich.ComputationalGrid.Lower.X = -10.0
drich.ComputationalGrid.Lower.Y = 10.0
drich.ComputationalGrid.Lower.Z = 1.0

drich.ComputationalGrid.DX = 8.8888888888888893
drich.ComputationalGrid.DY = 10.666666666666666
drich.ComputationalGrid.DZ = 1.0

drich.ComputationalGrid.NX = 10
drich.ComputationalGrid.NY = 10
drich.ComputationalGrid.NZ = 8

# ---------------------------------------------------------
# Time Cycles
# ---------------------------------------------------------

drich.Cycle.Names = "constant"
drich.Cycle.constant.Names = "alltime"
drich.Cycle.constant.alltime.Length = 1
drich.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

drich.TimingInfo.BaseUnit = 1.0
drich.TimingInfo.StartCount = 0
drich.TimingInfo.StartTime = 0.0
drich.TimingInfo.StopTime = 0.010
drich.TimingInfo.DumpInterval = -1
drich.TimeStep.Type = "Constant"
drich.TimeStep.Value = 0.001

# ---------------------------------------------------------
# Domain setup using builder
# ---------------------------------------------------------

DomainBuilder(drich).water("background").variably_saturated().box_domain(
    "domain_input",
    "domain",
    bounds=[-10.0, 150.0, 10.0, 170.0, 1.0, 9.0],  # x range  # y range  # z range
    patches="left right front back bottom top",
).box_domain(
    "background_input",
    "background",
    bounds=[-99999999.0, 99999999.0, -99999999.0, 99999999.0, -99999999.0, 99999999.0],
).box_domain(
    "source_region_input",
    "source_region",
    bounds=[65.56, 74.44, 79.34, 89.99, 4.5, 5.5],
).box_domain(
    "concen_region_input", "concen_region", bounds=[60.0, 80.0, 80.0, 100.0, 4.0, 6.0]
).slopes_mannings(
    "domain", slope_x=0.0, slope_y=0.0, mannings=0.0
).homogeneous_subsurface(
    "domain",
    specific_storage=1.0e-4,
    rel_perm={"Type": "VanGenuchten", "Alpha": 0.005, "N": 2.0},
    saturation={
        "Type": "VanGenuchten",
        "Alpha": 0.005,  # optional, uses RelPerm.Alpha if missing
        "N": 2.0,  # optional, uses RelPerm.N if missing
        "SRes": 0.2,
        "SSat": 0.99,
    },
).homogeneous_subsurface(
    "background", perm=4.0, porosity=1.0, isotropic=True
).zero_flux(
    patches="front back bottom top", cycle_name="constant", interval_name="alltime"
)

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

drich.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

drich.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

drich.Wells.Names = ""

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

drich.BCPressure.PatchNames = "left right front back bottom top"

drich.Patch.left.BCPressure.Type = "DirEquilRefPatch"
drich.Patch.left.BCPressure.Cycle = "constant"
drich.Patch.left.BCPressure.RefGeom = "domain"
drich.Patch.left.BCPressure.RefPatch = "bottom"
drich.Patch.left.BCPressure.alltime.Value = 5.0

drich.Patch.right.BCPressure.Type = "DirEquilRefPatch"
drich.Patch.right.BCPressure.Cycle = "constant"
drich.Patch.right.BCPressure.RefGeom = "domain"
drich.Patch.right.BCPressure.RefPatch = "bottom"
drich.Patch.right.BCPressure.alltime.Value = 3.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

drich.ICPressure.Type = "HydroStaticPatch"
drich.ICPressure.GeomNames = "domain"
drich.Geom.domain.ICPressure.Value = 3.0
drich.Geom.domain.ICPressure.RefGeom = "domain"
drich.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

drich.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Run ParFlow
# -----------------------------------------------------------------------------

drich.validate()

drich.write(file_format="yaml")

with open(get_absolute_path("drich.yaml")) as new, open(
    get_absolute_path("../../correct_output/domain_builder.yaml.ref")
) as ref:
    if new.read() == ref.read():
        print("Success we have the same file")
    else:
        print("Files are different")
        sys.exit(1)
