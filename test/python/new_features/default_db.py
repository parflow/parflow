# -----------------------------------------------------------------------------
# Testing loading of default database and adding run name argument to both the
# initial class definition and the apply
# -----------------------------------------------------------------------------

import sys

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.builders import SubsurfacePropertiesBuilder

db_test = Run("db_test", __file__)

db_test.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------

db_test.Process.Topology.P = 1
db_test.Process.Topology.Q = 1
db_test.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

db_test.ComputationalGrid.Lower.X = 0.0
db_test.ComputationalGrid.Lower.Y = 0.0
db_test.ComputationalGrid.Lower.Z = 0.0

db_test.ComputationalGrid.DX = 1000.0
db_test.ComputationalGrid.DY = 1000.0
db_test.ComputationalGrid.DZ = 2.0

db_test.ComputationalGrid.NX = 41
db_test.ComputationalGrid.NY = 41
db_test.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

db_test.GeomInput.Names = "box_input indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

db_test.GeomInput.box_input.InputType = "Box"
db_test.GeomInput.box_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

db_test.Geom.domain.Lower.X = 0.0
db_test.Geom.domain.Lower.Y = 0.0
db_test.Geom.domain.Lower.Z = 0.0

db_test.Geom.domain.Upper.X = 41000.0
db_test.Geom.domain.Upper.Y = 41000.0
db_test.Geom.domain.Upper.Z = 100.0
db_test.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

db_test.GeomInput.indi_input.InputType = "IndicatorField"
db_test.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
)
db_test.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"

db_test.GeomInput.s1.Value = 1
db_test.GeomInput.s2.Value = 2
db_test.GeomInput.s3.Value = 3
db_test.GeomInput.s4.Value = 4
db_test.GeomInput.s5.Value = 5
db_test.GeomInput.s6.Value = 6
db_test.GeomInput.s7.Value = 7
db_test.GeomInput.s8.Value = 8
db_test.GeomInput.s9.Value = 9
db_test.GeomInput.s10.Value = 10
db_test.GeomInput.s11.Value = 11
db_test.GeomInput.s12.Value = 12
db_test.GeomInput.s13.Value = 13
db_test.GeomInput.g1.Value = 21
db_test.GeomInput.g2.Value = 22
db_test.GeomInput.g3.Value = 23
db_test.GeomInput.g4.Value = 24
db_test.GeomInput.g5.Value = 25
db_test.GeomInput.g6.Value = 26
db_test.GeomInput.g7.Value = 27
db_test.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Reading subsurface properties from database
# -----------------------------------------------------------------------------

subsurf_mapping = {
    "bedrock_1": ["domain", "g2"],
    "sand": "s1",
    "loamy_sand": "s2",
    "sandy_loam": "s3",
    "silt_loam": "s4",
    "silt": "s5",
    "sandy_clay": "s6",
    "loam": "s7",
    "sandy_clay_loam": "s8",
    "silty_clay_loam": "s9",
    "clay_loam": "s10",
    "silty_clay": "s11",
    "clay": "s12",
    "organic": "s13",
    "sil_sedimentary": "g1",
    "bedrock_2": "g3",
    "crystalline": "g4",
    "fg_unconsolidated": "g5",
    "unconsolidated": "g6",
    "cg_sil_sedimentary": "g7",
    "carbonate": "g8",
}

# cloning run object to test optional run assignment
db_test_2 = db_test.clone("db_test_2")

# standard inline procedure calls in Python
SubsurfacePropertiesBuilder(db_test).load_default_properties().assign(
    mapping=subsurf_mapping
).apply().print_as_table()

# testing optional run assignment in apply
SubsurfacePropertiesBuilder().load_default_properties().assign(
    mapping=subsurf_mapping
).assign("bedrock_2", "g4").apply(db_test_2).print_as_table()

# or you can assign the properties one by one
# SubsurfacePropertiesBuilder()\
#   .load_default_properties() \
#   .assign('bedrock_1', ['domain', 'g2']) \
#   .assign('sand', 's1') \
#   .assign('loamy_sand', 's2') \
#   .apply(db_test_2) \
#   .print_as_table()


db_test.Perm.TensorType = "TensorByGeom"
db_test.Geom.Perm.TensorByGeom.Names = "domain"
db_test.Geom.domain.Perm.TensorValX = 1.0
db_test.Geom.domain.Perm.TensorValY = 1.0
db_test.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

db_test.SpecificStorage.Type = "Constant"
db_test.SpecificStorage.GeomNames = "domain"
db_test.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

db_test.Phase.Names = "water"
db_test.Phase.water.Density.Type = "Constant"
db_test.Phase.water.Density.Value = 1.0
db_test.Phase.water.Viscosity.Type = "Constant"
db_test.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

db_test.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

db_test.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

db_test.TimingInfo.BaseUnit = 1.0
db_test.TimingInfo.StartCount = 0
db_test.TimingInfo.StartTime = 0.0
db_test.TimingInfo.StopTime = 1000.0
db_test.TimingInfo.DumpInterval = 1.0
db_test.TimeStep.Type = "Constant"
db_test.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

db_test.Domain.GeomName = "domain"

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

db_test.Phase.water.Mobility.Type = "Constant"
db_test.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

db_test.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

db_test.Cycle.Names = "constant rainrec"
db_test.Cycle.constant.Names = "alltime"
db_test.Cycle.constant.alltime.Length = 1
db_test.Cycle.constant.Repeat = -1

db_test.Cycle.rainrec.Names = "rain rec"
db_test.Cycle.rainrec.rain.Length = 10
db_test.Cycle.rainrec.rec.Length = 150
db_test.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

db_test.BCPressure.PatchNames = db_test.Geom.domain.Patches

db_test.Patch.x_lower.BCPressure.Type = "FluxConst"
db_test.Patch.x_lower.BCPressure.Cycle = "constant"
db_test.Patch.x_lower.BCPressure.alltime.Value = 0.0

db_test.Patch.y_lower.BCPressure.Type = "FluxConst"
db_test.Patch.y_lower.BCPressure.Cycle = "constant"
db_test.Patch.y_lower.BCPressure.alltime.Value = 0.0

db_test.Patch.z_lower.BCPressure.Type = "FluxConst"
db_test.Patch.z_lower.BCPressure.Cycle = "constant"
db_test.Patch.z_lower.BCPressure.alltime.Value = 0.0

db_test.Patch.x_upper.BCPressure.Type = "FluxConst"
db_test.Patch.x_upper.BCPressure.Cycle = "constant"
db_test.Patch.x_upper.BCPressure.alltime.Value = 0.0

db_test.Patch.y_upper.BCPressure.Type = "FluxConst"
db_test.Patch.y_upper.BCPressure.Cycle = "constant"
db_test.Patch.y_upper.BCPressure.alltime.Value = 0.0

db_test.Patch.z_upper.BCPressure.Type = "OverlandFlow"
db_test.Patch.z_upper.BCPressure.Cycle = "rainrec"
db_test.Patch.z_upper.BCPressure.rain.Value = -0.1
db_test.Patch.z_upper.BCPressure.rec.Value = 0.0000

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

db_test.TopoSlopesX.Type = "PFBFile"
db_test.TopoSlopesX.GeomNames = "domain"
db_test.TopoSlopesX.FileName = "LW.slopex.pfb"

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

db_test.TopoSlopesY.Type = "PFBFile"
db_test.TopoSlopesY.GeomNames = "domain"
db_test.TopoSlopesY.FileName = "LW.slopey.pfb"

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

db_test.Mannings.Type = "Constant"
db_test.Mannings.GeomNames = "domain"
db_test.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

db_test.PhaseSources.water.Type = "Constant"
db_test.PhaseSources.water.GeomNames = "domain"
db_test.PhaseSources.water.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

db_test.ICPressure.Type = "PFBFile"
db_test.ICPressure.GeomNames = "domain"
db_test.Geom.domain.ICPressure.RefPatch = "z_upper"
db_test.Geom.domain.ICPressure.FileName = "press.init.pfb"

# ----------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------

db_test.Solver.PrintSubsurfData = False
db_test.Solver.PrintPressure = True
db_test.Solver.PrintSaturation = True
db_test.Solver.PrintMask = True

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

db_test.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

# ParFlow Solution
db_test.Solver = "Richards"
db_test.Solver.TerrainFollowingGrid = True
db_test.Solver.Nonlinear.VariableDz = False

db_test.Solver.MaxIter = 25000
db_test.Solver.Drop = 1e-20
db_test.Solver.AbsTol = 1e-8
db_test.Solver.MaxConvergenceFailures = 8
db_test.Solver.Nonlinear.MaxIter = 80
db_test.Solver.Nonlinear.ResidualTol = 1e-6

db_test.Solver.Nonlinear.EtaChoice = "EtaConstant"
db_test.Solver.Nonlinear.EtaValue = 0.001
db_test.Solver.Nonlinear.UseJacobian = True
db_test.Solver.Nonlinear.DerivativeEpsilon = 1e-16
db_test.Solver.Nonlinear.StepTol = 1e-30
db_test.Solver.Nonlinear.Globalization = "LineSearch"
db_test.Solver.Linear.KrylovDimension = 70
db_test.Solver.Linear.MaxRestarts = 2

db_test.Solver.Linear.Preconditioner = "PFMG"
