# -----------------------------------------------------------------------------
# 1D non-reactive tracer transport test (react_trans, Python Run API).
#
# Python translation of test/chemistry/tracer_crunch.tcl. CrunchFlow-backed
# Alquimia case reproducing the tracer profile of Molins et al. 2025
# (GMD 18:3241, Fig. 2a): a non-reactive front at 50 m by step 5. The engine
# input deck (1d-tracer-crunch.in), thermodynamic database (tracer.dbs), and
# reference output are shared with the TCL test under test/chemistry/.
#
# Runs only when ParFlow is built with PARFLOW_ENABLE_ALQUIMIA=ON against a
# CrunchFlow-backed Alquimia install (gated in CMake on PARFLOW_HAVE_ALQUIMIA).
# -----------------------------------------------------------------------------
import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "tracer_pf"
tracer = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
tracer.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
tracer.Process.Topology.P = 1
tracer.Process.Topology.Q = 1
tracer.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
tracer.ComputationalGrid.Lower.X = 0.0
tracer.ComputationalGrid.Lower.Y = 0.0
tracer.ComputationalGrid.Lower.Z = 0.0

tracer.ComputationalGrid.DX = 1.0
tracer.ComputationalGrid.DY = 1.0
tracer.ComputationalGrid.DZ = 1.0

tracer.ComputationalGrid.NX = 100
tracer.ComputationalGrid.NY = 1
tracer.ComputationalGrid.NZ = 1

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
tracer.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
tracer.GeomInput.domain_input.InputType = "Box"
tracer.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
tracer.Geom.domain.Lower.X = 0.0
tracer.Geom.domain.Lower.Y = 0.0
tracer.Geom.domain.Lower.Z = 0.0

tracer.Geom.domain.Upper.X = 100.0
tracer.Geom.domain.Upper.Y = 1.0
tracer.Geom.domain.Upper.Z = 1.0

tracer.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Background Geometry Input
# -----------------------------------------------------------------------------
tracer.GeomInput.background_input.InputType = "Box"
tracer.GeomInput.background_input.GeomName = "background"

# -----------------------------------------------------------------------------
# Background Geometry
# -----------------------------------------------------------------------------
tracer.Geom.background.Lower.X = -99999999.0
tracer.Geom.background.Lower.Y = -99999999.0
tracer.Geom.background.Lower.Z = -99999999.0

tracer.Geom.background.Upper.X = 99999999.0
tracer.Geom.background.Upper.Y = 99999999.0
tracer.Geom.background.Upper.Z = 99999999.0

# -----------------------------------------------------------------------------
# Source_Region Geometry Input
# -----------------------------------------------------------------------------
tracer.GeomInput.source_region_input.InputType = "Box"
tracer.GeomInput.source_region_input.GeomName = "source_region"

# -----------------------------------------------------------------------------
# Source_Region Geometry
# -----------------------------------------------------------------------------
tracer.Geom.source_region.Lower.X = 0.0
tracer.Geom.source_region.Lower.Y = 0.0
tracer.Geom.source_region.Lower.Z = 0.0

tracer.Geom.source_region.Upper.X = 100.0
tracer.Geom.source_region.Upper.Y = 1.0
tracer.Geom.source_region.Upper.Z = 1.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
tracer.Geom.Perm.Names = "background"

tracer.Geom.background.Perm.Type = "Constant"
tracer.Geom.background.Perm.Value = 0.25

tracer.Perm.TensorType = "TensorByGeom"

tracer.Geom.Perm.TensorByGeom.Names = "background"

tracer.Geom.background.Perm.TensorValX = 1.0
tracer.Geom.background.Perm.TensorValY = 1.0
tracer.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Concen_Region Geometry Input
# -----------------------------------------------------------------------------
tracer.GeomInput.concen_region_input.InputType = "Box"
tracer.GeomInput.concen_region_input.GeomName = "concen_region"

# -----------------------------------------------------------------------------
# Concen_Region Geometry
# -----------------------------------------------------------------------------
tracer.Geom.concen_region.Lower.X = 0.0
tracer.Geom.concen_region.Lower.Y = 0.0
tracer.Geom.concen_region.Lower.Z = 0.0

tracer.Geom.concen_region.Upper.X = 100.0
tracer.Geom.concen_region.Upper.Y = 1.0
tracer.Geom.concen_region.Upper.Z = 1.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
tracer.Phase.Names = "water"

tracer.Phase.water.Density.Type = "Constant"
tracer.Phase.water.Density.Value = 1.0

tracer.Phase.water.Viscosity.Type = "Constant"
tracer.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
tracer.Contaminants.Names = "tce"
tracer.Contaminants.tce.Degradation.Value = 0.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
tracer.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
tracer.TimingInfo.BaseUnit = 1.0
tracer.TimingInfo.StartCount = 0
tracer.TimingInfo.StartTime = 0.0
tracer.TimingInfo.StopTime = 50.0
tracer.TimingInfo.DumpInterval = 10.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
tracer.Geom.Porosity.GeomNames = "background"

tracer.Geom.background.Porosity.Type = "Constant"
tracer.Geom.background.Porosity.Value = 0.25

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
tracer.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
tracer.Phase.water.Mobility.Type = "Constant"
tracer.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
tracer.Geom.Retardation.GeomNames = "background"
tracer.Geom.background.tce.Retardation.Type = "Linear"
tracer.Geom.background.tce.Retardation.Rate = 0.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
tracer.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
tracer.Cycle.Names = "constant"
tracer.Cycle.constant.Names = "alltime"
tracer.Cycle.constant.alltime.Length = 20
tracer.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
tracer.BCPressure.PatchNames = "left right front back bottom top"

tracer.Patch.left.BCPressure.Type = "DirEquilRefPatch"
tracer.Patch.left.BCPressure.Cycle = "constant"
tracer.Patch.left.BCPressure.RefGeom = "domain"
tracer.Patch.left.BCPressure.RefPatch = "bottom"
tracer.Patch.left.BCPressure.alltime.Value = 200.0

tracer.Patch.right.BCPressure.Type = "DirEquilRefPatch"
tracer.Patch.right.BCPressure.Cycle = "constant"
tracer.Patch.right.BCPressure.RefGeom = "domain"
tracer.Patch.right.BCPressure.RefPatch = "bottom"
tracer.Patch.right.BCPressure.alltime.Value = 100.0

tracer.Patch.top.BCPressure.Type = "FluxConst"
tracer.Patch.top.BCPressure.Cycle = "constant"
tracer.Patch.top.BCPressure.alltime.Value = 0.0

tracer.Patch.bottom.BCPressure.Type = "FluxConst"
tracer.Patch.bottom.BCPressure.Cycle = "constant"
tracer.Patch.bottom.BCPressure.alltime.Value = 0.0

tracer.Patch.back.BCPressure.Type = "FluxConst"
tracer.Patch.back.BCPressure.Cycle = "constant"
tracer.Patch.back.BCPressure.alltime.Value = 0.0

tracer.Patch.front.BCPressure.Type = "FluxConst"
tracer.Patch.front.BCPressure.Cycle = "constant"
tracer.Patch.front.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
tracer.PhaseSources.water.Type = "Constant"
tracer.PhaseSources.water.GeomNames = "background"
tracer.PhaseSources.water.Geom.background.Value = 0.0

tracer.PhaseConcen.water.tce.Type = "Constant"
tracer.PhaseConcen.water.tce.GeomNames = "concen_region"
tracer.PhaseConcen.water.tce.Geom.concen_region.Value = 0.1

# The TCL deck also carries bare PhaseSources.* and TempSources.* keys; both are
# vestigial. The compiled phase-source module (problem_phase_source.c) reads
# only the per-phase form above, and TempSources has no reader in the build, so
# they are dropped here without changing the result.

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
tracer.SpecificStorage.Type = "Constant"
tracer.SpecificStorage.GeomNames = "background"
tracer.Geom.background.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Heat Capacity
# -----------------------------------------------------------------------------
tracer.Phase.water.HeatCapacity.Type = "Constant"
tracer.Phase.water.HeatCapacity.GeomNames = "background"
tracer.Phase.water.Geom.background.HeatCapacity.Value = 4000.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
tracer.TopoSlopesX.Type = "Constant"
tracer.TopoSlopesX.GeomNames = "domain"
tracer.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
tracer.TopoSlopesY.Type = "Constant"
tracer.TopoSlopesY.GeomNames = "domain"
tracer.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
tracer.Mannings.Type = "Constant"
tracer.Mannings.GeomNames = "domain"
tracer.Mannings.Geom.domain.Value = 2.3e-7

# ---------------------------------------------------------
# Alquimia chemistry input
# ---------------------------------------------------------
tracer.Solver.Chemistry = "Alquimia"
tracer.Chemistry.Engine = "CrunchFlow"
tracer.Chemistry.InputFile = "1d-tracer-crunch.in"

# order of geomnames matters: like every other PF geometry, geominputs listed
# later overwrite earlier ones where they overlap
tracer.GeochemCondition.Type = "Constant"
tracer.GeochemCondition.GeomNames = "concen_region"
tracer.GeochemCondition.Names = "initial"
tracer.GeochemCondition.Geom.concen_region.Value = "initial"

tracer.BCConcentration.GeochemCondition.Names = "west"
tracer.BCConcentration.PatchNames = "left"
tracer.Patch.left.BCConcentration.Type = "Constant"
tracer.Patch.left.BCConcentration.Value = "west"

tracer.Chemistry.ParFlowTimeUnits = "years"

tracer.Chemistry.PrintPrimaryMobile = True
tracer.Chemistry.PrintpH = False

# -----------------------------------------------------------------------------
# Solver settings
# -----------------------------------------------------------------------------
tracer.Solver.MaxIter = 50000
tracer.Solver.CFL = 0.6666666666667
tracer.Solver.AdvectOrder = 2
tracer.Solver.AdvectEnforceMinMax = True
tracer.Solver.RelTol = 1.0e-35
tracer.Solver.AbsTol = 1.0e-50
tracer.Solver.Nonlinear.ResidualTol = 1.0e-15

# -----------------------------------------------------------------------------
# Run and verify against the shared Molins et al. 2025 reference output
# -----------------------------------------------------------------------------
chem_dir = get_absolute_path("../../chemistry")
correct_output_dir_name = chem_dir + "/correct_output"
new_output_dir_name = get_absolute_path("test_output/tracer")
mkdir(new_output_dir_name)

# The engine reads the input deck and its database relative to the run cwd, so
# stage both into the working directory before running.
cp(chem_dir + "/1d-tracer-crunch.in", new_output_dir_name)
cp(chem_dir + "/tracer.dbs", new_output_dir_name)

tracer.run(working_directory=new_output_dir_name)

passed = True
test_file = "PrimaryMobile.00.tracer.00005"
filename = f"/{run_name}.out.{test_file}.pfb"
if not pf_test_file(
    new_output_dir_name + filename,
    correct_output_dir_name + filename,
    f"Max difference in {test_file}",
):
    passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
