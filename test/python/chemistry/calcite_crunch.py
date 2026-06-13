# -----------------------------------------------------------------------------
# 1D calcite-dissolution reactive-transport test (react_trans, Python Run API).
#
# Python translation of test/chemistry/calcite_crunch.tcl. CrunchFlow-backed
# Alquimia case reproducing the calcite profile of Molins et al. 2025
# (GMD 18:3241, Fig. 2): a dissolution front at ~22-23 m by step 5. The engine
# input deck (1d-calcite-crunch.in) and thermodynamic database (calcite.dbs)
# and the reference outputs are shared with the TCL test under test/chemistry/.
#
# Runs only when ParFlow is built with PARFLOW_ENABLE_ALQUIMIA=ON against a
# CrunchFlow-backed Alquimia install (gated in CMake on PARFLOW_HAVE_ALQUIMIA).
# -----------------------------------------------------------------------------
import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "calcite_pf"
calcite = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
calcite.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------
calcite.Process.Topology.P = 1
calcite.Process.Topology.Q = 1
calcite.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
calcite.ComputationalGrid.Lower.X = 0.0
calcite.ComputationalGrid.Lower.Y = 0.0
calcite.ComputationalGrid.Lower.Z = 0.0

calcite.ComputationalGrid.DX = 1.0
calcite.ComputationalGrid.DY = 1.0
calcite.ComputationalGrid.DZ = 1.0

calcite.ComputationalGrid.NX = 100
calcite.ComputationalGrid.NY = 1
calcite.ComputationalGrid.NZ = 1

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
calcite.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
calcite.GeomInput.domain_input.InputType = "Box"
calcite.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
calcite.Geom.domain.Lower.X = 0.0
calcite.Geom.domain.Lower.Y = 0.0
calcite.Geom.domain.Lower.Z = 0.0

calcite.Geom.domain.Upper.X = 100.0
calcite.Geom.domain.Upper.Y = 1.0
calcite.Geom.domain.Upper.Z = 1.0

calcite.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Background Geometry Input
# -----------------------------------------------------------------------------
calcite.GeomInput.background_input.InputType = "Box"
calcite.GeomInput.background_input.GeomName = "background"

# -----------------------------------------------------------------------------
# Background Geometry
# -----------------------------------------------------------------------------
calcite.Geom.background.Lower.X = -99999999.0
calcite.Geom.background.Lower.Y = -99999999.0
calcite.Geom.background.Lower.Z = -99999999.0

calcite.Geom.background.Upper.X = 99999999.0
calcite.Geom.background.Upper.Y = 99999999.0
calcite.Geom.background.Upper.Z = 99999999.0

# -----------------------------------------------------------------------------
# Source_Region Geometry Input
# -----------------------------------------------------------------------------
calcite.GeomInput.source_region_input.InputType = "Box"
calcite.GeomInput.source_region_input.GeomName = "source_region"

# -----------------------------------------------------------------------------
# Source_Region Geometry
# -----------------------------------------------------------------------------
calcite.Geom.source_region.Lower.X = 0.0
calcite.Geom.source_region.Lower.Y = 0.0
calcite.Geom.source_region.Lower.Z = 0.0

calcite.Geom.source_region.Upper.X = 100.0
calcite.Geom.source_region.Upper.Y = 1.0
calcite.Geom.source_region.Upper.Z = 1.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
calcite.Geom.Perm.Names = "background"

calcite.Geom.background.Perm.Type = "Constant"
calcite.Geom.background.Perm.Value = 0.25

calcite.Perm.TensorType = "TensorByGeom"

calcite.Geom.Perm.TensorByGeom.Names = "background"

calcite.Geom.background.Perm.TensorValX = 1.0
calcite.Geom.background.Perm.TensorValY = 1.0
calcite.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Concen_Region Geometry Input
# -----------------------------------------------------------------------------
calcite.GeomInput.concen_region_input.InputType = "Box"
calcite.GeomInput.concen_region_input.GeomName = "concen_region"

# -----------------------------------------------------------------------------
# Concen_Region Geometry
# -----------------------------------------------------------------------------
calcite.Geom.concen_region.Lower.X = 0.0
calcite.Geom.concen_region.Lower.Y = 0.0
calcite.Geom.concen_region.Lower.Z = 0.0

calcite.Geom.concen_region.Upper.X = 100.0
calcite.Geom.concen_region.Upper.Y = 1.0
calcite.Geom.concen_region.Upper.Z = 1.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
calcite.Phase.Names = "water"

calcite.Phase.water.Density.Type = "Constant"
calcite.Phase.water.Density.Value = 1.0

calcite.Phase.water.Viscosity.Type = "Constant"
calcite.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
calcite.Contaminants.Names = "tce dummy dummy2"
calcite.Contaminants.tce.Degradation.Value = 0.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
calcite.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
calcite.TimingInfo.BaseUnit = 1.0
calcite.TimingInfo.StartCount = 0
calcite.TimingInfo.StartTime = 0.0
calcite.TimingInfo.StopTime = 50.0
calcite.TimingInfo.DumpInterval = 10.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
calcite.Geom.Porosity.GeomNames = "background"

calcite.Geom.background.Porosity.Type = "Constant"
calcite.Geom.background.Porosity.Value = 0.25

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
calcite.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
calcite.Phase.water.Mobility.Type = "Constant"
calcite.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
calcite.Geom.Retardation.GeomNames = "background"
calcite.Geom.background.tce.Retardation.Type = "Linear"
calcite.Geom.background.tce.Retardation.Rate = 0.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
calcite.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
calcite.Cycle.Names = "constant"
calcite.Cycle.constant.Names = "alltime"
calcite.Cycle.constant.alltime.Length = 20
calcite.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
calcite.BCPressure.PatchNames = "left right front back bottom top"

calcite.Patch.left.BCPressure.Type = "DirEquilRefPatch"
calcite.Patch.left.BCPressure.Cycle = "constant"
calcite.Patch.left.BCPressure.RefGeom = "domain"
calcite.Patch.left.BCPressure.RefPatch = "bottom"
calcite.Patch.left.BCPressure.alltime.Value = 200.0

calcite.Patch.right.BCPressure.Type = "DirEquilRefPatch"
calcite.Patch.right.BCPressure.Cycle = "constant"
calcite.Patch.right.BCPressure.RefGeom = "domain"
calcite.Patch.right.BCPressure.RefPatch = "bottom"
calcite.Patch.right.BCPressure.alltime.Value = 100.0

calcite.Patch.top.BCPressure.Type = "FluxConst"
calcite.Patch.top.BCPressure.Cycle = "constant"
calcite.Patch.top.BCPressure.alltime.Value = 0.0

calcite.Patch.bottom.BCPressure.Type = "FluxConst"
calcite.Patch.bottom.BCPressure.Cycle = "constant"
calcite.Patch.bottom.BCPressure.alltime.Value = 0.0

calcite.Patch.back.BCPressure.Type = "FluxConst"
calcite.Patch.back.BCPressure.Cycle = "constant"
calcite.Patch.back.BCPressure.alltime.Value = 0.0

calcite.Patch.front.BCPressure.Type = "FluxConst"
calcite.Patch.front.BCPressure.Cycle = "constant"
calcite.Patch.front.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
calcite.PhaseSources.water.Type = "Constant"
calcite.PhaseSources.water.GeomNames = "background"
calcite.PhaseSources.water.Geom.background.Value = 0.0

calcite.PhaseConcen.water.tce.Type = "Constant"
calcite.PhaseConcen.water.tce.GeomNames = "concen_region"
calcite.PhaseConcen.water.tce.Geom.concen_region.Value = 0.1

# The TCL deck also carries bare PhaseSources.* and TempSources.* keys; both are
# vestigial. The compiled phase-source module (problem_phase_source.c) reads
# only the per-phase form above, and TempSources has no reader in the build, so
# they are dropped here without changing the result.

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
calcite.SpecificStorage.Type = "Constant"
calcite.SpecificStorage.GeomNames = "background"
calcite.Geom.background.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Heat Capacity
# -----------------------------------------------------------------------------
calcite.Phase.water.HeatCapacity.Type = "Constant"
calcite.Phase.water.HeatCapacity.GeomNames = "background"
calcite.Phase.water.Geom.background.HeatCapacity.Value = 4000.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
calcite.TopoSlopesX.Type = "Constant"
calcite.TopoSlopesX.GeomNames = "domain"
calcite.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
calcite.TopoSlopesY.Type = "Constant"
calcite.TopoSlopesY.GeomNames = "domain"
calcite.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
calcite.Mannings.Type = "Constant"
calcite.Mannings.GeomNames = "domain"
calcite.Mannings.Geom.domain.Value = 2.3e-7

# ---------------------------------------------------------
# Alquimia chemistry input
# ---------------------------------------------------------
calcite.Solver.Chemistry = "Alquimia"
calcite.Chemistry.Engine = "CrunchFlow"
calcite.Chemistry.InputFile = "1d-calcite-crunch.in"

# order of geomnames matters: like every other PF geometry, geominputs listed
# later overwrite earlier ones where they overlap
calcite.GeochemCondition.Type = "Constant"
calcite.GeochemCondition.GeomNames = "concen_region"
calcite.GeochemCondition.Names = "initial"
calcite.GeochemCondition.Geom.concen_region.Value = "initial"

calcite.BCConcentration.GeochemCondition.Names = "west"
calcite.BCConcentration.PatchNames = "left"
calcite.Patch.left.BCConcentration.Type = "Constant"
calcite.Patch.left.BCConcentration.Value = "west"

calcite.Chemistry.ParFlowTimeUnits = "years"

calcite.Chemistry.PrintPrimaryMobile = True
calcite.Chemistry.PrintMineralVolfx = True
calcite.Chemistry.PrintMineralSurfArea = True
calcite.Chemistry.PrintMineralRate = True
calcite.Chemistry.PrintPrimaryFreeIon = True
calcite.Chemistry.PrintSecondaryFreeIon = True
calcite.Chemistry.PrintpH = True

# -----------------------------------------------------------------------------
# Solver settings (the IMPES MaxIter default changed; restore the original)
# -----------------------------------------------------------------------------
calcite.Solver.MaxIter = 50000
calcite.Solver.CFL = 0.6
calcite.Solver.AdvectOrder = 2
calcite.Solver.AdvectEnforceMinMax = True
calcite.Solver.RelTol = 1.0e-35
calcite.Solver.AbsTol = 1.0e-50
calcite.Solver.Nonlinear.ResidualTol = 1.0e-15
calcite.Solver.PrintVelocities = True

# -----------------------------------------------------------------------------
# Run and verify against the shared Molins et al. 2025 reference outputs
# -----------------------------------------------------------------------------
chem_dir = get_absolute_path("../../chemistry")
correct_output_dir_name = chem_dir + "/correct_output"
new_output_dir_name = get_absolute_path("test_output/calcite")
mkdir(new_output_dir_name)

# The engine reads the input deck and its database relative to the run cwd, so
# stage both into the working directory before running.
cp(chem_dir + "/1d-calcite-crunch.in", new_output_dir_name)
cp(chem_dir + "/calcite.dbs", new_output_dir_name)

calcite.run(working_directory=new_output_dir_name)

passed = True
test_files = [
    "pH.00005",
    "PrimaryMobile.02.Ca++.00005",
    "MineralVolfx.00.Calcite.00005",
]
for test_file in test_files:
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
