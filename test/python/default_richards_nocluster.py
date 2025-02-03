# -----------------------------------------------------------------------------
#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "default_richards"
drich_n = Run(run_name, __file__)

# ---------------------------------------------------------

drich_n.FileVersion = 4

drich_n.UseClustering = False

drich_n.Process.Topology.P = 1
drich_n.Process.Topology.Q = 1
drich_n.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

drich_n.ComputationalGrid.Lower.X = -10.0
drich_n.ComputationalGrid.Lower.Y = 10.0
drich_n.ComputationalGrid.Lower.Z = 1.0

drich_n.ComputationalGrid.DX = 8.8888888888888893
drich_n.ComputationalGrid.DY = 10.666666666666666
drich_n.ComputationalGrid.DZ = 1.0

drich_n.ComputationalGrid.NX = 18
drich_n.ComputationalGrid.NY = 15
drich_n.ComputationalGrid.NZ = 8

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

drich_n.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------

drich_n.GeomInput.domain_input.InputType = "Box"
drich_n.GeomInput.domain_input.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

drich_n.Geom.domain.Lower.X = -10.0
drich_n.Geom.domain.Lower.Y = 10.0
drich_n.Geom.domain.Lower.Z = 1.0

drich_n.Geom.domain.Upper.X = 150.0
drich_n.Geom.domain.Upper.Y = 170.0
drich_n.Geom.domain.Upper.Z = 9.0

drich_n.Geom.domain.Patches = "left right front back bottom top"

# ---------------------------------------------------------
# Background Geometry Input
# ---------------------------------------------------------

drich_n.GeomInput.background_input.InputType = "Box"
drich_n.GeomInput.background_input.GeomName = "background"

# ---------------------------------------------------------
# Background Geometry
# ---------------------------------------------------------

drich_n.Geom.background.Lower.X = -99999999.0
drich_n.Geom.background.Lower.Y = -99999999.0
drich_n.Geom.background.Lower.Z = -99999999.0

drich_n.Geom.background.Upper.X = 99999999.0
drich_n.Geom.background.Upper.Y = 99999999.0
drich_n.Geom.background.Upper.Z = 99999999.0

# ---------------------------------------------------------
# Source_Region Geometry Input
# ---------------------------------------------------------

drich_n.GeomInput.source_region_input.InputType = "Box"
drich_n.GeomInput.source_region_input.GeomName = "source_region"

# ---------------------------------------------------------
# Source_Region Geometry
# ---------------------------------------------------------

drich_n.Geom.source_region.Lower.X = 65.56
drich_n.Geom.source_region.Lower.Y = 79.34
drich_n.Geom.source_region.Lower.Z = 4.5

drich_n.Geom.source_region.Upper.X = 74.44
drich_n.Geom.source_region.Upper.Y = 89.99
drich_n.Geom.source_region.Upper.Z = 5.5

# ---------------------------------------------------------
# Concen_Region Geometry Input
# ---------------------------------------------------------

drich_n.GeomInput.concen_region_input.InputType = "Box"
drich_n.GeomInput.concen_region_input.GeomName = "concen_region"

# ---------------------------------------------------------
# Concen_Region Geometry
# ---------------------------------------------------------

drich_n.Geom.concen_region.Lower.X = 60.0
drich_n.Geom.concen_region.Lower.Y = 80.0
drich_n.Geom.concen_region.Lower.Z = 4.0

drich_n.Geom.concen_region.Upper.X = 80.0
drich_n.Geom.concen_region.Upper.Y = 100.0
drich_n.Geom.concen_region.Upper.Z = 6.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

drich_n.Geom.Perm.Names = "background"

drich_n.Geom.background.Perm.Type = "Constant"
drich_n.Geom.background.Perm.Value = 4.0

drich_n.Perm.TensorType = "TensorByGeom"

drich_n.Geom.Perm.TensorByGeom.Names = "background"

drich_n.Geom.background.Perm.TensorValX = 1.0
drich_n.Geom.background.Perm.TensorValY = 1.0
drich_n.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

drich_n.SpecificStorage.Type = "Constant"
drich_n.SpecificStorage.GeomNames = "domain"
drich_n.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

drich_n.Phase.Names = "water"

drich_n.Phase.water.Density.Type = "Constant"
drich_n.Phase.water.Density.Value = 1.0

drich_n.Phase.water.Viscosity.Type = "Constant"
drich_n.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

drich_n.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

drich_n.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

drich_n.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

drich_n.TimingInfo.BaseUnit = 1.0
drich_n.TimingInfo.StartCount = 0
drich_n.TimingInfo.StartTime = 0.0
drich_n.TimingInfo.StopTime = 0.010
drich_n.TimingInfo.DumpInterval = -1
drich_n.TimeStep.Type = "Constant"
drich_n.TimeStep.Value = 0.001

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

drich_n.Geom.Porosity.GeomNames = "background"

drich_n.Geom.background.Porosity.Type = "Constant"
drich_n.Geom.background.Porosity.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

drich_n.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

drich_n.Phase.RelPerm.Type = "VanGenuchten"
drich_n.Phase.RelPerm.GeomNames = "domain"
drich_n.Geom.domain.RelPerm.Alpha = 0.005
drich_n.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

drich_n.Phase.Saturation.Type = "VanGenuchten"
drich_n.Phase.Saturation.GeomNames = "domain"
drich_n.Geom.domain.Saturation.Alpha = 0.005
drich_n.Geom.domain.Saturation.N = 2.0
drich_n.Geom.domain.Saturation.SRes = 0.2
drich_n.Geom.domain.Saturation.SSat = 0.99

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

drich_n.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

drich_n.Cycle.Names = "constant"
drich_n.Cycle.constant.Names = "alltime"
drich_n.Cycle.constant.alltime.Length = 1
drich_n.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

drich_n.BCPressure.PatchNames = "left right front back bottom top"

drich_n.Patch.left.BCPressure.Type = "DirEquilRefPatch"
drich_n.Patch.left.BCPressure.Cycle = "constant"
drich_n.Patch.left.BCPressure.RefGeom = "domain"
drich_n.Patch.left.BCPressure.RefPatch = "bottom"
drich_n.Patch.left.BCPressure.alltime.Value = 5.0

drich_n.Patch.right.BCPressure.Type = "DirEquilRefPatch"
drich_n.Patch.right.BCPressure.Cycle = "constant"
drich_n.Patch.right.BCPressure.RefGeom = "domain"
drich_n.Patch.right.BCPressure.RefPatch = "bottom"
drich_n.Patch.right.BCPressure.alltime.Value = 3.0

drich_n.Patch.front.BCPressure.Type = "FluxConst"
drich_n.Patch.front.BCPressure.Cycle = "constant"
drich_n.Patch.front.BCPressure.alltime.Value = 0.0

drich_n.Patch.back.BCPressure.Type = "FluxConst"
drich_n.Patch.back.BCPressure.Cycle = "constant"
drich_n.Patch.back.BCPressure.alltime.Value = 0.0

drich_n.Patch.bottom.BCPressure.Type = "FluxConst"
drich_n.Patch.bottom.BCPressure.Cycle = "constant"
drich_n.Patch.bottom.BCPressure.alltime.Value = 0.0

drich_n.Patch.top.BCPressure.Type = "FluxConst"
drich_n.Patch.top.BCPressure.Cycle = "constant"
drich_n.Patch.top.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

drich_n.TopoSlopesX.Type = "Constant"
drich_n.TopoSlopesX.GeomNames = "domain"
drich_n.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

drich_n.TopoSlopesY.Type = "Constant"
drich_n.TopoSlopesY.GeomNames = "domain"
drich_n.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

drich_n.Mannings.Type = "Constant"
drich_n.Mannings.GeomNames = "domain"
drich_n.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

drich_n.ICPressure.Type = "HydroStaticPatch"
drich_n.ICPressure.GeomNames = "domain"
drich_n.Geom.domain.ICPressure.Value = 3.0
drich_n.Geom.domain.ICPressure.RefGeom = "domain"
drich_n.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

drich_n.PhaseSources.water.Type = "Constant"
drich_n.PhaseSources.water.GeomNames = "background"
drich_n.PhaseSources.water.Geom.background.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

drich_n.KnownSolution = "NoKnownSolution"


# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

drich_n.Solver = "Richards"
drich_n.Solver.MaxIter = 5

drich_n.Solver.Nonlinear.MaxIter = 10
drich_n.Solver.Nonlinear.ResidualTol = 1e-9
drich_n.Solver.Nonlinear.EtaChoice = "EtaConstant"
drich_n.Solver.Nonlinear.EtaValue = 1e-5
drich_n.Solver.Nonlinear.UseJacobian = True
drich_n.Solver.Nonlinear.DerivativeEpsilon = 1e-2

drich_n.Solver.Linear.KrylovDimension = 10

drich_n.Solver.Linear.Preconditioner = "PFMG"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/default_richards")
mkdir(new_output_dir_name)
correct_output_dir_name = get_absolute_path("../correct_output")
drich_n.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in {test_file}",
    ):
        passed = False

for i in range(6):
    timestep = str(i).rjust(5, "0")
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Pressure for timestep {timestep}",
    ):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in Saturation for timestep {timestep}",
    ):
        passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
