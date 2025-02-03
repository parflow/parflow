# ---------------------------------------------------------
# Runs a simple sand draining problem, rectangular domain
# with variable dz and a heterogeneous subsurface with different K the top and bottom layers
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "var.dz.1d"
vardz = Run(run_name, __file__)

# ---------------------------------------------------------

vardz.FileVersion = 4

vardz.Process.Topology.P = 1
vardz.Process.Topology.Q = 1
vardz.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

vardz.ComputationalGrid.Lower.X = 0.0
vardz.ComputationalGrid.Lower.Y = 0.0
vardz.ComputationalGrid.Lower.Z = 0.0

vardz.ComputationalGrid.DX = 1.0
vardz.ComputationalGrid.DY = 1.0
vardz.ComputationalGrid.DZ = 0.1

vardz.ComputationalGrid.NX = 1
vardz.ComputationalGrid.NY = 1
vardz.ComputationalGrid.NZ = 14

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

vardz.GeomInput.Names = "domain_input het_input1 het_input2"

# ---------------------------------------------------------
# Geometry Input
# ---------------------------------------------------------

vardz.GeomInput.domain_input.InputType = "Box"
vardz.GeomInput.domain_input.GeomName = "domain"

vardz.GeomInput.het_input1.InputType = "Box"
vardz.GeomInput.het_input1.GeomName = "het1"

vardz.GeomInput.het_input2.InputType = "Box"
vardz.GeomInput.het_input2.GeomName = "het2"

# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

vardz.Geom.domain.Lower.X = 0.0
vardz.Geom.domain.Lower.Y = 0.0
vardz.Geom.domain.Lower.Z = 0.0

vardz.Geom.domain.Upper.X = 1.0
vardz.Geom.domain.Upper.Y = 1.0
vardz.Geom.domain.Upper.Z = 1.4

vardz.Geom.domain.Patches = "left right front back bottom top"

vardz.Geom.het1.Lower.X = 0.0
vardz.Geom.het1.Lower.Y = 0.0
vardz.Geom.het1.Lower.Z = 1.3

vardz.Geom.het1.Upper.X = 1.0
vardz.Geom.het1.Upper.Y = 1.0
vardz.Geom.het1.Upper.Z = 1.4

vardz.Geom.het2.Lower.X = 0.0
vardz.Geom.het2.Lower.Y = 0.0
vardz.Geom.het2.Lower.Z = 0.0

vardz.Geom.het2.Upper.X = 1.0
vardz.Geom.het2.Upper.Y = 1.0
vardz.Geom.het2.Upper.Z = 0.1

# --------------------------------------------
# variable dz assignments
# ------------------------------------------

vardz.Solver.Nonlinear.VariableDz = True
vardz.dzScale.GeomNames = "domain"
vardz.dzScale.Type = "nzList"
vardz.dzScale.nzListNumber = 14
vardz.Cell._0.dzScale.Value = 1.2
vardz.Cell._1.dzScale.Value = 1.0
vardz.Cell._2.dzScale.Value = 1.0
vardz.Cell._3.dzScale.Value = 1.0
vardz.Cell._4.dzScale.Value = 1.0
vardz.Cell._5.dzScale.Value = 1.0
vardz.Cell._6.dzScale.Value = 1.0
vardz.Cell._7.dzScale.Value = 1.0
vardz.Cell._8.dzScale.Value = 1.0
vardz.Cell._9.dzScale.Value = 1.0
vardz.Cell._10.dzScale.Value = 0.15
vardz.Cell._11.dzScale.Value = 0.1
vardz.Cell._12.dzScale.Value = 0.1
vardz.Cell._13.dzScale.Value = 0.05

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

vardz.Geom.Perm.Names = "domain het1 het2"

vardz.Geom.domain.Perm.Type = "Constant"
vardz.Geom.domain.Perm.Value = 5.129

vardz.Geom.het1.Perm.Type = "Constant"
vardz.Geom.het1.Perm.Value = 0.0001

vardz.Geom.het2.Perm.Type = "Constant"
vardz.Geom.het2.Perm.Value = 0.001

vardz.Perm.TensorType = "TensorByGeom"

vardz.Geom.Perm.TensorByGeom.Names = "domain"

vardz.Geom.domain.Perm.TensorValX = 1.0
vardz.Geom.domain.Perm.TensorValY = 1.0
vardz.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

vardz.SpecificStorage.Type = "Constant"
vardz.SpecificStorage.GeomNames = "domain"
vardz.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

vardz.Phase.Names = "water"

vardz.Phase.water.Density.Type = "Constant"
vardz.Phase.water.Density.Value = 1.0

vardz.Phase.water.Viscosity.Type = "Constant"
vardz.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

vardz.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

vardz.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

vardz.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

vardz.TimingInfo.BaseUnit = 1.0
vardz.TimingInfo.StartCount = 0
vardz.TimingInfo.StartTime = 0.0
vardz.TimingInfo.StopTime = 50.0
vardz.TimingInfo.DumpInterval = -100
vardz.TimeStep.Type = "Constant"
vardz.TimeStep.Value = 0.01
vardz.TimeStep.Value = 0.01

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

vardz.Geom.Porosity.GeomNames = "domain"
vardz.Geom.domain.Porosity.Type = "Constant"
vardz.Geom.domain.Porosity.Value = 0.4150

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

vardz.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

vardz.Phase.RelPerm.Type = "VanGenuchten"
vardz.Phase.RelPerm.GeomNames = "domain"
vardz.Geom.domain.RelPerm.Alpha = 2.7
vardz.Geom.domain.RelPerm.N = 3.8

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

vardz.Phase.Saturation.Type = "VanGenuchten"
vardz.Phase.Saturation.GeomNames = "domain"
vardz.Geom.domain.Saturation.Alpha = 2.7
vardz.Geom.domain.Saturation.N = 3.8
vardz.Geom.domain.Saturation.SRes = 0.106
vardz.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

vardz.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

vardz.Cycle.Names = "constant"
vardz.Cycle.constant.Names = "alltime"
vardz.Cycle.constant.alltime.Length = 1
vardz.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

vardz.BCPressure.PatchNames = "left right front back bottom top"

vardz.Patch.left.BCPressure.Type = "FluxConst"
vardz.Patch.left.BCPressure.Cycle = "constant"
vardz.Patch.left.BCPressure.RefGeom = "domain"
vardz.Patch.left.BCPressure.RefPatch = "bottom"
vardz.Patch.left.BCPressure.alltime.Value = 0.0

vardz.Patch.right.BCPressure.Type = "FluxConst"
vardz.Patch.right.BCPressure.Cycle = "constant"
vardz.Patch.right.BCPressure.RefGeom = "domain"
vardz.Patch.right.BCPressure.RefPatch = "bottom"
vardz.Patch.right.BCPressure.alltime.Value = 0.0

vardz.Patch.front.BCPressure.Type = "FluxConst"
vardz.Patch.front.BCPressure.Cycle = "constant"
vardz.Patch.front.BCPressure.alltime.Value = 0.0

vardz.Patch.back.BCPressure.Type = "FluxConst"
vardz.Patch.back.BCPressure.Cycle = "constant"
vardz.Patch.back.BCPressure.alltime.Value = 0.0

vardz.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
vardz.Patch.bottom.BCPressure.Type = "FluxConst"
vardz.Patch.bottom.BCPressure.Cycle = "constant"
vardz.Patch.bottom.BCPressure.RefGeom = "domain"
vardz.Patch.bottom.BCPressure.RefPatch = "bottom"
vardz.Patch.bottom.BCPressure.alltime.Value = 0.0

vardz.Patch.top.BCPressure.Type = "DirEquilRefPatch"
vardz.Patch.top.BCPressure.Type = "FluxConst"
vardz.Patch.top.BCPressure.Cycle = "constant"
vardz.Patch.top.BCPressure.RefGeom = "domain"
vardz.Patch.top.BCPressure.RefPatch = "bottom"
vardz.Patch.top.BCPressure.alltime.Value = -0.0001

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

vardz.TopoSlopesX.Type = "Constant"
vardz.TopoSlopesX.GeomNames = "domain"
vardz.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

vardz.TopoSlopesY.Type = "Constant"
vardz.TopoSlopesY.GeomNames = "domain"
vardz.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

vardz.Mannings.Type = "Constant"
vardz.Mannings.GeomNames = "domain"
vardz.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

vardz.ICPressure.Type = "Constant"
vardz.ICPressure.GeomNames = "domain"
vardz.Geom.domain.ICPressure.Value = -10.0
vardz.Geom.domain.ICPressure.RefGeom = "domain"
vardz.Geom.domain.ICPressure.RefPatch = "top"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

vardz.PhaseSources.water.Type = "Constant"
vardz.PhaseSources.water.GeomNames = "domain"
vardz.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

vardz.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

vardz.Solver = "Richards"
vardz.Solver.MaxIter = 2500

vardz.Solver.Nonlinear.MaxIter = 200
vardz.Solver.Nonlinear.ResidualTol = 1e-9
vardz.Solver.Nonlinear.EtaChoice = "Walker1"
vardz.Solver.Nonlinear.EtaValue = 1e-5
vardz.Solver.Nonlinear.UseJacobian = True
vardz.Solver.Nonlinear.DerivativeEpsilon = 1e-10

vardz.Solver.Linear.KrylovDimension = 10

vardz.Solver.Linear.Preconditioner = "MGSemi"
vardz.Solver.Linear.Preconditioner = "PFMG"
vardz.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
vardz.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

# -----------------------------------------------------------------------------
# Run and do tests
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/var_d5z_1D")
correct_output_dir_name = get_absolute_path("../correct_output")
mkdir(new_output_dir_name)
vardz.run(working_directory=new_output_dir_name)

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

for i in range(0, 30, 5):
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
