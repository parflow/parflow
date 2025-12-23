# ---------------------------------------------------------
# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "richards_ptest"
rbp = Run(run_name, __file__)

# ---------------------------------------------------------

rbp.FileVersion = 4

rbp.Process.Topology.P = 1
rbp.Process.Topology.Q = 1
rbp.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

rbp.ComputationalGrid.Lower.X = 0.0
rbp.ComputationalGrid.Lower.Y = 0.0
rbp.ComputationalGrid.Lower.Z = 0.0

rbp.ComputationalGrid.DX = 1.0
rbp.ComputationalGrid.DY = 1.0
rbp.ComputationalGrid.DZ = 1.0

rbp.ComputationalGrid.NX = 20
rbp.ComputationalGrid.NY = 50
rbp.ComputationalGrid.NZ = 20

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

rbp.GeomInput.Names = "domain_input"

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------

rbp.GeomInput.domain_input.InputType = "Box"
rbp.GeomInput.domain_input.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

rbp.Geom.domain.Lower.X = 0.0
rbp.Geom.domain.Lower.Y = 0.0
rbp.Geom.domain.Lower.Z = 0.0

rbp.Geom.domain.Upper.X = 20.0
rbp.Geom.domain.Upper.Y = 20.0
rbp.Geom.domain.Upper.Z = 20.0

rbp.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

rbp.Geom.Perm.Names = "domain"
rbp.Geom.domain.Perm.Type = "Constant"
rbp.Geom.domain.Perm.Value = 1.0

rbp.Perm.TensorType = "TensorByGeom"

rbp.Geom.Perm.TensorByGeom.Names = "domain"

rbp.Geom.domain.Perm.TensorValX = 1.0
rbp.Geom.domain.Perm.TensorValY = 1.0
rbp.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

rbp.SpecificStorage.Type = "Constant"
rbp.SpecificStorage.GeomNames = "domain"
rbp.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

rbp.Phase.Names = "water"

rbp.Phase.water.Density.Type = "Constant"
rbp.Phase.water.Density.Value = 1.0

rbp.Phase.water.Viscosity.Type = "Constant"
rbp.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

rbp.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

rbp.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

rbp.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

rbp.TimingInfo.BaseUnit = 10.0
rbp.TimingInfo.StartCount = 0
rbp.TimingInfo.StartTime = 0.0
rbp.TimingInfo.StopTime = 100.0
rbp.TimingInfo.DumpInterval = 10.0
rbp.TimeStep.Type = "Constant"
rbp.TimeStep.Value = 10.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

rbp.Geom.Porosity.GeomNames = "domain"
rbp.Geom.domain.Porosity.Type = "Constant"
rbp.Geom.domain.Porosity.Value = 0.25

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

rbp.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

rbp.Phase.RelPerm.Type = "VanGenuchten"
rbp.Phase.RelPerm.GeomNames = "domain"
rbp.Geom.domain.RelPerm.Alpha = 2.0
rbp.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

rbp.Phase.Saturation.Type = "VanGenuchten"
rbp.Phase.Saturation.GeomNames = "domain"
rbp.Geom.domain.Saturation.Alpha = 2.0
rbp.Geom.domain.Saturation.N = 2.0
rbp.Geom.domain.Saturation.SRes = 0.1
rbp.Geom.domain.Saturation.SSat = 1.0

# ---------------------------------------------------------
# Flow Barrier in X between cells 10 and 11 in all Z
# ---------------------------------------------------------

rbp.Solver.Nonlinear.FlowBarrierX = False

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

rbp.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

rbp.Cycle.Names = "constant"
rbp.Cycle.constant.Names = "alltime"
rbp.Cycle.constant.alltime.Length = 1
rbp.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

rbp.BCPressure.PatchNames = "left right front back bottom top"

rbp.Patch.left.BCPressure.Type = "DirEquilRefPatch"
rbp.Patch.left.BCPressure.Cycle = "constant"
rbp.Patch.left.BCPressure.RefGeom = "domain"
rbp.Patch.left.BCPressure.RefPatch = "bottom"
rbp.Patch.left.BCPressure.alltime.Value = 11.0

rbp.Patch.right.BCPressure.Type = "DirEquilRefPatch"
rbp.Patch.right.BCPressure.Cycle = "constant"
rbp.Patch.right.BCPressure.RefGeom = "domain"
rbp.Patch.right.BCPressure.RefPatch = "bottom"
rbp.Patch.right.BCPressure.alltime.Value = 15.0

rbp.Patch.front.BCPressure.Type = "FluxConst"
rbp.Patch.front.BCPressure.Cycle = "constant"
rbp.Patch.front.BCPressure.alltime.Value = 0.0

rbp.Patch.back.BCPressure.Type = "FluxConst"
rbp.Patch.back.BCPressure.Cycle = "constant"
rbp.Patch.back.BCPressure.alltime.Value = 0.0

rbp.Patch.bottom.BCPressure.Type = "FluxConst"
rbp.Patch.bottom.BCPressure.Cycle = "constant"
rbp.Patch.bottom.BCPressure.alltime.Value = 0.0

rbp.Patch.top.BCPressure.Type = "FluxConst"
rbp.Patch.top.BCPressure.Cycle = "constant"
rbp.Patch.top.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

rbp.TopoSlopesX.Type = "Constant"
rbp.TopoSlopesX.GeomNames = "domain"
rbp.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

rbp.TopoSlopesY.Type = "Constant"
rbp.TopoSlopesY.GeomNames = "domain"
rbp.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

rbp.Mannings.Type = "Constant"
rbp.Mannings.GeomNames = "domain"
rbp.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

rbp.ICPressure.Type = "HydroStaticPatch"
rbp.ICPressure.GeomNames = "domain"
rbp.Geom.domain.ICPressure.Value = 13.0
rbp.Geom.domain.ICPressure.RefGeom = "domain"
rbp.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

rbp.PhaseSources.water.Type = "Constant"
rbp.PhaseSources.water.GeomNames = "domain"
rbp.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

rbp.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

rbp.Solver = "Richards"
rbp.Solver.MaxIter = 50000

rbp.Solver.Nonlinear.MaxIter = 100
rbp.Solver.Nonlinear.ResidualTol = 1e-6
rbp.Solver.Nonlinear.EtaChoice = "EtaConstant"
rbp.Solver.Nonlinear.EtaValue = 1e-2
rbp.Solver.Nonlinear.UseJacobian = True

rbp.Solver.Nonlinear.DerivativeEpsilon = 1e-12

rbp.Solver.Linear.KrylovDimension = 100

rbp.Solver.Linear.Preconditioner = "PFMG"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

correct_output_dir_name = get_absolute_path("../correct_output")
new_output_dir_name = get_absolute_path("test_output/richards_ptest")
mkdir(new_output_dir_name)

rbp.run(working_directory=new_output_dir_name)
passed = True
for i in range(11):
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
