# ---------------------------------------------------------
#  Testing overland flow
# Running a parking lot sloping slab pointed in 8 directions
# With a suite of overlandflow BC options
# ---------------------------------------------------------

from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
import pandas as pd
import parflow as pf
from parflow.tools.compare import pf_test_file_with_abs
import numpy as np
import sys

run_name = "reservoir_mpi_test"
sloping_slab = Run(run_name, __file__)

# ---------------------------------------------------------

sloping_slab.FileVersion = 4

sloping_slab.Process.Topology.P = 1
sloping_slab.Process.Topology.Q = 1
sloping_slab.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

sloping_slab.ComputationalGrid.Lower.X = 0.0
sloping_slab.ComputationalGrid.Lower.Y = 0.0
sloping_slab.ComputationalGrid.Lower.Z = 0.0

sloping_slab.ComputationalGrid.NX = 5
sloping_slab.ComputationalGrid.NY = 5
sloping_slab.ComputationalGrid.NZ = 1

sloping_slab.ComputationalGrid.DX = 10.0
sloping_slab.ComputationalGrid.DY = 10.0
sloping_slab.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

sloping_slab.GeomInput.Names = "domaininput"
sloping_slab.GeomInput.domaininput.GeomName = "domain"
sloping_slab.GeomInput.domaininput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

sloping_slab.Geom.domain.Lower.X = 0.0
sloping_slab.Geom.domain.Lower.Y = 0.0
sloping_slab.Geom.domain.Lower.Z = 0.0

sloping_slab.Geom.domain.Upper.X = 50.0
sloping_slab.Geom.domain.Upper.Y = 50.0
sloping_slab.Geom.domain.Upper.Z = 0.05
sloping_slab.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

sloping_slab.Geom.Perm.Names = "domain"
sloping_slab.Geom.domain.Perm.Type = "Constant"
sloping_slab.Geom.domain.Perm.Value = 0.0000001

sloping_slab.Perm.TensorType = "TensorByGeom"

sloping_slab.Geom.Perm.TensorByGeom.Names = "domain"

sloping_slab.Geom.domain.Perm.TensorValX = 1.0
sloping_slab.Geom.domain.Perm.TensorValY = 1.0
sloping_slab.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

sloping_slab.SpecificStorage.Type = "Constant"
sloping_slab.SpecificStorage.GeomNames = "domain"
sloping_slab.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

sloping_slab.Phase.Names = "water"

sloping_slab.Phase.water.Density.Type = "Constant"
sloping_slab.Phase.water.Density.Value = 1.0

sloping_slab.Phase.water.Viscosity.Type = "Constant"
sloping_slab.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

sloping_slab.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

sloping_slab.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

sloping_slab.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

sloping_slab.TimingInfo.BaseUnit = 0.05
sloping_slab.TimingInfo.StartCount = 0
sloping_slab.TimingInfo.StartTime = 0.0
sloping_slab.TimingInfo.StopTime = 1.0
sloping_slab.TimingInfo.DumpInterval = -2
sloping_slab.TimeStep.Type = "Constant"
sloping_slab.TimeStep.Value = 0.05

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

sloping_slab.Geom.Porosity.GeomNames = "domain"
sloping_slab.Geom.domain.Porosity.Type = "Constant"
sloping_slab.Geom.domain.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

sloping_slab.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

sloping_slab.Phase.RelPerm.Type = "VanGenuchten"
sloping_slab.Phase.RelPerm.GeomNames = "domain"

sloping_slab.Geom.domain.RelPerm.Alpha = 6.0
sloping_slab.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

sloping_slab.Phase.Saturation.Type = "VanGenuchten"
sloping_slab.Phase.Saturation.GeomNames = "domain"

sloping_slab.Geom.domain.Saturation.Alpha = 6.0
sloping_slab.Geom.domain.Saturation.N = 2.0
sloping_slab.Geom.domain.Saturation.SRes = 0.2
sloping_slab.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

sloping_slab.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

sloping_slab.Cycle.Names = "constant rainrec"
sloping_slab.Cycle.constant.Names = "alltime"
sloping_slab.Cycle.constant.alltime.Length = 1
sloping_slab.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

sloping_slab.Cycle.rainrec.Names = "rain rec"
sloping_slab.Cycle.rainrec.rain.Length = 2
sloping_slab.Cycle.rainrec.rec.Length = 300
sloping_slab.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

sloping_slab.BCPressure.PatchNames = sloping_slab.Geom.domain.Patches

sloping_slab.Patch.x_lower.BCPressure.Type = "FluxConst"
sloping_slab.Patch.x_lower.BCPressure.Cycle = "constant"
sloping_slab.Patch.x_lower.BCPressure.alltime.Value = 0.0

sloping_slab.Patch.y_lower.BCPressure.Type = "FluxConst"
sloping_slab.Patch.y_lower.BCPressure.Cycle = "constant"
sloping_slab.Patch.y_lower.BCPressure.alltime.Value = 0.0

sloping_slab.Patch.z_lower.BCPressure.Type = "FluxConst"
sloping_slab.Patch.z_lower.BCPressure.Cycle = "constant"
sloping_slab.Patch.z_lower.BCPressure.alltime.Value = 0.0

sloping_slab.Patch.x_upper.BCPressure.Type = "FluxConst"
sloping_slab.Patch.x_upper.BCPressure.Cycle = "constant"
sloping_slab.Patch.x_upper.BCPressure.alltime.Value = 0.0

sloping_slab.Patch.y_upper.BCPressure.Type = "FluxConst"
sloping_slab.Patch.y_upper.BCPressure.Cycle = "constant"
sloping_slab.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
sloping_slab.Patch.z_upper.BCPressure.Type = "OverlandFlow"
sloping_slab.Patch.z_upper.BCPressure.Cycle = "rainrec"
sloping_slab.Patch.z_upper.BCPressure.rain.Value = -0.01
sloping_slab.Patch.z_upper.BCPressure.rec.Value = 0.0000

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

sloping_slab.Mannings.Type = "Constant"
sloping_slab.Mannings.GeomNames = "domain"
sloping_slab.Mannings.Geom.domain.Value = 3.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

sloping_slab.PhaseSources.water.Type = "Constant"
sloping_slab.PhaseSources.water.GeomNames = "domain"
sloping_slab.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

sloping_slab.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

sloping_slab.Solver = "Richards"
sloping_slab.Solver.MaxIter = 2500

sloping_slab.Solver.Nonlinear.MaxIter = 50
sloping_slab.Solver.Nonlinear.ResidualTol = 1e-9
sloping_slab.Solver.Nonlinear.EtaChoice = "EtaConstant"
sloping_slab.Solver.Nonlinear.EtaValue = 0.01
sloping_slab.Solver.Nonlinear.UseJacobian = False

sloping_slab.Solver.Nonlinear.DerivativeEpsilon = 1e-15
sloping_slab.Solver.Nonlinear.StepTol = 1e-20
sloping_slab.Solver.Nonlinear.Globalization = "LineSearch"
sloping_slab.Solver.Linear.KrylovDimension = 20
sloping_slab.Solver.Linear.MaxRestart = 2

sloping_slab.Solver.Linear.Preconditioner = "PFMG"
sloping_slab.Solver.PrintSubsurf = False
sloping_slab.Solver.Drop = 1e-20
sloping_slab.Solver.AbsTol = 1e-10

sloping_slab.Solver.OverlandKinematic.Epsilon = 1e-5

sloping_slab.Solver.WriteSiloSubsurfData = False
sloping_slab.Solver.WriteSiloPressure = False
sloping_slab.Solver.WriteSiloSlopes = False

sloping_slab.Solver.WriteSiloSaturation = False
sloping_slab.Solver.WriteSiloConcentration = False

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
sloping_slab.ICPressure.Type = "HydroStaticPatch"
sloping_slab.ICPressure.GeomNames = "domain"
sloping_slab.Geom.domain.ICPressure.Value = -3.0

sloping_slab.Geom.domain.ICPressure.RefGeom = "domain"
sloping_slab.Geom.domain.ICPressure.RefPatch = "z_upper"


#   #### Set the slopes. Will flow in -x direction
sloping_slab.TopoSlopesX.Type = "Constant"
sloping_slab.TopoSlopesX.GeomNames = "domain"
sloping_slab.TopoSlopesX.Geom.domain.Value = -0.01

sloping_slab.TopoSlopesY.Type = "Constant"
sloping_slab.TopoSlopesY.GeomNames = "domain"
sloping_slab.TopoSlopesY.Geom.domain.Value = 0.0

# turn on analytical jacobian and re-test
sloping_slab.Solver.Nonlinear.UseJacobian = True
sloping_slab.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"


sloping_slab.Reservoirs.Names = "reservoir"
sloping_slab.Reservoirs.Overland_Flow_Solver = "OverlandFlow"

sloping_slab.Reservoirs.reservoir.Intake_X = 25
sloping_slab.Reservoirs.reservoir.Intake_Y = 25
sloping_slab.Reservoirs.reservoir.Secondary_Intake_X = 25
sloping_slab.Reservoirs.reservoir.Secondary_Intake_Y = 35
sloping_slab.Reservoirs.reservoir.Release_X = 35
sloping_slab.Reservoirs.reservoir.Release_Y = 25
sloping_slab.Reservoirs.reservoir.Has_Secondary_Intake_Cell = 1

sloping_slab.Reservoirs.reservoir.Max_Storage = 7000000
sloping_slab.Reservoirs.reservoir.Storage = 5500000.0
sloping_slab.Reservoirs.reservoir.Min_Release_Storage = 4000000
sloping_slab.Reservoirs.reservoir.Release_Rate = 1

# run the base case on one processor. We then test that we get the same answer when using different
# processor configurations
base_case_dir = get_absolute_path("test_output/reservoir_mpi_test_1_processor")
mkdir(base_case_dir)
sloping_slab.run(working_directory=base_case_dir)

# test that when a reservoir spans two ranks we get the correct answer
sloping_slab.Process.Topology.P = 5
sloping_slab.Process.Topology.Q = 1
sloping_slab.Process.Topology.R = 1

passed = True

test1_dir = get_absolute_path("test_output/reservoir_mpi_test_5_processors")
mkdir(test1_dir)
sloping_slab.run(working_directory=test1_dir)


i = 10
timestep = str(i).rjust(5, "0")

sig_digits = 10
abs_value = 1e-18

test_files = ["press"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
    if not pf_test_file_with_abs(
        test1_dir + filename,
        base_case_dir + filename,
        f"Max difference in {filename}",
        abs_value,
        sig_digits,
    ):
        passed = False


# test that when a reservoir spans three ranks we get the correct answer
sloping_slab.Process.Topology.P = 5
sloping_slab.Process.Topology.Q = 5
sloping_slab.Process.Topology.R = 1

test2_dir = get_absolute_path("test_output/reservoir_mpi_test_25_processors")
mkdir(test2_dir)
sloping_slab.run(working_directory=test2_dir)

test_files = ["press"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
    if not pf_test_file_with_abs(
        test2_dir + filename,
        base_case_dir + filename,
        f"Max difference in {filename}",
        abs_value,
        sig_digits,
    ):
        passed = False


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
