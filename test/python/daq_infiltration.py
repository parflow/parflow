# ----------------------------------------------------------------
#  This runs the infiltration experiment using the DeepAquiferBC.
# ----------------------------------------------------------------

import sys, argparse
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs


run_name = "daq_infiltration"

daq = Run(run_name, __file__)
daq.FileVersion = 4

# ----------------------------------------------------------------

daq.Process.Topology.P = 1
daq.Process.Topology.Q = 1
daq.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

daq.ComputationalGrid.Lower.X = 0.0
daq.ComputationalGrid.Lower.Y = 0.0
daq.ComputationalGrid.Lower.Z = 0.0

daq.ComputationalGrid.DX = 10.0
daq.ComputationalGrid.DY = 10.0
daq.ComputationalGrid.DZ = 0.05

daq.ComputationalGrid.NX = 5
daq.ComputationalGrid.NY = 5
daq.ComputationalGrid.NZ = 200

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

daq.GeomInput.Names = "domain_input background_input"

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------

daq.GeomInput.domain_input.InputType = "Box"
daq.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

daq.Domain.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

daq.Geom.domain.Lower.X = 0.0
daq.Geom.domain.Lower.Y = 0.0
daq.Geom.domain.Lower.Z = 0.0

daq.Geom.domain.Upper.X = 50.0
daq.Geom.domain.Upper.Y = 50.0
daq.Geom.domain.Upper.Z = 10.0

daq.Geom.domain.Patches = "left right front back bottom top"

# ---------------------------------------------------------
# Background Geometry Input
# ---------------------------------------------------------

daq.GeomInput.background_input.InputType = "Box"
daq.GeomInput.background_input.GeomName = "background"

# ---------------------------------------------------------
# Background Geometry
# ---------------------------------------------------------

daq.Geom.background.Lower.X = -99999999.0
daq.Geom.background.Lower.Y = -99999999.0
daq.Geom.background.Lower.Z = -99999999.0

daq.Geom.background.Upper.X = 99999999.0
daq.Geom.background.Upper.Y = 99999999.0
daq.Geom.background.Upper.Z = 99999999.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

daq.TimingInfo.BaseUnit = 0.25
daq.TimingInfo.StartCount = 0
daq.TimingInfo.StartTime = 0.0
daq.TimingInfo.StopTime = 5.0
daq.TimingInfo.DumpInterval = -4
daq.TimeStep.Type = "Constant"
daq.TimeStep.Value = 0.25

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

daq.Cycle.Names = "constant rainfall"

daq.Cycle.constant.Names = "alltime"
daq.Cycle.constant.alltime.Length = 1
daq.Cycle.constant.Repeat = -1

daq.Cycle.rainfall.Names = "On Off"
daq.Cycle.rainfall.On.Length = 40
daq.Cycle.rainfall.Off.Length = 440
daq.Cycle.rainfall.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

daq.BCPressure.PatchNames = "left right front back bottom top"

daq.Patch.left.BCPressure.Type = "FluxConst"
daq.Patch.left.BCPressure.Cycle = "constant"
daq.Patch.left.BCPressure.alltime.Value = 0.0

daq.Patch.right.BCPressure.Type = "FluxConst"
daq.Patch.right.BCPressure.Cycle = "constant"
daq.Patch.right.BCPressure.alltime.Value = 0.0

daq.Patch.front.BCPressure.Type = "FluxConst"
daq.Patch.front.BCPressure.Cycle = "constant"
daq.Patch.front.BCPressure.alltime.Value = 0.0

daq.Patch.back.BCPressure.Type = "FluxConst"
daq.Patch.back.BCPressure.Cycle = "constant"
daq.Patch.back.BCPressure.alltime.Value = 0.0

daq.Patch.bottom.BCPressure.Type = "DeepAquifer"
daq.Patch.bottom.BCPressure.Cycle = "constant"
daq.Patch.BCPressure.DeepAquifer.SpecificYield.Type = "Constant"
daq.Patch.BCPressure.DeepAquifer.SpecificYield.Value = 0.1
daq.Patch.BCPressure.DeepAquifer.AquiferDepth.Type = "Constant"
daq.Patch.BCPressure.DeepAquifer.AquiferDepth.Value = 90.0

daq.Patch.top.BCPressure.Type = "FluxConst"
daq.Patch.top.BCPressure.Cycle = "rainfall"
daq.Patch.top.BCPressure.On.Value = -0.005
daq.Patch.top.BCPressure.Off.Value = 0.001

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

daq.ICPressure.Type = "HydroStaticPatch"
daq.ICPressure.GeomNames = "domain"
daq.Geom.domain.ICPressure.Value = -1.25
daq.Geom.domain.ICPressure.RefGeom = "domain"
daq.Geom.domain.ICPressure.RefPatch = "top"

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

daq.Gravity = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

daq.Geom.Porosity.GeomNames = "domain"
daq.Geom.domain.Porosity.Type = "Constant"
daq.Geom.domain.Porosity.Value = 0.489

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

daq.Geom.Perm.Names = "domain background"
daq.Geom.domain.Perm.Type = "Constant"
daq.Geom.domain.Perm.Value = 0.01836

daq.Perm.TensorType = "TensorByGeom"

daq.Geom.Perm.TensorByGeom.Names = "domain background"

daq.Geom.domain.Perm.TensorValX = 1.0
daq.Geom.domain.Perm.TensorValY = 1.0
daq.Geom.domain.Perm.TensorValZ = 1.0

daq.Geom.background.Perm.Type = "Constant"
daq.Geom.background.Perm.Value = 1.0

daq.Geom.background.Perm.TensorValX = 1.0
daq.Geom.background.Perm.TensorValY = 1.0
daq.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

daq.Phase.Names = "water"

daq.Phase.water.Density.Type = "Constant"
daq.Phase.water.Density.Value = 1.0

daq.Phase.water.Viscosity.Type = "Constant"
daq.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

daq.PhaseSources.water.Type = "Constant"
daq.PhaseSources.water.GeomNames = "domain"
daq.PhaseSources.water.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

daq.Phase.Saturation.Type = "VanGenuchten"
daq.Phase.Saturation.GeomNames = "domain"
daq.Geom.domain.Saturation.Alpha = 0.657658
daq.Geom.domain.Saturation.N = 2.678804
daq.Geom.domain.Saturation.SRes = 0.102249
daq.Geom.domain.Saturation.SSat = 1.0


# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

daq.Phase.RelPerm.Type = "VanGenuchten"
daq.Phase.RelPerm.GeomNames = "domain"
daq.Geom.domain.RelPerm.Alpha = 0.657658
daq.Geom.domain.RelPerm.N = 2.678804

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

daq.Mannings.Type = "Constant"
daq.Mannings.GeomNames = "domain"
daq.Mannings.Geom.domain.Value = 5.5e-5

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

daq.SpecificStorage.Type = "Constant"
daq.SpecificStorage.GeomNames = "domain"
daq.Geom.domain.SpecificStorage.Value = 1.0e-4

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

daq.TopoSlopesX.Type = "Constant"
daq.TopoSlopesX.GeomNames = "domain"
daq.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

daq.TopoSlopesY.Type = "Constant"
daq.TopoSlopesY.GeomNames = "domain"
daq.TopoSlopesY.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

daq.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

daq.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

daq.Wells.Names = ""

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

daq.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

daq.Solver = "Richards"
daq.Solver.MaxIter = 100000

daq.Solver.Nonlinear.MaxIter = 1000
daq.Solver.Nonlinear.ResidualTol = 1e-10
daq.Solver.Nonlinear.EtaChoice = "EtaConstant"
daq.Solver.Nonlinear.EtaValue = 1e-4
daq.Solver.Nonlinear.UseJacobian = True
daq.Solver.Nonlinear.DerivativeEpsilon = 1e-5
daq.Solver.Nonlinear.StepTol = 1e-5

daq.Solver.Linear.KrylovDimension = 30

daq.Solver.Linear.Preconditioner = "MGSemi"
daq.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
daq.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/daq_infiltration")
mkdir(new_output_dir_name)
correct_output_dir_name = get_absolute_path("../correct_output")
daq.run(working_directory=new_output_dir_name)


#
# Tests
#

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


sig_digits = 6
abs_value = 1e-12
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
