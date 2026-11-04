#
# This test is part of a series of tests for the DeepAquiferBC
# Here, we test a sloped slab domain with no flow on the sides
# and on the top. The bottom is the DeepAquiferBC. Water should
# flow from the effect of gravity.
#

#
# Import the ParFlow Python package
#
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
from parflow.tools.io import write_pfb
import sys, argparse
import numpy as np

run_name = "deep_aquifer_slab"
test = Run(run_name, __file__)
new_output_dir_name = get_absolute_path(f"test_output/{run_name}")
mkdir(new_output_dir_name)

test.FileVersion = 4

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()
print(args)

test.Process.Topology.P = args.p
test.Process.Topology.Q = args.q
test.Process.Topology.R = args.r

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
test.ComputationalGrid.Lower.X = 0.0
test.ComputationalGrid.Lower.Y = 0.0
test.ComputationalGrid.Lower.Z = 0.0

test.ComputationalGrid.DX = 2.0
test.ComputationalGrid.DY = 2.0
test.ComputationalGrid.DZ = 0.5

test.ComputationalGrid.NX = 25
test.ComputationalGrid.NY = 25
test.ComputationalGrid.NZ = 20

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
test.GeomInput.Names = "domain_input"

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------
test.GeomInput.domain_input.InputType = "Box"
test.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
test.Domain.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------
test.Geom.domain.Lower.X = 0.0
test.Geom.domain.Lower.Y = 0.0
test.Geom.domain.Lower.Z = 0.0

test.Geom.domain.Upper.X = 50.0
test.Geom.domain.Upper.Y = 50.0
test.Geom.domain.Upper.Z = 10.0

test.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
test.TimingInfo.BaseUnit = 1.0
test.TimingInfo.StartCount = 0
test.TimingInfo.StartTime = 0.0
test.TimingInfo.StopTime = 24.0
test.TimingInfo.DumpInterval = -3
test.TimeStep.Type = "Constant"
test.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
test.Cycle.Names = "constant"

test.Cycle.constant.Names = "alltime"
test.Cycle.constant.alltime.Length = 1
test.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Create DeepAquifer Files
# -----------------------------------------------------------------------------

permeability = f"{new_output_dir_name}/deep_aquifer_permeability.pfb"
data = np.full((1, 25, 25), 0.02)

permeability_pfb = write_pfb(
    get_absolute_path(permeability),
    data,
    p=1,
    q=1,
    r=1,
    x=0.0,
    y=0.0,
    z=0.0,
    dx=2,
    dy=2,
    dz=0.5,
)

test.dist(permeability)

# -----------------------------------------------------------------------------

specific_yield = f"{new_output_dir_name}/deep_aquifer_specific_yield.pfb"
data = np.full((1, 25, 25), 0.1)

specific_yield_pfb = write_pfb(
    get_absolute_path(specific_yield),
    data,
    p=1,
    q=1,
    r=1,
    x=0.0,
    y=0.0,
    z=0.0,
    dx=2,
    dy=2,
    dz=0.5,
)

test.dist(specific_yield)

# -----------------------------------------------------------------------------

elevations = f"{new_output_dir_name}/deep_aquifer_elevations.pfb"
data = np.full((1, 25, 25), 0.0)

for j in range(25):
    for i in range(25):
        data[0, j, i] = 0.8 * (i + 1 / 2) + 0.6 * (j + 1 / 2)

elevations_pfb = write_pfb(
    get_absolute_path(elevations),
    data,
    p=1,
    q=1,
    r=1,
    x=0.0,
    y=0.0,
    z=0.0,
    dx=2,
    dy=2,
    dz=0.5,
)

test.dist(elevations)

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
test.BCPressure.PatchNames = "left right front back bottom top"

test.Patch.left.BCPressure.Type = "FluxConst"
test.Patch.left.BCPressure.Cycle = "constant"
test.Patch.left.BCPressure.alltime.Value = 0.0

test.Patch.right.BCPressure.Type = "FluxConst"
test.Patch.right.BCPressure.Cycle = "constant"
test.Patch.right.BCPressure.alltime.Value = 0.0

test.Patch.front.BCPressure.Type = "FluxConst"
test.Patch.front.BCPressure.Cycle = "constant"
test.Patch.front.BCPressure.alltime.Value = 0.0

test.Patch.back.BCPressure.Type = "FluxConst"
test.Patch.back.BCPressure.Cycle = "constant"
test.Patch.back.BCPressure.alltime.Value = 0.0

# input files for DeepAquifer created above
test.Patch.bottom.BCPressure.Type = "DeepAquifer"
test.Patch.bottom.BCPressure.Cycle = "constant"
test.Patch.BCPressure.DeepAquifer.SpecificYield.Type = "PFBFile"
test.Patch.BCPressure.DeepAquifer.SpecificYield.FileName = specific_yield
test.Patch.BCPressure.DeepAquifer.AquiferDepth.Type = "Constant"
test.Patch.BCPressure.DeepAquifer.AquiferDepth.Value = 90.0
test.Patch.BCPressure.DeepAquifer.Permeability.Type = "PFBFile"
test.Patch.BCPressure.DeepAquifer.Permeability.FileName = permeability
test.Patch.BCPressure.DeepAquifer.Elevations.Type = "PFBFile"
test.Patch.BCPressure.DeepAquifer.Elevations.FileName = elevations

test.Patch.top.BCPressure.Type = "FluxConst"
test.Patch.top.BCPressure.Cycle = "constant"
test.Patch.top.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------
test.ICPressure.Type = "HydroStaticPatch"
test.ICPressure.GeomNames = "domain"
test.Geom.domain.ICPressure.Value = -5
test.Geom.domain.ICPressure.RefGeom = "domain"
test.Geom.domain.ICPressure.RefPatch = "top"

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
test.Gravity = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
test.Geom.Porosity.GeomNames = "domain"
test.Geom.domain.Porosity.Type = "Constant"
# Value for Silt soil
test.Geom.domain.Porosity.Value = 0.49

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
test.Geom.Perm.Names = "domain"
test.Geom.domain.Perm.Type = "Constant"
# Value for Silt soil in m/hour
test.Geom.domain.Perm.Value = 0.02

test.Perm.TensorType = "TensorByGeom"

test.Geom.Perm.TensorByGeom.Names = "domain"

test.Geom.domain.Perm.TensorValX = 0.0
test.Geom.domain.Perm.TensorValY = 0.0
test.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
test.Phase.Names = "water"

test.Phase.water.Density.Type = "Constant"
test.Phase.water.Density.Value = 1.0

test.Phase.water.Viscosity.Type = "Constant"
test.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
test.PhaseSources.water.Type = "Constant"
test.PhaseSources.water.GeomNames = "domain"
test.PhaseSources.water.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------
test.Phase.Saturation.Type = "VanGenuchten"
test.Phase.Saturation.GeomNames = "domain"
test.Geom.domain.Saturation.Alpha = 0.65
test.Geom.domain.Saturation.N = 2.00
test.Geom.domain.Saturation.SRes = 0.10
test.Geom.domain.Saturation.SSat = 1.0


# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
test.Phase.RelPerm.Type = "VanGenuchten"
test.Phase.RelPerm.GeomNames = "domain"
test.Geom.domain.RelPerm.Alpha = 0.65
test.Geom.domain.RelPerm.N = 2.00

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
test.Mannings.Type = "Constant"
test.Mannings.GeomNames = "domain"
test.Mannings.Geom.domain.Value = 5.5e-5

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
test.SpecificStorage.Type = "Constant"
test.SpecificStorage.GeomNames = "domain"
test.Geom.domain.SpecificStorage.Value = 1.0e-4

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
test.TopoSlopesX.Type = "Constant"
test.TopoSlopesX.GeomNames = "domain"
test.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
test.TopoSlopesY.Type = "Constant"
test.TopoSlopesY.GeomNames = "domain"
test.TopoSlopesY.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
test.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
test.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
test.Wells.Names = ""

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
test.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
test.Solver = "Richards"
test.Solver.MaxIter = 100000

test.Solver.Nonlinear.MaxIter = 250
test.Solver.Nonlinear.ResidualTol = 1e-8
test.Solver.Nonlinear.EtaChoice = "EtaConstant"
test.Solver.Nonlinear.EtaValue = 1e-12
test.Solver.Nonlinear.UseJacobian = True
test.Solver.Nonlinear.StepTol = 1e-16

test.Solver.Linear.KrylovDimension = 30

test.Solver.Linear.Preconditioner = "MGSemi"
test.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
test.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

test.Solver.PrintPressure = True
test.Solver.PrintSubsurfData = False
test.Solver.PrintSaturation = True
test.Solver.PrintMask = False

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------
correct_output_dir_name = get_absolute_path(f"../correct_output/{run_name}")
test.run(working_directory=new_output_dir_name, undist=True)

#
# Tests
#
passed = True

test_files = ["press", "satur"]
for i in range(9):
    for test_file in test_files:
        timestep = str(i).rjust(5, "0")
        filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in {test_file} for timestep {timestep}",
        ):
            passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
