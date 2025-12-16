# -----------------------------------------------------------------------------
# Test CLM with multiple reuse values for input.  The same test is run
# with different timesteps and reuse values set to match.   E.G. 1s = reuse 1, 0.1s = reuse 10.
# -----------------------------------------------------------------------------

import sys
import os
import math
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path, rm
from parflow.tools.io import read_pfb

run_name = "reuse"

clm = Run(run_name, __file__)

# -----------------------------------------------------------------------------
# Copying input files
# -----------------------------------------------------------------------------

dir_name = get_absolute_path("test_output/clm_reuse")
mkdir(dir_name)

cp("$PF_SRC/test/tcl/clm/clm-reuse/drv_clmin.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/clm-reuse/drv_vegm.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/clm-reuse/drv_vegp.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/clm-reuse/forcing_1.txt", dir_name)

# -----------------------------------------------------------------------------
# Setting test variables
# -----------------------------------------------------------------------------

# Total runtime of simulation
stopt = 100

# Reuse values to run with
reuseValues = [1, 4]

# This was set for reuse = 4 test; other reuse values will fail
relativeErrorTolerance = 0.2

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------

clm.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

clm.Process.Topology.P = 1
clm.Process.Topology.Q = 1
clm.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 2.0
clm.ComputationalGrid.DY = 2.0
clm.ComputationalGrid.DZ = 0.1

clm.ComputationalGrid.NX = 1
clm.ComputationalGrid.NY = 1
clm.ComputationalGrid.NZ = 100

nx = clm.ComputationalGrid.NX
dx = clm.ComputationalGrid.DX
ny = clm.ComputationalGrid.NY
dy = clm.ComputationalGrid.DY
nz = clm.ComputationalGrid.NZ
dz = clm.ComputationalGrid.DZ

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------

clm.GeomInput.Names = "domain_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

clm.GeomInput.domain_input.InputType = "Box"
clm.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

clm.Geom.domain.Lower.X = 0.0
clm.Geom.domain.Lower.Y = 0.0
clm.Geom.domain.Lower.Z = 0.0

clm.Geom.domain.Upper.X = nx * dx
clm.Geom.domain.Upper.Y = ny * dy
clm.Geom.domain.Upper.Z = nz * dz

clm.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

clm.Geom.Perm.Names = "domain"

clm.Geom.domain.Perm.Type = "Constant"
clm.Geom.domain.Perm.Value = 0.04465

clm.Perm.TensorType = "TensorByGeom"

clm.Geom.Perm.TensorByGeom.Names = "domain"

clm.Geom.domain.Perm.TensorValX = 1.0
clm.Geom.domain.Perm.TensorValY = 1.0
clm.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm.SpecificStorage.Type = "Constant"
clm.SpecificStorage.GeomNames = "domain"
clm.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm.Phase.Names = "water"

clm.Phase.water.Density.Type = "Constant"
clm.Phase.water.Density.Value = 1.0

clm.Phase.water.Viscosity.Type = "Constant"
clm.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

clm.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

clm.TimingInfo.BaseUnit = 1.0
clm.TimingInfo.StartCount = 0
clm.TimingInfo.StartTime = 0.0
clm.TimingInfo.StopTime = stopt
clm.TimingInfo.DumpInterval = 1.0
clm.TimeStep.Type = "Constant"

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm.Geom.Porosity.GeomNames = "domain"
clm.Geom.domain.Porosity.Type = "Constant"
clm.Geom.domain.Porosity.Value = 0.5

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

clm.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------

clm.Phase.water.Mobility.Type = "Constant"
clm.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

clm.Phase.RelPerm.Type = "VanGenuchten"
clm.Phase.RelPerm.GeomNames = "domain"

clm.Geom.domain.RelPerm.Alpha = 2.0
clm.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm.Phase.Saturation.Type = "VanGenuchten"
clm.Phase.Saturation.GeomNames = "domain"

clm.Geom.domain.Saturation.Alpha = 2.0
clm.Geom.domain.Saturation.N = 3.0
clm.Geom.domain.Saturation.SRes = 0.2
clm.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

clm.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

clm.Cycle.Names = "constant"
clm.Cycle.constant.Names = "alltime"
clm.Cycle.constant.alltime.Length = 1
clm.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

clm.BCPressure.PatchNames = clm.Geom.domain.Patches

clm.Patch.x_lower.BCPressure.Type = "FluxConst"
clm.Patch.x_lower.BCPressure.Cycle = "constant"
clm.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm.Patch.y_lower.BCPressure.Type = "FluxConst"
clm.Patch.y_lower.BCPressure.Cycle = "constant"
clm.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm.Patch.z_lower.BCPressure.Type = "FluxConst"
clm.Patch.z_lower.BCPressure.Cycle = "constant"
clm.Patch.z_lower.BCPressure.alltime.Value = -0.00

clm.Patch.x_upper.BCPressure.Type = "FluxConst"
clm.Patch.x_upper.BCPressure.Cycle = "constant"
clm.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm.Patch.y_upper.BCPressure.Type = "FluxConst"
clm.Patch.y_upper.BCPressure.Cycle = "constant"
clm.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm.Patch.z_upper.BCPressure.Cycle = "constant"
clm.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

clm.TopoSlopesX.Type = "Constant"
clm.TopoSlopesX.GeomNames = "domain"
clm.TopoSlopesX.Geom.domain.Value = 0.005

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

clm.TopoSlopesY.Type = "Constant"
clm.TopoSlopesY.GeomNames = "domain"
clm.TopoSlopesY.Geom.domain.Value = 0.00

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

clm.Mannings.Type = "Constant"
clm.Mannings.GeomNames = "domain"
clm.Mannings.Geom.domain.Value = 1e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm.PhaseSources.water.Type = "Constant"
clm.PhaseSources.water.GeomNames = "domain"
clm.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

clm.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

clm.Solver = "Richards"
clm.Solver.MaxIter = 90000

clm.Solver.Nonlinear.MaxIter = 100
clm.Solver.Nonlinear.ResidualTol = 1e-5
clm.Solver.Nonlinear.EtaChoice = "Walker1"
clm.Solver.Nonlinear.UseJacobian = True
clm.Solver.Nonlinear.DerivativeEpsilon = 1e-12
clm.Solver.Nonlinear.StepTol = 1e-30
clm.Solver.Nonlinear.Globalization = "LineSearch"
clm.Solver.Linear.KrylovDimension = 100
clm.Solver.Linear.MaxRestarts = 5

clm.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

clm.Solver.Linear.Preconditioner = "PFMG"
clm.Solver.PrintSubsurf = False
clm.Solver.Drop = 1e-20
clm.Solver.AbsTol = 1e-9

clm.Solver.LSM = "CLM"
clm.Solver.WriteSiloCLM = True
clm.Solver.CLM.MetForcing = "1D"
clm.Solver.CLM.MetFileName = "forcing_1.txt"
clm.Solver.CLM.MetFilePath = "./"

clm.Solver.CLM.EvapBeta = "Linear"

clm.Solver.PrintSubsurfData = True
clm.Solver.PrintPressure = False
clm.Solver.PrintSaturation = True
clm.Solver.PrintCLM = True
clm.Solver.PrintMask = True
clm.Solver.PrintSpecificStorage = True

clm.Solver.PrintLSMSink = False
clm.Solver.CLM.CLMDumpInterval = 1
clm.Solver.CLM.CLMFileDir = "output/"
clm.Solver.CLM.BinaryOutDir = False
clm.Solver.CLM.IstepStart = 1
clm.Solver.WriteCLMBinary = False
clm.Solver.WriteSiloCLM = False

clm.Solver.CLM.WriteLogs = False
clm.Solver.CLM.WriteLastRST = True
clm.Solver.CLM.DailyRST = False
clm.Solver.CLM.SingleFile = True

clm.Solver.CLM.EvapBeta = "Linear"
clm.Solver.CLM.VegWaterStress = "Saturation"
clm.Solver.CLM.ResSat = 0.2
clm.Solver.CLM.WiltingPoint = 0.2
clm.Solver.CLM.FieldCapacity = 1.00
clm.Solver.CLM.IrrigationType = "none"

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

clm.ICPressure.Type = "HydroStaticPatch"
clm.ICPressure.GeomNames = "domain"
clm.Geom.domain.ICPressure.Value = -1.0
clm.Geom.domain.ICPressure.RefGeom = "domain"
clm.Geom.domain.ICPressure.RefPatch = "z_upper"


# Run each of the cases
for reuseCount in reuseValues:
    clm.Solver.CLM.ReuseCount = reuseCount
    clm.TimeStep.Value = 1.0 / reuseCount

    new_dir_name = "clm-reuse-ts-{:.2f}".format(clm.TimeStep.Value)
    print(f"Running: {new_dir_name}")

    # Run and Unload the ParFlow output files
    clm.run(working_directory=dir_name)

    for k in range(1, stopt + 1):
        outfile1 = "{}.out.clm_output.{:05d}.C.pfb".format(run_name, k)

    rm(get_absolute_path(new_dir_name))
    mkdir(new_dir_name)
    new_dir_path = get_absolute_path(new_dir_name)
    os.system(
        "bash -c 'mv {}* {}'".format(os.path.join(dir_name, run_name), new_dir_path)
    )
    file_path = os.path.join(dir_name, "CLM.out.clm.log")
    os.system("mv {} {}".format(file_path, new_dir_path))
    file_path = os.path.join(dir_name, "clm.rst.00000.0")
    os.system("mv {} {}".format(file_path, new_dir_path))

# Dictionary to store norm values for each reuseCount
norm = {}
file_path = {}
ds = {}

# Post process output
with open("swe.out.csv", "w") as sweFile:
    sweFile.write("Time")
    for reuseCount in reuseValues:
        norm[reuseCount] = 0.0
        timeStep = 1.0 / reuseCount
        sweFile.write(",{:.10e}".format(timeStep))
    sweFile.write("\n")

    compareReuse = reuseValues[0]

    for k in range(1, stopt + 1):
        sweFile.write("{}".format(k))

        for reuseCount in reuseValues:
            timeStep = 1.0 / reuseCount
            dirname1 = get_absolute_path("clm-reuse-ts-{:.2f}".format(timeStep))
            file_path[reuseCount] = "{}/{}.out.clm_output.{:05d}.C.pfb".format(
                dirname1, run_name, k
            )
            ds[reuseCount] = read_pfb(file_path[reuseCount], z_first=False)

        sweFile.write("{:d}".format(k))

        for reuseCount in reuseValues:
            if reuseCount == compareReuse:
                norm[compareReuse] = (
                    norm.get(compareReuse)
                    + ds[compareReuse][0, 0, 10] * ds[compareReuse][0, 0, 10]
                )
            else:
                norm[reuseCount] = norm.get(reuseCount) + (
                    ds[compareReuse][0, 0, 10] - ds[reuseCount][0, 0, 10]
                ) * (ds[compareReuse][0, 0, 10] - ds[reuseCount][0, 0, 10])

        sweFile.write("\n")

# Calculate the square root of the norms
for reuseCount in reuseValues:
    norm[reuseCount] = math.sqrt(norm[reuseCount])

# Tests
passed = True

for reuseCount in reuseValues[1:]:
    relerror = norm[reuseCount] / norm[compareReuse]
    if relerror > relativeErrorTolerance:
        print(
            "FAILED: Relative error for reuse count = {} exceeds error tolerance ({} > {})".format(
                reuseCount, relerror, relativeErrorTolerance
            )
        )
        passed = False

if passed:
    print("clm-reuse: PASSED")
else:
    print("clm-reuse: FAILED")
    sys.exit(1)
