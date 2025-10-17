# ---------------------------------------------------------
#  This runs a simple 2D, terrain following problem with a 5% slope
#  R. Maxwell 1-11
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "stormflow.terrain.dz5.5pc"
tfgo = Run(run_name, __file__)

# ---------------------------------------------------------

tfgo.FileVersion = 4

tfgo.Process.Topology.P = 1
tfgo.Process.Topology.Q = 1
tfgo.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

tfgo.ComputationalGrid.Lower.X = 0.0
tfgo.ComputationalGrid.Lower.Y = 0.0
tfgo.ComputationalGrid.Lower.Z = 0.0

tfgo.ComputationalGrid.NX = 20
tfgo.ComputationalGrid.NY = 1
tfgo.ComputationalGrid.NZ = 30

tfgo.ComputationalGrid.DX = 5.0
tfgo.ComputationalGrid.DY = 1.0
tfgo.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

tfgo.GeomInput.Names = "boxinput"

tfgo.GeomInput.boxinput.InputType = "Box"
tfgo.GeomInput.boxinput.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

tfgo.Geom.domain.Lower.X = 0.0
tfgo.Geom.domain.Lower.Y = 0.0
tfgo.Geom.domain.Lower.Z = 0.0

tfgo.Geom.domain.Upper.X = 100.0
tfgo.Geom.domain.Upper.Y = 1.0
tfgo.Geom.domain.Upper.Z = 1.5

tfgo.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

tfgo.Geom.Perm.Names = "domain"

tfgo.Geom.domain.Perm.Type = "Constant"
tfgo.Geom.domain.Perm.Value = 10.0

tfgo.Perm.TensorType = "TensorByGeom"

tfgo.Geom.Perm.TensorByGeom.Names = "domain"

tfgo.Geom.domain.Perm.TensorValX = 1.0
tfgo.Geom.domain.Perm.TensorValY = 1.0
tfgo.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

tfgo.SpecificStorage.Type = "Constant"
tfgo.SpecificStorage.GeomNames = "domain"
tfgo.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

tfgo.Phase.Names = "water"

tfgo.Phase.water.Density.Type = "Constant"
tfgo.Phase.water.Density.Value = 1.0

tfgo.Phase.water.Viscosity.Type = "Constant"
tfgo.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

tfgo.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

tfgo.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

tfgo.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

# run for 2 hours @ 6min timesteps
tfgo.TimingInfo.BaseUnit = 1.0
tfgo.TimingInfo.StartCount = 0
tfgo.TimingInfo.StartTime = 0.0
tfgo.TimingInfo.StopTime = 2.0
tfgo.TimingInfo.DumpInterval = -1
tfgo.TimeStep.Type = "Constant"
tfgo.TimeStep.Value = 0.1

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

tfgo.Geom.Porosity.GeomNames = "domain"
tfgo.Geom.domain.Porosity.Type = "Constant"
tfgo.Geom.domain.Porosity.Value = 0.1

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

tfgo.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

tfgo.Phase.RelPerm.Type = "VanGenuchten"
tfgo.Phase.RelPerm.GeomNames = "domain"

tfgo.Geom.domain.RelPerm.Alpha = 6.0
tfgo.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

tfgo.Phase.Saturation.Type = "VanGenuchten"
tfgo.Phase.Saturation.GeomNames = "domain"

tfgo.Geom.domain.Saturation.Alpha = 6.0
tfgo.Geom.domain.Saturation.N = 2.0
tfgo.Geom.domain.Saturation.SRes = 0.2
tfgo.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

tfgo.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

tfgo.Cycle.Names = "constant rainrec"
tfgo.Cycle.constant.Names = "alltime"
tfgo.Cycle.constant.alltime.Length = 1
tfgo.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

tfgo.Cycle.rainrec.Names = "rain rec"
tfgo.Cycle.rainrec.rain.Length = 2
tfgo.Cycle.rainrec.rec.Length = 2
tfgo.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

tfgo.BCPressure.PatchNames = tfgo.Geom.domain.Patches

tfgo.Patch.x_lower.BCPressure.Type = "FluxConst"
tfgo.Patch.x_lower.BCPressure.Cycle = "constant"
tfgo.Patch.x_lower.BCPressure.alltime.Value = 0.0

tfgo.Patch.y_lower.BCPressure.Type = "FluxConst"
tfgo.Patch.y_lower.BCPressure.Cycle = "constant"
tfgo.Patch.y_lower.BCPressure.alltime.Value = 0.0

tfgo.Patch.z_lower.BCPressure.Type = "FluxConst"
tfgo.Patch.z_lower.BCPressure.Cycle = "constant"
tfgo.Patch.z_lower.BCPressure.alltime.Value = 0.0

tfgo.Patch.x_upper.BCPressure.Type = "FluxConst"
tfgo.Patch.x_upper.BCPressure.Cycle = "constant"
tfgo.Patch.x_upper.BCPressure.alltime.Value = 0.0

tfgo.Patch.y_upper.BCPressure.Type = "FluxConst"
tfgo.Patch.y_upper.BCPressure.Cycle = "constant"
tfgo.Patch.y_upper.BCPressure.alltime.Value = 0.0

tfgo.Patch.z_upper.BCPressure.Type = "OverlandFlow"
##pfset Patch.z-upper.BCPressure.Type		      FluxConst

tfgo.Patch.z_upper.BCPressure.Cycle = "constant"
tfgo.Patch.z_upper.BCPressure.alltime.Value = 0.00

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

tfgo.TopoSlopesX.Type = "Constant"
tfgo.TopoSlopesX.GeomNames = "domain"
tfgo.TopoSlopesX.Geom.domain.Value = 0.05

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

tfgo.TopoSlopesY.Type = "Constant"
tfgo.TopoSlopesY.GeomNames = "domain"
tfgo.TopoSlopesY.Geom.domain.Value = 0.00

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

tfgo.Mannings.Type = "Constant"
tfgo.Mannings.GeomNames = "domain"
tfgo.Mannings.Geom.domain.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

tfgo.PhaseSources.water.Type = "Constant"
tfgo.PhaseSources.water.GeomNames = "domain"
tfgo.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

tfgo.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

tfgo.Solver = "Richards"

# setting this to True sets a subsurface slope that is the same as the topo slopes
tfgo.Solver.TerrainFollowingGrid = True

tfgo.Solver.MaxIter = 2500

tfgo.Solver.Nonlinear.MaxIter = 300
tfgo.Solver.Nonlinear.ResidualTol = 1e-6
tfgo.Solver.Nonlinear.EtaChoice = "EtaConstant"
tfgo.Solver.Nonlinear.EtaValue = 1e-5
tfgo.Solver.Nonlinear.UseJacobian = False
tfgo.Solver.Nonlinear.DerivativeEpsilon = 1e-12
tfgo.Solver.Nonlinear.StepTol = 1e-20
tfgo.Solver.Nonlinear.Globalization = "LineSearch"
tfgo.Solver.Linear.KrylovDimension = 20
tfgo.Solver.Linear.MaxRestart = 2

tfgo.Solver.Linear.Preconditioner = "MGSemi"
tfgo.Solver.Linear.Preconditioner = "PFMG"
tfgo.Solver.Linear.Preconditioner.SymmetricMat = "Symmetric"
tfgo.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
tfgo.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10
tfgo.Solver.PrintSubsurf = False
tfgo.Solver.Drop = 1e-20
tfgo.Solver.AbsTol = 1e-12

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be 1m from the bottom of the domain, the top layer is initially dry
tfgo.ICPressure.Type = "HydroStaticPatch"
tfgo.ICPressure.GeomNames = "domain"
tfgo.Geom.domain.ICPressure.Value = 1.0

tfgo.Geom.domain.ICPressure.RefGeom = "domain"
tfgo.Geom.domain.ICPressure.RefPatch = "z_lower"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/tfgo")
mkdir(new_output_dir_name)
correct_output_dir_name = get_absolute_path("../correct_output")
tfgo.run(working_directory=new_output_dir_name)

passed = True


for i in range(21):
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
