# ---------------------------------------------------------
# Testing versioning of enumerated list domain for Python
# library
# ---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.settings import set_parflow_version

overland_FlatICP = Run("overland_FlatICP", __file__)

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

overland_FlatICP.ComputationalGrid.Lower.X = 0.0
overland_FlatICP.ComputationalGrid.Lower.Y = 0.0
overland_FlatICP.ComputationalGrid.Lower.Z = 0.0

overland_FlatICP.ComputationalGrid.NX = 10
overland_FlatICP.ComputationalGrid.NY = 10
overland_FlatICP.ComputationalGrid.NZ = 1

overland_FlatICP.ComputationalGrid.DX = 10.0
overland_FlatICP.ComputationalGrid.DY = 10.0
overland_FlatICP.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

overland_FlatICP.GeomInput.Names = "domaininput sourceinput"

overland_FlatICP.GeomInput.domaininput.GeomName = "domain"
overland_FlatICP.GeomInput.domaininput.InputType = "Box"

overland_FlatICP.GeomInput.sourceinput.GeomName = "icsource"
overland_FlatICP.GeomInput.sourceinput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

overland_FlatICP.Geom.domain.Lower.X = 0.0
overland_FlatICP.Geom.domain.Lower.Y = 0.0
overland_FlatICP.Geom.domain.Lower.Z = 0.0

overland_FlatICP.Geom.domain.Upper.X = 100.0
overland_FlatICP.Geom.domain.Upper.Y = 100.0
overland_FlatICP.Geom.domain.Upper.Z = 0.05
overland_FlatICP.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

overland_FlatICP.Geom.icsource.Lower.X = 40.0
overland_FlatICP.Geom.icsource.Lower.Y = 40.0
overland_FlatICP.Geom.icsource.Lower.Z = 0.0

overland_FlatICP.Geom.icsource.Upper.X = 60.0
overland_FlatICP.Geom.icsource.Upper.Y = 60.0
overland_FlatICP.Geom.icsource.Upper.Z = 0.05

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

overland_FlatICP.Geom.Perm.Names = "domain"

# Values in m/hour
overland_FlatICP.Geom.domain.Perm.Type = "Constant"
overland_FlatICP.Geom.domain.Perm.Value = 0.000694

overland_FlatICP.Perm.TensorType = "TensorByGeom"
overland_FlatICP.Geom.Perm.TensorByGeom.Names = "domain"

overland_FlatICP.Geom.domain.Perm.TensorValX = 1.0
overland_FlatICP.Geom.domain.Perm.TensorValY = 1.0
overland_FlatICP.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

overland_FlatICP.SpecificStorage.Type = "Constant"
overland_FlatICP.SpecificStorage.GeomNames = "domain"
overland_FlatICP.Geom.domain.SpecificStorage.Value = 5.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

overland_FlatICP.Phase.Names = "water"

overland_FlatICP.Phase.water.Density.Type = "Constant"
overland_FlatICP.Phase.water.Density.Value = 1.0

overland_FlatICP.Phase.water.Viscosity.Type = "Constant"
overland_FlatICP.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

overland_FlatICP.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

overland_FlatICP.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

overland_FlatICP.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

overland_FlatICP.TimingInfo.BaseUnit = 1.0
overland_FlatICP.TimingInfo.StartCount = 0
overland_FlatICP.TimingInfo.StartTime = 0.0
overland_FlatICP.TimingInfo.StopTime = 300.0
overland_FlatICP.TimingInfo.DumpInterval = 30.0
overland_FlatICP.TimeStep.Type = "Constant"
overland_FlatICP.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

overland_FlatICP.Geom.Porosity.GeomNames = "domain"
overland_FlatICP.Geom.domain.Porosity.Type = "Constant"
overland_FlatICP.Geom.domain.Porosity.Value = 0.001

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

overland_FlatICP.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

overland_FlatICP.Phase.RelPerm.Type = "VanGenuchten"
overland_FlatICP.Phase.RelPerm.GeomNames = "domain"

overland_FlatICP.Geom.domain.RelPerm.Alpha = 1.0
overland_FlatICP.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

overland_FlatICP.Phase.Saturation.Type = "VanGenuchten"
overland_FlatICP.Phase.Saturation.GeomNames = "domain"

overland_FlatICP.Geom.domain.Saturation.Alpha = 1.0
overland_FlatICP.Geom.domain.Saturation.N = 2.0
overland_FlatICP.Geom.domain.Saturation.SRes = 0.2
overland_FlatICP.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

overland_FlatICP.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

overland_FlatICP.Cycle.Names = "constant rainrec"
overland_FlatICP.Cycle.constant.Names = "alltime"
overland_FlatICP.Cycle.constant.alltime.Length = 1
overland_FlatICP.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland_FlatICP.Cycle.rainrec.Names = "rain rec"
overland_FlatICP.Cycle.rainrec.rain.Length = 200
overland_FlatICP.Cycle.rainrec.rec.Length = 100
overland_FlatICP.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

overland_FlatICP.BCPressure.PatchNames = overland_FlatICP.Geom.domain.Patches

overland_FlatICP.Patch.x_lower.BCPressure.Type = "FluxConst"
overland_FlatICP.Patch.x_lower.BCPressure.Cycle = "constant"
overland_FlatICP.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland_FlatICP.Patch.y_lower.BCPressure.Type = "FluxConst"
overland_FlatICP.Patch.y_lower.BCPressure.Cycle = "constant"
overland_FlatICP.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland_FlatICP.Patch.z_lower.BCPressure.Type = "FluxConst"
overland_FlatICP.Patch.z_lower.BCPressure.Cycle = "constant"
overland_FlatICP.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland_FlatICP.Patch.x_upper.BCPressure.Type = "FluxConst"
overland_FlatICP.Patch.x_upper.BCPressure.Cycle = "constant"
overland_FlatICP.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland_FlatICP.Patch.y_upper.BCPressure.Type = "FluxConst"
overland_FlatICP.Patch.y_upper.BCPressure.Cycle = "constant"
overland_FlatICP.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
overland_FlatICP.Patch.z_upper.BCPressure.Type = "OverlandFlow"
overland_FlatICP.Patch.z_upper.BCPressure.Cycle = "rainrec"
overland_FlatICP.Patch.z_upper.BCPressure.rain.Value = 0.0
overland_FlatICP.Patch.z_upper.BCPressure.rec.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

overland_FlatICP.TopoSlopesX.Type = "Constant"
overland_FlatICP.TopoSlopesX.GeomNames = "domain"
overland_FlatICP.TopoSlopesX.Geom.domain.Value = 0.00

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

overland_FlatICP.TopoSlopesY.Type = "Constant"
overland_FlatICP.TopoSlopesY.GeomNames = "domain"
overland_FlatICP.TopoSlopesY.Geom.domain.Value = 0.00

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

overland_FlatICP.Mannings.Type = "Constant"
overland_FlatICP.Mannings.GeomNames = "domain"
overland_FlatICP.Mannings.Geom.domain.Value = 0.0003312

# Phase sources:
# -----------------------------------------------------------------------------

overland_FlatICP.PhaseSources.water.Type = "Constant"
overland_FlatICP.PhaseSources.water.GeomNames = "domain"
overland_FlatICP.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

overland_FlatICP.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

overland_FlatICP.Solver = "Richards"
overland_FlatICP.Solver.MaxIter = 30000

overland_FlatICP.Solver.Nonlinear.MaxIter = 300
overland_FlatICP.Solver.Nonlinear.ResidualTol = 1e-8
overland_FlatICP.Solver.Nonlinear.EtaChoice = "Walker1"
overland_FlatICP.Solver.Nonlinear.EtaChoice = "EtaConstant"
overland_FlatICP.Solver.Nonlinear.EtaValue = 0.001
overland_FlatICP.Solver.Nonlinear.UseJacobian = False
overland_FlatICP.Solver.Nonlinear.DerivativeEpsilon = 1e-12
overland_FlatICP.Solver.Nonlinear.StepTol = 1e-30
overland_FlatICP.Solver.Nonlinear.Globalization = "LineSearch"
overland_FlatICP.Solver.Linear.KrylovDimension = 20
overland_FlatICP.Solver.Linear.MaxRestart = 2
overland_FlatICP.Solver.OverlandDiffusive.Epsilon = 1e-5

overland_FlatICP.Solver.Linear.Preconditioner = "PFMG"
overland_FlatICP.Solver.PrintSubsurf = False
overland_FlatICP.Solver.Drop = 1e-20
overland_FlatICP.Solver.AbsTol = 1e-12

overland_FlatICP.Solver.WriteSiloSubsurfData = False
overland_FlatICP.Solver.WriteSiloPressure = False
overland_FlatICP.Solver.WriteSiloSaturation = False

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------
overland_FlatICP.ICPressure.Type = "HydroStaticPatch"
overland_FlatICP.ICPressure.GeomNames = "domain icsource"

overland_FlatICP.Geom.domain.ICPressure.Value = 0.0
overland_FlatICP.Geom.domain.ICPressure.RefGeom = "domain"
overland_FlatICP.Geom.domain.ICPressure.RefPatch = "z_upper"

overland_FlatICP.Geom.icsource.ICPressure.Value = 0.1
overland_FlatICP.Geom.icsource.ICPressure.RefGeom = "domain"
overland_FlatICP.Geom.icsource.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------
runcheck = 1

# -----------------------------------------------------------------------------
# original approach from K&M AWR 2006
# -----------------------------------------------------------------------------

overland_FlatICP.Patch.z_upper.BCPressure.Type = "OverlandFlow"
overland_FlatICP.Solver.Nonlinear.UseJacobian = False
overland_FlatICP.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

# -----------------------------------------------------------------------------
# New kinematic formulation - this should exactly match the original formulation
#  for this flat test case
# -----------------------------------------------------------------------------

overland_FlatICP.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland_FlatICP.Solver.Nonlinear.UseJacobian = False
overland_FlatICP.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

# -----------------------------------------------------------------------------
# Diffusive formulation
# -----------------------------------------------------------------------------

# run with Jacobian False
overland_FlatICP.Patch.z_upper.BCPressure.Type = "OverlandDiffusive"

# -----------------------------------------------------------------------------
# Testing version compatibility
# -----------------------------------------------------------------------------

print("=" * 80)
print("Test with version 1.0.0 of ParFlow")
print("=" * 80)

set_parflow_version("1.0.0")
nb_error_v1 = overland_FlatICP.validate()

# -----------------------------------------------------------------------------

print("=" * 80)
print("Test with version 3.6.0 of ParFlow")
print("=" * 80)

set_parflow_version("3.6.0")
nb_error_v3 = overland_FlatICP.validate()

print("=" * 80)

# -----------------------------------------------------------------------------
# Asserts
# -----------------------------------------------------------------------------

found_error = False

if nb_error_v1 != 1:
    print(f"Expected to have 1 error with v1 but got {nb_error_v1}")
    found_error = True

if nb_error_v3 != 0:
    print(f"Expected no errors with v3.6.0 but got {nb_error_v3}")
    found_error = True

if found_error:
    sys.exit(1)
