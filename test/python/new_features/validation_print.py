# ---------------------------------------------------------
#  Same as default_richards.py run, with several errors to
#  demonstrate the skip_valid argument in validate
# ---------------------------------------------------------

from parflow import Run

drich = Run("default_richards", __file__)

# ---------------------------------------------------------

drich.Process.Topology.P = 1.5
drich.Process.Topology.Q = 1
drich.Process.Topology.R = -1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

drich.ComputationalGrid.Lower.X = -10.0
drich.ComputationalGrid.Lower.Y = 10.0
drich.ComputationalGrid.Lower.Z = 1.0

drich.ComputationalGrid.DX = 8.8888888888888893
drich.ComputationalGrid.DY = 10.666666666666666
drich.ComputationalGrid.DZ = 1.0

drich.ComputationalGrid.NX = 10
drich.ComputationalGrid.NY = 10
drich.ComputationalGrid.NZ = 8

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

drich.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------

drich.GeomInput.domain_input.InputType = "Box"
drich.GeomInput.domain_input.GeomName = "domain"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

drich.Geom.domain.Lower.X = -10.0
drich.Geom.domain.Lower.Y = 10.0
drich.Geom.domain.Lower.Z = 1.0

drich.Geom.domain.Upper.X = 150.0
drich.Geom.domain.Upper.Y = 170.0
drich.Geom.domain.Upper.Z = 9.0

drich.Geom.domain.Patches = "left right front back bottom top"

# ---------------------------------------------------------
# Background Geometry Input
# ---------------------------------------------------------

drich.GeomInput.background_input.InputType = "Box"
drich.GeomInput.background_input.GeomName = "background"

# ---------------------------------------------------------
# Background Geometry
# ---------------------------------------------------------

drich.Geom.background.Lower.X = -99999999.0
drich.Geom.background.Lower.Y = -99999999.0
drich.Geom.background.Lower.Z = -99999999.0

drich.Geom.background.Upper.X = 99999999.0
drich.Geom.background.Upper.Y = 99999999.0
drich.Geom.background.Upper.Z = 99999999.0

# ---------------------------------------------------------
# Source_Region Geometry Input
# ---------------------------------------------------------

drich.GeomInput.source_region_input.InputType = "Box"
drich.GeomInput.source_region_input.GeomName = "source_region"

# ---------------------------------------------------------
# Source_Region Geometry
# ---------------------------------------------------------

drich.Geom.source_region.Lower.X = 65.56
drich.Geom.source_region.Lower.Y = 79.34
drich.Geom.source_region.Lower.Z = 4.5

drich.Geom.source_region.Upper.X = 74.44
drich.Geom.source_region.Upper.Y = 89.99
drich.Geom.source_region.Upper.Z = 5.5

# ---------------------------------------------------------
# Concen_Region Geometry Input
# ---------------------------------------------------------

drich.GeomInput.concen_region_input.InputType = "Box"
drich.GeomInput.concen_region_input.GeomName = "concen_region"

# ---------------------------------------------------------
# Concen_Region Geometry
# ---------------------------------------------------------

drich.Geom.concen_region.Lower.X = 60.0
drich.Geom.concen_region.Lower.Y = 80.0
drich.Geom.concen_region.Lower.Z = 4.0

drich.Geom.concen_region.Upper.X = 80.0
drich.Geom.concen_region.Upper.Y = 100.0
drich.Geom.concen_region.Upper.Z = 6.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

drich.Geom.Perm.Names = "background"

drich.Geom.background.Perm.Type = "Constant"
drich.Geom.background.Perm.Value = 4.0

drich.Perm.TensorType = "TensorByGeom"

drich.Geom.Perm.TensorByGeom.Names = "background"

drich.Geom.background.Perm.TensorValX = 1.0
drich.Geom.background.Perm.TensorValY = 1.0
drich.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

drich.SpecificStorage.Type = "Constant"
drich.SpecificStorage.GeomNames = "domain"
drich.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

drich.Phase.Names = "water"

drich.Phase.water.Density.Type = "Constant"
drich.Phase.water.Density.Value = 1.0

drich.Phase.water.Viscosity.Type = "Constant"
drich.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

drich.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

drich.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

drich.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

drich.TimingInfo.BaseUnit = 1.0
drich.TimingInfo.StartCount = 0
drich.TimingInfo.StartTime = 0.0
drich.TimingInfo.StopTime = 0.010
drich.TimingInfo.DumpInterval = -1
drich.TimeStep.Type = "Random"
drich.TimeStep.Value = 0.001

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

drich.Geom.Porosity.GeomNames = "background"
drich.Geom.background.Porosity.Type = "Constant"
drich.Geom.background.Porosity.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

drich.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

drich.Phase.RelPerm.Type = "VanGenuchten"
drich.Phase.RelPerm.GeomNames = "domain"
drich.Geom.domain.RelPerm.Alpha = 0.005
drich.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

drich.Phase.Saturation.Type = "VanGenuchten"
drich.Phase.Saturation.GeomNames = "domain"
drich.Geom.domain.Saturation.Alpha = 0.005
drich.Geom.domain.Saturation.N = 2.0
drich.Geom.domain.Saturation.SRes = 0.2
drich.Geom.domain.Saturation.SSat = 1.99

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

drich.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

drich.Cycle.Names = "constant"
drich.Cycle.constant.Names = "alltime"
drich.Cycle.constant.alltime.Length = 1
drich.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

drich.BCPressure.PatchNames = "left right front back bottom top"

drich.Patch.left.BCPressure.Type = "DirEquilRefPatch"
drich.Patch.left.BCPressure.Cycle = "constant"
drich.Patch.left.BCPressure.RefGeom = "domain"
drich.Patch.left.BCPressure.RefPatch = "bottom"
drich.Patch.left.BCPressure.alltime.Value = 5.0

drich.Patch.right.BCPressure.Type = "DirEquilRefPatch"
drich.Patch.right.BCPressure.Cycle = "constant"
drich.Patch.right.BCPressure.RefGeom = "domain"
drich.Patch.right.BCPressure.RefPatch = "bottom"
drich.Patch.right.BCPressure.alltime.Value = 3.0

drich.Patch.front.BCPressure.Type = "FluxConst"
drich.Patch.front.BCPressure.Cycle = "constant"
drich.Patch.front.BCPressure.alltime.Value = 0.0

drich.Patch.back.BCPressure.Type = "FluxConst"
drich.Patch.back.BCPressure.Cycle = "constant"
drich.Patch.back.BCPressure.alltime.Value = 0.0

drich.Patch.bottom.BCPressure.Type = "FluxConst"
drich.Patch.bottom.BCPressure.Cycle = "constant"
drich.Patch.bottom.BCPressure.alltime.Value = 0.0

drich.Patch.top.BCPressure.Type = "FluxConst"
drich.Patch.top.BCPressure.Cycle = "constant"
drich.Patch.top.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

drich.TopoSlopesX.Type = "Constant"
drich.TopoSlopesX.GeomNames = "domain"
drich.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

drich.TopoSlopesY.Type = "Constant"
drich.TopoSlopesY.GeomNames = "domain"
drich.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

drich.Mannings.Type = "Constant"
drich.Mannings.GeomNames = "domain"
drich.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

drich.ICPressure.Type = "HydroStaticPatch"
drich.ICPressure.GeomNames = "domain"
drich.Geom.domain.ICPressure.Value = 3.0
drich.Geom.domain.ICPressure.RefGeom = "domain"
drich.Geom.domain.ICPressure.RefPatch = "bottom"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

drich.PhaseSources.water.Type = "Constant"
drich.PhaseSources.water.GeomNames = "background"
drich.PhaseSources.water.Geom.background.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

drich.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

drich.Solver = "Richards"
drich.Solver.MaxIter = 5

drich.Solver.Nonlinear.MaxIter = -10
drich.Solver.Nonlinear.ResidualTol = 1e-9
drich.Solver.Nonlinear.EtaChoice = "EtaConstant"
drich.Solver.Nonlinear.EtaValue = 1e-5
drich.Solver.Nonlinear.UseJacobian = True
drich.Solver.Nonlinear.DerivativeEpsilon = 1e-2

drich.Solver.Linear.KrylovDimension = 10

drich.Solver.Linear.Preconditioner = "PFMG"

# -----------------------------------------------------------------------------
# Run ParFlow
# -----------------------------------------------------------------------------

drich.validate(verbose=False)
