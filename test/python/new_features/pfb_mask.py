# -----------------------------------------------------------------------------
# example for pfsol generation
# Testing pfb to mask generation
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import load_patch_matrix_from_pfb_file
from parflow.tools.builders import SolidFileBuilder

sabino = Run("sabino", __file__)

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------

sabino.Process.Topology.P = 1
sabino.Process.Topology.Q = 1
sabino.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

sabino.ComputationalGrid.Lower.X = 0.0
sabino.ComputationalGrid.Lower.Y = 0.0
sabino.ComputationalGrid.Lower.Z = 0.0

sabino.ComputationalGrid.DX = 90.0
sabino.ComputationalGrid.DY = 90.0
sabino.ComputationalGrid.DZ = 100.0

sabino.ComputationalGrid.NX = 91
sabino.ComputationalGrid.NY = 70
sabino.ComputationalGrid.NZ = 20

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

sabino.GeomInput.Names = "domaininput"

sabino.GeomInput.domaininput.GeomName = "domain"
sabino.GeomInput.domaininput.GeomNames = "domain"

# -----------------------------------------------------------------------------
# PFSOL generation
# -----------------------------------------------------------------------------

sabino_mask = load_patch_matrix_from_pfb_file(
    get_absolute_path("$PF_SRC/test/input/sabino_mask.pfb")
)

SolidFileBuilder(top=1, bottom=2, side=3).mask(sabino_mask).write(
    "sabino_domain.pfsol", xllcorner=0, yllcorner=0, cellsize=90, vtk=True
).for_key(sabino.GeomInput.domaininput)

# -----------------------------------------------------------------------------

sabino.Geom.domain.Patches = "top bottom edge"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

sabino.Geom.domain.Lower.X = 0.0
sabino.Geom.domain.Lower.Y = 0.0
sabino.Geom.domain.Lower.Z = 0.0

sabino.Geom.domain.Upper.X = 8190.0
sabino.Geom.domain.Upper.Y = 6300.0
sabino.Geom.domain.Upper.Z = 2000.0

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

sabino.Geom.Perm.Names = "domain"

sabino.Geom.domain.Perm.Type = "Constant"
sabino.Geom.domain.Perm.Value = 0.0018

sabino.Perm.TensorType = "TensorByGeom"
sabino.Geom.Perm.TensorByGeom.Names = "domain"
sabino.Geom.domain.Perm.TensorValX = 1.0
sabino.Geom.domain.Perm.TensorValY = 1.0
sabino.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

sabino.SpecificStorage.Type = "Constant"
sabino.SpecificStorage.GeomNames = "domain"
sabino.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

sabino.Phase.Names = "water"
sabino.Phase.water.Density.Type = "Constant"
sabino.Phase.water.Density.Value = 1.0
sabino.Phase.water.Viscosity.Type = "Constant"
sabino.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

sabino.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

sabino.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

sabino.TimingInfo.BaseUnit = 1.0
sabino.TimingInfo.StartCount = 0
sabino.TimingInfo.StartTime = 0.0
sabino.TimingInfo.StopTime = 6.0
sabino.TimingInfo.DumpInterval = 1.0
sabino.TimeStep.Type = "Constant"
sabino.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

sabino.Geom.Porosity.GeomNames = "domain"
sabino.Geom.domain.Porosity.Type = "Constant"
sabino.Geom.domain.Porosity.Value = 0.1

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

sabino.Domain.GeomName = "domain"

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

sabino.Phase.water.Mobility.Type = "Constant"
sabino.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

sabino.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

sabino.Cycle.Names = "constant"
sabino.Cycle.constant.Names = "alltime"
sabino.Cycle.constant.alltime.Length = 1
sabino.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

sabino.BCPressure.PatchNames = "edge top bottom"

sabino.Patch.edge.BCPressure.Type = "FluxConst"
sabino.Patch.edge.BCPressure.Cycle = "constant"
sabino.Patch.edge.BCPressure.alltime.Value = 0.0

sabino.Patch.bottom.BCPressure.Type = "FluxConst"
sabino.Patch.bottom.BCPressure.Cycle = "constant"
sabino.Patch.bottom.BCPressure.alltime.Value = 0.0

sabino.Patch.top.BCPressure.alltime.Value = -0.0035
sabino.Patch.top.BCPressure.Type = "OverlandFlow"
sabino.Patch.top.BCPressure.Cycle = "constant"

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

sabino.TopoSlopesX.Type = "Constant"
sabino.TopoSlopesX.GeomNames = "domain"
sabino.TopoSlopesX.Geom.domain.Value = -0.005

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

sabino.TopoSlopesY.Type = "Constant"
sabino.TopoSlopesY.GeomNames = "domain"
sabino.TopoSlopesY.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

sabino.Mannings.Type = "Constant"
sabino.Mannings.GeomNames = "domain"
sabino.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

sabino.Phase.RelPerm.Type = "VanGenuchten"
sabino.Phase.RelPerm.GeomNames = "domain"
sabino.Geom.domain.RelPerm.Alpha = 3.5
sabino.Geom.domain.RelPerm.N = 2.0

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

sabino.Phase.Saturation.Type = "VanGenuchten"
sabino.Phase.Saturation.GeomNames = "domain"

sabino.Geom.domain.Saturation.Alpha = 3.5
sabino.Geom.domain.Saturation.N = 2.0
sabino.Geom.domain.Saturation.SRes = 0.1
sabino.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

sabino.PhaseSources.water.Type = "Constant"
sabino.PhaseSources.water.GeomNames = "domain"
sabino.PhaseSources.water.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

sabino.ICPressure.Type = "Constant"
sabino.ICPressure.GeomNames = "domain"
sabino.Geom.domain.ICPressure.RefPatch = "top"
sabino.Geom.domain.ICPressure.Value = -10

# ----------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

sabino.Solver.PrintSubsurf = True
sabino.Solver.PrintPressure = True
sabino.Solver.PrintSaturation = True
sabino.Solver.PrintMask = True

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

sabino.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

sabino.Solver = "Richards"
sabino.Solver.TerrainFollowingGrid = True

# -----------------------------------------------------------------------------
# Setting up vertical layering
# -----------------------------------------------------------------------------

sabino.Solver.Nonlinear.VariableDz = True
sabino.dzScale.GeomNames = "domain"
sabino.dzScale.Type = "nzList"
sabino.dzScale.nzListNumber = 20
sabino.Cell._0.dzScale.Value = 2.0
sabino.Cell._1.dzScale.Value = 2.0
sabino.Cell._2.dzScale.Value = 2.0
sabino.Cell._3.dzScale.Value = 1.0
sabino.Cell._4.dzScale.Value = 1.0
sabino.Cell._5.dzScale.Value = 0.50
sabino.Cell._6.dzScale.Value = 0.50
sabino.Cell._7.dzScale.Value = 0.20
sabino.Cell._8.dzScale.Value = 0.20
sabino.Cell._9.dzScale.Value = 0.20
sabino.Cell._10.dzScale.Value = 0.20
sabino.Cell._11.dzScale.Value = 0.10
sabino.Cell._12.dzScale.Value = 0.02
sabino.Cell._13.dzScale.Value = 0.02
sabino.Cell._14.dzScale.Value = 0.02
sabino.Cell._15.dzScale.Value = 0.02
sabino.Cell._16.dzScale.Value = 0.01
sabino.Cell._17.dzScale.Value = 0.006
sabino.Cell._18.dzScale.Value = 0.003
sabino.Cell._19.dzScale.Value = 0.001

sabino.Solver.MaxIter = 25000000
sabino.Solver.Drop = 1e-20
sabino.Solver.AbsTol = 1e-8
sabino.Solver.MaxConvergenceFailures = 8
sabino.Solver.Nonlinear.MaxIter = 80
sabino.Solver.Nonlinear.ResidualTol = 1e-6

sabino.Solver.Nonlinear.EtaChoice = "EtaConstant"
sabino.Solver.Nonlinear.EtaValue = 0.001
sabino.Solver.Nonlinear.UseJacobian = True
sabino.Solver.Nonlinear.DerivativeEpsilon = 1e-16
sabino.Solver.Nonlinear.StepTol = 1e-30
sabino.Solver.Nonlinear.Globalization = "LineSearch"
sabino.Solver.Linear.KrylovDimension = 70
sabino.Solver.Linear.MaxRestarts = 2

sabino.Solver.Linear.Preconditioner = "PFMG"
sabino.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

# -----------------------------------------------------------------------------
# Distribute inputs and run simulation
# -----------------------------------------------------------------------------

sabino.run()
