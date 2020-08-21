#  Testing pfset python function for setting keys that aren't in the library

from parflow import Run

pfset_test = Run('default_richards', __file__)


pfset_test.Process.Topology.P = 1
pfset_test.Process.Topology.Q = 1
pfset_test.Process.Topology.R = 1

pfset_test.pfset(key='NewKeyTest', value=1)

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset_test.ComputationalGrid.Lower.X = -10.0
pfset_test.ComputationalGrid.Lower.Y = 10.0
pfset_test.ComputationalGrid.Lower.Z = 1.0

pfset_test.ComputationalGrid.DX = 8.8888888888888893
pfset_test.ComputationalGrid.DY = 10.666666666666666
pfset_test.ComputationalGrid.DZ = 1.0

pfset_test.ComputationalGrid.NX = 10
pfset_test.ComputationalGrid.NY = 10
pfset_test.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset_test.GeomInput.Names = "domain_input background_input source_region_input concen_region_input"


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset_test.GeomInput.domain_input.InputType = 'Box'
pfset_test.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset_test.Geom.domain.Lower.X = -10.0
pfset_test.Geom.domain.Lower.Y = 10.0
pfset_test.Geom.domain.Lower.Z = 1.0

pfset_test.Geom.domain.Upper.X = 150.0
pfset_test.Geom.domain.Upper.Y = 170.0
pfset_test.Geom.domain.Upper.Z = 9.0

pfset_test.Geom.domain.Patches = "left right front back bottom top"

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfset_test.GeomInput.background_input.InputType = 'Box'
pfset_test.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfset_test.Geom.background.Lower.X = -99999999.0
pfset_test.Geom.background.Lower.Y = -99999999.0
pfset_test.Geom.background.Lower.Z = -99999999.0

pfset_test.Geom.background.Upper.X = 99999999.0
pfset_test.Geom.background.Upper.Y = 99999999.0
pfset_test.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
pfset_test.GeomInput.source_region_input.InputType = 'Box'
pfset_test.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
pfset_test.Geom.source_region.Lower.X = 65.56
pfset_test.Geom.source_region.Lower.Y = 79.34
pfset_test.Geom.source_region.Lower.Z = 4.5

pfset_test.Geom.source_region.Upper.X = 74.44
pfset_test.Geom.source_region.Upper.Y = 89.99
pfset_test.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
pfset_test.GeomInput.concen_region_input.InputType = 'Box'
pfset_test.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
pfset_test.Geom.concen_region.Lower.X = 60.0
pfset_test.Geom.concen_region.Lower.Y = 80.0
pfset_test.Geom.concen_region.Lower.Z = 4.0

pfset_test.Geom.concen_region.Upper.X = 80.0
pfset_test.Geom.concen_region.Upper.Y = 100.0
pfset_test.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset_test.Geom.Perm.Names = "background"

pfset_test.Geom.background.Perm.Type = 'Constant'
pfset_test.Geom.background.Perm.Value = 4.0

pfset_test.Perm.TensorType = 'TensorByGeom'

pfset_test.Geom.Perm.TensorByGeom.Names = "background"

pfset_test.Geom.background.Perm.TensorValX = 1.0
pfset_test.Geom.background.Perm.TensorValY = 1.0
pfset_test.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset_test.SpecificStorage.Type = 'Constant'
pfset_test.SpecificStorage.GeomNames = "domain"
pfset_test.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset_test.Phase.Names = "water"

pfset_test.Phase.water.Density.Type = 'Constant'
pfset_test.Phase.water.Density.Value = 1.0

pfset_test.Phase.water.Viscosity.Type = 'Constant'
pfset_test.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset_test.Contaminants.Names = ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset_test.Geom.Retardation.GeomNames = ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset_test.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset_test.TimingInfo.BaseUnit = 1.0
pfset_test.TimingInfo.StartCount = 0
pfset_test.TimingInfo.StartTime = 0.0
pfset_test.TimingInfo.StopTime = 0.010
pfset_test.TimingInfo.DumpInterval = -1
pfset_test.TimeStep.Type = 'Constant'
pfset_test.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset_test.Geom.Porosity.GeomNames = 'background'

pfset_test.Geom.background.Porosity.Type = 'Constant'
pfset_test.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset_test.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset_test.Phase.RelPerm.Type = 'VanGenuchten'
pfset_test.Phase.RelPerm.GeomNames = 'domain'
pfset_test.Geom.domain.RelPerm.Alpha = 0.005
# pfset_test.Geom.domain.RelPerm.Alpha.FileName = 'alpha_file.pfb'
pfset_test.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset_test.Phase.Saturation.Type = 'VanGenuchten'
pfset_test.Phase.Saturation.GeomNames = 'domain'
pfset_test.Geom.domain.Saturation.Alpha = 0.005
pfset_test.Geom.domain.Saturation.N = 2.0
pfset_test.Geom.domain.Saturation.SRes = 0.2
pfset_test.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset_test.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset_test.Cycle.Names = 'constant'
pfset_test.Cycle.constant.Names = "alltime"
pfset_test.Cycle.constant.alltime.Length = 1
pfset_test.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset_test.BCPressure.PatchNames = "left right front back bottom top"

pfset_test.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
pfset_test.Patch.left.BCPressure.Cycle = "constant"
pfset_test.Patch.left.BCPressure.RefGeom = 'domain'
pfset_test.Patch.left.BCPressure.RefPatch = 'bottom'
pfset_test.Patch.left.BCPressure.alltime.Value = 5.0

pfset_test.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
pfset_test.Patch.right.BCPressure.Cycle = "constant"
pfset_test.Patch.right.BCPressure.RefGeom = 'domain'
pfset_test.Patch.right.BCPressure.RefPatch = 'bottom'
pfset_test.Patch.right.BCPressure.alltime.Value = 3.0

pfset_test.Patch.front.BCPressure.Type = 'FluxConst'
pfset_test.Patch.front.BCPressure.Cycle = "constant"
pfset_test.Patch.front.BCPressure.alltime.Value = 0.0

pfset_test.Patch.back.BCPressure.Type = 'FluxConst'
pfset_test.Patch.back.BCPressure.Cycle = "constant"
pfset_test.Patch.back.BCPressure.alltime.Value = 0.0

pfset_test.Patch.bottom.BCPressure.Type = 'FluxConst'
pfset_test.Patch.bottom.BCPressure.Cycle = "constant"
pfset_test.Patch.bottom.BCPressure.alltime.Value = 0.0

pfset_test.Patch.top.BCPressure.Type = 'FluxConst'
pfset_test.Patch.top.BCPressure.Cycle = "constant"
pfset_test.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset_test.TopoSlopesX.Type = "Constant"
# seb added domain otherwise failing after
pfset_test.TopoSlopesX.GeomNames = "domain"

pfset_test.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset_test.TopoSlopesY.Type = "Constant"
# seb added domain otherwise failing after
pfset_test.TopoSlopesY.GeomNames = "domain"

pfset_test.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

pfset_test.Mannings.Type = "Constant"
# seb added domain otherwise failing after
pfset_test.Mannings.GeomNames = "domain"
pfset_test.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset_test.ICPressure.Type = 'HydroStaticPatch'
pfset_test.ICPressure.GeomNames = 'domain'
pfset_test.Geom.domain.ICPressure.Value = 3.0
pfset_test.Geom.domain.ICPressure.RefGeom = 'domain'
pfset_test.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset_test.PhaseSources.water.Type = 'Constant'
pfset_test.PhaseSources.water.GeomNames = 'background'
pfset_test.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset_test.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset_test.Solver = 'Richards'
pfset_test.Solver.MaxIter = 5

pfset_test.Solver.Nonlinear.MaxIter = 10
pfset_test.Solver.Nonlinear.ResidualTol = 1e-9
pfset_test.Solver.Nonlinear.EtaChoice = 'EtaConstant'
pfset_test.Solver.Nonlinear.EtaValue = 1e-5
pfset_test.Solver.Nonlinear.UseJacobian = True
pfset_test.Solver.Nonlinear.DerivativeEpsilon = 1e-2

pfset_test.Solver.Linear.KrylovDimension = 10

pfset_test.Solver.Linear.Preconditioner = 'PFMG'

# These keys are set in the ParFlow example file but are apparently not used
# pfset_test.Solver.Linear.Preconditioner.MGSemi.MaxIter  = 1
# pfset_test.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100


pfset_test.run()
