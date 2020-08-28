#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

from parflow import Run
octree_simple = Run("octree_simple", __file__)


octree_simple.FileVersion = 4

octree_simple.Process.Topology.P = 1
octree_simple.Process.Topology.Q = 1
octree_simple.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
octree_simple.ComputationalGrid.Lower.X = -10.0
octree_simple.ComputationalGrid.Lower.Y = 10.0
octree_simple.ComputationalGrid.Lower.Z = 1.0

octree_simple.ComputationalGrid.DX = 20.0
octree_simple.ComputationalGrid.DY = 20.0
octree_simple.ComputationalGrid.DZ = 1.0

octree_simple.ComputationalGrid.NX = 8
octree_simple.ComputationalGrid.NY = 8
octree_simple.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
octree_simple.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
octree_simple.GeomInput.domain_input.InputType = 'Box'
octree_simple.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
octree_simple.Geom.domain.Lower.X = -10.0
octree_simple.Geom.domain.Lower.Y = 10.0
octree_simple.Geom.domain.Lower.Z = 1.0

octree_simple.Geom.domain.Upper.X = 150.0
octree_simple.Geom.domain.Upper.Y = 170.0
octree_simple.Geom.domain.Upper.Z = 9.0

octree_simple.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
octree_simple.GeomInput.background_input.InputType = 'Box'
octree_simple.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
octree_simple.Geom.background.Lower.X = -99999999.0
octree_simple.Geom.background.Lower.Y = -99999999.0
octree_simple.Geom.background.Lower.Z = -99999999.0

octree_simple.Geom.background.Upper.X = 99999999.0
octree_simple.Geom.background.Upper.Y = 99999999.0
octree_simple.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
octree_simple.GeomInput.source_region_input.InputType = 'Box'
octree_simple.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
octree_simple.Geom.source_region.Lower.X = 65.56
octree_simple.Geom.source_region.Lower.Y = 79.34
octree_simple.Geom.source_region.Lower.Z = 4.5

octree_simple.Geom.source_region.Upper.X = 74.44
octree_simple.Geom.source_region.Upper.Y = 89.99
octree_simple.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
octree_simple.GeomInput.concen_region_input.InputType = 'Box'
octree_simple.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
octree_simple.Geom.concen_region.Lower.X = 60.0
octree_simple.Geom.concen_region.Lower.Y = 80.0
octree_simple.Geom.concen_region.Lower.Z = 4.0

octree_simple.Geom.concen_region.Upper.X = 80.0
octree_simple.Geom.concen_region.Upper.Y = 100.0
octree_simple.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
octree_simple.Geom.Perm.Names = 'background'

octree_simple.Geom.background.Perm.Type = 'Constant'
octree_simple.Geom.background.Perm.Value = 4.0

octree_simple.Perm.TensorType = 'TensorByGeom'

octree_simple.Geom.Perm.TensorByGeom.Names = 'background'

octree_simple.Geom.background.Perm.TensorValX = 1.0
octree_simple.Geom.background.Perm.TensorValY = 1.0
octree_simple.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

octree_simple.SpecificStorage.Type = 'Constant'
octree_simple.SpecificStorage.GeomNames = 'domain'
octree_simple.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

octree_simple.Phase.Names = 'water'

octree_simple.Phase.water.Density.Type = 'Constant'
octree_simple.Phase.water.Density.Value = 1.0

octree_simple.Phase.water.Viscosity.Type = 'Constant'
octree_simple.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
octree_simple.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
octree_simple.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

octree_simple.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

octree_simple.TimingInfo.BaseUnit = 1.0
octree_simple.TimingInfo.StartCount = 0
octree_simple.TimingInfo.StartTime = 0.0
octree_simple.TimingInfo.StopTime = 0.010
octree_simple.TimingInfo.DumpInterval = -1
octree_simple.TimeStep.Type = 'Constant'
octree_simple.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

octree_simple.Geom.Porosity.GeomNames = 'background'

octree_simple.Geom.background.Porosity.Type = 'Constant'
octree_simple.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
octree_simple.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

octree_simple.Phase.RelPerm.Type = 'VanGenuchten'
octree_simple.Phase.RelPerm.GeomNames = 'domain'
octree_simple.Geom.domain.RelPerm.Alpha = 0.005
octree_simple.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

octree_simple.Phase.Saturation.Type = 'VanGenuchten'
octree_simple.Phase.Saturation.GeomNames = 'domain'
octree_simple.Geom.domain.Saturation.Alpha = 0.005
octree_simple.Geom.domain.Saturation.N = 2.0
octree_simple.Geom.domain.Saturation.SRes = 0.2
octree_simple.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
octree_simple.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
octree_simple.Cycle.Names = 'constant'
octree_simple.Cycle.constant.Names = 'alltime'
octree_simple.Cycle.constant.alltime.Length = 1
octree_simple.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
octree_simple.BCPressure.PatchNames = 'left right front back bottom top'

octree_simple.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
octree_simple.Patch.left.BCPressure.Cycle = 'constant'
octree_simple.Patch.left.BCPressure.RefGeom = 'domain'
octree_simple.Patch.left.BCPressure.RefPatch = 'bottom'
octree_simple.Patch.left.BCPressure.alltime.Value = 5.0

octree_simple.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
octree_simple.Patch.right.BCPressure.Cycle = 'constant'
octree_simple.Patch.right.BCPressure.RefGeom = 'domain'
octree_simple.Patch.right.BCPressure.RefPatch = 'bottom'
octree_simple.Patch.right.BCPressure.alltime.Value = 3.0

octree_simple.Patch.front.BCPressure.Type = 'FluxConst'
octree_simple.Patch.front.BCPressure.Cycle = 'constant'
octree_simple.Patch.front.BCPressure.alltime.Value = 0.0

octree_simple.Patch.back.BCPressure.Type = 'FluxConst'
octree_simple.Patch.back.BCPressure.Cycle = 'constant'
octree_simple.Patch.back.BCPressure.alltime.Value = 0.0

octree_simple.Patch.bottom.BCPressure.Type = 'FluxConst'
octree_simple.Patch.bottom.BCPressure.Cycle = 'constant'
octree_simple.Patch.bottom.BCPressure.alltime.Value = 0.0

octree_simple.Patch.top.BCPressure.Type = 'FluxConst'
octree_simple.Patch.top.BCPressure.Cycle = 'constant'
octree_simple.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

octree_simple.TopoSlopesX.Type = 'Constant'
octree_simple.TopoSlopesX.GeomNames = 'domain'

octree_simple.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

octree_simple.TopoSlopesY.Type = 'Constant'
octree_simple.TopoSlopesY.GeomNames = 'domain'

octree_simple.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

octree_simple.Mannings.Type = 'Constant'
octree_simple.Mannings.GeomNames = 'domain'
octree_simple.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

octree_simple.ICPressure.Type = 'HydroStaticPatch'
octree_simple.ICPressure.GeomNames = 'domain'
octree_simple.Geom.domain.ICPressure.Value = 3.0
octree_simple.Geom.domain.ICPressure.RefGeom = 'domain'
octree_simple.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

octree_simple.PhaseSources.water.Type = 'Constant'
octree_simple.PhaseSources.water.GeomNames = 'background'
octree_simple.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

octree_simple.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
octree_simple.Solver = 'Richards'
octree_simple.Solver.MaxIter = 5

octree_simple.Solver.Nonlinear.MaxIter = 10
octree_simple.Solver.Nonlinear.ResidualTol = 1e-9
octree_simple.Solver.Nonlinear.EtaChoice = 'EtaConstant'
octree_simple.Solver.Nonlinear.EtaValue = 1e-5
octree_simple.Solver.Nonlinear.UseJacobian = True
octree_simple.Solver.Nonlinear.DerivativeEpsilon = 1e-2

octree_simple.Solver.Linear.KrylovDimension = 10

octree_simple.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


octree_simple.run()
