#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

from parflow import Run
octree_large_domain = Run("octree_large_domain", __file__)


octree_large_domain.FileVersion = 4

octree_large_domain.Process.Topology.P = 1
octree_large_domain.Process.Topology.Q = 1
octree_large_domain.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
octree_large_domain.ComputationalGrid.Lower.X = -10.0
octree_large_domain.ComputationalGrid.Lower.Y = 10.0
octree_large_domain.ComputationalGrid.Lower.Z = 1.0

octree_large_domain.ComputationalGrid.DX = 20.0
octree_large_domain.ComputationalGrid.DY = 20.0
octree_large_domain.ComputationalGrid.DZ = 1.0

octree_large_domain.ComputationalGrid.NX = 10
octree_large_domain.ComputationalGrid.NY = 10
octree_large_domain.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
octree_large_domain.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
octree_large_domain.GeomInput.domain_input.InputType = 'Box'
octree_large_domain.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
octree_large_domain.Geom.domain.Lower.X = -10.0
octree_large_domain.Geom.domain.Lower.Y = 10.0
octree_large_domain.Geom.domain.Lower.Z = 1.0

octree_large_domain.Geom.domain.Upper.X = 150.0
octree_large_domain.Geom.domain.Upper.Y = 170.0
octree_large_domain.Geom.domain.Upper.Z = 9.0

octree_large_domain.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
octree_large_domain.GeomInput.background_input.InputType = 'Box'
octree_large_domain.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
octree_large_domain.Geom.background.Lower.X = -99999999.0
octree_large_domain.Geom.background.Lower.Y = -99999999.0
octree_large_domain.Geom.background.Lower.Z = -99999999.0

octree_large_domain.Geom.background.Upper.X = 99999999.0
octree_large_domain.Geom.background.Upper.Y = 99999999.0
octree_large_domain.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
octree_large_domain.GeomInput.source_region_input.InputType = 'Box'
octree_large_domain.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
octree_large_domain.Geom.source_region.Lower.X = 65.56
octree_large_domain.Geom.source_region.Lower.Y = 79.34
octree_large_domain.Geom.source_region.Lower.Z = 4.5

octree_large_domain.Geom.source_region.Upper.X = 74.44
octree_large_domain.Geom.source_region.Upper.Y = 89.99
octree_large_domain.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
octree_large_domain.GeomInput.concen_region_input.InputType = 'Box'
octree_large_domain.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
octree_large_domain.Geom.concen_region.Lower.X = 60.0
octree_large_domain.Geom.concen_region.Lower.Y = 80.0
octree_large_domain.Geom.concen_region.Lower.Z = 4.0

octree_large_domain.Geom.concen_region.Upper.X = 80.0
octree_large_domain.Geom.concen_region.Upper.Y = 100.0
octree_large_domain.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
octree_large_domain.Geom.Perm.Names = 'background'

octree_large_domain.Geom.background.Perm.Type = 'Constant'
octree_large_domain.Geom.background.Perm.Value = 4.0

octree_large_domain.Perm.TensorType = 'TensorByGeom'

octree_large_domain.Geom.Perm.TensorByGeom.Names = 'background'

octree_large_domain.Geom.background.Perm.TensorValX = 1.0
octree_large_domain.Geom.background.Perm.TensorValY = 1.0
octree_large_domain.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

octree_large_domain.SpecificStorage.Type = 'Constant'
octree_large_domain.SpecificStorage.GeomNames = 'domain'
octree_large_domain.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

octree_large_domain.Phase.Names = 'water'

octree_large_domain.Phase.water.Density.Type = 'Constant'
octree_large_domain.Phase.water.Density.Value = 1.0

octree_large_domain.Phase.water.Viscosity.Type = 'Constant'
octree_large_domain.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
octree_large_domain.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
octree_large_domain.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

octree_large_domain.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

octree_large_domain.TimingInfo.BaseUnit = 1.0
octree_large_domain.TimingInfo.StartCount = 0
octree_large_domain.TimingInfo.StartTime = 0.0
octree_large_domain.TimingInfo.StopTime = 0.010
octree_large_domain.TimingInfo.DumpInterval = -1
octree_large_domain.TimeStep.Type = 'Constant'
octree_large_domain.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

octree_large_domain.Geom.Porosity.GeomNames = 'background'

octree_large_domain.Geom.background.Porosity.Type = 'Constant'
octree_large_domain.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
octree_large_domain.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

octree_large_domain.Phase.RelPerm.Type = 'VanGenuchten'
octree_large_domain.Phase.RelPerm.GeomNames = 'domain'
octree_large_domain.Geom.domain.RelPerm.Alpha = 0.005
octree_large_domain.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

octree_large_domain.Phase.Saturation.Type = 'VanGenuchten'
octree_large_domain.Phase.Saturation.GeomNames = 'domain'
octree_large_domain.Geom.domain.Saturation.Alpha = 0.005
octree_large_domain.Geom.domain.Saturation.N = 2.0
octree_large_domain.Geom.domain.Saturation.SRes = 0.2
octree_large_domain.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
octree_large_domain.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
octree_large_domain.Cycle.Names = 'constant'
octree_large_domain.Cycle.constant.Names = 'alltime'
octree_large_domain.Cycle.constant.alltime.Length = 1
octree_large_domain.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
octree_large_domain.BCPressure.PatchNames = 'left right front back bottom top'

octree_large_domain.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
octree_large_domain.Patch.left.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.left.BCPressure.RefGeom = 'domain'
octree_large_domain.Patch.left.BCPressure.RefPatch = 'bottom'
octree_large_domain.Patch.left.BCPressure.alltime.Value = 5.0

octree_large_domain.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
octree_large_domain.Patch.right.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.right.BCPressure.RefGeom = 'domain'
octree_large_domain.Patch.right.BCPressure.RefPatch = 'bottom'
octree_large_domain.Patch.right.BCPressure.alltime.Value = 3.0

octree_large_domain.Patch.front.BCPressure.Type = 'FluxConst'
octree_large_domain.Patch.front.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.front.BCPressure.alltime.Value = 0.0

octree_large_domain.Patch.back.BCPressure.Type = 'FluxConst'
octree_large_domain.Patch.back.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.back.BCPressure.alltime.Value = 0.0

octree_large_domain.Patch.bottom.BCPressure.Type = 'FluxConst'
octree_large_domain.Patch.bottom.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.bottom.BCPressure.alltime.Value = 0.0

octree_large_domain.Patch.top.BCPressure.Type = 'FluxConst'
octree_large_domain.Patch.top.BCPressure.Cycle = 'constant'
octree_large_domain.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

octree_large_domain.TopoSlopesX.Type = 'Constant'
octree_large_domain.TopoSlopesX.GeomNames = 'domain'

octree_large_domain.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

octree_large_domain.TopoSlopesY.Type = 'Constant'
octree_large_domain.TopoSlopesY.GeomNames = 'domain'

octree_large_domain.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

octree_large_domain.Mannings.Type = 'Constant'
octree_large_domain.Mannings.GeomNames = 'domain'
octree_large_domain.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

octree_large_domain.ICPressure.Type = 'HydroStaticPatch'
octree_large_domain.ICPressure.GeomNames = 'domain'
octree_large_domain.Geom.domain.ICPressure.Value = 3.0
octree_large_domain.Geom.domain.ICPressure.RefGeom = 'domain'
octree_large_domain.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

octree_large_domain.PhaseSources.water.Type = 'Constant'
octree_large_domain.PhaseSources.water.GeomNames = 'background'
octree_large_domain.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

octree_large_domain.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
octree_large_domain.Solver = 'Richards'
octree_large_domain.Solver.MaxIter = 5

octree_large_domain.Solver.Nonlinear.MaxIter = 10
octree_large_domain.Solver.Nonlinear.ResidualTol = 1e-9
octree_large_domain.Solver.Nonlinear.EtaChoice = 'EtaConstant'
octree_large_domain.Solver.Nonlinear.EtaValue = 1e-5
octree_large_domain.Solver.Nonlinear.UseJacobian = True
octree_large_domain.Solver.Nonlinear.DerivativeEpsilon = 1e-2

octree_large_domain.Solver.Linear.KrylovDimension = 10

octree_large_domain.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


octree_large_domain.run()
