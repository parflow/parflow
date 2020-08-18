#  This runs the basic pfmg test case based off of default richards
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
from parflow import Run
pfmg_octree = Run("pfmg_octree", __file__)

pfmg_octree.FileVersion = 4

pfmg_octree.Process.Topology.P = 1
pfmg_octree.Process.Topology.Q = 1
pfmg_octree.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfmg_octree.ComputationalGrid.Lower.X = -10.0
pfmg_octree.ComputationalGrid.Lower.Y = 10.0
pfmg_octree.ComputationalGrid.Lower.Z = 1.0

pfmg_octree.ComputationalGrid.DX = 8.8888888888888893
pfmg_octree.ComputationalGrid.DY = 10.666666666666666
pfmg_octree.ComputationalGrid.DZ = 1.0

pfmg_octree.ComputationalGrid.NX = 10
pfmg_octree.ComputationalGrid.NY = 10
pfmg_octree.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfmg_octree.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfmg_octree.GeomInput.domain_input.InputType = 'Box'
pfmg_octree.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfmg_octree.Geom.domain.Lower.X = -10.0
pfmg_octree.Geom.domain.Lower.Y = 10.0
pfmg_octree.Geom.domain.Lower.Z = 1.0

pfmg_octree.Geom.domain.Upper.X = 150.0
pfmg_octree.Geom.domain.Upper.Y = 170.0
pfmg_octree.Geom.domain.Upper.Z = 9.0

pfmg_octree.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfmg_octree.GeomInput.background_input.InputType = 'Box'
pfmg_octree.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfmg_octree.Geom.background.Lower.X = -99999999.0
pfmg_octree.Geom.background.Lower.Y = -99999999.0
pfmg_octree.Geom.background.Lower.Z = -99999999.0

pfmg_octree.Geom.background.Upper.X = 99999999.0
pfmg_octree.Geom.background.Upper.Y = 99999999.0
pfmg_octree.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
pfmg_octree.GeomInput.source_region_input.InputType = 'Box'
pfmg_octree.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
pfmg_octree.Geom.source_region.Lower.X = 65.56
pfmg_octree.Geom.source_region.Lower.Y = 79.34
pfmg_octree.Geom.source_region.Lower.Z = 4.5

pfmg_octree.Geom.source_region.Upper.X = 74.44
pfmg_octree.Geom.source_region.Upper.Y = 89.99
pfmg_octree.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
pfmg_octree.GeomInput.concen_region_input.InputType = 'Box'
pfmg_octree.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
pfmg_octree.Geom.concen_region.Lower.X = 60.0
pfmg_octree.Geom.concen_region.Lower.Y = 80.0
pfmg_octree.Geom.concen_region.Lower.Z = 4.0

pfmg_octree.Geom.concen_region.Upper.X = 80.0
pfmg_octree.Geom.concen_region.Upper.Y = 100.0
pfmg_octree.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfmg_octree.Geom.Perm.Names = 'background'

pfmg_octree.Geom.background.Perm.Type = 'Constant'
pfmg_octree.Geom.background.Perm.Value = 4.0

pfmg_octree.Perm.TensorType = 'TensorByGeom'

pfmg_octree.Geom.Perm.TensorByGeom.Names = 'background'

pfmg_octree.Geom.background.Perm.TensorValX = 1.0
pfmg_octree.Geom.background.Perm.TensorValY = 1.0
pfmg_octree.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfmg_octree.SpecificStorage.Type = 'Constant'
pfmg_octree.SpecificStorage.GeomNames = 'domain'
pfmg_octree.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfmg_octree.Phase.Names = 'water'

pfmg_octree.Phase.water.Density.Type = 'Constant'
pfmg_octree.Phase.water.Density.Value = 1.0

pfmg_octree.Phase.water.Viscosity.Type = 'Constant'
pfmg_octree.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfmg_octree.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfmg_octree.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfmg_octree.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfmg_octree.TimingInfo.BaseUnit = 1.0
pfmg_octree.TimingInfo.StartCount = 0
pfmg_octree.TimingInfo.StartTime = 0.0
pfmg_octree.TimingInfo.StopTime = 0.010
pfmg_octree.TimingInfo.DumpInterval = -1
pfmg_octree.TimeStep.Type = 'Constant'
pfmg_octree.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfmg_octree.Geom.Porosity.GeomNames = 'background'

pfmg_octree.Geom.background.Porosity.Type = 'Constant'
pfmg_octree.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfmg_octree.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfmg_octree.Phase.RelPerm.Type = 'VanGenuchten'
pfmg_octree.Phase.RelPerm.GeomNames = 'domain'
pfmg_octree.Geom.domain.RelPerm.Alpha = 0.005
pfmg_octree.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfmg_octree.Phase.Saturation.Type = 'VanGenuchten'
pfmg_octree.Phase.Saturation.GeomNames = 'domain'
pfmg_octree.Geom.domain.Saturation.Alpha = 0.005
pfmg_octree.Geom.domain.Saturation.N = 2.0
pfmg_octree.Geom.domain.Saturation.SRes = 0.2
pfmg_octree.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfmg_octree.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfmg_octree.Cycle.Names = 'constant'
pfmg_octree.Cycle.constant.Names = 'alltime'
pfmg_octree.Cycle.constant.alltime.Length = 1
pfmg_octree.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfmg_octree.BCPressure.PatchNames = 'left right front back bottom top'

pfmg_octree.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
pfmg_octree.Patch.left.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.left.BCPressure.RefGeom = 'domain'
pfmg_octree.Patch.left.BCPressure.RefPatch = 'bottom'
pfmg_octree.Patch.left.BCPressure.alltime.Value = 5.0

pfmg_octree.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
pfmg_octree.Patch.right.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.right.BCPressure.RefGeom = 'domain'
pfmg_octree.Patch.right.BCPressure.RefPatch = 'bottom'
pfmg_octree.Patch.right.BCPressure.alltime.Value = 3.0

pfmg_octree.Patch.front.BCPressure.Type = 'FluxConst'
pfmg_octree.Patch.front.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.front.BCPressure.alltime.Value = 0.0

pfmg_octree.Patch.back.BCPressure.Type = 'FluxConst'
pfmg_octree.Patch.back.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.back.BCPressure.alltime.Value = 0.0

pfmg_octree.Patch.bottom.BCPressure.Type = 'FluxConst'
pfmg_octree.Patch.bottom.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.bottom.BCPressure.alltime.Value = 0.0

pfmg_octree.Patch.top.BCPressure.Type = 'FluxConst'
pfmg_octree.Patch.top.BCPressure.Cycle = 'constant'
pfmg_octree.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfmg_octree.TopoSlopesX.Type = 'Constant'
pfmg_octree.TopoSlopesX.GeomNames = 'domain'

pfmg_octree.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfmg_octree.TopoSlopesY.Type = 'Constant'
pfmg_octree.TopoSlopesY.GeomNames = 'domain'

pfmg_octree.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfmg_octree.Mannings.Type = 'Constant'
pfmg_octree.Mannings.GeomNames = 'domain'
pfmg_octree.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfmg_octree.ICPressure.Type = 'HydroStaticPatch'
pfmg_octree.ICPressure.GeomNames = 'domain'
pfmg_octree.Geom.domain.ICPressure.Value = 3.0
pfmg_octree.Geom.domain.ICPressure.RefGeom = 'domain'
pfmg_octree.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfmg_octree.PhaseSources.water.Type = 'Constant'
pfmg_octree.PhaseSources.water.GeomNames = 'background'
pfmg_octree.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfmg_octree.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfmg_octree.Solver = 'Richards'
pfmg_octree.Solver.MaxIter = 5

pfmg_octree.Solver.Nonlinear.MaxIter = 10
pfmg_octree.Solver.Nonlinear.ResidualTol = 1e-9
pfmg_octree.Solver.Nonlinear.EtaChoice = 'EtaConstant'
pfmg_octree.Solver.Nonlinear.EtaValue = 1e-5
pfmg_octree.Solver.Nonlinear.UseJacobian = True
pfmg_octree.Solver.Nonlinear.DerivativeEpsilon = 1e-2

pfmg_octree.Solver.Linear.KrylovDimension = 10

pfmg_octree.Solver.Linear.Preconditioner = 'PFMGOctree'

#pfset Solver.Linear.Preconditioner.PFMGOctree.BoxSizePowerOf2 3                      
pfmg_octree.Solver.Linear.Preconditioner.PFMGOctree.BoxSizePowerOf2 = 2

#pfset Solver.Linear.Preconditioner.PFMG.MaxIter          1
#pfset Solver.Linear.Preconditioner.PFMG.NumPreRelax      100
#pfset Solver.Linear.Preconditioner.PFMG.NumPostRelax     100
#pfset Solver.Linear.Preconditioner.PFMG.Smoother         100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfmg_octree.run()
