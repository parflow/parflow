#  This runs the basic pfmg test case based off of default richards
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
from parflow import Run
pfmg_galerkin = Run("pfmg_galerkin", __file__)

pfmg_galerkin.FileVersion = 4

pfmg_galerkin.Process.Topology.P = 1
pfmg_galerkin.Process.Topology.Q = 1
pfmg_galerkin.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfmg_galerkin.ComputationalGrid.Lower.X = -10.0
pfmg_galerkin.ComputationalGrid.Lower.Y = 10.0
pfmg_galerkin.ComputationalGrid.Lower.Z = 1.0

pfmg_galerkin.ComputationalGrid.DX = 8.8888888888888893
pfmg_galerkin.ComputationalGrid.DY = 10.666666666666666
pfmg_galerkin.ComputationalGrid.DZ = 1.0

pfmg_galerkin.ComputationalGrid.NX = 10
pfmg_galerkin.ComputationalGrid.NY = 10
pfmg_galerkin.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfmg_galerkin.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfmg_galerkin.GeomInput.domain_input.InputType = 'Box'
pfmg_galerkin.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfmg_galerkin.Geom.domain.Lower.X = -10.0
pfmg_galerkin.Geom.domain.Lower.Y = 10.0
pfmg_galerkin.Geom.domain.Lower.Z = 1.0

pfmg_galerkin.Geom.domain.Upper.X = 150.0
pfmg_galerkin.Geom.domain.Upper.Y = 170.0
pfmg_galerkin.Geom.domain.Upper.Z = 9.0

pfmg_galerkin.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfmg_galerkin.GeomInput.background_input.InputType = 'Box'
pfmg_galerkin.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfmg_galerkin.Geom.background.Lower.X = -99999999.0
pfmg_galerkin.Geom.background.Lower.Y = -99999999.0
pfmg_galerkin.Geom.background.Lower.Z = -99999999.0

pfmg_galerkin.Geom.background.Upper.X = 99999999.0
pfmg_galerkin.Geom.background.Upper.Y = 99999999.0
pfmg_galerkin.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
pfmg_galerkin.GeomInput.source_region_input.InputType = 'Box'
pfmg_galerkin.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
pfmg_galerkin.Geom.source_region.Lower.X = 65.56
pfmg_galerkin.Geom.source_region.Lower.Y = 79.34
pfmg_galerkin.Geom.source_region.Lower.Z = 4.5

pfmg_galerkin.Geom.source_region.Upper.X = 74.44
pfmg_galerkin.Geom.source_region.Upper.Y = 89.99
pfmg_galerkin.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
pfmg_galerkin.GeomInput.concen_region_input.InputType = 'Box'
pfmg_galerkin.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
pfmg_galerkin.Geom.concen_region.Lower.X = 60.0
pfmg_galerkin.Geom.concen_region.Lower.Y = 80.0
pfmg_galerkin.Geom.concen_region.Lower.Z = 4.0

pfmg_galerkin.Geom.concen_region.Upper.X = 80.0
pfmg_galerkin.Geom.concen_region.Upper.Y = 100.0
pfmg_galerkin.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfmg_galerkin.Geom.Perm.Names = 'background'

pfmg_galerkin.Geom.background.Perm.Type = 'Constant'
pfmg_galerkin.Geom.background.Perm.Value = 4.0

pfmg_galerkin.Perm.TensorType = 'TensorByGeom'

pfmg_galerkin.Geom.Perm.TensorByGeom.Names = 'background'

pfmg_galerkin.Geom.background.Perm.TensorValX = 1.0
pfmg_galerkin.Geom.background.Perm.TensorValY = 1.0
pfmg_galerkin.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfmg_galerkin.SpecificStorage.Type = 'Constant'
pfmg_galerkin.SpecificStorage.GeomNames = 'domain'
pfmg_galerkin.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfmg_galerkin.Phase.Names = 'water'

pfmg_galerkin.Phase.water.Density.Type = 'Constant'
pfmg_galerkin.Phase.water.Density.Value = 1.0

pfmg_galerkin.Phase.water.Viscosity.Type = 'Constant'
pfmg_galerkin.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfmg_galerkin.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfmg_galerkin.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfmg_galerkin.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfmg_galerkin.TimingInfo.BaseUnit = 1.0
pfmg_galerkin.TimingInfo.StartCount = 0
pfmg_galerkin.TimingInfo.StartTime = 0.0
pfmg_galerkin.TimingInfo.StopTime = 0.010
pfmg_galerkin.TimingInfo.DumpInterval = -1
pfmg_galerkin.TimeStep.Type = 'Constant'
pfmg_galerkin.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfmg_galerkin.Geom.Porosity.GeomNames = 'background'

pfmg_galerkin.Geom.background.Porosity.Type = 'Constant'
pfmg_galerkin.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfmg_galerkin.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfmg_galerkin.Phase.RelPerm.Type = 'VanGenuchten'
pfmg_galerkin.Phase.RelPerm.GeomNames = 'domain'
pfmg_galerkin.Geom.domain.RelPerm.Alpha = 0.005
pfmg_galerkin.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfmg_galerkin.Phase.Saturation.Type = 'VanGenuchten'
pfmg_galerkin.Phase.Saturation.GeomNames = 'domain'
pfmg_galerkin.Geom.domain.Saturation.Alpha = 0.005
pfmg_galerkin.Geom.domain.Saturation.N = 2.0
pfmg_galerkin.Geom.domain.Saturation.SRes = 0.2
pfmg_galerkin.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfmg_galerkin.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfmg_galerkin.Cycle.Names = 'constant'
pfmg_galerkin.Cycle.constant.Names = 'alltime'
pfmg_galerkin.Cycle.constant.alltime.Length = 1
pfmg_galerkin.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfmg_galerkin.BCPressure.PatchNames = 'left right front back bottom top'

pfmg_galerkin.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
pfmg_galerkin.Patch.left.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.left.BCPressure.RefGeom = 'domain'
pfmg_galerkin.Patch.left.BCPressure.RefPatch = 'bottom'
pfmg_galerkin.Patch.left.BCPressure.alltime.Value = 5.0

pfmg_galerkin.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
pfmg_galerkin.Patch.right.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.right.BCPressure.RefGeom = 'domain'
pfmg_galerkin.Patch.right.BCPressure.RefPatch = 'bottom'
pfmg_galerkin.Patch.right.BCPressure.alltime.Value = 3.0

pfmg_galerkin.Patch.front.BCPressure.Type = 'FluxConst'
pfmg_galerkin.Patch.front.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.front.BCPressure.alltime.Value = 0.0

pfmg_galerkin.Patch.back.BCPressure.Type = 'FluxConst'
pfmg_galerkin.Patch.back.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.back.BCPressure.alltime.Value = 0.0

pfmg_galerkin.Patch.bottom.BCPressure.Type = 'FluxConst'
pfmg_galerkin.Patch.bottom.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.bottom.BCPressure.alltime.Value = 0.0

pfmg_galerkin.Patch.top.BCPressure.Type = 'FluxConst'
pfmg_galerkin.Patch.top.BCPressure.Cycle = 'constant'
pfmg_galerkin.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfmg_galerkin.TopoSlopesX.Type = 'Constant'
pfmg_galerkin.TopoSlopesX.GeomNames = 'domain'

pfmg_galerkin.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfmg_galerkin.TopoSlopesY.Type = 'Constant'
pfmg_galerkin.TopoSlopesY.GeomNames = 'domain'

pfmg_galerkin.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfmg_galerkin.Mannings.Type = 'Constant'
pfmg_galerkin.Mannings.GeomNames = 'domain'
pfmg_galerkin.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfmg_galerkin.ICPressure.Type = 'HydroStaticPatch'
pfmg_galerkin.ICPressure.GeomNames = 'domain'
pfmg_galerkin.Geom.domain.ICPressure.Value = 3.0
pfmg_galerkin.Geom.domain.ICPressure.RefGeom = 'domain'
pfmg_galerkin.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfmg_galerkin.PhaseSources.water.Type = 'Constant'
pfmg_galerkin.PhaseSources.water.GeomNames = 'background'
pfmg_galerkin.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfmg_galerkin.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfmg_galerkin.Solver = 'Richards'
pfmg_galerkin.Solver.MaxIter = 5

pfmg_galerkin.Solver.Nonlinear.MaxIter = 10
pfmg_galerkin.Solver.Nonlinear.ResidualTol = 1e-9
pfmg_galerkin.Solver.Nonlinear.EtaChoice = 'EtaConstant'
pfmg_galerkin.Solver.Nonlinear.EtaValue = 1e-5
pfmg_galerkin.Solver.Nonlinear.UseJacobian = True
pfmg_galerkin.Solver.Nonlinear.DerivativeEpsilon = 1e-2

pfmg_galerkin.Solver.Linear.KrylovDimension = 10

pfmg_galerkin.Solver.Linear.Preconditioner = 'PFMG'
pfmg_galerkin.Solver.Linear.Preconditioner.PFMG.Smoother = 'WJacobi'
pfmg_galerkin.Solver.Linear.Preconditioner.PFMG.RAPType = 'Galerkin'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfmg_galerkin.run()
