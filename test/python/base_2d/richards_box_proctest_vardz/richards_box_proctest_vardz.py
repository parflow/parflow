#----------------------------------------------------------------------------
# This runs a test case with the Richards' solver
# with a simple flow domain and different BCs on the top.
# The domain geometry is purposefully smaller than the computational grid
# making more than 1/2 the domain inactive in Y.  When run with topology
# 1 2 1 this will test PF behavior for inactive processors, for different BCs
# and solver configurations.
#----------------------------------------------------------------------------

from parflow import Run

richards_box_proctest_vardz = Run("richards_box_proctest_vardz", __file__)

#---------------------------------------------------------

richards_box_proctest_vardz.FileVersion = 4

richards_box_proctest_vardz.Process.Topology.P = 1
richards_box_proctest_vardz.Process.Topology.Q = 1
richards_box_proctest_vardz.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

richards_box_proctest_vardz.ComputationalGrid.Lower.X = 0.0
richards_box_proctest_vardz.ComputationalGrid.Lower.Y = 0.0
richards_box_proctest_vardz.ComputationalGrid.Lower.Z = 0.0

richards_box_proctest_vardz.ComputationalGrid.DX = 1.0
richards_box_proctest_vardz.ComputationalGrid.DY = 1.0
richards_box_proctest_vardz.ComputationalGrid.DZ = 1.0

richards_box_proctest_vardz.ComputationalGrid.NX = 20
richards_box_proctest_vardz.ComputationalGrid.NY = 50
richards_box_proctest_vardz.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

richards_box_proctest_vardz.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

richards_box_proctest_vardz.GeomInput.domain_input.InputType = 'Box'
richards_box_proctest_vardz.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

richards_box_proctest_vardz.Geom.domain.Lower.X = 0.0
richards_box_proctest_vardz.Geom.domain.Lower.Y = 0.0
richards_box_proctest_vardz.Geom.domain.Lower.Z = 0.0

richards_box_proctest_vardz.Geom.domain.Upper.X = 20.0
richards_box_proctest_vardz.Geom.domain.Upper.Y = 20.0
richards_box_proctest_vardz.Geom.domain.Upper.Z = 20.0

richards_box_proctest_vardz.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# variable dz assignments
#---------------------------------------------------------

richards_box_proctest_vardz.Solver.Nonlinear.VariableDz = True
richards_box_proctest_vardz.dzScale.GeomNames = 'domain'
richards_box_proctest_vardz.dzScale.Type = 'nzList'
richards_box_proctest_vardz.dzScale.nzListNumber = 20
richards_box_proctest_vardz.Cell.l0.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l1.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l2.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l3.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l4.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l5.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l6.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l7.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l8.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l9.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l10.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l11.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l12.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l13.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l14.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l15.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l16.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l17.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l18.dzScale.Value = 1.0
richards_box_proctest_vardz.Cell.l19.dzScale.Value = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Geom.Perm.Names = 'domain'
richards_box_proctest_vardz.Geom.domain.Perm.Type = 'Constant'
richards_box_proctest_vardz.Geom.domain.Perm.Value = 1.0

richards_box_proctest_vardz.Perm.TensorType = 'TensorByGeom'

richards_box_proctest_vardz.Geom.Perm.TensorByGeom.Names = 'domain'

richards_box_proctest_vardz.Geom.domain.Perm.TensorValX = 1.0
richards_box_proctest_vardz.Geom.domain.Perm.TensorValY = 1.0
richards_box_proctest_vardz.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.SpecificStorage.Type = 'Constant'
richards_box_proctest_vardz.SpecificStorage.GeomNames = 'domain'
richards_box_proctest_vardz.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Phase.Names = 'water'

richards_box_proctest_vardz.Phase.water.Density.Type = 'Constant'
richards_box_proctest_vardz.Phase.water.Density.Value = 1.0

richards_box_proctest_vardz.Phase.water.Viscosity.Type = 'Constant'
richards_box_proctest_vardz.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.TimingInfo.BaseUnit = 10.
richards_box_proctest_vardz.TimingInfo.StartCount = 0
richards_box_proctest_vardz.TimingInfo.StartTime = 0.0
richards_box_proctest_vardz.TimingInfo.StopTime = 100.0
richards_box_proctest_vardz.TimingInfo.DumpInterval = 10.0
richards_box_proctest_vardz.TimeStep.Type = 'Constant'
richards_box_proctest_vardz.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Geom.Porosity.GeomNames = 'domain'
richards_box_proctest_vardz.Geom.domain.Porosity.Type = 'Constant'
richards_box_proctest_vardz.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Phase.RelPerm.Type = 'VanGenuchten'
richards_box_proctest_vardz.Phase.RelPerm.GeomNames = 'domain'
richards_box_proctest_vardz.Geom.domain.RelPerm.Alpha = 2.0
richards_box_proctest_vardz.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_box_proctest_vardz.Phase.Saturation.Type = 'VanGenuchten'
richards_box_proctest_vardz.Phase.Saturation.GeomNames = 'domain'
richards_box_proctest_vardz.Geom.domain.Saturation.Alpha = 2.0
richards_box_proctest_vardz.Geom.domain.Saturation.N = 2.0
richards_box_proctest_vardz.Geom.domain.Saturation.SRes = 0.1
richards_box_proctest_vardz.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# No Flow Barrier
#---------------------------------------------------------

richards_box_proctest_vardz.Solver.Nonlinear.FlowBarrierX = False

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Cycle.Names = 'constant'
richards_box_proctest_vardz.Cycle.constant.Names = 'alltime'
richards_box_proctest_vardz.Cycle.constant.alltime.Length = 1
richards_box_proctest_vardz.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.BCPressure.PatchNames = 'left right front back bottom top'

richards_box_proctest_vardz.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
richards_box_proctest_vardz.Patch.left.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.left.BCPressure.RefGeom = 'domain'
richards_box_proctest_vardz.Patch.left.BCPressure.RefPatch = 'bottom'
richards_box_proctest_vardz.Patch.left.BCPressure.alltime.Value = 11.0

richards_box_proctest_vardz.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
richards_box_proctest_vardz.Patch.right.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.right.BCPressure.RefGeom = 'domain'
richards_box_proctest_vardz.Patch.right.BCPressure.RefPatch = 'bottom'
richards_box_proctest_vardz.Patch.right.BCPressure.alltime.Value = 15.0

richards_box_proctest_vardz.Patch.front.BCPressure.Type = 'FluxConst'
richards_box_proctest_vardz.Patch.front.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.front.BCPressure.alltime.Value = 0.0

richards_box_proctest_vardz.Patch.back.BCPressure.Type = 'FluxConst'
richards_box_proctest_vardz.Patch.back.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.back.BCPressure.alltime.Value = 0.0

richards_box_proctest_vardz.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_box_proctest_vardz.Patch.bottom.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.bottom.BCPressure.alltime.Value = 0.0

# used to cycle different BCs on the top of the domain, even with no
# overland flow
richards_box_proctest_vardz.Patch.top.BCPressure.Type = 'FluxConst'
richards_box_proctest_vardz.Patch.top.BCPressure.Type = 'OverlandFlow'
richards_box_proctest_vardz.Patch.top.BCPressure.Type = 'OverlandKinematic'

richards_box_proctest_vardz.Patch.top.BCPressure.Cycle = 'constant'
richards_box_proctest_vardz.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_box_proctest_vardz.TopoSlopesX.Type = 'Constant'
richards_box_proctest_vardz.TopoSlopesX.GeomNames = 'domain'
richards_box_proctest_vardz.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_box_proctest_vardz.TopoSlopesY.Type = 'Constant'
richards_box_proctest_vardz.TopoSlopesY.GeomNames = 'domain'
richards_box_proctest_vardz.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

richards_box_proctest_vardz.Mannings.Type = 'Constant'
richards_box_proctest_vardz.Mannings.GeomNames = 'domain'
richards_box_proctest_vardz.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_box_proctest_vardz.ICPressure.Type = 'HydroStaticPatch'
richards_box_proctest_vardz.ICPressure.GeomNames = 'domain'
richards_box_proctest_vardz.Geom.domain.ICPressure.Value = 13.0
richards_box_proctest_vardz.Geom.domain.ICPressure.RefGeom = 'domain'
richards_box_proctest_vardz.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.PhaseSources.water.Type = 'Constant'
richards_box_proctest_vardz.PhaseSources.water.GeomNames = 'domain'
richards_box_proctest_vardz.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.Solver = 'Richards'
richards_box_proctest_vardz.Solver.MaxIter = 50000

richards_box_proctest_vardz.Solver.Nonlinear.MaxIter = 100
richards_box_proctest_vardz.Solver.Nonlinear.ResidualTol = 1e-7

richards_box_proctest_vardz.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_box_proctest_vardz.Solver.Nonlinear.EtaValue = 1e-2

# used to test analytical and FD jacobian combinations
richards_box_proctest_vardz.Solver.Nonlinear.UseJacobian = True

richards_box_proctest_vardz.Solver.Nonlinear.DerivativeEpsilon = 1e-14

richards_box_proctest_vardz.Solver.Linear.KrylovDimension = 100

# used to test different linear preconditioners
richards_box_proctest_vardz.Solver.Linear.Preconditioner = 'PFMG'

richards_box_proctest_vardz.UseClustering = False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_box_proctest_vardz.run()
