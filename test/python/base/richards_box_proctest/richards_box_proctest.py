#---------------------------------------------------------
# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.
#---------------------------------------------------------

from parflow import Run

richards_box_proctest = Run("richards_box_proctest", __file__)

#---------------------------------------------------------

richards_box_proctest.FileVersion = 4

richards_box_proctest.Process.Topology.P = 1
richards_box_proctest.Process.Topology.Q = 1
richards_box_proctest.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

richards_box_proctest.ComputationalGrid.Lower.X = 0.0
richards_box_proctest.ComputationalGrid.Lower.Y = 0.0
richards_box_proctest.ComputationalGrid.Lower.Z = 0.0

richards_box_proctest.ComputationalGrid.DX = 1.0
richards_box_proctest.ComputationalGrid.DY = 1.0
richards_box_proctest.ComputationalGrid.DZ = 1.0

richards_box_proctest.ComputationalGrid.NX = 20
richards_box_proctest.ComputationalGrid.NY = 50
richards_box_proctest.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

richards_box_proctest.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

richards_box_proctest.GeomInput.domain_input.InputType = 'Box'
richards_box_proctest.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

richards_box_proctest.Geom.domain.Lower.X = 0.0
richards_box_proctest.Geom.domain.Lower.Y = 0.0
richards_box_proctest.Geom.domain.Lower.Z = 0.0

richards_box_proctest.Geom.domain.Upper.X = 20.0
richards_box_proctest.Geom.domain.Upper.Y = 20.0
richards_box_proctest.Geom.domain.Upper.Z = 20.0

richards_box_proctest.Geom.domain.Patches = 'left right front back bottom top'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_box_proctest.Geom.Perm.Names = 'domain'
richards_box_proctest.Geom.domain.Perm.Type = 'Constant'
richards_box_proctest.Geom.domain.Perm.Value = 1.0

richards_box_proctest.Perm.TensorType = 'TensorByGeom'

richards_box_proctest.Geom.Perm.TensorByGeom.Names = 'domain'

richards_box_proctest.Geom.domain.Perm.TensorValX = 1.0
richards_box_proctest.Geom.domain.Perm.TensorValY = 1.0
richards_box_proctest.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_box_proctest.SpecificStorage.Type = 'Constant'
richards_box_proctest.SpecificStorage.GeomNames = 'domain'
richards_box_proctest.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_box_proctest.Phase.Names = 'water'

richards_box_proctest.Phase.water.Density.Type = 'Constant'
richards_box_proctest.Phase.water.Density.Value = 1.0

richards_box_proctest.Phase.water.Viscosity.Type = 'Constant'
richards_box_proctest.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

richards_box_proctest.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

richards_box_proctest.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_box_proctest.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_box_proctest.TimingInfo.BaseUnit = 10.
richards_box_proctest.TimingInfo.StartCount = 0
richards_box_proctest.TimingInfo.StartTime = 0.0
richards_box_proctest.TimingInfo.StopTime = 100.0
richards_box_proctest.TimingInfo.DumpInterval = 10.0
richards_box_proctest.TimeStep.Type = 'Constant'
richards_box_proctest.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_box_proctest.Geom.Porosity.GeomNames = 'domain'
richards_box_proctest.Geom.domain.Porosity.Type = 'Constant'
richards_box_proctest.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

richards_box_proctest.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_box_proctest.Phase.RelPerm.Type = 'VanGenuchten'
richards_box_proctest.Phase.RelPerm.GeomNames = 'domain'
richards_box_proctest.Geom.domain.RelPerm.Alpha = 2.0
richards_box_proctest.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_box_proctest.Phase.Saturation.Type = 'VanGenuchten'
richards_box_proctest.Phase.Saturation.GeomNames = 'domain'
richards_box_proctest.Geom.domain.Saturation.Alpha = 2.0
richards_box_proctest.Geom.domain.Saturation.N = 2.0
richards_box_proctest.Geom.domain.Saturation.SRes = 0.1
richards_box_proctest.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# Flow Barrier in X between cells 10 and 11 in all Z
#---------------------------------------------------------

richards_box_proctest.Solver.Nonlinear.FlowBarrierX = False

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

richards_box_proctest.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

richards_box_proctest.Cycle.Names = 'constant'
richards_box_proctest.Cycle.constant.Names = 'alltime'
richards_box_proctest.Cycle.constant.alltime.Length = 1
richards_box_proctest.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_box_proctest.BCPressure.PatchNames = 'left right front back bottom top'

richards_box_proctest.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
richards_box_proctest.Patch.left.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.left.BCPressure.RefGeom = 'domain'
richards_box_proctest.Patch.left.BCPressure.RefPatch = 'bottom'
richards_box_proctest.Patch.left.BCPressure.alltime.Value = 11.0

richards_box_proctest.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
richards_box_proctest.Patch.right.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.right.BCPressure.RefGeom = 'domain'
richards_box_proctest.Patch.right.BCPressure.RefPatch = 'bottom'
richards_box_proctest.Patch.right.BCPressure.alltime.Value = 15.0

richards_box_proctest.Patch.front.BCPressure.Type = 'FluxConst'
richards_box_proctest.Patch.front.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.front.BCPressure.alltime.Value = 0.0

richards_box_proctest.Patch.back.BCPressure.Type = 'FluxConst'
richards_box_proctest.Patch.back.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.back.BCPressure.alltime.Value = 0.0

richards_box_proctest.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_box_proctest.Patch.bottom.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.bottom.BCPressure.alltime.Value = 0.0

richards_box_proctest.Patch.top.BCPressure.Type = 'FluxConst'
richards_box_proctest.Patch.top.BCPressure.Cycle = 'constant'
richards_box_proctest.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_box_proctest.TopoSlopesX.Type = 'Constant'
richards_box_proctest.TopoSlopesX.GeomNames = 'domain'
richards_box_proctest.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_box_proctest.TopoSlopesY.Type = 'Constant'
richards_box_proctest.TopoSlopesY.GeomNames = 'domain'
richards_box_proctest.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

richards_box_proctest.Mannings.Type = 'Constant'
richards_box_proctest.Mannings.GeomNames = 'domain'
richards_box_proctest.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_box_proctest.ICPressure.Type = 'HydroStaticPatch'
richards_box_proctest.ICPressure.GeomNames = 'domain'
richards_box_proctest.Geom.domain.ICPressure.Value = 13.0
richards_box_proctest.Geom.domain.ICPressure.RefGeom = 'domain'
richards_box_proctest.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_box_proctest.PhaseSources.water.Type = 'Constant'
richards_box_proctest.PhaseSources.water.GeomNames = 'domain'
richards_box_proctest.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_box_proctest.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

richards_box_proctest.Solver = 'Richards'
richards_box_proctest.Solver.MaxIter = 50000

richards_box_proctest.Solver.Nonlinear.MaxIter = 100
richards_box_proctest.Solver.Nonlinear.ResidualTol = 1e-6
richards_box_proctest.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_box_proctest.Solver.Nonlinear.EtaValue = 1e-2
richards_box_proctest.Solver.Nonlinear.UseJacobian = True

richards_box_proctest.Solver.Nonlinear.DerivativeEpsilon = 1e-12

richards_box_proctest.Solver.Linear.KrylovDimension = 100

richards_box_proctest.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_box_proctest.run()
