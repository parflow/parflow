# Runs a simple sand draining problem, rectangular domain
# with variable dz and a heterogenous subsurface with different K the top and bottom layers

from parflow import Run
var_dz_1D = Run("var_dz_1D", __file__)

var_dz_1D.FileVersion = 4

var_dz_1D.Process.Topology.P = 1
var_dz_1D.Process.Topology.Q = 1
var_dz_1D.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
var_dz_1D.ComputationalGrid.Lower.X = 0.0
var_dz_1D.ComputationalGrid.Lower.Y = 0.0
var_dz_1D.ComputationalGrid.Lower.Z = 0.0

var_dz_1D.ComputationalGrid.DX = 1.0
var_dz_1D.ComputationalGrid.DY = 1.0
var_dz_1D.ComputationalGrid.DZ = 0.1

var_dz_1D.ComputationalGrid.NX = 1
var_dz_1D.ComputationalGrid.NY = 1
var_dz_1D.ComputationalGrid.NZ = 14

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
var_dz_1D.GeomInput.Names = 'domain_input het_input1 het_input2'

#---------------------------------------------------------
# Geometry Input
#---------------------------------------------------------
var_dz_1D.GeomInput.domain_input.InputType = 'Box'
var_dz_1D.GeomInput.domain_input.GeomName = 'domain'

var_dz_1D.GeomInput.het_input1.InputType = 'Box'
var_dz_1D.GeomInput.het_input1.GeomName = 'het1'

var_dz_1D.GeomInput.het_input2.InputType = 'Box'
var_dz_1D.GeomInput.het_input2.GeomName = 'het2'

#---------------------------------------------------------
# Geometry
#---------------------------------------------------------
#domain
var_dz_1D.Geom.domain.Lower.X = 0.0
var_dz_1D.Geom.domain.Lower.Y = 0.0
var_dz_1D.Geom.domain.Lower.Z = 0.0

var_dz_1D.Geom.domain.Upper.X = 1.0
var_dz_1D.Geom.domain.Upper.Y = 1.0
var_dz_1D.Geom.domain.Upper.Z = 1.4

var_dz_1D.Geom.domain.Patches = 'left right front back bottom top'

#het1
var_dz_1D.Geom.het1.Lower.X = 0.0
var_dz_1D.Geom.het1.Lower.Y = 0.0
var_dz_1D.Geom.het1.Lower.Z = 1.3

var_dz_1D.Geom.het1.Upper.X = 1.0
var_dz_1D.Geom.het1.Upper.Y = 1.0
var_dz_1D.Geom.het1.Upper.Z = 1.4

#het2
var_dz_1D.Geom.het2.Lower.X = 0.0
var_dz_1D.Geom.het2.Lower.Y = 0.0
var_dz_1D.Geom.het2.Lower.Z = 0.0

var_dz_1D.Geom.het2.Upper.X = 1.0
var_dz_1D.Geom.het2.Upper.Y = 1.0
var_dz_1D.Geom.het2.Upper.Z = 0.1

#--------------------------------------------
# variable dz assignments
#------------------------------------------
var_dz_1D.Solver.Nonlinear.VariableDz = True
var_dz_1D.dzScale.GeomNames = 'domain'
var_dz_1D.dzScale.Type = 'nzList'
var_dz_1D.dzScale.nzListNumber = 14
var_dz_1D.Cell._0.dzScale.Value = 1.2
var_dz_1D.Cell._1.dzScale.Value = 1.0
var_dz_1D.Cell._2.dzScale.Value = 1.0
var_dz_1D.Cell._3.dzScale.Value = 1.0
var_dz_1D.Cell._4.dzScale.Value = 1.0
var_dz_1D.Cell._5.dzScale.Value = 1.0
var_dz_1D.Cell._6.dzScale.Value = 1.0
var_dz_1D.Cell._7.dzScale.Value = 1.0
var_dz_1D.Cell._8.dzScale.Value = 1.0
var_dz_1D.Cell._9.dzScale.Value = 1.0
var_dz_1D.Cell._10.dzScale.Value = 0.15
var_dz_1D.Cell._11.dzScale.Value = 0.1
var_dz_1D.Cell._12.dzScale.Value = 0.1
var_dz_1D.Cell._13.dzScale.Value = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
var_dz_1D.Geom.Perm.Names = 'domain het1 het2'

var_dz_1D.Geom.domain.Perm.Type = 'Constant'
var_dz_1D.Geom.domain.Perm.Value = 5.129

var_dz_1D.Geom.het1.Perm.Type = 'Constant'
var_dz_1D.Geom.het1.Perm.Value = 0.0001

var_dz_1D.Geom.het2.Perm.Type = 'Constant'
var_dz_1D.Geom.het2.Perm.Value = 0.001

var_dz_1D.Perm.TensorType = 'TensorByGeom'

var_dz_1D.Geom.Perm.TensorByGeom.Names = 'domain'

var_dz_1D.Geom.domain.Perm.TensorValX = 1.0
var_dz_1D.Geom.domain.Perm.TensorValY = 1.0
var_dz_1D.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

var_dz_1D.SpecificStorage.Type = 'Constant'
var_dz_1D.SpecificStorage.GeomNames = 'domain'
var_dz_1D.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

var_dz_1D.Phase.Names = 'water'

var_dz_1D.Phase.water.Density.Type = 'Constant'
var_dz_1D.Phase.water.Density.Value = 1.0

var_dz_1D.Phase.water.Viscosity.Type = 'Constant'
var_dz_1D.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
var_dz_1D.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
var_dz_1D.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
var_dz_1D.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

var_dz_1D.TimingInfo.BaseUnit = 1.0
var_dz_1D.TimingInfo.StartCount = 0
var_dz_1D.TimingInfo.StartTime = 0.0
var_dz_1D.TimingInfo.StopTime = 50.0
var_dz_1D.TimingInfo.DumpInterval = -100
var_dz_1D.TimeStep.Type = 'Constant'
var_dz_1D.TimeStep.Value = 0.01
var_dz_1D.TimeStep.Value = 0.01

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

var_dz_1D.Geom.Porosity.GeomNames = 'domain'

var_dz_1D.Geom.domain.Porosity.Type = 'Constant'
var_dz_1D.Geom.domain.Porosity.Value = 0.4150

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
var_dz_1D.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

var_dz_1D.Phase.RelPerm.Type = 'VanGenuchten'
var_dz_1D.Phase.RelPerm.GeomNames = 'domain'
var_dz_1D.Geom.domain.RelPerm.Alpha = 2.7
var_dz_1D.Geom.domain.RelPerm.N = 3.8

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

var_dz_1D.Phase.Saturation.Type = 'VanGenuchten'
var_dz_1D.Phase.Saturation.GeomNames = 'domain'
var_dz_1D.Geom.domain.Saturation.Alpha = 2.7
var_dz_1D.Geom.domain.Saturation.N = 3.8
var_dz_1D.Geom.domain.Saturation.SRes = 0.106
var_dz_1D.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
var_dz_1D.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
var_dz_1D.Cycle.Names = 'constant'
var_dz_1D.Cycle.constant.Names = 'alltime'
var_dz_1D.Cycle.constant.alltime.Length = 1
var_dz_1D.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
var_dz_1D.BCPressure.PatchNames = 'left right front back bottom top'

var_dz_1D.Patch.left.BCPressure.Type = 'FluxConst'
var_dz_1D.Patch.left.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.left.BCPressure.RefGeom = 'domain'
var_dz_1D.Patch.left.BCPressure.RefPatch = 'bottom'
var_dz_1D.Patch.left.BCPressure.alltime.Value = 0.0

var_dz_1D.Patch.right.BCPressure.Type = 'FluxConst'
var_dz_1D.Patch.right.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.right.BCPressure.RefGeom = 'domain'
var_dz_1D.Patch.right.BCPressure.RefPatch = 'bottom'
var_dz_1D.Patch.right.BCPressure.alltime.Value = 0.0

var_dz_1D.Patch.front.BCPressure.Type = 'FluxConst'
var_dz_1D.Patch.front.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.front.BCPressure.alltime.Value = 0.0

var_dz_1D.Patch.back.BCPressure.Type = 'FluxConst'
var_dz_1D.Patch.back.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.back.BCPressure.alltime.Value = 0.0

var_dz_1D.Patch.bottom.BCPressure.Type = 'DirEquilRefPatch'
var_dz_1D.Patch.bottom.BCPressure.Type = 'FluxConst'
var_dz_1D.Patch.bottom.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.bottom.BCPressure.RefGeom = 'domain'
var_dz_1D.Patch.bottom.BCPressure.RefPatch = 'bottom'
var_dz_1D.Patch.bottom.BCPressure.alltime.Value = 0.0

var_dz_1D.Patch.top.BCPressure.Type = 'DirEquilRefPatch'
var_dz_1D.Patch.top.BCPressure.Type = 'FluxConst'
#pfset Patch.top.BCPressure.Type			      OverlandFlow 
var_dz_1D.Patch.top.BCPressure.Cycle = 'constant'
var_dz_1D.Patch.top.BCPressure.RefGeom = 'domain'
var_dz_1D.Patch.top.BCPressure.RefPatch = 'bottom'
var_dz_1D.Patch.top.BCPressure.alltime.Value = -0.0001


#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

var_dz_1D.TopoSlopesX.Type = 'Constant'
var_dz_1D.TopoSlopesX.GeomNames = 'domain'

var_dz_1D.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

var_dz_1D.TopoSlopesY.Type = 'Constant'
var_dz_1D.TopoSlopesY.GeomNames = 'domain'

var_dz_1D.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

var_dz_1D.Mannings.Type = 'Constant'
var_dz_1D.Mannings.GeomNames = 'domain'
var_dz_1D.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

var_dz_1D.ICPressure.Type = 'Constant'
var_dz_1D.ICPressure.GeomNames = 'domain'
var_dz_1D.Geom.domain.ICPressure.Value = -10.0
var_dz_1D.Geom.domain.ICPressure.RefGeom = 'domain'
var_dz_1D.Geom.domain.ICPressure.RefPatch = 'top'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

var_dz_1D.PhaseSources.water.Type = 'Constant'
var_dz_1D.PhaseSources.water.GeomNames = 'domain'
var_dz_1D.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

var_dz_1D.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
var_dz_1D.Solver = 'Richards'
var_dz_1D.Solver.MaxIter = 2500

var_dz_1D.Solver.Nonlinear.MaxIter = 200
var_dz_1D.Solver.Nonlinear.ResidualTol = 1e-9
var_dz_1D.Solver.Nonlinear.EtaChoice = 'Walker1'
var_dz_1D.Solver.Nonlinear.EtaValue = 1e-5
var_dz_1D.Solver.Nonlinear.UseJacobian = True
#pfset Solver.Nonlinear.UseJacobian                     False
var_dz_1D.Solver.Nonlinear.DerivativeEpsilon = 1e-10

var_dz_1D.Solver.Linear.KrylovDimension = 10

var_dz_1D.Solver.Linear.Preconditioner = 'MGSemi'
var_dz_1D.Solver.Linear.Preconditioner = 'PFMG'
var_dz_1D.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
var_dz_1D.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10


#-----------------------------------------------------------------------------
# Run and do tests
#-----------------------------------------------------------------------------

var_dz_1D.run()
