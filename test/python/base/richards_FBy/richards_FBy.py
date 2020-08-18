# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.

# set runname richards_FBy
tcl_precision = 17

# Import the ParFlow TCL package
#
from parflow import Run
richards_FBy = Run("richards_FBy", __file__)

richards_FBy.FileVersion = 4

richards_FBy.Process.Topology.P = 1
richards_FBy.Process.Topology.Q = 1
richards_FBy.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
richards_FBy.ComputationalGrid.Lower.X = 0.0
richards_FBy.ComputationalGrid.Lower.Y = 0.0
richards_FBy.ComputationalGrid.Lower.Z = 0.0

richards_FBy.ComputationalGrid.DX = 1.0
richards_FBy.ComputationalGrid.DY = 1.0
richards_FBy.ComputationalGrid.DZ = 1.0

richards_FBy.ComputationalGrid.NX = 20
richards_FBy.ComputationalGrid.NY = 20
richards_FBy.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
richards_FBy.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
richards_FBy.GeomInput.domain_input.InputType = 'Box'
richards_FBy.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
richards_FBy.Geom.domain.Lower.X = 0.0
richards_FBy.Geom.domain.Lower.Y = 0.0
richards_FBy.Geom.domain.Lower.Z = 0.0

richards_FBy.Geom.domain.Upper.X = 20.0
richards_FBy.Geom.domain.Upper.Y = 20.0
richards_FBy.Geom.domain.Upper.Z = 20.0

richards_FBy.Geom.domain.Patches = 'left right front back bottom top'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_FBy.Geom.Perm.Names = 'domain'
richards_FBy.Geom.domain.Perm.Type = 'Constant'
richards_FBy.Geom.domain.Perm.Value = 1.0

richards_FBy.Perm.TensorType = 'TensorByGeom'

richards_FBy.Geom.Perm.TensorByGeom.Names = 'domain'

richards_FBy.Geom.domain.Perm.TensorValX = 1.0
richards_FBy.Geom.domain.Perm.TensorValY = 1.0
richards_FBy.Geom.domain.Perm.TensorValZ = 1.0



#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_FBy.SpecificStorage.Type = 'Constant'
richards_FBy.SpecificStorage.GeomNames = 'domain'
richards_FBy.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_FBy.Phase.Names = 'water'

richards_FBy.Phase.water.Density.Type = 'Constant'
richards_FBy.Phase.water.Density.Value = 1.0

richards_FBy.Phase.water.Viscosity.Type = 'Constant'
richards_FBy.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
richards_FBy.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
richards_FBy.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_FBy.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_FBy.TimingInfo.BaseUnit = 10.
richards_FBy.TimingInfo.StartCount = 0
richards_FBy.TimingInfo.StartTime = 0.0
richards_FBy.TimingInfo.StopTime = 100.0
richards_FBy.TimingInfo.DumpInterval = 10.0
richards_FBy.TimeStep.Type = 'Constant'
richards_FBy.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_FBy.Geom.Porosity.GeomNames = 'domain'

richards_FBy.Geom.domain.Porosity.Type = 'Constant'
richards_FBy.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
richards_FBy.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_FBy.Phase.RelPerm.Type = 'VanGenuchten'
richards_FBy.Phase.RelPerm.GeomNames = 'domain'
richards_FBy.Geom.domain.RelPerm.Alpha = 2.0
richards_FBy.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_FBy.Phase.Saturation.Type = 'VanGenuchten'
richards_FBy.Phase.Saturation.GeomNames = 'domain'
richards_FBy.Geom.domain.Saturation.Alpha = 2.0
richards_FBy.Geom.domain.Saturation.N = 2.0
richards_FBy.Geom.domain.Saturation.SRes = 0.1
richards_FBy.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# Flow Barrier in Y between cells 10 and 11 in all Z
#---------------------------------------------------------

richards_FBy.Solver.Nonlinear.FlowBarrierY = True
richards_FBy.FBy.Type = 'PFBFile'
richards_FBy.Geom.domain.FBy.FileName = 'Flow_Barrier_Y.pfb'

## write flow barrier file
# fileId = [open Flow_Barrier_Y.sa w]
# puts $fileId "20 20 20"
# for { set kk 0 } { $kk < 20 } { incr kk } {
# for { set jj 0 } { $jj < 20 } { incr jj } {
# for { set ii 0 } { $ii < 20 } { incr ii } {

# 	if {$jj == 9} {
# 		# from cell 10 (index 9) to cell 11
# 		# reduction of 1E-3
# 		puts $fileId "0.001"
# 	} else {
# 		puts $fileId "1.0"  }
# }
# }
# }
# close $fileId

# FBy = [pfload -sa Flow_Barrier_Y.sa]
# pfsetgrid {20 20 20} {0.0 0.0 0.0} {1.0 1.0 1.0} $FBy
# pfsave $FBy -pfb Flow_Barrier_Y.pfb

# pfdist  Flow_Barrier_Y.pfb

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
richards_FBy.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
richards_FBy.Cycle.Names = 'constant'
richards_FBy.Cycle.constant.Names = 'alltime'
richards_FBy.Cycle.constant.alltime.Length = 1
richards_FBy.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_FBy.BCPressure.PatchNames = 'left right front back bottom top'

richards_FBy.Patch.front.BCPressure.Type = 'DirEquilRefPatch'
richards_FBy.Patch.front.BCPressure.Cycle = 'constant'
richards_FBy.Patch.front.BCPressure.RefGeom = 'domain'
richards_FBy.Patch.front.BCPressure.RefPatch = 'bottom'
richards_FBy.Patch.front.BCPressure.alltime.Value = 11.0

richards_FBy.Patch.back.BCPressure.Type = 'DirEquilRefPatch'
richards_FBy.Patch.back.BCPressure.Cycle = 'constant'
richards_FBy.Patch.back.BCPressure.RefGeom = 'domain'
richards_FBy.Patch.back.BCPressure.RefPatch = 'bottom'
richards_FBy.Patch.back.BCPressure.alltime.Value = 15.0

richards_FBy.Patch.left.BCPressure.Type = 'FluxConst'
richards_FBy.Patch.left.BCPressure.Cycle = 'constant'
richards_FBy.Patch.left.BCPressure.alltime.Value = 0.0

richards_FBy.Patch.right.BCPressure.Type = 'FluxConst'
richards_FBy.Patch.right.BCPressure.Cycle = 'constant'
richards_FBy.Patch.right.BCPressure.alltime.Value = 0.0

richards_FBy.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_FBy.Patch.bottom.BCPressure.Cycle = 'constant'
richards_FBy.Patch.bottom.BCPressure.alltime.Value = 0.0

richards_FBy.Patch.top.BCPressure.Type = 'FluxConst'
richards_FBy.Patch.top.BCPressure.Cycle = 'constant'
richards_FBy.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_FBy.TopoSlopesX.Type = 'Constant'
richards_FBy.TopoSlopesX.GeomNames = 'domain'

richards_FBy.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_FBy.TopoSlopesY.Type = 'Constant'
richards_FBy.TopoSlopesY.GeomNames = 'domain'

richards_FBy.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

richards_FBy.Mannings.Type = 'Constant'
richards_FBy.Mannings.GeomNames = 'domain'
richards_FBy.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_FBy.ICPressure.Type = 'HydroStaticPatch'
richards_FBy.ICPressure.GeomNames = 'domain'
richards_FBy.Geom.domain.ICPressure.Value = 13.0
richards_FBy.Geom.domain.ICPressure.RefGeom = 'domain'
richards_FBy.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_FBy.PhaseSources.water.Type = 'Constant'
richards_FBy.PhaseSources.water.GeomNames = 'domain'
richards_FBy.PhaseSources.water.Geom.domain.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_FBy.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
richards_FBy.Solver = 'Richards'
richards_FBy.Solver.MaxIter = 50000

richards_FBy.Solver.Nonlinear.MaxIter = 100
richards_FBy.Solver.Nonlinear.ResidualTol = 1e-6
richards_FBy.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_FBy.Solver.Nonlinear.EtaValue = 1e-2
richards_FBy.Solver.Nonlinear.UseJacobian = True

richards_FBy.Solver.Nonlinear.DerivativeEpsilon = 1e-12

richards_FBy.Solver.Linear.KrylovDimension = 100

richards_FBy.Solver.Linear.Preconditioner = 'PFMG'


#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_FBy.run()
