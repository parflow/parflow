# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.

# set runname richards_FBx
tcl_precision = 17

# Import the ParFlow TCL package
#
from parflow import Run
richards_FBx = Run("richards_FBx", __file__)

richards_FBx.FileVersion = 4

richards_FBx.Process.Topology.P = 1
richards_FBx.Process.Topology.Q = 1
richards_FBx.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
richards_FBx.ComputationalGrid.Lower.X = 0.0
richards_FBx.ComputationalGrid.Lower.Y = 0.0
richards_FBx.ComputationalGrid.Lower.Z = 0.0

richards_FBx.ComputationalGrid.DX = 1.0
richards_FBx.ComputationalGrid.DY = 1.0
richards_FBx.ComputationalGrid.DZ = 1.0

richards_FBx.ComputationalGrid.NX = 20
richards_FBx.ComputationalGrid.NY = 20
richards_FBx.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
richards_FBx.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
richards_FBx.GeomInput.domain_input.InputType = 'Box'
richards_FBx.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
richards_FBx.Geom.domain.Lower.X = 0.0
richards_FBx.Geom.domain.Lower.Y = 0.0
richards_FBx.Geom.domain.Lower.Z = 0.0

richards_FBx.Geom.domain.Upper.X = 20.0
richards_FBx.Geom.domain.Upper.Y = 20.0
richards_FBx.Geom.domain.Upper.Z = 20.0

richards_FBx.Geom.domain.Patches = 'left right front back bottom top'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

richards_FBx.Geom.Perm.Names = 'domain'
richards_FBx.Geom.domain.Perm.Type = 'Constant'
richards_FBx.Geom.domain.Perm.Value = 1.0

richards_FBx.Perm.TensorType = 'TensorByGeom'

richards_FBx.Geom.Perm.TensorByGeom.Names = 'domain'

richards_FBx.Geom.domain.Perm.TensorValX = 1.0
richards_FBx.Geom.domain.Perm.TensorValY = 1.0
richards_FBx.Geom.domain.Perm.TensorValZ = 1.0



#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

richards_FBx.SpecificStorage.Type = 'Constant'
richards_FBx.SpecificStorage.GeomNames = 'domain'
richards_FBx.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

richards_FBx.Phase.Names = 'water'

richards_FBx.Phase.water.Density.Type = 'Constant'
richards_FBx.Phase.water.Density.Value = 1.0

richards_FBx.Phase.water.Viscosity.Type = 'Constant'
richards_FBx.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
richards_FBx.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
richards_FBx.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

richards_FBx.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

richards_FBx.TimingInfo.BaseUnit = 10.
richards_FBx.TimingInfo.StartCount = 0
richards_FBx.TimingInfo.StartTime = 0.0
richards_FBx.TimingInfo.StopTime = 100.0
richards_FBx.TimingInfo.DumpInterval = 10.0
richards_FBx.TimeStep.Type = 'Constant'
richards_FBx.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

richards_FBx.Geom.Porosity.GeomNames = 'domain'

richards_FBx.Geom.domain.Porosity.Type = 'Constant'
richards_FBx.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
richards_FBx.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

richards_FBx.Phase.RelPerm.Type = 'VanGenuchten'
richards_FBx.Phase.RelPerm.GeomNames = 'domain'
richards_FBx.Geom.domain.RelPerm.Alpha = 2.0
richards_FBx.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

richards_FBx.Phase.Saturation.Type = 'VanGenuchten'
richards_FBx.Phase.Saturation.GeomNames = 'domain'
richards_FBx.Geom.domain.Saturation.Alpha = 2.0
richards_FBx.Geom.domain.Saturation.N = 2.0
richards_FBx.Geom.domain.Saturation.SRes = 0.1
richards_FBx.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# Flow Barrier in X between cells 10 and 11 in all Z
#---------------------------------------------------------

richards_FBx.Solver.Nonlinear.FlowBarrierX = True
richards_FBx.FBx.Type = 'PFBFile'
richards_FBx.Geom.domain.FBx.FileName = 'Flow_Barrier_X.pfb'

## write flow boundary file
# fileId = [open Flow_Barrier_X.sa w]
# # puts $fileId "20 20 20"
# # for { set kk 0 } { $kk < 20 } { incr kk } {
# # for { set jj 0 } { $jj < 20 } { incr jj } {
# # for { set ii 0 } { $ii < 20 } { incr ii } {
#
# # 	if {$ii == 9} {
# # 		# from cell 10 (index 9) to cell 11
# # 		# reduction of 1E-3
# # 		puts $fileId "0.001"
# # 	} else {
# # 		puts $fileId "1.0"  }
# # }
# # }
# # }
# # close $fileId
#
# FBx = [pfload -sa Flow_Barrier_X.sa]
# pfsetgrid {20 20 20} {0.0 0.0 0.0} {1.0 1.0 1.0} $FBx
# pfsave $FBx -pfb Flow_Barrier_X.pfb

richards_FBx.dist('Flow_Barrier_X.pfb')

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
richards_FBx.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
richards_FBx.Cycle.Names = 'constant'
richards_FBx.Cycle.constant.Names = 'alltime'
richards_FBx.Cycle.constant.alltime.Length = 1
richards_FBx.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

richards_FBx.BCPressure.PatchNames = 'left right front back bottom top'

richards_FBx.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
richards_FBx.Patch.left.BCPressure.Cycle = 'constant'
richards_FBx.Patch.left.BCPressure.RefGeom = 'domain'
richards_FBx.Patch.left.BCPressure.RefPatch = 'bottom'
richards_FBx.Patch.left.BCPressure.alltime.Value = 11.0

richards_FBx.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
richards_FBx.Patch.right.BCPressure.Cycle = 'constant'
richards_FBx.Patch.right.BCPressure.RefGeom = 'domain'
richards_FBx.Patch.right.BCPressure.RefPatch = 'bottom'
richards_FBx.Patch.right.BCPressure.alltime.Value = 15.0

richards_FBx.Patch.front.BCPressure.Type = 'FluxConst'
richards_FBx.Patch.front.BCPressure.Cycle = 'constant'
richards_FBx.Patch.front.BCPressure.alltime.Value = 0.0

richards_FBx.Patch.back.BCPressure.Type = 'FluxConst'
richards_FBx.Patch.back.BCPressure.Cycle = 'constant'
richards_FBx.Patch.back.BCPressure.alltime.Value = 0.0

richards_FBx.Patch.bottom.BCPressure.Type = 'FluxConst'
richards_FBx.Patch.bottom.BCPressure.Cycle = 'constant'
richards_FBx.Patch.bottom.BCPressure.alltime.Value = 0.0

richards_FBx.Patch.top.BCPressure.Type = 'FluxConst'
richards_FBx.Patch.top.BCPressure.Cycle = 'constant'
richards_FBx.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

richards_FBx.TopoSlopesX.Type = 'Constant'
richards_FBx.TopoSlopesX.GeomNames = 'domain'

richards_FBx.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

richards_FBx.TopoSlopesY.Type = 'Constant'
richards_FBx.TopoSlopesY.GeomNames = 'domain'

richards_FBx.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

richards_FBx.Mannings.Type = 'Constant'
richards_FBx.Mannings.GeomNames = 'domain'
richards_FBx.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

richards_FBx.ICPressure.Type = 'HydroStaticPatch'
richards_FBx.ICPressure.GeomNames = 'domain'
richards_FBx.Geom.domain.ICPressure.Value = 13.0
richards_FBx.Geom.domain.ICPressure.RefGeom = 'domain'
richards_FBx.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

richards_FBx.PhaseSources.water.Type = 'Constant'
richards_FBx.PhaseSources.water.GeomNames = 'domain'
richards_FBx.PhaseSources.water.Geom.domain.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

richards_FBx.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
richards_FBx.Solver = 'Richards'
richards_FBx.Solver.MaxIter = 50000

richards_FBx.Solver.Nonlinear.MaxIter = 100
richards_FBx.Solver.Nonlinear.ResidualTol = 1e-6
richards_FBx.Solver.Nonlinear.EtaChoice = 'EtaConstant'
richards_FBx.Solver.Nonlinear.EtaValue = 1e-2
richards_FBx.Solver.Nonlinear.UseJacobian = True

richards_FBx.Solver.Nonlinear.DerivativeEpsilon = 1e-12

richards_FBx.Solver.Linear.KrylovDimension = 100

richards_FBx.Solver.Linear.Preconditioner = 'PFMG'


#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

richards_FBx.run()
