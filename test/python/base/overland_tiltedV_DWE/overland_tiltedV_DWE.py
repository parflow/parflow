#running different configuraitons of tilted V

tcl_precision = 17

# Import the ParFlow TCL package
#
from parflow import Run
overland_tiltedV_DWE = Run("overland_tiltedV_DWE", __file__)

overland_tiltedV_DWE.FileVersion = 4


overland_tiltedV_DWE.Process.Topology.P = 1
overland_tiltedV_DWE.Process.Topology.Q = 1
overland_tiltedV_DWE.Process.Topology.R = 1


#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
overland_tiltedV_DWE.ComputationalGrid.Lower.X = 0.0
overland_tiltedV_DWE.ComputationalGrid.Lower.Y = 0.0
overland_tiltedV_DWE.ComputationalGrid.Lower.Z = 0.0

overland_tiltedV_DWE.ComputationalGrid.NX = 5
overland_tiltedV_DWE.ComputationalGrid.NY = 5
overland_tiltedV_DWE.ComputationalGrid.NZ = 1

overland_tiltedV_DWE.ComputationalGrid.DX = 10.0
overland_tiltedV_DWE.ComputationalGrid.DY = 10.0
overland_tiltedV_DWE.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
overland_tiltedV_DWE.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

overland_tiltedV_DWE.GeomInput.domaininput.GeomName = 'domain'
overland_tiltedV_DWE.GeomInput.leftinput.GeomName = 'left'
overland_tiltedV_DWE.GeomInput.rightinput.GeomName = 'right'
overland_tiltedV_DWE.GeomInput.channelinput.GeomName = 'channel'

overland_tiltedV_DWE.GeomInput.domaininput.InputType = 'Box'
overland_tiltedV_DWE.GeomInput.leftinput.InputType = 'Box'
overland_tiltedV_DWE.GeomInput.rightinput.InputType = 'Box'
overland_tiltedV_DWE.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
overland_tiltedV_DWE.Geom.domain.Lower.X = 0.0
overland_tiltedV_DWE.Geom.domain.Lower.Y = 0.0
overland_tiltedV_DWE.Geom.domain.Lower.Z = 0.0

overland_tiltedV_DWE.Geom.domain.Upper.X = 50.0
overland_tiltedV_DWE.Geom.domain.Upper.Y = 50.0
overland_tiltedV_DWE.Geom.domain.Upper.Z = 0.05
overland_tiltedV_DWE.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry
#---------------------------------------------------------
overland_tiltedV_DWE.Geom.left.Lower.X = 0.0
overland_tiltedV_DWE.Geom.left.Lower.Y = 0.0
overland_tiltedV_DWE.Geom.left.Lower.Z = 0.0

overland_tiltedV_DWE.Geom.left.Upper.X = 20.0
overland_tiltedV_DWE.Geom.left.Upper.Y = 50.0
overland_tiltedV_DWE.Geom.left.Upper.Z = 0.05

#---------------------------------------------------------
# Right Slope Geometry
#---------------------------------------------------------
overland_tiltedV_DWE.Geom.right.Lower.X = 30.0
overland_tiltedV_DWE.Geom.right.Lower.Y = 0.0
overland_tiltedV_DWE.Geom.right.Lower.Z = 0.0

overland_tiltedV_DWE.Geom.right.Upper.X = 50.0
overland_tiltedV_DWE.Geom.right.Upper.Y = 50.0
overland_tiltedV_DWE.Geom.right.Upper.Z = 0.05

#---------------------------------------------------------
# Channel Geometry
#---------------------------------------------------------
overland_tiltedV_DWE.Geom.channel.Lower.X = 20.0
overland_tiltedV_DWE.Geom.channel.Lower.Y = 0.0
overland_tiltedV_DWE.Geom.channel.Lower.Z = 0.0

overland_tiltedV_DWE.Geom.channel.Upper.X = 30.0
overland_tiltedV_DWE.Geom.channel.Upper.Y = 50.0
overland_tiltedV_DWE.Geom.channel.Upper.Z = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Geom.Perm.Names = 'domain'
overland_tiltedV_DWE.Geom.domain.Perm.Type = 'Constant'
overland_tiltedV_DWE.Geom.domain.Perm.Value = 0.0000001

overland_tiltedV_DWE.Perm.TensorType = 'TensorByGeom'

overland_tiltedV_DWE.Geom.Perm.TensorByGeom.Names = 'domain'

overland_tiltedV_DWE.Geom.domain.Perm.TensorValX = 1.0
overland_tiltedV_DWE.Geom.domain.Perm.TensorValY = 1.0
overland_tiltedV_DWE.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.SpecificStorage.Type = 'Constant'
overland_tiltedV_DWE.SpecificStorage.GeomNames = 'domain'
overland_tiltedV_DWE.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Phase.Names = 'water'

overland_tiltedV_DWE.Phase.water.Density.Type = 'Constant'
overland_tiltedV_DWE.Phase.water.Density.Value = 1.0

overland_tiltedV_DWE.Phase.water.Viscosity.Type = 'Constant'
overland_tiltedV_DWE.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
overland_tiltedV_DWE.TimingInfo.BaseUnit = 0.05
overland_tiltedV_DWE.TimingInfo.StartCount = 0
overland_tiltedV_DWE.TimingInfo.StartTime = 0.0
overland_tiltedV_DWE.TimingInfo.StopTime = 2.0
overland_tiltedV_DWE.TimingInfo.DumpInterval = -2
overland_tiltedV_DWE.TimeStep.Type = 'Constant'
overland_tiltedV_DWE.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Geom.Porosity.GeomNames = 'domain'
overland_tiltedV_DWE.Geom.domain.Porosity.Type = 'Constant'
overland_tiltedV_DWE.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Phase.RelPerm.Type = 'VanGenuchten'
overland_tiltedV_DWE.Phase.RelPerm.GeomNames = 'domain'

overland_tiltedV_DWE.Geom.domain.RelPerm.Alpha = 6.0
overland_tiltedV_DWE.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

overland_tiltedV_DWE.Phase.Saturation.Type = 'VanGenuchten'
overland_tiltedV_DWE.Phase.Saturation.GeomNames = 'domain'

overland_tiltedV_DWE.Geom.domain.Saturation.Alpha = 6.0
overland_tiltedV_DWE.Geom.domain.Saturation.N = 2.
overland_tiltedV_DWE.Geom.domain.Saturation.SRes = 0.2
overland_tiltedV_DWE.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
overland_tiltedV_DWE.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
overland_tiltedV_DWE.Cycle.Names = 'constant rainrec'
overland_tiltedV_DWE.Cycle.constant.Names = 'alltime'
overland_tiltedV_DWE.Cycle.constant.alltime.Length = 1
overland_tiltedV_DWE.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland_tiltedV_DWE.Cycle.rainrec.Names = 'rain rec'
overland_tiltedV_DWE.Cycle.rainrec.rain.Length = 2
overland_tiltedV_DWE.Cycle.rainrec.rec.Length = 300
overland_tiltedV_DWE.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
overland_tiltedV_DWE.BCPressure.PatchNames = overland_tiltedV_DWE.Geom.domain.Patches

overland_tiltedV_DWE.Patch.x_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_DWE.Patch.x_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_DWE.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_DWE.Patch.y_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_DWE.Patch.y_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_DWE.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_DWE.Patch.z_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_DWE.Patch.z_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_DWE.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_DWE.Patch.x_upper.BCPressure.Type = 'FluxConst'
overland_tiltedV_DWE.Patch.x_upper.BCPressure.Cycle = 'constant'
overland_tiltedV_DWE.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland_tiltedV_DWE.Patch.y_upper.BCPressure.Type = 'FluxConst'
overland_tiltedV_DWE.Patch.y_upper.BCPressure.Cycle = 'constant'
overland_tiltedV_DWE.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall
overland_tiltedV_DWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_tiltedV_DWE.Patch.z_upper.BCPressure.Cycle = 'rainrec'
overland_tiltedV_DWE.Patch.z_upper.BCPressure.rain.Value = -0.01
overland_tiltedV_DWE.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

overland_tiltedV_DWE.Mannings.Type = 'Constant'
overland_tiltedV_DWE.Mannings.GeomNames = 'domain'
overland_tiltedV_DWE.Mannings.Geom.domain.Value = 3.e-6


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.PhaseSources.water.Type = 'Constant'
overland_tiltedV_DWE.PhaseSources.water.GeomNames = 'domain'
overland_tiltedV_DWE.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

overland_tiltedV_DWE.Solver = 'Richards'
overland_tiltedV_DWE.Solver.MaxIter = 2500

overland_tiltedV_DWE.Solver.Nonlinear.MaxIter = 100
overland_tiltedV_DWE.Solver.Nonlinear.ResidualTol = 1e-9
overland_tiltedV_DWE.Solver.Nonlinear.EtaChoice = 'EtaConstant'
overland_tiltedV_DWE.Solver.Nonlinear.EtaValue = 0.01
overland_tiltedV_DWE.Solver.Nonlinear.UseJacobian = False
overland_tiltedV_DWE.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland_tiltedV_DWE.Solver.Nonlinear.StepTol = 1e-20
overland_tiltedV_DWE.Solver.Nonlinear.Globalization = 'LineSearch'
overland_tiltedV_DWE.Solver.Linear.KrylovDimension = 50
overland_tiltedV_DWE.Solver.Linear.MaxRestart = 2

overland_tiltedV_DWE.Solver.Linear.Preconditioner = 'PFMG'
overland_tiltedV_DWE.Solver.PrintSubsurf = False
overland_tiltedV_DWE.Solver.Drop = 1E-20
overland_tiltedV_DWE.Solver.AbsTol = 1E-10

overland_tiltedV_DWE.Solver.WriteSiloSubsurfData = False
overland_tiltedV_DWE.Solver.WriteSiloPressure = False
overland_tiltedV_DWE.Solver.WriteSiloSlopes = False

overland_tiltedV_DWE.Solver.WriteSiloSaturation = False
overland_tiltedV_DWE.Solver.WriteSiloConcentration = False

overland_tiltedV_DWE.Solver.OverlandDiffusive.Epsilon = 1E-5

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland_tiltedV_DWE.ICPressure.Type = 'HydroStaticPatch'
overland_tiltedV_DWE.ICPressure.GeomNames = 'domain'
overland_tiltedV_DWE.Geom.domain.ICPressure.Value = -3.0

overland_tiltedV_DWE.Geom.domain.ICPressure.RefGeom = 'domain'
overland_tiltedV_DWE.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#set runcheck to 1 if you want to run the pass fail tests
runcheck = 1
# source pftest.tcl

#-----------------------------------------------------------------------------
# New diffusive formulations without the zero channel (as compared to the first
#    tests in overland_tiltedV_KWE.tcl)
# Note: The difference in configuration here is to be consistent with the way
#   the upwinding is handled for the new and original fomulations.
#   These two results should be almost identical for the new and old formulations
#-----------------------------------------------------------------------------
overland_tiltedV_DWE.TopoSlopesX.Type = 'Constant'
overland_tiltedV_DWE.TopoSlopesX.GeomNames = 'left right channel'
overland_tiltedV_DWE.TopoSlopesX.Geom.left.Value = -0.01
overland_tiltedV_DWE.TopoSlopesX.Geom.right.Value = 0.01
overland_tiltedV_DWE.TopoSlopesX.Geom.channel.Value = 0.01

overland_tiltedV_DWE.TopoSlopesY.Type = 'Constant'
overland_tiltedV_DWE.TopoSlopesY.GeomNames = 'domain'
overland_tiltedV_DWE.TopoSlopesY.Geom.domain.Value = 0.01

# run with DWE
overland_tiltedV_DWE.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
overland_tiltedV_DWE.Solver.Nonlinear.UseJacobian = False
overland_tiltedV_DWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname TiltedV_OverlandDif
# puts "##########"
# puts $runname
# pfrun $runname
# pfundist $runname
# if $runcheck==1 {
#   set passed 1
#   foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#     if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#       set passed 0
#     }
#     if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#       set passed 0
#     }
#   }
#   if $passed {
#     puts "$runname : PASSED"
#   } {
#     puts "$runname : FAILED"
#   }
# }

# run with KWE upwinding and analytical jacobian
overland_tiltedV_DWE.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
overland_tiltedV_DWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_DWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname TiltedV_OverlandDif
# puts "##########"
# puts "Running $runname Jacobian True"
# pfrun $runname
# pfundist $runname
# if $runcheck==1 {
#   set passed 1
#   foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#     if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#       set passed 0
#     }
#     if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#       set passed 0
#     }
#   }
#   if $passed {
#     puts "$runname : PASSED"
#   } {
#     puts "$runname : FAILED"
#   }
# }

# run with KWE upwinding and analytical jacobian and nonsymmetric preconditioner
overland_tiltedV_DWE.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
overland_tiltedV_DWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_DWE.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

# set runname TiltedV_OverlandDif
# puts "##########"
# puts "Running $runname Jacobian True Nonsymmetric Preconditioner"
# pfrun $runname
# pfundist $runname
# if $runcheck==1 {
#   set passed 1
#   foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#     if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#       set passed 0
#     }
#     if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#       set passed 0
#     }
#   }
#   if $passed {
#     puts "$runname : PASSED"
#   } {
#     puts "$runname : FAILED"
#   }
# }
overland_tiltedV_DWE.run()
