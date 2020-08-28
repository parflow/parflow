#running different configuraitons of tilted V

tcl_precision = 17

from parflow import Run
overland_tiltedV_KWE = Run("overland_tiltedV_KWE", __file__)

overland_tiltedV_KWE.FileVersion = 4


overland_tiltedV_KWE.Process.Topology.P = 1
overland_tiltedV_KWE.Process.Topology.Q = 1
overland_tiltedV_KWE.Process.Topology.R = 1


#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
overland_tiltedV_KWE.ComputationalGrid.Lower.X = 0.0
overland_tiltedV_KWE.ComputationalGrid.Lower.Y = 0.0
overland_tiltedV_KWE.ComputationalGrid.Lower.Z = 0.0

overland_tiltedV_KWE.ComputationalGrid.NX = 5
overland_tiltedV_KWE.ComputationalGrid.NY = 5
overland_tiltedV_KWE.ComputationalGrid.NZ = 1

overland_tiltedV_KWE.ComputationalGrid.DX = 10.0
overland_tiltedV_KWE.ComputationalGrid.DY = 10.0
overland_tiltedV_KWE.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
overland_tiltedV_KWE.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

overland_tiltedV_KWE.GeomInput.domaininput.GeomName = 'domain'
overland_tiltedV_KWE.GeomInput.leftinput.GeomName = 'left'
overland_tiltedV_KWE.GeomInput.rightinput.GeomName = 'right'
overland_tiltedV_KWE.GeomInput.channelinput.GeomName = 'channel'

overland_tiltedV_KWE.GeomInput.domaininput.InputType = 'Box'
overland_tiltedV_KWE.GeomInput.leftinput.InputType = 'Box'
overland_tiltedV_KWE.GeomInput.rightinput.InputType = 'Box'
overland_tiltedV_KWE.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
overland_tiltedV_KWE.Geom.domain.Lower.X = 0.0
overland_tiltedV_KWE.Geom.domain.Lower.Y = 0.0
overland_tiltedV_KWE.Geom.domain.Lower.Z = 0.0

overland_tiltedV_KWE.Geom.domain.Upper.X = 50.0
overland_tiltedV_KWE.Geom.domain.Upper.Y = 50.0
overland_tiltedV_KWE.Geom.domain.Upper.Z = 0.05
overland_tiltedV_KWE.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry
#---------------------------------------------------------
overland_tiltedV_KWE.Geom.left.Lower.X = 0.0
overland_tiltedV_KWE.Geom.left.Lower.Y = 0.0
overland_tiltedV_KWE.Geom.left.Lower.Z = 0.0

overland_tiltedV_KWE.Geom.left.Upper.X = 20.0
overland_tiltedV_KWE.Geom.left.Upper.Y = 50.0
overland_tiltedV_KWE.Geom.left.Upper.Z = 0.05

#---------------------------------------------------------
# Right Slope Geometry
#---------------------------------------------------------
overland_tiltedV_KWE.Geom.right.Lower.X = 30.0
overland_tiltedV_KWE.Geom.right.Lower.Y = 0.0
overland_tiltedV_KWE.Geom.right.Lower.Z = 0.0

overland_tiltedV_KWE.Geom.right.Upper.X = 50.0
overland_tiltedV_KWE.Geom.right.Upper.Y = 50.0
overland_tiltedV_KWE.Geom.right.Upper.Z = 0.05

#---------------------------------------------------------
# Channel Geometry
#---------------------------------------------------------
overland_tiltedV_KWE.Geom.channel.Lower.X = 20.0
overland_tiltedV_KWE.Geom.channel.Lower.Y = 0.0
overland_tiltedV_KWE.Geom.channel.Lower.Z = 0.0

overland_tiltedV_KWE.Geom.channel.Upper.X = 30.0
overland_tiltedV_KWE.Geom.channel.Upper.Y = 50.0
overland_tiltedV_KWE.Geom.channel.Upper.Z = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Geom.Perm.Names = 'domain'
overland_tiltedV_KWE.Geom.domain.Perm.Type = 'Constant'
overland_tiltedV_KWE.Geom.domain.Perm.Value = 0.0000001

overland_tiltedV_KWE.Perm.TensorType = 'TensorByGeom'

overland_tiltedV_KWE.Geom.Perm.TensorByGeom.Names = 'domain'

overland_tiltedV_KWE.Geom.domain.Perm.TensorValX = 1.0
overland_tiltedV_KWE.Geom.domain.Perm.TensorValY = 1.0
overland_tiltedV_KWE.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.SpecificStorage.Type = 'Constant'
overland_tiltedV_KWE.SpecificStorage.GeomNames = 'domain'
overland_tiltedV_KWE.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Phase.Names = 'water'

overland_tiltedV_KWE.Phase.water.Density.Type = 'Constant'
overland_tiltedV_KWE.Phase.water.Density.Value = 1.0

overland_tiltedV_KWE.Phase.water.Viscosity.Type = 'Constant'
overland_tiltedV_KWE.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.TimingInfo.BaseUnit = 0.05
overland_tiltedV_KWE.TimingInfo.StartCount = 0
overland_tiltedV_KWE.TimingInfo.StartTime = 0.0
overland_tiltedV_KWE.TimingInfo.StopTime = 2.0
overland_tiltedV_KWE.TimingInfo.DumpInterval = -2
overland_tiltedV_KWE.TimeStep.Type = 'Constant'
overland_tiltedV_KWE.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Geom.Porosity.GeomNames = 'domain'
overland_tiltedV_KWE.Geom.domain.Porosity.Type = 'Constant'
overland_tiltedV_KWE.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Phase.RelPerm.Type = 'VanGenuchten'
overland_tiltedV_KWE.Phase.RelPerm.GeomNames = 'domain'

overland_tiltedV_KWE.Geom.domain.RelPerm.Alpha = 6.0
overland_tiltedV_KWE.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

overland_tiltedV_KWE.Phase.Saturation.Type = 'VanGenuchten'
overland_tiltedV_KWE.Phase.Saturation.GeomNames = 'domain'

overland_tiltedV_KWE.Geom.domain.Saturation.Alpha = 6.0
overland_tiltedV_KWE.Geom.domain.Saturation.N = 2.
overland_tiltedV_KWE.Geom.domain.Saturation.SRes = 0.2
overland_tiltedV_KWE.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.Cycle.Names = 'constant rainrec'
overland_tiltedV_KWE.Cycle.constant.Names = 'alltime'
overland_tiltedV_KWE.Cycle.constant.alltime.Length = 1
overland_tiltedV_KWE.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland_tiltedV_KWE.Cycle.rainrec.Names = 'rain rec'
overland_tiltedV_KWE.Cycle.rainrec.rain.Length = 2
overland_tiltedV_KWE.Cycle.rainrec.rec.Length = 300
overland_tiltedV_KWE.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.BCPressure.PatchNames = overland_tiltedV_KWE.Geom.domain.Patches

overland_tiltedV_KWE.Patch.x_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_KWE.Patch.x_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_KWE.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_KWE.Patch.y_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_KWE.Patch.y_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_KWE.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_KWE.Patch.z_lower.BCPressure.Type = 'FluxConst'
overland_tiltedV_KWE.Patch.z_lower.BCPressure.Cycle = 'constant'
overland_tiltedV_KWE.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland_tiltedV_KWE.Patch.x_upper.BCPressure.Type = 'FluxConst'
overland_tiltedV_KWE.Patch.x_upper.BCPressure.Cycle = 'constant'
overland_tiltedV_KWE.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland_tiltedV_KWE.Patch.y_upper.BCPressure.Type = 'FluxConst'
overland_tiltedV_KWE.Patch.y_upper.BCPressure.Cycle = 'constant'
overland_tiltedV_KWE.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Cycle = 'rainrec'
overland_tiltedV_KWE.Patch.z_upper.BCPressure.rain.Value = -0.01
overland_tiltedV_KWE.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

overland_tiltedV_KWE.Mannings.Type = 'Constant'
overland_tiltedV_KWE.Mannings.GeomNames = 'domain'
overland_tiltedV_KWE.Mannings.Geom.domain.Value = 3.e-6


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.PhaseSources.water.Type = 'Constant'
overland_tiltedV_KWE.PhaseSources.water.GeomNames = 'domain'
overland_tiltedV_KWE.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

overland_tiltedV_KWE.Solver = 'Richards'
overland_tiltedV_KWE.Solver.MaxIter = 2500

overland_tiltedV_KWE.Solver.Nonlinear.MaxIter = 100
overland_tiltedV_KWE.Solver.Nonlinear.ResidualTol = 1e-9
overland_tiltedV_KWE.Solver.Nonlinear.EtaChoice = 'EtaConstant'
overland_tiltedV_KWE.Solver.Nonlinear.EtaValue = 0.01
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = False
overland_tiltedV_KWE.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland_tiltedV_KWE.Solver.Nonlinear.StepTol = 1e-20
overland_tiltedV_KWE.Solver.Nonlinear.Globalization = 'LineSearch'
overland_tiltedV_KWE.Solver.Linear.KrylovDimension = 50
overland_tiltedV_KWE.Solver.Linear.MaxRestart = 2
overland_tiltedV_KWE.Solver.OverlandKinematic.Epsilon = 1E-5

overland_tiltedV_KWE.Solver.Linear.Preconditioner = 'PFMG'
overland_tiltedV_KWE.Solver.PrintSubsurf = False
overland_tiltedV_KWE.Solver.Drop = 1E-20
overland_tiltedV_KWE.Solver.AbsTol = 1E-10

overland_tiltedV_KWE.Solver.WriteSiloSubsurfData = False
overland_tiltedV_KWE.Solver.WriteSiloPressure = False
overland_tiltedV_KWE.Solver.WriteSiloSlopes = False

overland_tiltedV_KWE.Solver.WriteSiloSaturation = False
overland_tiltedV_KWE.Solver.WriteSiloConcentration = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland_tiltedV_KWE.ICPressure.Type = 'HydroStaticPatch'
overland_tiltedV_KWE.ICPressure.GeomNames = 'domain'
overland_tiltedV_KWE.Geom.domain.ICPressure.Value = -3.0

overland_tiltedV_KWE.Geom.domain.ICPressure.RefGeom = 'domain'
overland_tiltedV_KWE.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#set runcheck to 1 if you want to run the pass fail tests
runcheck = 1
# source pftest.tcl

#-----------------------------------------------------------------------------
# Oringial formulation with a zero value channel
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.TopoSlopesX.Type = 'Constant'
overland_tiltedV_KWE.TopoSlopesX.GeomNames = 'left right channel'
overland_tiltedV_KWE.TopoSlopesX.Geom.left.Value = -0.01
overland_tiltedV_KWE.TopoSlopesX.Geom.right.Value = 0.01
overland_tiltedV_KWE.TopoSlopesX.Geom.channel.Value = 0.00

overland_tiltedV_KWE.TopoSlopesY.Type = 'Constant'
overland_tiltedV_KWE.TopoSlopesY.GeomNames = 'domain'
overland_tiltedV_KWE.TopoSlopesY.Geom.domain.Value = 0.01

#original approach from K&M AWR 2006
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = False
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# Commented lines are from original TCL test - will need to convert to Python if running with
# Python pftools

# set runname TiltedV_Overland
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

#original approach from K&M AWR 2006 with analytical jacobian
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname TiltedV_Overland
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

#original approach from K&M AWR 2006 with analytical jacobian and nonsymmetric preconditioner
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

# set runname TiltedV_Overland
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

#-----------------------------------------------------------------------------
# New kinematic formulations without the zero channel
# Note: The difference in configuration here is to be consistent with the way
#   the upwinding is handled for the new and original fomulations.
#   These two results should be almost identiacl for the new and old formulations
#-----------------------------------------------------------------------------
overland_tiltedV_KWE.TopoSlopesX.Type = 'Constant'
overland_tiltedV_KWE.TopoSlopesX.GeomNames = 'left right channel'
overland_tiltedV_KWE.TopoSlopesX.Geom.left.Value = -0.01
overland_tiltedV_KWE.TopoSlopesX.Geom.right.Value = 0.01
overland_tiltedV_KWE.TopoSlopesX.Geom.channel.Value = 0.01

overland_tiltedV_KWE.TopoSlopesY.Type = 'Constant'
overland_tiltedV_KWE.TopoSlopesY.GeomNames = 'domain'
overland_tiltedV_KWE.TopoSlopesY.Geom.domain.Value = 0.01

# run with KWE upwinding
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandKinematic'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = False
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname TiltedV_OverlandKin
# puts "##########"
# puts "Running $runname"
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
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandKinematic'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname TiltedV_OverlandKin
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
overland_tiltedV_KWE.Patch.z_upper.BCPressure.Type = 'OverlandKinematic'
overland_tiltedV_KWE.Solver.Nonlinear.UseJacobian = True
overland_tiltedV_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

# set runname TiltedV_OverlandKin
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
overland_tiltedV_KWE.run()
