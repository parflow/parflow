#  Testing overland flow
# Running a parking lot sloping slab pointed in 8 directions
# With a suite of overlandflow BC options

tcl_precision = 17

from parflow import Run
overland_slopingslab_KWE = Run("overland_slopingslab_KWE", __file__)

overland_slopingslab_KWE.FileVersion = 4


overland_slopingslab_KWE.Process.Topology.P = 1
overland_slopingslab_KWE.Process.Topology.Q = 1
overland_slopingslab_KWE.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
overland_slopingslab_KWE.ComputationalGrid.Lower.X = 0.0
overland_slopingslab_KWE.ComputationalGrid.Lower.Y = 0.0
overland_slopingslab_KWE.ComputationalGrid.Lower.Z = 0.0

overland_slopingslab_KWE.ComputationalGrid.NX = 5
overland_slopingslab_KWE.ComputationalGrid.NY = 5
overland_slopingslab_KWE.ComputationalGrid.NZ = 1

overland_slopingslab_KWE.ComputationalGrid.DX = 10.0
overland_slopingslab_KWE.ComputationalGrid.DY = 10.0
overland_slopingslab_KWE.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
overland_slopingslab_KWE.GeomInput.Names = 'domaininput'
overland_slopingslab_KWE.GeomInput.domaininput.GeomName = 'domain'
overland_slopingslab_KWE.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
overland_slopingslab_KWE.Geom.domain.Lower.X = 0.0
overland_slopingslab_KWE.Geom.domain.Lower.Y = 0.0
overland_slopingslab_KWE.Geom.domain.Lower.Z = 0.0

overland_slopingslab_KWE.Geom.domain.Upper.X = 50.0
overland_slopingslab_KWE.Geom.domain.Upper.Y = 50.0
overland_slopingslab_KWE.Geom.domain.Upper.Z = 0.05
overland_slopingslab_KWE.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Geom.Perm.Names = 'domain'
overland_slopingslab_KWE.Geom.domain.Perm.Type = 'Constant'
overland_slopingslab_KWE.Geom.domain.Perm.Value = 0.0000001

overland_slopingslab_KWE.Perm.TensorType = 'TensorByGeom'

overland_slopingslab_KWE.Geom.Perm.TensorByGeom.Names = 'domain'

overland_slopingslab_KWE.Geom.domain.Perm.TensorValX = 1.0
overland_slopingslab_KWE.Geom.domain.Perm.TensorValY = 1.0
overland_slopingslab_KWE.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.SpecificStorage.Type = 'Constant'
overland_slopingslab_KWE.SpecificStorage.GeomNames = 'domain'
overland_slopingslab_KWE.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Phase.Names = 'water'

overland_slopingslab_KWE.Phase.water.Density.Type = 'Constant'
overland_slopingslab_KWE.Phase.water.Density.Value = 1.0

overland_slopingslab_KWE.Phase.water.Viscosity.Type = 'Constant'
overland_slopingslab_KWE.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

#
overland_slopingslab_KWE.TimingInfo.BaseUnit = 0.05
overland_slopingslab_KWE.TimingInfo.StartCount = 0
overland_slopingslab_KWE.TimingInfo.StartTime = 0.0
overland_slopingslab_KWE.TimingInfo.StopTime = 1.0
overland_slopingslab_KWE.TimingInfo.DumpInterval = -2
overland_slopingslab_KWE.TimeStep.Type = 'Constant'
overland_slopingslab_KWE.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Geom.Porosity.GeomNames = 'domain'

overland_slopingslab_KWE.Geom.domain.Porosity.Type = 'Constant'
overland_slopingslab_KWE.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Phase.RelPerm.Type = 'VanGenuchten'
overland_slopingslab_KWE.Phase.RelPerm.GeomNames = 'domain'

overland_slopingslab_KWE.Geom.domain.RelPerm.Alpha = 6.0
overland_slopingslab_KWE.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

overland_slopingslab_KWE.Phase.Saturation.Type = 'VanGenuchten'
overland_slopingslab_KWE.Phase.Saturation.GeomNames = 'domain'

overland_slopingslab_KWE.Geom.domain.Saturation.Alpha = 6.0
overland_slopingslab_KWE.Geom.domain.Saturation.N = 2.
overland_slopingslab_KWE.Geom.domain.Saturation.SRes = 0.2
overland_slopingslab_KWE.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
overland_slopingslab_KWE.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
overland_slopingslab_KWE.Cycle.Names = 'constant rainrec'
overland_slopingslab_KWE.Cycle.constant.Names = 'alltime'
overland_slopingslab_KWE.Cycle.constant.alltime.Length = 1
overland_slopingslab_KWE.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland_slopingslab_KWE.Cycle.rainrec.Names = 'rain rec'
overland_slopingslab_KWE.Cycle.rainrec.rain.Length = 2
overland_slopingslab_KWE.Cycle.rainrec.rec.Length = 300
overland_slopingslab_KWE.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
overland_slopingslab_KWE.BCPressure.PatchNames = overland_slopingslab_KWE.Geom.domain.Patches

overland_slopingslab_KWE.Patch.x_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_KWE.Patch.x_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_KWE.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_KWE.Patch.y_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_KWE.Patch.y_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_KWE.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_KWE.Patch.z_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_KWE.Patch.z_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_KWE.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_KWE.Patch.x_upper.BCPressure.Type = 'FluxConst'
overland_slopingslab_KWE.Patch.x_upper.BCPressure.Cycle = 'constant'
overland_slopingslab_KWE.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland_slopingslab_KWE.Patch.y_upper.BCPressure.Type = 'FluxConst'
overland_slopingslab_KWE.Patch.y_upper.BCPressure.Cycle = 'constant'
overland_slopingslab_KWE.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
overland_slopingslab_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_slopingslab_KWE.Patch.z_upper.BCPressure.Cycle = 'rainrec'
overland_slopingslab_KWE.Patch.z_upper.BCPressure.rain.Value = -0.01
overland_slopingslab_KWE.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

overland_slopingslab_KWE.Mannings.Type = 'Constant'
overland_slopingslab_KWE.Mannings.GeomNames = 'domain'
overland_slopingslab_KWE.Mannings.Geom.domain.Value = 3.e-6


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.PhaseSources.water.Type = 'Constant'
overland_slopingslab_KWE.PhaseSources.water.GeomNames = 'domain'
overland_slopingslab_KWE.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

overland_slopingslab_KWE.Solver = 'Richards'
overland_slopingslab_KWE.Solver.MaxIter = 2500

overland_slopingslab_KWE.Solver.Nonlinear.MaxIter = 50
overland_slopingslab_KWE.Solver.Nonlinear.ResidualTol = 1e-9
overland_slopingslab_KWE.Solver.Nonlinear.EtaChoice = 'EtaConstant'
overland_slopingslab_KWE.Solver.Nonlinear.EtaValue = 0.01
overland_slopingslab_KWE.Solver.Nonlinear.UseJacobian = False

overland_slopingslab_KWE.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland_slopingslab_KWE.Solver.Nonlinear.StepTol = 1e-20
overland_slopingslab_KWE.Solver.Nonlinear.Globalization = 'LineSearch'
overland_slopingslab_KWE.Solver.Linear.KrylovDimension = 20
overland_slopingslab_KWE.Solver.Linear.MaxRestart = 2

overland_slopingslab_KWE.Solver.Linear.Preconditioner = 'PFMG'
overland_slopingslab_KWE.Solver.PrintSubsurf = False
overland_slopingslab_KWE.Solver.Drop = 1E-20
overland_slopingslab_KWE.Solver.AbsTol = 1E-10

overland_slopingslab_KWE.Solver.OverlandKinematic.Epsilon = 1E-5


overland_slopingslab_KWE.Solver.WriteSiloSubsurfData = False
overland_slopingslab_KWE.Solver.WriteSiloPressure = False
overland_slopingslab_KWE.Solver.WriteSiloSlopes = False

overland_slopingslab_KWE.Solver.WriteSiloSaturation = False
overland_slopingslab_KWE.Solver.WriteSiloConcentration = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland_slopingslab_KWE.ICPressure.Type = 'HydroStaticPatch'
overland_slopingslab_KWE.ICPressure.GeomNames = 'domain'
overland_slopingslab_KWE.Geom.domain.ICPressure.Value = -3.0

overland_slopingslab_KWE.Geom.domain.ICPressure.RefGeom = 'domain'
overland_slopingslab_KWE.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# Running all 8 direction combinations with the upwind formulation on and off (i.e. 16 total)
# Commented lines are from original TCL test - will need to convert to Python if running with
# Python pftools
#-----------------------------------------------------------------------------
#set runcheck to 1 if you want to run the pass fail tests
# runcheck = 1
# first = 1
# source pftest.tcl

###############################
# Looping over slop configurations
###############################
# foreach xslope [list 0.01 -0.01] yslope [list 0.01 -0.01] name [list posxposy negxnegy] {
#   puts "$xslope $yslope $name"
#set name negy

#   #### Set the slopes
overland_slopingslab_KWE.TopoSlopesX.Type = "Constant"
overland_slopingslab_KWE.TopoSlopesX.GeomNames = "domain"
overland_slopingslab_KWE.TopoSlopesX.Geom.domain.Value = 0.01

overland_slopingslab_KWE.TopoSlopesY.Type = "Constant"
overland_slopingslab_KWE.TopoSlopesY.GeomNames = "domain"
overland_slopingslab_KWE.TopoSlopesY.Geom.domain.Value = 0.01

#   #original approach from K&M AWR 2006
overland_slopingslab_KWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_slopingslab_KWE.Solver.Nonlinear.UseJacobian = False
overland_slopingslab_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'


#   set runname Slab.$name.OverlandModule
#   puts "##########"
#   if $first==1 {
#     puts $runname
#     set first 0
#   } else {
#     puts "Running $runname Jacobian True"
#   }
#   pfrun $runname
#   pfundist $runname
#   if $runcheck==1 {
#     set passed 1
#     foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#       if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#         set passed 0
#       }
#       if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#         set passed 0
#       }
#     }
#     if $passed {
#       puts "$runname : PASSED"
#     } {
#       puts "$runname : FAILED"
#     }
#   }

# turn on analytical jacobian and re-test
overland_slopingslab_KWE.Solver.Nonlinear.UseJacobian = True
overland_slopingslab_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

# set runname Slab.$name.OverlandModule
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

# turn on non-symmetric Preconditioner and re-test
overland_slopingslab_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

# set runname Slab.$name.OverlandModule
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

# run with KWE upwinding
overland_slopingslab_KWE.Patch.z_upper.BCPressure.Type = 'OverlandKinematic'
overland_slopingslab_KWE.Solver.Nonlinear.UseJacobian = False
overland_slopingslab_KWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

#   set runname Slab.$name.OverlandKin
#   puts "##########"
#   puts "Running $runname"
#   pfrun $runname
#   pfundist $runname
#   if $runcheck==1 {
#     set passed 1
#     foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#       if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#         set passed 0
#       }
#       if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#         set passed 0
#       }
#     }
#     if $passed {
#       puts "$runname : PASSED"
#     } {
#       puts "$runname  : FAILED"
#     }
#   }

#   # run with KWE upwinding jacobian true
#   pfset Patch.z-upper.BCPressure.Type		      OverlandKinematic
#   pfset Solver.Nonlinear.UseJacobian                       True
#   pfset Solver.Linear.Preconditioner.PCMatrixType PFSymmetric

#     set runname Slab.$name.OverlandKin
#     puts "##########"
#     puts "Running $runname Jacobian True"
#     pfrun $runname
#     pfundist $runname
#     if $runcheck==1 {
#       set passed 1
#       foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#         if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#           set passed 0
#         }
#         if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#           set passed 0
#         }
#       }
#       if $passed {
#         puts "$runname : PASSED"
#       } {
#         puts "$runname  : FAILED"
#       }
#     }

#     # run with KWE upwinding jacobian true and nonsymmetric preconditioner
#     pfset Patch.z-upper.BCPressure.Type		      OverlandKinematic
#     pfset Solver.Nonlinear.UseJacobian                       True
#     pfset Solver.Linear.Preconditioner.PCMatrixType FullJacobian

#       set runname Slab.$name.OverlandKin
#       puts "##########"
#       puts "Running $runname Jacobian True Nonsymmetric Preconditioner"
#       pfrun $runname
#       pfundist $runname
#       if $runcheck==1 {
#         set passed 1
#         foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#           if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#             set passed 0
#           }
#           if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#             set passed 0
#           }
#         }
#         if $passed {
#           puts "$runname : PASSED"
#         } {
#           puts "$runname  : FAILED"
#         }
#       }
# }
overland_slopingslab_KWE.run()
