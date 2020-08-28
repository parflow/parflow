#---------------------------------------------------------
#  Testing overland flow diffusive wave
# Running a parking lot sloping slab pointed in 8 directions
# With diffusive BC options
#---------------------------------------------------------
from parflow import Run

run = Run("overland_slopingslab_DWE", __file__)

#---------------------------------------------------------

run.FileVersion = 4

run.Process.Topology.P = 1
run.Process.Topology.Q = 1
run.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

run.ComputationalGrid.Lower.X = 0.0
run.ComputationalGrid.Lower.Y = 0.0
run.ComputationalGrid.Lower.Z = 0.0

run.ComputationalGrid.NX = 5
run.ComputationalGrid.NY = 5
run.ComputationalGrid.NZ = 1

run.ComputationalGrid.DX = 10.0
run.ComputationalGrid.DY = 10.0
run.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

run.GeomInput.Names = 'domaininput'
run.GeomInput.domaininput.GeomName = 'domain'
run.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

run.Geom.domain.Lower.X = 0.0
run.Geom.domain.Lower.Y = 0.0
run.Geom.domain.Lower.Z = 0.0

run.Geom.domain.Upper.X = 50.0
run.Geom.domain.Upper.Y = 50.0
run.Geom.domain.Upper.Z = 0.05
run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

run.Geom.Perm.Names = 'domain'
run.Geom.domain.Perm.Type = 'Constant'
run.Geom.domain.Perm.Value = 0.0000001

run.Perm.TensorType = 'TensorByGeom'

run.Geom.Perm.TensorByGeom.Names = 'domain'

run.Geom.domain.Perm.TensorValX = 1.0
run.Geom.domain.Perm.TensorValY = 1.0
run.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

run.SpecificStorage.Type = 'Constant'
run.SpecificStorage.GeomNames = 'domain'
run.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

run.Phase.Names = 'water'

run.Phase.water.Density.Type = 'Constant'
run.Phase.water.Density.Value = 1.0

run.Phase.water.Viscosity.Type = 'Constant'
run.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

run.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

run.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

run.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

run.TimingInfo.BaseUnit = 0.05
run.TimingInfo.StartCount = 0
run.TimingInfo.StartTime = 0.0
run.TimingInfo.StopTime = 1.0
run.TimingInfo.DumpInterval = -2
run.TimeStep.Type = 'Constant'
run.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

run.Geom.Porosity.GeomNames = 'domain'

run.Geom.domain.Porosity.Type = 'Constant'
run.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

run.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

run.Phase.RelPerm.Type = 'VanGenuchten'
run.Phase.RelPerm.GeomNames = 'domain'

run.Geom.domain.RelPerm.Alpha = 6.0
run.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

run.Phase.Saturation.Type = 'VanGenuchten'
run.Phase.Saturation.GeomNames = 'domain'

run.Geom.domain.Saturation.Alpha = 6.0
run.Geom.domain.Saturation.N = 2.
run.Geom.domain.Saturation.SRes = 0.2
run.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

run.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

run.Cycle.Names = 'constant rainrec'
run.Cycle.constant.Names = 'alltime'
run.Cycle.constant.alltime.Length = 1
run.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

run.Cycle.rainrec.Names = 'rain rec'
run.Cycle.rainrec.rain.Length = 2
run.Cycle.rainrec.rec.Length = 300
run.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

run.BCPressure.PatchNames = run.Geom.domain.Patches

run.Patch.x_lower.BCPressure.Type = 'FluxConst'
run.Patch.x_lower.BCPressure.Cycle = 'constant'
run.Patch.x_lower.BCPressure.alltime.Value = 0.0

run.Patch.y_lower.BCPressure.Type = 'FluxConst'
run.Patch.y_lower.BCPressure.Cycle = 'constant'
run.Patch.y_lower.BCPressure.alltime.Value = 0.0

run.Patch.z_lower.BCPressure.Type = 'FluxConst'
run.Patch.z_lower.BCPressure.Cycle = 'constant'
run.Patch.z_lower.BCPressure.alltime.Value = 0.0

run.Patch.x_upper.BCPressure.Type = 'FluxConst'
run.Patch.x_upper.BCPressure.Cycle = 'constant'
run.Patch.x_upper.BCPressure.alltime.Value = 0.0

run.Patch.y_upper.BCPressure.Type = 'FluxConst'
run.Patch.y_upper.BCPressure.Cycle = 'constant'
run.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
run.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
run.Patch.z_upper.BCPressure.Cycle = 'rainrec'
run.Patch.z_upper.BCPressure.rain.Value = -0.01
run.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

run.Mannings.Type = 'Constant'
run.Mannings.GeomNames = 'domain'
run.Mannings.Geom.domain.Value = 3.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

run.PhaseSources.water.Type = 'Constant'
run.PhaseSources.water.GeomNames = 'domain'
run.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

run.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

run.Solver = 'Richards'
run.Solver.MaxIter = 2500

run.Solver.Nonlinear.MaxIter = 50
run.Solver.Nonlinear.ResidualTol = 1e-9
run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
run.Solver.Nonlinear.EtaValue = 0.01
run.Solver.Nonlinear.UseJacobian = False
#pfset Solver.Nonlinear.UseJacobian                       True

run.Solver.Nonlinear.DerivativeEpsilon = 1e-15
run.Solver.Nonlinear.StepTol = 1e-20
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Linear.KrylovDimension = 20
run.Solver.Linear.MaxRestart = 2

run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.PrintSubsurf = False
run.Solver.Drop = 1E-20
run.Solver.AbsTol = 1E-10

run.Solver.OverlandDiffusive.Epsilon = 1E-5

#pfset Solver.Linear.Preconditioner.PCMatrixType         FullJacobian

run.Solver.WriteSiloSubsurfData = False
run.Solver.WriteSiloPressure = False
run.Solver.WriteSiloSlopes = False

run.Solver.WriteSiloSaturation = False
run.Solver.WriteSiloConcentration = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
run.ICPressure.Type = 'HydroStaticPatch'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.Value = -3.0

run.Geom.domain.ICPressure.RefGeom = 'domain'
run.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# Running all 8 direction combinations with the upwind formulation on and off (i.e. 16 total)
# Commented lines are from original TCL test - will need to convert to Python if running with
# Python pftools
#-----------------------------------------------------------------------------

#set runcheck to 1 if you want to run the pass fail tests
# runcheck = 1
# source pftest.tcl
# first = 1

###############################
# Looping over slab configurations
###############################
# foreach xslope [list 0.01 -0.01] yslope [list 0.01 -0.01] name [list posxposy negxnegy] {
#   puts "$xslope $yslope $name"

#   #### Set the slopes
#   pfset TopoSlopesX.Type "Constant"
#   pfset TopoSlopesX.GeomNames "domain"
#   pfset TopoSlopesX.Geom.domain.Value $xslope

#   pfset TopoSlopesY.Type "Constant"
#   pfset TopoSlopesY.GeomNames "domain"
#   pfset TopoSlopesY.Geom.domain.Value $yslope

#    #new BC
#    pfset Patch.z-upper.BCPressure.Type		      OverlandDiffusive
#    #pfset Solver.Nonlinear.UseJacobian                       True
#    #pfset Solver.Linear.Preconditioner.PCMatrixType         FullJacobian

#    set runname Slab.$name.OverlandDif
#    if $first==1 {
#      puts $runname
#      set first 0
#    } else {
#      puts "Running $runname OverlandDiffusive"
#    }
#    pfrun $runname
#    pfundist $runname
#    if $runcheck==1 {
#      set passed 1
#      foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#        if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#          set passed 0
#        }
#        if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#          set passed 0
#        }
#      }
#      if $passed {
#        puts "$runname : PASSED"
#      } {
#        puts "$runname : FAILED"
#      }
#    }

# }

#set to 1 if you want to run the pass fail tests
# runcheck = 1
# source pftest.tcl

###############################
# Looping over slop configurations
###############################
# foreach xslope [list 0.01 -0.01] yslope [list 0.01 -0.01] name [list posxposy negxnegy] {
#   puts "$xslope $yslope $name"

#   #### Set the slopes
run.TopoSlopesX.Type = "Constant"
run.TopoSlopesX.GeomNames = "domain"
run.TopoSlopesX.Geom.domain.Value = 0.01

run.TopoSlopesY.Type = "Constant"
run.TopoSlopesY.GeomNames = "domain"
run.TopoSlopesY.Geom.domain.Value = 0.01

#   #original approach from K&M AWR 2006
run.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
run.Solver.Nonlinear.UseJacobian = False
run.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

#    set runname Slab.$name.OverlandDif
#    puts "Running $runname OverlandDiffusive Jacobian True"
#    pfrun $runname
#    pfundist $runname
#    if $runcheck==1 {
#      set passed 1
#      foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#        if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#          set passed 0
#        }
#        if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#          set passed 0
#        }
#      }
#      if $passed {
#        puts "$runname : PASSED"
#      } {
#        puts "$runname : FAILED"
#      }
#    }


# }

#set to 1 if you want to run the pass fail tests
# runcheck = 1
# source pftest.tcl

###############################
# Looping over slop configurations
###############################
# foreach xslope [list 0.01 -0.01] yslope [list 0.01 -0.01] name [list posxposy negxnegy] {
#   puts "$xslope $yslope $name"

#   #### Set the slopes
#   pfset TopoSlopesX.Type "Constant"
#   pfset TopoSlopesX.GeomNames "domain"
#   pfset TopoSlopesX.Geom.domain.Value $xslope

#   pfset TopoSlopesY.Type "Constant"
#   pfset TopoSlopesY.GeomNames "domain"
#   pfset TopoSlopesY.Geom.domain.Value $yslope

#    #new BC
#    pfset Patch.z-upper.BCPressure.Type		      OverlandDiffusive
#    pfset Solver.Nonlinear.UseJacobian                       True
#    pfset Solver.Linear.Preconditioner.PCMatrixType         FullJacobian

#    set runname Slab.$name.OverlandDif
#    puts "Running $runname OverlandDiffusive Jacobian True Nonsymmetric Preconditioner"
#    pfrun $runname
#    pfundist $runname
#    if $runcheck==1 {
#      set passed 1
#      foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
#        if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
#          set passed 0
#        }
#        if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
#          set passed 0
#        }
#      }
#      if $passed {
#        puts "$runname : PASSED"
#      } {
#        puts "$runname : FAILED"
#      }
#    }

# }
run.run()
