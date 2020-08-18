#  Testing overland flow diffusive wave
# Running a parking lot sloping slab pointed in 8 directions
# With diffusive BC options

tcl_precision = 17

#
# Import the ParFlow TCL package
#
from parflow import Run
overland_slopingslab_DWE = Run("overland_slopingslab_DWE", __file__)

overland_slopingslab_DWE.FileVersion = 4


overland_slopingslab_DWE.Process.Topology.P = 1
overland_slopingslab_DWE.Process.Topology.Q = 1
overland_slopingslab_DWE.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
overland_slopingslab_DWE.ComputationalGrid.Lower.X = 0.0
overland_slopingslab_DWE.ComputationalGrid.Lower.Y = 0.0
overland_slopingslab_DWE.ComputationalGrid.Lower.Z = 0.0

overland_slopingslab_DWE.ComputationalGrid.NX = 5
overland_slopingslab_DWE.ComputationalGrid.NY = 5
overland_slopingslab_DWE.ComputationalGrid.NZ = 1

overland_slopingslab_DWE.ComputationalGrid.DX = 10.0
overland_slopingslab_DWE.ComputationalGrid.DY = 10.0
overland_slopingslab_DWE.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
overland_slopingslab_DWE.GeomInput.Names = 'domaininput'
overland_slopingslab_DWE.GeomInput.domaininput.GeomName = 'domain'
overland_slopingslab_DWE.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
overland_slopingslab_DWE.Geom.domain.Lower.X = 0.0
overland_slopingslab_DWE.Geom.domain.Lower.Y = 0.0
overland_slopingslab_DWE.Geom.domain.Lower.Z = 0.0

overland_slopingslab_DWE.Geom.domain.Upper.X = 50.0
overland_slopingslab_DWE.Geom.domain.Upper.Y = 50.0
overland_slopingslab_DWE.Geom.domain.Upper.Z = 0.05
overland_slopingslab_DWE.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Geom.Perm.Names = 'domain'
overland_slopingslab_DWE.Geom.domain.Perm.Type = 'Constant'
overland_slopingslab_DWE.Geom.domain.Perm.Value = 0.0000001

overland_slopingslab_DWE.Perm.TensorType = 'TensorByGeom'

overland_slopingslab_DWE.Geom.Perm.TensorByGeom.Names = 'domain'

overland_slopingslab_DWE.Geom.domain.Perm.TensorValX = 1.0
overland_slopingslab_DWE.Geom.domain.Perm.TensorValY = 1.0
overland_slopingslab_DWE.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.SpecificStorage.Type = 'Constant'
overland_slopingslab_DWE.SpecificStorage.GeomNames = 'domain'
overland_slopingslab_DWE.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Phase.Names = 'water'

overland_slopingslab_DWE.Phase.water.Density.Type = 'Constant'
overland_slopingslab_DWE.Phase.water.Density.Value = 1.0

overland_slopingslab_DWE.Phase.water.Viscosity.Type = 'Constant'
overland_slopingslab_DWE.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

#
overland_slopingslab_DWE.TimingInfo.BaseUnit = 0.05
overland_slopingslab_DWE.TimingInfo.StartCount = 0
overland_slopingslab_DWE.TimingInfo.StartTime = 0.0
overland_slopingslab_DWE.TimingInfo.StopTime = 1.0
overland_slopingslab_DWE.TimingInfo.DumpInterval = -2
overland_slopingslab_DWE.TimeStep.Type = 'Constant'
overland_slopingslab_DWE.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Geom.Porosity.GeomNames = 'domain'

overland_slopingslab_DWE.Geom.domain.Porosity.Type = 'Constant'
overland_slopingslab_DWE.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Phase.RelPerm.Type = 'VanGenuchten'
overland_slopingslab_DWE.Phase.RelPerm.GeomNames = 'domain'

overland_slopingslab_DWE.Geom.domain.RelPerm.Alpha = 6.0
overland_slopingslab_DWE.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

overland_slopingslab_DWE.Phase.Saturation.Type = 'VanGenuchten'
overland_slopingslab_DWE.Phase.Saturation.GeomNames = 'domain'

overland_slopingslab_DWE.Geom.domain.Saturation.Alpha = 6.0
overland_slopingslab_DWE.Geom.domain.Saturation.N = 2.
overland_slopingslab_DWE.Geom.domain.Saturation.SRes = 0.2
overland_slopingslab_DWE.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
overland_slopingslab_DWE.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
overland_slopingslab_DWE.Cycle.Names = 'constant rainrec'
overland_slopingslab_DWE.Cycle.constant.Names = 'alltime'
overland_slopingslab_DWE.Cycle.constant.alltime.Length = 1
overland_slopingslab_DWE.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland_slopingslab_DWE.Cycle.rainrec.Names = 'rain rec'
overland_slopingslab_DWE.Cycle.rainrec.rain.Length = 2
overland_slopingslab_DWE.Cycle.rainrec.rec.Length = 300
overland_slopingslab_DWE.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
overland_slopingslab_DWE.BCPressure.PatchNames = overland_slopingslab_DWE.Geom.domain.Patches

overland_slopingslab_DWE.Patch.x_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_DWE.Patch.x_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_DWE.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_DWE.Patch.y_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_DWE.Patch.y_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_DWE.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_DWE.Patch.z_lower.BCPressure.Type = 'FluxConst'
overland_slopingslab_DWE.Patch.z_lower.BCPressure.Cycle = 'constant'
overland_slopingslab_DWE.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland_slopingslab_DWE.Patch.x_upper.BCPressure.Type = 'FluxConst'
overland_slopingslab_DWE.Patch.x_upper.BCPressure.Cycle = 'constant'
overland_slopingslab_DWE.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland_slopingslab_DWE.Patch.y_upper.BCPressure.Type = 'FluxConst'
overland_slopingslab_DWE.Patch.y_upper.BCPressure.Cycle = 'constant'
overland_slopingslab_DWE.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
overland_slopingslab_DWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_slopingslab_DWE.Patch.z_upper.BCPressure.Cycle = 'rainrec'
overland_slopingslab_DWE.Patch.z_upper.BCPressure.rain.Value = -0.01
overland_slopingslab_DWE.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

overland_slopingslab_DWE.Mannings.Type = 'Constant'
overland_slopingslab_DWE.Mannings.GeomNames = 'domain'
overland_slopingslab_DWE.Mannings.Geom.domain.Value = 3.e-6


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.PhaseSources.water.Type = 'Constant'
overland_slopingslab_DWE.PhaseSources.water.GeomNames = 'domain'
overland_slopingslab_DWE.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

overland_slopingslab_DWE.Solver = 'Richards'
overland_slopingslab_DWE.Solver.MaxIter = 2500

overland_slopingslab_DWE.Solver.Nonlinear.MaxIter = 50
overland_slopingslab_DWE.Solver.Nonlinear.ResidualTol = 1e-9
overland_slopingslab_DWE.Solver.Nonlinear.EtaChoice = 'EtaConstant'
overland_slopingslab_DWE.Solver.Nonlinear.EtaValue = 0.01
overland_slopingslab_DWE.Solver.Nonlinear.UseJacobian = False
#pfset Solver.Nonlinear.UseJacobian                       True

overland_slopingslab_DWE.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland_slopingslab_DWE.Solver.Nonlinear.StepTol = 1e-20
overland_slopingslab_DWE.Solver.Nonlinear.Globalization = 'LineSearch'
overland_slopingslab_DWE.Solver.Linear.KrylovDimension = 20
overland_slopingslab_DWE.Solver.Linear.MaxRestart = 2

overland_slopingslab_DWE.Solver.Linear.Preconditioner = 'PFMG'
overland_slopingslab_DWE.Solver.PrintSubsurf = False
overland_slopingslab_DWE.Solver.Drop = 1E-20
overland_slopingslab_DWE.Solver.AbsTol = 1E-10

overland_slopingslab_DWE.Solver.OverlandDiffusive.Epsilon = 1E-5

#pfset Solver.Linear.Preconditioner.PCMatrixType         FullJacobian

overland_slopingslab_DWE.Solver.WriteSiloSubsurfData = False
overland_slopingslab_DWE.Solver.WriteSiloPressure = False
overland_slopingslab_DWE.Solver.WriteSiloSlopes = False

overland_slopingslab_DWE.Solver.WriteSiloSaturation = False
overland_slopingslab_DWE.Solver.WriteSiloConcentration = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland_slopingslab_DWE.ICPressure.Type = 'HydroStaticPatch'
overland_slopingslab_DWE.ICPressure.GeomNames = 'domain'
overland_slopingslab_DWE.Geom.domain.ICPressure.Value = -3.0

overland_slopingslab_DWE.Geom.domain.ICPressure.RefGeom = 'domain'
overland_slopingslab_DWE.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# Running all 8 direction combinations with the upwind formulation on and off (i.e. 16 total)
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
overland_slopingslab_DWE.TopoSlopesX.Type = "Constant"
overland_slopingslab_DWE.TopoSlopesX.GeomNames = "domain"
overland_slopingslab_DWE.TopoSlopesX.Geom.domain.Value = 0.01

overland_slopingslab_DWE.TopoSlopesY.Type = "Constant"
overland_slopingslab_DWE.TopoSlopesY.GeomNames = "domain"
overland_slopingslab_DWE.TopoSlopesY.Geom.domain.Value = 0.01

#   #original approach from K&M AWR 2006
overland_slopingslab_DWE.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland_slopingslab_DWE.Solver.Nonlinear.UseJacobian = False
overland_slopingslab_DWE.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'

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
overland_slopingslab_DWE.run()
