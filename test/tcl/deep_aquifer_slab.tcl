#
# This test is part of a series of tests for the DeepAquiferBC
# Here, we test a sloped slab domain with no flow on the sides
# and on the top. The bottom is the DeepAquiferBC. Water should
# flow from the effect of gravity.
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

set p [lindex $argv 0]
set q [lindex $argv 1]
set r [lindex $argv 2]

pfset Process.Topology.P $p
pfset Process.Topology.Q $q
pfset Process.Topology.R $r

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X   0.0
pfset ComputationalGrid.Lower.Y   0.0
pfset ComputationalGrid.Lower.Z   0.0

pfset ComputationalGrid.DX        2.0
pfset ComputationalGrid.DY        2.0
pfset ComputationalGrid.DZ        0.5

pfset ComputationalGrid.NX        25
pfset ComputationalGrid.NY        25
pfset ComputationalGrid.NZ        20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType    Box
pfset GeomInput.domain_input.GeomName     domain

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X         0.0 
pfset Geom.domain.Lower.Y         0.0
pfset Geom.domain.Lower.Z         0.0

pfset Geom.domain.Upper.X         50.0
pfset Geom.domain.Upper.Y         50.0
pfset Geom.domain.Upper.Z         10.0

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
pfset TimingInfo.BaseUnit       1.0
pfset TimingInfo.StartCount     0
pfset TimingInfo.StartTime      0.0
pfset TimingInfo.StopTime       24.0
pfset TimingInfo.DumpInterval  -3
pfset TimeStep.Type             Constant
pfset TimeStep.Value            1.0

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names                    "constant"

pfset Cycle.constant.Names           "alltime"
pfset Cycle.constant.alltime.Length   1
pfset Cycle.constant.Repeat          -1

#-----------------------------------------------------------------------------
# Create DeepAquifer Files
#-----------------------------------------------------------------------------

set permeability "./deep_aquifer_permeability.pfb"
set datafile     "./deep_aquifer_permeability.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    puts $data 0.02
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $permeability
pfdist -nz 1 $permeability

#-----------------------------------------------------------------------------

set specific_yield "./deep_aquifer_specific_yield.pfb"
set datafile       "./deep_aquifer_specific_yield.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    puts $data 0.1
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $specific_yield
pfdist -nz 1 $specific_yield

#-----------------------------------------------------------------------------

set elevations "./deep_aquifer_elevations.pfb"
set datafile   "./deep_aquifer_elevations.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    puts $data [expr 0.8 * ($ii + 0.5) + 0.6 * ($jj + 0.5)]
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $elevations
pfdist -nz 1 $elevations

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames    [pfget Geom.domain.Patches]

pfset Patch.left.BCPressure.Type             FluxConst
pfset Patch.left.BCPressure.Cycle            constant
pfset Patch.left.BCPressure.alltime.Value    0.0

pfset Patch.right.BCPressure.Type            FluxConst
pfset Patch.right.BCPressure.Cycle           constant
pfset Patch.right.BCPressure.alltime.Value   0.0

pfset Patch.front.BCPressure.Type            FluxConst
pfset Patch.front.BCPressure.Cycle           constant
pfset Patch.front.BCPressure.alltime.Value   0.0

pfset Patch.back.BCPressure.Type             FluxConst
pfset Patch.back.BCPressure.Cycle            constant
pfset Patch.back.BCPressure.alltime.Value    0.0

# input files for DeepAquifer created above
pfset Patch.bottom.BCPressure.Type           DeepAquifer
pfset Patch.bottom.BCPressure.Cycle          constant
pfset Patch.BCPressure.DeepAquifer.SpecificYield.Type      PFBFile
pfset Patch.BCPressure.DeepAquifer.SpecificYield.FileName  $specific_yield
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Type       Constant
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Value      90.0
pfset Patch.BCPressure.DeepAquifer.Permeability.Type       PFBFile
pfset Patch.BCPressure.DeepAquifer.Permeability.FileName   $permeability
pfset Patch.BCPressure.DeepAquifer.Elevations.Type         PFBFile
pfset Patch.BCPressure.DeepAquifer.Elevations.FileName     $elevations

pfset Patch.top.BCPressure.Type              FluxConst
pfset Patch.top.BCPressure.Cycle             constant
pfset Patch.top.BCPressure.alltime.Value     0.0

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type                        HydroStaticPatch
pfset ICPressure.GeomNames                   domain
pfset Geom.domain.ICPressure.Value           -5
pfset Geom.domain.ICPressure.RefGeom         domain
pfset Geom.domain.ICPressure.RefPatch        top

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity  1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
pfset Geom.Porosity.GeomNames       "domain"
pfset Geom.domain.Porosity.Type     Constant
# Value for Silt soil
pfset Geom.domain.Porosity.Value    0.49

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names               "domain"
pfset Geom.domain.Perm.Type         Constant
# Value for Silt soil in m/hour
pfset Geom.domain.Perm.Value        0.02

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  0.0
pfset Geom.domain.Perm.TensorValY  0.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names "water"

pfset Phase.water.Density.Type      Constant
pfset Phase.water.Density.Value     1.0

pfset Phase.water.Viscosity.Type    Constant
pfset Phase.water.Viscosity.Value   1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                Constant
pfset PhaseSources.water.GeomNames           domain
pfset PhaseSources.water.Geom.domain.Value   0.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------
pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     0.65
pfset Geom.domain.Saturation.N         2.00
pfset Geom.domain.Saturation.SRes      0.10
pfset Geom.domain.Saturation.SSat      1.0


#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        0.65
pfset Geom.domain.RelPerm.N            2.00

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
pfset Mannings.Type               "Constant"
pfset Mannings.GeomNames          "domain"
pfset Mannings.Geom.domain.Value  5.5e-5

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type               Constant
pfset SpecificStorage.GeomNames          "domain"
pfset Geom.domain.SpecificStorage.Value  1.0e-4
 
#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
pfset TopoSlopesX.Type               "Constant"
pfset TopoSlopesX.GeomNames          "domain"
pfset TopoSlopesX.Geom.domain.Value  0.0
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
pfset TopoSlopesY.Type               "Constant"
pfset TopoSlopesY.GeomNames          "domain"
pfset TopoSlopesY.Geom.domain.Value  0.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names          ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames  ""

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                 ""

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     100000

pfset Solver.Nonlinear.MaxIter                           250
pfset Solver.Nonlinear.ResidualTol                       1e-8
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-12
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.StepTol                           1e-16

pfset Solver.Linear.KrylovDimension                      30

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10

pfset Solver.PrintPressure     True
pfset Solver.PrintSubsurfData  False
pfset Solver.PrintSaturation   True
pfset Solver.PrintMask         False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
set runname deep_aquifer_slab
pfrun $runname
pfundist $runname

pfundist $permeability
pfundist $specific_yield
pfundist $elevations

#
# Tests 
#
source pftest.tcl

set sig_digits 6

set passed 1

foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits ../correct_output/$runname] {
	      set passed 0
    }
    if ![pftestFile $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits ../correct_output/$runname] {
	      set passed 0
    }
}

if $passed {
    puts "${runname} : PASSED"
} {
    puts "${runname} : FAILED"
}