#
# This test is part of a series of tests for the DeepAquiferBC
# Here, we test a sloped slab domain with no flow on the sides.
# The bottom is the DeepAquiferBC and the top OverlandKinematic. 
# Water should flow from the effect of gravity and overland flow.
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

pfset ComputationalGrid.DX   2.0
pfset ComputationalGrid.DY   2.0
pfset ComputationalGrid.DZ   0.5

pfset ComputationalGrid.NX   25
pfset ComputationalGrid.NY   25
pfset ComputationalGrid.NZ   20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names   "domain_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType   "Box"
pfset GeomInput.domain_input.GeomName   "domain"

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName   "domain"

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X   0.0 
pfset Geom.domain.Lower.Y   0.0
pfset Geom.domain.Lower.Z   0.0

pfset Geom.domain.Upper.X   50.0
pfset Geom.domain.Upper.Y   50.0
pfset Geom.domain.Upper.Z   10.0

pfset Geom.domain.Patches   "left right front back bottom top"

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
pfset TimingInfo.BaseUnit   1.0
pfset TimingInfo.StartCount   0
pfset TimingInfo.StartTime   0.0
pfset TimingInfo.StopTime   8.0
pfset TimingInfo.DumpInterval   -1
pfset TimeStep.Type   "Constant"
pfset TimeStep.Value   1.0

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names   "constant rainfall"

pfset Cycle.constant.Names   "alltime"
pfset Cycle.constant.alltime.Length   1
pfset Cycle.constant.Repeat   -1

pfset Cycle.rainfall.Names   "rain sunny"
pfset Cycle.rainfall.rain.Length   1
pfset Cycle.rainfall.sunny.Length   2
pfset Cycle.rainfall.Repeat   -1

#-----------------------------------------------------------------------------
# Create DeepAquifer and OverlandFlow Files
#-----------------------------------------------------------------------------

set sx   -0.05
set sy   0.2

set elevations "./deep_aquifer_elevations.pfb"
set datafile   "./deep_aquifer_elevations.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    puts $data [expr 2 * $sx * ($ii + 0.5) + $sy * abs(2 * ($jj + 0.5) - 25)]
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $elevations
pfdist -nz 1 $elevations

#-----------------------------------------------------------------------------

set slopes_x   "./deep_aquifer_slopes_x.pfb"
set datafile   "./deep_aquifer_slopes_x.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    puts $data $sx
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $slopes_x
pfdist -nz 1 $slopes_x

#-----------------------------------------------------------------------------

set slopes_y   "./deep_aquifer_slopes_y.pfb"
set datafile   "./deep_aquifer_slopes_y.sa"
set data [open $datafile w]
puts $data "25 25 1"
for { set jj 0 } { $jj < 25 } { incr jj } {
  for { set ii 0 } { $ii < 25 } { incr ii } {
    if { [expr (2 * ($jj + 1.0) - 25) > 0] } {
      puts $data $sy
    } else {
      puts $data [expr -1 * $sy]
    }
  }
}
close $data
set data [pfload -sa $datafile]
pfsetgrid { 25 25 1 } { 0.0 0.0 0.0 } { 50.0 50.0 0.5 } $data
pfsave $data -pfb $slopes_y
pfdist -nz 1 $slopes_y

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames   "left right front back bottom top"

pfset Patch.left.BCPressure.Type   "FluxConst"
pfset Patch.left.BCPressure.Cycle   "constant"
pfset Patch.left.BCPressure.alltime.Value   0.0

pfset Patch.right.BCPressure.Type   "FluxConst"
pfset Patch.right.BCPressure.Cycle   "constant"
pfset Patch.right.BCPressure.alltime.Value   0.0

pfset Patch.front.BCPressure.Type   "FluxConst"
pfset Patch.front.BCPressure.Cycle   "constant"
pfset Patch.front.BCPressure.alltime.Value   0.0

pfset Patch.back.BCPressure.Type   "FluxConst"
pfset Patch.back.BCPressure.Cycle   "constant"
pfset Patch.back.BCPressure.alltime.Value   0.0

# input files for DeepAquifer created above
pfset Patch.bottom.BCPressure.Type   "DeepAquifer"
pfset Patch.bottom.BCPressure.Cycle   "constant"
pfset Patch.BCPressure.DeepAquifer.SpecificYield.Type   "Constant"
pfset Patch.BCPressure.DeepAquifer.SpecificYield.Value   0.1
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Type   "Constant"
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Value   90.0
pfset Patch.BCPressure.DeepAquifer.Permeability.Type   "SameAsBottomLayer"
pfset Patch.BCPressure.DeepAquifer.Elevations.Type   "PFBFile"
pfset Patch.BCPressure.DeepAquifer.Elevations.FileName   $elevations

pfset Patch.top.BCPressure.Type   "OverlandKinematic"
pfset Patch.top.BCPressure.Cycle   "rainfall"
pfset Patch.top.BCPressure.rain.Value   -0.05
pfset Patch.top.BCPressure.sunny.Value   0.01

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type   "HydroStaticPatch"
pfset ICPressure.GeomNames   "domain"
pfset Geom.domain.ICPressure.Value   -2
pfset Geom.domain.ICPressure.RefGeom   "domain"
pfset Geom.domain.ICPressure.RefPatch   "top"

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity   1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
pfset Geom.Porosity.GeomNames   "domain"
pfset Geom.domain.Porosity.Type   "Constant"
# Value for Silt soil
pfset Geom.domain.Porosity.Value   0.49

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names   "domain"
pfset Geom.domain.Perm.Type   "Constant"
# Value for Silt soil in m/hour
pfset Geom.domain.Perm.Value   0.05

pfset Perm.TensorType   "TensorByGeom"

pfset Geom.Perm.TensorByGeom.Names   "domain"

pfset Geom.domain.Perm.TensorValX   0.0
pfset Geom.domain.Perm.TensorValY   0.0
pfset Geom.domain.Perm.TensorValZ   1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names   "water"

pfset Phase.water.Density.Type   "Constant"
pfset Phase.water.Density.Value   1.0

pfset Phase.water.Viscosity.Type   "Constant"
pfset Phase.water.Viscosity.Value   1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type   "Constant"
pfset PhaseSources.water.GeomNames   "domain"
pfset PhaseSources.water.Geom.domain.Value   0.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------
pfset Phase.Saturation.Type   "VanGenuchten"
pfset Phase.Saturation.GeomNames   "domain"
pfset Geom.domain.Saturation.Alpha   0.65
pfset Geom.domain.Saturation.N   2.0
pfset Geom.domain.Saturation.SRes   0.10
pfset Geom.domain.Saturation.SSat   1.0


#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type   "VanGenuchten"
pfset Phase.RelPerm.GeomNames   "domain"
pfset Geom.domain.RelPerm.Alpha   0.65
pfset Geom.domain.RelPerm.N   2.00

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
pfset Mannings.Type   "Constant"
pfset Mannings.GeomNames   "domain"
pfset Mannings.Geom.domain.Value   5.5e-5

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type   "Constant"
pfset SpecificStorage.GeomNames   "domain"
pfset Geom.domain.SpecificStorage.Value   1.0e-4
 
#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
pfset TopoSlopesX.Type   "PFBFile"
pfset TopoSlopesX.GeomNames   "domain"
pfset TopoSlopesX.FileName   $slopes_x
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
pfset TopoSlopesY.Type   "PFBFile"
pfset TopoSlopesY.GeomNames   "domain"
pfset TopoSlopesY.FileName   $slopes_y

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names   ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames   ""

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names   ""

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution   "NoKnownSolution"

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver   "Richards"
pfset Solver.MaxIter   100000

pfset Solver.Nonlinear.MaxIter   250
pfset Solver.Nonlinear.ResidualTol   1e-10
pfset Solver.Nonlinear.EtaChoice   "EtaConstant"
pfset Solver.Nonlinear.EtaValue   1e-12
pfset Solver.Nonlinear.UseJacobian   True
pfset Solver.Nonlinear.StepTol   1e-16

pfset Solver.Linear.KrylovDimension   30

pfset Solver.Linear.Preconditioner   "MGSemi"
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter   1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels   10

pfset Solver.OverlandKinematic.Epsilon   1e-5

pfset Solver.PrintPressure   True
pfset Solver.PrintSubsurfData   False
pfset Solver.PrintSaturation   True
pfset Solver.PrintMask   False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
set runname deep_aquifer_vcatch_kin
pfrun $runname
pfundist $runname

pfundist $elevations
pfundist $slopes_x
pfundist $slopes_y

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