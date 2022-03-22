#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*


#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
pfset FileVersion 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

pfset Process.Topology.P        [lindex $argv 0]
pfset Process.Topology.Q        [lindex $argv 1]
pfset Process.Topology.R        [lindex $argv 2]

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X                -10.0
pfset ComputationalGrid.Lower.Y                 10.0
pfset ComputationalGrid.Lower.Z                  1.0

pfset ComputationalGrid.DX	                 8.8888888888888893
pfset ComputationalGrid.DY                      10.666666666666666
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                      18
pfset ComputationalGrid.NY                      15
pfset ComputationalGrid.NZ                       8

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input background_input source_region_input concen_region_input"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X                        -10.0 
pfset Geom.domain.Lower.Y                         10.0
pfset Geom.domain.Lower.Z                          1.0

pfset Geom.domain.Upper.X                        150.0
pfset Geom.domain.Upper.Y                        170.0
pfset Geom.domain.Upper.Z                          9.0

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Background Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#-----------------------------------------------------------------------------
# Background Geometry
#-----------------------------------------------------------------------------
pfset Geom.background.Lower.X -99999999.0
pfset Geom.background.Lower.Y -99999999.0
pfset Geom.background.Lower.Z -99999999.0

pfset Geom.background.Upper.X  99999999.0
pfset Geom.background.Upper.Y  99999999.0
pfset Geom.background.Upper.Z  99999999.0


#-----------------------------------------------------------------------------
# Source_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.source_region_input.InputType      Box
pfset GeomInput.source_region_input.GeomName       source_region

#-----------------------------------------------------------------------------
# Source_Region Geometry
#-----------------------------------------------------------------------------
pfset Geom.source_region.Lower.X    65.56
pfset Geom.source_region.Lower.Y    79.34
pfset Geom.source_region.Lower.Z     4.5

pfset Geom.source_region.Upper.X    74.44
pfset Geom.source_region.Upper.Y    89.99
pfset Geom.source_region.Upper.Z     5.5


#-----------------------------------------------------------------------------
# Concen_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.concen_region_input.InputType       Box
pfset GeomInput.concen_region_input.GeomName        concen_region

#-----------------------------------------------------------------------------
# Concen_Region Geometry
#-----------------------------------------------------------------------------
pfset Geom.concen_region.Lower.X   60.0
pfset Geom.concen_region.Lower.Y   80.0
pfset Geom.concen_region.Lower.Z    4.0

pfset Geom.concen_region.Upper.X   80.0
pfset Geom.concen_region.Upper.Y  100.0
pfset Geom.concen_region.Upper.Z    6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background"

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    4.0

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       ""
pfset Geom.domain.SpecificStorage.Value 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	Constant
pfset Phase.water.Density.Value	1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names			"tce"
pfset Contaminants.tce.Degradation.Value	 0.0

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime            1000.0
pfset TimingInfo.DumpInterval	       -1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          background

pfset Geom.background.Porosity.Type    Constant
pfset Geom.background.Porosity.Value   1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type        Constant
pfset Phase.water.Mobility.Value       1.0

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames           background
pfset Geom.background.tce.Retardation.Type     Linear
pfset Geom.background.tce.Retardation.Rate     0.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names snoopy

pfset Wells.snoopy.InputType                Recirc

pfset Wells.snoopy.Cycle		    constant

pfset Wells.snoopy.ExtractionType	    Flux
pfset Wells.snoopy.InjectionType            Flux

pfset Wells.snoopy.X			    71.0 
pfset Wells.snoopy.Y			    90.0
pfset Wells.snoopy.ExtractionZLower	     5.0
pfset Wells.snoopy.ExtractionZUpper	     5.0
pfset Wells.snoopy.InjectionZLower	     2.0
pfset Wells.snoopy.InjectionZUpper	     2.0

pfset Wells.snoopy.ExtractionMethod	    Standard
pfset Wells.snoopy.InjectionMethod          Standard

pfset Wells.snoopy.alltime.Extraction.Flux.water.Value        	     5.0
pfset Wells.snoopy.alltime.Injection.Flux.water.Value		     7.5
pfset Wells.snoopy.alltime.Injection.Concentration.water.tce.Fraction 0.1

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames "left right front back bottom top"

pfset Patch.left.BCPressure.Type			DirEquilRefPatch
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.RefPatch			bottom
pfset Patch.left.BCPressure.alltime.Value		14.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		9.0

pfset Patch.front.BCPressure.Type			FluxConst
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.Value		0.0

pfset Patch.back.BCPressure.Type			FluxConst
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.Value		0.0

pfset Patch.bottom.BCPressure.Type			FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.alltime.Value		0.0

pfset Patch.top.BCPressure.Type			        FluxConst
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.alltime.Value		0.0


#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames ""

pfset TopoSlopesX.Geom.domain.Value 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames ""

pfset TopoSlopesY.Geom.domain.Value 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames ""
pfset Mannings.Geom.domain.Value 0.

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0

pfset PhaseConcen.water.tce.Type                      Constant
pfset PhaseConcen.water.tce.GeomNames                 concen_region
pfset PhaseConcen.water.tce.Geom.concen_region.Value  0.8


pfset Solver.WriteSiloSubsurfData True
pfset Solver.WriteSiloPressure True
pfset Solver.WriteSiloSaturation True
pfset Solver.WriteSiloConcentration True

pfset Solver.PrintVelocities True

#-----------------------------------------------------------------------------
# The Solver Impes MaxIter default value changed so to get previous
# results we need to set it back to what it was
#-----------------------------------------------------------------------------
pfset Solver.MaxIter 5
pfset Solver.AbsTol 1e-25

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun default_single
pfundist default_single

# To run with debugging
# pfrun default_single -g {0 1}
# will debug process 0 and 1

#
# Tests 
#
source pftest.tcl

set sig_digits 4

set passed 1

if ![pftestFile default_single.out.press.00000.pfb "Max difference in Pressure" $sig_digits] {
    set passed 0
}

# use abs value test to prevent machine precision effects
set abs_value 1e-12
if ![pftestFileWithAbs default_single.out.phasex.0.00000.pfb "Max difference in x-velocity" $sig_digits $abs_value] {
    set passed 0
}

if ![pftestFileWithAbs default_single.out.phasey.0.00000.pfb "Max difference in y-velocity" $sig_digits $abs_value] {
    set passed 0
}

if ![pftestFileWithAbs default_single.out.phasez.0.00000.pfb "Max difference in z-velocity" $sig_digits $abs_value] {
    set passed 0
}

if ![pftestFile default_single.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile default_single.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile default_single.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001 00002 00003 00004 00005" {
    if ![pftestFile default_single.out.concen.0.00.$i.pfsb "Max difference in concen timestep $i" $sig_digits] {
    set passed 0
    }
}

if $passed {
    puts "default_single : PASSED"
} {
    puts "default_single : FAILED"
}
