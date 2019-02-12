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

pfset Process.Topology.P        1
pfset Process.Topology.Q        1
pfset Process.Topology.R        1

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
pfset GeomInput.Names "domain_input background_input concen_region_input"


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
# Concen_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.concen_region_input.InputType       Box
pfset GeomInput.concen_region_input.GeomName        concen_region

#-----------------------------------------------------------------------------
# Concen_Region Geometry
#-----------------------------------------------------------------------------

pfset Geom.concen_region.Lower.X   0.0
pfset Geom.concen_region.Lower.Y   15.0
pfset Geom.concen_region.Lower.Z    1.0

pfset Geom.concen_region.Upper.X   20.0
pfset Geom.concen_region.Upper.Y   35.0
pfset Geom.concen_region.Upper.Z    5.0

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
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	Constant
pfset Phase.water.Density.Value	1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "background"
pfset Geom.background.SpecificStorage.Value          1.0e-5

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "domain"
pfset TopoSlopesX.Geom.domain.Value 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "domain"
pfset TopoSlopesY.Geom.domain.Value 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 2.3e-7

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names			"tce1 tce2"
pfset Contaminants.tce1.Degradation.Value	 0.0
pfset Contaminants.tce2.Degradation.Value	 0.0

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
pfset TimingInfo.StopTime             100.0
pfset TimingInfo.DumpInterval	        2.0


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
pfset Geom.background.tce1.Retardation.Type     Linear
pfset Geom.background.tce1.Retardation.Rate     0.0
pfset Geom.background.tce2.Retardation.Type     Linear
pfset Geom.background.tce2.Retardation.Rate     0.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names "inj_test ext_test rec_test pre_test"

pfset Wells.inj_test.InputType              Vertical
pfset Wells.inj_test.Cycle		    onoff
pfset Wells.inj_test.Action		    Injection
pfset Wells.inj_test.Type		    Flux
pfset Wells.inj_test.X			     0.0 
pfset Wells.inj_test.Y			   150.0
pfset Wells.inj_test.ZLower	             1.0
pfset Wells.inj_test.ZUpper	             5.0
pfset Wells.inj_test.Method		    Standard
pfset Wells.inj_test.on.Flux.water.Value 20.0
pfset Wells.inj_test.on.Concentration.water.tce1.Value   0.0
pfset Wells.inj_test.on.Concentration.water.tce2.Value   0.9
pfset Wells.inj_test.on.Saturation.water.Value   1.0
pfset Wells.inj_test.off.Flux.water.Value 20.0
pfset Wells.inj_test.off.Concentration.water.tce1.Value   0.0
pfset Wells.inj_test.off.Concentration.water.tce2.Value   0.0
pfset Wells.inj_test.off.Saturation.water.Value   1.0


pfset Wells.ext_test.InputType              Vertical
pfset Wells.ext_test.Cycle		    constant
pfset Wells.ext_test.Action		    Extraction
pfset Wells.ext_test.Type		    Flux
pfset Wells.ext_test.X			    50.0 
pfset Wells.ext_test.Y			   160.0
pfset Wells.ext_test.ZLower	             1.0
pfset Wells.ext_test.ZUpper	             5.0
pfset Wells.ext_test.Method		    Standard
pfset Wells.ext_test.alltime.Flux.water.Value 20.0


pfset Wells.pre_test.InputType              Vertical
pfset Wells.pre_test.Cycle		    constant
pfset Wells.pre_test.Action		    Extraction
pfset Wells.pre_test.Type		    Pressure
pfset Wells.pre_test.X			   140.0 
pfset Wells.pre_test.Y			    90.0
pfset Wells.pre_test.ZLower	             1.0
pfset Wells.pre_test.ZUpper	             9.0
pfset Wells.pre_test.Method		    Standard
pfset Wells.pre_test.alltime.Pressure.Value  6.0

pfset Wells.rec_test.InputType              Recirc
pfset Wells.rec_test.Cycle		    constant
pfset Wells.rec_test.ExtractionType	    Flux
pfset Wells.rec_test.InjectionType          Flux
pfset Wells.rec_test.X			    50.0 
pfset Wells.rec_test.Y			    25.0
pfset Wells.rec_test.ExtractionZLower	     3.5
pfset Wells.rec_test.ExtractionZUpper	     4.5
pfset Wells.rec_test.InjectionZLower	     7.5
pfset Wells.rec_test.InjectionZUpper	     7.5
pfset Wells.rec_test.ExtractionMethod	    Standard
pfset Wells.rec_test.InjectionMethod        Standard
pfset Wells.rec_test.alltime.Extraction.Flux.water.Value       	       20.0
pfset Wells.rec_test.alltime.Injection.Flux.water.Value		       40.0
pfset Wells.rec_test.alltime.Injection.Concentration.water.tce1.Fraction   0.001
pfset Wells.rec_test.alltime.Injection.Concentration.water.tce2.Fraction   0.001

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant onoff"
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

pfset Cycle.onoff.Names                 "on off"
pfset Cycle.onoff.on.Length             2.0
pfset Cycle.onoff.off.Length            8.0
pfset Cycle.onoff.Repeat               -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames "left right front back bottom top"

pfset Patch.left.BCPressure.Type			DirEquilRefPatch
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.RefPatch			bottom
pfset Patch.left.BCPressure.alltime.Value		15.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		5.0

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


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0

pfset PhaseConcen.water.tce1.Type                     Constant
pfset PhaseConcen.water.tce1.GeomNames                concen_region
pfset PhaseConcen.water.tce1.Geom.concen_region.Value 0.9

pfset PhaseConcen.water.tce2.Type                     Constant
pfset PhaseConcen.water.tce2.GeomNames                concen_region
pfset PhaseConcen.water.tce2.Geom.concen_region.Value 0.0

pfrun well.01
pfundist well.01

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST well.01
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    if ![pftestFile $TEST.out.press.00045.pfb "Max difference in Pressure" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $TEST.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $TEST.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.porosity.pfb "Max difference in porosity" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.concen.0.00.00050.pfsb "Max difference in concen" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.concen.0.01.00050.pfsb "Max difference in concen" $sig_digits] {
	set passed 0
    }

    
    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
