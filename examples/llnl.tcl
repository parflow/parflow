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
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input background_input concen_region_input layers"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X                        1641506.0
pfset Geom.domain.Lower.Y                        423484.7
pfset Geom.domain.Lower.Z                        230.0

pfset Geom.domain.Upper.X                        1654206.0
pfset Geom.domain.Upper.Y                        436184.7
pfset Geom.domain.Upper.Z                        550.0

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Background Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#-----------------------------------------------------------------------------
# Background Geometry
#-----------------------------------------------------------------------------
pfset Geom.background.Lower.X -999999999.0
pfset Geom.background.Lower.Y -999999999.0
pfset Geom.background.Lower.Z -999999999.0

pfset Geom.background.Upper.X  999999999.0
pfset Geom.background.Upper.Y  999999999.0
pfset Geom.background.Upper.Z  999999999.0

#-----------------------------------------------------------------------------
# Concen_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.concen_region_input.InputType       Box
pfset GeomInput.concen_region_input.GeomName        concen_region

#-----------------------------------------------------------------------------
# Concen_Region Geometry
#-----------------------------------------------------------------------------

pfset Geom.concen_region.Lower.X   1647203.0
pfset Geom.concen_region.Lower.Y   431721.0
pfset Geom.concen_region.Lower.Z   360.0

pfset Geom.concen_region.Upper.X  1649803.0 
pfset Geom.concen_region.Upper.Y  433221.0 
pfset Geom.concen_region.Upper.Z    420.0

#-----------------------------------------------------------------------------
# Layer Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.layers.InputType       SolidFile
pfset GeomInput.layers.GeomNames       "ground layer1b layer2 layer3a layer3b \
	layer4 layer5 clay uplifted"

pfset Geom.ground.Patches ""
pfset GeomInput.layers.FileName "llnl.pfsol"

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X                1641506.0
pfset ComputationalGrid.Lower.Y                 423484.7
pfset ComputationalGrid.Lower.Z                 230.0

pfset ComputationalGrid.DX	                 384.84848484848487
pfset ComputationalGrid.DY                       384.84848484848487
pfset ComputationalGrid.DZ	                 18.823529411764707

pfset ComputationalGrid.NX                      33
pfset ComputationalGrid.NY                      33
pfset ComputationalGrid.NZ                      17 

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background ground layer1b layer2 layer3a layer3b layer4 \
	layer5 clay uplifted"

#
# Constant values for vizing
#
#  pfset Geom.ground.Perm.Type Constant
#  pfset Geom.ground.Perm.Value 1.0
#  pfset Geom.layer1b.Perm.Type Constant
#  pfset Geom.layer1b.Perm.Value 2.0
#  pfset Geom.layer2.Perm.Type Constant
#  pfset Geom.layer2.Perm.Value 3.0
#  pfset Geom.layer3a.Perm.Type Constant
#  pfset Geom.layer3a.Perm.Value 4.0
#  pfset Geom.layer3b.Perm.Type Constant
#  pfset Geom.layer3b.Perm.Value 5.0
#  pfset Geom.layer4.Perm.Type Constant
#  pfset Geom.layer4.Perm.Value 6.0
#  pfset Geom.layer5.Perm.Type Constant
#  pfset Geom.layer5.Perm.Value 7.0
#  pfset Geom.clay.Perm.Type Constant
#  pfset Geom.clay.Perm.Value 8.0
#  pfset Geom.uplifted.Perm.Type Constant
#  pfset Geom.uplifted.Perm.Value 9.0

# Background
pfset Geom.background.Perm.Type Constant
pfset Geom.background.Perm.Value 0.00001

pfset Geom.ground.Perm.Type "TurnBands"
pfset Geom.ground.Perm.LambdaX 200.0
pfset Geom.ground.Perm.LambdaY 200.0
pfset Geom.ground.Perm.LambdaZ 10.0
pfset Geom.ground.Perm.GeomMean 4.56
pfset Geom.ground.Perm.Sigma   2.08
pfset Geom.ground.Perm.NumLines 100
pfset Geom.ground.Perm.RZeta 5.0
pfset Geom.ground.Perm.KMax 100.0
pfset Geom.ground.Perm.DelK 0.2
pfset Geom.ground.Perm.Seed 1
pfset Geom.ground.Perm.LogNormal LogTruncated
pfset Geom.ground.Perm.StratType Bottom
pfset Geom.ground.Perm.LowCutoff    0.0
pfset Geom.ground.Perm.HighCutoff 100.0

pfset Geom.layer1b.Perm.Type "TurnBands"
pfset Geom.layer1b.Perm.LambdaX 200.0
pfset Geom.layer1b.Perm.LambdaY 200.0
pfset Geom.layer1b.Perm.LambdaZ 10.0
pfset Geom.layer1b.Perm.GeomMean 5.92
pfset Geom.layer1b.Perm.Sigma   1.55
pfset Geom.layer1b.Perm.NumLines 100
pfset Geom.layer1b.Perm.RZeta 5.0
pfset Geom.layer1b.Perm.KMax 100.0
pfset Geom.layer1b.Perm.DelK 0.2
pfset Geom.layer1b.Perm.Seed 2
pfset Geom.layer1b.Perm.LogNormal LogTruncated
pfset Geom.layer1b.Perm.StratType Bottom
pfset Geom.layer1b.Perm.LowCutoff    0.0
pfset Geom.layer1b.Perm.HighCutoff 100.0

pfset Geom.layer2.Perm.Type "TurnBands"
pfset Geom.layer2.Perm.LambdaX 200.0
pfset Geom.layer2.Perm.LambdaY 200.0
pfset Geom.layer2.Perm.LambdaZ 10.0
pfset Geom.layer2.Perm.GeomMean 1.98
pfset Geom.layer2.Perm.Sigma   1.43
pfset Geom.layer2.Perm.NumLines 100
pfset Geom.layer2.Perm.RZeta 5.0
pfset Geom.layer2.Perm.KMax 100.0
pfset Geom.layer2.Perm.DelK 0.2
pfset Geom.layer2.Perm.Seed 3
pfset Geom.layer2.Perm.LogNormal LogTruncated
pfset Geom.layer2.Perm.StratType Bottom
pfset Geom.layer2.Perm.LowCutoff    0.0
pfset Geom.layer2.Perm.HighCutoff 100.0

pfset Geom.layer3a.Perm.Type "TurnBands"
pfset Geom.layer3a.Perm.LambdaX 200.0
pfset Geom.layer3a.Perm.LambdaY 200.0
pfset Geom.layer3a.Perm.LambdaZ 10.0
pfset Geom.layer3a.Perm.GeomMean 1.47
pfset Geom.layer3a.Perm.Sigma   2.29
pfset Geom.layer3a.Perm.NumLines 100
pfset Geom.layer3a.Perm.RZeta 5.0
pfset Geom.layer3a.Perm.KMax 100.0
pfset Geom.layer3a.Perm.DelK 0.2
pfset Geom.layer3a.Perm.Seed 4
pfset Geom.layer3a.Perm.LogNormal LogTruncated
pfset Geom.layer3a.Perm.StratType Bottom
pfset Geom.layer3a.Perm.LowCutoff    0.0
pfset Geom.layer3a.Perm.HighCutoff 100.0

pfset Geom.layer3b.Perm.Type "TurnBands"
pfset Geom.layer3b.Perm.LambdaX 200.0
pfset Geom.layer3b.Perm.LambdaY 200.0
pfset Geom.layer3b.Perm.LambdaZ 10.0
pfset Geom.layer3b.Perm.GeomMean 1.86
pfset Geom.layer3b.Perm.Sigma   2.29
pfset Geom.layer3b.Perm.NumLines 100
pfset Geom.layer3b.Perm.RZeta 5.0
pfset Geom.layer3b.Perm.KMax 100.0
pfset Geom.layer3b.Perm.DelK 0.2
pfset Geom.layer3b.Perm.Seed 5
pfset Geom.layer3b.Perm.LogNormal LogTruncated
pfset Geom.layer3b.Perm.StratType Bottom
pfset Geom.layer3b.Perm.LowCutoff    0.0
pfset Geom.layer3b.Perm.HighCutoff 100.0

pfset Geom.layer4.Perm.Type "TurnBands"
pfset Geom.layer4.Perm.LambdaX 200.0
pfset Geom.layer4.Perm.LambdaY 200.0
pfset Geom.layer4.Perm.LambdaZ 10.0
pfset Geom.layer4.Perm.GeomMean 2.55
pfset Geom.layer4.Perm.Sigma   1.45
pfset Geom.layer4.Perm.NumLines 100
pfset Geom.layer4.Perm.RZeta 5.0
pfset Geom.layer4.Perm.KMax 100.0
pfset Geom.layer4.Perm.DelK 0.2
pfset Geom.layer4.Perm.Seed 6
pfset Geom.layer4.Perm.LogNormal LogTruncated
pfset Geom.layer4.Perm.StratType Bottom
pfset Geom.layer4.Perm.LowCutoff    0.0
pfset Geom.layer4.Perm.HighCutoff 100.0

pfset Geom.layer5.Perm.Type "TurnBands"
pfset Geom.layer5.Perm.LambdaX 200.0
pfset Geom.layer5.Perm.LambdaY 200.0
pfset Geom.layer5.Perm.LambdaZ 10.0
pfset Geom.layer5.Perm.GeomMean 1.39
pfset Geom.layer5.Perm.Sigma   1.98
pfset Geom.layer5.Perm.NumLines 100
pfset Geom.layer5.Perm.RZeta 5.0
pfset Geom.layer5.Perm.KMax 100.0
pfset Geom.layer5.Perm.DelK 0.2
pfset Geom.layer5.Perm.Seed 7
pfset Geom.layer5.Perm.LogNormal LogTruncated
pfset Geom.layer5.Perm.StratType Bottom
pfset Geom.layer5.Perm.LowCutoff    0.0
pfset Geom.layer5.Perm.HighCutoff 100.0

pfset Geom.clay.Perm.Type "TurnBands"
pfset Geom.clay.Perm.LambdaX 200.0
pfset Geom.clay.Perm.LambdaY 200.0
pfset Geom.clay.Perm.LambdaZ 10.0
pfset Geom.clay.Perm.GeomMean 0.1
pfset Geom.clay.Perm.Sigma   1.46
pfset Geom.clay.Perm.NumLines 100
pfset Geom.clay.Perm.RZeta 5.0
pfset Geom.clay.Perm.KMax 100.0
pfset Geom.clay.Perm.DelK 0.2
pfset Geom.clay.Perm.Seed 8
pfset Geom.clay.Perm.LogNormal LogTruncated
pfset Geom.clay.Perm.StratType Top
pfset Geom.clay.Perm.LowCutoff    0.0
pfset Geom.clay.Perm.HighCutoff 100.0

pfset Geom.uplifted.Perm.Type "Constant"
pfset Geom.uplifted.Perm.Value 0.004

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
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0

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
pfset Contaminants.Names			"tce1"
pfset Contaminants.tce1.Degradation.Value	 0.0

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
pfset TimingInfo.DumpInterval	       10

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

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names "davy wente"

pfset Wells.davy.InputType              Vertical
pfset Wells.davy.Cycle		    constant
pfset Wells.davy.Action		    Extraction
pfset Wells.davy.Type		    Flux
pfset Wells.davy.X			   1645884.9 
pfset Wells.davy.Y			   425455.75
pfset Wells.davy.ZLower	             430.0
pfset Wells.davy.ZUpper	             445.0
pfset Wells.davy.Method		    Standard
pfset Wells.davy.alltime.Flux.water.Value 9600

pfset Wells.wente.InputType              Vertical
pfset Wells.wente.Cycle		    constant
pfset Wells.wente.Action		    Extraction
pfset Wells.wente.Type		    Flux
pfset Wells.wente.X			   1645115.5
pfset Wells.wente.Y			   427327.86
pfset Wells.wente.ZLower	             325.0
pfset Wells.wente.ZUpper	             445.0
pfset Wells.wente.Method		    Standard
pfset Wells.wente.alltime.Flux.water.Value 9600

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


#=============================================================================
#
# Use the capabilities of the TCL scripting to set up the arrays.
#
# This helper routine is used to construct the piecewise linear values from
# an array.  This should be easier to edit and maintain than manually
# entering each value as a pfset command.
#
# Input is the patch name and the array with location/value pairs.
proc CreateBCPLinearArray { patch array } {

    # Set is used to set a variable to a value (i.e. = in C)
    set i 0

    # The foreach command loops over a list, here we use two looping
    # variables at the same time to get the location/value pairs.
    foreach {location value} $array {
	pfset Patch.$patch.BCPressure.alltime.$i.Location  $location
	pfset Patch.$patch.BCPressure.alltime.$i.Value $value

	# Increment i by 1
	incr i
    }

    # Since we have the count of the number of points along the line
    # set it so we don't have to manually compute it.
    pfset Patch.$patch.BCPressure.alltime.NumPoints $i
}
#=============================================================================

pfset BCPressure.PatchNames "left right front back bottom top"

#-----------------------------------------------------------------------------
# Left boundary patch
#-----------------------------------------------------------------------------
pfset Patch.left.BCPressure.Type			DirEquilPLinear
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.alltime.XLower          1641506.0 
pfset Patch.left.BCPressure.alltime.YLower           423484.7
pfset Patch.left.BCPressure.alltime.XUpper          1641506.0
pfset Patch.left.BCPressure.alltime.YUpper           436184.7

set location_value_array { 0.0   576.0 \
  	0.042 570.0 \
  	0.095 560.0 \
  	0.126 550.0 \
  	0.163 540.0 \
  	0.2   530.0 \
	0.605 520.0 \
	1.0   511.0 }

CreateBCPLinearArray left $location_value_array

#-----------------------------------------------------------------------------
# Right boundary patch
#-----------------------------------------------------------------------------
pfset Patch.right.BCPressure.Type			DirEquilPLinear
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.alltime.XLower           1654206.0  
pfset Patch.right.BCPressure.alltime.YLower            423484.7  
pfset Patch.right.BCPressure.alltime.XUpper           1654206.0  
pfset Patch.right.BCPressure.alltime.YUpper            436184.7

set location_value_array { \
	0.0   665.0 \
	0.037 660.0 \
	0.074 650.0 \
	0.116 640.0 \
	0.153 630.0 \
	0.184 620.0 \
	0.211 610.0 \
	0.253 600.0 \
	0.305 590.0 \
	0.337 580.0 \
	0.421 580.0 \
	0.505 590.0 \
	0.605 595.0 \
	0.721 590.0 \
	0.811 580.0 \
	1.0   578.0}

CreateBCPLinearArray right $location_value_array


#-----------------------------------------------------------------------------
# Front boundary patch
#-----------------------------------------------------------------------------
pfset Patch.front.BCPressure.Type			DirEquilPLinear
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.XLower             1641506.0   
pfset Patch.front.BCPressure.alltime.YLower              423484.7 
pfset Patch.front.BCPressure.alltime.XUpper             1654206.0  
pfset Patch.front.BCPressure.alltime.YUpper              423484.7  

set location_value_array { \
	0.0   576.0 \
	0.042 580.0 \
	0.284 590.0 \
	0.337 600.0 \
	0.411 590.0 \
	0.453 590.0 \
	0.484 600.0 \
	0.495 610.0 \
	0.505 620.0 \
	0.516 630.0 \
	0.526 640.0 \
	0.558 650.0 \
	0.589 660.0 \
	0.611 670.0 \
	0.653 680.0 \
	0.758 685.0 \
	0.884 680.0 \
	0.953 670.0 \
	1.0   665.0 }

CreateBCPLinearArray front $location_value_array

#-----------------------------------------------------------------------------
# Back boundary patch
#-----------------------------------------------------------------------------
pfset Patch.back.BCPressure.Type			DirEquilPLinear
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.XLower           1641506.0   
pfset Patch.back.BCPressure.alltime.YLower            436184.7 
pfset Patch.back.BCPressure.alltime.XUpper           1654206.0  
pfset Patch.back.BCPressure.alltime.YUpper            436184.7   

set location_value_array { \
	0.0   511.0 \
	0.163 520.0 \
	0.347 530.0 \
	0.589 540.0 \
	0.758 550.0 \
	0.853 560.0 \
	0.937 570.0 \
	1.0   578.0 }

CreateBCPLinearArray back $location_value_array


#-----------------------------------------------------------------------------
# Top and Bottom
#-----------------------------------------------------------------------------

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
pfset PhaseConcen.water.tce1.Geom.concen_region.Value 0.8

pfrun llnl
pfundist llnl

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST llnl
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    if ![pftestFile $TEST.out.press.00000.pfb "Max difference in Pressure" $sig_digits] {
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

    if ![pftestFile $TEST.out.concen.0.00.00100.pfsb "Max difference in concen" $sig_digits] {
	set passed 0
    }
    
    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
