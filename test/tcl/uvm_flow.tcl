# Test case submitted by:
# Joshua B. Kollat, Ph.D. 
# Research Associate
# Penn State University

#
# Tests running solver impes with no advection.
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

set runname uvm.flow

#set PARFLOW_DIR {/home/juk124/parflow/parflow}
set INPUT_DIR {.}
set OUTPUT_DIR {.}

#Base Units:
# Mass - g
# Length - m
# Time - days

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

#These are based on Donna's formulation using MODFLOW/MT3DMS
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

#These are based on Donna's formulation using MODFLOW/MT3DMS
pfset ComputationalGrid.NX                45
pfset ComputationalGrid.NY                63
pfset ComputationalGrid.NZ                33
#These are the UVM model resolutions
#pfset ComputationalGrid.NX                35
#pfset ComputationalGrid.NY                50
#pfset ComputationalGrid.NZ                33

#These are based on Donna's formulation using MODFLOW/MT3DMS
set   UpperX                              2.7
set   UpperY                              3.8
set   UpperZ                              2.0
#These are the UVM model limits:
#set   UpperX                              2.74295
#set   UpperY                              3.937
#set   UpperZ                              2.0066
#These are the true UVM tank dimensions
#set   UpperX                              2.540
#set   UpperY                              3.560
#set   UpperZ                              2.030

set   LowerX                              [pfget ComputationalGrid.Lower.X]
set   LowerY                              [pfget ComputationalGrid.Lower.Y]
set   LowerZ                              [pfget ComputationalGrid.Lower.Z]

set   NX                                  [pfget ComputationalGrid.NX]
set   NY                                  [pfget ComputationalGrid.NY]
set   NZ                                  [pfget ComputationalGrid.NZ]

pfset ComputationalGrid.DX                [expr ($UpperX - $LowerX) / $NX]
pfset ComputationalGrid.DY                [expr ($UpperY - $LowerY) / $NY]
pfset ComputationalGrid.DZ                [expr ($UpperZ - $LowerZ) / $NZ]

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain layer1 layer2 layer3 layer4 layer5 lens"

#-----------------------------------------------------------------------------
# Entire domain
#-----------------------------------------------------------------------------
pfset GeomInput.domain.InputType   Box
pfset GeomInput.domain.GeomName    domain

pfset Geom.domain.Lower.X          0.0 
pfset Geom.domain.Lower.Y          0.0
pfset Geom.domain.Lower.Z          0.0
pfset Geom.domain.Upper.X          $UpperX
pfset Geom.domain.Upper.Y          $UpperY
pfset Geom.domain.Upper.Z          $UpperZ

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Layer 1 - coarse sand
#-----------------------------------------------------------------------------
pfset GeomInput.layer1.InputType    Box
pfset GeomInput.layer1.GeomName     layer1

pfset Geom.layer1.Lower.X           0.0 
pfset Geom.layer1.Lower.Y           0.0
pfset Geom.layer1.Lower.Z           0.0
pfset Geom.layer1.Upper.X           $UpperX
pfset Geom.layer1.Upper.Y           $UpperY
pfset Geom.layer1.Upper.Z           0.51

pfset Geom.layer1.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Layer 2 - silt
#-----------------------------------------------------------------------------
pfset GeomInput.layer2.InputType     Box
pfset GeomInput.layer2.GeomName      layer2

pfset Geom.layer2.Lower.X            0.0 
pfset Geom.layer2.Lower.Y            0.0
pfset Geom.layer2.Lower.Z            0.51
pfset Geom.layer2.Upper.X            $UpperX
pfset Geom.layer2.Upper.Y            $UpperY
pfset Geom.layer2.Upper.Z            0.69

pfset Geom.layer2.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Layer 3 - medium sand
#-----------------------------------------------------------------------------
pfset GeomInput.layer3.InputType     Box
pfset GeomInput.layer3.GeomName      layer3

pfset Geom.layer3.Lower.X            0.0 
pfset Geom.layer3.Lower.Y            0.0
pfset Geom.layer3.Lower.Z            0.69
pfset Geom.layer3.Upper.X            $UpperX
pfset Geom.layer3.Upper.Y            $UpperY
pfset Geom.layer3.Upper.Z            1.14

pfset Geom.layer3.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Layer4 - medium sand
#-----------------------------------------------------------------------------
pfset GeomInput.layer4.InputType      Box
pfset GeomInput.layer4.GeomName       layer4

pfset Geom.layer4.Lower.X             0.0 
pfset Geom.layer4.Lower.Y             0.0
pfset Geom.layer4.Lower.Z             1.14
pfset Geom.layer4.Upper.X             $UpperX
pfset Geom.layer4.Upper.Y             $UpperY
pfset Geom.layer4.Upper.Z             1.60

pfset Geom.layer4.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Layer 5 - medium sand
#-----------------------------------------------------------------------------
pfset GeomInput.layer5.InputType      Box
pfset GeomInput.layer5.GeomName       layer5

pfset Geom.layer5.Lower.X             0.0 
pfset Geom.layer5.Lower.Y             0.0
pfset Geom.layer5.Lower.Z             1.60
pfset Geom.layer5.Upper.X             $UpperX
pfset Geom.layer5.Upper.Y             $UpperY
pfset Geom.layer5.Upper.Z             $UpperZ

pfset Geom.layer5.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Fine lens block - specify after layers to replace layer section
#-----------------------------------------------------------------------------
pfset GeomInput.lens.InputType  Box
pfset GeomInput.lens.GeomName   lens

pfset Geom.lens.Lower.X         0.86 
pfset Geom.lens.Lower.Y         1.34
pfset Geom.lens.Lower.Z         1.14
pfset Geom.lens.Upper.X         2.11
pfset Geom.lens.Upper.Y         2.68
pfset Geom.lens.Upper.Z         1.60

pfset Geom.lens.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
#pfset Geom.Perm.Names "domain"
pfset Geom.Perm.Names "layer1 layer2 layer3 layer4 layer5 lens"

#Note: Since gravity, viscosity, and density have been normalized to 1, we are
#dealing with hydraulic conductivity rather than permiability.  The value below
#have been specified as such.

#Note: Parflow.exe is running form the debug folder
pfset Perm.Conditioning.FileName $INPUT_DIR/uvm.flow.KConditioningPts.txt

#Gravel Layer
pfset Geom.layer1.Perm.Type      "TurnBands"
pfset Geom.layer1.Perm.LambdaX   0.30
pfset Geom.layer1.Perm.LambdaY   0.30
pfset Geom.layer1.Perm.LambdaZ   0.30
pfset Geom.layer1.Perm.GeomMean  134.0
pfset Geom.layer1.Perm.Sigma     21.4344
pfset Geom.layer1.Perm.NumLines  100
pfset Geom.layer1.Perm.RZeta     5.0
pfset Geom.layer1.Perm.KMax      100.0
pfset Geom.layer1.Perm.DelK      0.2
pfset Geom.layer1.Perm.MaxNPts   100
pfset Geom.layer1.Perm.MaxCpts   8
pfset Geom.layer1.Perm.LogNormal Normal
pfset Geom.layer1.Perm.StratType Bottom

#Silt Layer
pfset Geom.layer2.Perm.Type    "Constant"
#Note: You cannot specify a K value of 0, it just needs to be really small
pfset Geom.layer2.Perm.Value   0.00001

#Medium Sand Layer 3
pfset Geom.layer3.Perm.Type      "TurnBands"
pfset Geom.layer3.Perm.LambdaX   0.30
pfset Geom.layer3.Perm.LambdaY   0.30
pfset Geom.layer3.Perm.LambdaZ   0.30
pfset Geom.layer3.Perm.GeomMean  17.3542
pfset Geom.layer3.Perm.Sigma     1.6632
pfset Geom.layer3.Perm.NumLines  100
pfset Geom.layer3.Perm.RZeta     5.0
pfset Geom.layer3.Perm.KMax      100.0
pfset Geom.layer3.Perm.DelK      0.2
pfset Geom.layer3.Perm.MaxNPts   100
pfset Geom.layer3.Perm.MaxCpts   8
pfset Geom.layer3.Perm.LogNormal Normal
pfset Geom.layer3.Perm.StratType Bottom

#Medium Sand Layer 4
pfset Geom.layer4.Perm.Type     "TurnBands"
pfset Geom.layer4.Perm.LambdaX  0.30
pfset Geom.layer4.Perm.LambdaY  0.30
pfset Geom.layer4.Perm.LambdaZ  0.30
pfset Geom.layer4.Perm.GeomMean 18.1849
pfset Geom.layer4.Perm.Sigma    1.0392
pfset Geom.layer4.Perm.NumLines 100
pfset Geom.layer4.Perm.RZeta    5.0
pfset Geom.layer4.Perm.KMax     100.0
pfset Geom.layer4.Perm.DelK      0.2
pfset Geom.layer4.Perm.MaxNPts   100
pfset Geom.layer4.Perm.MaxCpts   8
pfset Geom.layer4.Perm.LogNormal Normal
pfset Geom.layer4.Perm.StratType Bottom

#Fine Sand Lens
pfset Geom.lens.Perm.Type        "TurnBands"
pfset Geom.lens.Perm.LambdaX     0.30
pfset Geom.lens.Perm.LambdaY     0.30
pfset Geom.lens.Perm.LambdaZ     0.30
pfset Geom.lens.Perm.GeomMean    10.89139
pfset Geom.lens.Perm.Sigma       2.0664
pfset Geom.lens.Perm.NumLines    100
pfset Geom.lens.Perm.RZeta       5.0
pfset Geom.lens.Perm.KMax        100.0
pfset Geom.lens.Perm.DelK        0.2
pfset Geom.lens.Perm.MaxNPts     100
pfset Geom.lens.Perm.MaxCpts     8
pfset Geom.lens.Perm.LogNormal   Normal
pfset Geom.lens.Perm.StratType   Bottom

#Medium Sand Layer 5
pfset Geom.layer5.Perm.Type       "TurnBands"
pfset Geom.layer5.Perm.LambdaX    0.30
pfset Geom.layer5.Perm.LambdaY    0.30
pfset Geom.layer5.Perm.LambdaZ    0.30
pfset Geom.layer5.Perm.GeomMean   14.142552
pfset Geom.layer5.Perm.Sigma      1.4112
pfset Geom.layer5.Perm.NumLines   100
pfset Geom.layer5.Perm.RZeta      5.0
pfset Geom.layer5.Perm.KMax       100.0
pfset Geom.layer5.Perm.DelK       0.2
pfset Geom.layer5.Perm.MaxNPts    100
pfset Geom.layer5.Perm.MaxCpts    8
pfset Geom.layer5.Perm.LogNormal  Normal
pfset Geom.layer5.Perm.StratType  Bottom

#K tensor is specified over the whole domain
pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it
pfset SpecificStorage.Type              Constant
pfset SpecificStorage.GeomNames         ""
pfset Geom.domain.SpecificStorage.Value 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names "water"

pfset Phase.water.Density.Type     Constant
pfset Phase.water.Density.Value    1.0

pfset Phase.water.Viscosity.Type   Constant
pfset Phase.water.Viscosity.Value  1.0

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity  1.0

#-----------------------------------------------------------------------------
# Contaminants - only needed is advecting
#-----------------------------------------------------------------------------
#This key is needed no matter what
pfset Contaminants.Names   ""

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

#This sets up the stress periods - i.e., the injection well is on for 360 
#steps and off the remaining 120.  Note: PF is smart about this.  It only dumps 
#output for step 360 when the well is turned off, regardless of the values
#entered below.
pfset TimingInfo.BaseUnit        1.0
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        20.0
pfset TimingInfo.DumpInterval     1.0

#-----------------------------------------------------------------------------
# Porosity - set similar to MODFLOW/MT3DMS
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames      "domain layer1 layer2"

#Define a porosity of 0.3 throughout the entire domain
pfset Geom.domain.Porosity.Type    Constant
pfset Geom.domain.Porosity.Value   0.300

#Replace layer 1 (gravel) with...
pfset Geom.layer1.Porosity.Type    Constant
pfset Geom.layer1.Porosity.Value   0.600

#Replace layer 2 (silt) with...
pfset Geom.layer2.Porosity.Type    Constant
pfset Geom.layer2.Porosity.Value   0.320

#-----------------------------------------------------------------------------
# Domain - specifies which geometry is the problem domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        0.005
pfset Geom.domain.RelPerm.N            2.0    

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     0.005
pfset Geom.domain.Saturation.N         2.0
pfset Geom.domain.Saturation.SRes      0.2
pfset Geom.domain.Saturation.SSat      0.99

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type    Constant
pfset Phase.water.Mobility.Value   1.0

#-----------------------------------------------------------------------------
# Wells - tracer source well
#-----------------------------------------------------------------------------

#This key is needed no matter what
#pfset Wells.Names ""

#Here we define the source well for the tracer - confirmed location on 9-25-08

pfset Wells.Names "B4"
pfset Wells.B4.InputType                         Vertical
pfset Wells.B4.Action                            Injection
pfset Wells.B4.Type                              Flux
pfset Wells.B4.X                                 1.326
pfset Wells.B4.Y                                 0.495
#Just assuming one cell for source
pfset Wells.B4.ZUpper                            1.35
pfset Wells.B4.ZLower                            1.25
pfset Wells.B4.Method                            Standard
pfset Wells.B4.Cycle                             "onoff"
# 
#Source on
# 1500 cm^3/hr or 1.5 L/hr or 0.036 m^3/day
pfset Wells.B4.on.Flux.water.Value               0.036
pfset Wells.B4.on.Saturation.water.Value         1.0
#Source off
pfset Wells.B4.off.Flux.water.Value              0.0
pfset Wells.B4.off.Saturation.water.Value        1.0
# 
#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant onoff"

pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      1
pfset Cycle.constant.Repeat             -1

pfset Cycle.onoff.Names                  "on off"
pfset Cycle.onoff.on.Length              15
#Use this for getting the output
set CycleOffOutputFileNum                "00015"
pfset Cycle.onoff.off.Length             5
pfset Cycle.onoff.Repeat                -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames "left right front back bottom top"

#Front boundary - constant head reservoir
pfset Patch.front.BCPressure.Type             DirEquilRefPatch
pfset Patch.front.BCPressure.Cycle            "constant"
pfset Patch.front.BCPressure.RefGeom          domain
pfset Patch.front.BCPressure.RefPatch         bottom
pfset Patch.front.BCPressure.alltime.Value    2.032

#Back boundary - constant head reservoir
pfset Patch.back.BCPressure.Type              DirEquilRefPatch
pfset Patch.back.BCPressure.Cycle             "constant"
pfset Patch.back.BCPressure.RefGeom           domain
pfset Patch.back.BCPressure.RefPatch          bottom
pfset Patch.back.BCPressure.alltime.Value     2.007

#Left boundary - no flow
pfset Patch.left.BCPressure.Type              FluxConst
pfset Patch.left.BCPressure.Cycle             "constant"
pfset Patch.left.BCPressure.alltime.Value     0.0

#Right boundary - no flow
pfset Patch.right.BCPressure.Type             FluxConst
pfset Patch.right.BCPressure.Cycle            "constant"
pfset Patch.right.BCPressure.alltime.Value    0.0

#Bottom boundary - no flow
pfset Patch.bottom.BCPressure.Type            FluxConst
pfset Patch.bottom.BCPressure.Cycle           "constant"
pfset Patch.bottom.BCPressure.alltime.Value   0.0

#Top boundary - no flow
pfset Patch.top.BCPressure.Type               FluxConst
pfset Patch.top.BCPressure.Cycle              "constant"
pfset Patch.top.BCPressure.alltime.Value      0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

pfset TopoSlopesX.Type              "Constant"
pfset TopoSlopesX.GeomNames         ""
pfset TopoSlopesX.Geom.domain.Value 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type              "Constant"
pfset TopoSlopesY.GeomNames         ""
pfset TopoSlopesY.Geom.domain.Value 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

pfset Mannings.Type               "Constant"
pfset Mannings.GeomNames          ""
pfset Mannings.Geom.domain.Value  0.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                 Constant
pfset PhaseSources.water.GeomNames            domain
pfset PhaseSources.water.Geom.domain.Value    0.0

#-----------------------------------------------------------------------------
#  Solver Impes  
#-----------------------------------------------------------------------------

pfset Solver.AbsTol             1E-20

pfset Solver.PrintSubsurf       True
pfset Solver.PrintPressure      True
pfset Solver.PrintVelocities    True
pfset Solver.PrintSaturation    True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

#Loop through runs
set k 1

##########################
# Seeds
##########################

#Set the random seed to be different for every run
pfset Geom.layer1.Perm.Seed  [ expr 23586+2*$k ] 
pfset Geom.layer2.Perm.Seed  [ expr 71649+2*$k ] 
pfset Geom.layer3.Perm.Seed  [ expr 46382+2*$k ] 
pfset Geom.layer4.Perm.Seed  [ expr 54987+2*$k ] 
pfset Geom.layer5.Perm.Seed  [ expr 93216+2*$k ] 
pfset Geom.lens.Perm.Seed    [ expr 61329+2*$k ] 

##########################
# Run
##########################
pfrun    $runname
pfundist $runname

#
# Tests 
#
source pftest.tcl
set passed 1

# This test is not as stable as some of the others.
# Difference between optimized and debug version is this large.
# The absolute error is very small however so use a test the 
# checks not only the sig digits but also the absolute value of the difference.
set sig_digits 5
set abs_diff 1e-12

if ![pftestFile $runname.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
	set passed 0
    }

    if ![pftestFileWithAbs $runname.out.phasex.0.$i.pfb "Max difference in x-velocity at timestep $i" $sig_digits $abs_diff] {
	set passed 0
    }

    if ![pftestFileWithAbs $runname.out.phasey.0.$i.pfb "Max difference in y-velocity at timestep $i" $sig_digits $abs_diff] {
	set passed 0
    }

    if ![pftestFileWithAbs $runname.out.phasez.0.$i.pfb "Max difference in z-velocity at timestep $i" $sig_digits $abs_diff] {
	set passed 0
    }
}

if $passed {
    puts "$runname : PASSED"
} {
    puts "$runname : FAILED"
}
