# this runs the Cape Cod site flow case for the Harvey and Garabedian bacterial
# injection experiment from Maxwell, et al, 2007.

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
pfset ComputationalGrid.Lower.X                0.0
pfset ComputationalGrid.Lower.Y                0.0
pfset ComputationalGrid.Lower.Z                 0.0

pfset ComputationalGrid.DX	                 0.34
pfset ComputationalGrid.DY                      0.34
pfset ComputationalGrid.DZ	                 0.038

pfset ComputationalGrid.NX                      50
pfset ComputationalGrid.NY                      30
pfset ComputationalGrid.NZ                      100

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input upper_aquifer_input lower_aquifer_input"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X                        0.0
pfset Geom.domain.Lower.Y                        0.0
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                        17.0
pfset Geom.domain.Upper.Y                        10.2
pfset Geom.domain.Upper.Z                        3.8

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Upper Aquifer Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.upper_aquifer_input.InputType            Box
pfset GeomInput.upper_aquifer_input.GeomName             upper_aquifer

#-----------------------------------------------------------------------------
# Upper Aquifer Geometry
#-----------------------------------------------------------------------------
pfset Geom.upper_aquifer.Lower.X                        0.0
pfset Geom.upper_aquifer.Lower.Y                        0.0
pfset Geom.upper_aquifer.Lower.Z                        1.5
#pfset Geom.upper_aquifer.Lower.Z                        0.0

pfset Geom.upper_aquifer.Upper.X                        17.0
pfset Geom.upper_aquifer.Upper.Y                        10.2
pfset Geom.upper_aquifer.Upper.Z                        3.8

#-----------------------------------------------------------------------------
# Lower Aquifer Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.lower_aquifer_input.InputType            Box
pfset GeomInput.lower_aquifer_input.GeomName             lower_aquifer

#-----------------------------------------------------------------------------
# Lower Aquifer Geometry
#-----------------------------------------------------------------------------
pfset Geom.lower_aquifer.Lower.X                        0.0
pfset Geom.lower_aquifer.Lower.Y                        0.0
pfset Geom.lower_aquifer.Lower.Z                        0.0

pfset Geom.lower_aquifer.Upper.X                        17.0
pfset Geom.lower_aquifer.Upper.Y                        10.2
pfset Geom.lower_aquifer.Upper.Z                        1.5


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "upper_aquifer lower_aquifer"
# we open a file, in this case from PEST to set upper and lower kg and sigma
#
set fileId [open ../input/stats4.txt r 0600]
set kgu [gets $fileId]
set varu [gets $fileId]
set kgl [gets $fileId]
set varl [gets $fileId]
close $fileId


## we use the parallel turning bands formulation in ParFlow to simulate
## GRF for upper and lower aquifer
##


pfset Geom.upper_aquifer.Perm.LambdaX  3.60
pfset Geom.upper_aquifer.Perm.LambdaY  3.60
pfset Geom.upper_aquifer.Perm.LambdaZ  0.19
pfset Geom.upper_aquifer.Perm.GeomMean  112.00

pfset Geom.upper_aquifer.Perm.Sigma   1.0
pfset Geom.upper_aquifer.Perm.Sigma   0.48989794
pfset Geom.upper_aquifer.Perm.NumLines 150
pfset Geom.upper_aquifer.Perm.MaxSearchRad  4
pfset Geom.upper_aquifer.Perm.RZeta  5.0
pfset Geom.upper_aquifer.Perm.KMax  100.0000001
pfset Geom.upper_aquifer.Perm.DelK  0.2
pfset Geom.upper_aquifer.Perm.Seed  33333
pfset Geom.upper_aquifer.Perm.LogNormal Log
pfset Geom.upper_aquifer.Perm.StratType Bottom

pfset Geom.lower_aquifer.Perm.LambdaX  3.60
pfset Geom.lower_aquifer.Perm.LambdaY  3.60
pfset Geom.lower_aquifer.Perm.LambdaZ  0.19

pfset Geom.lower_aquifer.Perm.GeomMean  77.0
pfset Geom.lower_aquifer.Perm.Sigma   1.0
pfset Geom.lower_aquifer.Perm.Sigma   0.48989794
pfset Geom.lower_aquifer.Perm.MaxSearchRad 4
pfset Geom.lower_aquifer.Perm.NumLines 150
pfset Geom.lower_aquifer.Perm.RZeta  5.0
pfset Geom.lower_aquifer.Perm.KMax  100.0000001
pfset Geom.lower_aquifer.Perm.DelK  0.2
pfset Geom.lower_aquifer.Perm.Seed  33333
pfset Geom.lower_aquifer.Perm.LogNormal Log
pfset Geom.lower_aquifer.Perm.StratType Bottom

pfset Geom.upper_aquifer.Perm.Seed 1
pfset Geom.upper_aquifer.Perm.MaxNPts 70.0
pfset Geom.upper_aquifer.Perm.MaxCpts 20

pfset Geom.lower_aquifer.Perm.Seed 1
pfset Geom.lower_aquifer.Perm.MaxNPts 70.0
pfset Geom.lower_aquifer.Perm.MaxCpts 20

#pfset Geom.lower_aquifer.Perm.Type "TurnBands"
#pfset Geom.upper_aquifer.Perm.Type "TurnBands"

# uncomment the lines below to run parallel gaussian instead
# of parallel turning bands

pfset Geom.lower_aquifer.Perm.Type "ParGauss"
pfset Geom.upper_aquifer.Perm.Type "ParGauss"

#pfset lower aqu and upper aq stats to pest/read in values

pfset Geom.upper_aquifer.Perm.GeomMean  $kgu
pfset Geom.upper_aquifer.Perm.Sigma  $varu

pfset Geom.lower_aquifer.Perm.GeomMean  $kgl
pfset Geom.lower_aquifer.Perm.Sigma  $varl


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
pfset Contaminants.Names			""


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		-1
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime            0.0
pfset TimingInfo.DumpInterval	       -1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          domain

pfset Geom.domain.Porosity.Type    Constant
pfset Geom.domain.Porosity.Value   0.390

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
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names ""


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
pfset Patch.left.BCPressure.alltime.Value		10.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		9.97501

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
pfset PhaseSources.water.GeomNames                    domain
pfset PhaseSources.water.Geom.domain.Value        0.0

#-----------------------------------------------------------------------------
#  Solver Impes
#-----------------------------------------------------------------------------
pfset Solver.MaxIter 50
pfset Solver.AbsTol  1E-10
pfset Solver.Drop   1E-15

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

# this script is setup to run 100 realizations, for testing we just run one
###set n_runs 100
set n_runs 1

#
#  Loop through runs
#
for {set k 1} {$k <= $n_runs} {incr k 1} {
#
# set the random seed to be different for every run
#
pfset Geom.upper_aquifer.Perm.Seed  [ expr 33333+2*$k ]
pfset Geom.lower_aquifer.Perm.Seed  [ expr 31313+2*$k ]



pfrun harvey_flow_pgs.$k
pfundist harvey_flow_pgs.$k

# we use pf tools to convert from pressure to head
# we could do a number of other things here like copy files to different format
set press [pfload harvey_flow_pgs.$k.out.press.pfb]
set head [pfhhead $press]
pfsave $head -pfb harvey_flow_pgs.$k.head.pfb
}

# this could run other tcl scripts now an example is below
#puts stdout "running SLIM"
#source bromide_trans.sm.tcl

#
# Tests
#
source pftest.tcl

set passed 1

if ![pftestFile harvey_flow_pgs.1.out.press.pfb "Max difference in Pressure" $sig_digits] {
    set passed 0
}

if ![pftestFile harvey_flow_pgs.1.out.porosity.pfb "Max difference in Porosity" $sig_digits] {
    set passed 0
}

if ![pftestFile harvey_flow_pgs.1.head.pfb "Max difference in Head" $sig_digits] {
    set passed 0
}

if ![pftestFile harvey_flow_pgs.1.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile harvey_flow_pgs.1.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile harvey_flow_pgs.1.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

if $passed {
    puts "harvey_flow_pgs.1 : PASSED"
} {
    puts "harvey_flow_pgs.1 : FAILED"
}
