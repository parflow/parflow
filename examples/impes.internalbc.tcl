# this runs the Cape Cod site flow case for the Harvey and Garabedian bacterial 
# injection experiment from Maxwell, et al, 2007.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

if { [info exists ::env(PARFLOW_HAVE_SILO) ] } {
    set HaveSilo 1
} else {
    set HaveSilo 0
}

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

pfset ComputationalGrid.DX	                 10.
pfset ComputationalGrid.DY                       10.
pfset ComputationalGrid.DZ	                 1.

pfset ComputationalGrid.NX                      100
pfset ComputationalGrid.NY                      100
pfset ComputationalGrid.NZ                      100

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input"


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

pfset Geom.domain.Upper.X                        1000.0
pfset Geom.domain.Upper.Y                        1000.
pfset Geom.domain.Upper.Z                        100.0

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "domain"

pfset Geom.domain.Perm.Type        Constant
pfset Geom.domain.Perm.Value       1.0


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
pfset Geom.domain.SpecificStorage.Value 1.0e-5

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

# we use the PLinear BC to set a linear range of head values on the X=0 face.

pfset Patch.left.BCPressure.Type			DirEquilPLinear
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.alltime.XLower               0.0
pfset Patch.left.BCPressure.alltime.YLower               0.0
pfset Patch.left.BCPressure.alltime.XUpper               0.0
pfset Patch.left.BCPressure.alltime.YUpper            1000.0
pfset Patch.left.BCPressure.alltime.NumPoints            5
pfset Patch.left.BCPressure.alltime.0.Location   	 0.0
pfset Patch.left.BCPressure.alltime.0.Value	        99.9
pfset Patch.left.BCPressure.alltime.1.Location   	 0.25
pfset Patch.left.BCPressure.alltime.1.Value	       100.0
pfset Patch.left.BCPressure.alltime.2.Location   	 0.5
pfset Patch.left.BCPressure.alltime.2.Value	       100.2
pfset Patch.left.BCPressure.alltime.3.Location   	 0.75
pfset Patch.left.BCPressure.alltime.3.Value	       100.6
pfset Patch.left.BCPressure.alltime.4.Location   	 1.0
pfset Patch.left.BCPressure.alltime.4.Value	       101.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		99.0

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
# Internal BC
#---------------------------------------------------------
pfset InternalBC.Names     "r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15"
# note that the pressure values are in head potential, that is the
# rho*g*z is subtracted from these values.
pfset InternalBC.r1.X                                  500.0
pfset InternalBC.r1.Y                                  150.0
pfset InternalBC.r1.Z                                   99.0
pfset InternalBC.r1.Value                                1.0

pfset InternalBC.r2.X                                  500.0
pfset InternalBC.r2.Y                                  140.0
pfset InternalBC.r2.Z                                   99.0
pfset InternalBC.r2.Value                               01.01

pfset InternalBC.r3.X                                  500.0
pfset InternalBC.r3.Y                                  130.0
pfset InternalBC.r3.Z                                   99.0
pfset InternalBC.r3.Value                                0.990

pfset InternalBC.r4.X                                  500.0
pfset InternalBC.r4.Y                                  120.0
pfset InternalBC.r4.Z                                   99.0
pfset InternalBC.r4.Value                                0.40

pfset InternalBC.r5.X                                  500.0
pfset InternalBC.r5.Y                                  110.0
pfset InternalBC.r5.Z                                   99.0
pfset InternalBC.r5.Value                                0.20

pfset InternalBC.r6.X                                  500.0
pfset InternalBC.r6.Y                                  100.0
pfset InternalBC.r6.Z                                   99.0
pfset InternalBC.r6.Value                                0.10

pfset InternalBC.r7.X                                  500.0
pfset InternalBC.r7.Y                                   90.0
pfset InternalBC.r7.Z                                   99.0
pfset InternalBC.r7.Value                                0.20

pfset InternalBC.r8.X                                  500.0
pfset InternalBC.r8.Y                                   80.0
pfset InternalBC.r8.Z                                   99.0
pfset InternalBC.r8.Value                                0.10

pfset InternalBC.r9.X                                  500.0
pfset InternalBC.r9.Y                                   70.0
pfset InternalBC.r9.Z                                   99.0
pfset InternalBC.r9.Value                                0.10

pfset InternalBC.r10.X                                  500.0
pfset InternalBC.r10.Y                                   60.0
pfset InternalBC.r10.Z                                   99.0
pfset InternalBC.r10.Value                                0.20

pfset InternalBC.r11.X                                  500.0
pfset InternalBC.r11.Y                                   50.0
pfset InternalBC.r11.Z                                   99.0
pfset InternalBC.r11.Value                                0.50

pfset InternalBC.r12.X                                  500.0
pfset InternalBC.r12.Y                                   40.0
pfset InternalBC.r12.Z                                   99.0
pfset InternalBC.r12.Value                                0.50

pfset InternalBC.r13.X                                  500.0
pfset InternalBC.r13.Y                                   30.0
pfset InternalBC.r13.Z                                   99.0
pfset InternalBC.r13.Value                                0.50

pfset InternalBC.r14.X                                  500.0
pfset InternalBC.r14.Y                                   20.0
pfset InternalBC.r14.Z                                   99.0
pfset InternalBC.r14.Value                                0.30

pfset InternalBC.r15.X                                  500.0
pfset InternalBC.r15.Y                                   10.0
pfset InternalBC.r15.Z                                   99.0
pfset InternalBC.r15.Value                                0.20

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
pfset Solver  Impes
pfset Solver.MaxIter 50
pfset Solver.AbsTol  1E-10
pfset Solver.Drop   1E-15

# we set all output to write as SILO in addition to pfb
# so we can visualize w/ VisIt
if $HaveSilo {
    pfset Solver.WriteSiloSubsurfData True
    pfset Solver.WriteSiloPressure True
    pfset Solver.WriteSiloSaturation True
    pfset Solver.WriteSiloConcentration True
}

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfrun impes.internalbc
pfundist impes.internalbc

# we use pf tools to convert from pressure to head
if $HaveSilo {
    set press [pfload impes.internalbc.out.press.silo]
    set head [pfhhead $press]
    pfsave $head -silo impes.internalbc.head.silo
}

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST impes.internalbc
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    if ![pftestFile $TEST.out.press.pfb "Max difference in Pressure" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.porosity.pfb "Max difference in Porosity" $sig_digits] {
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
    

    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}

