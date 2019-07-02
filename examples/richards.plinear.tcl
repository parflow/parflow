# this runs the Cape Cod site flow case for the Harvey and Garabedian bacterial 
# injection experiment from Maxwell, et al, 2007.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

set runname richards.plinear 

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

pfset Process.Topology.P        2
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
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          "domain"
pfset Geom.domain.RelPerm.Alpha         .5
pfset Geom.domain.RelPerm.N             2. 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         "domain"
pfset Geom.domain.Saturation.Alpha        .5
pfset Geom.domain.Saturation.N            2.
pfset Geom.domain.Saturation.SRes         0.0001
pfset Geom.domain.Saturation.SSat         1.0

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
pfset TimingInfo.StartCount             0	
pfset TimingInfo.StartTime		0.0

# If testing only solve 2 timesteps, example runs for 20 timesteps
if { [info exists ::env(PF_TEST) ] } {
    pfset TimingInfo.StopTime           1.0
} {
    pfset TimingInfo.StopTime          10.0
}
pfset TimingInfo.DumpInterval	       -1
pfset TimeStep.Type                    Constant
pfset TimeStep.Value                    0.5

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
pfset Patch.left.BCPressure.alltime.0.Value	       49.9
pfset Patch.left.BCPressure.alltime.1.Location   	 0.25
pfset Patch.left.BCPressure.alltime.1.Value	       50.0
pfset Patch.left.BCPressure.alltime.2.Location   	 0.5
pfset Patch.left.BCPressure.alltime.2.Value	       50.2
pfset Patch.left.BCPressure.alltime.3.Location   	 0.75
pfset Patch.left.BCPressure.alltime.3.Value	       50.6
pfset Patch.left.BCPressure.alltime.4.Location   	 1.0
pfset Patch.left.BCPressure.alltime.4.Value	       51.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		49.0

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
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      50.0
pfset Geom.domain.ICPressure.RefPatch                   bottom 
pfset Geom.domain.ICPressure.RefGeom                    domain

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
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
#  Solver Richards 
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                             Richards
pfset Solver.MaxIter                                     25000
pfset Solver.TerrainFollowingGrid                    False 

pfset Solver.Nonlinear.MaxIter                           300
pfset Solver.Nonlinear.ResidualTol                       1e-6
pfset Solver.Nonlinear.EtaChoice                         Walker1 
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-12
pfset Solver.Nonlinear.StepTol				 1e-30
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      20
pfset Solver.Linear.MaxRestart                           2

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10
pfset Solver.PrintSubsurf				False
pfset  Solver.Drop                                      1E-20
pfset Solver.AbsTol                                     1E-12

# we set all output to write as SILO in addition to pfb
# so we can visualize w/ VisIt
if $HaveSilo {
    pfset Solver.WriteSiloSubsurfData                   True
    pfset Solver.WriteSiloPressure                      True
    pfset Solver.WriteSiloSaturation                    True
    pfset Solver.WriteSiloConcentration                 True
    pfset Solver.WriteSiloSlopes                        True
}

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfrun $runname 
pfundist $runname

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST richards.plinear
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
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
    
    for {set i 0} {$i < 2} {incr i} {
    	set fi [format "%05d" $i]
	if ![pftestFile $TEST.out.press.$fi.pfb "Max difference in pressure timestep $i" $sig_digits] {
	    set passed 0
	}
	if ![pftestFile $TEST.out.satur.$fi.pfb "Max difference in saturation timestep $i" $sig_digits] {
	    set passed 0
	}
    }

    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
