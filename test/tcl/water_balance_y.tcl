#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR

set tcl_precision 16

set verbose 0

#---------------------------------------------------------
# Some controls for the test
#---------------------------------------------------------

#---------------------------------------------------------
# Control slopes 
#-1 = slope to lower-y
# 0 = flat top (no overland flow)
# 1 = slope to upper-y 
#---------------------------------------------------------
set use_slopes 1

#---------------------------------------------------------
# Flux on the top surface
#---------------------------------------------------------
set rain_flux -0.05
set rec_flux  0.0

#---------------------------------------------------------
# Import the ParFlow TCL package
#---------------------------------------------------------
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

#---------------------------------------------------------
# Name of the run
#---------------------------------------------------------
set runname water_balance

pfset FileVersion 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------
pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

pfset ComputationalGrid.NX                30
pfset ComputationalGrid.NY                30
pfset ComputationalGrid.NZ                30

pfset ComputationalGrid.DX	         10.0
pfset ComputationalGrid.DY               10.0
pfset ComputationalGrid.DZ	          0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names                 "domaininput leftinput rightinput channelinput"

pfset GeomInput.domaininput.GeomName  domain
pfset GeomInput.leftinput.GeomName  left
pfset GeomInput.rightinput.GeomName  right
pfset GeomInput.channelinput.GeomName  channel

pfset GeomInput.domaininput.InputType  Box 
pfset GeomInput.leftinput.InputType  Box 
pfset GeomInput.rightinput.InputType  Box 
pfset GeomInput.channelinput.InputType  Box 

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
pfset Geom.domain.Lower.X                        0.0
pfset Geom.domain.Lower.Y                        0.0
pfset Geom.domain.Lower.Z                        0.0
 
pfset Geom.domain.Upper.X                        300.0
pfset Geom.domain.Upper.Y                        300.0
pfset Geom.domain.Upper.Z                          1.5
pfset Geom.domain.Patches             "x-lower x-upper y-lower y-upper z-lower z-upper"

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------
pfset Geom.left.Lower.X                        0.0
pfset Geom.left.Lower.Y                        0.0
pfset Geom.left.Lower.Z                        0.0
 
pfset Geom.left.Upper.X                        140.0
pfset Geom.left.Upper.Y                        300.0
pfset Geom.left.Upper.Z                          1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------
pfset Geom.right.Lower.X                        160.0
pfset Geom.right.Lower.Y                        0.0
pfset Geom.right.Lower.Z                        0.0
 
pfset Geom.right.Upper.X                        300.0
pfset Geom.right.Upper.Y                        300.0
pfset Geom.right.Upper.Z                          1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------
pfset Geom.channel.Lower.X                        140.0
pfset Geom.channel.Lower.Y                        0.0
pfset Geom.channel.Lower.Z                        0.0
 
pfset Geom.channel.Upper.X                        160.0
pfset Geom.channel.Upper.Y                        300.0
pfset Geom.channel.Upper.Z                          1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

pfset Geom.Perm.Names                 "left right channel"

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

pfset Geom.left.Perm.Type "TurnBands"
pfset Geom.left.Perm.LambdaX  50.
pfset Geom.left.Perm.LambdaY  50.
pfset Geom.left.Perm.LambdaZ  0.5
pfset Geom.left.Perm.GeomMean  0.01

pfset Geom.left.Perm.Sigma   0.5
pfset Geom.left.Perm.NumLines 40
pfset Geom.left.Perm.RZeta  5.0
pfset Geom.left.Perm.KMax  100.0
pfset Geom.left.Perm.DelK  0.2
pfset Geom.left.Perm.Seed  33333
pfset Geom.left.Perm.LogNormal Log
pfset Geom.left.Perm.StratType Bottom


pfset Geom.right.Perm.Type "TurnBands"
pfset Geom.right.Perm.LambdaX  50.
pfset Geom.right.Perm.LambdaY  50.
pfset Geom.right.Perm.LambdaZ  0.5
pfset Geom.right.Perm.GeomMean  0.05

pfset Geom.right.Perm.Sigma   0.5
pfset Geom.right.Perm.NumLines 40
pfset Geom.right.Perm.RZeta  5.0
pfset Geom.right.Perm.KMax  100.0
pfset Geom.right.Perm.DelK  0.2
pfset Geom.right.Perm.Seed  13333
pfset Geom.right.Perm.LogNormal Log
pfset Geom.right.Perm.StratType Bottom

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface
#

pfset Geom.left.Perm.Type            Constant
pfset Geom.left.Perm.Value           0.001

pfset Geom.right.Perm.Type            Constant
pfset Geom.right.Perm.Value           0.01

pfset Geom.channel.Perm.Type            Constant
pfset Geom.channel.Perm.Value           0.00001

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0d0
pfset Geom.domain.Perm.TensorValY  1.0d0
pfset Geom.domain.Perm.TensorValZ  1.0d0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	        Constant
pfset Phase.water.Density.Value	        1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

pfset Contaminants.Names			""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

pfset Geom.Retardation.GeomNames           ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit        0.1
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        2.0
pfset TimingInfo.DumpInterval    0.1
pfset TimeStep.Type              Constant
pfset TimeStep.Value             0.1
 
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          "left right channel"

pfset Geom.left.Porosity.Type          Constant
pfset Geom.left.Porosity.Value         0.25

pfset Geom.right.Porosity.Type          Constant
pfset Geom.right.Porosity.Value         0.25

pfset Geom.channel.Porosity.Type          Constant
pfset Geom.channel.Porosity.Value         0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          "domain"

pfset Geom.domain.RelPerm.Alpha         6.0
pfset Geom.domain.RelPerm.N             2. 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         "domain"

pfset Geom.domain.Saturation.Alpha        6.0
pfset Geom.domain.Saturation.N            2.
pfset Geom.domain.Saturation.SRes         0.2
pfset Geom.domain.Saturation.SSat         1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant rainrec"
pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      1
pfset Cycle.constant.Repeat             -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

pfset Cycle.rainrec.Names                 "0 1 2 3 4 5 6"
pfset Cycle.rainrec.0.Length           1
pfset Cycle.rainrec.1.Length           1
pfset Cycle.rainrec.2.Length           1
pfset Cycle.rainrec.3.Length           1
pfset Cycle.rainrec.4.Length           1
pfset Cycle.rainrec.5.Length           1
pfset Cycle.rainrec.6.Length           1

pfset Cycle.rainrec.Repeat             1
 
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                           [pfget Geom.domain.Patches]

pfset Patch.x-lower.BCPressure.Type		      FluxConst
pfset Patch.x-lower.BCPressure.Cycle		      "constant"
pfset Patch.x-lower.BCPressure.alltime.Value	      0.0

pfset Patch.y-lower.BCPressure.Type		      FluxConst
pfset Patch.y-lower.BCPressure.Cycle		      "constant"
pfset Patch.y-lower.BCPressure.alltime.Value	      0.0

pfset Patch.z-lower.BCPressure.Type		      FluxConst
pfset Patch.z-lower.BCPressure.Cycle		      "constant"
pfset Patch.z-lower.BCPressure.alltime.Value	      0.0

pfset Patch.x-upper.BCPressure.Type		      FluxConst
pfset Patch.x-upper.BCPressure.Cycle		      "constant"
pfset Patch.x-upper.BCPressure.alltime.Value	      0.0

pfset Patch.y-upper.BCPressure.Type		      FluxConst
pfset Patch.y-upper.BCPressure.Cycle		      "constant"
pfset Patch.y-upper.BCPressure.alltime.Value	      0.0


pfset Patch.z-upper.BCPressure.Type		      OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle	              "rainrec"
pfset Patch.z-upper.BCPressure.0.Value	              $rec_flux
pfset Patch.z-upper.BCPressure.1.Value	              $rec_flux
pfset Patch.z-upper.BCPressure.2.Value	              $rain_flux
pfset Patch.z-upper.BCPressure.3.Value	              $rain_flux
pfset Patch.z-upper.BCPressure.4.Value	              $rec_flux
pfset Patch.z-upper.BCPressure.5.Value	              $rec_flux
pfset Patch.z-upper.BCPressure.6.Value	              $rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "left right channel"
if $use_slopes {
    pfset TopoSlopesX.Geom.left.Value -0.005
    pfset TopoSlopesX.Geom.right.Value 0.005
    pfset TopoSlopesX.Geom.channel.Value 0.00
} {
    pfset TopoSlopesX.Geom.left.Value    0.00
    pfset TopoSlopesX.Geom.right.Value   0.00
    pfset TopoSlopesX.Geom.channel.Value 0.00
}

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------


pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "left right channel"
if $use_slopes {
    pfset TopoSlopesY.Geom.left.Value    0.000
    pfset TopoSlopesY.Geom.right.Value   0.000
    pfset TopoSlopesY.Geom.channel.Value [expr 0.001 * $use_slopes]
} {
    pfset TopoSlopesY.Geom.left.Value    0.000
    pfset TopoSlopesY.Geom.right.Value   0.000
    pfset TopoSlopesY.Geom.channel.Value 0.000
}

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "left right channel"
pfset Mannings.Geom.left.Value 5.e-6
pfset Mannings.Geom.right.Value 5.e-6
pfset Mannings.Geom.channel.Value 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                     Constant
pfset PhaseSources.water.GeomNames                domain
pfset PhaseSources.water.Geom.domain.Value        0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                             Richards
pfset Solver.MaxIter                                     2500

pfset Solver.AbsTol                                      1E-12
pfset Solver.Nonlinear.MaxIter                           300
pfset Solver.Nonlinear.ResidualTol                       1e-12
pfset Solver.Nonlinear.EtaChoice                         Walker1 
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-16
pfset Solver.Nonlinear.StepTol				 1e-30
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      20
pfset Solver.Linear.MaxRestart                           2

pfset Solver.Linear.Preconditioner                       PFMG
pfset Solver.Linear.Preconditioner.PFMG.MaxIter           1
pfset Solver.Linear.Preconditioner.PFMG.Smoother          RBGaussSeidelNonSymmetric
pfset Solver.Linear.Preconditioner.PFMG.NumPreRelax       1
pfset Solver.Linear.Preconditioner.PFMG.NumPostRelax      1


pfset Solver.WriteSiloSubsurfData                       True
pfset Solver.WriteSiloPressure                          True
pfset Solver.WriteSiloSaturation                        True
pfset Solver.WriteSiloConcentration                     True
pfset Solver.WriteSiloSlopes                            True
pfset Solver.WriteSiloMask                              True
pfset Solver.WriteSiloEvapTrans                         True
pfset Solver.WriteSiloEvapTransSum                      True
pfset Solver.WriteSiloOverlandSum                       True
pfset Solver.WriteSiloMannings                          True
pfset Solver.WriteSiloSpecificStorage                   True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain

pfset Geom.domain.ICPressure.Value                      -3.0    

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   z-upper

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfrun $runname
pfundist $runname

#
# Tests 
#
source pftest.tcl
set passed 1

set slope_x          [pfload $runname.out.slope_x.silo]
set slope_y          [pfload $runname.out.slope_y.silo]
set mannings         [pfload $runname.out.mannings.silo]
set specific_storage [pfload $runname.out.specific_storage.silo]
set porosity         [pfload $runname.out.porosity.silo]

set mask             [pfload $runname.out.mask.silo]
set top              [pfcomputetop $mask]

set surface_area_of_domain [expr [pfget ComputationalGrid.DX] * [pfget ComputationalGrid.DY] * [pfget ComputationalGrid.NX] * [pfget ComputationalGrid.NY]]

set prev_total_water_balance 0.0

for {set i 0} {$i <= 19} {incr i} {
    if $verbose {
	puts "======================================================"
	puts "Timestep $i"
	puts "======================================================"
    }
    set total_water_in_domain 0.0

    set filename [format "%s.out.press.%05d.pfb" $runname $i]
    set pressure [pfload $filename]
    set surface_storage [pfsurfacestorage $top $pressure]
    pfsave $surface_storage -silo "surface_storage.$i.silo"
    set total_surface_storage [pfsum $surface_storage]
    if $verbose {
	puts [format "Surface storage\t\t\t\t\t : %.16e" $total_surface_storage]
    }
    set total_water_in_domain [expr $total_water_in_domain + $total_surface_storage]

    set filename [format "%s.out.satur.%05d.pfb" $runname $i]
    set saturation [pfload $filename]

    set water_table_depth [pfwatertabledepth $top $saturation]
    pfsave $water_table_depth -silo "water_table_depth.$i.silo"

    set subsurface_storage [pfsubsurfacestorage $mask $porosity $pressure $saturation $specific_storage]
    pfsave $subsurface_storage -silo "subsurface_storage.$i.silo"
    set total_subsurface_storage [pfsum $subsurface_storage]
    if $verbose {
	puts [format "Subsurface storage\t\t\t\t : %.16e" $total_subsurface_storage]
    }
    set total_water_in_domain [expr $total_water_in_domain + $total_subsurface_storage]

    if $verbose {
	puts [format "Total water in domain\t\t\t\t : %.16e" $total_water_in_domain]
	puts ""
    }

    set total_surface_runoff 0.0
    if { $i > 0} {
	set surface_runoff [pfsurfacerunoff $top $slope_x $slope_y $mannings $pressure]
	pfsave $surface_runoff -silo "surface_runoff.$i.silo"
	set total_surface_runoff [expr [pfsum $surface_runoff] * [pfget TimeStep.Value]]
	if $verbose {
	    puts [format "Surface runoff from pftools\t\t\t : %.16e" $total_surface_runoff]
	}

	set filename [format "%s.out.overlandsum.%05d.silo" $runname $i]
	set surface_runoff2 [pfload $filename]
	set total_surface_runoff2 [pfsum $surface_runoff2]
	if $verbose {
	    puts [format "Surface runoff from pfsimulator\t\t\t : %.16e" $total_surface_runoff2]
	}
	
	if ![pftestIsEqual $total_surface_runoff $total_surface_runoff2 "Surface runoff comparison" ] {
	    puts "FAILED: Surface runoff comparison"
	    set passed 0
	}
    }

    if [expr $i < 1] {
	set bc_index 0
    } elseif [expr $i > 0 && $i < 7] {
	set bc_index [expr $i - 1]
    } {
	set bc_index 6
    }
    set bc_flux [pfget Patch.z-upper.BCPressure.$bc_index.Value]

    set boundary_flux [expr $bc_flux * $surface_area_of_domain * [pfget TimeStep.Value]]
    if $verbose {
	puts [format "BC flux\t\t\t\t\t\t : %.16e" $boundary_flux]
    }

    # Note flow into domain is negative
    set expected_difference [expr $boundary_flux + $total_surface_runoff]
    if $verbose {
	puts [format "Total Flux\t\t\t\t\t : %.16e" $expected_difference]
    }

    if { $i > 0 } {

	if $verbose {
	    puts ""
	    puts [format "Diff from prev total\t\t\t\t : %.16e" [expr $total_water_in_domain - $prev_total_water_balance]]
	}

	if [expr $expected_difference != 0.0] {
	    set percent_diff [expr (abs(($prev_total_water_balance - $total_water_in_domain) - $expected_difference)) / abs($expected_difference) * 100]
	    if $verbose {
		puts [format "Percent diff from expected difference\t\t : %.12e" $percent_diff]
	    }
	}

	set expected_water_balance [expr $prev_total_water_balance - $expected_difference]
	set percent_diff [expr abs(($total_water_in_domain - $expected_water_balance)) / $expected_water_balance * 100]
	if $verbose {
	    puts [format "Percent diff from expected total water sum\t : %.12e" $percent_diff]
	}

	if [expr $percent_diff > 0.005] {
	    puts "FAILED: Water balance is not correct"
	    set passed 0
	}

    }

    set prev_total_water_balance [expr $total_water_in_domain]
}

if $verbose {
    puts "\n\n"
}

if $passed {
    puts "$runname : PASSED"
} {
    puts "$runname : FAILED"
}

