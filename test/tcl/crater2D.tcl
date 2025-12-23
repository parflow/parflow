#  This is a 2D crater problem w/ time varying input and topography
#    Reed Maxwell, 11/06
#
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

pfset FileVersion 4

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

pfset ComputationalGrid.NX                100
pfset ComputationalGrid.NY                1
pfset ComputationalGrid.NZ                100

set   UpperX                              400
set   UpperY                              1.0
set   UpperZ                              200

set   LowerX                              [pfget ComputationalGrid.Lower.X]
set   LowerY                              [pfget ComputationalGrid.Lower.Y]
set   LowerZ                              [pfget ComputationalGrid.Lower.Z]

set   NX                                  [pfget ComputationalGrid.NX]
set   NY                                  [pfget ComputationalGrid.NY]
set   NZ                                  [pfget ComputationalGrid.NZ]

pfset ComputationalGrid.DX	          [expr ($UpperX - $LowerX) / $NX]
pfset ComputationalGrid.DY                [expr ($UpperY - $LowerY) / $NY]
pfset ComputationalGrid.DZ	          [expr ($UpperZ - $LowerZ) / $NZ]

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
set   Zones                           "zone1 zone2 zone3above4 zone3left4 \
                                      zone3right4 zone3below4 zone4"

pfset GeomInput.Names                 "solidinput $Zones background"

pfset GeomInput.solidinput.InputType  SolidFile
pfset GeomInput.solidinput.GeomNames  domain
pfset GeomInput.solidinput.FileName   ../input/crater2D.pfsol

pfset GeomInput.zone1.InputType       Box
pfset GeomInput.zone1.GeomName        zone1

pfset Geom.zone1.Lower.X              0.0
pfset Geom.zone1.Lower.Y              0.0
pfset Geom.zone1.Lower.Z              0.0
pfset Geom.zone1.Upper.X              400.0
pfset Geom.zone1.Upper.Y              1.0
pfset Geom.zone1.Upper.Z              200.0

pfset GeomInput.zone2.InputType       Box
pfset GeomInput.zone2.GeomName        zone2

pfset Geom.zone2.Lower.X              0.0
pfset Geom.zone2.Lower.Y              0.0
pfset Geom.zone2.Lower.Z              60.0
pfset Geom.zone2.Upper.X              200.0
pfset Geom.zone2.Upper.Y              1.0
pfset Geom.zone2.Upper.Z              80.0

pfset GeomInput.zone3above4.InputType Box
pfset GeomInput.zone3above4.GeomName  zone3above4

pfset Geom.zone3above4.Lower.X        0.0
pfset Geom.zone3above4.Lower.Y        0.0
pfset Geom.zone3above4.Lower.Z        180.0
pfset Geom.zone3above4.Upper.X        200.0
pfset Geom.zone3above4.Upper.Y        1.0
pfset Geom.zone3above4.Upper.Z        200.0

pfset GeomInput.zone3left4.InputType  Box
pfset GeomInput.zone3left4.GeomName   zone3left4

pfset Geom.zone3left4.Lower.X         0.0
pfset Geom.zone3left4.Lower.Y         0.0
pfset Geom.zone3left4.Lower.Z         190.0
pfset Geom.zone3left4.Upper.X         100.0
pfset Geom.zone3left4.Upper.Y         1.0
pfset Geom.zone3left4.Upper.Z         200.0

pfset GeomInput.zone3right4.InputType  Box
pfset GeomInput.zone3right4.GeomName   zone3right4

pfset Geom.zone3right4.Lower.X        30.0
pfset Geom.zone3right4.Lower.Y        0.0
pfset Geom.zone3right4.Lower.Z        90.0
pfset Geom.zone3right4.Upper.X        80.0
pfset Geom.zone3right4.Upper.Y        1.0
pfset Geom.zone3right4.Upper.Z        100.0

pfset GeomInput.zone3below4.InputType Box
pfset GeomInput.zone3below4.GeomName  zone3below4

pfset Geom.zone3below4.Lower.X        0.0
pfset Geom.zone3below4.Lower.Y        0.0
pfset Geom.zone3below4.Lower.Z        0.0
pfset Geom.zone3below4.Upper.X        400.0
pfset Geom.zone3below4.Upper.Y        1.0
pfset Geom.zone3below4.Upper.Z        20.0

pfset GeomInput.zone4.InputType       Box
pfset GeomInput.zone4.GeomName        zone4

pfset Geom.zone4.Lower.X              0.0
pfset Geom.zone4.Lower.Y              0.0
pfset Geom.zone4.Lower.Z              100.0
pfset Geom.zone4.Upper.X              300.0
pfset Geom.zone4.Upper.Y              1.0
pfset Geom.zone4.Upper.Z              150.0

pfset GeomInput.background.InputType  Box
pfset GeomInput.background.GeomName   background

pfset Geom.background.Lower.X         -99999999.0
pfset Geom.background.Lower.Y         -99999999.0
pfset Geom.background.Lower.Z         -99999999.0
pfset Geom.background.Upper.X         99999999.0
pfset Geom.background.Upper.Y         99999999.0
pfset Geom.background.Upper.Z         99999999.0

pfset Geom.domain.Patches             "infiltration z-upper x-lower y-lower \
                                      x-upper y-upper z-lower"


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names                 $Zones



pfset Geom.zone1.Perm.Type            Constant
pfset Geom.zone1.Perm.Value           9.1496

pfset Geom.zone2.Perm.Type            Constant
pfset Geom.zone2.Perm.Value           5.4427

pfset Geom.zone3above4.Perm.Type      Constant
pfset Geom.zone3above4.Perm.Value     4.8033

pfset Geom.zone3left4.Perm.Type       Constant
pfset Geom.zone3left4.Perm.Value      4.8033

pfset Geom.zone3right4.Perm.Type      Constant
pfset Geom.zone3right4.Perm.Value     4.8033

pfset Geom.zone3below4.Perm.Type      Constant
pfset Geom.zone3below4.Perm.Value     4.8033

pfset Geom.zone4.Perm.Type            Constant
pfset Geom.zone4.Perm.Value           .48033

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

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

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime               300.0
pfset TimingInfo.DumpInterval	        30.0
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames           $Zones

pfset Geom.zone1.Porosity.Type          Constant
pfset Geom.zone1.Porosity.Value         0.3680

pfset Geom.zone2.Porosity.Type          Constant
pfset Geom.zone2.Porosity.Value         0.3510

pfset Geom.zone3above4.Porosity.Type    Constant
pfset Geom.zone3above4.Porosity.Value   0.3250

pfset Geom.zone3left4.Porosity.Type     Constant
pfset Geom.zone3left4.Porosity.Value    0.3250

pfset Geom.zone3right4.Porosity.Type    Constant
pfset Geom.zone3right4.Porosity.Value   0.3250

pfset Geom.zone3below4.Porosity.Type    Constant
pfset Geom.zone3below4.Porosity.Value   0.3250

pfset Geom.zone4.Porosity.Type          Constant
pfset Geom.zone4.Porosity.Value         0.3250

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          $Zones

pfset Geom.zone1.RelPerm.Alpha         3.34
pfset Geom.zone1.RelPerm.N             1.982

pfset Geom.zone2.RelPerm.Alpha         3.63
pfset Geom.zone2.RelPerm.N             1.632

pfset Geom.zone3above4.RelPerm.Alpha   3.45
pfset Geom.zone3above4.RelPerm.N       1.573

pfset Geom.zone3left4.RelPerm.Alpha    3.45
pfset Geom.zone3left4.RelPerm.N        1.573

pfset Geom.zone3right4.RelPerm.Alpha   3.45
pfset Geom.zone3right4.RelPerm.N       1.573

pfset Geom.zone3below4.RelPerm.Alpha   3.45
pfset Geom.zone3below4.RelPerm.N       1.573

pfset Geom.zone4.RelPerm.Alpha         3.45
pfset Geom.zone4.RelPerm.N             1.573

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         $Zones

pfset Geom.zone1.Saturation.Alpha        3.34
pfset Geom.zone1.Saturation.N            1.982
pfset Geom.zone1.Saturation.SRes         0.2771
pfset Geom.zone1.Saturation.SSat         1.0

pfset Geom.zone2.Saturation.Alpha        3.63
pfset Geom.zone2.Saturation.N            1.632
pfset Geom.zone2.Saturation.SRes         0.2806
pfset Geom.zone2.Saturation.SSat         1.0

pfset Geom.zone3above4.Saturation.Alpha  3.45
pfset Geom.zone3above4.Saturation.N      1.573
pfset Geom.zone3above4.Saturation.SRes   0.2643
pfset Geom.zone3above4.Saturation.SSat   1.0

pfset Geom.zone3left4.Saturation.Alpha   3.45
pfset Geom.zone3left4.Saturation.N       1.573
pfset Geom.zone3left4.Saturation.SRes    0.2643
pfset Geom.zone3left4.Saturation.SSat    1.0

pfset Geom.zone3right4.Saturation.Alpha  3.45
pfset Geom.zone3right4.Saturation.N      1.573
pfset Geom.zone3right4.Saturation.SRes   0.2643
pfset Geom.zone3right4.Saturation.SSat   1.0

pfset Geom.zone3below4.Saturation.Alpha  3.45
pfset Geom.zone3below4.Saturation.N      1.573
pfset Geom.zone3below4.Saturation.SRes   0.2643
pfset Geom.zone3below4.Saturation.SSat   1.0

pfset Geom.zone3below4.Saturation.Alpha  3.45
pfset Geom.zone3below4.Saturation.N      1.573
pfset Geom.zone3below4.Saturation.SRes   0.2643
pfset Geom.zone3below4.Saturation.SSat   1.0

pfset Geom.zone4.Saturation.Alpha        0.345
pfset Geom.zone4.Saturation.N            1.573
pfset Geom.zone4.Saturation.SRes         0.2643
pfset Geom.zone4.Saturation.SSat         1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant onoff"
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

pfset Cycle.onoff.Names                 "on off"
pfset Cycle.onoff.on.Length             10
pfset Cycle.onoff.off.Length            90
pfset Cycle.onoff.Repeat               -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

pfset Patch.infiltration.BCPressure.Type	      FluxConst
pfset Patch.infiltration.BCPressure.Cycle	      "onoff"
pfset Patch.infiltration.BCPressure.on.Value     	-0.10
pfset Patch.infiltration.BCPressure.off.Value     	0.0

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

pfset Patch.z-upper.BCPressure.Type		      FluxConst
pfset Patch.z-upper.BCPressure.Cycle		      "constant"
pfset Patch.z-upper.BCPressure.alltime.Value	      0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

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

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames ""
pfset Mannings.Geom.domain.Value 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              "domain"

pfset Geom.domain.ICPressure.Value                      1.0
pfset Geom.domain.ICPressure.RefPatch                  z-lower
pfset Geom.domain.ICPressure.RefGeom                  domain

pfset Geom.infiltration.ICPressure.Value                      10.0
pfset Geom.infiltration.ICPressure.RefPatch                  infiltration
pfset Geom.infiltration.ICPressure.RefGeom                  domain

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     10000

#pfset Solver.Nonlinear.MaxIter                           100
pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.StepTol                           1e-9
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-7
pfset Solver.Linear.KrylovDimension                      25
pfset Solver.Linear.MaxRestarts                          2

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

# Test Top writing
pfset Solver.PrintTop                                    True

source pftest.tcl
#
# Check for regressions
#
proc check_output {runname} {
    global passed sig_digits

    if ![pftestFile $runname.out.perm_x.pfb "$runname: Max difference in perm_x" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $runname.out.perm_y.pfb "$runname: Max difference in perm_y" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $runname.out.perm_z.pfb "$runname: Max difference in perm_z" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $runname.out.porosity.pfb "$runname: Max difference in porosity" $sig_digits] {
	set passed 0
    }
    
    if ![pftestFile $runname.out.top_patch.pfb "$runname: Max difference in top patch" $sig_digits] {
	set passed 0
    }
    
    if ![pftestFile $runname.out.top_zindex.pfb "$runname: Max difference in top zindex" $sig_digits] {
	set passed 0
    }
    
    foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010" {
	if ![pftestFile $runname.out.press.$i.pfb "$runname: Max difference in Pressure for timestep $i" $sig_digits] {
	    set passed 0
	}
	if ![pftestFile $runname.out.satur.$i.pfb "$runname: Max difference in Saturation for timestep $i" $sig_digits] {
	    set passed 0
	}
    }
}

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

set passed 1

#-----------------------------------------------------------------------------
# EtaConstant
#-----------------------------------------------------------------------------
set runname "crater2D_etaconstant"

puts "Running with EtaConstant"

pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-5

pfrun $runname
pfundist $runname
check_output $runname

pfunset Solver.Nonlinear.EtaValue

#-----------------------------------------------------------------------------
# Walker1
#-----------------------------------------------------------------------------
set runname "crater2D_walker1"

puts "Running with Walker1"

pfset Solver.Nonlinear.EtaChoice                         Walker1

pfrun $runname
pfundist $runname
check_output $runname

#-----------------------------------------------------------------------------
# Walker2
#-----------------------------------------------------------------------------
set runname "crater2D_walker2"

puts "Running with Walker2"

pfset Solver.Nonlinear.EtaChoice                         Walker2
# Default EtaGamma yields differences between the embedded Kinsol and current external Sundials library.
pfset Solver.Nonlinear.EtaGamma                          0.100000

pfrun $runname
pfundist $runname
check_output $runname

if $passed {
    puts "creater2D : PASSED"
} {
    puts "creater2D : FAILED"
}
