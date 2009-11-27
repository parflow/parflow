#  This runs Problem 2 in the paper
#     "Robust Numerical Methods for Saturated-Unsaturated Flow with
#      Dry Initial Conditions", Forsyth, Wu and Pruess, 
#      Advances in Water Resources, 1995.

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

pfset ComputationalGrid.NX                96
pfset ComputationalGrid.NY                1
pfset ComputationalGrid.NZ                67

set   UpperX                              800.0
set   UpperY                              1.0
set   UpperZ                              650.0

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
pfset GeomInput.solidinput.FileName   fors2_hf.pfsol

pfset GeomInput.zone1.InputType       Box
pfset GeomInput.zone1.GeomName        zone1

pfset Geom.zone1.Lower.X              0.0
pfset Geom.zone1.Lower.Y              0.0
pfset Geom.zone1.Lower.Z              610.0
pfset Geom.zone1.Upper.X              800.0
pfset Geom.zone1.Upper.Y              1.0
pfset Geom.zone1.Upper.Z              650.0

pfset GeomInput.zone2.InputType       Box
pfset GeomInput.zone2.GeomName        zone2

pfset Geom.zone2.Lower.X              0.0
pfset Geom.zone2.Lower.Y              0.0
pfset Geom.zone2.Lower.Z              560.0
pfset Geom.zone2.Upper.X              800.0
pfset Geom.zone2.Upper.Y              1.0
pfset Geom.zone2.Upper.Z              610.0

pfset GeomInput.zone3above4.InputType Box
pfset GeomInput.zone3above4.GeomName  zone3above4

pfset Geom.zone3above4.Lower.X        0.0
pfset Geom.zone3above4.Lower.Y        0.0
pfset Geom.zone3above4.Lower.Z        500.0
pfset Geom.zone3above4.Upper.X        800.0
pfset Geom.zone3above4.Upper.Y        1.0
pfset Geom.zone3above4.Upper.Z        560.0

pfset GeomInput.zone3left4.InputType  Box
pfset GeomInput.zone3left4.GeomName   zone3left4

pfset Geom.zone3left4.Lower.X         0.0
pfset Geom.zone3left4.Lower.Y         0.0
pfset Geom.zone3left4.Lower.Z         400.0
pfset Geom.zone3left4.Upper.X         100.0
pfset Geom.zone3left4.Upper.Y         1.0
pfset Geom.zone3left4.Upper.Z         500.0

pfset GeomInput.zone3right4.InputType  Box
pfset GeomInput.zone3right4.GeomName   zone3right4

pfset Geom.zone3right4.Lower.X        300.0
pfset Geom.zone3right4.Lower.Y        0.0
pfset Geom.zone3right4.Lower.Z        400.0
pfset Geom.zone3right4.Upper.X        800.0
pfset Geom.zone3right4.Upper.Y        1.0
pfset Geom.zone3right4.Upper.Z        500.0

pfset GeomInput.zone3below4.InputType Box
pfset GeomInput.zone3below4.GeomName  zone3below4

pfset Geom.zone3below4.Lower.X        0.0
pfset Geom.zone3below4.Lower.Y        0.0
pfset Geom.zone3below4.Lower.Z        0.0
pfset Geom.zone3below4.Upper.X        800.0
pfset Geom.zone3below4.Upper.Y        1.0
pfset Geom.zone3below4.Upper.Z        400.0

pfset GeomInput.zone4.InputType       Box
pfset GeomInput.zone4.GeomName        zone4

pfset Geom.zone4.Lower.X              100.0
pfset Geom.zone4.Lower.Y              0.0
pfset Geom.zone4.Lower.Z              400.0
pfset Geom.zone4.Upper.X              300.0
pfset Geom.zone4.Upper.Y              1.0
pfset Geom.zone4.Upper.Z              500.0

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

# Values in cm^2

pfset Geom.zone1.Perm.Type            Constant
pfset Geom.zone1.Perm.Value           9.1496e-5

pfset Geom.zone2.Perm.Type            Constant
pfset Geom.zone2.Perm.Value           5.4427e-5

pfset Geom.zone3above4.Perm.Type      Constant
pfset Geom.zone3above4.Perm.Value     4.8033e-5

pfset Geom.zone3left4.Perm.Type       Constant
pfset Geom.zone3left4.Perm.Value      4.8033e-5

pfset Geom.zone3right4.Perm.Type      Constant
pfset Geom.zone3right4.Perm.Value     4.8033e-5

pfset Geom.zone3below4.Perm.Type      Constant
pfset Geom.zone3below4.Perm.Value     4.8033e-5

pfset Geom.zone4.Perm.Type            Constant
pfset Geom.zone4.Perm.Value           4.8033e-4

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
pfset Phase.water.Viscosity.Value	1.124e-2

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

pfset Contaminants.Names			"tce"
pfset Contaminants.tce.Degradation.Value	 0.0

pfset PhaseConcen.water.tce.Type                 Constant
pfset PhaseConcen.water.tce.GeomNames            domain
pfset PhaseConcen.water.tce.Geom.domain.Value    0.0

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

pfset Geom.Retardation.GeomNames           background
pfset Geom.background.tce.Retardation.Type     Linear
pfset Geom.background.tce.Retardation.Rate     0.0

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
pfset TimingInfo.StopTime               2592000.0
pfset TimingInfo.StopTime               8640.0
#pfset TimingInfo.DumpInterval	        86400.0
pfset TimingInfo.DumpInterval	        -1
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    8640.0

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

pfset Geom.zone1.RelPerm.Alpha         0.0334
pfset Geom.zone1.RelPerm.N             1.982 

pfset Geom.zone2.RelPerm.Alpha         0.0363
pfset Geom.zone2.RelPerm.N             1.632 

pfset Geom.zone3above4.RelPerm.Alpha   0.0345
pfset Geom.zone3above4.RelPerm.N       1.573 

pfset Geom.zone3left4.RelPerm.Alpha    0.0345
pfset Geom.zone3left4.RelPerm.N        1.573 

pfset Geom.zone3right4.RelPerm.Alpha   0.0345
pfset Geom.zone3right4.RelPerm.N       1.573 

pfset Geom.zone3below4.RelPerm.Alpha   0.0345
pfset Geom.zone3below4.RelPerm.N       1.573 

pfset Geom.zone4.RelPerm.Alpha         0.0345
pfset Geom.zone4.RelPerm.N             1.573 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         $Zones

pfset Geom.zone1.Saturation.Alpha        0.0334
pfset Geom.zone1.Saturation.N            1.982
pfset Geom.zone1.Saturation.SRes         0.2771
pfset Geom.zone1.Saturation.SSat         1.0

pfset Geom.zone2.Saturation.Alpha        0.0363
pfset Geom.zone2.Saturation.N            1.632
pfset Geom.zone2.Saturation.SRes         0.2806
pfset Geom.zone2.Saturation.SSat         1.0

pfset Geom.zone3above4.Saturation.Alpha  0.0345
pfset Geom.zone3above4.Saturation.N      1.573
pfset Geom.zone3above4.Saturation.SRes   0.2643
pfset Geom.zone3above4.Saturation.SSat   1.0

pfset Geom.zone3left4.Saturation.Alpha   0.0345
pfset Geom.zone3left4.Saturation.N       1.573
pfset Geom.zone3left4.Saturation.SRes    0.2643
pfset Geom.zone3left4.Saturation.SSat    1.0

pfset Geom.zone3right4.Saturation.Alpha  0.0345
pfset Geom.zone3right4.Saturation.N      1.573
pfset Geom.zone3right4.Saturation.SRes   0.2643
pfset Geom.zone3right4.Saturation.SSat   1.0

pfset Geom.zone3below4.Saturation.Alpha  0.0345
pfset Geom.zone3below4.Saturation.N      1.573
pfset Geom.zone3below4.Saturation.SRes   0.2643
pfset Geom.zone3below4.Saturation.SSat   1.0

pfset Geom.zone3below4.Saturation.Alpha  0.0345
pfset Geom.zone3below4.Saturation.N      1.573
pfset Geom.zone3below4.Saturation.SRes   0.2643
pfset Geom.zone3below4.Saturation.SSat   1.0

pfset Geom.zone4.Saturation.Alpha        0.0345
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
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

pfset Patch.infiltration.BCPressure.Type	      FluxConst
pfset Patch.infiltration.BCPressure.Cycle	      "constant"
pfset Patch.infiltration.BCPressure.alltime.Value     -2.3148e-5

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

pfset ICPressure.Type                                   Constant
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -734.0

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

pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.StepTol                           1e-9
pfset Solver.Nonlinear.EtaValue                          1e-5
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-7

pfset Solver.Linear.KrylovDimension                      25
pfset Solver.Linear.MaxRestarts                          2

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun forsyth2
pfundist forsyth2




#
# Tests 
#
source pftest.tcl
set passed 1

if ![pftestFile forsyth2.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile forsyth2.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile forsyth2.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001" {
    if ![pftestFile forsyth2.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
    set passed 0
}
    if ![pftestFile forsyth2.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
    set passed 0
}
}


if $passed {
    puts "forsyth2 : PASSED"
} {
    puts "forsyth2 : FAILED"
}
