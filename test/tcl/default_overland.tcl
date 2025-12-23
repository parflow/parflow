#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR

set tcl_precision 17

set runname default_overland

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

pfset Process.Topology.P        [lindex $argv 0]
pfset Process.Topology.Q        [lindex $argv 1]
pfset Process.Topology.R        [lindex $argv 2]

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
pfset ComputationalGrid.DZ	            .05

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

# 
pfset TimingInfo.BaseUnit        0.1
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        0.4
pfset TimingInfo.DumpInterval    -1
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

pfset Cycle.rainrec.Names                 "rain rec"
pfset Cycle.rainrec.rain.Length           1
pfset Cycle.rainrec.rec.Length            2
pfset Cycle.rainrec.Repeat                -1
 
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

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

## overland flow boundary condition with very heavy rainfall then slight ET
pfset Patch.z-upper.BCPressure.Type		      OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle		      "rainrec"
pfset Patch.z-upper.BCPressure.rain.Value	      -0.05
pfset Patch.z-upper.BCPressure.rec.Value	      0.000001

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "left right channel"
pfset TopoSlopesX.Geom.left.Value -0.005
pfset TopoSlopesX.Geom.right.Value 0.005
pfset TopoSlopesX.Geom.channel.Value 0.00

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------


pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "left right channel"
pfset TopoSlopesY.Geom.left.Value 0.001
pfset TopoSlopesY.Geom.right.Value 0.001
pfset TopoSlopesY.Geom.channel.Value 0.001

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

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    domain
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

pfset Solver.Nonlinear.MaxIter                           20
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.01
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-8
pfset Solver.Nonlinear.StepTol				 1e-20
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      20
pfset Solver.Linear.MaxRestart                           2

pfset Solver.Linear.Preconditioner                       PFMGOctree
pfset Solver.PrintSubsurf				False
pfset  Solver.Drop                                      1E-20
pfset Solver.AbsTol                                     1E-9
 
pfset Solver.WriteSiloSubsurfData True
pfset Solver.WriteSiloPressure True
pfset Solver.WriteSiloSaturation True
pfset Solver.WriteSiloConcentration True

pfset Solver.PrintVelocities True

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

#
# SGS this test fails with 6 sigdigits
#
set sig_digits 5

if ![pftestFile $runname.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001 00002 00003 00004" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
	set passed 0
    }
    # use abs value test to prevent machine precision effects
    set abs_value 1e-12
    if ![pftestFileWithAbs $runname.out.velx.$i.pfb "Max difference in x-velocity for timestep $i" $sig_digits $abs_value] {
    set passed 0
    }
    if ![pftestFileWithAbs $runname.out.vely.$i.pfb "Max difference in y-velocity for timestep $i" $sig_digits $abs_value] {
    set passed 0
    }
    if ![pftestFileWithAbs $runname.out.velz.$i.pfb "Max difference in z-velocity for timestep $i" $sig_digits $abs_value] {
    set passed 0
    }
}


if $passed {
    puts "$runname : PASSED"
} {
    puts "$runname : FAILED"
}
