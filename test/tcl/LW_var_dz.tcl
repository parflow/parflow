#  This runs the a Little Washita Test Problem with variable dz
#  and a constant rain forcing.  The full Jacobian is use and there is no dampening in the
#  overland flow.

set tcl_precision 17

set runname LW_var_dz

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

pfset ComputationalGrid.NX                45
pfset ComputationalGrid.NY                32
pfset ComputationalGrid.NZ               25
pfset ComputationalGrid.NZ               10
pfset ComputationalGrid.NZ               6

pfset ComputationalGrid.DX	         1000.0
pfset ComputationalGrid.DY               1000.0
#"native" grid resolution is 2m everywhere X NZ=25 for 50m
#computational domain.
pfset ComputationalGrid.DZ		2.0

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names                 "domaininput"

pfset GeomInput.domaininput.GeomName  domain
pfset GeomInput.domaininput.InputType  Box

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X                        0.0
pfset Geom.domain.Lower.Y                        0.0
pfset Geom.domain.Lower.Z                        0.0

pfset Geom.domain.Upper.X                        45000.0
pfset Geom.domain.Upper.Y                        32000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
pfset Geom.domain.Upper.Z                        12.0
pfset Geom.domain.Patches             "x-lower x-upper y-lower y-upper z-lower z-upper"

#--------------------------------------------
# variable dz assignments
#------------------------------------------
pfset Solver.Nonlinear.VariableDz   True
pfset dzScale.GeomNames            domain
pfset dzScale.Type            nzList
pfset dzScale.nzListNumber       6

#pfset dzScale.Type            nzList
#pfset dzScale.nzListNumber       3
pfset Cell.0.dzScale.Value 1.0
pfset Cell.1.dzScale.Value 1.00
pfset Cell.2.dzScale.Value 1.000
pfset Cell.3.dzScale.Value 1.000
pfset Cell.4.dzScale.Value 1.000
pfset Cell.5.dzScale.Value 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

pfset Geom.Perm.Names                 "domain"

# Values in m/hour


pfset Geom.domain.Perm.Type            Constant

pfset Geom.domain.Perm.Type "TurnBands"
pfset Geom.domain.Perm.LambdaX  5000.0
pfset Geom.domain.Perm.LambdaY  5000.0
pfset Geom.domain.Perm.LambdaZ  50.0
pfset Geom.domain.Perm.GeomMean  0.0001427686

pfset Geom.domain.Perm.Sigma   0.20
pfset Geom.domain.Perm.Sigma   1.20
#pfset Geom.domain.Perm.Sigma   0.48989794
pfset Geom.domain.Perm.NumLines 150
pfset Geom.domain.Perm.RZeta  10.0
pfset Geom.domain.Perm.KMax  100.0000001
pfset Geom.domain.Perm.DelK  0.2
pfset Geom.domain.Perm.Seed  33333
pfset Geom.domain.Perm.LogNormal Log
pfset Geom.domain.Perm.StratType Bottom


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
pfset Geom.domain.SpecificStorage.Value 1.0e-5
#pfset Geom.domain.SpecificStorage.Value 0.0

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
pfset TimingInfo.BaseUnit        10.0
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        200.0
pfset TimingInfo.DumpInterval    20.0
pfset TimeStep.Type              Constant
pfset TimeStep.Value             10.0
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          "domain"

pfset Geom.domain.Porosity.Type          Constant
pfset Geom.domain.Porosity.Value         0.25
#pfset Geom.domain.Porosity.Value         0.


#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          "domain"

pfset Geom.domain.RelPerm.Alpha         1.
pfset Geom.domain.RelPerm.Alpha         1.0
pfset Geom.domain.RelPerm.N             3.
#pfset Geom.domain.RelPerm.NumSamplePoints   10000
#pfset Geom.domain.RelPerm.MinPressureHead   -200
#pfset Geom.domain.RelPerm.InterpolationMethod   "Linear"
#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         "domain"

pfset Geom.domain.Saturation.Alpha        1.0
pfset Geom.domain.Saturation.Alpha        1.0
pfset Geom.domain.Saturation.N            3.
pfset Geom.domain.Saturation.SRes         0.1
pfset Geom.domain.Saturation.SSat         1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant rainrec"
pfset Cycle.Names "constant"
pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      10000000
pfset Cycle.constant.Repeat             -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

pfset Cycle.rainrec.Names                 "rain rec"
pfset Cycle.rainrec.rain.Length           10
pfset Cycle.rainrec.rec.Length            20
pfset Cycle.rainrec.Repeat                14

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
pfset Patch.z-lower.BCPressure.alltime.Value	       0.0

pfset Patch.x-upper.BCPressure.Type		      FluxConst
pfset Patch.x-upper.BCPressure.Cycle		      "constant"
pfset Patch.x-upper.BCPressure.alltime.Value	      0.0

pfset Patch.y-upper.BCPressure.Type		      FluxConst
pfset Patch.y-upper.BCPressure.Cycle		      "constant"
pfset Patch.y-upper.BCPressure.alltime.Value	      0.0

## overland flow boundary condition with very heavy rainfall
pfset Patch.z-upper.BCPressure.Type		      OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle		      "constant"
# constant recharge at 100 mm / y
pfset Patch.z-upper.BCPressure.alltime.Value	      -0.005

#---------------
# Copy slopes to working dir
#----------------

file copy -force ../input/lw.1km.slope_x.10x.pfb .
file copy -force ../input/lw.1km.slope_y.10x.pfb .

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "PFBFile"
pfset TopoSlopesX.GeomNames "domain"

pfset TopoSlopesX.FileName lw.1km.slope_x.10x.pfb


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "PFBFile"
pfset TopoSlopesY.GeomNames "domain"

pfset TopoSlopesY.FileName lw.1km.slope_y.10x.pfb

#---------
##  Distribute slopes
#---------

pfset ComputationalGrid.NX                45
pfset ComputationalGrid.NY                32
pfset ComputationalGrid.NZ                6

# Slope files 1D files so distribute with -nz 1
pfdist -nz 1 lw.1km.slope_x.10x.pfb
pfdist -nz 1 lw.1km.slope_y.10x.pfb

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 0.00005


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

pfset Solver.TerrainFollowingGrid                        True


pfset Solver.Nonlinear.MaxIter                           80
pfset Solver.Nonlinear.ResidualTol                       1e-5
pfset Solver.Nonlinear.EtaValue                          0.001


pfset Solver.PrintSubsurf				False
pfset  Solver.Drop                                      1E-20
pfset Solver.AbsTol                                     1E-10


pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       True
#pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-14
pfset Solver.Nonlinear.StepTol				 1e-25
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      80
pfset Solver.Linear.MaxRestarts                           2

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner                       PFMG
pfset Solver.Linear.Preconditioner.PCMatrixType     FullJacobian

#pfset Solver.WriteSiloSubsurfData True
#pfset Solver.WriteSiloPressure True
#pfset Solver.WriteSiloSaturation True
#pfset Solver.WriteSiloConcentration True
#pfset Solver.WriteSiloSlopes True
#pfset Solver.WriteSiloMask True

pfset Solver.PrintVelocities True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -10.0

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   z-upper


#spinup key
# True=skim pressures, False = regular (default)
#pfset Solver.Spinup           True
pfset Solver.Spinup           False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfrun $runname
pfundist $runname
pfundist lw.1km.slope_x.10x.pfb
pfundist lw.1km.slope_y.10x.pfb


source pftest.tcl
set passed 1

if ![pftestFile $runname.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile $runname.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00002 00004 00006 00008 00010" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
    set passed 0
    }
    if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
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
    puts "LW_var_dz : PASSED"
} {
    puts "LW_var_dz : FAILED"
}




#puts "[exec tail $runname.out.kinsol.log]"
#puts "[exec tail $runname.out.log]"
