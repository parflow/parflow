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

set py [pfget Process.Topology.P]
#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

pfset ComputationalGrid.NX                 100
pfset ComputationalGrid.NY                50 
pfset ComputationalGrid.NZ                50

pfset ComputationalGrid.DX	           0.1
pfset ComputationalGrid.DY                 0.1
pfset ComputationalGrid.DZ	           0.1

set nx [pfget ComputationalGrid.NX]
set dx [pfget ComputationalGrid.DX]
set ny [pfget ComputationalGrid.NY]
set dy [pfget ComputationalGrid.DY]
set nz [pfget ComputationalGrid.NZ]
set dz [pfget ComputationalGrid.DZ]
#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names                 "domain_input heat_input"

pfset GeomInput.solidinput.GeomNames  domain

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

pfset GeomInput.heat_input.InputType            Box
pfset GeomInput.heat_input.GeomName             heat
#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X                         0.0 
pfset Geom.domain.Lower.Y                         0.0
pfset Geom.domain.Lower.Z                         0.0

pfset Geom.domain.Upper.X                         [expr ($nx * $dx)]
pfset Geom.domain.Upper.Y                         [expr ($ny * $dy)]
pfset Geom.domain.Upper.Z                         [expr ($nz * $dz)] 

pfset Geom.domain.Patches "left right front back bottom top"

pfset Geom.heat.Lower.X           [ expr (($nx/2)-1)*$dx ]
pfset Geom.heat.Lower.Y           [ expr (($ny/2)-1)*$dy ]
pfset Geom.heat.Lower.Z           [ expr (($nz/2)-1)*$dz ]

pfset Geom.heat.Upper.X           [ expr (($nx/2)+1)*$dx ]
pfset Geom.heat.Upper.Y           [ expr (($ny/2)+1)*$dy ]
pfset Geom.heat.Upper.Z           [ expr (($nz/2)+1)*$dz ]

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names                 "domain"

# Values in m/d

pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           1.0e-11
pfset Geom.domain.Perm.Value           1.0e-9
#pfset Geom.domain.Perm.Value           0.0

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value          1.0e-5
#pfset Geom.domain.SpecificStorage.Value          1.0e-9
#pfset Geom.domain.SpecificStorage.Value          0.0

#-----------------------------------------------------------------------------
# Heat Capacity 
#-----------------------------------------------------------------------------

pfset Phase.water.HeatCapacity.Type                      Constant
pfset Phase.water.HeatCapacity.GeomNames                 "domain"
pfset Phase.water.Geom.domain.HeatCapacity.Value        4000. 
#pfset Phase.water.Geom.domain.HeatCapacity.Value         1.0 

pfset Phase.rock.HeatCapacity.Type                       Constant
pfset Phase.rock.HeatCapacity.GeomNames                  "domain"
pfset Phase.rock.Geom.domain.HeatCapacity.Value          837. 
#pfset Phase.rock.Geom.domain.HeatCapacity.Value          1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water rock"
pfset Phase.Names "water"

#-----------------------------------------------------------------------------
# Density
#-----------------------------------------------------------------------------

pfset Phase.water.Density.Type	        Constant
#pfset Phase.water.Density.Type	EquationOfState
pfset Phase.water.Density.Value	        1000.0
pfset Phase.water.Density.ReferenceDensity   1.0
pfset Phase.water.Density.CompressibiltyConstant 0.4

pfset Phase.rock.Density.Type	        Constant
pfset Phase.rock.Density.Value	        1.0

#-----------------------------------------------------------------------------
# Viscosity
#-----------------------------------------------------------------------------

pfset Phase.water.Viscosity.Type	Constant
#pfset Phase.water.Viscosity.Type	EquationOfState
pfset Phase.water.Viscosity.Value	0.001

pfset Phase.rock.Viscosity.Type	        Constant
pfset Phase.rock.Viscosity.Value	1.0

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

pfset Gravity				9.81

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
set dtime 100000.0
set dtime 10000.0
set dtime 36000.
#set dtime 0.001
set fac  1000.0
set dump 1.0
set fac  10.0
#set dump 1.0
pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime              [expr ($dtime * $fac)]
pfset TimingInfo.StopTime              360000. 
pfset TimingInfo.DumpInterval	       [expr ($dtime * $dump)]
pfset TimingInfo.DumpInterval	       36000. 
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                   $dtime 

#pfset TimeStep.Type                     Growth
pfset TimeStep.MinStep                  1.0e-4
pfset TimeStep.InitialStep              1.0e-4
pfset TimeStep.GrowthFactor             1.2
pfset TimeStep.MaxStep                  2.0
#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames           domain

pfset Geom.domain.Porosity.Type          Constant
pfset Geom.domain.Porosity.Value         0.4

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
#pfset Phase.RelPerm.Type               Constant
pfset Phase.RelPerm.GeomNames          "domain"
pfset Geom.domain.RelPerm.Value         1.0 

pfset Geom.domain.RelPerm.Alpha         1.0e-4
pfset Geom.domain.RelPerm.N             2. 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
#pfset Phase.Saturation.Type              Constant
pfset Phase.Saturation.GeomNames         "domain"
pfset Geom.domain.Saturation.Value             1.0

pfset Geom.domain.Saturation.Alpha        1.0e-4
pfset Geom.domain.Saturation.N            2.0
pfset Geom.domain.Saturation.SRes         0.2
pfset Geom.domain.Saturation.SSat         1.0

#-------------------------------------------------------
# Thermal Conductivity
#-------------------------------------------------------

pfset Phase.ThermalConductivity.Type   Constant
#pfset Phase.ThermalConductivity.Type   Function1 
pfset Phase.ThermalConductivity.GeomNames "domain"
pfset Geom.domain.ThermalConductivity.Value 2.0
pfset Geom.domain.ThermalConductivity.KDry  1.8
pfset Geom.domain.ThermalConductivity.KWet  2.2

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames        "left right front back bottom top"
 

#pfset Patch.left.BCPressure.Type		      FluxConst
pfset Patch.left.BCPressure.Cycle		      "constant"
pfset Patch.left.BCPressure.Type		      DirEquilRefPatch
pfset Patch.left.BCPressure.RefGeom                   domain
pfset Patch.left.BCPressure.RefPatch                  bottom
pfset Patch.left.BCPressure.alltime.Value            120937.76
#pfset Patch.left.BCPressure.alltime.Value             0.0

#pfset Patch.right.BCPressure.Type		      FluxConst
pfset Patch.right.BCPressure.Cycle		      "constant"
pfset Patch.right.BCPressure.Type		      DirEquilRefPatch
pfset Patch.right.BCPressure.RefGeom                  domain
pfset Patch.right.BCPressure.RefPatch                 bottom
pfset Patch.right.BCPressure.alltime.Value        120937.76 
#pfset Patch.right.BCPressure.alltime.Value            0.0

pfset Patch.front.BCPressure.Type		      FluxConst
pfset Patch.front.BCPressure.Cycle		      "constant"
#pfset Patch.front.BCPressure.Type                      DirEquilRefPatch  
pfset Patch.front.BCPressure.RefGeom                  domain
pfset Patch.front.BCPressure.RefPatch                 bottom
#pfset Patch.front.BCPressure.alltime.Value	      490318.25715
pfset Patch.front.BCPressure.alltime.Value	      0.0

pfset Patch.back.BCPressure.Type		      FluxConst
pfset Patch.back.BCPressure.Cycle		      "constant"
#pfset Patch.back.BCPressure.Type                      DirEquilRefPatch  
pfset Patch.back.BCPressure.RefGeom                  domain
pfset Patch.back.BCPressure.RefPatch                 bottom
#pfset Patch.back.BCPressure.alltime.Value	      490318.25715
pfset Patch.back.BCPressure.alltime.Value	      0.0

#---- Bottom BC
pfset Patch.bottom.BCPressure.Type		      FluxConst
pfset Patch.bottom.BCPressure.Cycle		      "constant"
pfset Patch.bottom.BCPressure.alltime.Value           0.0
#---- End Bottom BC

#---- Top BC
pfset Patch.top.BCPressure.Type		              FluxConst
pfset Patch.top.BCPressure.Cycle		      "constant"
pfset Patch.top.BCPressure.alltime.Value              0.0
#---- End Top BC

#-----------------------------------------------------------------------------
# Boundary Conditions: Temperature 
#-----------------------------------------------------------------------------
pfset BCTemperature.PatchNames        "left right front back bottom top"
 
 
pfset Patch.left.BCTemperature.Type                      DirConst
#pfset Patch.left.BCTemperature.Type                      FluxConst
pfset Patch.left.BCTemperature.Cycle                     "constant"
pfset Patch.left.BCTemperature.alltime.Value             288.15
#pfset Patch.left.BCTemperature.alltime.Value             305. 
#pfset Patch.left.BCTemperature.alltime.Value             0.0
 
pfset Patch.right.BCTemperature.Type                     DirConst
#pfset Patch.right.BCTemperature.Type                     FluxConst
pfset Patch.right.BCTemperature.Cycle                    "constant"
pfset Patch.right.BCTemperature.alltime.Value            295.
pfset Patch.right.BCTemperature.alltime.Value           293.15
#pfset Patch.right.BCTemperature.alltime.Value            0.0
 
pfset Patch.front.BCTemperature.Type                     DirConst 
pfset Patch.front.BCTemperature.Type                     FluxConst 
pfset Patch.front.BCTemperature.Cycle                    "constant"
pfset Patch.front.BCTemperature.alltime.Value            305.
#pfset Patch.front.BCTemperature.alltime.Value            295.
pfset Patch.front.BCTemperature.alltime.Value           0.0 
 
pfset Patch.back.BCTemperature.Type                      DirConst 
pfset Patch.back.BCTemperature.Type                      FluxConst 
pfset Patch.back.BCTemperature.Cycle                     "constant"
pfset Patch.back.BCTemperature.alltime.Value             295.
pfset Patch.back.BCTemperature.alltime.Value             0.0
 
pfset Patch.bottom.BCTemperature.Type                    FluxConst 
pfset Patch.bottom.BCTemperature.Cycle                   "constant"
pfset Patch.bottom.BCTemperature.alltime.Value           0.0
 
pfset Patch.top.BCTemperature.Type                       FluxConst 
pfset Patch.top.BCTemperature.Cycle                      "constant"
pfset Patch.top.BCTemperature.alltime.Value              0.0

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

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                  Constant 
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              "domain"
pfset Geom.domain.ICPressure.Value                  120937.76 
#pfset Geom.domain.ICPressure.Value                     0.0 

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   bottom

#pfset ICPressure.Type                                  PFBFile
#pfset Geom.domain.ICPressure.FileName                  "ptest.press.dp2.pfb"
#pfset Geom.domain.ICPressure.FileName                  "heat.out.press.00050.pfb"
#---------------------------------------------------------
# Initial conditions: water temperature
#---------------------------------------------------------
pfset ICTemperature.Type                                  Constant 
pfset ICTemperature.GeomNames                              "domain"
pfset Geom.heat.ICTemperature.Value                      300. 
pfset Geom.domain.ICTemperature.Value                     288.15 
#pfset Geom.domain.ICTemperature.Value                     29.55 

pfset Geom.domain.ICTemperature.RefGeom                    domain
pfset Geom.domain.ICTemperature.RefPatch                   bottom

#pfset ICTemperature.Type                                  PFBFile
#pfset Geom.domain.ICTemperature.FileName                  "temp3d.C.in.pfb"
#pfset Geom.domain.ICTemperature.FileName                  "heat.out.temp.00050.pfb"
#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                   "domain"
pfset PhaseSources.water.Geom.domain.Value            0.0

#-----------------------------------------------------------------------------
# Temperature sources:
#-----------------------------------------------------------------------------
pfset TempSources.Type                         Constant
pfset TempSources.GeomNames                   "domain"
pfset TempSources.Geom.domain.Value           0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters 
#-----------------------------------------------------------------------------
 
pfset Solver                                             Richards
pfset Solver.MaxIter                                     50000 
pfset Solver.Nonlinear.PrintFlag                         HighVerbosity

pfset Solver.Nonlinear.MaxIter                           500

pfset Solver.Nonlinear.ResidualTol                       1.e-2
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaChoice                         Walker1
pfset Solver.Nonlinear.EtaValue                          1.0e-6
pfset Solver.Nonlinear.EtaValue                          1.0e-4
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-16
pfset Solver.Nonlinear.StepTol                           1.e-16
pfset Solver.Nonlinear.Globalization                     LineSearch


#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#pfdist ptest.press.dp2.pfb
#pfdist temp3d.C.in.pfb
#pfdist heat.out.press.00050.pfb
#pfdist heat.out.temp.00040.pfb
pfrun cond
pfundist cond


#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST cond
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    foreach i "00000 00001 00002 00003 00004 00005" {
	if ![pftestFile $TEST.out.press.$i.pfb "Max difference in press timestep $i" $sig_digits] {
	    set passed 0
	}

	if ![pftestFile $TEST.out.satur.$i.pfb "Max difference in satur timestep $i" $sig_digits] {
	    set passed 0
	}
    }

    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
