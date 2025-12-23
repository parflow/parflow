#  This test problem runs the Richards' equation solvers
#  on the eqn:  - div (p grad p) = f where p = x and f
#  is chosen to guarantee the correct solution.
#  For 64 unknowns, the following line should be printed on
#  the screen:
#
#  l2-error in pressure:       5.84881366e-05

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

#---------------------------------------------------------
# Academic test problem name
#---------------------------------------------------------
set   TestName           X

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                  0.0
pfset ComputationalGrid.Lower.Y                  0.0
pfset ComputationalGrid.Lower.Z                  0.0

pfset ComputationalGrid.DX	                 0.015625
pfset ComputationalGrid.DY                       1.0
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                       64
pfset ComputationalGrid.NY                       1
pfset ComputationalGrid.NZ                       1

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input background_input"


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X                          0.0 
pfset Geom.domain.Lower.Y                          0.0
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                          1.0
pfset Geom.domain.Upper.Y                          1.0
pfset Geom.domain.Upper.Z                          1.0

pfset Geom.domain.Patches "left right front back bottom top"

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfset Geom.background.Lower.X -99999999.0
pfset Geom.background.Lower.Y -99999999.0
pfset Geom.background.Lower.Z -99999999.0

pfset Geom.background.Upper.X  99999999.0
pfset Geom.background.Upper.Y  99999999.0
pfset Geom.background.Upper.Z  99999999.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background"

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    1.0

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

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

pfset Gravity				0.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime               1.0
pfset TimingInfo.DumpInterval	       -1
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          background

pfset Geom.background.Porosity.Type    Constant
pfset Geom.background.Porosity.Value   1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               Polynomial
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Degree       1
pfset Geom.domain.RelPerm.Coeff.0      0.0   
pfset Geom.domain.RelPerm.Coeff.1      1.0   

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            Constant
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Value     0.0

#-------------------------------------------------------
# Thermal Conductivity
#-------------------------------------------------------
 
pfset Phase.ThermalConductivity.Type   Constant
pfset Phase.ThermalConductivity.GeomNames "domain"
pfset Geom.domain.ThermalConductivity.Value 2.0
pfset Geom.domain.ThermalConductivity.KDry  1.8
pfset Geom.domain.ThermalConductivity.KWet  2.2

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
pfset BCPressure.PatchNames "left right front back bottom top"

pfset Patch.left.BCPressure.Type			ExactSolution
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.alltime.PredefinedFunction  $TestName

pfset Patch.right.BCPressure.Type			ExactSolution
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.alltime.PredefinedFunction $TestName

pfset Patch.front.BCPressure.Type			FluxConst
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.Value              0.0

pfset Patch.back.BCPressure.Type			FluxConst
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.Value               0.0

pfset Patch.bottom.BCPressure.Type			FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.alltime.Value             0.0

pfset Patch.top.BCPressure.Type			        FluxConst
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.alltime.Value                0.0


#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                   Constant
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      1.0

#-----------------------------------------------------------------------------
# Boundary Conditions: Temperature 
#-----------------------------------------------------------------------------
pfset BCTemperature.PatchNames        "left right front back bottom top"
 
pfset Patch.left.BCTemperature.Type                      FluxConst
pfset Patch.left.BCTemperature.Cycle                     "constant"
pfset Patch.left.BCTemperature.alltime.Value             0.0
 
pfset Patch.right.BCTemperature.Type                     FluxConst
pfset Patch.right.BCTemperature.Cycle                    "constant"
pfset Patch.right.BCTemperature.alltime.Value            0.0
 
pfset Patch.front.BCTemperature.Type                     FluxConst
pfset Patch.front.BCTemperature.Cycle                    "constant"
pfset Patch.front.BCTemperature.alltime.Value            0.0
 
pfset Patch.back.BCTemperature.Type                      FluxConst
pfset Patch.back.BCTemperature.Cycle                     "constant"
pfset Patch.back.BCTemperature.alltime.Value             0.0
 
pfset Patch.bottom.BCTemperature.Type                    FluxConst
pfset Patch.bottom.BCTemperature.Cycle                   "constant"
pfset Patch.bottom.BCTemperature.alltime.Value           0.0
 
pfset Patch.top.BCTemperature.Type                       FluxConst
pfset Patch.top.BCTemperature.Cycle                      "constant"
pfset Patch.top.BCTemperature.alltime.Value              0.0
 
#---------------------------------------------------------
# Initial conditions: water temperature
#---------------------------------------------------------
pfset ICTemperature.Type                                  Constant
pfset ICTemperature.GeomNames                              "domain"
pfset Geom.domain.ICTemperature.Value                     288.15
 
pfset Geom.domain.ICTemperature.RefGeom                    domain
pfset Geom.domain.ICTemperature.RefPatch                   bottom

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         PredefinedFunction
pfset PhaseSources.water.PredefinedFunction           $TestName

#-----------------------------------------------------------------------------
# Temperature sources:
#-----------------------------------------------------------------------------
pfset TempSources.Type                         Constant
pfset TempSources.GeomNames                   "background"
pfset TempSources.Geom.background.Value        0.0

#-----------------------------------------------------------------------------
# Heat Capacity 
#-----------------------------------------------------------------------------
 
pfset Phase.water.HeatCapacity.Type                      Constant
pfset Phase.water.HeatCapacity.GeomNames                 "background"
pfset Phase.water.Geom.background.HeatCapacity.Value        4000.

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                   $TestName


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     5

pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaValue                          1e-2
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-8

pfset Solver.Linear.KrylovDimension                      10
pfset Solver.Linear.Preconditioner                       MGSemi

pfset Solver.Linear.Preconditioner.SymmetricMat          Symmetric
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
 
pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "domain"
pfset TopoSlopesX.Geom.domain.Value 0.0005
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
 
pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "domain"
pfset TopoSlopesY.Geom.domain.Value 0.0005
 
#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
 
pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 2.3e-7
 
#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
 
pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value 1.0e-4

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun test_X
# pfrun test_X -g {0}
pfundist test_X

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
source pftest.tcl


# Expects setting of pressure_l2_error(1) is in output
pftestParseAndEvaluateOutputForTCL test_X.out.txt

set passed 1

if ![pftestIsEqual $pressure_l2_error(1) 5.84881366e-05 "Pressure l2_error is not correct" ] {
    set passed 0
}

if $passed {
    puts "test_X : PASSED"
} {
    puts "test_X : FAILED"
}
