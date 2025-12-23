#  This test problem runs the Richards' equation solvers
#  on the eqn:  - div (p grad p) = f where p = x^3y^2 + sin(xy) + 1
#  and f is chosen to guarantee the correct solution.
#  For 128 x 128 unknowns, the following line should be printed on
#  the screen:
#
#  l2-error in pressure:       2.91683678e-04

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
set   TestName           X3Y2PlusSinXYPlus1

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                  0.0
pfset ComputationalGrid.Lower.Y                  0.0
pfset ComputationalGrid.Lower.Z                  0.0

pfset ComputationalGrid.DX	                 0.0078125
pfset ComputationalGrid.DY                       0.0078125
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                       128
pfset ComputationalGrid.NY                       128
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
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "background"
pfset Geom.background.SpecificStorage.Value          1.0e-5

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

pfset Patch.front.BCPressure.Type			ExactSolution
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.PredefinedFunction $TestName

pfset Patch.back.BCPressure.Type			ExactSolution
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.PredefinedFunction  $TestName

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
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         PredefinedFunction
pfset PhaseSources.water.PredefinedFunction           $TestName


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
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-2
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-8

pfset Solver.Linear.KrylovDimension                      10
pfset Solver.Linear.Preconditioner                       MGSemi

pfset Solver.Linear.Preconditioner.SymmetricMat          Symmetric
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

pfset Solver.TerrainFollowingGrid.SlopeUpwindFormulation Upwind

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfrun test_$TestName
pfundist test_$TestName

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST test_$TestName
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    if ![pftestFile $TEST.out.press.00001.pfb "Max difference in Pressure" $sig_digits] {
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

    if ![pftestFile $TEST.out.porosity.pfb "Max difference in porosity" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.satur.00001.pfb "Max difference in concen" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.specific_storage.pfb "Max difference in specific storage" $sig_digits] {
	set passed 0
    }
    
    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
