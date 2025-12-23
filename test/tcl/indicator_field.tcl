#
# Problem to test use of indicator field.
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

pfset FileVersion 4

set name "indicator_field"

#
# Control use of indicator field:
# 0 = use domain
# 1 = use indicator field
#
set useIndicatorField 1

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

pfset ComputationalGrid.NX                12
pfset ComputationalGrid.NY                12
pfset ComputationalGrid.NZ                12

set   UpperX                              440
set   UpperY                              120
set   UpperZ                              220

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

pfset GeomInput.Names                 "solid_input indicator_input"

pfset GeomInput.solid_input.InputType  SolidFile
pfset GeomInput.solid_input.GeomNames  domain
pfset GeomInput.solid_input.FileName   ../input/small_domain.pfsol

pfset Geom.domain.Patches             "infiltration z-upper x-lower y-lower \
                                      x-upper y-upper z-lower"

pfset GeomInput.indicator_input.InputType    IndicatorField
pfset GeomInput.indicator_input.GeomNames    "indicator"
pfset Geom.indicator_input.FileName          "small_domain_indicator_field.pfb"

pfset GeomInput.indicator.Value		1

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names                 domain

pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           1.0

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
pfset TimingInfo.StopTime               [expr 30.0*1]
pfset TimingInfo.DumpInterval	        0
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    10.0
pfset TimingInfo.DumpAtEnd              True

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames           domain

pfset Geom.domain.Porosity.Type          Constant
pfset Geom.domain.Porosity.Value         0.3680

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten

if $useIndicatorField {
    pfset Phase.RelPerm.GeomNames          indicator
    pfset Geom.indicator.RelPerm.Alpha         3.34
    pfset Geom.indicator.RelPerm.N             1.982
} {
    pfset Phase.RelPerm.GeomNames          domain
    pfset Geom.domain.RelPerm.Alpha         3.34
    pfset Geom.domain.RelPerm.N             1.982
}

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         domain

pfset Geom.domain.Saturation.Alpha        3.34
pfset Geom.domain.Saturation.N            1.982
pfset Geom.domain.Saturation.SRes         0.2771
pfset Geom.domain.Saturation.SSat         1.0

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
pfset Patch.infiltration.BCPressure.Cycle	      "constant"
pfset Patch.infiltration.BCPressure.alltime.Value     	-0.10
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
pfset Solver.MaxIter                                     1

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

pfset Solver.PrintSubsurfData False
pfset Solver.PrintPressure False
pfset Solver.PrintSaturation False
pfset Solver.PrintConcentration False

pfset Solver.WriteSiloSubsurfData True
pfset Solver.WriteSiloPressure True
pfset Solver.WriteSiloSaturation True
pfset Solver.WriteSiloConcentration True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

file copy -force ../input/small_domain_indicator_field.pfb small_domain_indicator_field.pfb
pfdist small_domain_indicator_field.pfb
pfrun $name

#
# Tests
#
source pftest.tcl
set passed 1

if ![pftestFile $name.out.perm_x.silo "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile $name.out.perm_y.silo "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile $name.out.perm_z.silo "Max difference in perm_z" $sig_digits] {
    set passed 0
}
if ![pftestFile $name.out.porosity.silo "Max difference in porosity" $sig_digits] {
    set passed 0
}

foreach i "00000 00001" {
    if ![pftestFile $name.out.press.$i.silo "Max difference in Pressure for timestep $i" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $name.out.satur.$i.silo "Max difference in Saturation for timestep $i" $sig_digits] {
	set passed 0
    }
}

if $passed {
    puts "$name : PASSED"
} {
    puts "$name : FAILED"
}
