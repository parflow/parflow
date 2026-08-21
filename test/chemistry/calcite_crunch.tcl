#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
pfset FileVersion 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

pfset Process.Topology.P        [lindex $argv 0]
pfset Process.Topology.Q        [lindex $argv 1]
pfset Process.Topology.R        [lindex $argv 2]

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X              0.0
pfset ComputationalGrid.Lower.Y              0.0
pfset ComputationalGrid.Lower.Z              0.0

pfset ComputationalGrid.DX	                 1.0
pfset ComputationalGrid.DY                   1.0
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                   100
pfset ComputationalGrid.NY                   1
pfset ComputationalGrid.NZ                   1

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input background_input source_region_input concen_region_input"

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
pfset Geom.domain.Lower.Z                        0.0

pfset Geom.domain.Upper.X                        100.0
pfset Geom.domain.Upper.Y                        1.0
pfset Geom.domain.Upper.Z                        1.0

pfset Geom.domain.Patches "left right front back bottom top"

#-----------------------------------------------------------------------------
# Background Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#-----------------------------------------------------------------------------
# Background Geometry
#-----------------------------------------------------------------------------
pfset Geom.background.Lower.X -99999999.0
pfset Geom.background.Lower.Y -99999999.0
pfset Geom.background.Lower.Z -99999999.0

pfset Geom.background.Upper.X  99999999.0
pfset Geom.background.Upper.Y  99999999.0
pfset Geom.background.Upper.Z  99999999.0

#-----------------------------------------------------------------------------
# Source_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.source_region_input.InputType      Box
pfset GeomInput.source_region_input.GeomName       source_region

#-----------------------------------------------------------------------------
# Source_Region Geometry
#-----------------------------------------------------------------------------
pfset Geom.source_region.Lower.X    0.0
pfset Geom.source_region.Lower.Y    0.0
pfset Geom.source_region.Lower.Z    0.0

pfset Geom.source_region.Upper.X    100.0
pfset Geom.source_region.Upper.Y    1.0
pfset Geom.source_region.Upper.Z    1.0



#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background"

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    0.25

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0


#-----------------------------------------------------------------------------
# Concen_Region Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.concen_region_input.InputType       Box
pfset GeomInput.concen_region_input.GeomName        concen_region

#-----------------------------------------------------------------------------
# Concen_Region Geometry
#-----------------------------------------------------------------------------
pfset Geom.concen_region.Lower.X  0.0
pfset Geom.concen_region.Lower.Y  0.0
pfset Geom.concen_region.Lower.Z  0.0

pfset Geom.concen_region.Upper.X  100.0
pfset Geom.concen_region.Upper.Y  1.0
pfset Geom.concen_region.Upper.Z  1.0

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
pfset Contaminants.Names			"tce dummy dummy2"
pfset Contaminants.tce.Degradation.Value	 0.0

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
pfset TimingInfo.StopTime            50.0
pfset TimingInfo.DumpInterval	     10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          background

pfset Geom.background.Porosity.Type    Constant
pfset Geom.background.Porosity.Value   0.25

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
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames           background
pfset Geom.background.tce.Retardation.Type     Linear
pfset Geom.background.tce.Retardation.Rate     0.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 20.0
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames "left right front back bottom top"

pfset Patch.left.BCPressure.Type			DirEquilRefPatch
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.RefPatch			bottom
pfset Patch.left.BCPressure.alltime.Value		200.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		100.0

pfset Patch.top.BCPressure.Type				FluxConst
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.alltime.Value		0.0

pfset Patch.bottom.BCPressure.Type			FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.alltime.Value		0.0

pfset Patch.back.BCPressure.Type			FluxConst
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.Value		0.0

pfset Patch.front.BCPressure.Type			FluxConst
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.Value		0.0



#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0

pfset PhaseSources.Type                         Constant
pfset PhaseSources.GeomNames                    background
pfset PhaseSources.Geom.background.Value               0.0

pfset PhaseConcen.water.tce.Type                      Constant
pfset PhaseConcen.water.tce.GeomNames                 "concen_region"
pfset PhaseConcen.water.tce.Geom.concen_region.Value  0.1


#-----------------------------------------------------------------------------
# Temperature sources:
#-----------------------------------------------------------------------------
pfset TempSources.Type                         Constant
pfset TempSources.GeomNames                   "background"
pfset TempSources.Geom.background.Value        0.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "background"
pfset Geom.background.SpecificStorage.Value          1.0e-5

#-----------------------------------------------------------------------------
# Heat Capacity 
#-----------------------------------------------------------------------------

pfset Phase.water.HeatCapacity.Type                      Constant
pfset Phase.water.HeatCapacity.GeomNames                 "background"
pfset Phase.water.Geom.background.HeatCapacity.Value        4000. 

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
# ALQUIMIA INPUT VARS
#---------------------------------------------------------
pfset Solver.Chemistry Alquimia
pfset Chemistry.Engine CrunchFlow
pfset Chemistry.InputFile 1d-calcite-crunch.in


# order of geomnames matters
# just like everything else w/ PF geometries,
# geominputs listed later will overwrite those
# listed earlier if they overlap
pfset GeochemCondition.Type "Constant"
pfset GeochemCondition.GeomNames "concen_region"
pfset GeochemCondition.Names "initial"
pfset GeochemCondition.Geom.concen_region.Value "initial"

pfset BCConcentration.GeochemCondition.Names "west"
pfset BCConcentration.PatchNames "left"
pfset Patch.left.BCConcentration.Type Constant
pfset Patch.left.BCConcentration.Value west


#pfset Solver.WriteSiloConcentration True
pfset Chemistry.ParFlowTimeUnits years



pfset Chemistry.PrintPrimaryMobile True
pfset Chemistry.PrintMineralVolfx True
pfset Chemistry.PrintMineralVolfx True
pfset Chemistry.PrintMineralSurfArea True
pfset Chemistry.PrintMineralRate True
pfset Chemistry.PrintPrimaryFreeIon True
pfset Chemistry.PrintSecondaryFreeIon True
pfset Chemistry.PrintpH True

#-----------------------------------------------------------------------------
# The Solver Impes MaxIter default value changed so to get previous
# results we need to set it back to what it was
#-----------------------------------------------------------------------------
pfset Solver.MaxIter 50000
pfset Solver.CFL 0.6
pfset Solver.AdvectOrder 2
pfset Solver.AdvectEnforceMinMax True
pfset Solver.RelTol 1.0e-35
pfset Solver.AbsTol 1.0e-50
pfset Solver.Nonlinear.ResidualTol 1.0e-15
pfset Solver.PrintVelocities True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun calcite_pf
pfundist calcite_pf

#-----------------------------------------------------------------------------
# Verify against published Molins et al. 2025 (GMD 18:3241) Fig. 2 profiles.
# Reference outputs in correct_output/ were generated by this port and match
# the paper's calcite-dissolution front (~22-23 m at step 5).
#-----------------------------------------------------------------------------
source ../tcl/pftest.tcl
set passed 1
set sig_digits 6

foreach f { pH.00005 PrimaryMobile.02.Ca++.00005 MineralVolfx.00.Calcite.00005 } {
    if ![pftestFile calcite_pf.out.$f.pfb "Max difference in $f" $sig_digits "correct_output"] {
        set passed 0
    }
}

if $passed {
    puts "calcite_crunch : PASSED"
} {
    puts "calcite_crunch : FAILED"
}
