#
# This test is part of a series of tests for the DeepAquiferBC
# Here, we test a flat domain with no flow on the sides.
# The bottom is the DeepAquiferBC and on the top there is some
# water infiltration and evaporation.
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

set p [lindex $argv 0]
set q [lindex $argv 1]

pfset Process.Topology.P $p
pfset Process.Topology.Q $q
pfset Process.Topology.R  1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X   0.0
pfset ComputationalGrid.Lower.Y   0.0
pfset ComputationalGrid.Lower.Z   0.0

pfset ComputationalGrid.DX        4.0
pfset ComputationalGrid.DY        4.0
pfset ComputationalGrid.DZ        0.25

pfset ComputationalGrid.NX         24
pfset ComputationalGrid.NY         24
pfset ComputationalGrid.NZ         40

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input background_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType    Box
pfset GeomInput.domain_input.GeomName     domain

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X         0.0 
pfset Geom.domain.Lower.Y         0.0
pfset Geom.domain.Lower.Z         0.0

pfset Geom.domain.Upper.X        96.0
pfset Geom.domain.Upper.Y        96.0
pfset Geom.domain.Upper.Z        10.0

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
# Setup timing info
#-----------------------------------------------------------------------------
pfset TimingInfo.BaseUnit       1.0
pfset TimingInfo.StartCount     0.0
pfset TimingInfo.StartTime      0.0
pfset TimingInfo.StopTime       48.0
pfset TimingInfo.DumpInterval   6
pfset TimeStep.Type             Constant
pfset TimeStep.Value            6.0

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names                    "constant rainfall"

pfset Cycle.constant.Names           "alltime"
pfset Cycle.constant.alltime.Length   1
pfset Cycle.constant.Repeat          -1

pfset Cycle.rainfall.Names           "rain sunny"
pfset Cycle.rainfall.rain.Length      8
pfset Cycle.rainfall.sunny.Length     40
pfset Cycle.rainfall.Repeat           -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames    [pfget Geom.domain.Patches]

pfset Patch.left.BCPressure.Type             FluxConst
pfset Patch.left.BCPressure.Cycle            constant
pfset Patch.left.BCPressure.alltime.Value    0.0

pfset Patch.right.BCPressure.Type            FluxConst
pfset Patch.right.BCPressure.Cycle           constant
pfset Patch.right.BCPressure.alltime.Value   0.0

pfset Patch.front.BCPressure.Type            FluxConst
pfset Patch.front.BCPressure.Cycle           constant
pfset Patch.front.BCPressure.alltime.Value   0.0

pfset Patch.back.BCPressure.Type             FluxConst
pfset Patch.back.BCPressure.Cycle            constant
pfset Patch.back.BCPressure.alltime.Value    0.0

pfset Patch.bottom.BCPressure.Type           DeepAquifer
pfset Patch.bottom.BCPressure.Cycle          constant
pfset Patch.BCPressure.DeepAquifer.SpecificYield.Type   Constant
pfset Patch.BCPressure.DeepAquifer.SpecificYield.Value  0.1
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Type    Constant
pfset Patch.BCPressure.DeepAquifer.AquiferDepth.Value   90.0
pfset Patch.BCPressure.DeepAquifer.Permeability.Type    Constant
pfset Patch.BCPressure.DeepAquifer.Permeability.Value   0.01836
pfset Patch.BCPressure.DeepAquifer.Elevations.Type      Constant
pfset Patch.BCPressure.DeepAquifer.Elevations.Value     0.0

pfset Patch.top.BCPressure.Type              FluxConst
pfset Patch.top.BCPressure.Cycle             rainfall
pfset Patch.top.BCPressure.rain.Value       -0.075
pfset Patch.top.BCPressure.sunny.Value       0.02

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type                        HydroStaticPatch
pfset ICPressure.GeomNames                   domain
pfset Geom.domain.ICPressure.Value          -2.5
pfset Geom.domain.ICPressure.RefGeom         domain
pfset Geom.domain.ICPressure.RefPatch        top

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity  1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
pfset Geom.Porosity.GeomNames       "domain"
pfset Geom.domain.Porosity.Type     Constant
# Value for Silt soil
pfset Geom.domain.Porosity.Value    0.489

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names               "domain background"
pfset Geom.domain.Perm.Type         Constant
# Value for Silt soil in m/hour
pfset Geom.domain.Perm.Value        0.01836

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain background"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    1.0

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names "water"

pfset Phase.water.Density.Type      Constant
pfset Phase.water.Density.Value     1.0

pfset Phase.water.Viscosity.Type    Constant
pfset Phase.water.Viscosity.Value   1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                Constant
pfset PhaseSources.water.GeomNames           domain
pfset PhaseSources.water.Geom.domain.Value   0.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------
pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     0.657658
pfset Geom.domain.Saturation.N         2.678804
pfset Geom.domain.Saturation.SRes      0.102249
pfset Geom.domain.Saturation.SSat      1.0


#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        0.657658
pfset Geom.domain.RelPerm.N            2.678804

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
pfset Mannings.Type               "Constant"
pfset Mannings.GeomNames          "domain"
pfset Mannings.Geom.domain.Value  5.5e-5

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type               Constant
pfset SpecificStorage.GeomNames          "domain"
pfset Geom.domain.SpecificStorage.Value  1.0e-4
 
#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
pfset TopoSlopesX.Type               "Constant"
pfset TopoSlopesX.GeomNames          "domain"
pfset TopoSlopesX.Geom.domain.Value  0.0
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
pfset TopoSlopesY.Type               "Constant"
pfset TopoSlopesY.GeomNames          "domain"
pfset TopoSlopesY.Geom.domain.Value  0.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names          ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames  ""

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                 ""

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     100000

pfset Solver.Nonlinear.MaxIter                           1000
pfset Solver.Nonlinear.ResidualTol                       1e-8
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-12
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.StepTol                           1e-16

pfset Solver.Linear.KrylovDimension                      30

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10

pfset Solver.PrintPressure     True
pfset Solver.PrintSubsurfData  False
pfset Solver.PrintSaturation   True
pfset Solver.PrintMask         False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
set runname "deep_aquifer_infiltration"
pfrun $runname
pfundist $runname


#
# Tests 
#
source pftest.tcl

set sig_digits 6

set passed 1

foreach i "00000 00001 00002 00003 00004 00005 00006 00007 00008" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits ../correct_output/$runname] {
	      set passed 0
    }
    if ![pftestFile $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits ../correct_output/$runname] {
	      set passed 0
    }
}

if $passed {
    puts "default_single : PASSED"
} {
    puts "default_single : FAILED"
}
