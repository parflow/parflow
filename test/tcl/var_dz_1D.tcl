# Runs a simple sand draining problem, rectangular domain
# with variable dz and a heterogeneous subsurface with different K the top and bottom layers

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
pfset ComputationalGrid.Lower.X                0.0
pfset ComputationalGrid.Lower.Y                0.0
pfset ComputationalGrid.Lower.Z                0.0

pfset ComputationalGrid.DX	                 1.0
pfset ComputationalGrid.DY                       1.0
pfset ComputationalGrid.DZ	                 0.1

pfset ComputationalGrid.NX                      1
pfset ComputationalGrid.NY                      1
pfset ComputationalGrid.NZ                     14

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input het_input1 het_input2"

#---------------------------------------------------------
# Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

pfset GeomInput.het_input1.InputType            Box
pfset GeomInput.het_input1.GeomName            het1

pfset GeomInput.het_input2.InputType            Box
pfset GeomInput.het_input2.GeomName            het2

#---------------------------------------------------------
# Geometry
#---------------------------------------------------------
#domain
pfset Geom.domain.Lower.X                        0.0
pfset Geom.domain.Lower.Y                         0.0
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                        1.0
pfset Geom.domain.Upper.Y                        1.0
pfset Geom.domain.Upper.Z                          1.4

pfset Geom.domain.Patches "left right front back bottom top"

#het1
pfset Geom.het1.Lower.X                        0.0
pfset Geom.het1.Lower.Y                         0.0
pfset Geom.het1.Lower.Z                          1.3

pfset Geom.het1.Upper.X                        1.0
pfset Geom.het1.Upper.Y                        1.0
pfset Geom.het1.Upper.Z                          1.4

#het2
pfset Geom.het2.Lower.X                        0.0
pfset Geom.het2.Lower.Y                        0.0
pfset Geom.het2.Lower.Z                        0.0

pfset Geom.het2.Upper.X                        1.0
pfset Geom.het2.Upper.Y                        1.0
pfset Geom.het2.Upper.Z                        0.1

#--------------------------------------------
# variable dz assignments
#------------------------------------------
pfset Solver.Nonlinear.VariableDz     True
#pfset Solver.Nonlinear.VariableDz     False
pfset dzScale.GeomNames            domain
pfset dzScale.Type            nzList
pfset dzScale.nzListNumber       14
pfset Cell.0.dzScale.Value 1.2
pfset Cell.1.dzScale.Value 1.0
pfset Cell.2.dzScale.Value 1.0
pfset Cell.3.dzScale.Value 1.0
pfset Cell.4.dzScale.Value 1.0
pfset Cell.5.dzScale.Value 1.0
pfset Cell.6.dzScale.Value 1.0
pfset Cell.7.dzScale.Value 1.0
pfset Cell.8.dzScale.Value 1.0
pfset Cell.9.dzScale.Value 1.0
pfset Cell.10.dzScale.Value 0.15
pfset Cell.11.dzScale.Value 0.1
pfset Cell.12.dzScale.Value 0.1
pfset Cell.13.dzScale.Value 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "domain het1 het2"

pfset Geom.domain.Perm.Type     Constant
pfset Geom.domain.Perm.Value    5.129

pfset Geom.het1.Perm.Type     Constant
pfset Geom.het1.Perm.Value    0.0001

pfset Geom.het2.Perm.Type     Constant
pfset Geom.het2.Perm.Value    0.001

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
pfset Geom.domain.SpecificStorage.Value 1.0e-6

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
pfset TimingInfo.StopTime               50.0
pfset TimingInfo.DumpInterval	       -100
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    0.01
pfset TimeStep.Value                    0.01

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames         domain

pfset Geom.domain.Porosity.Type    Constant
pfset Geom.domain.Porosity.Value   0.4150

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        2.7
pfset Geom.domain.RelPerm.N            3.8

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     2.7
pfset Geom.domain.Saturation.N         3.8
pfset Geom.domain.Saturation.SRes      0.106
pfset Geom.domain.Saturation.SSat      1.0

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

pfset Patch.left.BCPressure.Type			FluxConst
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.RefPatch			bottom
pfset Patch.left.BCPressure.alltime.Value		0.0

pfset Patch.right.BCPressure.Type			FluxConst
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		0.0

pfset Patch.front.BCPressure.Type			FluxConst
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.Value		0.0

pfset Patch.back.BCPressure.Type			FluxConst
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.Value		0.0

pfset Patch.bottom.BCPressure.Type		 DirEquilRefPatch
pfset Patch.bottom.BCPressure.Type		 FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.RefGeom			domain
pfset Patch.bottom.BCPressure.RefPatch			bottom
pfset Patch.bottom.BCPressure.alltime.Value		0.0

pfset Patch.top.BCPressure.Type			       DirEquilRefPatch
pfset Patch.top.BCPressure.Type			      FluxConst
#pfset Patch.top.BCPressure.Type			      OverlandFlow 
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.RefGeom			domain
pfset Patch.top.BCPressure.RefPatch		 	bottom
pfset Patch.top.BCPressure.alltime.Value		-0.0001


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
pfset Geom.domain.ICPressure.Value                      -10.0
pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   top

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

pfset Solver.Nonlinear.MaxIter                           200
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaChoice                         Walker1
pfset Solver.Nonlinear.EtaValue                          1e-5
pfset Solver.Nonlinear.UseJacobian                      True
#pfset Solver.Nonlinear.UseJacobian                     False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-10

pfset Solver.Linear.KrylovDimension                      10

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner                       PFMG
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10


#-----------------------------------------------------------------------------
# Run and do tests
#-----------------------------------------------------------------------------
set runname var.dz.1d
pfrun $runname
pfundist $runname

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

foreach i "00000 00005 00010 00015 00020 00025" {
    if ![pftestFile $runname.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
    set passed 0
}
    if ![pftestFile  $runname.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
    set passed 0
}
}

if $passed {
    puts "1d_var_dz : PASSED"
} {
    puts "1d_var_dz : FAILED"
}

#set k1 [pfload -pfb var.dz.1d.out.perm_x.pfb]
#set press1 [pfload -pfb var.dz.1d.out.press.00001.pfb]
#set press2 [pfload -pfb var.dz.1d.out.press.00010.pfb]
#set press3 [pfload -pfb var.dz.1d.out.press.00024.pfb]

#for {set k 13} {$k >= 0} {incr k -1} {
#	set outp1 [pfgetelt $press1 0 0 $k]
#	set outp2 [pfgetelt $press2 0 0 $k]
#	set outp3 [pfgetelt $press3 0 0 $k]
#		set outk1 [pfgetelt $k1 0 0 $k]
#	puts stdout "$k $outp1 $outp2 $outp3 $outk1"
#}
