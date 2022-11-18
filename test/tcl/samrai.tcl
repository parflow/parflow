#  This runs the basic pfmg test case based off of default richards
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

set P  [lindex $argv 0]
set Q  [lindex $argv 1]
set R  [lindex $argv 2]

set NumPatches [lindex $argv 3]

pfset Process.Topology.P $P
pfset Process.Topology.Q $Q   
pfset Process.Topology.R $R

set NumProcs [expr $P * $Q * $R]

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                -10.0
pfset ComputationalGrid.Lower.Y                 10.0
pfset ComputationalGrid.Lower.Z                  1.0

pfset ComputationalGrid.DX	                 8.8888888888888893
pfset ComputationalGrid.DY                      10.666666666666666
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                      10
pfset ComputationalGrid.NY                      10
pfset ComputationalGrid.NZ                       8

#---------------------------------------------------------
# Processor grid
#---------------------------------------------------------
if {[expr $NumProcs == 1]} {
    if {[expr $NumPatches == 1]} {
	# {ix = 0, iy = 0, iz = 0, nx = 10, ny = 10, nz = 8, sx = 1, sy = 1, sz = 1, rx = 0, ry = 0, rz = 0, level = 0, process = 0}

	
	pfset ProcessGrid.NumSubgrids 1
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	
	pfset ProcessGrid.0.NX 10
	pfset ProcessGrid.0.NY 10
	pfset ProcessGrid.0.NZ 8
    } elseif {[expr $NumPatches == 2]} {
	# {ix = 0, iy = 0, iz = 0, nx = 10, ny = 5, nz = 8, sx = 1, sy = 1, sz = 1, rx = 0, ry = 0, rz = 0, level = 0, process = 0}
	# {ix = 0, iy = 5, iz = 0, nx = 10, ny = 5, nz = 8, sx = 1, sy = 1, sz = 1, rx = 0, ry = 0, rz = 0, level = 0, process = 1}
	
	pfset ProcessGrid.NumSubgrids 2
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	
	pfset ProcessGrid.0.NX 10
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 0
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	
	pfset ProcessGrid.1.NX 10
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8
    } elseif {[expr $NumPatches == 3]} {
	
	pfset ProcessGrid.NumSubgrids 3
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	pfset ProcessGrid.0.NX 5
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 0
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	pfset ProcessGrid.1.NX 5
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8

	pfset ProcessGrid.2.P 0
	pfset ProcessGrid.2.IX 5
	pfset ProcessGrid.2.IY 0
	pfset ProcessGrid.2.IZ 0
	pfset ProcessGrid.2.NX 5
	pfset ProcessGrid.2.NY 10
	pfset ProcessGrid.2.NZ 8
    } else {
	puts "Invalid processor/number of subgrid option"
	exit
    }
} elseif {[expr $NumProcs == 2]} {

    if {[expr $NumPatches == 2]} {
	
	pfset ProcessGrid.NumSubgrids 2
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	pfset ProcessGrid.0.NX 10
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 1
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	pfset ProcessGrid.1.NX 10
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8

    } elseif {[expr $NumPatches == 3]} {
	
	pfset ProcessGrid.NumSubgrids 3
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	pfset ProcessGrid.0.NX 5
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 0
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	pfset ProcessGrid.1.NX 5
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8

	pfset ProcessGrid.2.P 1
	pfset ProcessGrid.2.IX 5
	pfset ProcessGrid.2.IY 0
	pfset ProcessGrid.2.IZ 0
	pfset ProcessGrid.2.NX 5
	pfset ProcessGrid.2.NY 10
	pfset ProcessGrid.2.NZ 8
    } elseif {[expr $NumPatches == 4]} {
	pfset ProcessGrid.NumSubgrids 4
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	pfset ProcessGrid.0.NX 5
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 0
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	pfset ProcessGrid.1.NX 5
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8

	pfset ProcessGrid.2.P 1
	pfset ProcessGrid.2.IX 5
	pfset ProcessGrid.2.IY 0
	pfset ProcessGrid.2.IZ 0
	pfset ProcessGrid.2.NX 5
	pfset ProcessGrid.2.NY 5
	pfset ProcessGrid.2.NZ 8


	pfset ProcessGrid.3.P 1
	pfset ProcessGrid.3.IX 5
	pfset ProcessGrid.3.IY 5
	pfset ProcessGrid.3.IZ 0
	pfset ProcessGrid.3.NX 5
	pfset ProcessGrid.3.NY 5
	pfset ProcessGrid.3.NZ 8
    } else {

	puts "Invalid processor/number of subgrid option"
	exit
    }
} else {
    puts "Invalid processor/number of subgrid option"
    exit
}

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input background_input source_region_input \
		       concen_region_input"


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X                        -10.0 
pfset Geom.domain.Lower.Y                         10.0
pfset Geom.domain.Lower.Z                          1.0

pfset Geom.domain.Upper.X                        150.0
pfset Geom.domain.Upper.Y                        170.0
pfset Geom.domain.Upper.Z                          9.0

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


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
pfset GeomInput.source_region_input.InputType      Box
pfset GeomInput.source_region_input.GeomName       source_region

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
pfset Geom.source_region.Lower.X    65.56
pfset Geom.source_region.Lower.Y    79.34
pfset Geom.source_region.Lower.Z     4.5

pfset Geom.source_region.Upper.X    74.44
pfset Geom.source_region.Upper.Y    89.99
pfset Geom.source_region.Upper.Z     5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
pfset GeomInput.concen_region_input.InputType       Box
pfset GeomInput.concen_region_input.GeomName        concen_region

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
pfset Geom.concen_region.Lower.X   60.0
pfset Geom.concen_region.Lower.Y   80.0
pfset Geom.concen_region.Lower.Z    4.0

pfset Geom.concen_region.Upper.X   80.0
pfset Geom.concen_region.Upper.Y  100.0
pfset Geom.concen_region.Upper.Z    6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background"

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    4.0

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
pfset TimingInfo.StopTime               0.010
pfset TimingInfo.DumpInterval	       -1
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    0.001

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

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        0.005
pfset Geom.domain.RelPerm.N            2.0    

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     0.005
pfset Geom.domain.Saturation.N         2.0
pfset Geom.domain.Saturation.SRes      0.2
pfset Geom.domain.Saturation.SSat      0.99

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

pfset Patch.left.BCPressure.Type			DirEquilRefPatch
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.RefGeom			domain
pfset Patch.left.BCPressure.RefPatch			bottom
pfset Patch.left.BCPressure.alltime.Value		5.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		3.0

pfset Patch.front.BCPressure.Type			FluxConst
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.Value		0.0

pfset Patch.back.BCPressure.Type			FluxConst
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.Value		0.0

pfset Patch.bottom.BCPressure.Type			FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.alltime.Value		0.0

pfset Patch.top.BCPressure.Type			        FluxConst
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.alltime.Value		0.0

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
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      3.0
pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   bottom

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
pfset Solver.MaxIter                                     5

pfset Solver.Nonlinear.MaxIter                           10
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-5
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-2

pfset Solver.Linear.KrylovDimension                      10

pfset Solver.Linear.Preconditioner                       PFMGOctree

pfset Solver.Linear.Preconditioner.PFMGOctree.BoxSizePowerOf2 2

#pfset Solver.Linear.Preconditioner.PFMG.MaxIter          1
#pfset Solver.Linear.Preconditioner.PFMG.NumPreRelax      100
#pfset Solver.Linear.Preconditioner.PFMG.NumPostRelax     100
#pfset Solver.Linear.Preconditioner.PFMG.Smoother         100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun samrai
pfundist samrai

#
# Tests 
#
source pftest.tcl
set passed 1

if ![pftestFile samrai.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile samrai.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile samrai.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001 00002 00003 00004 00005" {
    if ![pftestFile samrai.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
    set passed 0
}
    if ![pftestFile samrai.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
    set passed 0
}
}


if $passed {
    puts "samrai : PASSED"
} {
    puts "samrai : FAILED"
}
