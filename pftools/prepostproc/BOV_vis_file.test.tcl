# this runs a simple test problem to show how tcl scripting can be used to 
# to automatically generate a header for the free, visualization tool, VISIT.
# it writes a BOV or "Box of Values" header file (.bov) for a number of parflow
# output files, allowing the .pfb files to be directly read by VISIT.
#  Visit may be downloaded (for free) at http://www.llnl.gov/visit/
#
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

pfset Process.Topology.P        1
pfset Process.Topology.Q        1
pfset Process.Topology.R        1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X                0.0
pfset ComputationalGrid.Lower.Y                0.0
pfset ComputationalGrid.Lower.Z                 0.0

pfset ComputationalGrid.DX	                10.0
pfset ComputationalGrid.DY                      10.0
pfset ComputationalGrid.DZ	                1.0

pfset ComputationalGrid.NX                      20
pfset ComputationalGrid.NY                      20
pfset ComputationalGrid.NZ                      20

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input"


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
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                        200.0
pfset Geom.domain.Upper.Y                        200.0
pfset Geom.domain.Upper.Z                        20.0

pfset Geom.domain.Patches "left right front back bottom top"



#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "domain"


pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           4.0


pfset Perm.TensorType               TensorByGeom
pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       ""
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
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		-1
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime            0.0
pfset TimingInfo.DumpInterval	       -1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          domain

pfset Geom.domain.Porosity.Type    Constant
pfset Geom.domain.Porosity.Value   0.390

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
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names ""


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
pfset Patch.left.BCPressure.alltime.Value		21.0

pfset Patch.right.BCPressure.Type			DirEquilRefPatch
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.RefGeom			domain
pfset Patch.right.BCPressure.RefPatch			bottom
pfset Patch.right.BCPressure.alltime.Value		20.0

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
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

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
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames ""
pfset Mannings.Geom.domain.Value 0.

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.Type                         Constant
pfset PhaseSources.GeomNames                    domain
pfset PhaseSources.Geom.domain.Value        0.0

#-----------------------------------------------------------------------------
#  Solver Impes  
#-----------------------------------------------------------------------------
pfset Solver.MaxIter 50
pfset Solver.AbsTol  1E-10
pfset Solver.Drop   1E-15

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

set run_name "bov_test"

pfrun $run_name
pfundist $run_name

# we use tcl scripting to write a BOV (Box of Values) header for VISIT allowing
# VISIT to read a parflow simpe binary (sb) file directly
#
# NOTE: this currently only works for SINGLE PROCESSOR runs Q=P=R=1
#
# we open a file, one .bov file for each .pfb output file we want to write a VISIT header for
#

set bov_file(1) "press"
set bov_file(2) "perm_x"
set bov_file(3) "perm_y"
set bov_file(4) "perm_z"

set n_bov 4

# use the pfget command to get the domain information
# we could also do this directly

set dx [ pfget ComputationalGrid.DX ]
set dy [ pfget ComputationalGrid.DY ]
set dz [ pfget ComputationalGrid.DZ ]

set nx [ pfget ComputationalGrid.NX ]
set ny [ pfget ComputationalGrid.NY ]
set nz [ pfget ComputationalGrid.NZ ]

set x0 [ pfget ComputationalGrid.Lower.X ]
set y0 [ pfget ComputationalGrid.Lower.Y ]
set z0 [ pfget ComputationalGrid.Lower.Z ]

for {set k 1} {$k <= $n_bov} {incr k 1} {

# here we use tcl to build the bov and pfb file names
#
set bov_fname "$run_name.out.$bov_file($k).bov"
set pfb_fname "$run_name.out.$bov_file($k).pfb"
#
#
set fileId [open $bov_fname w 0600]
puts $fileId "TIME: 0.00"
puts $fileId "DATA_FILE: $pfb_fname"
puts $fileId "DATA_SIZE: $nx $ny $nz"
puts $fileId "DATA_FORMAT: DOUBLE"
puts $fileId "VARIABLE: $bov_file($k)"
puts $fileId "DATA_ENDIAN: BIG"
puts $fileId "CENTERING: zonal"
puts $fileId "BRICK_ORIGIN: $x0 $y0 $z0" 
puts $fileId "BRICK_SIZE: $dx $dy $dz"
puts $fileId "BYTE_OFFSET: 100"
puts $fileId "DIVIDE_BRICK: FALSE"
puts $fileId "DATA_COMPONENTS: 1"

close $fileId

}