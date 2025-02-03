# this runs CLM_slope test case

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

foreach dir {qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg qflx_top_soil} {
    file mkdir $dir
}

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
pfset ComputationalGrid.Lower.X                0.0
pfset ComputationalGrid.Lower.Y                0.0
pfset ComputationalGrid.Lower.Z                0.0

pfset ComputationalGrid.DX	                   1000.
pfset ComputationalGrid.DY                     1000.
pfset ComputationalGrid.DZ	                   0.5

pfset ComputationalGrid.NX                     5
pfset ComputationalGrid.NY                     5
pfset ComputationalGrid.NZ                     10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input center_input north_input south_input east_input west_input northeast_input southeast_input southwest_input northwest_input"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------

pfset GeomInput.domain_input.GeomName             domain
pfset GeomInput.center_input.GeomName             center
pfset GeomInput.north_input.GeomName              north
pfset GeomInput.south_input.GeomName              south
pfset GeomInput.east_input.GeomName               east
pfset GeomInput.west_input.GeomName               west
pfset GeomInput.northeast_input.GeomName          northeast
pfset GeomInput.southeast_input.GeomName          southeast
pfset GeomInput.southwest_input.GeomName          southwest
pfset GeomInput.northwest_input.GeomName          northwest

pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.center_input.InputType            Box
pfset GeomInput.north_input.InputType             Box
pfset GeomInput.south_input.InputType             Box
pfset GeomInput.east_input.InputType              Box
pfset GeomInput.west_input.InputType              Box
pfset GeomInput.northeast_input.InputType         Box
pfset GeomInput.southeast_input.InputType         Box
pfset GeomInput.southwest_input.InputType         Box
pfset GeomInput.northwest_input.InputType         Box

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X                        0.0
pfset Geom.domain.Lower.Y                        0.0
pfset Geom.domain.Lower.Z                        0.0

pfset Geom.domain.Upper.X                        5000.
pfset Geom.domain.Upper.Y                        5000.
pfset Geom.domain.Upper.Z                        5.

pfset Geom.domain.Patches  "x-lower x-upper y-lower y-upper z-lower z-upper"

# ---------------------------------------------------------
# Center tile Geometry
# ---------------------------------------------------------
pfset Geom.center.Lower.X                        2000
pfset Geom.center.Lower.Y                        2000
pfset Geom.center.Lower.Z                        0.0

pfset Geom.center.Upper.X                        3000
pfset Geom.center.Upper.Y                        3000
pfset Geom.center.Upper.Z                        5.0
# was 0.05
#
#
# ---------------------------------------------------------
# North Slope Geometry
# ---------------------------------------------------------
pfset Geom.north.Lower.X                        2000
pfset Geom.north.Lower.Y                        3000
pfset Geom.north.Lower.Z                        0.0

pfset Geom.north.Upper.X                        3000
pfset Geom.north.Upper.Y                        5000
pfset Geom.north.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# South Slope Geometry
# ---------------------------------------------------------
pfset Geom.south.Lower.X                        2000
pfset Geom.south.Lower.Y                        0.0
pfset Geom.south.Lower.Z                        0.0

pfset Geom.south.Upper.X                        3000
pfset Geom.south.Upper.Y                        2000
pfset Geom.south.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# East Slope Geometry
# ---------------------------------------------------------
pfset Geom.east.Lower.X                        3000
pfset Geom.east.Lower.Y                        2000
pfset Geom.east.Lower.Z                        0.0

pfset Geom.east.Upper.X                        5000
pfset Geom.east.Upper.Y                        3000
pfset Geom.east.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# West Slope Geometry
# ---------------------------------------------------------
pfset Geom.west.Lower.X                        0.0
pfset Geom.west.Lower.Y                        2000
pfset Geom.west.Lower.Z                        0.0

pfset Geom.west.Upper.X                        2000
pfset Geom.west.Upper.Y                        3000
pfset Geom.west.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# Northeast Slope Geometry
# ---------------------------------------------------------
pfset Geom.northeast.Lower.X                        3000
pfset Geom.northeast.Lower.Y                        3000
pfset Geom.northeast.Lower.Z                        0.0

pfset Geom.northeast.Upper.X                        5000
pfset Geom.northeast.Upper.Y                        5000
pfset Geom.northeast.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# Southeast Slope Geometry
# ---------------------------------------------------------
pfset Geom.southeast.Lower.X                        3000
pfset Geom.southeast.Lower.Y                        0.0
pfset Geom.southeast.Lower.Z                        0.0

pfset Geom.southeast.Upper.X                        5000
pfset Geom.southeast.Upper.Y                        2000
pfset Geom.southeast.Upper.Z                        5.0
#was 0.05
#
# ---------------------------------------------------------
# Southwest Slope Geometry
# ---------------------------------------------------------
pfset Geom.southwest.Lower.X                        0.0
pfset Geom.southwest.Lower.Y                        0.0
pfset Geom.southwest.Lower.Z                        0.0

pfset Geom.southwest.Upper.X                        2000
pfset Geom.southwest.Upper.Y                        2000
pfset Geom.southwest.Upper.Z                        5.0
# was 0.05
#
# ---------------------------------------------------------
# Northwest Slope Geometry
# ---------------------------------------------------------
pfset Geom.northwest.Lower.X                        0.0
pfset Geom.northwest.Lower.Y                        3000
pfset Geom.northwest.Lower.Z                        0.0

pfset Geom.northwest.Upper.X                        2000
pfset Geom.northwest.Upper.Y                        5000
pfset Geom.northwest.Upper.Z                        5.0
# was 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "domain"

pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           0.2


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
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit        1.0
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        5
pfset TimingInfo.DumpInterval    -1
pfset TimeStep.Type              Constant
pfset TimeStep.Value             1.0


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
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          "domain"

pfset Geom.domain.RelPerm.Alpha         3.5
pfset Geom.domain.RelPerm.N             2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         "domain"

pfset Geom.domain.Saturation.Alpha        3.5
pfset Geom.domain.Saturation.N            2.
pfset Geom.domain.Saturation.SRes         0.01
pfset Geom.domain.Saturation.SSat         1.0

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
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

pfset Patch.x-lower.BCPressure.Type                   FluxConst
pfset Patch.x-lower.BCPressure.Cycle                  "constant"
pfset Patch.x-lower.BCPressure.alltime.Value          0.0

pfset Patch.y-lower.BCPressure.Type                   FluxConst
pfset Patch.y-lower.BCPressure.Cycle                  "constant"
pfset Patch.y-lower.BCPressure.alltime.Value          0.0

pfset Patch.z-lower.BCPressure.Type                   FluxConst
pfset Patch.z-lower.BCPressure.Cycle                  "constant"
pfset Patch.z-lower.BCPressure.alltime.Value          0.0

pfset Patch.x-upper.BCPressure.Type                   FluxConst
pfset Patch.x-upper.BCPressure.Cycle                  "constant"
pfset Patch.x-upper.BCPressure.alltime.Value          0.0

pfset Patch.y-upper.BCPressure.Type                   FluxConst
pfset Patch.y-upper.BCPressure.Cycle                  "constant"
pfset Patch.y-upper.BCPressure.alltime.Value          0.0

pfset Patch.z-upper.BCPressure.Type                   OverlandFlow
##pfset Patch.z-upper.BCPressure.Type                FluxConst
pfset Patch.z-upper.BCPressure.Cycle                  "constant"
pfset Patch.z-upper.BCPressure.alltime.Value          0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "center north south east west northeast southeast southwest northwest"
pfset TopoSlopesX.Geom.center.Value 0.0
pfset TopoSlopesX.Geom.north.Value 0.0
pfset TopoSlopesX.Geom.south.Value 0.0
pfset TopoSlopesX.Geom.west.Value 10.0
pfset TopoSlopesX.Geom.east.Value -10.0
pfset TopoSlopesX.Geom.northeast.Value -10.0
pfset TopoSlopesX.Geom.southeast.Value -10.0
pfset TopoSlopesX.Geom.southwest.Value 10.0
pfset TopoSlopesX.Geom.northwest.Value 10.0


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "center north south east west northeast southeast southwest northwest"
pfset TopoSlopesY.Geom.center.Value 0.0
pfset TopoSlopesY.Geom.north.Value -10.0
pfset TopoSlopesY.Geom.south.Value 10.0
pfset TopoSlopesY.Geom.west.Value 0.0
pfset TopoSlopesY.Geom.east.Value 0.0
pfset TopoSlopesY.Geom.northeast.Value -10.0
pfset TopoSlopesY.Geom.southeast.Value 10.0
pfset TopoSlopesY.Geom.southwest.Value 10.0
pfset TopoSlopesY.Geom.northwest.Value -10.0


#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    domain
pfset PhaseSources.water.Geom.domain.Value        0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                      NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                             Richards
pfset Solver.MaxIter                                     500

pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.01
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.StepTol                           1e-20
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      15
pfset Solver.Linear.MaxRestart                           2

pfset Solver.Linear.Preconditioner                       PFMG
pfset Solver.PrintSubsurf                                False
pfset Solver.Drop                                        1E-20
pfset Solver.AbsTol                                      1E-9

pfset Solver.LSM                                         CLM
pfset Solver.WriteSiloCLM                                False
pfset Solver.CLM.MetForcing                              1D
pfset Solver.CLM.MetFileName                             narr_1hr.sc3.txt.0
pfset Solver.CLM.MetFilePath                             ./

pfset Solver.WriteSiloEvapTrans                          False
pfset Solver.WriteSiloOverlandBCFlux                     False
pfset Solver.PrintCLM                                    False
pfset Solver.CLM.UseSlopeAspect                          True


#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -2.0

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   z-upper



set num_processors [expr [pfget Process.Topology.P] * [pfget Process.Topology.Q] * [pfget Process.Topology.R]]
for {set i 0} { $i <= $num_processors } {incr i} {
    file delete drv_vegm.dat.$i
    file copy  drv_vegm.dat drv_vegm.dat.$i
    file delete drv_clmin.dat.$i
    file copy drv_clmin.dat drv_clmin.dat.$i
}

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


pfrun clm_slope
pfundist clm_slope

#
# Tests
#
source ../pftest.tcl
set passed 1

set correct_output_dir "../../correct_output/clm_output"

if ![pftestFile clm_slope.out.perm_x.pfb "Max difference in perm_x" $sig_digits $correct_output_dir] {
    set passed 0
}
if ![pftestFile clm_slope.out.perm_y.pfb "Max difference in perm_y" $sig_digits $correct_output_dir] {
    set passed 0
}
if ![pftestFile clm_slope.out.perm_z.pfb "Max difference in perm_z" $sig_digits $correct_output_dir] {
    set passed 0
}

for {set i 0} { $i <= 5 } {incr i} {
    set i_string [format "%05d" $i]
    if ![pftestFile clm_slope.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits $correct_output_dir] {
    set passed 0
    }
    if ![pftestFile clm_slope.out.satur.$i_string.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
    set passed 0
    }
}

set mask [pfload clm_slope.out.mask.pfb]
set top [Parflow::pfcomputetop $mask]

pfsave $top -pfb "clm_slope.out.top_index.pfb"

set data [pfload clm_slope.out.press.00000.pfb]
set top_data [Parflow::pfextracttop $top $data]

pfsave $data -pfb "clm_slope.out.press.00000.pfb"
pfsave $top_data -pfb "clm_slope.out.top.press.00000.pfb"

pfdelete $mask
pfdelete $top
pfdelete $data
pfdelete $top_data

if ![pftestFile clm_slope.out.top_index.pfb "Max difference in top_index" $sig_digits $correct_output_dir] {
    set passed 0
}

if ![pftestFile clm_slope.out.top.press.00000.pfb "Max difference in top_clm_slope.out.press.00000.pfb" $sig_digits $correct_output_dir] {
    set passed 0
}



if $passed {
    puts "clm_slope : PASSED"
} {
    puts "clm_slope : FAILED"
}
