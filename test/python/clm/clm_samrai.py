# this runs CLM test case

#
# Import the ParFlow TCL package
#
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
import sys

clm_samrai = Run("clm_samrai", __file__)

dir_name = get_absolute_path("test_output/clm_samrai")
mkdir(dir_name)

# foreach dir {qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg qflx_top_soil} {
#     file mkdir $dir
# }

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
clm_samrai.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

P = 1
Q = 1
R = 1

NumPatches = sys.argv[1]

clm_samrai.Process.Topology.P = P
clm_samrai.Process.Topology.Q = Q
clm_samrai.Process.Topology.R = R

NumProcs = P * Q * R

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

clm_samrai.ComputationalGrid.Lower.X = 0.0
clm_samrai.ComputationalGrid.Lower.Y = 0.0
clm_samrai.ComputationalGrid.Lower.Z = 0.0

clm_samrai.ComputationalGrid.DX = 1000.0
clm_samrai.ComputationalGrid.DY = 1000.0
clm_samrai.ComputationalGrid.DZ = 0.5

clm_samrai.ComputationalGrid.NX = 5
clm_samrai.ComputationalGrid.NY = 5
clm_samrai.ComputationalGrid.NZ = 10

# if {[expr $NumProcs == 1]} {
#     if {[expr $NumPatches == 1]} {
# 	pfset ProcessGrid.NumSubgrids 1
# 	pfset ProcessGrid.0.P 0
# 	pfset ProcessGrid.0.IX 0
# 	pfset ProcessGrid.0.IY 0
# 	pfset ProcessGrid.0.IZ 0
#
# 	pfset ProcessGrid.0.NX 5
# 	pfset ProcessGrid.0.NY 5
# 	pfset ProcessGrid.0.NZ 10
#     } elseif {[expr $NumPatches == 2]} {
# 	pfset ProcessGrid.NumSubgrids 2
# 	pfset ProcessGrid.0.P 0
# 	pfset ProcessGrid.0.IX 0
# 	pfset ProcessGrid.0.IY 0
# 	pfset ProcessGrid.0.IZ 0
#
# 	pfset ProcessGrid.0.NX 2
# 	pfset ProcessGrid.0.NY 5
# 	pfset ProcessGrid.0.NZ 10
#
# 	pfset ProcessGrid.1.P 0
# 	pfset ProcessGrid.1.IX 2
# 	pfset ProcessGrid.1.IY 0
# 	pfset ProcessGrid.1.IZ 0
#
# 	pfset ProcessGrid.1.NX 3
# 	pfset ProcessGrid.1.NY 5
# 	pfset ProcessGrid.1.NZ 10
#     } elseif {[expr $NumPatches == 3]} {
#
# 	pfset ProcessGrid.NumSubgrids 3
# 	pfset ProcessGrid.0.P 0
# 	pfset ProcessGrid.0.IX 0
# 	pfset ProcessGrid.0.IY 0
# 	pfset ProcessGrid.0.IZ 0
# 	pfset ProcessGrid.0.NX 5
# 	pfset ProcessGrid.0.NY 5
# 	pfset ProcessGrid.0.NZ 8
#
# 	pfset ProcessGrid.1.P 0
# 	pfset ProcessGrid.1.IX 0
# 	pfset ProcessGrid.1.IY 5
# 	pfset ProcessGrid.1.IZ 0
# 	pfset ProcessGrid.1.NX 5
# 	pfset ProcessGrid.1.NY 5
# 	pfset ProcessGrid.1.NZ 8

# 	pfset ProcessGrid.2.P 0
# 	pfset ProcessGrid.2.IX 5
# 	pfset ProcessGrid.2.IY 0
# 	pfset ProcessGrid.2.IZ 0
# 	pfset ProcessGrid.2.NX 5
# 	pfset ProcessGrid.2.NY 10
# 	pfset ProcessGrid.2.NZ 8
#     } else {
# 	puts "Invalid processor/number of subgrid option"
# 	exit
#     }
# }


# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
clm_samrai.GeomInput.Names = "domain_input"


# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
clm_samrai.GeomInput.domain_input.InputType = "Box"
clm_samrai.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
clm_samrai.Geom.domain.Lower.X = 0.0
clm_samrai.Geom.domain.Lower.Y = 0.0
clm_samrai.Geom.domain.Lower.Z = 0.0

clm_samrai.Geom.domain.Upper.X = 5000.0
clm_samrai.Geom.domain.Upper.Y = 5000.0
clm_samrai.Geom.domain.Upper.Z = 5.0

clm_samrai.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
clm_samrai.Geom.Perm.Names = "domain"

clm_samrai.Geom.domain.Perm.Type = "Constant"
clm_samrai.Geom.domain.Perm.Value = 0.2


clm_samrai.Perm.TensorType = "TensorByGeom"

clm_samrai.Geom.Perm.TensorByGeom.Names = "domain"

clm_samrai.Geom.domain.Perm.TensorValX = 1.0
clm_samrai.Geom.domain.Perm.TensorValY = 1.0
clm_samrai.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_samrai.SpecificStorage.Type = "Constant"
clm_samrai.SpecificStorage.GeomNames = "domain"
clm_samrai.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm_samrai.Phase.Names = "water"

clm_samrai.Phase.water.Density.Type = "Constant"
clm_samrai.Phase.water.Density.Value = 1.0

clm_samrai.Phase.water.Viscosity.Type = "Constant"
clm_samrai.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
clm_samrai.Contaminants.Names = ""


# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm_samrai.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
#
clm_samrai.TimingInfo.BaseUnit = 1.0
clm_samrai.TimingInfo.StartCount = 0
clm_samrai.TimingInfo.StartTime = 0.0
clm_samrai.TimingInfo.StopTime = 5
clm_samrai.TimingInfo.DumpInterval = -1
clm_samrai.TimeStep.Type = "Constant"
clm_samrai.TimeStep.Value = 1.0
#

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm_samrai.Geom.Porosity.GeomNames = "domain"

clm_samrai.Geom.domain.Porosity.Type = "Constant"
clm_samrai.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
clm_samrai.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
clm_samrai.Phase.water.Mobility.Type = "Constant"
clm_samrai.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
#
clm_samrai.Phase.RelPerm.Type = "VanGenuchten"
clm_samrai.Phase.RelPerm.GeomNames = "domain"
#
clm_samrai.Geom.domain.RelPerm.Alpha = 3.5
clm_samrai.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm_samrai.Phase.Saturation.Type = "VanGenuchten"
clm_samrai.Phase.Saturation.GeomNames = "domain"
#
clm_samrai.Geom.domain.Saturation.Alpha = 3.5
clm_samrai.Geom.domain.Saturation.N = 2.0
clm_samrai.Geom.domain.Saturation.SRes = 0.01
clm_samrai.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
clm_samrai.Wells.Names = ""


# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
clm_samrai.Cycle.Names = "constant"
clm_samrai.Cycle.constant.Names = "alltime"
clm_samrai.Cycle.constant.alltime.Length = 1
clm_samrai.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
clm_samrai.BCPressure.PatchNames = clm_samrai.Geom.domain.Patches
#
clm_samrai.Patch.x_lower.BCPressure.Type = "FluxConst"
clm_samrai.Patch.x_lower.BCPressure.Cycle = "constant"
clm_samrai.Patch.x_lower.BCPressure.alltime.Value = 0.0
#
clm_samrai.Patch.y_lower.BCPressure.Type = "FluxConst"
clm_samrai.Patch.y_lower.BCPressure.Cycle = "constant"
clm_samrai.Patch.y_lower.BCPressure.alltime.Value = 0.0
#
clm_samrai.Patch.z_lower.BCPressure.Type = "FluxConst"
clm_samrai.Patch.z_lower.BCPressure.Cycle = "constant"
clm_samrai.Patch.z_lower.BCPressure.alltime.Value = 0.0
#
clm_samrai.Patch.x_upper.BCPressure.Type = "FluxConst"
clm_samrai.Patch.x_upper.BCPressure.Cycle = "constant"
clm_samrai.Patch.x_upper.BCPressure.alltime.Value = 0.0
#
clm_samrai.Patch.y_upper.BCPressure.Type = "FluxConst"
clm_samrai.Patch.y_upper.BCPressure.Cycle = "constant"
clm_samrai.Patch.y_upper.BCPressure.alltime.Value = 0.0
#
clm_samrai.Patch.z_upper.BCPressure.Type = "OverlandFlow"
##pfset Patch.z-upper.BCPressure.Type                FluxConst
clm_samrai.Patch.z_upper.BCPressure.Cycle = "constant"
clm_samrai.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
#
clm_samrai.TopoSlopesX.Type = "Constant"
clm_samrai.TopoSlopesX.GeomNames = "domain"
clm_samrai.TopoSlopesX.Geom.domain.Value = -0.001
#
# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------
#
clm_samrai.TopoSlopesY.Type = "Constant"
clm_samrai.TopoSlopesY.GeomNames = "domain"
clm_samrai.TopoSlopesY.Geom.domain.Value = 0.001
#
# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
#
clm_samrai.Mannings.Type = "Constant"
clm_samrai.Mannings.GeomNames = "domain"
clm_samrai.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm_samrai.PhaseSources.water.Type = "Constant"
clm_samrai.PhaseSources.water.GeomNames = "domain"
clm_samrai.PhaseSources.water.Geom.domain.Value = 0.0
#
# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
#
clm_samrai.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
#
clm_samrai.Solver = "Richards"
clm_samrai.Solver.MaxIter = 500
#
clm_samrai.Solver.Nonlinear.MaxIter = 15
clm_samrai.Solver.Nonlinear.ResidualTol = 1e-9
clm_samrai.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_samrai.Solver.Nonlinear.EtaValue = 0.01
clm_samrai.Solver.Nonlinear.UseJacobian = False
clm_samrai.Solver.Nonlinear.DerivativeEpsilon = 1e-12
clm_samrai.Solver.Nonlinear.StepTol = 1e-20
clm_samrai.Solver.Nonlinear.Globalization = "LineSearch"
clm_samrai.Solver.Linear.KrylovDimension = 15
clm_samrai.Solver.Linear.MaxRestart = 2
#
clm_samrai.Solver.Linear.Preconditioner = "PFMGOctree"

clm_samrai.Solver.Linear.Preconditioner.PFMGOctree.BoxSizePowerOf2 = 2
#
clm_samrai.Solver.LSM = "CLM"
clm_samrai.Solver.WriteSiloCLM = True
clm_samrai.Solver.CLM.MetForcing = "1D"
clm_samrai.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_samrai.Solver.CLM.MetFilePath = "./"


clm_samrai.Solver.WriteSiloEvapTrans = True
clm_samrai.Solver.WriteSiloOverlandBCFlux = True


# Initial conditions: water pressure
# ---------------------------------------------------------
#
clm_samrai.ICPressure.Type = "HydroStaticPatch"
clm_samrai.ICPressure.GeomNames = "domain"
clm_samrai.Geom.domain.ICPressure.Value = -2.0
#
clm_samrai.Geom.domain.ICPressure.RefGeom = "domain"
clm_samrai.Geom.domain.ICPressure.RefPatch = "z_upper"


# num_processors = NumProcs
# for {set i 0} { $i <= $num_processors } {incr i} {
#     file delete drv_vegm.dat.$i
#     file copy  drv_vegm.dat drv_vegm.dat.$i
#     file delete drv_clmin.dat.$i
#     file copy drv_clmin.dat drv_clmin.dat.$i
# }

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------


# pfrun clm
# pfundist clm

#
# Tests
#
# source ../pftest.tcl
passed = 1

# if ![pftestFile clm.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
#     set passed 0
# }
# if ![pftestFile clm.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
#     set passed 0
# }
# if ![pftestFile clm.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
#     set passed 0
# }

# for {set i 0} { $i <= 5 } {incr i} {
#     set i_string [format "%05d" $i]
#     if ![pftestFile clm.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits] {
#     set passed 0
#     }
#     if ![pftestFile clm.out.satur.$i_string.pfb "Max difference in Saturation for timestep $i_string" $sig_digits] {
#     set passed 0
#     }
# }

# mask = [pfload clm.out.mask.pfb]
# top = [Parflow::pfcomputetop $mask]

# pfsave $top -pfb "clm.out.top_index.pfb"

# data = [pfload clm.out.press.00000.pfb]
# top_data = [Parflow::pfextracttop $top $data]

# pfsave $data -pfb "clm.out.press.00000.pfb"
# pfsave $top_data -pfb "clm.out.top.press.00000.pfb"

# pfdelete $mask
# pfdelete $top
# pfdelete $data
# pfdelete $top_data

# if ![pftestFile clm.out.top_index.pfb "Max difference in top_index" $sig_digits] {
#     set passed 0
# }

# if ![pftestFile clm.out.top.press.00000.pfb "Max difference in top_clm.out.press.00000.pfb" $sig_digits] {
#     set passed 0
# }


# if $passed {
#     puts "clm : PASSED"
# } {
#     puts "clm : FAILED"
# }

clm_samrai.run(working_directory=dir_name)
