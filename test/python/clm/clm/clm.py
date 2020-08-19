# this runs CLM test case

#
# Import the ParFlow TCL package
#
from parflow import Run
clm = Run("clm", __file__)

# foreach dir {qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg qflx_top_soil} {
#     file mkdir $dir
# }

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
clm.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm.Process.Topology.P = [lindex $argv 0]
clm.Process.Topology.Q = [lindex $argv 1]
clm.Process.Topology.R = [lindex $argv 2]

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 1000.
clm.ComputationalGrid.DY = 1000.
clm.ComputationalGrid.DZ = 0.5

clm.ComputationalGrid.NX = 5
clm.ComputationalGrid.NY = 5
clm.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
clm.GeomInput.Names = 'domain_input'


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
clm.GeomInput.domain_input.InputType = 'Box'
clm.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
clm.Geom.domain.Lower.X = 0.0
clm.Geom.domain.Lower.Y = 0.0
clm.Geom.domain.Lower.Z = 0.0

clm.Geom.domain.Upper.X = 5000.
clm.Geom.domain.Upper.Y = 5000.
clm.Geom.domain.Upper.Z = 5.

clm.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clm.Geom.Perm.Names = 'domain'

clm.Geom.domain.Perm.Type = 'Constant'
clm.Geom.domain.Perm.Value = 0.2


clm.Perm.TensorType = 'TensorByGeom'

clm.Geom.Perm.TensorByGeom.Names = 'domain'

clm.Geom.domain.Perm.TensorValX = 1.0
clm.Geom.domain.Perm.TensorValY = 1.0
clm.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm.SpecificStorage.Type = 'Constant'
clm.SpecificStorage.GeomNames = 'domain'
clm.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm.Phase.Names = 'water'

clm.Phase.water.Density.Type = 'Constant'
clm.Phase.water.Density.Value = 1.0

clm.Phase.water.Viscosity.Type = 'Constant'
clm.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
clm.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
#  
clm.TimingInfo.BaseUnit = 1.0
clm.TimingInfo.StartCount = 0
clm.TimingInfo.StartTime = 0.0
clm.TimingInfo.StopTime = 5
clm.TimingInfo.DumpInterval = -1
clm.TimeStep.Type = 'Constant'
clm.TimeStep.Value = 1.0
#  

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm.Geom.Porosity.GeomNames = 'domain'

clm.Geom.domain.Porosity.Type = 'Constant'
clm.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
clm.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
clm.Phase.water.Mobility.Type = 'Constant'
clm.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
#  
clm.Phase.RelPerm.Type = 'VanGenuchten'
clm.Phase.RelPerm.GeomNames = 'domain'
#  
clm.Geom.domain.RelPerm.Alpha = 3.5
clm.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm.Phase.Saturation.Type = 'VanGenuchten'
clm.Phase.Saturation.GeomNames = 'domain'
#  
clm.Geom.domain.Saturation.Alpha = 3.5
clm.Geom.domain.Saturation.N = 2.
clm.Geom.domain.Saturation.SRes = 0.01
clm.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clm.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clm.Cycle.Names = 'constant'
clm.Cycle.constant.Names = 'alltime'
clm.Cycle.constant.alltime.Length = 1
clm.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clm.BCPressure.PatchNames = [pfget Geom.domain.Patches]
#  
clm.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm.Patch.x_lower.BCPressure.Cycle = 'constant'
clm.Patch.x_lower.BCPressure.alltime.Value = 0.0
#  
clm.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm.Patch.y_lower.BCPressure.Cycle = 'constant'
clm.Patch.y_lower.BCPressure.alltime.Value = 0.0
#  
clm.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm.Patch.z_lower.BCPressure.Cycle = 'constant'
clm.Patch.z_lower.BCPressure.alltime.Value = 0.0
#  
clm.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm.Patch.x_upper.BCPressure.Cycle = 'constant'
clm.Patch.x_upper.BCPressure.alltime.Value = 0.0
#  
clm.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm.Patch.y_upper.BCPressure.Cycle = 'constant'
clm.Patch.y_upper.BCPressure.alltime.Value = 0.0
#  
clm.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
##pfset Patch.z-upper.BCPressure.Type                FluxConst 
clm.Patch.z_upper.BCPressure.Cycle = 'constant'
clm.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
#  
clm.TopoSlopesX.Type = 'Constant'
clm.TopoSlopesX.GeomNames = 'domain'
clm.TopoSlopesX.Geom.domain.Value = -0.001
#  
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
#  
clm.TopoSlopesY.Type = 'Constant'
clm.TopoSlopesY.GeomNames = 'domain'
clm.TopoSlopesY.Geom.domain.Value = 0.001
#  
#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
#  
clm.Mannings.Type = 'Constant'
clm.Mannings.GeomNames = 'domain'
clm.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm.PhaseSources.water.Type = 'Constant'
clm.PhaseSources.water.GeomNames = 'domain'
clm.PhaseSources.water.Geom.domain.Value = 0.0
#  
#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
#  
clm.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
#  
clm.Solver = 'Richards'
clm.Solver.MaxIter = 500
#  
clm.Solver.Nonlinear.MaxIter = 15
clm.Solver.Nonlinear.ResidualTol = 1e-9
clm.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm.Solver.Nonlinear.EtaValue = 0.01
clm.Solver.Nonlinear.UseJacobian = True
clm.Solver.Nonlinear.StepTol = 1e-20
clm.Solver.Nonlinear.Globalization = 'LineSearch'
clm.Solver.Linear.KrylovDimension = 15
clm.Solver.Linear.MaxRestart = 2
#  
clm.Solver.Linear.Preconditioner = 'PFMG'
clm.Solver.PrintSubsurf = False
clm.Solver.Drop = 1E-20
clm.Solver.AbsTol = 1E-9
#  
clm.Solver.LSM = 'CLM'
clm.Solver.WriteSiloCLM = True
clm.Solver.CLM.MetForcing = 1D
clm.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm.Solver.CLM.MetFilePath = ./

clm.Solver.WriteSiloEvapTrans = True
clm.Solver.WriteSiloOverlandBCFlux = True
clm.Solver.PrintCLM = True

# Initial conditions: water pressure
#---------------------------------------------------------
#  
clm.ICPressure.Type = 'HydroStaticPatch'
clm.ICPressure.GeomNames = 'domain'
clm.Geom.domain.ICPressure.Value = -2.0
#  
clm.Geom.domain.ICPressure.RefGeom = 'domain'
clm.Geom.domain.ICPressure.RefPatch = 'z_upper'



num_processors = [expr [pfget Process.Topology.P] * [pfget Process.Topology.Q] * [pfget Process.Topology.R]]
# for {set i 0} { $i <= $num_processors } {incr i} {
#     file delete drv_vegm.dat.$i
#     file copy  drv_vegm.dat drv_vegm.dat.$i
#     file delete drv_clmin.dat.$i
#     file copy drv_clmin.dat drv_clmin.dat.$i
# }

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


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

mask = [pfload clm.out.mask.pfb]
top = [Parflow::pfcomputetop $mask]

# pfsave $top -pfb "clm.out.top_index.pfb"

data = [pfload clm.out.press.00000.pfb]
top_data = [Parflow::pfextracttop $top $data]

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

clm.run()
