# this runs CLM test case

#
# Import the ParFlow TCL package
#
from parflow import Run
clm_4levels = Run("clm_4levels", __file__)

# foreach dir {qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg qflx_top_soil} {
#     file mkdir $dir
# }

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
clm_4levels.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm_4levels.Process.Topology.P = 1
clm_4levels.Process.Topology.Q = 1
clm_4levels.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm_4levels.ComputationalGrid.Lower.X = 0.0
clm_4levels.ComputationalGrid.Lower.Y = 0.0
clm_4levels.ComputationalGrid.Lower.Z = 0.0

clm_4levels.ComputationalGrid.DX = 1000.
clm_4levels.ComputationalGrid.DY = 1000.
clm_4levels.ComputationalGrid.DZ = 0.5

clm_4levels.ComputationalGrid.NX = 5
clm_4levels.ComputationalGrid.NY = 5
clm_4levels.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
clm_4levels.GeomInput.Names = 'domain_input'


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
clm_4levels.GeomInput.domain_input.InputType = 'Box'
clm_4levels.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
clm_4levels.Geom.domain.Lower.X = 0.0
clm_4levels.Geom.domain.Lower.Y = 0.0
clm_4levels.Geom.domain.Lower.Z = 0.0

clm_4levels.Geom.domain.Upper.X = 5000.
clm_4levels.Geom.domain.Upper.Y = 5000.
clm_4levels.Geom.domain.Upper.Z = 5.

clm_4levels.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clm_4levels.Geom.Perm.Names = 'domain'

clm_4levels.Geom.domain.Perm.Type = 'Constant'
clm_4levels.Geom.domain.Perm.Value = 0.2


clm_4levels.Perm.TensorType = 'TensorByGeom'

clm_4levels.Geom.Perm.TensorByGeom.Names = 'domain'

clm_4levels.Geom.domain.Perm.TensorValX = 1.0
clm_4levels.Geom.domain.Perm.TensorValY = 1.0
clm_4levels.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_4levels.SpecificStorage.Type = 'Constant'
clm_4levels.SpecificStorage.GeomNames = 'domain'
clm_4levels.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_4levels.Phase.Names = 'water'

clm_4levels.Phase.water.Density.Type = 'Constant'
clm_4levels.Phase.water.Density.Value = 1.0

clm_4levels.Phase.water.Viscosity.Type = 'Constant'
clm_4levels.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
clm_4levels.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_4levels.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
#  
clm_4levels.TimingInfo.BaseUnit = 1.0
clm_4levels.TimingInfo.StartCount = 0
clm_4levels.TimingInfo.StartTime = 0.0
clm_4levels.TimingInfo.StopTime = 5
clm_4levels.TimingInfo.DumpInterval = -1
clm_4levels.TimeStep.Type = 'Constant'
clm_4levels.TimeStep.Value = 1.0
#  

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_4levels.Geom.Porosity.GeomNames = 'domain'

clm_4levels.Geom.domain.Porosity.Type = 'Constant'
clm_4levels.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
clm_4levels.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
clm_4levels.Phase.water.Mobility.Type = 'Constant'
clm_4levels.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
#  
clm_4levels.Phase.RelPerm.Type = 'VanGenuchten'
clm_4levels.Phase.RelPerm.GeomNames = 'domain'
#  
clm_4levels.Geom.domain.RelPerm.Alpha = 3.5
clm_4levels.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_4levels.Phase.Saturation.Type = 'VanGenuchten'
clm_4levels.Phase.Saturation.GeomNames = 'domain'
#  
clm_4levels.Geom.domain.Saturation.Alpha = 3.5
clm_4levels.Geom.domain.Saturation.N = 2.
clm_4levels.Geom.domain.Saturation.SRes = 0.01
clm_4levels.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clm_4levels.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clm_4levels.Cycle.Names = 'constant'
clm_4levels.Cycle.constant.Names = 'alltime'
clm_4levels.Cycle.constant.alltime.Length = 1
clm_4levels.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clm_4levels.BCPressure.PatchNames = clm_4levels.Geom.domain.Patches
#  
clm_4levels.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_4levels.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_4levels.Patch.x_lower.BCPressure.alltime.Value = 0.0
#  
clm_4levels.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_4levels.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_4levels.Patch.y_lower.BCPressure.alltime.Value = 0.0
#  
clm_4levels.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm_4levels.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_4levels.Patch.z_lower.BCPressure.alltime.Value = 0.0
#  
clm_4levels.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_4levels.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_4levels.Patch.x_upper.BCPressure.alltime.Value = 0.0
#  
clm_4levels.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_4levels.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_4levels.Patch.y_upper.BCPressure.alltime.Value = 0.0
#  
clm_4levels.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
##pfset Patch.z-upper.BCPressure.Type                FluxConst 
clm_4levels.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_4levels.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
#  
clm_4levels.TopoSlopesX.Type = 'Constant'
clm_4levels.TopoSlopesX.GeomNames = 'domain'
clm_4levels.TopoSlopesX.Geom.domain.Value = -0.001
#  
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
#  
clm_4levels.TopoSlopesY.Type = 'Constant'
clm_4levels.TopoSlopesY.GeomNames = 'domain'
clm_4levels.TopoSlopesY.Geom.domain.Value = 0.001
#  
#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
#  
clm_4levels.Mannings.Type = 'Constant'
clm_4levels.Mannings.GeomNames = 'domain'
clm_4levels.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_4levels.PhaseSources.water.Type = 'Constant'
clm_4levels.PhaseSources.water.GeomNames = 'domain'
clm_4levels.PhaseSources.water.Geom.domain.Value = 0.0
#  
#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
#  
clm_4levels.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
#  
clm_4levels.Solver = 'Richards'
clm_4levels.Solver.MaxIter = 500
#  
clm_4levels.Solver.Nonlinear.MaxIter = 15
clm_4levels.Solver.Nonlinear.ResidualTol = 1e-9
clm_4levels.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm_4levels.Solver.Nonlinear.EtaValue = 0.01
clm_4levels.Solver.Nonlinear.UseJacobian = True
clm_4levels.Solver.Nonlinear.StepTol = 1e-20
clm_4levels.Solver.Nonlinear.Globalization = 'LineSearch'
clm_4levels.Solver.Linear.KrylovDimension = 15
clm_4levels.Solver.Linear.MaxRestart = 2
#  
clm_4levels.Solver.Linear.Preconditioner = 'PFMG'
clm_4levels.Solver.PrintSubsurf = False
clm_4levels.Solver.Drop = 1E-20
clm_4levels.Solver.AbsTol = 1E-9
#  
clm_4levels.Solver.LSM = 'CLM'
clm_4levels.Solver.WriteSiloCLM = True
clm_4levels.Solver.CLM.MetForcing = '1D'
clm_4levels.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm_4levels.Solver.CLM.MetFilePath = '../'

clm_4levels.Solver.CLM.RootZoneNZ = 4

clm_4levels.Solver.WriteSiloEvapTrans = True
clm_4levels.Solver.WriteSiloOverlandBCFlux = True
clm_4levels.Solver.PrintCLM = True

# Initial conditions: water pressure
#---------------------------------------------------------
#  
clm_4levels.ICPressure.Type = 'HydroStaticPatch'
clm_4levels.ICPressure.GeomNames = 'domain'
clm_4levels.Geom.domain.ICPressure.Value = -2.0
#  
clm_4levels.Geom.domain.ICPressure.RefGeom = 'domain'
clm_4levels.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


clm_4levels.run()
