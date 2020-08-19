# BH: this runs CLM with variable DZ test case
# It has been observed that root fraction distribution in CLM was based on constant DZ,
# regardless of whether variable DZ was an input in PF. This is fixed in this version.
# however the layer thicknesses and layer interfaces do not match between CLM and ParFlow,
# only center of cells location match. As a consequence, the volume of the column is not
# conserved. This has been correctd in this version, and the root fraction distribution
# is calculated based on layer interfaces which match ParFlow layer interfaces.
#
# Import the ParFlow TCL package
#
from parflow import Run
clm_varDZ = Run("clm_varDZ", __file__)

# foreach dir {qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg qflx_top_soil} {
#     file mkdir $dir
# }


#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
clm_varDZ.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm_varDZ.Process.Topology.P = 1
clm_varDZ.Process.Topology.Q = 1
clm_varDZ.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm_varDZ.ComputationalGrid.Lower.X = 0.0
clm_varDZ.ComputationalGrid.Lower.Y = 0.0
clm_varDZ.ComputationalGrid.Lower.Z = 0.0

clm_varDZ.ComputationalGrid.DX = 1000.
clm_varDZ.ComputationalGrid.DY = 1000.
clm_varDZ.ComputationalGrid.DZ = 0.5

clm_varDZ.ComputationalGrid.NX = 5
clm_varDZ.ComputationalGrid.NY = 5
clm_varDZ.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
clm_varDZ.GeomInput.Names = 'domain_input'


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
clm_varDZ.GeomInput.domain_input.InputType = 'Box'
clm_varDZ.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
clm_varDZ.Geom.domain.Lower.X = 0.0
clm_varDZ.Geom.domain.Lower.Y = 0.0
clm_varDZ.Geom.domain.Lower.Z = 0.0

clm_varDZ.Geom.domain.Upper.X = 5000.
clm_varDZ.Geom.domain.Upper.Y = 5000.
clm_varDZ.Geom.domain.Upper.Z = 5.

clm_varDZ.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#--------------------------------------------
# variable dz assignments
#------------------------------------------
clm_varDZ.Solver.Nonlinear.VariableDz = True
clm_varDZ.dzScale.GeomNames = 'domain'
clm_varDZ.dzScale.Type = 'nzList'
clm_varDZ.dzScale.nzListNumber = 10

clm_varDZ.Cell._0.dzScale.Value = 2.5
clm_varDZ.Cell._1.dzScale.Value = 2
clm_varDZ.Cell._2.dzScale.Value = 1.5
clm_varDZ.Cell._3.dzScale.Value = 1.25
clm_varDZ.Cell._4.dzScale.Value = 1.
clm_varDZ.Cell._5.dzScale.Value = 0.75
clm_varDZ.Cell._6.dzScale.Value = 0.5
clm_varDZ.Cell._7.dzScale.Value = 0.25
clm_varDZ.Cell._8.dzScale.Value = 0.125
clm_varDZ.Cell._9.dzScale.Value = 0.125

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clm_varDZ.Geom.Perm.Names = 'domain'

clm_varDZ.Geom.domain.Perm.Type = 'Constant'
clm_varDZ.Geom.domain.Perm.Value = 0.2


clm_varDZ.Perm.TensorType = 'TensorByGeom'

clm_varDZ.Geom.Perm.TensorByGeom.Names = 'domain'

clm_varDZ.Geom.domain.Perm.TensorValX = 1.0
clm_varDZ.Geom.domain.Perm.TensorValY = 1.0
clm_varDZ.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_varDZ.SpecificStorage.Type = 'Constant'
clm_varDZ.SpecificStorage.GeomNames = 'domain'
clm_varDZ.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_varDZ.Phase.Names = 'water'

clm_varDZ.Phase.water.Density.Type = 'Constant'
clm_varDZ.Phase.water.Density.Value = 1.0

clm_varDZ.Phase.water.Viscosity.Type = 'Constant'
clm_varDZ.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
clm_varDZ.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_varDZ.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
#  
clm_varDZ.TimingInfo.BaseUnit = 1.0
clm_varDZ.TimingInfo.StartCount = 0
clm_varDZ.TimingInfo.StartTime = 0.0
clm_varDZ.TimingInfo.StopTime = 5
clm_varDZ.TimingInfo.DumpInterval = -1
clm_varDZ.TimeStep.Type = 'Constant'
clm_varDZ.TimeStep.Value = 1.0
#  

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_varDZ.Geom.Porosity.GeomNames = 'domain'

clm_varDZ.Geom.domain.Porosity.Type = 'Constant'
clm_varDZ.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
clm_varDZ.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
clm_varDZ.Phase.water.Mobility.Type = 'Constant'
clm_varDZ.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
#  
clm_varDZ.Phase.RelPerm.Type = 'VanGenuchten'
clm_varDZ.Phase.RelPerm.GeomNames = 'domain'
#  
clm_varDZ.Geom.domain.RelPerm.Alpha = 3.5
clm_varDZ.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_varDZ.Phase.Saturation.Type = 'VanGenuchten'
clm_varDZ.Phase.Saturation.GeomNames = 'domain'
#  
clm_varDZ.Geom.domain.Saturation.Alpha = 3.5
clm_varDZ.Geom.domain.Saturation.N = 2.
clm_varDZ.Geom.domain.Saturation.SRes = 0.01
clm_varDZ.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clm_varDZ.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clm_varDZ.Cycle.Names = 'constant'
clm_varDZ.Cycle.constant.Names = 'alltime'
clm_varDZ.Cycle.constant.alltime.Length = 1
clm_varDZ.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clm_varDZ.BCPressure.PatchNames = clm_varDZ.Geom.domain.Patches
#  
clm_varDZ.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_varDZ.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.x_lower.BCPressure.alltime.Value = 0.0
#  
clm_varDZ.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_varDZ.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.y_lower.BCPressure.alltime.Value = 0.0
#  
clm_varDZ.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm_varDZ.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.z_lower.BCPressure.alltime.Value = 0.0
#  
clm_varDZ.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_varDZ.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.x_upper.BCPressure.alltime.Value = 0.0
#  
clm_varDZ.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_varDZ.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.y_upper.BCPressure.alltime.Value = 0.0
#  
clm_varDZ.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
##pfset Patch.z-upper.BCPressure.Type                FluxConst 
clm_varDZ.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_varDZ.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
#  
clm_varDZ.TopoSlopesX.Type = 'Constant'
clm_varDZ.TopoSlopesX.GeomNames = 'domain'
clm_varDZ.TopoSlopesX.Geom.domain.Value = -0.001
#  
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
#  
clm_varDZ.TopoSlopesY.Type = 'Constant'
clm_varDZ.TopoSlopesY.GeomNames = 'domain'
clm_varDZ.TopoSlopesY.Geom.domain.Value = 0.001
#  
#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
#  
clm_varDZ.Mannings.Type = 'Constant'
clm_varDZ.Mannings.GeomNames = 'domain'
clm_varDZ.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_varDZ.PhaseSources.water.Type = 'Constant'
clm_varDZ.PhaseSources.water.GeomNames = 'domain'
clm_varDZ.PhaseSources.water.Geom.domain.Value = 0.0
#  
#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
#  
clm_varDZ.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
#  
clm_varDZ.Solver = 'Richards'
clm_varDZ.Solver.MaxIter = 500
#  
clm_varDZ.Solver.Nonlinear.MaxIter = 15
clm_varDZ.Solver.Nonlinear.ResidualTol = 1e-9
clm_varDZ.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm_varDZ.Solver.Nonlinear.EtaValue = 0.01
clm_varDZ.Solver.Nonlinear.UseJacobian = True
clm_varDZ.Solver.Nonlinear.StepTol = 1e-20
clm_varDZ.Solver.Nonlinear.Globalization = 'LineSearch'
clm_varDZ.Solver.Linear.KrylovDimension = 15
clm_varDZ.Solver.Linear.MaxRestart = 2
#  
clm_varDZ.Solver.Linear.Preconditioner = 'PFMG'
clm_varDZ.Solver.PrintSubsurf = False
clm_varDZ.Solver.Drop = 1E-20
clm_varDZ.Solver.AbsTol = 1E-9
#  
clm_varDZ.Solver.LSM = 'CLM'
clm_varDZ.Solver.WriteSiloCLM = True
clm_varDZ.Solver.CLM.MetForcing = '1D'
clm_varDZ.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm_varDZ.Solver.CLM.MetFilePath = '../'
clm_varDZ.Solver.CLM.ForceVegetation = False

clm_varDZ.Solver.WriteSiloEvapTrans = True
clm_varDZ.Solver.WriteSiloOverlandBCFlux = True
clm_varDZ.Solver.PrintCLM = True

# Initial conditions: water pressure
#---------------------------------------------------------
#  
clm_varDZ.ICPressure.Type = 'HydroStaticPatch'
clm_varDZ.ICPressure.GeomNames = 'domain'
clm_varDZ.Geom.domain.ICPressure.Value = -2.0
#  
clm_varDZ.Geom.domain.ICPressure.RefGeom = 'domain'
clm_varDZ.Geom.domain.ICPressure.RefPatch = 'z_upper'


#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


clm_varDZ.run()
