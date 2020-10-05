#-----------------------------------------------------------------------------
# this runs CLM test case
#-----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path

#-----------------------------------------------------------------------------
# Making output directories and copying input files
#-----------------------------------------------------------------------------

clm_jac = Run("clm_jac", __file__)

dir_name = get_absolute_path('test_output/clm_jac')
mkdir(dir_name)

directories = ['qflx_evap_grnd', 'eflx_lh_tot', 'qflx_evap_tot', 'qflx_tran_veg', 'correct_output',
               'qflx_infl', 'swe_out', 'eflx_lwrad_out', 't_grnd', 'diag_out', 'qflx_evap_soi', 'eflx_soil_grnd',
               'eflx_sh_tot', 'qflx_evap_veg', 'qflx_top_soil']

for directory in directories:
    mkdir(dir_name + '/' + directory)

cp('$PF_SRC/test/tcl/clm/drv_clmin.dat', dir_name)
cp('$PF_SRC/test/tcl/clm/drv_vegm.dat', dir_name)
cp('$PF_SRC/test/tcl/clm/drv_vegp.dat', dir_name)
cp('$PF_SRC/test/tcl/clm/narr_1hr.sc3.txt.0', dir_name)

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------

clm_jac.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm_jac.Process.Topology.P = 1
clm_jac.Process.Topology.Q = 1
clm_jac.Process.Topology.R =1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------

clm_jac.ComputationalGrid.Lower.X = 0.0
clm_jac.ComputationalGrid.Lower.Y = 0.0
clm_jac.ComputationalGrid.Lower.Z = 0.0

clm_jac.ComputationalGrid.DX = 1000.
clm_jac.ComputationalGrid.DY = 1000.
clm_jac.ComputationalGrid.DZ = 0.5

clm_jac.ComputationalGrid.NX = 5
clm_jac.ComputationalGrid.NY = 5
clm_jac.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------

clm_jac.GeomInput.Names = 'domain_input'

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------

clm_jac.GeomInput.domain_input.InputType = 'Box'
clm_jac.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------

clm_jac.Geom.domain.Lower.X = 0.0
clm_jac.Geom.domain.Lower.Y = 0.0
clm_jac.Geom.domain.Lower.Z = 0.0

clm_jac.Geom.domain.Upper.X = 5000.
clm_jac.Geom.domain.Upper.Y = 5000.
clm_jac.Geom.domain.Upper.Z = 5.

clm_jac.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

clm_jac.Geom.Perm.Names = 'domain'

clm_jac.Geom.domain.Perm.Type = 'Constant'
clm_jac.Geom.domain.Perm.Value = 0.2

clm_jac.Perm.TensorType = 'TensorByGeom'

clm_jac.Geom.Perm.TensorByGeom.Names = 'domain'

clm_jac.Geom.domain.Perm.TensorValX = 1.0
clm_jac.Geom.domain.Perm.TensorValY = 1.0
clm_jac.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_jac.SpecificStorage.Type = 'Constant'
clm_jac.SpecificStorage.GeomNames = 'domain'
clm_jac.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_jac.Phase.Names = 'water'

clm_jac.Phase.water.Density.Type = 'Constant'
clm_jac.Phase.water.Density.Value = 1.0

clm_jac.Phase.water.Viscosity.Type = 'Constant'
clm_jac.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

clm_jac.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_jac.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

clm_jac.TimingInfo.BaseUnit = 1.0
clm_jac.TimingInfo.StartCount = 0
clm_jac.TimingInfo.StartTime = 0.0
clm_jac.TimingInfo.StopTime = 5
clm_jac.TimingInfo.DumpInterval = -1
clm_jac.TimeStep.Type = 'Constant'
clm_jac.TimeStep.Value = 1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_jac.Geom.Porosity.GeomNames = 'domain'
clm_jac.Geom.domain.Porosity.Type = 'Constant'
clm_jac.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

clm_jac.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------

clm_jac.Phase.water.Mobility.Type = 'Constant'
clm_jac.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

clm_jac.Phase.RelPerm.Type = 'VanGenuchten'
clm_jac.Phase.RelPerm.GeomNames = 'domain'

clm_jac.Geom.domain.RelPerm.Alpha = 3.5
clm_jac.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_jac.Phase.Saturation.Type = 'VanGenuchten'
clm_jac.Phase.Saturation.GeomNames = 'domain'

clm_jac.Geom.domain.Saturation.Alpha = 3.5
clm_jac.Geom.domain.Saturation.N = 2.
clm_jac.Geom.domain.Saturation.SRes = 0.01
clm_jac.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

clm_jac.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

clm_jac.Cycle.Names = 'constant'
clm_jac.Cycle.constant.Names = 'alltime'
clm_jac.Cycle.constant.alltime.Length = 1
clm_jac.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

clm_jac.BCPressure.PatchNames = clm_jac.Geom.domain.Patches

clm_jac.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_jac.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_jac.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_jac.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_jac.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm_jac.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_jac.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm_jac.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_jac.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_jac.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_jac.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_jac.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_jac.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_jac.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
clm_jac.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_jac.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

clm_jac.TopoSlopesX.Type = 'Constant'
clm_jac.TopoSlopesX.GeomNames = 'domain'
clm_jac.TopoSlopesX.Geom.domain.Value = -0.001

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

clm_jac.TopoSlopesY.Type = 'Constant'
clm_jac.TopoSlopesY.GeomNames = 'domain'
clm_jac.TopoSlopesY.Geom.domain.Value = 0.001

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

clm_jac.Mannings.Type = 'Constant'
clm_jac.Mannings.GeomNames = 'domain'
clm_jac.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_jac.PhaseSources.water.Type = 'Constant'
clm_jac.PhaseSources.water.GeomNames = 'domain'
clm_jac.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

clm_jac.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

clm_jac.Solver = 'Richards'
clm_jac.Solver.MaxIter = 500

clm_jac.Solver.Nonlinear.MaxIter = 15
clm_jac.Solver.Nonlinear.ResidualTol = 1e-9
clm_jac.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm_jac.Solver.Nonlinear.EtaValue = 0.01
clm_jac.Solver.Nonlinear.UseJacobian = True
clm_jac.Solver.Nonlinear.DerivativeEpsilon = 1e-12
clm_jac.Solver.Nonlinear.StepTol = 1e-20
clm_jac.Solver.Nonlinear.Globalization = 'LineSearch'
clm_jac.Solver.Linear.KrylovDimension = 15
clm_jac.Solver.Linear.MaxRestart = 2

clm_jac.Solver.Linear.Preconditioner = 'PFMG'
clm_jac.Solver.PrintSubsurf = False
clm_jac.Solver.Drop = 1E-20
clm_jac.Solver.AbsTol = 1E-9

clm_jac.Solver.LSM = 'CLM'
clm_jac.Solver.WriteSiloCLM = True
clm_jac.Solver.CLM.MetForcing = '1D'
clm_jac.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm_jac.Solver.CLM.MetFilePath = '.'

clm_jac.Solver.WriteSiloEvapTrans = True
clm_jac.Solver.WriteSiloOverlandBCFlux = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

clm_jac.ICPressure.Type = 'HydroStaticPatch'
clm_jac.ICPressure.GeomNames = 'domain'
clm_jac.Geom.domain.ICPressure.Value = -2.0

clm_jac.Geom.domain.ICPressure.RefGeom = 'domain'
clm_jac.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

clm_jac.run(working_directory=dir_name)
