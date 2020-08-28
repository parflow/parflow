# this runs CLM_slope test case

#
# Import the ParFlow TCL package
#
from parflow import Run
from parflow.tools.fs import mkdir, cp

clm_slope = Run("clm_slope", __file__)

directories = ['qflx_evap_grnd', 'eflx_lh_tot', 'qflx_evap_tot', 'qflx_tran_veg', 'correct_output',
               'qflx_infl', 'swe_out', 'eflx_lwrad_out', 't_grnd', 'diag_out', 'qflx_evap_soi', 'eflx_soil_grnd',
               'eflx_sh_tot', 'qflx_evap_veg', 'qflx_top_soil']

for directory in directories:
    mkdir(directory)


cp('$PF_SRC/test/clm/drv_clmin.dat')
cp('$PF_SRC/test/clm/drv_vegm.dat')
cp('$PF_SRC/test/clm/drv_vegp.dat')
cp('$PF_SRC/test/clm/narr_1hr.sc3.txt.0')

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
clm_slope.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm_slope.Process.Topology.P = 1
clm_slope.Process.Topology.Q = 1
clm_slope.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm_slope.ComputationalGrid.Lower.X = 0.0
clm_slope.ComputationalGrid.Lower.Y = 0.0
clm_slope.ComputationalGrid.Lower.Z = 0.0

clm_slope.ComputationalGrid.DX = 1000.
clm_slope.ComputationalGrid.DY = 1000.
clm_slope.ComputationalGrid.DZ = 0.5

clm_slope.ComputationalGrid.NX = 5
clm_slope.ComputationalGrid.NY = 5
clm_slope.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
clm_slope.GeomInput.Names = 'domain_input center_input north_input south_input \
    east_input west_input northeast_input southeast_input southwest_input northwest_input'


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------

clm_slope.GeomInput.domain_input.GeomName = 'domain'
clm_slope.GeomInput.center_input.GeomName = 'center'
clm_slope.GeomInput.north_input.GeomName = 'north'
clm_slope.GeomInput.south_input.GeomName = 'south'
clm_slope.GeomInput.east_input.GeomName = 'east'
clm_slope.GeomInput.west_input.GeomName = 'west'
clm_slope.GeomInput.northeast_input.GeomName = 'northeast'
clm_slope.GeomInput.southeast_input.GeomName = 'southeast'
clm_slope.GeomInput.southwest_input.GeomName = 'southwest'
clm_slope.GeomInput.northwest_input.GeomName = 'northwest'

clm_slope.GeomInput.domain_input.InputType = 'Box'
clm_slope.GeomInput.center_input.InputType = 'Box'
clm_slope.GeomInput.north_input.InputType = 'Box'
clm_slope.GeomInput.south_input.InputType = 'Box'
clm_slope.GeomInput.east_input.InputType = 'Box'
clm_slope.GeomInput.west_input.InputType = 'Box'
clm_slope.GeomInput.northeast_input.InputType = 'Box'
clm_slope.GeomInput.southeast_input.InputType = 'Box'
clm_slope.GeomInput.southwest_input.InputType = 'Box'
clm_slope.GeomInput.northwest_input.InputType = 'Box'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
clm_slope.Geom.domain.Lower.X = 0.0
clm_slope.Geom.domain.Lower.Y = 0.0
clm_slope.Geom.domain.Lower.Z = 0.0

clm_slope.Geom.domain.Upper.X = 5000.
clm_slope.Geom.domain.Upper.Y = 5000.
clm_slope.Geom.domain.Upper.Z = 5.

clm_slope.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

# ---------------------------------------------------------
# Center tile Geometry
# ---------------------------------------------------------
clm_slope.Geom.center.Lower.X = 2000
clm_slope.Geom.center.Lower.Y = 2000
clm_slope.Geom.center.Lower.Z = 0.0

clm_slope.Geom.center.Upper.X = 3000
clm_slope.Geom.center.Upper.Y = 3000
clm_slope.Geom.center.Upper.Z = 5.0
# was 0.05
#
#
# ---------------------------------------------------------
# North Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.north.Lower.X = 2000
clm_slope.Geom.north.Lower.Y = 3000
clm_slope.Geom.north.Lower.Z = 0.0

clm_slope.Geom.north.Upper.X = 3000
clm_slope.Geom.north.Upper.Y = 5000
clm_slope.Geom.north.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# South Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.south.Lower.X = 2000
clm_slope.Geom.south.Lower.Y = 0.0
clm_slope.Geom.south.Lower.Z = 0.0

clm_slope.Geom.south.Upper.X = 3000
clm_slope.Geom.south.Upper.Y = 2000
clm_slope.Geom.south.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# East Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.east.Lower.X = 3000
clm_slope.Geom.east.Lower.Y = 2000
clm_slope.Geom.east.Lower.Z = 0.0

clm_slope.Geom.east.Upper.X = 5000
clm_slope.Geom.east.Upper.Y = 3000
clm_slope.Geom.east.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# West Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.west.Lower.X = 0.0
clm_slope.Geom.west.Lower.Y = 2000
clm_slope.Geom.west.Lower.Z = 0.0

clm_slope.Geom.west.Upper.X = 2000
clm_slope.Geom.west.Upper.Y = 3000
clm_slope.Geom.west.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# Northeast Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.northeast.Lower.X = 3000
clm_slope.Geom.northeast.Lower.Y = 3000
clm_slope.Geom.northeast.Lower.Z = 0.0

clm_slope.Geom.northeast.Upper.X = 5000
clm_slope.Geom.northeast.Upper.Y = 5000
clm_slope.Geom.northeast.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# Southeast Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.southeast.Lower.X = 3000
clm_slope.Geom.southeast.Lower.Y = 0.0
clm_slope.Geom.southeast.Lower.Z = 0.0

clm_slope.Geom.southeast.Upper.X = 5000
clm_slope.Geom.southeast.Upper.Y = 2000
clm_slope.Geom.southeast.Upper.Z = 5.0
#was 0.05
#
# ---------------------------------------------------------
# Southwest Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.southwest.Lower.X = 0.0
clm_slope.Geom.southwest.Lower.Y = 0.0
clm_slope.Geom.southwest.Lower.Z = 0.0

clm_slope.Geom.southwest.Upper.X = 2000
clm_slope.Geom.southwest.Upper.Y = 2000
clm_slope.Geom.southwest.Upper.Z = 5.0
# was 0.05
#
# ---------------------------------------------------------
# Northwest Slope Geometry
# ---------------------------------------------------------
clm_slope.Geom.northwest.Lower.X = 0.0
clm_slope.Geom.northwest.Lower.Y = 3000
clm_slope.Geom.northwest.Lower.Z = 0.0

clm_slope.Geom.northwest.Upper.X = 2000
clm_slope.Geom.northwest.Upper.Y = 5000
clm_slope.Geom.northwest.Upper.Z = 5.0
# was 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clm_slope.Geom.Perm.Names = 'domain'

clm_slope.Geom.domain.Perm.Type = 'Constant'
clm_slope.Geom.domain.Perm.Value = 0.2


clm_slope.Perm.TensorType = 'TensorByGeom'

clm_slope.Geom.Perm.TensorByGeom.Names = 'domain'

clm_slope.Geom.domain.Perm.TensorValX = 1.0
clm_slope.Geom.domain.Perm.TensorValY = 1.0
clm_slope.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_slope.SpecificStorage.Type = 'Constant'
clm_slope.SpecificStorage.GeomNames = 'domain'
clm_slope.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_slope.Phase.Names = 'water'

clm_slope.Phase.water.Density.Type = 'Constant'
clm_slope.Phase.water.Density.Value = 1.0

clm_slope.Phase.water.Viscosity.Type = 'Constant'
clm_slope.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
clm_slope.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_slope.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

clm_slope.TimingInfo.BaseUnit = 1.0
clm_slope.TimingInfo.StartCount = 0
clm_slope.TimingInfo.StartTime = 0.0
clm_slope.TimingInfo.StopTime = 5
clm_slope.TimingInfo.DumpInterval = -1
clm_slope.TimeStep.Type = 'Constant'
clm_slope.TimeStep.Value = 1.0


#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_slope.Geom.Porosity.GeomNames = 'domain'

clm_slope.Geom.domain.Porosity.Type = 'Constant'
clm_slope.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
clm_slope.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
clm_slope.Phase.water.Mobility.Type = 'Constant'
clm_slope.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

clm_slope.Phase.RelPerm.Type = 'VanGenuchten'
clm_slope.Phase.RelPerm.GeomNames = 'domain'

clm_slope.Geom.domain.RelPerm.Alpha = 3.5
clm_slope.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_slope.Phase.Saturation.Type = 'VanGenuchten'
clm_slope.Phase.Saturation.GeomNames = 'domain'

clm_slope.Geom.domain.Saturation.Alpha = 3.5
clm_slope.Geom.domain.Saturation.N = 2.
clm_slope.Geom.domain.Saturation.SRes = 0.01
clm_slope.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clm_slope.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clm_slope.Cycle.Names = 'constant'
clm_slope.Cycle.constant.Names = 'alltime'
clm_slope.Cycle.constant.alltime.Length = 1
clm_slope.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clm_slope.BCPressure.PatchNames = clm_slope.Geom.domain.Patches

clm_slope.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_slope.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_slope.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_slope.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_slope.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_slope.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_slope.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm_slope.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_slope.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm_slope.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_slope.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_slope.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_slope.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_slope.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_slope.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_slope.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
##pfset Patch.z-upper.BCPressure.Type                FluxConst
clm_slope.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_slope.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

clm_slope.TopoSlopesX.Type = 'Constant'
clm_slope.TopoSlopesX.GeomNames = 'center north south east west northeast southeast southwest northwest'
clm_slope.TopoSlopesX.Geom.center.Value = 0.0
clm_slope.TopoSlopesX.Geom.north.Value = 0.0
clm_slope.TopoSlopesX.Geom.south.Value = 0.0
clm_slope.TopoSlopesX.Geom.west.Value = 10.0
clm_slope.TopoSlopesX.Geom.east.Value = -10.0
clm_slope.TopoSlopesX.Geom.northeast.Value = -10.0
clm_slope.TopoSlopesX.Geom.southeast.Value = -10.0
clm_slope.TopoSlopesX.Geom.southwest.Value = 10.0
clm_slope.TopoSlopesX.Geom.northwest.Value = 10.0


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

clm_slope.TopoSlopesY.Type = 'Constant'
clm_slope.TopoSlopesY.GeomNames = 'center north south east west northeast southeast southwest northwest'
clm_slope.TopoSlopesY.Geom.center.Value = 0.0
clm_slope.TopoSlopesY.Geom.north.Value = -10.0
clm_slope.TopoSlopesY.Geom.south.Value = 10.0
clm_slope.TopoSlopesY.Geom.west.Value = 0.0
clm_slope.TopoSlopesY.Geom.east.Value = 0.0
clm_slope.TopoSlopesY.Geom.northeast.Value = -10.0
clm_slope.TopoSlopesY.Geom.southeast.Value = 10.0
clm_slope.TopoSlopesY.Geom.southwest.Value = 10.0
clm_slope.TopoSlopesY.Geom.northwest.Value = -10.0


#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

clm_slope.Mannings.Type = 'Constant'
clm_slope.Mannings.GeomNames = 'domain'
clm_slope.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_slope.PhaseSources.water.Type = 'Constant'
clm_slope.PhaseSources.water.GeomNames = 'domain'
clm_slope.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

clm_slope.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

clm_slope.Solver = 'Richards'
clm_slope.Solver.MaxIter = 500

clm_slope.Solver.Nonlinear.MaxIter = 15
clm_slope.Solver.Nonlinear.ResidualTol = 1e-9
clm_slope.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm_slope.Solver.Nonlinear.EtaValue = 0.01
clm_slope.Solver.Nonlinear.UseJacobian = True
clm_slope.Solver.Nonlinear.StepTol = 1e-20
clm_slope.Solver.Nonlinear.Globalization = 'LineSearch'
clm_slope.Solver.Linear.KrylovDimension = 15
clm_slope.Solver.Linear.MaxRestart = 2

clm_slope.Solver.Linear.Preconditioner = 'PFMG'
clm_slope.Solver.PrintSubsurf = False
clm_slope.Solver.Drop = 1E-20
clm_slope.Solver.AbsTol = 1E-9

clm_slope.Solver.LSM = 'CLM'
clm_slope.Solver.WriteSiloCLM = False
clm_slope.Solver.CLM.MetForcing = '1D'
clm_slope.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm_slope.Solver.CLM.MetFilePath = '.'

clm_slope.Solver.WriteSiloEvapTrans = False
clm_slope.Solver.WriteSiloOverlandBCFlux = False
clm_slope.Solver.PrintCLM = False
clm_slope.Solver.SlopeAccountingCLM = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

clm_slope.ICPressure.Type = 'HydroStaticPatch'
clm_slope.ICPressure.GeomNames = 'domain'
clm_slope.Geom.domain.ICPressure.Value = -2.0

clm_slope.Geom.domain.ICPressure.RefGeom = 'domain'
clm_slope.Geom.domain.ICPressure.RefPatch = 'z_upper'


#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

clm_slope.run()
