# -----------------------------------------------------------------------------
# this runs CLM_slope test case
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path

clm = Run("clm_slope", __file__)

# -----------------------------------------------------------------------------
# Making output directories and copying input files
# -----------------------------------------------------------------------------

dir_name = get_absolute_path("test_output/clm_slopes")
mkdir(dir_name)

directories = [
    "qflx_evap_grnd",
    "eflx_lh_tot",
    "qflx_evap_tot",
    "qflx_tran_veg",
    "correct_output",
    "qflx_infl",
    "swe_out",
    "eflx_lwrad_out",
    "t_grnd",
    "diag_out",
    "qflx_evap_soi",
    "eflx_soil_grnd",
    "eflx_sh_tot",
    "qflx_evap_veg",
    "qflx_top_soil",
]

for directory in directories:
    mkdir(dir_name + "/" + directory)

cp("$PF_SRC/test/tcl/clm/drv_clmin.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegm.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/drv_vegp.dat", dir_name)
cp("$PF_SRC/test/tcl/clm/narr_1hr.sc3.txt.0", dir_name)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------

clm.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

clm.Process.Topology.P = 1
clm.Process.Topology.Q = 1
clm.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 1000.0
clm.ComputationalGrid.DY = 1000.0
clm.ComputationalGrid.DZ = 0.5

clm.ComputationalGrid.NX = 5
clm.ComputationalGrid.NY = 5
clm.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------

clm.GeomInput.Names = "domain_input center_input north_input south_input \
    east_input west_input northeast_input southeast_input southwest_input northwest_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

clm.GeomInput.domain_input.GeomName = "domain"
clm.GeomInput.center_input.GeomName = "center"
clm.GeomInput.north_input.GeomName = "north"
clm.GeomInput.south_input.GeomName = "south"
clm.GeomInput.east_input.GeomName = "east"
clm.GeomInput.west_input.GeomName = "west"
clm.GeomInput.northeast_input.GeomName = "northeast"
clm.GeomInput.southeast_input.GeomName = "southeast"
clm.GeomInput.southwest_input.GeomName = "southwest"
clm.GeomInput.northwest_input.GeomName = "northwest"

clm.GeomInput.domain_input.InputType = "Box"
clm.GeomInput.center_input.InputType = "Box"
clm.GeomInput.north_input.InputType = "Box"
clm.GeomInput.south_input.InputType = "Box"
clm.GeomInput.east_input.InputType = "Box"
clm.GeomInput.west_input.InputType = "Box"
clm.GeomInput.northeast_input.InputType = "Box"
clm.GeomInput.southeast_input.InputType = "Box"
clm.GeomInput.southwest_input.InputType = "Box"
clm.GeomInput.northwest_input.InputType = "Box"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

clm.Geom.domain.Lower.X = 0.0
clm.Geom.domain.Lower.Y = 0.0
clm.Geom.domain.Lower.Z = 0.0

clm.Geom.domain.Upper.X = 5000.0
clm.Geom.domain.Upper.Y = 5000.0
clm.Geom.domain.Upper.Z = 5.0

clm.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# ---------------------------------------------------------
# Center tile Geometry
# ---------------------------------------------------------

clm.Geom.center.Lower.X = 2000
clm.Geom.center.Lower.Y = 2000
clm.Geom.center.Lower.Z = 0.0

clm.Geom.center.Upper.X = 3000
clm.Geom.center.Upper.Y = 3000
clm.Geom.center.Upper.Z = 5.0

# ---------------------------------------------------------
# North Slope Geometry
# ---------------------------------------------------------

clm.Geom.north.Lower.X = 2000
clm.Geom.north.Lower.Y = 3000
clm.Geom.north.Lower.Z = 0.0

clm.Geom.north.Upper.X = 3000
clm.Geom.north.Upper.Y = 5000
clm.Geom.north.Upper.Z = 5.0

# ---------------------------------------------------------
# South Slope Geometry
# ---------------------------------------------------------

clm.Geom.south.Lower.X = 2000
clm.Geom.south.Lower.Y = 0.0
clm.Geom.south.Lower.Z = 0.0

clm.Geom.south.Upper.X = 3000
clm.Geom.south.Upper.Y = 2000
clm.Geom.south.Upper.Z = 5.0

# ---------------------------------------------------------
# East Slope Geometry
# ---------------------------------------------------------

clm.Geom.east.Lower.X = 3000
clm.Geom.east.Lower.Y = 2000
clm.Geom.east.Lower.Z = 0.0

clm.Geom.east.Upper.X = 5000
clm.Geom.east.Upper.Y = 3000
clm.Geom.east.Upper.Z = 5.0

# ---------------------------------------------------------
# West Slope Geometry
# ---------------------------------------------------------

clm.Geom.west.Lower.X = 0.0
clm.Geom.west.Lower.Y = 2000
clm.Geom.west.Lower.Z = 0.0

clm.Geom.west.Upper.X = 2000
clm.Geom.west.Upper.Y = 3000
clm.Geom.west.Upper.Z = 5.0

# ---------------------------------------------------------
# Northeast Slope Geometry
# ---------------------------------------------------------

clm.Geom.northeast.Lower.X = 3000
clm.Geom.northeast.Lower.Y = 3000
clm.Geom.northeast.Lower.Z = 0.0

clm.Geom.northeast.Upper.X = 5000
clm.Geom.northeast.Upper.Y = 5000
clm.Geom.northeast.Upper.Z = 5.0

# ---------------------------------------------------------
# Southeast Slope Geometry
# ---------------------------------------------------------

clm.Geom.southeast.Lower.X = 3000
clm.Geom.southeast.Lower.Y = 0.0
clm.Geom.southeast.Lower.Z = 0.0

clm.Geom.southeast.Upper.X = 5000
clm.Geom.southeast.Upper.Y = 2000
clm.Geom.southeast.Upper.Z = 5.0

# ---------------------------------------------------------
# Southwest Slope Geometry
# ---------------------------------------------------------

clm.Geom.southwest.Lower.X = 0.0
clm.Geom.southwest.Lower.Y = 0.0
clm.Geom.southwest.Lower.Z = 0.0

clm.Geom.southwest.Upper.X = 2000
clm.Geom.southwest.Upper.Y = 2000
clm.Geom.southwest.Upper.Z = 5.0

# ---------------------------------------------------------
# Northwest Slope Geometry
# ---------------------------------------------------------

clm.Geom.northwest.Lower.X = 0.0
clm.Geom.northwest.Lower.Y = 3000
clm.Geom.northwest.Lower.Z = 0.0

clm.Geom.northwest.Upper.X = 2000
clm.Geom.northwest.Upper.Y = 5000
clm.Geom.northwest.Upper.Z = 5.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

clm.Geom.Perm.Names = "domain"

clm.Geom.domain.Perm.Type = "Constant"
clm.Geom.domain.Perm.Value = 0.2

clm.Perm.TensorType = "TensorByGeom"

clm.Geom.Perm.TensorByGeom.Names = "domain"

clm.Geom.domain.Perm.TensorValX = 1.0
clm.Geom.domain.Perm.TensorValY = 1.0
clm.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm.SpecificStorage.Type = "Constant"
clm.SpecificStorage.GeomNames = "domain"
clm.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm.Phase.Names = "water"

clm.Phase.water.Density.Type = "Constant"
clm.Phase.water.Density.Value = 1.0

clm.Phase.water.Viscosity.Type = "Constant"
clm.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

clm.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

clm.TimingInfo.BaseUnit = 1.0
clm.TimingInfo.StartCount = 0
clm.TimingInfo.StartTime = 0.0
clm.TimingInfo.StopTime = 5
clm.TimingInfo.DumpInterval = -1
clm.TimeStep.Type = "Constant"
clm.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm.Geom.Porosity.GeomNames = "domain"

clm.Geom.domain.Porosity.Type = "Constant"
clm.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

clm.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
clm.Phase.water.Mobility.Type = "Constant"
clm.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

clm.Phase.RelPerm.Type = "VanGenuchten"
clm.Phase.RelPerm.GeomNames = "domain"

clm.Geom.domain.RelPerm.Alpha = 3.5
clm.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm.Phase.Saturation.Type = "VanGenuchten"
clm.Phase.Saturation.GeomNames = "domain"

clm.Geom.domain.Saturation.Alpha = 3.5
clm.Geom.domain.Saturation.N = 2.0
clm.Geom.domain.Saturation.SRes = 0.01
clm.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

clm.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

clm.Cycle.Names = "constant"
clm.Cycle.constant.Names = "alltime"
clm.Cycle.constant.alltime.Length = 1
clm.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

clm.BCPressure.PatchNames = clm.Geom.domain.Patches

clm.Patch.x_lower.BCPressure.Type = "FluxConst"
clm.Patch.x_lower.BCPressure.Cycle = "constant"
clm.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm.Patch.y_lower.BCPressure.Type = "FluxConst"
clm.Patch.y_lower.BCPressure.Cycle = "constant"
clm.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm.Patch.z_lower.BCPressure.Type = "FluxConst"
clm.Patch.z_lower.BCPressure.Cycle = "constant"
clm.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm.Patch.x_upper.BCPressure.Type = "FluxConst"
clm.Patch.x_upper.BCPressure.Cycle = "constant"
clm.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm.Patch.y_upper.BCPressure.Type = "FluxConst"
clm.Patch.y_upper.BCPressure.Cycle = "constant"
clm.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm.Patch.z_upper.BCPressure.Cycle = "constant"
clm.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

clm.TopoSlopesX.Type = "Constant"
clm.TopoSlopesX.GeomNames = (
    "center north south east west northeast southeast southwest northwest"
)
clm.TopoSlopesX.Geom.center.Value = 0.0
clm.TopoSlopesX.Geom.north.Value = 0.0
clm.TopoSlopesX.Geom.south.Value = 0.0
clm.TopoSlopesX.Geom.west.Value = 10.0
clm.TopoSlopesX.Geom.east.Value = -10.0
clm.TopoSlopesX.Geom.northeast.Value = -10.0
clm.TopoSlopesX.Geom.southeast.Value = -10.0
clm.TopoSlopesX.Geom.southwest.Value = 10.0
clm.TopoSlopesX.Geom.northwest.Value = 10.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

clm.TopoSlopesY.Type = "Constant"
clm.TopoSlopesY.GeomNames = (
    "center north south east west northeast southeast southwest northwest"
)
clm.TopoSlopesY.Geom.center.Value = 0.0
clm.TopoSlopesY.Geom.north.Value = -10.0
clm.TopoSlopesY.Geom.south.Value = 10.0
clm.TopoSlopesY.Geom.west.Value = 0.0
clm.TopoSlopesY.Geom.east.Value = 0.0
clm.TopoSlopesY.Geom.northeast.Value = -10.0
clm.TopoSlopesY.Geom.southeast.Value = 10.0
clm.TopoSlopesY.Geom.southwest.Value = 10.0
clm.TopoSlopesY.Geom.northwest.Value = -10.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

clm.Mannings.Type = "Constant"
clm.Mannings.GeomNames = "domain"
clm.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm.PhaseSources.water.Type = "Constant"
clm.PhaseSources.water.GeomNames = "domain"
clm.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

clm.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

clm.Solver = "Richards"
clm.Solver.MaxIter = 500

clm.Solver.Nonlinear.MaxIter = 15
clm.Solver.Nonlinear.ResidualTol = 1e-9
clm.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm.Solver.Nonlinear.EtaValue = 0.01
clm.Solver.Nonlinear.UseJacobian = True
clm.Solver.Nonlinear.StepTol = 1e-20
clm.Solver.Nonlinear.Globalization = "LineSearch"
clm.Solver.Linear.KrylovDimension = 15
clm.Solver.Linear.MaxRestart = 2

clm.Solver.Linear.Preconditioner = "PFMG"
clm.Solver.PrintSubsurf = False
clm.Solver.Drop = 1e-20
clm.Solver.AbsTol = 1e-9

clm.Solver.LSM = "CLM"
clm.Solver.WriteSiloCLM = False
clm.Solver.CLM.MetForcing = "1D"
clm.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm.Solver.CLM.MetFilePath = "."

clm.Solver.WriteSiloEvapTrans = False
clm.Solver.WriteSiloOverlandBCFlux = False
clm.Solver.PrintCLM = False
clm.Solver.SlopeAccountingCLM = True

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

clm.ICPressure.Type = "HydroStaticPatch"
clm.ICPressure.GeomNames = "domain"
clm.Geom.domain.ICPressure.Value = -2.0

clm.Geom.domain.ICPressure.RefGeom = "domain"
clm.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

clm.run(working_directory=dir_name)
