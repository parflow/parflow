# -----------------------------------------------------------------------------
# this runs CLM test case with vtk (not included in the ParFlow test suite)
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path

clm_vtk = Run("clm_vtk", __file__)

# -----------------------------------------------------------------------------
# Making output directories and copying input files
# -----------------------------------------------------------------------------

dir_name = get_absolute_path("test_output/clm_vtk")
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

clm_vtk.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

clm_vtk.Process.Topology.P = 1
clm_vtk.Process.Topology.Q = 1
clm_vtk.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

clm_vtk.ComputationalGrid.Lower.X = 0.0
clm_vtk.ComputationalGrid.Lower.Y = 0.0
clm_vtk.ComputationalGrid.Lower.Z = 0.0

clm_vtk.ComputationalGrid.DX = 1000.0
clm_vtk.ComputationalGrid.DY = 1000.0
clm_vtk.ComputationalGrid.DZ = 0.5

clm_vtk.ComputationalGrid.NX = 5
clm_vtk.ComputationalGrid.NY = 5
clm_vtk.ComputationalGrid.NZ = 10

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------

clm_vtk.GeomInput.Names = "domain_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

clm_vtk.GeomInput.domain_input.InputType = "Box"
clm_vtk.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

clm_vtk.Geom.domain.Lower.X = 0.0
clm_vtk.Geom.domain.Lower.Y = 0.0
clm_vtk.Geom.domain.Lower.Z = 0.0

clm_vtk.Geom.domain.Upper.X = 5000.0
clm_vtk.Geom.domain.Upper.Y = 5000.0
clm_vtk.Geom.domain.Upper.Z = 5.0

clm_vtk.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

clm_vtk.Geom.Perm.Names = "domain"

clm_vtk.Geom.domain.Perm.Type = "TurnBands"
clm_vtk.Geom.domain.Perm.LambdaX = 3000.0
clm_vtk.Geom.domain.Perm.LambdaY = 2000.0
clm_vtk.Geom.domain.Perm.LambdaZ = 5.0
clm_vtk.Geom.domain.Perm.GeomMean = 0.2
clm_vtk.Geom.domain.Perm.Sigma = 0.5
clm_vtk.Geom.domain.Perm.NumLines = 40
clm_vtk.Geom.domain.Perm.RZeta = 5.0
clm_vtk.Geom.domain.Perm.KMax = 100.0
clm_vtk.Geom.domain.Perm.DelK = 0.2
clm_vtk.Geom.domain.Perm.Seed = 23333
clm_vtk.Geom.domain.Perm.LogNormal = "Log"
clm_vtk.Geom.domain.Perm.StratType = "Bottom"

clm_vtk.Perm.TensorType = "TensorByGeom"

clm_vtk.Geom.Perm.TensorByGeom.Names = "domain"

clm_vtk.Geom.domain.Perm.TensorValX = 1.0
clm_vtk.Geom.domain.Perm.TensorValY = 1.0
clm_vtk.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_vtk.SpecificStorage.Type = "Constant"
clm_vtk.SpecificStorage.GeomNames = "domain"
clm_vtk.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

clm_vtk.Phase.Names = "water"

clm_vtk.Phase.water.Density.Type = "Constant"
clm_vtk.Phase.water.Density.Value = 1.0

clm_vtk.Phase.water.Viscosity.Type = "Constant"
clm_vtk.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

clm_vtk.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

clm_vtk.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

clm_vtk.TimingInfo.BaseUnit = 1.0
clm_vtk.TimingInfo.StartCount = 0
clm_vtk.TimingInfo.StartTime = 0.0
clm_vtk.TimingInfo.StopTime = 5
clm_vtk.TimingInfo.DumpInterval = -1
clm_vtk.TimeStep.Type = "Constant"
clm_vtk.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

clm_vtk.Geom.Porosity.GeomNames = "domain"

clm_vtk.Geom.domain.Porosity.Type = "Constant"
clm_vtk.Geom.domain.Porosity.Value = 0.390

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

clm_vtk.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------

clm_vtk.Phase.water.Mobility.Type = "Constant"
clm_vtk.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

clm_vtk.Phase.RelPerm.Type = "VanGenuchten"
clm_vtk.Phase.RelPerm.GeomNames = "domain"

clm_vtk.Geom.domain.RelPerm.Alpha = 3.5
clm_vtk.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

clm_vtk.Phase.Saturation.Type = "VanGenuchten"
clm_vtk.Phase.Saturation.GeomNames = "domain"

clm_vtk.Geom.domain.Saturation.Alpha = 3.5
clm_vtk.Geom.domain.Saturation.N = 2.0
clm_vtk.Geom.domain.Saturation.SRes = 0.01
clm_vtk.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

clm_vtk.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

clm_vtk.Cycle.Names = "constant"
clm_vtk.Cycle.constant.Names = "alltime"
clm_vtk.Cycle.constant.alltime.Length = 1
clm_vtk.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

clm_vtk.BCPressure.PatchNames = clm_vtk.Geom.domain.Patches

clm_vtk.Patch.x_lower.BCPressure.Type = "FluxConst"
clm_vtk.Patch.x_lower.BCPressure.Cycle = "constant"
clm_vtk.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_vtk.Patch.y_lower.BCPressure.Type = "FluxConst"
clm_vtk.Patch.y_lower.BCPressure.Cycle = "constant"
clm_vtk.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_vtk.Patch.z_lower.BCPressure.Type = "FluxConst"
clm_vtk.Patch.z_lower.BCPressure.Cycle = "constant"
clm_vtk.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm_vtk.Patch.x_upper.BCPressure.Type = "FluxConst"
clm_vtk.Patch.x_upper.BCPressure.Cycle = "constant"
clm_vtk.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_vtk.Patch.y_upper.BCPressure.Type = "FluxConst"
clm_vtk.Patch.y_upper.BCPressure.Cycle = "constant"
clm_vtk.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_vtk.Patch.z_upper.BCPressure.Type = "OverlandFlow"
clm_vtk.Patch.z_upper.BCPressure.Cycle = "constant"
clm_vtk.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

clm_vtk.TopoSlopesX.Type = "Constant"
clm_vtk.TopoSlopesX.GeomNames = "domain"
clm_vtk.TopoSlopesX.Geom.domain.Value = -0.001

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

clm_vtk.TopoSlopesY.Type = "Constant"
clm_vtk.TopoSlopesY.GeomNames = "domain"
clm_vtk.TopoSlopesY.Geom.domain.Value = 0.001

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

clm_vtk.Mannings.Type = "Constant"
clm_vtk.Mannings.GeomNames = "domain"
clm_vtk.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

clm_vtk.PhaseSources.water.Type = "Constant"
clm_vtk.PhaseSources.water.GeomNames = "domain"
clm_vtk.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

clm_vtk.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

clm_vtk.Solver = "Richards"
clm_vtk.Solver.MaxIter = 500

clm_vtk.Solver.Nonlinear.MaxIter = 75
clm_vtk.Solver.Nonlinear.ResidualTol = 1e-9
clm_vtk.Solver.Nonlinear.EtaChoice = "EtaConstant"
clm_vtk.Solver.Nonlinear.EtaValue = 0.01
clm_vtk.Solver.Nonlinear.UseJacobian = True
clm_vtk.Solver.Nonlinear.StepTol = 1e-20
clm_vtk.Solver.Nonlinear.Globalization = "LineSearch"
clm_vtk.Solver.Linear.KrylovDimension = 15
clm_vtk.Solver.Linear.MaxRestart = 2

clm_vtk.Solver.Linear.Preconditioner = "PFMG"
clm_vtk.Solver.PrintSubsurf = False
clm_vtk.Solver.Drop = 1e-20
clm_vtk.Solver.AbsTol = 1e-9

clm_vtk.Solver.LSM = "CLM"
clm_vtk.Solver.WriteSiloCLM = False
clm_vtk.Solver.CLM.MetForcing = "1D"
clm_vtk.Solver.CLM.MetFileName = "narr_1hr.sc3.txt.0"
clm_vtk.Solver.CLM.MetFilePath = "."

clm_vtk.Solver.WriteSiloEvapTrans = False
clm_vtk.Solver.WriteSiloOverlandBCFlux = False
clm_vtk.Solver.PrintCLM = True

clm_vtk.Solver.CLM.SingleFile = True

clm_vtk.Solver.WriteCLMBinary = False
clm_vtk.Solver.WriteSiloCLM = False

clm_vtk.Solver.PrintLSMSink = False
clm_vtk.Solver.CLM.CLMFileDir = "./"
clm_vtk.Solver.CLM.BinaryOutDir = False

clm_vtk.Solver.CLM.WriteLastRST = True
clm_vtk.Solver.CLM.WriteLogs = False
clm_vtk.Solver.CLM.DailyRST = False

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

clm_vtk.ICPressure.Type = "HydroStaticPatch"
clm_vtk.ICPressure.GeomNames = "domain"
clm_vtk.Geom.domain.ICPressure.Value = -2.0

clm_vtk.Geom.domain.ICPressure.RefGeom = "domain"
clm_vtk.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

clm_vtk.run(working_directory=dir_name)
