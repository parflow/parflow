# -----------------------------------------------------------------------------
# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID
# DETAILS
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import cp, get_absolute_path, rm
from parflow.tools.builders import CLMImporter


washita = Run("washita_clm_keys", __file__)

# -----------------------------------------------------------------------------
# Import the driver files, and remove them.
# -----------------------------------------------------------------------------

# Set up the computational grid
washita.ComputationalGrid.NX = 41
washita.ComputationalGrid.NY = 41
washita.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Copy input files
# -----------------------------------------------------------------------------

cp("$PF_SRC/test/tcl/washita/parflow_input/LW.slopex.pfb")
cp("$PF_SRC/test/tcl/washita/parflow_input/LW.slopey.pfb")
cp("$PF_SRC/test/tcl/washita/parflow_input/IndicatorFile_Gleeson.50z.pfb")
cp("$PF_SRC/test/tcl/washita/parflow_input/press.init.pfb")


nldas_files = [
    "NLDAS.DSWR.000001_to_000024.pfb",
    "NLDAS.DLWR.000001_to_000024.pfb",
    "NLDAS.APCP.000001_to_000024.pfb",
    "NLDAS.Temp.000001_to_000024.pfb",
    "NLDAS.UGRD.000001_to_000024.pfb",
    "NLDAS.VGRD.000001_to_000024.pfb",
    "NLDAS.Press.000001_to_000024.pfb",
    "NLDAS.SPFH.000001_to_000024.pfb",
]

for file in nldas_files:
    cp("$PF_SRC/test/tcl/washita/NLDAS/" + file)
    washita.dist(file, R=24)

# -----------------------------------------------------------------------------

washita.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------
washita.Process.Topology.P = 1
washita.Process.Topology.Q = 1
washita.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

washita.ComputationalGrid.Lower.X = 0.0
washita.ComputationalGrid.Lower.Y = 0.0
washita.ComputationalGrid.Lower.Z = 0.0

washita.ComputationalGrid.DX = 1000.0
washita.ComputationalGrid.DY = 1000.0
washita.ComputationalGrid.DZ = 2.0

# NX, NY, and NZ already set above

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

washita.GeomInput.Names = "box_input indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

washita.GeomInput.box_input.InputType = "Box"
washita.GeomInput.box_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

washita.Geom.domain.Lower.X = 0.0
washita.Geom.domain.Lower.Y = 0.0
washita.Geom.domain.Lower.Z = 0.0
#
washita.Geom.domain.Upper.X = 41000.0
washita.Geom.domain.Upper.Y = 41000.0
washita.Geom.domain.Upper.Z = 100.0
washita.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

washita.GeomInput.indi_input.InputType = "IndicatorField"
washita.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
)
washita.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"

washita.GeomInput.s1.Value = 1
washita.GeomInput.s2.Value = 2
washita.GeomInput.s3.Value = 3
washita.GeomInput.s4.Value = 4
washita.GeomInput.s5.Value = 5
washita.GeomInput.s6.Value = 6
washita.GeomInput.s7.Value = 7
washita.GeomInput.s8.Value = 8
washita.GeomInput.s9.Value = 9
washita.GeomInput.s10.Value = 10
washita.GeomInput.s11.Value = 11
washita.GeomInput.s12.Value = 12
washita.GeomInput.s13.Value = 13
washita.GeomInput.g1.Value = 21
washita.GeomInput.g2.Value = 22
washita.GeomInput.g3.Value = 23
washita.GeomInput.g4.Value = 24
washita.GeomInput.g5.Value = 25
washita.GeomInput.g6.Value = 26
washita.GeomInput.g7.Value = 27
washita.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

washita.Geom.Perm.Names = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8"

washita.Geom.domain.Perm.Type = "Constant"
washita.Geom.domain.Perm.Value = 0.2

washita.Geom.s1.Perm.Type = "Constant"
washita.Geom.s1.Perm.Value = 0.269022595

washita.Geom.s2.Perm.Type = "Constant"
washita.Geom.s2.Perm.Value = 0.043630356

washita.Geom.s3.Perm.Type = "Constant"
washita.Geom.s3.Perm.Value = 0.015841225

washita.Geom.s4.Perm.Type = "Constant"
washita.Geom.s4.Perm.Value = 0.007582087

washita.Geom.s5.Perm.Type = "Constant"
washita.Geom.s5.Perm.Value = 0.01818816

washita.Geom.s6.Perm.Type = "Constant"
washita.Geom.s6.Perm.Value = 0.005009435

washita.Geom.s7.Perm.Type = "Constant"
washita.Geom.s7.Perm.Value = 0.005492736

washita.Geom.s8.Perm.Type = "Constant"
washita.Geom.s8.Perm.Value = 0.004675077

washita.Geom.s9.Perm.Type = "Constant"
washita.Geom.s9.Perm.Value = 0.003386794

washita.Geom.g2.Perm.Type = "Constant"
washita.Geom.g2.Perm.Value = 0.025

washita.Geom.g3.Perm.Type = "Constant"
washita.Geom.g3.Perm.Value = 0.059

washita.Geom.g6.Perm.Type = "Constant"
washita.Geom.g6.Perm.Value = 0.2

washita.Geom.g8.Perm.Type = "Constant"
washita.Geom.g8.Perm.Value = 0.68

washita.Perm.TensorType = "TensorByGeom"
washita.Geom.Perm.TensorByGeom.Names = "domain"
washita.Geom.domain.Perm.TensorValX = 1.0
washita.Geom.domain.Perm.TensorValY = 1.0
washita.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

washita.SpecificStorage.Type = "Constant"
washita.SpecificStorage.GeomNames = "domain"
washita.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

washita.Phase.Names = "water"
washita.Phase.water.Density.Type = "Constant"
washita.Phase.water.Density.Value = 1.0
washita.Phase.water.Viscosity.Type = "Constant"
washita.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

washita.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

washita.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

washita.TimingInfo.BaseUnit = 1.0
washita.TimingInfo.StartCount = 0
washita.TimingInfo.StartTime = 0.0
washita.TimingInfo.StopTime = 12.0
washita.TimingInfo.DumpInterval = 24.0
washita.TimeStep.Type = "Constant"
washita.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

washita.Geom.Porosity.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9"

washita.Geom.domain.Porosity.Type = "Constant"
washita.Geom.domain.Porosity.Value = 0.4

washita.Geom.s1.Porosity.Type = "Constant"
washita.Geom.s1.Porosity.Value = 0.375

washita.Geom.s2.Porosity.Type = "Constant"
washita.Geom.s2.Porosity.Value = 0.39

washita.Geom.s3.Porosity.Type = "Constant"
washita.Geom.s3.Porosity.Value = 0.387

washita.Geom.s4.Porosity.Type = "Constant"
washita.Geom.s4.Porosity.Value = 0.439

washita.Geom.s5.Porosity.Type = "Constant"
washita.Geom.s5.Porosity.Value = 0.489

washita.Geom.s6.Porosity.Type = "Constant"
washita.Geom.s6.Porosity.Value = 0.399

washita.Geom.s7.Porosity.Type = "Constant"
washita.Geom.s7.Porosity.Value = 0.384

washita.Geom.s8.Porosity.Type = "Constant"
washita.Geom.s8.Porosity.Value = 0.482

washita.Geom.s9.Porosity.Type = "Constant"
washita.Geom.s9.Porosity.Value = 0.442

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

washita.Domain.GeomName = "domain"

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

washita.Phase.water.Mobility.Type = "Constant"
washita.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

washita.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

washita.Cycle.Names = "constant"
washita.Cycle.constant.Names = "alltime"
washita.Cycle.constant.alltime.Length = 1
washita.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

washita.BCPressure.PatchNames = washita.Geom.domain.Patches

washita.Patch.x_lower.BCPressure.Type = "FluxConst"
washita.Patch.x_lower.BCPressure.Cycle = "constant"
washita.Patch.x_lower.BCPressure.alltime.Value = 0.0

washita.Patch.y_lower.BCPressure.Type = "FluxConst"
washita.Patch.y_lower.BCPressure.Cycle = "constant"
washita.Patch.y_lower.BCPressure.alltime.Value = 0.0

washita.Patch.z_lower.BCPressure.Type = "FluxConst"
washita.Patch.z_lower.BCPressure.Cycle = "constant"
washita.Patch.z_lower.BCPressure.alltime.Value = 0.0

washita.Patch.x_upper.BCPressure.Type = "FluxConst"
washita.Patch.x_upper.BCPressure.Cycle = "constant"
washita.Patch.x_upper.BCPressure.alltime.Value = 0.0

washita.Patch.y_upper.BCPressure.Type = "FluxConst"
washita.Patch.y_upper.BCPressure.Cycle = "constant"
washita.Patch.y_upper.BCPressure.alltime.Value = 0.0

washita.Patch.z_upper.BCPressure.Type = "OverlandFlow"
washita.Patch.z_upper.BCPressure.Cycle = "constant"
washita.Patch.z_upper.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

washita.TopoSlopesX.Type = "PFBFile"
washita.TopoSlopesX.GeomNames = "domain"
washita.TopoSlopesX.FileName = "LW.slopex.pfb"

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

washita.TopoSlopesY.Type = "PFBFile"
washita.TopoSlopesY.GeomNames = "domain"
washita.TopoSlopesY.FileName = "LW.slopey.pfb"

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

washita.Mannings.Type = "Constant"
washita.Mannings.GeomNames = "domain"
washita.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

washita.Phase.RelPerm.Type = "VanGenuchten"
washita.Phase.RelPerm.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

washita.Geom.domain.RelPerm.Alpha = 3.5
washita.Geom.domain.RelPerm.N = 2.0

washita.Geom.s1.RelPerm.Alpha = 3.548
washita.Geom.s1.RelPerm.N = 4.162

washita.Geom.s2.RelPerm.Alpha = 3.467
washita.Geom.s2.RelPerm.N = 2.738

washita.Geom.s3.RelPerm.Alpha = 2.692
washita.Geom.s3.RelPerm.N = 2.445

washita.Geom.s4.RelPerm.Alpha = 0.501
washita.Geom.s4.RelPerm.N = 2.659

washita.Geom.s5.RelPerm.Alpha = 0.661
washita.Geom.s5.RelPerm.N = 2.659

washita.Geom.s6.RelPerm.Alpha = 1.122
washita.Geom.s6.RelPerm.N = 2.479

washita.Geom.s7.RelPerm.Alpha = 2.089
washita.Geom.s7.RelPerm.N = 2.318

washita.Geom.s8.RelPerm.Alpha = 0.832
washita.Geom.s8.RelPerm.N = 2.514

washita.Geom.s9.RelPerm.Alpha = 1.585
washita.Geom.s9.RelPerm.N = 2.413

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

washita.Phase.Saturation.Type = "VanGenuchten"
washita.Phase.Saturation.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

washita.Geom.domain.Saturation.Alpha = 3.5
washita.Geom.domain.Saturation.N = 2.0
washita.Geom.domain.Saturation.SRes = 0.2
washita.Geom.domain.Saturation.SSat = 1.0

washita.Geom.s1.Saturation.Alpha = 3.548
washita.Geom.s1.Saturation.N = 4.162
washita.Geom.s1.Saturation.SRes = 0.000001
washita.Geom.s1.Saturation.SSat = 1.0

washita.Geom.s2.Saturation.Alpha = 3.467
washita.Geom.s2.Saturation.N = 2.738
washita.Geom.s2.Saturation.SRes = 0.000001
washita.Geom.s2.Saturation.SSat = 1.0

washita.Geom.s3.Saturation.Alpha = 2.692
washita.Geom.s3.Saturation.N = 2.445
washita.Geom.s3.Saturation.SRes = 0.000001
washita.Geom.s3.Saturation.SSat = 1.0

washita.Geom.s4.Saturation.Alpha = 0.501
washita.Geom.s4.Saturation.N = 2.659
washita.Geom.s4.Saturation.SRes = 0.000001
washita.Geom.s4.Saturation.SSat = 1.0

washita.Geom.s5.Saturation.Alpha = 0.661
washita.Geom.s5.Saturation.N = 2.659
washita.Geom.s5.Saturation.SRes = 0.000001
washita.Geom.s5.Saturation.SSat = 1.0

washita.Geom.s6.Saturation.Alpha = 1.122
washita.Geom.s6.Saturation.N = 2.479
washita.Geom.s6.Saturation.SRes = 0.000001
washita.Geom.s6.Saturation.SSat = 1.0

washita.Geom.s7.Saturation.Alpha = 2.089
washita.Geom.s7.Saturation.N = 2.318
washita.Geom.s7.Saturation.SRes = 0.000001
washita.Geom.s7.Saturation.SSat = 1.0

washita.Geom.s8.Saturation.Alpha = 0.832
washita.Geom.s8.Saturation.N = 2.514
washita.Geom.s8.Saturation.SRes = 0.000001
washita.Geom.s8.Saturation.SSat = 1.0

washita.Geom.s9.Saturation.Alpha = 1.585
washita.Geom.s9.Saturation.N = 2.413
washita.Geom.s9.Saturation.SRes = 0.000001
washita.Geom.s9.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

washita.PhaseSources.water.Type = "Constant"
washita.PhaseSources.water.GeomNames = "domain"
washita.PhaseSources.water.Geom.domain.Value = 0.0

# ----------------------------------------------------------------
# CLM Settings:
# ------------------------------------------------------------

washita.Solver.LSM = "CLM"
washita.Solver.CLM.CLMFileDir = "."
washita.Solver.CLM.Print1dOut = False
washita.Solver.CLM.CLMDumpInterval = 1

washita.Solver.CLM.MetForcing = "3D"
washita.Solver.CLM.MetFileName = "NLDAS"
washita.Solver.CLM.MetFilePath = "."
washita.Solver.CLM.MetFileNT = 24
washita.Solver.CLM.IstepStart = 1

washita.Solver.CLM.EvapBeta = "Linear"
washita.Solver.CLM.VegWaterStress = "Saturation"
washita.Solver.CLM.ResSat = 0.1
washita.Solver.CLM.WiltingPoint = 0.12
washita.Solver.CLM.FieldCapacity = 0.98
washita.Solver.CLM.IrrigationType = "none"

# ---------------------------------------------------------
# New keys to generate drv files for CLM
# ---------------------------------------------------------

washita.Solver.CLM.Input.File.ActiveRestart = "clm.rst."
washita.Solver.CLM.Input.File.MetInput = "narr_1hr.dat"
washita.Solver.CLM.Input.File.Output = "clm_out.txt"
washita.Solver.CLM.Input.File.ParamOutput = "clm_para_out.dat"
washita.Solver.CLM.Input.File.VegTileSpecification = "drv_vegm.alluv.dat"

washita.Solver.CLM.Input.ICSource.Code = 2
washita.Solver.CLM.Input.InitCond.SnowCover = 0

# Time range
washita.Solver.CLM.Input.Timing.RestartCode = 2
washita.Solver.CLM.Input.Timing.StartDate = "1998/10/01"
washita.Solver.CLM.Input.Timing.StartTime = "05:00:00"
washita.Solver.CLM.Input.Timing.EndDate = "1999/10/01"
washita.Solver.CLM.Input.Timing.EndTime = "04:00:00"

# Default constants
washita.Solver.CLM.Vegetation.Map.Latitude.Value = 34.75
washita.Solver.CLM.Vegetation.Map.Longitude.Value = -98.14
washita.Solver.CLM.Vegetation.Map.Color.Value = 2
washita.Solver.CLM.Vegetation.Map.Clay.Value = 0.26
washita.Solver.CLM.Vegetation.Map.Sand.Value = 0.16

# Setup default land cover names with corresponding vegetation parameters
washita.Solver.CLM.Vegetation.Parameters.pfset(
    yaml_file="clm_input/washita_veg_params_ref.yaml"
)

# Override only existing landfrac pfb files
land_frac = washita.Solver.CLM.Vegetation.Map.LandFrac
for land_cover_name in ["croplands", "forest_dn", "grasslands", "savannas"]:
    land_frac[land_cover_name].Type = "PFBFile"
    land_frac[land_cover_name].FileName = (
        f"clm_input/washita_clm_{land_cover_name}_landfrac.pfb"
    )

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

washita.ICPressure.Type = "PFBFile"
washita.ICPressure.GeomNames = "domain"
washita.Geom.domain.ICPressure.RefPatch = "z_upper"
washita.Geom.domain.ICPressure.FileName = "press.init.pfb"

# -------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

# Writing output (all pfb):
washita.Solver.PrintSubsurfData = False
washita.Solver.PrintPressure = True
washita.Solver.PrintSaturation = True
washita.Solver.PrintMask = True

washita.Solver.WriteCLMBinary = False
washita.Solver.PrintCLM = True
washita.Solver.WriteSiloSpecificStorage = False
washita.Solver.WriteSiloMannings = False
washita.Solver.WriteSiloMask = False
washita.Solver.WriteSiloSlopes = False
washita.Solver.WriteSiloSubsurfData = False
washita.Solver.WriteSiloPressure = False
washita.Solver.WriteSiloSaturation = False
washita.Solver.WriteSiloEvapTrans = False
washita.Solver.WriteSiloEvapTransSum = False
washita.Solver.WriteSiloOverlandSum = False
washita.Solver.WriteSiloCLM = False

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

washita.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

# ParFlow Solution
washita.Solver = "Richards"
washita.Solver.TerrainFollowingGrid = True
washita.Solver.Nonlinear.VariableDz = False

washita.Solver.MaxIter = 25000
washita.Solver.Drop = 1e-20
washita.Solver.AbsTol = 1e-8
washita.Solver.MaxConvergenceFailures = 8
washita.Solver.Nonlinear.MaxIter = 80
washita.Solver.Nonlinear.ResidualTol = 1e-6

## new solver settings for Terrain Following Grid
washita.Solver.Nonlinear.EtaChoice = "EtaConstant"
washita.Solver.Nonlinear.EtaValue = 0.001
washita.Solver.Nonlinear.UseJacobian = True
washita.Solver.Nonlinear.DerivativeEpsilon = 1e-16
washita.Solver.Nonlinear.StepTol = 1e-30
washita.Solver.Nonlinear.Globalization = "LineSearch"
washita.Solver.Linear.KrylovDimension = 70
washita.Solver.Linear.MaxRestarts = 2

washita.Solver.Linear.Preconditioner = "PFMGOctree"
washita.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"

# -----------------------------------------------------------------------------
# Distribute CLM and ParFlow inputs
# -----------------------------------------------------------------------------

washita.dist("LW.slopex.pfb")
washita.dist("LW.slopey.pfb")
washita.dist("IndicatorFile_Gleeson.50z.pfb")
washita.dist("press.init.pfb")

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------

# Write out the yaml file
washita.write("washita_clm_keys", file_format="yaml")

washita.run()
