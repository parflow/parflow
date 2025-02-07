# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID
# DETAILS:
# Arguments are 1) runname 2) year

# Import the ParFlow TCL package
# lappend   auto_path $env(PARFLOW_DIR)/bin
# package   require parflow
LW_Timing = Run("LW_Timing", __file__)

# -----------------------------------------------------------------------------
# Make a directory for the simulation run, files will be copied to this
# directory for running.
# -----------------------------------------------------------------------------
# file mkdir "Outputs"
# cd "./Outputs"

LW_Timing.FileVersion = "FileVersion 4"

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------
LW_Timing.Process.Topology.P = 1
LW_Timing.Process.Topology.Q = 1
LW_Timing.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
LW_Timing.ComputationalGrid.Lower.X = 0.0
LW_Timing.ComputationalGrid.Lower.Y = 0.0
LW_Timing.ComputationalGrid.Lower.Z = 0.0

LW_Timing.ComputationalGrid.DX = 1000.0
LW_Timing.ComputationalGrid.DY = 1000.0
LW_Timing.ComputationalGrid.DZ = 2.0

LW_Timing.ComputationalGrid.NX = 41
LW_Timing.ComputationalGrid.NY = 41
LW_Timing.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------
LW_Timing.GeomInput.Names = "box_input indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
LW_Timing.GeomInput.box_input.InputType = "Box"
LW_Timing.GeomInput.box_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
LW_Timing.Geom.domain.Lower.X = 0.0
LW_Timing.Geom.domain.Lower.Y = 0.0
LW_Timing.Geom.domain.Lower.Z = 0.0
#
LW_Timing.Geom.domain.Upper.X = 41000.0
LW_Timing.Geom.domain.Upper.Y = 41000.0
LW_Timing.Geom.domain.Upper.Z = 100.0
LW_Timing.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------
LW_Timing.GeomInput.indi_input.InputType = "IndicatorField"
LW_Timing.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
)
LW_Timing.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"

LW_Timing.GeomInput.s1.Value = 1
LW_Timing.GeomInput.s2.Value = 2
LW_Timing.GeomInput.s3.Value = 3
LW_Timing.GeomInput.s4.Value = 4
LW_Timing.GeomInput.s5.Value = 5
LW_Timing.GeomInput.s6.Value = 6
LW_Timing.GeomInput.s7.Value = 7
LW_Timing.GeomInput.s8.Value = 8
LW_Timing.GeomInput.s9.Value = 9
LW_Timing.GeomInput.s10.Value = 10
LW_Timing.GeomInput.s11.Value = 11
LW_Timing.GeomInput.s12.Value = 12
LW_Timing.GeomInput.s13.Value = 13
LW_Timing.GeomInput.g1.Value = 21
LW_Timing.GeomInput.g2.Value = 22
LW_Timing.GeomInput.g3.Value = 23
LW_Timing.GeomInput.g4.Value = 24
LW_Timing.GeomInput.g5.Value = 25
LW_Timing.GeomInput.g6.Value = 26
LW_Timing.GeomInput.g7.Value = 27
LW_Timing.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------
LW_Timing.Geom.Perm.Names = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8"

LW_Timing.Geom.domain.Perm.Type = "Constant"
LW_Timing.Geom.domain.Perm.Value = 0.2

LW_Timing.Geom.s1.Perm.Type = "Constant"
LW_Timing.Geom.s1.Perm.Value = 0.269022595

LW_Timing.Geom.s2.Perm.Type = "Constant"
LW_Timing.Geom.s2.Perm.Value = 0.043630356

LW_Timing.Geom.s3.Perm.Type = "Constant"
LW_Timing.Geom.s3.Perm.Value = 0.015841225

LW_Timing.Geom.s4.Perm.Type = "Constant"
LW_Timing.Geom.s4.Perm.Value = 0.007582087

LW_Timing.Geom.s5.Perm.Type = "Constant"
LW_Timing.Geom.s5.Perm.Value = 0.01818816

LW_Timing.Geom.s6.Perm.Type = "Constant"
LW_Timing.Geom.s6.Perm.Value = 0.005009435

LW_Timing.Geom.s7.Perm.Type = "Constant"
LW_Timing.Geom.s7.Perm.Value = 0.005492736

LW_Timing.Geom.s8.Perm.Type = "Constant"
LW_Timing.Geom.s8.Perm.Value = 0.004675077

LW_Timing.Geom.s9.Perm.Type = "Constant"
LW_Timing.Geom.s9.Perm.Value = 0.003386794

LW_Timing.Geom.g2.Perm.Type = "Constant"
LW_Timing.Geom.g2.Perm.Value = 0.025

LW_Timing.Geom.g3.Perm.Type = "Constant"
LW_Timing.Geom.g3.Perm.Value = 0.059

LW_Timing.Geom.g6.Perm.Type = "Constant"
LW_Timing.Geom.g6.Perm.Value = 0.2

LW_Timing.Geom.g8.Perm.Type = "Constant"
LW_Timing.Geom.g8.Perm.Value = 0.68

LW_Timing.Perm.TensorType = "TensorByGeom"
LW_Timing.Geom.Perm.TensorByGeom.Names = "domain"
LW_Timing.Geom.domain.Perm.TensorValX = 1.0
LW_Timing.Geom.domain.Perm.TensorValY = 1.0
LW_Timing.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
LW_Timing.SpecificStorage.Type = "Constant"
LW_Timing.SpecificStorage.GeomNames = "domain"
LW_Timing.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
LW_Timing.Phase.Names = "water"
LW_Timing.Phase.water.Density.Type = "Constant"
LW_Timing.Phase.water.Density.Value = 1.0
LW_Timing.Phase.water.Viscosity.Type = "Constant"
LW_Timing.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
LW_Timing.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
LW_Timing.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------
LW_Timing.TimingInfo.BaseUnit = 1.0
LW_Timing.TimingInfo.StartCount = 0.0
LW_Timing.TimingInfo.StartTime = 0.0
LW_Timing.TimingInfo.StopTime = 100.0
LW_Timing.TimingInfo.DumpInterval = 10000.0
LW_Timing.TimeStep.Type = "Constant"
LW_Timing.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------
LW_Timing.Geom.Porosity.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9"

LW_Timing.Geom.domain.Porosity.Type = "Constant"
LW_Timing.Geom.domain.Porosity.Value = 0.4

LW_Timing.Geom.s1.Porosity.Type = "Constant"
LW_Timing.Geom.s1.Porosity.Value = 0.375

LW_Timing.Geom.s2.Porosity.Type = "Constant"
LW_Timing.Geom.s2.Porosity.Value = 0.39

LW_Timing.Geom.s3.Porosity.Type = "Constant"
LW_Timing.Geom.s3.Porosity.Value = 0.387

LW_Timing.Geom.s4.Porosity.Type = "Constant"
LW_Timing.Geom.s4.Porosity.Value = 0.439

LW_Timing.Geom.s5.Porosity.Type = "Constant"
LW_Timing.Geom.s5.Porosity.Value = 0.489

LW_Timing.Geom.s6.Porosity.Type = "Constant"
LW_Timing.Geom.s6.Porosity.Value = 0.399

LW_Timing.Geom.s7.Porosity.Type = "Constant"
LW_Timing.Geom.s7.Porosity.Value = 0.384

LW_Timing.Geom.s8.Porosity.Type = "Constant"
LW_Timing.Geom.s8.Porosity.Value = 0.482

LW_Timing.Geom.s9.Porosity.Type = "Constant"
LW_Timing.Geom.s9.Porosity.Value = 0.442

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
LW_Timing.Domain.GeomName = "domain"

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------
LW_Timing.Phase.water.Mobility.Type = "Constant"
LW_Timing.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
LW_Timing.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
LW_Timing.Cycle.Names = "constant"
LW_Timing.Cycle.constant.Names = "alltime"
LW_Timing.Cycle.constant.alltime.Length = 1
LW_Timing.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------
LW_Timing.BCPressure.PatchNames = LW_Timing.Geom.domain.Patches

LW_Timing.Patch.x_lower.BCPressure.Type = "FluxConst"
LW_Timing.Patch.x_lower.BCPressure.Cycle = "constant"
LW_Timing.Patch.x_lower.BCPressure.alltime.Value = 0.0

LW_Timing.Patch.y_lower.BCPressure.Type = "FluxConst"
LW_Timing.Patch.y_lower.BCPressure.Cycle = "constant"
LW_Timing.Patch.y_lower.BCPressure.alltime.Value = 0.0

LW_Timing.Patch.z_lower.BCPressure.Type = "FluxConst"
LW_Timing.Patch.z_lower.BCPressure.Cycle = "constant"
LW_Timing.Patch.z_lower.BCPressure.alltime.Value = 0.0

LW_Timing.Patch.x_upper.BCPressure.Type = "FluxConst"
LW_Timing.Patch.x_upper.BCPressure.Cycle = "constant"
LW_Timing.Patch.x_upper.BCPressure.alltime.Value = 0.0

LW_Timing.Patch.y_upper.BCPressure.Type = "FluxConst"
LW_Timing.Patch.y_upper.BCPressure.Cycle = "constant"
LW_Timing.Patch.y_upper.BCPressure.alltime.Value = 0.0

LW_Timing.Patch.z_upper.BCPressure.Type = "OverlandFlow"
LW_Timing.Patch.z_upper.BCPressure.Cycle = "constant"
LW_Timing.Patch.z_upper.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------
LW_Timing.TopoSlopesX.Type = "PFBFile"
LW_Timing.TopoSlopesX.GeomNames = "domain"
LW_Timing.TopoSlopesX.FileName = "LW.slopex.pfb"

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------
LW_Timing.TopoSlopesY.Type = "PFBFile"
LW_Timing.TopoSlopesY.GeomNames = "domain"
LW_Timing.TopoSlopesY.FileName = "LW.slopey.pfb"

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------
LW_Timing.Mannings.Type = "Constant"
LW_Timing.Mannings.GeomNames = "domain"
LW_Timing.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
LW_Timing.Phase.RelPerm.Type = "VanGenuchten"
LW_Timing.Phase.RelPerm.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

LW_Timing.Geom.domain.RelPerm.Alpha = 3.5
LW_Timing.Geom.domain.RelPerm.N = 2.0

LW_Timing.Geom.s1.RelPerm.Alpha = 3.548
LW_Timing.Geom.s1.RelPerm.N = 4.162

LW_Timing.Geom.s2.RelPerm.Alpha = 3.467
LW_Timing.Geom.s2.RelPerm.N = 2.738

LW_Timing.Geom.s3.RelPerm.Alpha = 2.692
LW_Timing.Geom.s3.RelPerm.N = 2.445

LW_Timing.Geom.s4.RelPerm.Alpha = 0.501
LW_Timing.Geom.s4.RelPerm.N = 2.659

LW_Timing.Geom.s5.RelPerm.Alpha = 0.661
LW_Timing.Geom.s5.RelPerm.N = 2.659

LW_Timing.Geom.s6.RelPerm.Alpha = 1.122
LW_Timing.Geom.s6.RelPerm.N = 2.479

LW_Timing.Geom.s7.RelPerm.Alpha = 2.089
LW_Timing.Geom.s7.RelPerm.N = 2.318

LW_Timing.Geom.s8.RelPerm.Alpha = 0.832
LW_Timing.Geom.s8.RelPerm.N = 2.514

LW_Timing.Geom.s9.RelPerm.Alpha = 1.585
LW_Timing.Geom.s9.RelPerm.N = 2.413

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------
LW_Timing.Phase.Saturation.Type = "VanGenuchten"
LW_Timing.Phase.Saturation.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

LW_Timing.Geom.domain.Saturation.Alpha = 3.5
LW_Timing.Geom.domain.Saturation.N = 2.0
LW_Timing.Geom.domain.Saturation.SRes = 0.2
LW_Timing.Geom.domain.Saturation.SSat = 1.0

LW_Timing.Geom.s1.Saturation.Alpha = 3.548
LW_Timing.Geom.s1.Saturation.N = 4.162
LW_Timing.Geom.s1.Saturation.SRes = 0.000001
LW_Timing.Geom.s1.Saturation.SSat = 1.0

LW_Timing.Geom.s2.Saturation.Alpha = 3.467
LW_Timing.Geom.s2.Saturation.N = 2.738
LW_Timing.Geom.s2.Saturation.SRes = 0.000001
LW_Timing.Geom.s2.Saturation.SSat = 1.0

LW_Timing.Geom.s3.Saturation.Alpha = 2.692
LW_Timing.Geom.s3.Saturation.N = 2.445
LW_Timing.Geom.s3.Saturation.SRes = 0.000001
LW_Timing.Geom.s3.Saturation.SSat = 1.0

LW_Timing.Geom.s4.Saturation.Alpha = 0.501
LW_Timing.Geom.s4.Saturation.N = 2.659
LW_Timing.Geom.s4.Saturation.SRes = 0.000001
LW_Timing.Geom.s4.Saturation.SSat = 1.0

LW_Timing.Geom.s5.Saturation.Alpha = 0.661
LW_Timing.Geom.s5.Saturation.N = 2.659
LW_Timing.Geom.s5.Saturation.SRes = 0.000001
LW_Timing.Geom.s5.Saturation.SSat = 1.0

LW_Timing.Geom.s6.Saturation.Alpha = 1.122
LW_Timing.Geom.s6.Saturation.N = 2.479
LW_Timing.Geom.s6.Saturation.SRes = 0.000001
LW_Timing.Geom.s6.Saturation.SSat = 1.0

LW_Timing.Geom.s7.Saturation.Alpha = 2.089
LW_Timing.Geom.s7.Saturation.N = 2.318
LW_Timing.Geom.s7.Saturation.SRes = 0.000001
LW_Timing.Geom.s7.Saturation.SSat = 1.0

LW_Timing.Geom.s8.Saturation.Alpha = 0.832
LW_Timing.Geom.s8.Saturation.N = 2.514
LW_Timing.Geom.s8.Saturation.SRes = 0.000001
LW_Timing.Geom.s8.Saturation.SSat = 1.0

LW_Timing.Geom.s9.Saturation.Alpha = 1.585
LW_Timing.Geom.s9.Saturation.N = 2.413
LW_Timing.Geom.s9.Saturation.SRes = 0.000001
LW_Timing.Geom.s9.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
LW_Timing.PhaseSources.water.Type = "Constant"
LW_Timing.PhaseSources.water.GeomNames = "domain"
LW_Timing.PhaseSources.water.Geom.domain.Value = 0.0

# ----------------------------------------------------------------
# CLM Settings:
# ------------------------------------------------------------
LW_Timing.Solver.LSM = "CLM"
LW_Timing.Solver.CLM.CLMFileDir = "clm_output/"
LW_Timing.Solver.CLM.Print1dOut = False
LW_Timing.Solver.BinaryOutDir = False
LW_Timing.Solver.CLM.CLMDumpInterval = 1000000

LW_Timing.Solver.CLM.MetForcing = "3D"
LW_Timing.Solver.CLM.MetFileName = "NLDAS"
LW_Timing.Solver.CLM.MetFilePath = "."
LW_Timing.Solver.CLM.MetFileNT = 24
LW_Timing.Solver.CLM.IstepStart = 1

LW_Timing.Solver.CLM.EvapBeta = "Linear"
LW_Timing.Solver.CLM.VegWaterStress = "Saturation"
LW_Timing.Solver.CLM.ResSat = 0.1
LW_Timing.Solver.CLM.WiltingPoint = 0.12
LW_Timing.Solver.CLM.FieldCapacity = 0.98
LW_Timing.Solver.CLM.IrrigationType = "none"

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------
LW_Timing.ICPressure.Type = "PFBFile"
LW_Timing.ICPressure.GeomNames = "domain"
LW_Timing.Geom.domain.ICPressure.RefPatch = "z_upper"
LW_Timing.Geom.domain.ICPressure.FileName = "press.init.pfb"

# ----------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
# Writing output (all pfb):
LW_Timing.Solver.PrintSubsurfData = False
LW_Timing.Solver.PrintPressure = True
LW_Timing.Solver.PrintSaturation = True
LW_Timing.Solver.PrintMask = True

LW_Timing.Solver.WriteCLMBinary = False
LW_Timing.Solver.PrintCLM = True
LW_Timing.Solver.WriteSiloSpecificStorage = False
LW_Timing.Solver.WriteSiloMannings = False
LW_Timing.Solver.WriteSiloMask = False
LW_Timing.Solver.WriteSiloSlopes = False
LW_Timing.Solver.WriteSiloSubsurfData = False
LW_Timing.Solver.WriteSiloPressure = False
LW_Timing.Solver.WriteSiloSaturation = False
LW_Timing.Solver.WriteSiloEvapTrans = False
LW_Timing.Solver.WriteSiloEvapTransSum = False
LW_Timing.Solver.WriteSiloOverlandSum = False
LW_Timing.Solver.WriteSiloCLM = False


# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
LW_Timing.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
# ParFlow Solution
LW_Timing.Solver = "Richards"
LW_Timing.Solver.TerrainFollowingGrid = True
LW_Timing.Solver.Nonlinear.VariableDz = False

LW_Timing.Solver.MaxIter = 25000
LW_Timing.Solver.Drop = 1e-20
LW_Timing.Solver.AbsTol = 1e-8
LW_Timing.Solver.MaxConvergenceFailures = 8
LW_Timing.Solver.Nonlinear.MaxIter = 80
LW_Timing.Solver.Nonlinear.ResidualTol = 1e-6

## new solver settings for Terrain Following Grid
LW_Timing.Solver.Nonlinear.EtaChoice = "EtaConstant"
LW_Timing.Solver.Nonlinear.EtaValue = 0.001
LW_Timing.Solver.Nonlinear.UseJacobian = True
LW_Timing.Solver.Nonlinear.DerivativeEpsilon = 1e-16
LW_Timing.Solver.Nonlinear.StepTol = 1e-30
LW_Timing.Solver.Nonlinear.Globalization = "LineSearch"
LW_Timing.Solver.Linear.KrylovDimension = 70
LW_Timing.Solver.Linear.MaxRestarts = 2

LW_Timing.Solver.Linear.Preconditioner = "PFMGOctree"
LW_Timing.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"


# -----------------------------------------------------------------------------
# Copy files and distribute.
# -----------------------------------------------------------------------------

# ParFlow Inputs
path = "../../parflow_input"
# foreach file "LW.slopex LW.slopey IndicatorFile_Gleeson.50z press.init" {
#     file copy -force [format "%s/%s.pfb" $path $file] .
# }

# -----------------------------------------------------------------------------
# Distribute inputs
# -----------------------------------------------------------------------------
# pfdist -nz 1 LW.slopex.pfb
# pfdist -nz 1 LW.slopey.pfb

# pfdist IndicatorFile_Gleeson.50z.pfb
# pfdist press.init.pfb

# CLM Inputs
path = "../../clm_input"
# foreach file "drv_clmin drv_vegp drv_vegm.alluv" {
#     file copy -force [format "%s/%s.dat" $path $file] .
# }

path = "../../NLDAS"

# files = [glob "$path/NLDAS.DSWR.*.pfb"]

time_periods = []
# foreach file $files {
#     regexp {NLDAS\.DSWR\.(.*)\.pfb} $file full time
#     lappend time_periods $time
# }

# NldasVariables = [list "DSWR" "DLWR" "APCP" "Temp" "UGRD" "VGRD" "Press" "SPFH"]

# foreach time_period $time_periods {
#     foreach variable $NldasVariables {
# 	set file NLDAS.$variable.$time_period.pfb
# 	file copy -force [format "%s/%s" $path $file] .
# 	pfdist -nz 24 $file
#     }
# }

# file delete correct_output
# file link -symbolic correct_output "../correct_output"

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------
# set runname "LW"
# pfrun    $runname

# puts "ParFlow run Complete"

# -----------------------------------------------------------------------------
# Undistribute outputs
# -----------------------------------------------------------------------------
# pfundist $runname

# StartTime = [expr int([pfget TimingInfo.StartTime])]
# StopTime = [expr int([pfget TimingInfo.StopTime])]

# ClmVariables = [list "eflx_lh_tot" "qflx_evap_soi" "swe_out" "eflx_lwrad_out" "qflx_evap_tot" "t_grnd" "eflx_sh_tot" "qflx_evap_veg" "t_soil" "eflx_soil_grnd" "qflx_infl" "qflx_evap_grnd" "qflx_tran_veg" ]
# for {set i $StartTime} { $i <= $StopTime } {incr i} {
#     set step [format "%05d" $i]
#     foreach variable $ClmVariables {
#         pfundist $runname.out.$variable.$step.pfb
#     }
# }

# -----------------------------------------------------------------------------
# Verify output
# -----------------------------------------------------------------------------

# source ../../../pftest.tcl

sig_digits = 4

passed = 1

# ParflowVariables = [list "satur" "press"]
# step = [format "%05d" 0]
# foreach variable $ParflowVariables {
#     set file $runname.out.$variable.$step.pfb
#     if ![pftestFile $file "Max difference in $file" $sig_digits] {
# 	set passed 0
#     }
# }

# step = [format "%05d" 12]
# foreach variable $ClmVariables {
#     set file $runname.out.$variable.$step.pfb
#     if ![pftestFile $file "Max difference in $file" $sig_digits] {
# 	set passed 0
#     }
# }

# if $passed {
#     puts "default_single : PASSED"
# } {
#     puts "default_single : FAILED"
# }


LW_Timing.run()
