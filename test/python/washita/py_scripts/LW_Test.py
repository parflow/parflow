# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID
# DETAILS:
# Arugments are 1) runname 2) year

# Import the ParFlow TCL package
# lappend   auto_path $env(PARFLOW_DIR)/bin
# package   require parflow
LW_Test = Run("LW_Test", __file__)

#-----------------------------------------------------------------------------
# Make a directory for the simulation run, files will be copied to this
# directory for running.
#-----------------------------------------------------------------------------
# file mkdir "Outputs"
# cd "./Outputs"

LW_Test.FileVersion = 4

#-----------------------------------------------------------------------------
# Set Processor topology 
#-----------------------------------------------------------------------------
LW_Test.Process.Topology.P = 1
LW_Test.Process.Topology.Q = 1
LW_Test.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
LW_Test.ComputationalGrid.Lower.X = 0.0
LW_Test.ComputationalGrid.Lower.Y = 0.0
LW_Test.ComputationalGrid.Lower.Z = 0.0

LW_Test.ComputationalGrid.DX = 1000.0
LW_Test.ComputationalGrid.DY = 1000.0
LW_Test.ComputationalGrid.DZ = 2.0

LW_Test.ComputationalGrid.NX = 41
LW_Test.ComputationalGrid.NY = 41
LW_Test.ComputationalGrid.NZ = 50

#-----------------------------------------------------------------------------
# Names of the GeomInputs
#-----------------------------------------------------------------------------
LW_Test.GeomInput.Names = 'box_input indi_input'

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
LW_Test.GeomInput.box_input.InputType = 'Box'
LW_Test.GeomInput.box_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry 
#-----------------------------------------------------------------------------
LW_Test.Geom.domain.Lower.X = 0.0
LW_Test.Geom.domain.Lower.Y = 0.0
LW_Test.Geom.domain.Lower.Z = 0.0
#  
LW_Test.Geom.domain.Upper.X = 41000.0
LW_Test.Geom.domain.Upper.Y = 41000.0
LW_Test.Geom.domain.Upper.Z = 100.0
LW_Test.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Indicator Geometry Input
#-----------------------------------------------------------------------------
LW_Test.GeomInput.indi_input.InputType = 'IndicatorField'
LW_Test.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8'
LW_Test.Geom.indi_input.FileName = 'IndicatorFile_Gleeson.50z.pfb'

LW_Test.GeomInput.s1.Value = 1
LW_Test.GeomInput.s2.Value = 2
LW_Test.GeomInput.s3.Value = 3
LW_Test.GeomInput.s4.Value = 4
LW_Test.GeomInput.s5.Value = 5
LW_Test.GeomInput.s6.Value = 6
LW_Test.GeomInput.s7.Value = 7
LW_Test.GeomInput.s8.Value = 8
LW_Test.GeomInput.s9.Value = 9
LW_Test.GeomInput.s10.Value = 10
LW_Test.GeomInput.s11.Value = 11
LW_Test.GeomInput.s12.Value = 12
LW_Test.GeomInput.s13.Value = 13
LW_Test.GeomInput.g1.Value = 21
LW_Test.GeomInput.g2.Value = 22
LW_Test.GeomInput.g3.Value = 23
LW_Test.GeomInput.g4.Value = 24
LW_Test.GeomInput.g5.Value = 25
LW_Test.GeomInput.g6.Value = 26
LW_Test.GeomInput.g7.Value = 27
LW_Test.GeomInput.g8.Value = 28

#-----------------------------------------------------------------------------
# Permeability (values in m/hr)
#-----------------------------------------------------------------------------
LW_Test.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8'

LW_Test.Geom.domain.Perm.Type = 'Constant'
LW_Test.Geom.domain.Perm.Value = 0.2

LW_Test.Geom.s1.Perm.Type = 'Constant'
LW_Test.Geom.s1.Perm.Value = 0.269022595

LW_Test.Geom.s2.Perm.Type = 'Constant'
LW_Test.Geom.s2.Perm.Value = 0.043630356

LW_Test.Geom.s3.Perm.Type = 'Constant'
LW_Test.Geom.s3.Perm.Value = 0.015841225

LW_Test.Geom.s4.Perm.Type = 'Constant'
LW_Test.Geom.s4.Perm.Value = 0.007582087

LW_Test.Geom.s5.Perm.Type = 'Constant'
LW_Test.Geom.s5.Perm.Value = 0.01818816

LW_Test.Geom.s6.Perm.Type = 'Constant'
LW_Test.Geom.s6.Perm.Value = 0.005009435

LW_Test.Geom.s7.Perm.Type = 'Constant'
LW_Test.Geom.s7.Perm.Value = 0.005492736

LW_Test.Geom.s8.Perm.Type = 'Constant'
LW_Test.Geom.s8.Perm.Value = 0.004675077

LW_Test.Geom.s9.Perm.Type = 'Constant'
LW_Test.Geom.s9.Perm.Value = 0.003386794

LW_Test.Geom.g2.Perm.Type = 'Constant'
LW_Test.Geom.g2.Perm.Value = 0.025

LW_Test.Geom.g3.Perm.Type = 'Constant'
LW_Test.Geom.g3.Perm.Value = 0.059

LW_Test.Geom.g6.Perm.Type = 'Constant'
LW_Test.Geom.g6.Perm.Value = 0.2

LW_Test.Geom.g8.Perm.Type = 'Constant'
LW_Test.Geom.g8.Perm.Value = 0.68

LW_Test.Perm.TensorType = 'TensorByGeom'
LW_Test.Geom.Perm.TensorByGeom.Names = 'domain'
LW_Test.Geom.domain.Perm.TensorValX = 1.0
LW_Test.Geom.domain.Perm.TensorValY = 1.0
LW_Test.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
LW_Test.SpecificStorage.Type = 'Constant'
LW_Test.SpecificStorage.GeomNames = 'domain'
LW_Test.Geom.domain.SpecificStorage.Value = 1.0e-5

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
LW_Test.Phase.Names = 'water'
LW_Test.Phase.water.Density.Type = 'Constant'
LW_Test.Phase.water.Density.Value = 1.0
LW_Test.Phase.water.Viscosity.Type = 'Constant'
LW_Test.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
LW_Test.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
LW_Test.Gravity = 1.0

#-----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
#-----------------------------------------------------------------------------
LW_Test.TimingInfo.BaseUnit = 1.0
LW_Test.TimingInfo.StartCount = 0.0
LW_Test.TimingInfo.StartTime = 0.0
LW_Test.TimingInfo.StopTime = 12.0
LW_Test.TimingInfo.DumpInterval = 24.0
LW_Test.TimeStep.Type = 'Constant'
LW_Test.TimeStep.Value = 1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
LW_Test.Geom.Porosity.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9'

LW_Test.Geom.domain.Porosity.Type = 'Constant'
LW_Test.Geom.domain.Porosity.Value = 0.4

LW_Test.Geom.s1.Porosity.Type = 'Constant'
LW_Test.Geom.s1.Porosity.Value = 0.375

LW_Test.Geom.s2.Porosity.Type = 'Constant'
LW_Test.Geom.s2.Porosity.Value = 0.39

LW_Test.Geom.s3.Porosity.Type = 'Constant'
LW_Test.Geom.s3.Porosity.Value = 0.387

LW_Test.Geom.s4.Porosity.Type = 'Constant'
LW_Test.Geom.s4.Porosity.Value = 0.439

LW_Test.Geom.s5.Porosity.Type = 'Constant'
LW_Test.Geom.s5.Porosity.Value = 0.489

LW_Test.Geom.s6.Porosity.Type = 'Constant'
LW_Test.Geom.s6.Porosity.Value = 0.399

LW_Test.Geom.s7.Porosity.Type = 'Constant'
LW_Test.Geom.s7.Porosity.Value = 0.384

LW_Test.Geom.s8.Porosity.Type = 'Constant'
LW_Test.Geom.s8.Porosity.Value = 0.482

LW_Test.Geom.s9.Porosity.Type = 'Constant'
LW_Test.Geom.s9.Porosity.Value = 0.442

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
LW_Test.Domain.GeomName = 'domain'

#----------------------------------------------------------------------------
# Mobility
#----------------------------------------------------------------------------
LW_Test.Phase.water.Mobility.Type = 'Constant'
LW_Test.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
LW_Test.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
LW_Test.Cycle.Names = 'constant'
LW_Test.Cycle.constant.Names = 'alltime'
LW_Test.Cycle.constant.alltime.Length = 1
LW_Test.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions
#-----------------------------------------------------------------------------
LW_Test.BCPressure.PatchNames = LW_Test.Geom.domain.Patches

LW_Test.Patch.x_lower.BCPressure.Type = 'FluxConst'
LW_Test.Patch.x_lower.BCPressure.Cycle = 'constant'
LW_Test.Patch.x_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.y_lower.BCPressure.Type = 'FluxConst'
LW_Test.Patch.y_lower.BCPressure.Cycle = 'constant'
LW_Test.Patch.y_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.z_lower.BCPressure.Type = 'FluxConst'
LW_Test.Patch.z_lower.BCPressure.Cycle = 'constant'
LW_Test.Patch.z_lower.BCPressure.alltime.Value = 0.0

LW_Test.Patch.x_upper.BCPressure.Type = 'FluxConst'
LW_Test.Patch.x_upper.BCPressure.Cycle = 'constant'
LW_Test.Patch.x_upper.BCPressure.alltime.Value = 0.0

LW_Test.Patch.y_upper.BCPressure.Type = 'FluxConst'
LW_Test.Patch.y_upper.BCPressure.Cycle = 'constant'
LW_Test.Patch.y_upper.BCPressure.alltime.Value = 0.0

LW_Test.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
LW_Test.Patch.z_upper.BCPressure.Cycle = 'constant'
LW_Test.Patch.z_upper.BCPressure.alltime.Value = 0.0

#-----------------------------------------------------------------------------
# Topo slopes in x-direction
#-----------------------------------------------------------------------------
LW_Test.TopoSlopesX.Type = 'PFBFile'
LW_Test.TopoSlopesX.GeomNames = 'domain'
LW_Test.TopoSlopesX.FileName = 'LW.slopex.pfb'

#-----------------------------------------------------------------------------
# Topo slopes in y-direction
#-----------------------------------------------------------------------------
LW_Test.TopoSlopesY.Type = 'PFBFile'
LW_Test.TopoSlopesY.GeomNames = 'domain'
LW_Test.TopoSlopesY.FileName = 'LW.slopey.pfb'

#-----------------------------------------------------------------------------
# Mannings coefficient
#-----------------------------------------------------------------------------
LW_Test.Mannings.Type = 'Constant'
LW_Test.Mannings.GeomNames = 'domain'
LW_Test.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
LW_Test.Phase.RelPerm.Type = 'VanGenuchten'
LW_Test.Phase.RelPerm.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 '

LW_Test.Geom.domain.RelPerm.Alpha = 3.5
LW_Test.Geom.domain.RelPerm.N = 2.0

LW_Test.Geom.s1.RelPerm.Alpha = 3.548
LW_Test.Geom.s1.RelPerm.N = 4.162

LW_Test.Geom.s2.RelPerm.Alpha = 3.467
LW_Test.Geom.s2.RelPerm.N = 2.738

LW_Test.Geom.s3.RelPerm.Alpha = 2.692
LW_Test.Geom.s3.RelPerm.N = 2.445

LW_Test.Geom.s4.RelPerm.Alpha = 0.501
LW_Test.Geom.s4.RelPerm.N = 2.659

LW_Test.Geom.s5.RelPerm.Alpha = 0.661
LW_Test.Geom.s5.RelPerm.N = 2.659

LW_Test.Geom.s6.RelPerm.Alpha = 1.122
LW_Test.Geom.s6.RelPerm.N = 2.479

LW_Test.Geom.s7.RelPerm.Alpha = 2.089
LW_Test.Geom.s7.RelPerm.N = 2.318

LW_Test.Geom.s8.RelPerm.Alpha = 0.832
LW_Test.Geom.s8.RelPerm.N = 2.514

LW_Test.Geom.s9.RelPerm.Alpha = 1.585
LW_Test.Geom.s9.RelPerm.N = 2.413

#-----------------------------------------------------------------------------
# Saturation
#-----------------------------------------------------------------------------
LW_Test.Phase.Saturation.Type = 'VanGenuchten'
LW_Test.Phase.Saturation.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 '

LW_Test.Geom.domain.Saturation.Alpha = 3.5
LW_Test.Geom.domain.Saturation.N = 2.
LW_Test.Geom.domain.Saturation.SRes = 0.2
LW_Test.Geom.domain.Saturation.SSat = 1.0

LW_Test.Geom.s1.Saturation.Alpha = 3.548
LW_Test.Geom.s1.Saturation.N = 4.162
LW_Test.Geom.s1.Saturation.SRes = 0.000001
LW_Test.Geom.s1.Saturation.SSat = 1.0

LW_Test.Geom.s2.Saturation.Alpha = 3.467
LW_Test.Geom.s2.Saturation.N = 2.738
LW_Test.Geom.s2.Saturation.SRes = 0.000001
LW_Test.Geom.s2.Saturation.SSat = 1.0

LW_Test.Geom.s3.Saturation.Alpha = 2.692
LW_Test.Geom.s3.Saturation.N = 2.445
LW_Test.Geom.s3.Saturation.SRes = 0.000001
LW_Test.Geom.s3.Saturation.SSat = 1.0

LW_Test.Geom.s4.Saturation.Alpha = 0.501
LW_Test.Geom.s4.Saturation.N = 2.659
LW_Test.Geom.s4.Saturation.SRes = 0.000001
LW_Test.Geom.s4.Saturation.SSat = 1.0

LW_Test.Geom.s5.Saturation.Alpha = 0.661
LW_Test.Geom.s5.Saturation.N = 2.659
LW_Test.Geom.s5.Saturation.SRes = 0.000001
LW_Test.Geom.s5.Saturation.SSat = 1.0

LW_Test.Geom.s6.Saturation.Alpha = 1.122
LW_Test.Geom.s6.Saturation.N = 2.479
LW_Test.Geom.s6.Saturation.SRes = 0.000001
LW_Test.Geom.s6.Saturation.SSat = 1.0

LW_Test.Geom.s7.Saturation.Alpha = 2.089
LW_Test.Geom.s7.Saturation.N = 2.318
LW_Test.Geom.s7.Saturation.SRes = 0.000001
LW_Test.Geom.s7.Saturation.SSat = 1.0

LW_Test.Geom.s8.Saturation.Alpha = 0.832
LW_Test.Geom.s8.Saturation.N = 2.514
LW_Test.Geom.s8.Saturation.SRes = 0.000001
LW_Test.Geom.s8.Saturation.SSat = 1.0

LW_Test.Geom.s9.Saturation.Alpha = 1.585
LW_Test.Geom.s9.Saturation.N = 2.413
LW_Test.Geom.s9.Saturation.SRes = 0.000001
LW_Test.Geom.s9.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
LW_Test.PhaseSources.water.Type = 'Constant'
LW_Test.PhaseSources.water.GeomNames = 'domain'
LW_Test.PhaseSources.water.Geom.domain.Value = 0.0

#----------------------------------------------------------------
# CLM Settings:
# ------------------------------------------------------------
LW_Test.Solver.LSM = 'CLM'
LW_Test.Solver.CLM.CLMFileDir = 'clm_output/'
LW_Test.Solver.CLM.Print1dOut = False
LW_Test.Solver.BinaryOutDir = False
LW_Test.Solver.CLM.CLMDumpInterval = 1

LW_Test.Solver.CLM.MetForcing = '3D'
LW_Test.Solver.CLM.MetFileName = 'NLDAS'
LW_Test.Solver.CLM.MetFilePath = '.'
LW_Test.Solver.CLM.MetFileNT = 24
LW_Test.Solver.CLM.IstepStart = 1

LW_Test.Solver.CLM.EvapBeta = 'Linear'
LW_Test.Solver.CLM.VegWaterStress = 'Saturation'
LW_Test.Solver.CLM.ResSat = 0.1
LW_Test.Solver.CLM.WiltingPoint = 0.12
LW_Test.Solver.CLM.FieldCapacity = 0.98
LW_Test.Solver.CLM.IrrigationType = 'none'

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
LW_Test.ICPressure.Type = 'PFBFile'
LW_Test.ICPressure.GeomNames = 'domain'
LW_Test.Geom.domain.ICPressure.RefPatch = 'z_upper'
LW_Test.Geom.domain.ICPressure.FileName = 'press.init.pfb'

#----------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
#Writing output (all pfb):
LW_Test.Solver.PrintSubsurfData = False
LW_Test.Solver.PrintPressure = True
LW_Test.Solver.PrintSaturation = True
LW_Test.Solver.PrintMask = True

LW_Test.Solver.WriteCLMBinary = False
LW_Test.Solver.PrintCLM = True
LW_Test.Solver.WriteSiloSpecificStorage = False
LW_Test.Solver.WriteSiloMannings = False
LW_Test.Solver.WriteSiloMask = False
LW_Test.Solver.WriteSiloSlopes = False
LW_Test.Solver.WriteSiloSubsurfData = False
LW_Test.Solver.WriteSiloPressure = False
LW_Test.Solver.WriteSiloSaturation = False
LW_Test.Solver.WriteSiloEvapTrans = False
LW_Test.Solver.WriteSiloEvapTransSum = False
LW_Test.Solver.WriteSiloOverlandSum = False
LW_Test.Solver.WriteSiloCLM = False


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
LW_Test.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
# ParFlow Solution
LW_Test.Solver = 'Richards'
LW_Test.Solver.TerrainFollowingGrid = True
LW_Test.Solver.Nonlinear.VariableDz = False

LW_Test.Solver.MaxIter = 25000
LW_Test.Solver.Drop = 1E-20
LW_Test.Solver.AbsTol = 1E-8
LW_Test.Solver.MaxConvergenceFailures = 8
LW_Test.Solver.Nonlinear.MaxIter = 80
LW_Test.Solver.Nonlinear.ResidualTol = 1e-6

## new solver settings for Terrain Following Grid
LW_Test.Solver.Nonlinear.EtaChoice = 'EtaConstant'
LW_Test.Solver.Nonlinear.EtaValue = 0.001
LW_Test.Solver.Nonlinear.UseJacobian = True
LW_Test.Solver.Nonlinear.DerivativeEpsilon = 1e-16
LW_Test.Solver.Nonlinear.StepTol = 1e-30
LW_Test.Solver.Nonlinear.Globalization = 'LineSearch'
LW_Test.Solver.Linear.KrylovDimension = 70
LW_Test.Solver.Linear.MaxRestarts = 2

LW_Test.Solver.Linear.Preconditioner = 'PFMGOctree'
LW_Test.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'


#-----------------------------------------------------------------------------
# Copy files and distribute.
#-----------------------------------------------------------------------------

# ParFlow Inputs
path = "../../parflow_input"
# foreach file "LW.slopex LW.slopey IndicatorFile_Gleeson.50z press.init" {
#     file copy -force [format "%s/%s.pfb" $path $file] .
# }

#-----------------------------------------------------------------------------
# Distribute inputs
#-----------------------------------------------------------------------------
# pfdist -nz 1 LW.slopex.pfb
# pfdist -nz 1 LW.slopey.pfb

# pfdist IndicatorFile_Gleeson.50z.pfb
# pfdist press.init.pfb

#CLM Inputs
path = "../../clm_input"
# foreach file "drv_clmin drv_vegp drv_vegm.alluv" {
#     file copy -force [format "%s/%s.dat" $path $file] .
# }

path = "../../NLDAS"
# foreach file "NLDAS.DSWR.000001_to_000024 NLDAS.DLWR.000001_to_000024 NLDAS.APCP.000001_to_000024 NLDAS.Temp.000001_to_000024 NLDAS.UGRD.000001_to_000024 NLDAS.VGRD.000001_to_000024 NLDAS.Press.000001_to_000024 NLDAS.SPFH.000001_to_000024" {
#     file copy -force [format "%s/%s.pfb" $path $file] .
#     pfdist -nz 24 [format "%s.pfb" $file]
# }

# file delete correct_output
# file link -symbolic correct_output "../correct_output"

#-----------------------------------------------------------------------------
# Run Simulation 
#-----------------------------------------------------------------------------
# set runname "LW"
# pfrun    $runname

# puts "ParFlow run Complete"

#-----------------------------------------------------------------------------
# Undistribute outputs
#-----------------------------------------------------------------------------
# pfundist $runname

StartTime = [expr int([pfget TimingInfo.StartTime])]
StopTime = [expr int([pfget TimingInfo.StopTime])]

ClmVariables = [list "eflx_lh_tot" "qflx_evap_soi" "swe_out" "eflx_lwrad_out" "qflx_evap_tot" "t_grnd" "eflx_sh_tot" "qflx_evap_veg" "t_soil" "eflx_soil_grnd" "qflx_infl" "qflx_evap_grnd" "qflx_tran_veg" ]
# for {set i $StartTime} { $i <= $StopTime } {incr i} { 
#     set step [format "%05d" $i]
#     foreach variable $ClmVariables {
#         pfundist $runname.out.$variable.$step.pfb
#     }
# }

#-----------------------------------------------------------------------------
# Verify output
#-----------------------------------------------------------------------------

# source ../../../pftest.tcl

sig_digits = 4

passed = 1

ParflowVariables = [list "satur" "press"]
step = [format "%05d" 0]
# foreach variable $ParflowVariables {
#     set file $runname.out.$variable.$step.pfb 
#     if ![pftestFile $file "Max difference in $file" $sig_digits] {
# 	set passed 0
#     }
# }

step = [format "%05d" 12]
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


LW_Test.run()
