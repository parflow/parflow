#
# Test CLM with multiple reuse values for input.  The same test is run
# with different timesteps and reuse values set to match.   E.G. 1s = reuse 1, 0.1s = reuse 10.
#

# Import the ParFlow TCL package
#
from parflow import Run
clm_reuse = Run("clm_reuse", __file__)

# Total runtime of simulation
#set stopt 7762
stopt = 100
# Reuse values to run with
reuseValues = [1,4]

# This was set for reuse = 4 test; other reuse values will fail
relativeErrorTolerance = 0.2

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
clm_reuse.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

clm_reuse.Process.Topology.P = 1
clm_reuse.Process.Topology.Q = 1
clm_reuse.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm_reuse.ComputationalGrid.Lower.X = 0.0
clm_reuse.ComputationalGrid.Lower.Y = 0.0
clm_reuse.ComputationalGrid.Lower.Z = 0.0

clm_reuse.ComputationalGrid.DX = 2.0
clm_reuse.ComputationalGrid.DY = 2.0
clm_reuse.ComputationalGrid.DZ = 0.1

clm_reuse.ComputationalGrid.NX = 1
clm_reuse.ComputationalGrid.NY = 1
clm_reuse.ComputationalGrid.NZ = 100

nx = clm_reuse.ComputationalGrid.NX
dx = clm_reuse.ComputationalGrid.DX
ny = clm_reuse.ComputationalGrid.NY
dy = clm_reuse.ComputationalGrid.DY
nz = clm_reuse.ComputationalGrid.NZ
dz = clm_reuse.ComputationalGrid.DZ

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
clm_reuse.GeomInput.Names = 'domain_input'

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
clm_reuse.GeomInput.domain_input.InputType = 'Box'
clm_reuse.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
clm_reuse.Geom.domain.Lower.X = 0.0
clm_reuse.Geom.domain.Lower.Y = 0.0
clm_reuse.Geom.domain.Lower.Z = 0.0

clm_reuse.Geom.domain.Upper.X = (nx * dx)
clm_reuse.Geom.domain.Upper.Y = (ny * dy)
clm_reuse.Geom.domain.Upper.Z = (nz * dz)

clm_reuse.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clm_reuse.Geom.Perm.Names = 'domain'

clm_reuse.Geom.domain.Perm.Type = 'Constant'
clm_reuse.Geom.domain.Perm.Value = 0.04465

clm_reuse.Perm.TensorType = 'TensorByGeom'

clm_reuse.Geom.Perm.TensorByGeom.Names = 'domain'

clm_reuse.Geom.domain.Perm.TensorValX = 1.0
clm_reuse.Geom.domain.Perm.TensorValY = 1.0
clm_reuse.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_reuse.SpecificStorage.Type = 'Constant'
clm_reuse.SpecificStorage.GeomNames = 'domain'
clm_reuse.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_reuse.Phase.Names = 'water'

clm_reuse.Phase.water.Density.Type = 'Constant'
clm_reuse.Phase.water.Density.Value = 1.0

clm_reuse.Phase.water.Viscosity.Type = 'Constant'
clm_reuse.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
clm_reuse.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_reuse.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

clm_reuse.TimingInfo.BaseUnit = 1.0
clm_reuse.TimingInfo.StartCount = 0
clm_reuse.TimingInfo.StartTime = 0.0
clm_reuse.TimingInfo.StopTime = stopt
clm_reuse.TimingInfo.DumpInterval = 1.0
clm_reuse.TimeStep.Type = 'Constant'
# pfset TimeStep.Value             1.0


#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_reuse.Geom.Porosity.GeomNames = 'domain'

clm_reuse.Geom.domain.Porosity.Type = 'Constant'
clm_reuse.Geom.domain.Porosity.Value = 0.5

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
clm_reuse.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
clm_reuse.Phase.water.Mobility.Type = 'Constant'
clm_reuse.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

clm_reuse.Phase.RelPerm.Type = 'VanGenuchten'
clm_reuse.Phase.RelPerm.GeomNames = 'domain'

clm_reuse.Geom.domain.RelPerm.Alpha = 2.0
clm_reuse.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_reuse.Phase.Saturation.Type = 'VanGenuchten'
clm_reuse.Phase.Saturation.GeomNames = 'domain'

clm_reuse.Geom.domain.Saturation.Alpha = 2.0
clm_reuse.Geom.domain.Saturation.N = 3.0
clm_reuse.Geom.domain.Saturation.SRes = 0.2
clm_reuse.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clm_reuse.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clm_reuse.Cycle.Names = 'constant'
clm_reuse.Cycle.constant.Names = 'alltime'
clm_reuse.Cycle.constant.alltime.Length = 1
clm_reuse.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clm_reuse.BCPressure.PatchNames = clm_reuse.Geom.domain.Patches

clm_reuse.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_reuse.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_reuse.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_reuse.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_reuse.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_reuse.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_reuse.Patch.z_lower.BCPressure.Type = 'FluxConst'
#pfset Patch.z-lower.BCPressure.Type                   DirEquilRefPatch
clm_reuse.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_reuse.Patch.z_lower.BCPressure.alltime.Value = -0.00

clm_reuse.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_reuse.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_reuse.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_reuse.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_reuse.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_reuse.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_reuse.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
#pfset Patch.z-upper.BCPressure.Type                FluxConst
clm_reuse.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_reuse.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

clm_reuse.TopoSlopesX.Type = 'Constant'
clm_reuse.TopoSlopesX.GeomNames = 'domain'
clm_reuse.TopoSlopesX.Geom.domain.Value = 0.005

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

clm_reuse.TopoSlopesY.Type = 'Constant'
clm_reuse.TopoSlopesY.GeomNames = 'domain'
clm_reuse.TopoSlopesY.Geom.domain.Value = 0.00

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

clm_reuse.Mannings.Type = 'Constant'
clm_reuse.Mannings.GeomNames = 'domain'
clm_reuse.Mannings.Geom.domain.Value = 1e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_reuse.PhaseSources.water.Type = 'Constant'
clm_reuse.PhaseSources.water.GeomNames = 'domain'
clm_reuse.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

clm_reuse.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

clm_reuse.Solver = 'Richards'
clm_reuse.Solver.MaxIter = 90000

clm_reuse.Solver.Nonlinear.MaxIter = 100
clm_reuse.Solver.Nonlinear.ResidualTol = 1e-5
clm_reuse.Solver.Nonlinear.EtaChoice = 'Walker1'
clm_reuse.Solver.Nonlinear.EtaValue = 0.01
clm_reuse.Solver.Nonlinear.UseJacobian = True
clm_reuse.Solver.Nonlinear.DerivativeEpsilon = 1e-12
clm_reuse.Solver.Nonlinear.StepTol = 1e-30
clm_reuse.Solver.Nonlinear.Globalization = 'LineSearch'
clm_reuse.Solver.Linear.KrylovDimension = 100
clm_reuse.Solver.Linear.MaxRestarts = 5

clm_reuse.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'

clm_reuse.Solver.Linear.Preconditioner = 'PFMG'
#pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
#pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10
clm_reuse.Solver.PrintSubsurf = False
clm_reuse.Solver.Drop = 1E-20
clm_reuse.Solver.AbsTol = 1E-9

clm_reuse.Solver.LSM = 'CLM'
clm_reuse.Solver.WriteSiloCLM = True
clm_reuse.Solver.CLM.MetForcing = '1D'
clm_reuse.Solver.CLM.MetFileName = 'forcing_1.txt'
clm_reuse.Solver.CLM.MetFilePath = './'

#pfset Solver.TerrainFollowingGrid                       True
clm_reuse.Solver.CLM.EvapBeta = 'Linear'

#Writing output: PFB only no SILO
clm_reuse.Solver.PrintSubsurfData = True
clm_reuse.Solver.PrintPressure = False
clm_reuse.Solver.PrintSaturation = True
clm_reuse.Solver.PrintCLM = True
clm_reuse.Solver.PrintMask = True
clm_reuse.Solver.PrintSpecificStorage = True

#pfset Solver.WriteSiloSpecificStorage                 True
clm_reuse.Solver.WriteSiloMannings = False
clm_reuse.Solver.WriteSiloMask = False
clm_reuse.Solver.WriteSiloSlopes = False
#pfset Solver.WriteSiloSubsurfData                     True
#pfset Solver.WriteSiloPressure                        True
clm_reuse.Solver.WriteSiloSaturation = False
#pfset Solver.WriteSiloEvapTrans                       True
#pfset Solver.WriteSiloEvapTransSum                    True
#pfset Solver.WriteSiloOverlandSum                     True
#pfset Solver.WriteSiloCLM                             True
#pfset Solver.WriteSiloOverlandBCFlux                  True

clm_reuse.Solver.PrintLSMSink = False
clm_reuse.Solver.CLM.CLMDumpInterval = 1
clm_reuse.Solver.CLM.CLMFileDir = 'output/'
clm_reuse.Solver.CLM.BinaryOutDir = False
clm_reuse.Solver.CLM.IstepStart = 1
clm_reuse.Solver.WriteCLMBinary = False
clm_reuse.Solver.WriteSiloCLM = False

clm_reuse.Solver.CLM.WriteLogs = False
clm_reuse.Solver.CLM.WriteLastRST = True
clm_reuse.Solver.CLM.DailyRST = False
clm_reuse.Solver.CLM.SingleFile = True


#  pfset Solver.CLM.EvapBeta                             Linear
#  pfset Solver.CLM.VegWaterStress                       Saturation
#  pfset Solver.CLM.ResSat                               0.2
#  pfset Solver.CLM.WiltingPoint                         0.2
#  pfset Solver.CLM.FieldCapacity                        1.00
#  pfset Solver.CLM.IrrigationType                       none

# Initial conditions: water pressure
#---------------------------------------------------------

clm_reuse.ICPressure.Type = 'HydroStaticPatch'
clm_reuse.ICPressure.GeomNames = 'domain'
clm_reuse.Geom.domain.ICPressure.Value = -1.0
clm_reuse.Geom.domain.ICPressure.RefGeom = 'domain'
clm_reuse.Geom.domain.ICPressure.RefPatch = 'z_upper'

for reuseCount in reuseValues:
  new_name = f'clm_reuse_ts_{reuseCount}'
  new_name = clm_reuse.clone(f'{new_name}')
  new_name.Solver.CLM.ReuseCount = reuseCount
  new_name.TimeStep.Value = (1.0 / reuseCount)

  new_name.run()
