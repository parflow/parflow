import sys
import argparse
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path, rm

clayL_benchmark = Run("clayL_benchmark", __file__)

clayL_benchmark.FileVersion = 4

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1, type=int)
parser.add_argument("-q", "--q", default=1, type=int)
parser.add_argument("-xcells", "--xcells", default=1, type=int)
parser.add_argument("-ycells", "--ycells", default=1, type=int)
parser.add_argument("-j", "--use_jacobian", default=False, type=bool)
args = parser.parse_args()

clayL_benchmark.Process.Topology.P = args.p
clayL_benchmark.Process.Topology.Q = args.q
clayL_benchmark.Process.Topology.R = 1


#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

nx = args.xcells
ny = args.ycells
nz = 240

dx = 1.0
dy = 1.0
dz = 0.025

ux = nx * dx
uy = ny * dy
uz = nz * dz

clayL_benchmark.ComputationalGrid.Lower.X = 0.0
clayL_benchmark.ComputationalGrid.Lower.Y = 0.0
clayL_benchmark.ComputationalGrid.Lower.Z = 0.0

clayL_benchmark.ComputationalGrid.NX = nx
clayL_benchmark.ComputationalGrid.NY = ny
clayL_benchmark.ComputationalGrid.NZ = nz

clayL_benchmark.ComputationalGrid.DX = dx
clayL_benchmark.ComputationalGrid.DY = dy
clayL_benchmark.ComputationalGrid.DZ = dz

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
clayL_benchmark.GeomInput.Names = 'domain_input'

clayL_benchmark.GeomInput.domain_input.InputType = 'Box'
clayL_benchmark.GeomInput.domain_input.GeomName = 'domain'

clayL_benchmark.Geom.domain.Patches = 'left right front back bottom top'

clayL_benchmark.Geom.domain.Lower.X = 0.0
clayL_benchmark.Geom.domain.Lower.Y = 0.0
clayL_benchmark.Geom.domain.Lower.Z = 0.0

clayL_benchmark.Geom.domain.Upper.X = ux
clayL_benchmark.Geom.domain.Upper.Y = uy
clayL_benchmark.Geom.domain.Upper.Z = uz

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
clayL_benchmark.Geom.Perm.Names = 'domain'

# Values in m/hour

clayL_benchmark.Geom.domain.Perm.Type = 'Constant'
clayL_benchmark.Geom.domain.Perm.Value = 1.0e-3

#-----------------------------------------------------------------------------
# Perm Tensors
#-----------------------------------------------------------------------------

clayL_benchmark.Perm.TensorType = 'TensorByGeom'

clayL_benchmark.Geom.Perm.TensorByGeom.Names = 'domain'

clayL_benchmark.Geom.domain.Perm.TensorValX = 1.0
clayL_benchmark.Geom.domain.Perm.TensorValY = 1.0
clayL_benchmark.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

clayL_benchmark.SpecificStorage.Type = 'Constant'
clayL_benchmark.SpecificStorage.GeomNames = 'domain'
clayL_benchmark.Geom.domain.SpecificStorage.Value = 1.0e-8

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clayL_benchmark.Phase.Names = 'water'

clayL_benchmark.Phase.water.Density.Type = 'Constant'
clayL_benchmark.Phase.water.Density.Value = 1.0

clayL_benchmark.Phase.water.Viscosity.Type = 'Constant'
clayL_benchmark.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

clayL_benchmark.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

clayL_benchmark.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clayL_benchmark.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
time = 1.
fac = 10.0
clayL_benchmark.TimingInfo.BaseUnit = 1.0
clayL_benchmark.TimingInfo.StartCount = 0
clayL_benchmark.TimingInfo.StartTime = 0.0
clayL_benchmark.TimingInfo.StopTime = time * fac
clayL_benchmark.TimingInfo.DumpInterval = time * fac
clayL_benchmark.TimeStep.Type = 'Constant'
clayL_benchmark.TimeStep.Value = time

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clayL_benchmark.Geom.Porosity.GeomNames = 'domain'

clayL_benchmark.Geom.domain.Porosity.Type = 'Constant'
clayL_benchmark.Geom.domain.Porosity.Value = 0.451

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

clayL_benchmark.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------


clayL_benchmark.Phase.RelPerm.Type = 'VanGenuchten'
clayL_benchmark.Phase.RelPerm.GeomNames = 'domain'

clayL_benchmark.Geom.domain.RelPerm.Alpha = 1.0
clayL_benchmark.Geom.domain.RelPerm.N = 4.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clayL_benchmark.Phase.Saturation.Type = 'VanGenuchten'
clayL_benchmark.Phase.Saturation.GeomNames = 'domain'

clayL_benchmark.Geom.domain.Saturation.Alpha = 1.0
clayL_benchmark.Geom.domain.Saturation.N = 4.
clayL_benchmark.Geom.domain.Saturation.SRes = 0.15
clayL_benchmark.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
clayL_benchmark.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
clayL_benchmark.Cycle.Names = 'constant'
clayL_benchmark.Cycle.constant.Names = 'alltime'
clayL_benchmark.Cycle.constant.alltime.Length = 1
clayL_benchmark.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
clayL_benchmark.BCPressure.PatchNames = 'left right front back bottom top'

clayL_benchmark.Patch.left.BCPressure.Type = 'FluxConst'
clayL_benchmark.Patch.left.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.left.BCPressure.alltime.Value = 0.0

clayL_benchmark.Patch.right.BCPressure.Type = 'FluxConst'
clayL_benchmark.Patch.right.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.right.BCPressure.alltime.Value = 0.0

clayL_benchmark.Patch.front.BCPressure.Type = 'FluxConst'
clayL_benchmark.Patch.front.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.front.BCPressure.alltime.Value = 0.0

clayL_benchmark.Patch.back.BCPressure.Type = 'FluxConst'
clayL_benchmark.Patch.back.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.back.BCPressure.alltime.Value = 0.0

#---- Bottom BC
clayL_benchmark.Patch.bottom.BCPressure.Type = 'DirEquilRefPatch'
clayL_benchmark.Patch.bottom.BCPressure.RefGeom = 'domain'
clayL_benchmark.Patch.bottom.BCPressure.RefPatch = 'bottom'
clayL_benchmark.Patch.bottom.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.bottom.BCPressure.alltime.Value = 0.0
#---- End Bottom BC

#---- Top BC
clayL_benchmark.Patch.top.BCPressure.Type = 'FluxConst'
clayL_benchmark.Patch.top.BCPressure.Cycle = 'constant'
clayL_benchmark.Patch.top.BCPressure.alltime.Value = -0.0008
#---- End Top BC

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

clayL_benchmark.TopoSlopesX.Type = 'Constant'
clayL_benchmark.TopoSlopesX.GeomNames = 'domain'
clayL_benchmark.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

clayL_benchmark.TopoSlopesY.Type = 'Constant'
clayL_benchmark.TopoSlopesY.GeomNames = 'domain'
clayL_benchmark.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

clayL_benchmark.Mannings.Type = 'Constant'
clayL_benchmark.Mannings.GeomNames = 'domain'
clayL_benchmark.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clayL_benchmark.PhaseSources.water.Type = 'Constant'
clayL_benchmark.PhaseSources.water.GeomNames = 'domain'
clayL_benchmark.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

clayL_benchmark.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

clayL_benchmark.Solver = 'Richards'
clayL_benchmark.Solver.MaxIter = 250000

clayL_benchmark.Solver.Nonlinear.MaxIter = 300
clayL_benchmark.Solver.Nonlinear.ResidualTol = 1e-5
clayL_benchmark.Solver.Nonlinear.EtaChoice = 'Walker1'
clayL_benchmark.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clayL_benchmark.Solver.Nonlinear.EtaValue = 0.001
clayL_benchmark.Solver.Nonlinear.UseJacobian = args.use_jacobian
clayL_benchmark.Solver.Nonlinear.DerivativeEpsilon = 1e-16
clayL_benchmark.Solver.Nonlinear.StepTol = 1e-10
clayL_benchmark.Solver.Nonlinear.Globalization = 'LineSearch'
clayL_benchmark.Solver.Linear.KrylovDimension = 20
clayL_benchmark.Solver.Linear.MaxRestart = 2

clayL_benchmark.Solver.Linear.Preconditioner = 'MGSemi'
clayL_benchmark.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
clayL_benchmark.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

# pfset Solver.Drop                                       1E-20
# pfset Solver.AbsTol                                     1E-12

clayL_benchmark.Solver.PrintSaturation = False
clayL_benchmark.Solver.PrintSubsurf = False
clayL_benchmark.Solver.PrintPressure = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

clayL_benchmark.ICPressure.Type = 'HydroStaticPatch'
clayL_benchmark.ICPressure.GeomNames = 'domain'
clayL_benchmark.Geom.domain.ICPressure.Value = -3.0

clayL_benchmark.Geom.domain.ICPressure.RefGeom = 'domain'
clayL_benchmark.Geom.domain.ICPressure.RefPatch = 'bottom'

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path('clayL_out')
mkdir(new_output_dir_name)
clayL_benchmark.run(working_directory=new_output_dir_name)
