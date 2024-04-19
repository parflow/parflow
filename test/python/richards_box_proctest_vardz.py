#----------------------------------------------------------------------------
# This runs a test case with the Richards' solver
# with a simple flow domain and different BCs on the top.
# The domain geometry is purposefully smaller than the computational grid
# making more than 1/2 the domain inactive in Y.  When run with topology
# 1 2 1 this will test PF behavior for inactive processors, for different BCs
# and solver configurations.
#----------------------------------------------------------------------------

import sys, argparse
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "richards_ptest_vdz"
rbpv = Run(run_name, __file__)

#---------------------------------------------------------

rbpv.FileVersion = 4


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--p', default=1)
parser.add_argument('-q', '--q', default=1)
parser.add_argument('-r', '--r', default=1)
args = parser.parse_args()

rbpv.Process.Topology.P = args.p
rbpv.Process.Topology.Q = args.q
rbpv.Process.Topology.R = args.r

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

rbpv.ComputationalGrid.Lower.X = 0.0
rbpv.ComputationalGrid.Lower.Y = 0.0
rbpv.ComputationalGrid.Lower.Z = 0.0

rbpv.ComputationalGrid.DX = 1.0
rbpv.ComputationalGrid.DY = 1.0
rbpv.ComputationalGrid.DZ = 1.0

rbpv.ComputationalGrid.NX = 20
rbpv.ComputationalGrid.NY = 50
rbpv.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

rbpv.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

rbpv.GeomInput.domain_input.InputType = 'Box'
rbpv.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

rbpv.Geom.domain.Lower.X = 0.0
rbpv.Geom.domain.Lower.Y = 0.0
rbpv.Geom.domain.Lower.Z = 0.0

rbpv.Geom.domain.Upper.X = 20.0
rbpv.Geom.domain.Upper.Y = 20.0
rbpv.Geom.domain.Upper.Z = 20.0

rbpv.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# variable dz assignments
#---------------------------------------------------------

rbpv.Solver.Nonlinear.VariableDz = True
rbpv.dzScale.GeomNames = 'domain'
rbpv.dzScale.Type = 'nzList'
rbpv.dzScale.nzListNumber = 20
rbpv.Cell._0.dzScale.Value = 1.0
rbpv.Cell._1.dzScale.Value = 1.0
rbpv.Cell._2.dzScale.Value = 1.0
rbpv.Cell._3.dzScale.Value = 1.0
rbpv.Cell._4.dzScale.Value = 1.0
rbpv.Cell._5.dzScale.Value = 1.0
rbpv.Cell._6.dzScale.Value = 1.0
rbpv.Cell._7.dzScale.Value = 1.0
rbpv.Cell._8.dzScale.Value = 1.0
rbpv.Cell._9.dzScale.Value = 1.0
rbpv.Cell._10.dzScale.Value = 1.0
rbpv.Cell._11.dzScale.Value = 1.0
rbpv.Cell._12.dzScale.Value = 1.0
rbpv.Cell._13.dzScale.Value = 1.0
rbpv.Cell._14.dzScale.Value = 1.0
rbpv.Cell._15.dzScale.Value = 1.0
rbpv.Cell._16.dzScale.Value = 1.0
rbpv.Cell._17.dzScale.Value = 1.0
rbpv.Cell._18.dzScale.Value = 1.0
rbpv.Cell._19.dzScale.Value = 0.05

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

rbpv.Geom.Perm.Names = 'domain'
rbpv.Geom.domain.Perm.Type = 'Constant'
rbpv.Geom.domain.Perm.Value = 1.0

rbpv.Perm.TensorType = 'TensorByGeom'

rbpv.Geom.Perm.TensorByGeom.Names = 'domain'

rbpv.Geom.domain.Perm.TensorValX = 1.0
rbpv.Geom.domain.Perm.TensorValY = 1.0
rbpv.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

rbpv.SpecificStorage.Type = 'Constant'
rbpv.SpecificStorage.GeomNames = 'domain'
rbpv.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

rbpv.Phase.Names = 'water'

rbpv.Phase.water.Density.Type = 'Constant'
rbpv.Phase.water.Density.Value = 1.0

rbpv.Phase.water.Viscosity.Type = 'Constant'
rbpv.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

rbpv.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

rbpv.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

rbpv.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

rbpv.TimingInfo.BaseUnit = 10.
rbpv.TimingInfo.StartCount = 0
rbpv.TimingInfo.StartTime = 0.0
rbpv.TimingInfo.StopTime = 100.0
rbpv.TimingInfo.DumpInterval = 10.0
rbpv.TimeStep.Type = 'Constant'
rbpv.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

rbpv.Geom.Porosity.GeomNames = 'domain'
rbpv.Geom.domain.Porosity.Type = 'Constant'
rbpv.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

rbpv.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

rbpv.Phase.RelPerm.Type = 'VanGenuchten'
rbpv.Phase.RelPerm.GeomNames = 'domain'
rbpv.Geom.domain.RelPerm.Alpha = 2.0
rbpv.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

rbpv.Phase.Saturation.Type = 'VanGenuchten'
rbpv.Phase.Saturation.GeomNames = 'domain'
rbpv.Geom.domain.Saturation.Alpha = 2.0
rbpv.Geom.domain.Saturation.N = 2.0
rbpv.Geom.domain.Saturation.SRes = 0.1
rbpv.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# No Flow Barrier
#---------------------------------------------------------

rbpv.Solver.Nonlinear.FlowBarrierX = False

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

rbpv.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

rbpv.Cycle.Names = 'constant'
rbpv.Cycle.constant.Names = 'alltime'
rbpv.Cycle.constant.alltime.Length = 1
rbpv.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

rbpv.BCPressure.PatchNames = 'left right front back bottom top'

rbpv.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
rbpv.Patch.left.BCPressure.Cycle = 'constant'
rbpv.Patch.left.BCPressure.RefGeom = 'domain'
rbpv.Patch.left.BCPressure.RefPatch = 'bottom'
rbpv.Patch.left.BCPressure.alltime.Value = 11.0

rbpv.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
rbpv.Patch.right.BCPressure.Cycle = 'constant'
rbpv.Patch.right.BCPressure.RefGeom = 'domain'
rbpv.Patch.right.BCPressure.RefPatch = 'bottom'
rbpv.Patch.right.BCPressure.alltime.Value = 15.0

rbpv.Patch.front.BCPressure.Type = 'FluxConst'
rbpv.Patch.front.BCPressure.Cycle = 'constant'
rbpv.Patch.front.BCPressure.alltime.Value = 0.0

rbpv.Patch.back.BCPressure.Type = 'FluxConst'
rbpv.Patch.back.BCPressure.Cycle = 'constant'
rbpv.Patch.back.BCPressure.alltime.Value = 0.0

rbpv.Patch.bottom.BCPressure.Type = 'FluxConst'
rbpv.Patch.bottom.BCPressure.Cycle = 'constant'
rbpv.Patch.bottom.BCPressure.alltime.Value = 0.0

# used to cycle different BCs on the top of the domain, even with no
# overland flow
rbpv.Patch.top.BCPressure.Type = 'FluxConst'
rbpv.Patch.top.BCPressure.Type = 'OverlandFlow'
rbpv.Patch.top.BCPressure.Type = 'OverlandKinematic'

rbpv.Patch.top.BCPressure.Cycle = 'constant'
rbpv.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

rbpv.TopoSlopesX.Type = 'Constant'
rbpv.TopoSlopesX.GeomNames = ''
rbpv.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

rbpv.TopoSlopesY.Type = 'Constant'
rbpv.TopoSlopesY.GeomNames = ''
rbpv.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

rbpv.Mannings.Type = 'Constant'
rbpv.Mannings.GeomNames = ''
rbpv.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

rbpv.ICPressure.Type = 'HydroStaticPatch'
rbpv.ICPressure.GeomNames = 'domain'
rbpv.Geom.domain.ICPressure.Value = 13.0
rbpv.Geom.domain.ICPressure.RefGeom = 'domain'
rbpv.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

rbpv.PhaseSources.water.Type = 'Constant'
rbpv.PhaseSources.water.GeomNames = 'domain'
rbpv.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

rbpv.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

rbpv.Solver = 'Richards'
rbpv.Solver.MaxIter = 50000

rbpv.Solver.Nonlinear.MaxIter = 100
rbpv.Solver.Nonlinear.ResidualTol = 1e-7

rbpv.Solver.Nonlinear.EtaChoice = 'EtaConstant'
rbpv.Solver.Nonlinear.EtaValue = 1e-2

# used to test analytical and FD jacobian combinations
rbpv.Solver.Nonlinear.UseJacobian = True

rbpv.Solver.Nonlinear.DerivativeEpsilon = 1e-14

rbpv.Solver.Linear.KrylovDimension = 100

# used to test different linear preconditioners
rbpv.Solver.Linear.Preconditioner = 'PFMG'

rbpv.UseClustering = False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

correct_output_dir_name = get_absolute_path('../correct_output')
new_output_dir_name = get_absolute_path('test_output/richards_ptest_vdz')
mkdir(new_output_dir_name)

rbpv.run(working_directory=new_output_dir_name)
passed = True
for i in range(11):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in Pressure for timestep {timestep}"):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in Saturation for timestep {timestep}"):
        passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
