#------------------------------------------------------------------
# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.
#------------------------------------------------------------------

import sys
from parflow import Run, write_pfb
from parflow.tools.fs import get_absolute_path, mkdir, chdir
from parflow.tools.compare import pf_test_file
import numpy as np

run_name = "richards_FBx"
rich_fbx = Run(run_name, __file__)

#---------------------------------------------------------
# Creating and navigating to output directory
#---------------------------------------------------------

correct_output_dir_name = get_absolute_path('../correct_output')
new_output_dir_name = get_absolute_path('test_output/richards_fbx')
mkdir(new_output_dir_name)

#------------------------------------------------------------------

rich_fbx.FileVersion = 4

rich_fbx.Process.Topology.P = 1
rich_fbx.Process.Topology.Q = 1
rich_fbx.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

rich_fbx.ComputationalGrid.Lower.X = 0.0
rich_fbx.ComputationalGrid.Lower.Y = 0.0
rich_fbx.ComputationalGrid.Lower.Z = 0.0

rich_fbx.ComputationalGrid.DX = 1.0
rich_fbx.ComputationalGrid.DY = 1.0
rich_fbx.ComputationalGrid.DZ = 1.0

rich_fbx.ComputationalGrid.NX = 20
rich_fbx.ComputationalGrid.NY = 20
rich_fbx.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

rich_fbx.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

rich_fbx.GeomInput.domain_input.InputType = 'Box'
rich_fbx.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

rich_fbx.Geom.domain.Lower.X = 0.0
rich_fbx.Geom.domain.Lower.Y = 0.0
rich_fbx.Geom.domain.Lower.Z = 0.0

rich_fbx.Geom.domain.Upper.X = 20.0
rich_fbx.Geom.domain.Upper.Y = 20.0
rich_fbx.Geom.domain.Upper.Z = 20.0

rich_fbx.Geom.domain.Patches = 'left right front back bottom top'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

rich_fbx.Geom.Perm.Names = 'domain'
rich_fbx.Geom.domain.Perm.Type = 'Constant'
rich_fbx.Geom.domain.Perm.Value = 1.0

rich_fbx.Perm.TensorType = 'TensorByGeom'

rich_fbx.Geom.Perm.TensorByGeom.Names = 'domain'

rich_fbx.Geom.domain.Perm.TensorValX = 1.0
rich_fbx.Geom.domain.Perm.TensorValY = 1.0
rich_fbx.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

rich_fbx.SpecificStorage.Type = 'Constant'
rich_fbx.SpecificStorage.GeomNames = 'domain'
rich_fbx.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

rich_fbx.Phase.Names = 'water'

rich_fbx.Phase.water.Density.Type = 'Constant'
rich_fbx.Phase.water.Density.Value = 1.0

rich_fbx.Phase.water.Viscosity.Type = 'Constant'
rich_fbx.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

rich_fbx.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

rich_fbx.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

rich_fbx.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

rich_fbx.TimingInfo.BaseUnit = 10.
rich_fbx.TimingInfo.StartCount = 0
rich_fbx.TimingInfo.StartTime = 0.0
rich_fbx.TimingInfo.StopTime = 100.0
rich_fbx.TimingInfo.DumpInterval = 10.0
rich_fbx.TimeStep.Type = 'Constant'
rich_fbx.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

rich_fbx.Geom.Porosity.GeomNames = 'domain'
rich_fbx.Geom.domain.Porosity.Type = 'Constant'
rich_fbx.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

rich_fbx.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

rich_fbx.Phase.RelPerm.Type = 'VanGenuchten'
rich_fbx.Phase.RelPerm.GeomNames = 'domain'
rich_fbx.Geom.domain.RelPerm.Alpha = 2.0
rich_fbx.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

rich_fbx.Phase.Saturation.Type = 'VanGenuchten'
rich_fbx.Phase.Saturation.GeomNames = 'domain'
rich_fbx.Geom.domain.Saturation.Alpha = 2.0
rich_fbx.Geom.domain.Saturation.N = 2.0
rich_fbx.Geom.domain.Saturation.SRes = 0.1
rich_fbx.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# Flow Barrier in X between cells 10 and 11 in all Z
#---------------------------------------------------------

rich_fbx.Solver.Nonlinear.FlowBarrierX = True
rich_fbx.FBx.Type = 'PFBFile'
rich_fbx.Geom.domain.FBx.FileName = 'Flow_Barrier_X.pfb'

## write flow boundary file
FBx_data = np.full((20, 20, 20), 1.0)
# from cell 10 (index 9) to cell 11
# reduction of 1E-3
FBx_data[:, :, 9] = 0.001
write_pfb(get_absolute_path(new_output_dir_name + '/Flow_Barrier_X.pfb'), FBx_data)
rich_fbx.dist(new_output_dir_name + '/Flow_Barrier_X.pfb')

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

rich_fbx.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

rich_fbx.Cycle.Names = 'constant'
rich_fbx.Cycle.constant.Names = 'alltime'
rich_fbx.Cycle.constant.alltime.Length = 1
rich_fbx.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

rich_fbx.BCPressure.PatchNames = 'left right front back bottom top'

rich_fbx.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
rich_fbx.Patch.left.BCPressure.Cycle = 'constant'
rich_fbx.Patch.left.BCPressure.RefGeom = 'domain'
rich_fbx.Patch.left.BCPressure.RefPatch = 'bottom'
rich_fbx.Patch.left.BCPressure.alltime.Value = 11.0

rich_fbx.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
rich_fbx.Patch.right.BCPressure.Cycle = 'constant'
rich_fbx.Patch.right.BCPressure.RefGeom = 'domain'
rich_fbx.Patch.right.BCPressure.RefPatch = 'bottom'
rich_fbx.Patch.right.BCPressure.alltime.Value = 15.0

rich_fbx.Patch.front.BCPressure.Type = 'FluxConst'
rich_fbx.Patch.front.BCPressure.Cycle = 'constant'
rich_fbx.Patch.front.BCPressure.alltime.Value = 0.0

rich_fbx.Patch.back.BCPressure.Type = 'FluxConst'
rich_fbx.Patch.back.BCPressure.Cycle = 'constant'
rich_fbx.Patch.back.BCPressure.alltime.Value = 0.0

rich_fbx.Patch.bottom.BCPressure.Type = 'FluxConst'
rich_fbx.Patch.bottom.BCPressure.Cycle = 'constant'
rich_fbx.Patch.bottom.BCPressure.alltime.Value = 0.0

rich_fbx.Patch.top.BCPressure.Type = 'FluxConst'
rich_fbx.Patch.top.BCPressure.Cycle = 'constant'
rich_fbx.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

rich_fbx.TopoSlopesX.Type = 'Constant'
rich_fbx.TopoSlopesX.GeomNames = 'domain'
rich_fbx.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

rich_fbx.TopoSlopesY.Type = 'Constant'
rich_fbx.TopoSlopesY.GeomNames = 'domain'
rich_fbx.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

rich_fbx.Mannings.Type = 'Constant'
rich_fbx.Mannings.GeomNames = 'domain'
rich_fbx.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

rich_fbx.ICPressure.Type = 'HydroStaticPatch'
rich_fbx.ICPressure.GeomNames = 'domain'
rich_fbx.Geom.domain.ICPressure.Value = 13.0
rich_fbx.Geom.domain.ICPressure.RefGeom = 'domain'
rich_fbx.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

rich_fbx.PhaseSources.water.Type = 'Constant'
rich_fbx.PhaseSources.water.GeomNames = 'domain'
rich_fbx.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

rich_fbx.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

rich_fbx.Solver = 'Richards'
rich_fbx.Solver.MaxIter = 50000

rich_fbx.Solver.Nonlinear.MaxIter = 100
rich_fbx.Solver.Nonlinear.ResidualTol = 1e-6
rich_fbx.Solver.Nonlinear.EtaChoice = 'EtaConstant'
rich_fbx.Solver.Nonlinear.EtaValue = 1e-2
rich_fbx.Solver.Nonlinear.UseJacobian = True

rich_fbx.Solver.Nonlinear.DerivativeEpsilon = 1e-12

rich_fbx.Solver.Linear.KrylovDimension = 100

rich_fbx.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

rich_fbx.run(working_directory=new_output_dir_name)
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
        
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
                
