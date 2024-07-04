#------------------------------------------------------------------
# This runs a test case with the Richards' solver
# with simple flow domains, like a wall or a fault.
#------------------------------------------------------------------

import sys
from parflow import Run, write_pfb
from parflow.tools.fs import get_absolute_path, mkdir, chdir
from parflow.tools.compare import pf_test_file
import numpy as np

run_name = "richards_FBy"
rich_fby = Run(run_name, __file__)

#---------------------------------------------------------
# Creating and navigating to output directory
#---------------------------------------------------------

correct_output_dir_name = get_absolute_path('../correct_output')
new_output_dir_name = get_absolute_path('test_output/richards_fby')
mkdir(new_output_dir_name)

#------------------------------------------------------------------

rich_fby.FileVersion = 4

rich_fby.Process.Topology.P = 1
rich_fby.Process.Topology.Q = 1
rich_fby.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

rich_fby.ComputationalGrid.Lower.X = 0.0
rich_fby.ComputationalGrid.Lower.Y = 0.0
rich_fby.ComputationalGrid.Lower.Z = 0.0

rich_fby.ComputationalGrid.DX = 1.0
rich_fby.ComputationalGrid.DY = 1.0
rich_fby.ComputationalGrid.DZ = 1.0

rich_fby.ComputationalGrid.NX = 20
rich_fby.ComputationalGrid.NY = 20
rich_fby.ComputationalGrid.NZ = 20

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

rich_fby.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

rich_fby.GeomInput.domain_input.InputType = 'Box'
rich_fby.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

rich_fby.Geom.domain.Lower.X = 0.0
rich_fby.Geom.domain.Lower.Y = 0.0
rich_fby.Geom.domain.Lower.Z = 0.0

rich_fby.Geom.domain.Upper.X = 20.0
rich_fby.Geom.domain.Upper.Y = 20.0
rich_fby.Geom.domain.Upper.Z = 20.0

rich_fby.Geom.domain.Patches = 'left right front back bottom top'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

rich_fby.Geom.Perm.Names = 'domain'
rich_fby.Geom.domain.Perm.Type = 'Constant'
rich_fby.Geom.domain.Perm.Value = 1.0

rich_fby.Perm.TensorType = 'TensorByGeom'

rich_fby.Geom.Perm.TensorByGeom.Names = 'domain'

rich_fby.Geom.domain.Perm.TensorValX = 1.0
rich_fby.Geom.domain.Perm.TensorValY = 1.0
rich_fby.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

rich_fby.SpecificStorage.Type = 'Constant'
rich_fby.SpecificStorage.GeomNames = 'domain'
rich_fby.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

rich_fby.Phase.Names = 'water'

rich_fby.Phase.water.Density.Type = 'Constant'
rich_fby.Phase.water.Density.Value = 1.0

rich_fby.Phase.water.Viscosity.Type = 'Constant'
rich_fby.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

rich_fby.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

rich_fby.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

rich_fby.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

rich_fby.TimingInfo.BaseUnit = 10.
rich_fby.TimingInfo.StartCount = 0
rich_fby.TimingInfo.StartTime = 0.0
rich_fby.TimingInfo.StopTime = 100.0
rich_fby.TimingInfo.DumpInterval = 10.0
rich_fby.TimeStep.Type = 'Constant'
rich_fby.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

rich_fby.Geom.Porosity.GeomNames = 'domain'
rich_fby.Geom.domain.Porosity.Type = 'Constant'
rich_fby.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

rich_fby.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

rich_fby.Phase.RelPerm.Type = 'VanGenuchten'
rich_fby.Phase.RelPerm.GeomNames = 'domain'
rich_fby.Geom.domain.RelPerm.Alpha = 2.0
rich_fby.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

rich_fby.Phase.Saturation.Type = 'VanGenuchten'
rich_fby.Phase.Saturation.GeomNames = 'domain'
rich_fby.Geom.domain.Saturation.Alpha = 2.0
rich_fby.Geom.domain.Saturation.N = 2.0
rich_fby.Geom.domain.Saturation.SRes = 0.1
rich_fby.Geom.domain.Saturation.SSat = 1.0

#---------------------------------------------------------
# Flow Barrier in Y between cells 10 and 11 in all Z
#---------------------------------------------------------

rich_fby.Solver.Nonlinear.FlowBarrierY = True
rich_fby.FBy.Type = 'PFBFile'
rich_fby.Geom.domain.FBy.FileName = 'Flow_Barrier_Y.pfb'

## write flow barrier file
FBy_data = np.full((20, 20, 20), 1.0)
# from cell 10 (index 9) to cell 11
# reduction of 1E-3
FBy_data[:, 9, :] = 0.001
write_pfb(get_absolute_path(new_output_dir_name + '/Flow_Barrier_Y.pfb'), FBy_data)
rich_fby.dist(new_output_dir_name + '/Flow_Barrier_Y.pfb')

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

rich_fby.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

rich_fby.Cycle.Names = 'constant'
rich_fby.Cycle.constant.Names = 'alltime'
rich_fby.Cycle.constant.alltime.Length = 1
rich_fby.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

rich_fby.BCPressure.PatchNames = 'left right front back bottom top'

rich_fby.Patch.front.BCPressure.Type = 'DirEquilRefPatch'
rich_fby.Patch.front.BCPressure.Cycle = 'constant'
rich_fby.Patch.front.BCPressure.RefGeom = 'domain'
rich_fby.Patch.front.BCPressure.RefPatch = 'bottom'
rich_fby.Patch.front.BCPressure.alltime.Value = 11.0

rich_fby.Patch.back.BCPressure.Type = 'DirEquilRefPatch'
rich_fby.Patch.back.BCPressure.Cycle = 'constant'
rich_fby.Patch.back.BCPressure.RefGeom = 'domain'
rich_fby.Patch.back.BCPressure.RefPatch = 'bottom'
rich_fby.Patch.back.BCPressure.alltime.Value = 15.0

rich_fby.Patch.left.BCPressure.Type = 'FluxConst'
rich_fby.Patch.left.BCPressure.Cycle = 'constant'
rich_fby.Patch.left.BCPressure.alltime.Value = 0.0

rich_fby.Patch.right.BCPressure.Type = 'FluxConst'
rich_fby.Patch.right.BCPressure.Cycle = 'constant'
rich_fby.Patch.right.BCPressure.alltime.Value = 0.0

rich_fby.Patch.bottom.BCPressure.Type = 'FluxConst'
rich_fby.Patch.bottom.BCPressure.Cycle = 'constant'
rich_fby.Patch.bottom.BCPressure.alltime.Value = 0.0

rich_fby.Patch.top.BCPressure.Type = 'FluxConst'
rich_fby.Patch.top.BCPressure.Cycle = 'constant'
rich_fby.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

rich_fby.TopoSlopesX.Type = 'Constant'
rich_fby.TopoSlopesX.GeomNames = 'domain'
rich_fby.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

rich_fby.TopoSlopesY.Type = 'Constant'
rich_fby.TopoSlopesY.GeomNames = 'domain'
rich_fby.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

rich_fby.Mannings.Type = 'Constant'
rich_fby.Mannings.GeomNames = 'domain'
rich_fby.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

rich_fby.ICPressure.Type = 'HydroStaticPatch'
rich_fby.ICPressure.GeomNames = 'domain'
rich_fby.Geom.domain.ICPressure.Value = 13.0
rich_fby.Geom.domain.ICPressure.RefGeom = 'domain'
rich_fby.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

rich_fby.PhaseSources.water.Type = 'Constant'
rich_fby.PhaseSources.water.GeomNames = 'domain'
rich_fby.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

rich_fby.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

rich_fby.Solver = 'Richards'
rich_fby.Solver.MaxIter = 50000

rich_fby.Solver.Nonlinear.MaxIter = 100
rich_fby.Solver.Nonlinear.ResidualTol = 1e-6
rich_fby.Solver.Nonlinear.EtaChoice = 'EtaConstant'
rich_fby.Solver.Nonlinear.EtaValue = 1e-2
rich_fby.Solver.Nonlinear.UseJacobian = True

rich_fby.Solver.Nonlinear.DerivativeEpsilon = 1e-12

rich_fby.Solver.Linear.KrylovDimension = 100

rich_fby.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

rich_fby.run(working_directory=new_output_dir_name)
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
                
