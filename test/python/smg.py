#------------------------------------------------------------------
#  This runs the basic smg test case based off of default richards
#  This run, as written in this input file, should take
#  3 nonlinear iterations.
#------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file

run_name = "smg"
smg = Run(run_name, __file__)

#------------------------------------------------------------------

smg.FileVersion = 4

smg.Process.Topology.P = 1
smg.Process.Topology.Q = 1
smg.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

smg.ComputationalGrid.Lower.X = -10.0
smg.ComputationalGrid.Lower.Y = 10.0
smg.ComputationalGrid.Lower.Z = 1.0

smg.ComputationalGrid.DX = 8.8888888888888893
smg.ComputationalGrid.DY = 10.666666666666666
smg.ComputationalGrid.DZ = 1.0

smg.ComputationalGrid.NX = 10
smg.ComputationalGrid.NY = 10
smg.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

smg.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

smg.GeomInput.domain_input.InputType = 'Box'
smg.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

smg.Geom.domain.Lower.X = -10.0
smg.Geom.domain.Lower.Y = 10.0
smg.Geom.domain.Lower.Z = 1.0

smg.Geom.domain.Upper.X = 150.0
smg.Geom.domain.Upper.Y = 170.0
smg.Geom.domain.Upper.Z = 9.0

smg.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------

smg.GeomInput.background_input.InputType = 'Box'
smg.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------

smg.Geom.background.Lower.X = -99999999.0
smg.Geom.background.Lower.Y = -99999999.0
smg.Geom.background.Lower.Z = -99999999.0

smg.Geom.background.Upper.X = 99999999.0
smg.Geom.background.Upper.Y = 99999999.0
smg.Geom.background.Upper.Z = 99999999.0

#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------

smg.GeomInput.source_region_input.InputType = 'Box'
smg.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------

smg.Geom.source_region.Lower.X = 65.56
smg.Geom.source_region.Lower.Y = 79.34
smg.Geom.source_region.Lower.Z = 4.5

smg.Geom.source_region.Upper.X = 74.44
smg.Geom.source_region.Upper.Y = 89.99
smg.Geom.source_region.Upper.Z = 5.5

#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------

smg.GeomInput.concen_region_input.InputType = 'Box'
smg.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------

smg.Geom.concen_region.Lower.X = 60.0
smg.Geom.concen_region.Lower.Y = 80.0
smg.Geom.concen_region.Lower.Z = 4.0

smg.Geom.concen_region.Upper.X = 80.0
smg.Geom.concen_region.Upper.Y = 100.0
smg.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

smg.Geom.Perm.Names = 'background'

smg.Geom.background.Perm.Type = 'Constant'
smg.Geom.background.Perm.Value = 4.0

smg.Perm.TensorType = 'TensorByGeom'

smg.Geom.Perm.TensorByGeom.Names = 'background'

smg.Geom.background.Perm.TensorValX = 1.0
smg.Geom.background.Perm.TensorValY = 1.0
smg.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

smg.SpecificStorage.Type = 'Constant'
smg.SpecificStorage.GeomNames = 'domain'
smg.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

smg.Phase.Names = 'water'

smg.Phase.water.Density.Type = 'Constant'
smg.Phase.water.Density.Value = 1.0

smg.Phase.water.Viscosity.Type = 'Constant'
smg.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

smg.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

smg.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

smg.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

smg.TimingInfo.BaseUnit = 1.0
smg.TimingInfo.StartCount = 0
smg.TimingInfo.StartTime = 0.0
smg.TimingInfo.StopTime = 0.010
smg.TimingInfo.DumpInterval = -1
smg.TimeStep.Type = 'Constant'
smg.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

smg.Geom.Porosity.GeomNames = 'background'

smg.Geom.background.Porosity.Type = 'Constant'
smg.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

smg.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

smg.Phase.RelPerm.Type = 'VanGenuchten'
smg.Phase.RelPerm.GeomNames = 'domain'
smg.Geom.domain.RelPerm.Alpha = 0.005
smg.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

smg.Phase.Saturation.Type = 'VanGenuchten'
smg.Phase.Saturation.GeomNames = 'domain'
smg.Geom.domain.Saturation.Alpha = 0.005
smg.Geom.domain.Saturation.N = 2.0
smg.Geom.domain.Saturation.SRes = 0.2
smg.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

smg.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

smg.Cycle.Names = 'constant'
smg.Cycle.constant.Names = 'alltime'
smg.Cycle.constant.alltime.Length = 1
smg.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

smg.BCPressure.PatchNames = 'left right front back bottom top'

smg.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
smg.Patch.left.BCPressure.Cycle = 'constant'
smg.Patch.left.BCPressure.RefGeom = 'domain'
smg.Patch.left.BCPressure.RefPatch = 'bottom'
smg.Patch.left.BCPressure.alltime.Value = 5.0

smg.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
smg.Patch.right.BCPressure.Cycle = 'constant'
smg.Patch.right.BCPressure.RefGeom = 'domain'
smg.Patch.right.BCPressure.RefPatch = 'bottom'
smg.Patch.right.BCPressure.alltime.Value = 3.0

smg.Patch.front.BCPressure.Type = 'FluxConst'
smg.Patch.front.BCPressure.Cycle = 'constant'
smg.Patch.front.BCPressure.alltime.Value = 0.0

smg.Patch.back.BCPressure.Type = 'FluxConst'
smg.Patch.back.BCPressure.Cycle = 'constant'
smg.Patch.back.BCPressure.alltime.Value = 0.0

smg.Patch.bottom.BCPressure.Type = 'FluxConst'
smg.Patch.bottom.BCPressure.Cycle = 'constant'
smg.Patch.bottom.BCPressure.alltime.Value = 0.0

smg.Patch.top.BCPressure.Type = 'FluxConst'
smg.Patch.top.BCPressure.Cycle = 'constant'
smg.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

smg.TopoSlopesX.Type = 'Constant'
smg.TopoSlopesX.GeomNames = 'domain'
smg.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

smg.TopoSlopesY.Type = 'Constant'
smg.TopoSlopesY.GeomNames = 'domain'
smg.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

smg.Mannings.Type = 'Constant'
smg.Mannings.GeomNames = 'domain'
smg.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

smg.ICPressure.Type = 'HydroStaticPatch'
smg.ICPressure.GeomNames = 'domain'
smg.Geom.domain.ICPressure.Value = 3.0
smg.Geom.domain.ICPressure.RefGeom = 'domain'
smg.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

smg.PhaseSources.water.Type = 'Constant'
smg.PhaseSources.water.GeomNames = 'background'
smg.PhaseSources.water.Geom.background.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

smg.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

smg.Solver = 'Richards'
smg.Solver.MaxIter = 5

smg.Solver.Nonlinear.MaxIter = 10
smg.Solver.Nonlinear.ResidualTol = 1e-9
smg.Solver.Nonlinear.EtaChoice = 'EtaConstant'
smg.Solver.Nonlinear.EtaValue = 1e-5
smg.Solver.Nonlinear.UseJacobian = True
smg.Solver.Nonlinear.DerivativeEpsilon = 1e-2

smg.Solver.Linear.KrylovDimension = 10

smg.Solver.Linear.Preconditioner = 'SMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path('test_output/smg')
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)
smg.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in {test_file}"):
        passed = False

for i in range(6):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Pressure for timestep {timestep}"):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Saturation for timestep {timestep}"):
        passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
