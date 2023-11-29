#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "octree-large-domain"
octree = Run(run_name, __file__)

#-----------------------------------------------------------------------------

octree.FileVersion = 4

octree.Process.Topology.P = 1
octree.Process.Topology.Q = 1
octree.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------

octree.ComputationalGrid.Lower.X = -10.0
octree.ComputationalGrid.Lower.Y = 10.0
octree.ComputationalGrid.Lower.Z = 1.0

octree.ComputationalGrid.DX = 20.0
octree.ComputationalGrid.DY = 20.0
octree.ComputationalGrid.DZ = 1.0

octree.ComputationalGrid.NX = 10
octree.ComputationalGrid.NY = 10
octree.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

octree.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

octree.GeomInput.domain_input.InputType = 'Box'
octree.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

octree.Geom.domain.Lower.X = -10.0
octree.Geom.domain.Lower.Y = 10.0
octree.Geom.domain.Lower.Z = 1.0

octree.Geom.domain.Upper.X = 150.0
octree.Geom.domain.Upper.Y = 170.0
octree.Geom.domain.Upper.Z = 9.0

octree.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------

octree.GeomInput.background_input.InputType = 'Box'
octree.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------

octree.Geom.background.Lower.X = -99999999.0
octree.Geom.background.Lower.Y = -99999999.0
octree.Geom.background.Lower.Z = -99999999.0

octree.Geom.background.Upper.X = 99999999.0
octree.Geom.background.Upper.Y = 99999999.0
octree.Geom.background.Upper.Z = 99999999.0

#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------

octree.GeomInput.source_region_input.InputType = 'Box'
octree.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------

octree.Geom.source_region.Lower.X = 65.56
octree.Geom.source_region.Lower.Y = 79.34
octree.Geom.source_region.Lower.Z = 4.5

octree.Geom.source_region.Upper.X = 74.44
octree.Geom.source_region.Upper.Y = 89.99
octree.Geom.source_region.Upper.Z = 5.5

#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------

octree.GeomInput.concen_region_input.InputType = 'Box'
octree.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------

octree.Geom.concen_region.Lower.X = 60.0
octree.Geom.concen_region.Lower.Y = 80.0
octree.Geom.concen_region.Lower.Z = 4.0

octree.Geom.concen_region.Upper.X = 80.0
octree.Geom.concen_region.Upper.Y = 100.0
octree.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

octree.Geom.Perm.Names = 'background'

octree.Geom.background.Perm.Type = 'Constant'
octree.Geom.background.Perm.Value = 4.0

octree.Perm.TensorType = 'TensorByGeom'

octree.Geom.Perm.TensorByGeom.Names = 'background'

octree.Geom.background.Perm.TensorValX = 1.0
octree.Geom.background.Perm.TensorValY = 1.0
octree.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

octree.SpecificStorage.Type = 'Constant'
octree.SpecificStorage.GeomNames = 'domain'
octree.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

octree.Phase.Names = 'water'

octree.Phase.water.Density.Type = 'Constant'
octree.Phase.water.Density.Value = 1.0

octree.Phase.water.Viscosity.Type = 'Constant'
octree.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

octree.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

octree.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

octree.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

octree.TimingInfo.BaseUnit = 1.0
octree.TimingInfo.StartCount = 0
octree.TimingInfo.StartTime = 0.0
octree.TimingInfo.StopTime = 0.010
octree.TimingInfo.DumpInterval = -1
octree.TimeStep.Type = 'Constant'
octree.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

octree.Geom.Porosity.GeomNames = 'background'

octree.Geom.background.Porosity.Type = 'Constant'
octree.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

octree.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

octree.Phase.RelPerm.Type = 'VanGenuchten'
octree.Phase.RelPerm.GeomNames = 'domain'
octree.Geom.domain.RelPerm.Alpha = 0.005
octree.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

octree.Phase.Saturation.Type = 'VanGenuchten'
octree.Phase.Saturation.GeomNames = 'domain'
octree.Geom.domain.Saturation.Alpha = 0.005
octree.Geom.domain.Saturation.N = 2.0
octree.Geom.domain.Saturation.SRes = 0.2
octree.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

octree.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

octree.Cycle.Names = 'constant'
octree.Cycle.constant.Names = 'alltime'
octree.Cycle.constant.alltime.Length = 1
octree.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

octree.BCPressure.PatchNames = 'left right front back bottom top'

octree.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
octree.Patch.left.BCPressure.Cycle = 'constant'
octree.Patch.left.BCPressure.RefGeom = 'domain'
octree.Patch.left.BCPressure.RefPatch = 'bottom'
octree.Patch.left.BCPressure.alltime.Value = 5.0

octree.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
octree.Patch.right.BCPressure.Cycle = 'constant'
octree.Patch.right.BCPressure.RefGeom = 'domain'
octree.Patch.right.BCPressure.RefPatch = 'bottom'
octree.Patch.right.BCPressure.alltime.Value = 3.0

octree.Patch.front.BCPressure.Type = 'FluxConst'
octree.Patch.front.BCPressure.Cycle = 'constant'
octree.Patch.front.BCPressure.alltime.Value = 0.0

octree.Patch.back.BCPressure.Type = 'FluxConst'
octree.Patch.back.BCPressure.Cycle = 'constant'
octree.Patch.back.BCPressure.alltime.Value = 0.0

octree.Patch.bottom.BCPressure.Type = 'FluxConst'
octree.Patch.bottom.BCPressure.Cycle = 'constant'
octree.Patch.bottom.BCPressure.alltime.Value = 0.0

octree.Patch.top.BCPressure.Type = 'FluxConst'
octree.Patch.top.BCPressure.Cycle = 'constant'
octree.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

octree.TopoSlopesX.Type = 'Constant'
octree.TopoSlopesX.GeomNames = 'domain'

octree.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

octree.TopoSlopesY.Type = 'Constant'
octree.TopoSlopesY.GeomNames = 'domain'

octree.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

octree.Mannings.Type = 'Constant'
octree.Mannings.GeomNames = 'domain'
octree.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

octree.ICPressure.Type = 'HydroStaticPatch'
octree.ICPressure.GeomNames = 'domain'
octree.Geom.domain.ICPressure.Value = 3.0
octree.Geom.domain.ICPressure.RefGeom = 'domain'
octree.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

octree.PhaseSources.water.Type = 'Constant'
octree.PhaseSources.water.GeomNames = 'background'
octree.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

octree.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

octree.Solver = 'Richards'
octree.Solver.MaxIter = 5

octree.Solver.Nonlinear.MaxIter = 10
octree.Solver.Nonlinear.ResidualTol = 1e-9
octree.Solver.Nonlinear.EtaChoice = 'EtaConstant'
octree.Solver.Nonlinear.EtaValue = 1e-5
octree.Solver.Nonlinear.UseJacobian = True
octree.Solver.Nonlinear.DerivativeEpsilon = 1e-2

octree.Solver.Linear.KrylovDimension = 10

octree.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
new_output_dir_name = get_absolute_path('test_output/octree-large-domain')
mkdir(new_output_dir_name)
correct_output_dir_name = get_absolute_path('../correct_output')
octree.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in {test_file}", sig_digits=4):
        passed = False

for i in range(6):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in Pressure for timestep {timestep}", sig_digits=4):
        passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
