#-----------------------------------------------------------------------------
#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.
#-----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

run_name = "default_richards_wells"
drich = Run(run_name, __file__)
#---------------------------------------------------------
drich.FileVersion = 4

drich.Process.Topology.P = 1
drich.Process.Topology.Q = 1
drich.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

drich.ComputationalGrid.Lower.X = -10.0
drich.ComputationalGrid.Lower.Y = 10.0
drich.ComputationalGrid.Lower.Z = 1.0

drich.ComputationalGrid.DX = 8.8888888888888893
drich.ComputationalGrid.DY = 10.666666666666666
drich.ComputationalGrid.DZ = 1.0

drich.ComputationalGrid.NX = 10
drich.ComputationalGrid.NY = 10
drich.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

drich.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

drich.GeomInput.domain_input.InputType = 'Box'
drich.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

drich.Geom.domain.Lower.X = -10.0
drich.Geom.domain.Lower.Y = 10.0
drich.Geom.domain.Lower.Z = 1.0

drich.Geom.domain.Upper.X = 150.0
drich.Geom.domain.Upper.Y = 170.0
drich.Geom.domain.Upper.Z = 9.0

drich.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------

drich.GeomInput.background_input.InputType = 'Box'
drich.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------

drich.Geom.background.Lower.X = -99999999.0
drich.Geom.background.Lower.Y = -99999999.0
drich.Geom.background.Lower.Z = -99999999.0

drich.Geom.background.Upper.X = 99999999.0
drich.Geom.background.Upper.Y = 99999999.0
drich.Geom.background.Upper.Z = 99999999.0

#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------

drich.GeomInput.source_region_input.InputType = 'Box'
drich.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------

drich.Geom.source_region.Lower.X = 65.56
drich.Geom.source_region.Lower.Y = 79.34
drich.Geom.source_region.Lower.Z = 4.5

drich.Geom.source_region.Upper.X = 74.44
drich.Geom.source_region.Upper.Y = 89.99
drich.Geom.source_region.Upper.Z = 5.5

#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------

drich.GeomInput.concen_region_input.InputType = 'Box'
drich.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------

drich.Geom.concen_region.Lower.X = 60.0
drich.Geom.concen_region.Lower.Y = 80.0
drich.Geom.concen_region.Lower.Z = 4.0

drich.Geom.concen_region.Upper.X = 80.0
drich.Geom.concen_region.Upper.Y = 100.0
drich.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

drich.Geom.Perm.Names = 'background'

drich.Geom.background.Perm.Type = 'Constant'
drich.Geom.background.Perm.Value = 4.0

drich.Perm.TensorType = 'TensorByGeom'

drich.Geom.Perm.TensorByGeom.Names = 'background'

drich.Geom.background.Perm.TensorValX = 1.0
drich.Geom.background.Perm.TensorValY = 1.0
drich.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

drich.SpecificStorage.Type = 'Constant'
drich.SpecificStorage.GeomNames = 'domain'
drich.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

drich.Phase.Names = 'water'

drich.Phase.water.Density.Type = 'Constant'
drich.Phase.water.Density.Value = 1.0

drich.Phase.water.Viscosity.Type = 'Constant'
drich.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

drich.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

drich.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

drich.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

drich.TimingInfo.BaseUnit = 1.0
drich.TimingInfo.StartCount = 0
drich.TimingInfo.StartTime = 0.0
drich.TimingInfo.StopTime = 0.010
drich.TimingInfo.DumpInterval = -1
drich.TimeStep.Type = 'Constant'
drich.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

drich.Geom.Porosity.GeomNames = 'background'

drich.Geom.background.Porosity.Type = 'Constant'
drich.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
drich.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

drich.Phase.RelPerm.Type = 'VanGenuchten'
drich.Phase.RelPerm.GeomNames = 'domain'
drich.Geom.domain.RelPerm.Alpha = 0.005
drich.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

drich.Phase.Saturation.Type = 'VanGenuchten'
drich.Phase.Saturation.GeomNames = 'domain'
drich.Geom.domain.Saturation.Alpha = 0.005
drich.Geom.domain.Saturation.N = 2.0
drich.Geom.domain.Saturation.SRes = 0.2
drich.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

drich.Cycle.Names = 'constant'
drich.Cycle.constant.Names = 'alltime'
drich.Cycle.constant.alltime.Length = 1
drich.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

drich.Wells.Names = 'pumping_well'
drich.Wells.pumping_well.InputType = 'Vertical'
drich.Wells.pumping_well.Action = 'Extraction'
drich.Wells.pumping_well.Type = 'Pressure'
drich.Wells.pumping_well.X = 0
drich.Wells.pumping_well.Y = 80
drich.Wells.pumping_well.ZUpper = 3.0
drich.Wells.pumping_well.ZLower = 2.00
drich.Wells.pumping_well.Method = 'Standard'
drich.Wells.pumping_well.Cycle = 'constant'
drich.Wells.pumping_well.alltime.Pressure.Value = 0.5
drich.Wells.pumping_well.alltime.Saturation.water.Value = 1.0

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

drich.BCPressure.PatchNames = 'left right front back bottom top'

drich.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
drich.Patch.left.BCPressure.Cycle = 'constant'
drich.Patch.left.BCPressure.RefGeom = 'domain'
drich.Patch.left.BCPressure.RefPatch = 'bottom'
drich.Patch.left.BCPressure.alltime.Value = 5.0

drich.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
drich.Patch.right.BCPressure.Cycle = 'constant'
drich.Patch.right.BCPressure.RefGeom = 'domain'
drich.Patch.right.BCPressure.RefPatch = 'bottom'
drich.Patch.right.BCPressure.alltime.Value = 5.0

drich.Patch.front.BCPressure.Type = 'FluxConst'
drich.Patch.front.BCPressure.Cycle = 'constant'
drich.Patch.front.BCPressure.alltime.Value = 0.0

drich.Patch.back.BCPressure.Type = 'FluxConst'
drich.Patch.back.BCPressure.Cycle = 'constant'
drich.Patch.back.BCPressure.alltime.Value = 0.0

drich.Patch.bottom.BCPressure.Type = 'FluxConst'
drich.Patch.bottom.BCPressure.Cycle = 'constant'
drich.Patch.bottom.BCPressure.alltime.Value = 0.0

drich.Patch.top.BCPressure.Type = 'FluxConst'
drich.Patch.top.BCPressure.Cycle = 'constant'
drich.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

drich.TopoSlopesX.Type = 'Constant'
drich.TopoSlopesX.GeomNames = 'domain'

drich.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

drich.TopoSlopesY.Type = 'Constant'
drich.TopoSlopesY.GeomNames = 'domain'
drich.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

drich.Mannings.Type = 'Constant'
drich.Mannings.GeomNames = 'domain'
drich.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

drich.ICPressure.Type = 'HydroStaticPatch'
drich.ICPressure.GeomNames = 'domain'
drich.Geom.domain.ICPressure.Value = 5.0
drich.Geom.domain.ICPressure.RefGeom = 'domain'
drich.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

drich.PhaseSources.water.Type = 'Constant'
drich.PhaseSources.water.GeomNames = 'background'
drich.PhaseSources.water.Geom.background.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

drich.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

drich.Solver = 'Richards'
drich.Solver.MaxIter = 5

drich.Solver.Nonlinear.MaxIter = 10
drich.Solver.Nonlinear.ResidualTol = 1e-9
drich.Solver.Nonlinear.EtaChoice = 'EtaConstant'
drich.Solver.Nonlinear.EtaValue = 1e-5
drich.Solver.Nonlinear.UseJacobian = True
drich.Solver.Nonlinear.DerivativeEpsilon = 1e-2

drich.Solver.Linear.KrylovDimension = 10

drich.Solver.Linear.Preconditioner = 'MGSemi'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
new_output_dir_name = get_absolute_path('test_output/' + run_name)
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)
drich.run(working_directory=new_output_dir_name)

passed = True

filename = f"/{run_name}.out.perm_x.pfb"
if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, "Max difference in perm_x"):
    passed = False
filename = f"/{run_name}.out.perm_y.pfb"
if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, "Max difference in perm_y"):
    passed = False
filename = f"/{run_name}.out.perm_z.pfb"
if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, "Max difference in perm_z"):
    passed = False

timesteps = ["00000", "00001", "00002", "00003", "00004", "00005"]
for i in timesteps:
    filename = f"/{run_name}.out.press.{i}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Pressure for timestep {i}"):
        passed = False
    filename = f"/{run_name}.out.satur.{i}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Saturation for timestep {i}"):
        passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
