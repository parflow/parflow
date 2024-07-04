#-----------------------------------------------------------------------------
#  This is a 2D crater problem w/ time varying input and topography
#-----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs

run_name = "crater2D_vangtable_spline"
crater = Run(run_name, __file__)

# ---------------------------------------------------------
# Copy testing data in test directory
# ---------------------------------------------------------

dir_name = get_absolute_path('test_output/crater_vs')
mkdir(dir_name)

cp('$PF_SRC/test/input/crater2D.pfsol', dir_name)

#---------------------------------------------------------
# Controls for the VanG curves used later.
#---------------------------------------------------------

VG_points = 20000
VG_alpha = 1.0
VG_N = 2.0

#---------------------------------------------------------

crater.FileVersion = 4

crater.Process.Topology.P = 1
crater.Process.Topology.Q = 1
crater.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

crater.ComputationalGrid.Lower.X = 0.0
crater.ComputationalGrid.Lower.Y = 0.0
crater.ComputationalGrid.Lower.Z = 0.0

crater.ComputationalGrid.NX = 100
crater.ComputationalGrid.NY = 1
crater.ComputationalGrid.NZ = 100

UpperX = 400
UpperY = 1.0
UpperZ = 200

LowerX = crater.ComputationalGrid.Lower.X
LowerY = crater.ComputationalGrid.Lower.Y
LowerZ = crater.ComputationalGrid.Lower.Z

NX = crater.ComputationalGrid.NX
NY = crater.ComputationalGrid.NY
NZ = crater.ComputationalGrid.NZ

crater.ComputationalGrid.DX = (UpperX - LowerX) / NX
crater.ComputationalGrid.DY = (UpperY - LowerY) / NY
crater.ComputationalGrid.DZ = (UpperZ - LowerZ) / NZ

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

Zones = "zone1 zone2 zone3above4 zone3left4 zone3right4 zone3below4 zone4"

crater.GeomInput.Names = f'solidinput {Zones} background'

crater.GeomInput.solidinput.InputType = 'SolidFile'
crater.GeomInput.solidinput.GeomNames = 'domain'
crater.GeomInput.solidinput.FileName = 'crater2D.pfsol'

crater.GeomInput.zone1.InputType = 'Box'
crater.GeomInput.zone1.GeomName = 'zone1'

crater.Geom.zone1.Lower.X = 0.0
crater.Geom.zone1.Lower.Y = 0.0
crater.Geom.zone1.Lower.Z = 0.0
crater.Geom.zone1.Upper.X = 400.0
crater.Geom.zone1.Upper.Y = 1.0
crater.Geom.zone1.Upper.Z = 200.0

crater.GeomInput.zone2.InputType = 'Box'
crater.GeomInput.zone2.GeomName = 'zone2'

crater.Geom.zone2.Lower.X = 0.0
crater.Geom.zone2.Lower.Y = 0.0
crater.Geom.zone2.Lower.Z = 60.0
crater.Geom.zone2.Upper.X = 200.0
crater.Geom.zone2.Upper.Y = 1.0
crater.Geom.zone2.Upper.Z = 80.0

crater.GeomInput.zone3above4.InputType = 'Box'
crater.GeomInput.zone3above4.GeomName = 'zone3above4'

crater.Geom.zone3above4.Lower.X = 0.0
crater.Geom.zone3above4.Lower.Y = 0.0
crater.Geom.zone3above4.Lower.Z = 180.0
crater.Geom.zone3above4.Upper.X = 200.0
crater.Geom.zone3above4.Upper.Y = 1.0
crater.Geom.zone3above4.Upper.Z = 200.0

crater.GeomInput.zone3left4.InputType = 'Box'
crater.GeomInput.zone3left4.GeomName = 'zone3left4'

crater.Geom.zone3left4.Lower.X = 0.0
crater.Geom.zone3left4.Lower.Y = 0.0
crater.Geom.zone3left4.Lower.Z = 190.0
crater.Geom.zone3left4.Upper.X = 100.0
crater.Geom.zone3left4.Upper.Y = 1.0
crater.Geom.zone3left4.Upper.Z = 200.0

crater.GeomInput.zone3right4.InputType = 'Box'
crater.GeomInput.zone3right4.GeomName = 'zone3right4'

crater.Geom.zone3right4.Lower.X = 30.0
crater.Geom.zone3right4.Lower.Y = 0.0
crater.Geom.zone3right4.Lower.Z = 90.0
crater.Geom.zone3right4.Upper.X = 80.0
crater.Geom.zone3right4.Upper.Y = 1.0
crater.Geom.zone3right4.Upper.Z = 100.0

crater.GeomInput.zone3below4.InputType = 'Box'
crater.GeomInput.zone3below4.GeomName = 'zone3below4'

crater.Geom.zone3below4.Lower.X = 0.0
crater.Geom.zone3below4.Lower.Y = 0.0
crater.Geom.zone3below4.Lower.Z = 0.0
crater.Geom.zone3below4.Upper.X = 400.0
crater.Geom.zone3below4.Upper.Y = 1.0
crater.Geom.zone3below4.Upper.Z = 20.0

crater.GeomInput.zone4.InputType = 'Box'
crater.GeomInput.zone4.GeomName = 'zone4'

crater.Geom.zone4.Lower.X = 0.0
crater.Geom.zone4.Lower.Y = 0.0
crater.Geom.zone4.Lower.Z = 100.0
crater.Geom.zone4.Upper.X = 300.0
crater.Geom.zone4.Upper.Y = 1.0
crater.Geom.zone4.Upper.Z = 150.0

crater.GeomInput.background.InputType = 'Box'
crater.GeomInput.background.GeomName = 'background'

crater.Geom.background.Lower.X = -99999999.0
crater.Geom.background.Lower.Y = -99999999.0
crater.Geom.background.Lower.Z = -99999999.0
crater.Geom.background.Upper.X = 99999999.0
crater.Geom.background.Upper.Y = 99999999.0
crater.Geom.background.Upper.Z = 99999999.0

crater.Geom.domain.Patches = 'infiltration z_upper x_lower y_lower \
    x_upper y_upper z_lower'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

crater.Geom.Perm.Names = Zones

crater.Geom.zone1.Perm.Type = 'Constant'
crater.Geom.zone1.Perm.Value = 9.1496

crater.Geom.zone2.Perm.Type = 'Constant'
crater.Geom.zone2.Perm.Value = 5.4427

crater.Geom.zone3above4.Perm.Type = 'Constant'
crater.Geom.zone3above4.Perm.Value = 4.8033

crater.Geom.zone3left4.Perm.Type = 'Constant'
crater.Geom.zone3left4.Perm.Value = 4.8033

crater.Geom.zone3right4.Perm.Type = 'Constant'
crater.Geom.zone3right4.Perm.Value = 4.8033

crater.Geom.zone3below4.Perm.Type = 'Constant'
crater.Geom.zone3below4.Perm.Value = 4.8033

crater.Geom.zone4.Perm.Type = 'Constant'
crater.Geom.zone4.Perm.Value = .48033

crater.Perm.TensorType = 'TensorByGeom'

crater.Geom.Perm.TensorByGeom.Names = 'background'

crater.Geom.background.Perm.TensorValX = 1.0
crater.Geom.background.Perm.TensorValY = 1.0
crater.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

crater.SpecificStorage.Type = 'Constant'
crater.SpecificStorage.GeomNames = 'domain'
crater.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

crater.Phase.Names = 'water'

crater.Phase.water.Density.Type = 'Constant'
crater.Phase.water.Density.Value = 1.0

crater.Phase.water.Viscosity.Type = 'Constant'
crater.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

crater.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

crater.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

crater.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

crater.TimingInfo.BaseUnit = 1.0
crater.TimingInfo.StartCount = 0
crater.TimingInfo.StartTime = 0.0
crater.TimingInfo.StopTime = 20.0
crater.TimingInfo.DumpInterval = 10.0
crater.TimeStep.Type = 'Constant'
crater.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

crater.Geom.Porosity.GeomNames = Zones

crater.Geom.zone1.Porosity.Type = 'Constant'
crater.Geom.zone1.Porosity.Value = 0.3680

crater.Geom.zone2.Porosity.Type = 'Constant'
crater.Geom.zone2.Porosity.Value = 0.3510

crater.Geom.zone3above4.Porosity.Type = 'Constant'
crater.Geom.zone3above4.Porosity.Value = 0.3250

crater.Geom.zone3left4.Porosity.Type = 'Constant'
crater.Geom.zone3left4.Porosity.Value = 0.3250

crater.Geom.zone3right4.Porosity.Type = 'Constant'
crater.Geom.zone3right4.Porosity.Value = 0.3250

crater.Geom.zone3below4.Porosity.Type = 'Constant'
crater.Geom.zone3below4.Porosity.Value = 0.3250

crater.Geom.zone4.Porosity.Type = 'Constant'
crater.Geom.zone4.Porosity.Value = 0.3250

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

crater.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

crater.Phase.RelPerm.Type = 'VanGenuchten'
crater.Phase.RelPerm.GeomNames = Zones

crater.Geom.zone1.RelPerm.Alpha = VG_alpha
crater.Geom.zone1.RelPerm.N = VG_N
crater.Geom.zone1.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone1.RelPerm.MinPressureHead = -300

crater.Geom.zone2.RelPerm.Alpha = VG_alpha
crater.Geom.zone2.RelPerm.N = VG_N
crater.Geom.zone2.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone2.RelPerm.MinPressureHead = -300

crater.Geom.zone3above4.RelPerm.Alpha = VG_alpha
crater.Geom.zone3above4.RelPerm.N = VG_N
crater.Geom.zone3above4.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone3above4.RelPerm.MinPressureHead = -300

crater.Geom.zone3left4.RelPerm.Alpha = VG_alpha
crater.Geom.zone3left4.RelPerm.N = VG_N
crater.Geom.zone3left4.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone3left4.RelPerm.MinPressureHead = -300

crater.Geom.zone3right4.RelPerm.Alpha = VG_alpha
crater.Geom.zone3right4.RelPerm.N = VG_N
crater.Geom.zone3right4.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone3right4.RelPerm.MinPressureHead = -300

crater.Geom.zone3below4.RelPerm.Alpha = VG_alpha
crater.Geom.zone3below4.RelPerm.N = VG_N
crater.Geom.zone3below4.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone3below4.RelPerm.MinPressureHead = -300

crater.Geom.zone4.RelPerm.Alpha = VG_alpha
crater.Geom.zone4.RelPerm.N = VG_N
crater.Geom.zone4.RelPerm.NumSamplePoints = VG_points
crater.Geom.zone4.RelPerm.MinPressureHead = -300

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

crater.Phase.Saturation.Type = 'VanGenuchten'
crater.Phase.Saturation.GeomNames = Zones

crater.Geom.zone1.Saturation.Alpha = VG_alpha
crater.Geom.zone1.Saturation.N = VG_N
crater.Geom.zone1.Saturation.SRes = 0.2771
crater.Geom.zone1.Saturation.SSat = 1.0

crater.Geom.zone2.Saturation.Alpha = VG_alpha
crater.Geom.zone2.Saturation.N = VG_N
crater.Geom.zone2.Saturation.SRes = 0.2806
crater.Geom.zone2.Saturation.SSat = 1.0

crater.Geom.zone3above4.Saturation.Alpha = VG_alpha
crater.Geom.zone3above4.Saturation.N = VG_N
crater.Geom.zone3above4.Saturation.SRes = 0.2643
crater.Geom.zone3above4.Saturation.SSat = 1.0

crater.Geom.zone3left4.Saturation.Alpha = VG_alpha
crater.Geom.zone3left4.Saturation.N = VG_N
crater.Geom.zone3left4.Saturation.SRes = 0.2643
crater.Geom.zone3left4.Saturation.SSat = 1.0

crater.Geom.zone3right4.Saturation.Alpha = VG_alpha
crater.Geom.zone3right4.Saturation.N = VG_N
crater.Geom.zone3right4.Saturation.SRes = 0.2643
crater.Geom.zone3right4.Saturation.SSat = 1.0

crater.Geom.zone3below4.Saturation.Alpha = VG_alpha
crater.Geom.zone3below4.Saturation.N = VG_N
crater.Geom.zone3below4.Saturation.SRes = 0.2643
crater.Geom.zone3below4.Saturation.SSat = 1.0

crater.Geom.zone4.Saturation.Alpha = VG_alpha
crater.Geom.zone4.Saturation.N = VG_N
crater.Geom.zone4.Saturation.SRes = 0.2643
crater.Geom.zone4.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

crater.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

crater.Cycle.Names = 'constant onoff'
crater.Cycle.constant.Names = 'alltime'
crater.Cycle.constant.alltime.Length = 1
crater.Cycle.constant.Repeat = -1

crater.Cycle.onoff.Names = 'on off'
crater.Cycle.onoff.on.Length = 10
crater.Cycle.onoff.off.Length = 90
crater.Cycle.onoff.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

crater.BCPressure.PatchNames = crater.Geom.domain.Patches

crater.Patch.infiltration.BCPressure.Type = 'FluxConst'
crater.Patch.infiltration.BCPressure.Cycle = 'onoff'
crater.Patch.infiltration.BCPressure.on.Value = -0.10
crater.Patch.infiltration.BCPressure.off.Value = 0.0

crater.Patch.x_lower.BCPressure.Type = 'FluxConst'
crater.Patch.x_lower.BCPressure.Cycle = 'constant'
crater.Patch.x_lower.BCPressure.alltime.Value = 0.0

crater.Patch.y_lower.BCPressure.Type = 'FluxConst'
crater.Patch.y_lower.BCPressure.Cycle = 'constant'
crater.Patch.y_lower.BCPressure.alltime.Value = 0.0

crater.Patch.z_lower.BCPressure.Type = 'FluxConst'
crater.Patch.z_lower.BCPressure.Cycle = 'constant'
crater.Patch.z_lower.BCPressure.alltime.Value = 0.0

crater.Patch.x_upper.BCPressure.Type = 'FluxConst'
crater.Patch.x_upper.BCPressure.Cycle = 'constant'
crater.Patch.x_upper.BCPressure.alltime.Value = 0.0

crater.Patch.y_upper.BCPressure.Type = 'FluxConst'
crater.Patch.y_upper.BCPressure.Cycle = 'constant'
crater.Patch.y_upper.BCPressure.alltime.Value = 0.0

crater.Patch.z_upper.BCPressure.Type = 'FluxConst'
crater.Patch.z_upper.BCPressure.Cycle = 'constant'
crater.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

crater.TopoSlopesX.Type = 'Constant'
crater.TopoSlopesX.GeomNames = 'domain'
crater.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

crater.TopoSlopesY.Type = 'Constant'
crater.TopoSlopesY.GeomNames = 'domain'
crater.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

crater.Mannings.Type = 'Constant'
crater.Mannings.GeomNames = 'domain'
crater.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

crater.ICPressure.Type = 'HydroStaticPatch'
crater.ICPressure.GeomNames = 'domain'

crater.Geom.domain.ICPressure.Value = 1.0
crater.Geom.domain.ICPressure.RefPatch = 'z_lower'
crater.Geom.domain.ICPressure.RefGeom = 'domain'

crater.Geom.infiltration.ICPressure.Value = 10.0
crater.Geom.infiltration.ICPressure.RefPatch = 'infiltration'
crater.Geom.infiltration.ICPressure.RefGeom = 'domain'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

crater.PhaseSources.water.Type = 'Constant'
crater.PhaseSources.water.GeomNames = 'background'
crater.PhaseSources.water.Geom.background.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

crater.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
crater.Solver = 'Richards'
crater.Solver.MaxIter = 10000

crater.Solver.Nonlinear.MaxIter = 15
crater.Solver.Nonlinear.ResidualTol = 1e-9
crater.Solver.Nonlinear.StepTol = 1e-9
crater.Solver.Nonlinear.EtaValue = 1e-5
crater.Solver.Nonlinear.UseJacobian = True
crater.Solver.Nonlinear.DerivativeEpsilon = 1e-7

crater.Solver.Linear.KrylovDimension = 25
crater.Solver.Linear.MaxRestarts = 10

crater.Solver.Linear.Preconditioner = 'MGSemi'
crater.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
crater.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

new_output_dir_name = dir_name
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)
crater.run(working_directory=new_output_dir_name)

passed = True
sig_digits = 5
abs_diff = 1e-200
test_files = ["perm_x", "perm_y", "perm_z", "porosity"]

for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in {test_file}", sig_digits):
        passed = False
        

for i in range(3):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file_with_abs(new_output_dir_name + filename, correct_output_dir_name + filename,
                                 f"Max difference in Pressure for timestep {timestep}", sig_digits, abs_diff):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file_with_abs(new_output_dir_name + filename, correct_output_dir_name + filename,
                                 f"Max difference in Saturation for timestep {timestep}", sig_digits, abs_diff):
        passed = False


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
