# Test for using the "PressureFile" option for BC.
# Test is based on default_richards with modified BC
# specification.

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
from parflow.tools.io import write_pfb
import numpy as np

run_name = "bc_pressure_file"
bcp = Run(run_name, __file__)

new_output_dir_name = get_absolute_path('test_output/bcp')
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)

bcp.FileVersion = 4

bcp.Process.Topology.P = 1
bcp.Process.Topology.Q = 1
bcp.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
bcp.ComputationalGrid.Lower.X = -10.0
bcp.ComputationalGrid.Lower.Y = 10.0
bcp.ComputationalGrid.Lower.Z = 1.0

bcp.ComputationalGrid.DX = 8.8888888888888893
bcp.ComputationalGrid.DY = 10.666666666666666
bcp.ComputationalGrid.DZ = 1.0

bcp.ComputationalGrid.NX = 18
bcp.ComputationalGrid.NY = 15
bcp.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
bcp.GeomInput.Names = "domain_input background_input source_region_input concen_region_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
bcp.GeomInput.domain_input.InputType = 'Box'
bcp.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
bcp.Geom.domain.Lower.X = -10.0
bcp.Geom.domain.Lower.Y = 10.0
bcp.Geom.domain.Lower.Z = 1.0

bcp.Geom.domain.Upper.X = 150.0
bcp.Geom.domain.Upper.Y = 170.0
bcp.Geom.domain.Upper.Z = 9.0

bcp.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
bcp.GeomInput.background_input.InputType = 'Box'
bcp.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
bcp.Geom.background.Lower.X = -99999999.0
bcp.Geom.background.Lower.Y = -99999999.0
bcp.Geom.background.Lower.Z = -99999999.0

bcp.Geom.background.Upper.X = 99999999.0
bcp.Geom.background.Upper.Y = 99999999.0
bcp.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
bcp.GeomInput.source_region_input.InputType = 'Box'
bcp.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
bcp.Geom.source_region.Lower.X = 65.56
bcp.Geom.source_region.Lower.Y = 79.34
bcp.Geom.source_region.Lower.Z = 4.5

bcp.Geom.source_region.Upper.X = 74.44
bcp.Geom.source_region.Upper.Y = 89.99
bcp.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
bcp.GeomInput.concen_region_input.InputType = 'Box'
bcp.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
bcp.Geom.concen_region.Lower.X = 60.0
bcp.Geom.concen_region.Lower.Y = 80.0
bcp.Geom.concen_region.Lower.Z = 4.0

bcp.Geom.concen_region.Upper.X = 80.0
bcp.Geom.concen_region.Upper.Y = 100.0
bcp.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
bcp.Geom.Perm.Names = "background"

bcp.Geom.background.Perm.Type = 'Constant'
bcp.Geom.background.Perm.Value = 4.0

bcp.Perm.TensorType = "TensorByGeom"

bcp.Geom.Perm.TensorByGeom.Names = "background"

bcp.Geom.background.Perm.TensorValX = 1.0
bcp.Geom.background.Perm.TensorValY = 1.0
bcp.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

bcp.SpecificStorage.Type = 'Constant'
bcp.SpecificStorage.GeomNames = "domain"
bcp.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

bcp.Phase.Names = "water"

bcp.Phase.water.Density.Type = "Constant"
bcp.Phase.water.Density.Value = 1.0

bcp.Phase.water.Viscosity.Type = "Constant"
bcp.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
bcp.Contaminants.Names = ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
bcp.Geom.Retardation.GeomNames = ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

bcp.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

bcp.TimingInfo.BaseUnit = 1.0
bcp.TimingInfo.StartCount = 0
bcp.TimingInfo.StartTime = 0.0
bcp.TimingInfo.StopTime = 0.010
bcp.TimingInfo.DumpInterval = -1
bcp.TimeStep.Type = "Constant"
bcp.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

bcp.Geom.Porosity.GeomNames = "background"

bcp.Geom.background.Porosity.Type = "Constant"
bcp.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
bcp.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

bcp.Phase.RelPerm.Type = 'VanGenuchten'
bcp.Phase.RelPerm.GeomNames = 'domain'
bcp.Geom.domain.RelPerm.Alpha = 0.005
bcp.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

bcp.Phase.Saturation.Type = 'VanGenuchten'
bcp.Phase.Saturation.GeomNames = 'domain'
bcp.Geom.domain.Saturation.Alpha = 0.005
bcp.Geom.domain.Saturation.N = 2.0
bcp.Geom.domain.Saturation.SRes = 0.2
bcp.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
bcp.Wells.Names = ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
bcp.Cycle.Names = 'constant'
bcp.Cycle.constant.Names = "alltime"
bcp.Cycle.constant.alltime.Length = 1
bcp.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
bcp.BCPressure.PatchNames = 'left right front back bottom top'

bcp.Patch.left.BCPressure.Type = "DirEquilRefPatch"
bcp.Patch.left.BCPressure.Cycle = "constant"
bcp.Patch.left.BCPressure.RefGeom = "domain"
bcp.Patch.left.BCPressure.RefPatch = "bottom"
bcp.Patch.left.BCPressure.alltime.Value = 5.0

bcp.Patch.right.BCPressure.Type = "DirEquilRefPatch"
bcp.Patch.right.BCPressure.Cycle = "constant"
bcp.Patch.right.BCPressure.RefGeom = 'domain'
bcp.Patch.right.BCPressure.RefPatch = "bottom"
bcp.Patch.right.BCPressure.alltime.Value = 3.0

bcp.Patch.front.BCPressure.Type = "FluxConst"
bcp.Patch.front.BCPressure.Cycle = "constant"
bcp.Patch.front.BCPressure.alltime.Value = 0.0

bcp.Patch.back.BCPressure.Type = "FluxConst"
bcp.Patch.back.BCPressure.Cycle = "constant"
bcp.Patch.back.BCPressure.alltime.Value = 0.0

bcp.Patch.bottom.BCPressure.Type = "FluxConst"
bcp.Patch.bottom.BCPressure.Cycle = "constant"
bcp.Patch.bottom.BCPressure.alltime.Value = 0.0

# Testing using the PressureFile option, create a file with 0's for all elements
pressure_filename = "pressure_test.pfb"

pressure_array = np.ndarray((bcp.ComputationalGrid.NX, bcp.ComputationalGrid.NY, bcp.ComputationalGrid.NZ))

write_pfb(new_output_dir_name + '/' +  pressure_filename,
          pressure_array,
          bcp.ComputationalGrid.NX,
          bcp.ComputationalGrid.NY,
          bcp.ComputationalGrid.NZ,
          bcp.ComputationalGrid.Lower.X,
          bcp.ComputationalGrid.Lower.Y,
          bcp.ComputationalGrid.Lower.Z,
          bcp.ComputationalGrid.DX,
          bcp.ComputationalGrid.DY,
          bcp.ComputationalGrid.DZ,
          z_first=False
)

bcp.Patch.top.BCPressure.Type = "PressureFile"
bcp.Patch.top.BCPressure.Cycle = "constant"
bcp.Patch.top.BCPressure.alltime.FileName = pressure_filename

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

bcp.TopoSlopesX.Type = "Constant"
bcp.TopoSlopesX.GeomNames = ""

bcp.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

bcp.TopoSlopesY.Type = "Constant"
bcp.TopoSlopesY.GeomNames = ""

bcp.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

bcp.Mannings.Type = "Constant"
bcp.Mannings.GeomNames = ""
bcp.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

bcp.ICPressure.Type = "HydroStaticPatch"
bcp.ICPressure.GeomNames = 'domain'
bcp.Geom.domain.ICPressure.Value = 3.0
bcp.Geom.domain.ICPressure.RefGeom = 'domain'
bcp.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

bcp.PhaseSources.water.Type = 'Constant'
bcp.PhaseSources.water.GeomNames = 'background'
bcp.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

bcp.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
bcp.Solver = 'Richards'
bcp.Solver.MaxIter = 5

bcp.Solver.Nonlinear.MaxIter = 10
bcp.Solver.Nonlinear.ResidualTol = 1e-9
bcp.Solver.Nonlinear.EtaChoice = 'EtaConstant'
bcp.Solver.Nonlinear.EtaValue = 1e-5
bcp.Solver.Nonlinear.UseJacobian = True
bcp.Solver.Nonlinear.DerivativeEpsilon = 1e-2

bcp.Solver.Linear.KrylovDimension = 10

bcp.Solver.Linear.Preconditioner = 'PFMG'
#bcp.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
#bcp.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

bcp.Solver.PrintVelocities = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
bcp.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in {test_file}"):
        passed = False

sig_digits=6
abs_value = 1e-12
for i in range(0, 6):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in Pressure for timestep {timestep}"):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                        f"Max difference in Saturation for timestep {timestep}"):
        passed = False
    filename = f"/{run_name}.out.velx.{timestep}.pfb"
    if not pf_test_file_with_abs(new_output_dir_name + filename, correct_output_dir_name + filename,
                                 f"Max difference in x-velocity for timestep {timestep}", sig_digits, abs_value):
        passed = False
    filename = f"/{run_name}.out.vely.{timestep}.pfb"
    if not pf_test_file_with_abs(new_output_dir_name + filename, correct_output_dir_name + filename,
                                 f"Max difference in y-velocity for timestep {timestep}", sig_digits, abs_value):
        passed = False
    filename = f"/{run_name}.out.vely.{timestep}.pfb"
    if not pf_test_file_with_abs(new_output_dir_name + filename, correct_output_dir_name + filename,
                                 f"Max difference in z-velocity for timestep {timestep}", sig_digits, abs_value):
        passed = False

rm(new_output_dir_name)
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
