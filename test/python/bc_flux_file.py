# Test for using the "PressureFile" option for BC.
# Test is based on default_richards with modified BC
# specification.

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
from parflow.tools.io import write_pfb
import numpy as np

run_name = "bc_flux_file"
bcf = Run(run_name, __file__)

bcf.FileVersion = 4

new_output_dir_name = get_absolute_path('test_output/bcf')
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)

bcf.Process.Topology.P = 1
bcf.Process.Topology.Q = 1
bcf.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
bcf.ComputationalGrid.Lower.X = -10.0
bcf.ComputationalGrid.Lower.Y = 10.0
bcf.ComputationalGrid.Lower.Z = 1.0

bcf.ComputationalGrid.DX = 8.8888888888888893
bcf.ComputationalGrid.DY = 10.666666666666666
bcf.ComputationalGrid.DZ = 1.0

bcf.ComputationalGrid.NX = 18
bcf.ComputationalGrid.NY = 15
bcf.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
bcf.GeomInput.Names = "domain_input background_input source_region_input concen_region_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
bcf.GeomInput.domain_input.InputType = 'Box'
bcf.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
bcf.Geom.domain.Lower.X = -10.0
bcf.Geom.domain.Lower.Y = 10.0
bcf.Geom.domain.Lower.Z = 1.0

bcf.Geom.domain.Upper.X = 150.0
bcf.Geom.domain.Upper.Y = 170.0
bcf.Geom.domain.Upper.Z = 9.0

bcf.Geom.domain.Patches = "left right front back bottom top"

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
bcf.GeomInput.background_input.InputType = 'Box'
bcf.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
bcf.Geom.background.Lower.X = -99999999.0
bcf.Geom.background.Lower.Y = -99999999.0
bcf.Geom.background.Lower.Z = -99999999.0

bcf.Geom.background.Upper.X = 99999999.0
bcf.Geom.background.Upper.Y = 99999999.0
bcf.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
bcf.GeomInput.source_region_input.InputType = 'Box'
bcf.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
bcf.Geom.source_region.Lower.X = 65.56
bcf.Geom.source_region.Lower.Y = 79.34
bcf.Geom.source_region.Lower.Z = 4.5

bcf.Geom.source_region.Upper.X = 74.44
bcf.Geom.source_region.Upper.Y = 89.99
bcf.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
bcf.GeomInput.concen_region_input.InputType = 'Box'
bcf.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
bcf.Geom.concen_region.Lower.X = 60.0
bcf.Geom.concen_region.Lower.Y = 80.0
bcf.Geom.concen_region.Lower.Z = 4.0

bcf.Geom.concen_region.Upper.X = 80.0
bcf.Geom.concen_region.Upper.Y = 100.0
bcf.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
bcf.Geom.Perm.Names = "background"

bcf.Geom.background.Perm.Type = 'Constant'
bcf.Geom.background.Perm.Value = 4.0

bcf.Perm.TensorType = 'TensorByGeom'

bcf.Geom.Perm.TensorByGeom.Names = "background"

bcf.Geom.background.Perm.TensorValX = 1.0
bcf.Geom.background.Perm.TensorValY = 1.0
bcf.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

bcf.SpecificStorage.Type = 'Constant'
bcf.SpecificStorage.GeomNames = "domain"
bcf.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

bcf.Phase.Names = "water"

bcf.Phase.water.Density.Type = 'Constant'
bcf.Phase.water.Density.Value = 1.0

bcf.Phase.water.Viscosity.Type = 'Constant'
bcf.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
bcf.Contaminants.Names = ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
bcf.Geom.Retardation.GeomNames = ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

bcf.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

bcf.TimingInfo.BaseUnit = 1.0
bcf.TimingInfo.StartCount = 0
bcf.TimingInfo.StartTime = 0.0
bcf.TimingInfo.StopTime = 0.010
bcf.TimingInfo.DumpInterval = -1
bcf.TimeStep.Type = 'Constant'
bcf.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

bcf.Geom.Porosity.GeomNames = 'background'

bcf.Geom.background.Porosity.Type = 'Constant'
bcf.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
bcf.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

bcf.Phase.RelPerm.Type = 'VanGenuchten'
bcf.Phase.RelPerm.GeomNames = 'domain'
bcf.Geom.domain.RelPerm.Alpha = 0.005
bcf.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

bcf.Phase.Saturation.Type = 'VanGenuchten'
bcf.Phase.Saturation.GeomNames = 'domain'
bcf.Geom.domain.Saturation.Alpha = 0.005
bcf.Geom.domain.Saturation.N = 2.0
bcf.Geom.domain.Saturation.SRes = 0.2
bcf.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
bcf.Wells.Names = ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
bcf.Cycle.Names = 'constant'
bcf.Cycle.constant.Names = "alltime"
bcf.Cycle.constant.alltime.Length = 1
bcf.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
bcf.BCPressure.PatchNames = "left right front back bottom top"

bcf.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
bcf.Patch.left.BCPressure.Cycle = "constant"
bcf.Patch.left.BCPressure.RefGeom = 'domain'
bcf.Patch.left.BCPressure.RefPatch = 'bottom'
bcf.Patch.left.BCPressure.alltime.Value = 5.0

bcf.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
bcf.Patch.right.BCPressure.Cycle = "constant"
bcf.Patch.right.BCPressure.RefGeom = 'domain'
bcf.Patch.right.BCPressure.RefPatch = 'bottom'
bcf.Patch.right.BCPressure.alltime.Value = 3.0

bcf.Patch.front.BCPressure.Type = 'FluxConst'
bcf.Patch.front.BCPressure.Cycle = "constant"
bcf.Patch.front.BCPressure.alltime.Value = 0.0

bcf.Patch.back.BCPressure.Type = 'FluxConst'
bcf.Patch.back.BCPressure.Cycle = "constant"
bcf.Patch.back.BCPressure.alltime.Value = 0.0

bcf.Patch.bottom.BCPressure.Type = 'FluxConst'
bcf.Patch.bottom.BCPressure.Cycle = "constant"
bcf.Patch.bottom.BCPressure.alltime.Value = 0.0

#bcf.Patch.top.BCPressure.Type = 'FluxConst'
#bcf.Patch.top.BCPressure.Cycle = "constant"
#bcf.Patch.top.BCPressure.alltime.Value = 0.0

#
# Testing using the PressureFile option, create a file with 0's for all elements
flux_filename = "pressure_test.pfb"

flux_array = np.ndarray((bcf.ComputationalGrid.NX, bcf.ComputationalGrid.NY, bcf.ComputationalGrid.NZ))

write_pfb(new_output_dir_name + '/' + flux_filename,
          flux_array,
          bcf.ComputationalGrid.NX,
          bcf.ComputationalGrid.NY,
          bcf.ComputationalGrid.NZ,
          bcf.ComputationalGrid.Lower.X,
          bcf.ComputationalGrid.Lower.Y,
          bcf.ComputationalGrid.Lower.Z,
          bcf.ComputationalGrid.DX,
          bcf.ComputationalGrid.DY,
          bcf.ComputationalGrid.DZ,
          z_first=False
)

bcf.Patch.top.BCPressure.Type = "FluxFile"
bcf.Patch.top.BCPressure.Cycle = "constant"
bcf.Patch.top.BCPressure.alltime.FileName = flux_filename

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

bcf.TopoSlopesX.Type = "Constant"
bcf.TopoSlopesX.GeomNames = ""

bcf.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

bcf.TopoSlopesY.Type = "Constant"
bcf.TopoSlopesY.GeomNames = ""

bcf.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

bcf.Mannings.Type = "Constant"
bcf.Mannings.GeomNames = ""
bcf.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

bcf.ICPressure.Type = 'HydroStaticPatch'
bcf.ICPressure.GeomNames = 'domain'
bcf.Geom.domain.ICPressure.Value = 3.0
bcf.Geom.domain.ICPressure.RefGeom = 'domain'
bcf.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

bcf.PhaseSources.water.Type = 'Constant'
bcf.PhaseSources.water.GeomNames = 'background'
bcf.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

bcf.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
bcf.Solver = 'Richards'
bcf.Solver.MaxIter = 5

bcf.Solver.Nonlinear.MaxIter = 10
bcf.Solver.Nonlinear.ResidualTol = 1e-9
bcf.Solver.Nonlinear.EtaChoice = 'EtaConstant'
bcf.Solver.Nonlinear.EtaValue = 1e-5
bcf.Solver.Nonlinear.UseJacobian = True
bcf.Solver.Nonlinear.DerivativeEpsilon = 1e-2

bcf.Solver.Linear.KrylovDimension = 10

bcf.Solver.Linear.Preconditioner = 'PFMG'
#bcf.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
#bcf.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

bcf.Solver.PrintVelocities = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
bcf.run(working_directory=new_output_dir_name)

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
