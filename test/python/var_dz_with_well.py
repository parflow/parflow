#---------------------------------------------------------
# Runs a simple sand draining problem, rectangular domain
# with variable dz
#---------------------------------------------------------

from parflow.tools.fs import mkdir, get_absolute_path
import numpy as np

import parflow as pf

vardz = pf.Run("model", __file__)

#---------------------------------------------------------

vardz.FileVersion = 4

vardz.Process.Topology.P = 1
vardz.Process.Topology.Q = 1
vardz.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

vardz.ComputationalGrid.Lower.X = 0.0
vardz.ComputationalGrid.Lower.Y = 0.0
vardz.ComputationalGrid.Lower.Z = 0.0

vardz.ComputationalGrid.DX = 1.0
vardz.ComputationalGrid.DY = 1.0


vardz.ComputationalGrid.NX = 1
vardz.ComputationalGrid.NY = 1
vardz.ComputationalGrid.NZ = 14

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

vardz.GeomInput.Names = 'domain_input'

#---------------------------------------------------------
# Geometry Input
#---------------------------------------------------------

vardz.GeomInput.domain_input.InputType = 'Box'
vardz.GeomInput.domain_input.GeomName = 'domain'


#---------------------------------------------------------
# Geometry
#---------------------------------------------------------

vardz.Geom.domain.Lower.X = 0.0
vardz.Geom.domain.Lower.Y = 0.0
vardz.Geom.domain.Lower.Z = 0.0

vardz.Geom.domain.Upper.X = 1.0
vardz.Geom.domain.Upper.Y = 1.0
vardz.Geom.domain.Upper.Z = 14

vardz.Geom.domain.Patches = 'left right front back bottom top'

#--------------------------------------------
# variable dz assignments
#------------------------------------------
vardz.ComputationalGrid.DZ = 1.0
vardz.Solver.Nonlinear.VariableDz = True
vardz.dzScale.GeomNames = 'domain'
vardz.dzScale.Type = 'nzList'
vardz.dzScale.nzListNumber = 14
vardz.Cell._0.dzScale.Value = 1
vardz.Cell._1.dzScale.Value = 2.
vardz.Cell._2.dzScale.Value = 1.
vardz.Cell._3.dzScale.Value = 1.
vardz.Cell._4.dzScale.Value = 1.
vardz.Cell._5.dzScale.Value = 1.
vardz.Cell._6.dzScale.Value = 1.
vardz.Cell._7.dzScale.Value = 1.
vardz.Cell._8.dzScale.Value = 1.
vardz.Cell._9.dzScale.Value = 1
vardz.Cell._10.dzScale.Value = 1
vardz.Cell._11.dzScale.Value = 1
vardz.Cell._12.dzScale.Value = 1
vardz.Cell._13.dzScale.Value = 1

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

vardz.Geom.Perm.Names = 'domain'

vardz.Geom.domain.Perm.Type = 'Constant'
vardz.Geom.domain.Perm.Value = 5.129



vardz.Perm.TensorType = 'TensorByGeom'

vardz.Geom.Perm.TensorByGeom.Names = 'domain'

vardz.Geom.domain.Perm.TensorValX = 1.0
vardz.Geom.domain.Perm.TensorValY = 1.0
vardz.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

vardz.SpecificStorage.Type = 'Constant'
vardz.SpecificStorage.GeomNames = 'domain'
vardz.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

vardz.Phase.Names = 'water'

vardz.Phase.water.Density.Type = 'Constant'
vardz.Phase.water.Density.Value = 1.0

vardz.Phase.water.Viscosity.Type = 'Constant'
vardz.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

vardz.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

vardz.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

vardz.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

vardz.TimingInfo.BaseUnit = 1.0
vardz.TimingInfo.StartCount = 0
vardz.TimingInfo.StartTime = 0.0
vardz.TimingInfo.StopTime = 10.0
vardz.TimingInfo.DumpInterval = 1.0
vardz.TimeStep.Type = 'Constant'
vardz.TimeStep.Value = 0.01
vardz.TimeStep.Value = 0.01

# vardz.Reservoirs.Names = ""

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

vardz.Geom.Porosity.GeomNames = 'domain'
vardz.Geom.domain.Porosity.Type = 'Constant'
vardz.Geom.domain.Porosity.Value = 0.4150

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

vardz.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

vardz.Phase.RelPerm.Type = 'VanGenuchten'
vardz.Phase.RelPerm.GeomNames = 'domain'
vardz.Geom.domain.RelPerm.Alpha = 2.7
vardz.Geom.domain.RelPerm.N = 3.8

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

vardz.Phase.Saturation.Type = 'VanGenuchten'
vardz.Phase.Saturation.GeomNames = 'domain'
vardz.Geom.domain.Saturation.Alpha = 2.7
vardz.Geom.domain.Saturation.N = 3.8
vardz.Geom.domain.Saturation.SRes = 0.106
vardz.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

vardz.Cycle.Names = 'constant'
vardz.Cycle.constant.Names = 'alltime'
vardz.Cycle.constant.alltime.Length = 1
vardz.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

vardz.BCPressure.PatchNames = 'left right front back bottom top'

vardz.Patch.left.BCPressure.Type = 'FluxConst'
vardz.Patch.left.BCPressure.Cycle = 'constant'
vardz.Patch.left.BCPressure.RefGeom = 'domain'
vardz.Patch.left.BCPressure.RefPatch = 'bottom'
vardz.Patch.left.BCPressure.alltime.Value = 0.0

vardz.Patch.right.BCPressure.Type = 'FluxConst'
vardz.Patch.right.BCPressure.Cycle = 'constant'
vardz.Patch.right.BCPressure.RefGeom = 'domain'
vardz.Patch.right.BCPressure.RefPatch = 'bottom'
vardz.Patch.right.BCPressure.alltime.Value = 0.0

vardz.Patch.front.BCPressure.Type = 'FluxConst'
vardz.Patch.front.BCPressure.Cycle = 'constant'
vardz.Patch.front.BCPressure.alltime.Value = 0.0

vardz.Patch.back.BCPressure.Type = 'FluxConst'
vardz.Patch.back.BCPressure.Cycle = 'constant'
vardz.Patch.back.BCPressure.alltime.Value = 0.0

vardz.Patch.bottom.BCPressure.Type = 'DirEquilRefPatch'
vardz.Patch.bottom.BCPressure.Type = 'FluxConst'
vardz.Patch.bottom.BCPressure.Cycle = 'constant'
vardz.Patch.bottom.BCPressure.RefGeom = 'domain'
vardz.Patch.bottom.BCPressure.RefPatch = 'bottom'
vardz.Patch.bottom.BCPressure.alltime.Value = 0.0

vardz.Patch.top.BCPressure.Type = 'DirEquilRefPatch'
vardz.Patch.top.BCPressure.Type = 'FluxConst'
vardz.Patch.top.BCPressure.Cycle = 'constant'
vardz.Patch.top.BCPressure.RefGeom = 'domain'
vardz.Patch.top.BCPressure.RefPatch = 'bottom'
vardz.Patch.top.BCPressure.alltime.Value = -0.0001

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

vardz.TopoSlopesX.Type = 'Constant'
vardz.TopoSlopesX.GeomNames = 'domain'
vardz.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

vardz.TopoSlopesY.Type = 'Constant'
vardz.TopoSlopesY.GeomNames = 'domain'
vardz.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

vardz.Mannings.Type = 'Constant'
vardz.Mannings.GeomNames = 'domain'
vardz.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

vardz.ICPressure.Type = 'Constant'
vardz.ICPressure.GeomNames = 'domain'
vardz.Geom.domain.ICPressure.Value = -10.0
vardz.Geom.domain.ICPressure.RefGeom = 'domain'
vardz.Geom.domain.ICPressure.RefPatch = 'top'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

vardz.PhaseSources.water.Type = 'Constant'
vardz.PhaseSources.water.GeomNames = 'domain'
vardz.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

vardz.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

vardz.Solver = 'Richards'
vardz.Solver.MaxIter = 2500

vardz.Solver.Nonlinear.MaxIter = 200
vardz.Solver.Nonlinear.ResidualTol = 1e-9
vardz.Solver.Nonlinear.EtaChoice = 'Walker1'
vardz.Solver.Nonlinear.EtaValue = 1e-5
vardz.Solver.Nonlinear.UseJacobian = True
vardz.Solver.Nonlinear.DerivativeEpsilon = 1e-10

vardz.Solver.Linear.KrylovDimension = 10

vardz.Solver.Linear.Preconditioner = 'MGSemi'
vardz.Solver.Linear.Preconditioner = 'PFMG'
vardz.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
vardz.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

vardz.Wells.Names = 'pressure_well'
vardz.Wells.pressure_well.InputType = 'Vertical'
vardz.Wells.pressure_well.Action = 'Extraction'
vardz.Wells.pressure_well.Type = 'Pressure'
vardz.Wells.pressure_well.X = 0.5
vardz.Wells.pressure_well.Y = 0.5
vardz.Wells.pressure_well.ZUpper = 10.5
vardz.Wells.pressure_well.ZLower = .5
vardz.Wells.pressure_well.Method = 'Standard'
vardz.Wells.pressure_well.Cycle = 'constant'
vardz.Wells.pressure_well.alltime.Pressure.Value = 0.5
vardz.Wells.pressure_well.alltime.Saturation.water.Value = 1.0

#-----------------------------------------------------------------------------
# Run and do tests
#-----------------------------------------------------------------------------

# For our tests we will be comparing the pressure field at the 10th timestep
# We use np.allclose to compare instead of np.equals because changing the var
# dz causes tiny differences from floating point arithmatic. These changes make
# total sense and are unavoidable.

pressure_file = "model.out.press.00010.pfb"

# base case single column
dir_name = get_absolute_path('test_output/single_column_1')
mkdir(dir_name)
vardz.run(working_directory=dir_name)

base_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")

# single column test 1
dir_name = get_absolute_path('test_output/single_column_2')
mkdir(dir_name)

vardz.ComputationalGrid.DZ = 10.0
vardz.Geom.domain.Upper.Z = 140.0
vardz.dzScale.nzListNumber = 14
vardz.Cell._0.dzScale.Value = .1
vardz.Cell._1.dzScale.Value = .2
vardz.Cell._2.dzScale.Value = .1
vardz.Cell._3.dzScale.Value = .1
vardz.Cell._4.dzScale.Value = .1
vardz.Cell._5.dzScale.Value = .1
vardz.Cell._6.dzScale.Value = .1
vardz.Cell._7.dzScale.Value = .1
vardz.Cell._8.dzScale.Value = .1
vardz.Cell._9.dzScale.Value = .1
vardz.Cell._10.dzScale.Value = .1
vardz.Cell._11.dzScale.Value = .1
vardz.Cell._12.dzScale.Value = .1
vardz.Cell._13.dzScale.Value = .1

vardz.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert(np.allclose(base_case_pressure, test_case_pressure))

# single column test 2
dir_name = get_absolute_path('test_output/single_column_3')
mkdir(dir_name)

vardz.ComputationalGrid.DZ = 0.1
vardz.Geom.domain.Upper.Z = 1.4
vardz.Cell._0.dzScale.Value = 10
vardz.Cell._1.dzScale.Value = 20
vardz.Cell._2.dzScale.Value = 10
vardz.Cell._3.dzScale.Value = 10
vardz.Cell._4.dzScale.Value = 10
vardz.Cell._5.dzScale.Value = 10
vardz.Cell._6.dzScale.Value = 10
vardz.Cell._7.dzScale.Value = 10
vardz.Cell._8.dzScale.Value = 10
vardz.Cell._9.dzScale.Value = 10
vardz.Cell._10.dzScale.Value = 10
vardz.Cell._11.dzScale.Value = 10
vardz.Cell._12.dzScale.Value = 10
vardz.Cell._13.dzScale.Value = 10

vardz.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert(np.allclose(base_case_pressure, test_case_pressure))

# Next we switch to a multicolumn setup and add a flux well in to make sure this works for both types
# of wells
vardz.ComputationalGrid.NX = 2
vardz.ComputationalGrid.NY = 2
vardz.Geom.domain.Upper.X = 2.0
vardz.Geom.domain.Upper.Y = 2.0

vardz.Wells.Names = 'pressure_well flux_well'

vardz.Wells.pressure_well.InputType = 'Vertical'
vardz.Wells.pressure_well.Action = 'Extraction'
vardz.Wells.pressure_well.Type = 'Pressure'
vardz.Wells.pressure_well.X = 0.5
vardz.Wells.pressure_well.Y = 0.5
vardz.Wells.pressure_well.ZUpper = 10.5
vardz.Wells.pressure_well.ZLower = .5
vardz.Wells.pressure_well.Method = 'Standard'
vardz.Wells.pressure_well.Cycle = 'constant'
vardz.Wells.pressure_well.alltime.Pressure.Value = 0.5
vardz.Wells.pressure_well.alltime.Saturation.water.Value = 1.0

vardz.Wells.flux_well.InputType = 'Vertical'
vardz.Wells.flux_well.Type = 'Flux'
vardz.Wells.flux_well.Action = 'Extraction'
vardz.Wells.flux_well.Cycle = 'constant'
vardz.Wells.flux_well.X = 1.5
vardz.Wells.flux_well.Y = 0.5
vardz.Wells.flux_well.ZLower = .5
vardz.Wells.flux_well.ZUpper = 10.5
vardz.Wells.flux_well.Method = 'Standard'
vardz.Wells.flux_well.alltime.Flux.water.Value = 7.5

# Multi column  base case
dir_name = get_absolute_path('test_output/multi_column_1')
mkdir(dir_name)

vardz.ComputationalGrid.DZ = 1.0
vardz.Geom.domain.Upper.Z = 14.0
vardz.dzScale.nzListNumber = 14
vardz.Cell._0.dzScale.Value = 1
vardz.Cell._1.dzScale.Value = 2.
vardz.Cell._2.dzScale.Value = 1.
vardz.Cell._3.dzScale.Value = 1.
vardz.Cell._4.dzScale.Value = 1.
vardz.Cell._5.dzScale.Value = 1.
vardz.Cell._6.dzScale.Value = 1.
vardz.Cell._7.dzScale.Value = 1.
vardz.Cell._8.dzScale.Value = 1.
vardz.Cell._9.dzScale.Value = 1
vardz.Cell._10.dzScale.Value = 1
vardz.Cell._11.dzScale.Value = 1
vardz.Cell._12.dzScale.Value = 1
vardz.Cell._13.dzScale.Value = 1

vardz.run(working_directory=dir_name)
base_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")

dir_name = get_absolute_path('test_output/multi_column_2')
mkdir(dir_name)

vardz.ComputationalGrid.DZ = 10.0
vardz.Geom.domain.Upper.Z = 140.0
vardz.dzScale.nzListNumber = 14
vardz.Cell._0.dzScale.Value = .1
vardz.Cell._1.dzScale.Value = .2
vardz.Cell._2.dzScale.Value = .1
vardz.Cell._3.dzScale.Value = .1
vardz.Cell._4.dzScale.Value = .1
vardz.Cell._5.dzScale.Value = .1
vardz.Cell._6.dzScale.Value = .1
vardz.Cell._7.dzScale.Value = .1
vardz.Cell._8.dzScale.Value = .1
vardz.Cell._9.dzScale.Value = .1
vardz.Cell._10.dzScale.Value = .1
vardz.Cell._11.dzScale.Value = .1
vardz.Cell._12.dzScale.Value = .1
vardz.Cell._13.dzScale.Value = .1

vardz.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert(np.allclose(base_case_pressure, test_case_pressure))

dir_name = get_absolute_path('test_output/multi_column_3')
mkdir(dir_name)

vardz.ComputationalGrid.DZ = 0.1
vardz.Geom.domain.Upper.Z = 1.4
vardz.Cell._0.dzScale.Value = 10
vardz.Cell._1.dzScale.Value = 20
vardz.Cell._2.dzScale.Value = 10
vardz.Cell._3.dzScale.Value = 10
vardz.Cell._4.dzScale.Value = 10
vardz.Cell._5.dzScale.Value = 10
vardz.Cell._6.dzScale.Value = 10
vardz.Cell._7.dzScale.Value = 10
vardz.Cell._8.dzScale.Value = 10
vardz.Cell._9.dzScale.Value = 10
vardz.Cell._10.dzScale.Value = 10
vardz.Cell._11.dzScale.Value = 10
vardz.Cell._12.dzScale.Value = 10
vardz.Cell._13.dzScale.Value = 10

vardz.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert(np.allclose(base_case_pressure, test_case_pressure))
