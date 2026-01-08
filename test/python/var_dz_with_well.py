# ---------------------------------------------------------
# Runs a simple sand draining problem, rectangular domain
# with variable dz
# ---------------------------------------------------------

from parflow.tools.fs import mkdir, get_absolute_path
import numpy as np

import parflow as pf

var_dz_with_well = pf.Run("var_dz_with_well", __file__)

# ---------------------------------------------------------

var_dz_with_well.FileVersion = 4

var_dz_with_well.Process.Topology.P = 1
var_dz_with_well.Process.Topology.Q = 1
var_dz_with_well.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

var_dz_with_well.ComputationalGrid.Lower.X = 0.0
var_dz_with_well.ComputationalGrid.Lower.Y = 0.0
var_dz_with_well.ComputationalGrid.Lower.Z = 0.0

var_dz_with_well.ComputationalGrid.DX = 1.0
var_dz_with_well.ComputationalGrid.DY = 1.0


var_dz_with_well.ComputationalGrid.NX = 1
var_dz_with_well.ComputationalGrid.NY = 1
var_dz_with_well.ComputationalGrid.NZ = 14

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

var_dz_with_well.GeomInput.Names = "domain_input"

# ---------------------------------------------------------
# Geometry Input
# ---------------------------------------------------------

var_dz_with_well.GeomInput.domain_input.InputType = "Box"
var_dz_with_well.GeomInput.domain_input.GeomName = "domain"


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

var_dz_with_well.Geom.domain.Lower.X = 0.0
var_dz_with_well.Geom.domain.Lower.Y = 0.0
var_dz_with_well.Geom.domain.Lower.Z = 0.0

var_dz_with_well.Geom.domain.Upper.X = 1.0
var_dz_with_well.Geom.domain.Upper.Y = 144.0
var_dz_with_well.Geom.domain.Upper.Z = 14

var_dz_with_well.Geom.domain.Patches = "left right front back bottom top"

# --------------------------------------------
# variable dz assignments
# ------------------------------------------
var_dz_with_well.ComputationalGrid.DZ = 1.0
var_dz_with_well.Solver.Nonlinear.VariableDz = True
var_dz_with_well.dzScale.GeomNames = "domain"
var_dz_with_well.dzScale.Type = "nzList"
var_dz_with_well.dzScale.nzListNumber = 14
var_dz_with_well.Cell._0.dzScale.Value = 1
var_dz_with_well.Cell._1.dzScale.Value = 2.0
var_dz_with_well.Cell._2.dzScale.Value = 1.0
var_dz_with_well.Cell._3.dzScale.Value = 1.0
var_dz_with_well.Cell._4.dzScale.Value = 1.0
var_dz_with_well.Cell._5.dzScale.Value = 1.0
var_dz_with_well.Cell._6.dzScale.Value = 1.0
var_dz_with_well.Cell._7.dzScale.Value = 1.0
var_dz_with_well.Cell._8.dzScale.Value = 1.0
var_dz_with_well.Cell._9.dzScale.Value = 1
var_dz_with_well.Cell._10.dzScale.Value = 1
var_dz_with_well.Cell._11.dzScale.Value = 1
var_dz_with_well.Cell._12.dzScale.Value = 1
var_dz_with_well.Cell._13.dzScale.Value = 1

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

var_dz_with_well.Geom.Perm.Names = "domain"

var_dz_with_well.Geom.domain.Perm.Type = "Constant"
var_dz_with_well.Geom.domain.Perm.Value = 5.129


var_dz_with_well.Perm.TensorType = "TensorByGeom"

var_dz_with_well.Geom.Perm.TensorByGeom.Names = "domain"

var_dz_with_well.Geom.domain.Perm.TensorValX = 1.0
var_dz_with_well.Geom.domain.Perm.TensorValY = 1.0
var_dz_with_well.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

var_dz_with_well.SpecificStorage.Type = "Constant"
var_dz_with_well.SpecificStorage.GeomNames = "domain"
var_dz_with_well.Geom.domain.SpecificStorage.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

var_dz_with_well.Phase.Names = "water"

var_dz_with_well.Phase.water.Density.Type = "Constant"
var_dz_with_well.Phase.water.Density.Value = 1.0

var_dz_with_well.Phase.water.Viscosity.Type = "Constant"
var_dz_with_well.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

var_dz_with_well.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

var_dz_with_well.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

var_dz_with_well.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

var_dz_with_well.TimingInfo.BaseUnit = 1.0
var_dz_with_well.TimingInfo.StartCount = 0
var_dz_with_well.TimingInfo.StartTime = 0.0
var_dz_with_well.TimingInfo.StopTime = 10.0
var_dz_with_well.TimingInfo.DumpInterval = 1.0
var_dz_with_well.TimeStep.Type = "Constant"
var_dz_with_well.TimeStep.Value = 0.01
var_dz_with_well.TimeStep.Value = 0.01

# vardz.Reservoirs.Names = ""

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

var_dz_with_well.Geom.Porosity.GeomNames = "domain"
var_dz_with_well.Geom.domain.Porosity.Type = "Constant"
var_dz_with_well.Geom.domain.Porosity.Value = 0.4150

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

var_dz_with_well.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

var_dz_with_well.Phase.RelPerm.Type = "VanGenuchten"
var_dz_with_well.Phase.RelPerm.GeomNames = "domain"
var_dz_with_well.Geom.domain.RelPerm.Alpha = 2.7
var_dz_with_well.Geom.domain.RelPerm.N = 3.8

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

var_dz_with_well.Phase.Saturation.Type = "VanGenuchten"
var_dz_with_well.Phase.Saturation.GeomNames = "domain"
var_dz_with_well.Geom.domain.Saturation.Alpha = 2.7
var_dz_with_well.Geom.domain.Saturation.N = 3.8
var_dz_with_well.Geom.domain.Saturation.SRes = 0.106
var_dz_with_well.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

var_dz_with_well.Cycle.Names = "constant"
var_dz_with_well.Cycle.constant.Names = "alltime"
var_dz_with_well.Cycle.constant.alltime.Length = 1
var_dz_with_well.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

var_dz_with_well.BCPressure.PatchNames = "left right front back bottom top"

var_dz_with_well.Patch.left.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.left.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.left.BCPressure.RefGeom = "domain"
var_dz_with_well.Patch.left.BCPressure.RefPatch = "bottom"
var_dz_with_well.Patch.left.BCPressure.alltime.Value = 0.0

var_dz_with_well.Patch.right.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.right.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.right.BCPressure.RefGeom = "domain"
var_dz_with_well.Patch.right.BCPressure.RefPatch = "bottom"
var_dz_with_well.Patch.right.BCPressure.alltime.Value = 0.0

var_dz_with_well.Patch.front.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.front.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.front.BCPressure.alltime.Value = 0.0

var_dz_with_well.Patch.back.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.back.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.back.BCPressure.alltime.Value = 0.0

var_dz_with_well.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
var_dz_with_well.Patch.bottom.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.bottom.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.bottom.BCPressure.RefGeom = "domain"
var_dz_with_well.Patch.bottom.BCPressure.RefPatch = "bottom"
var_dz_with_well.Patch.bottom.BCPressure.alltime.Value = 0.0

var_dz_with_well.Patch.top.BCPressure.Type = "DirEquilRefPatch"
var_dz_with_well.Patch.top.BCPressure.Type = "FluxConst"
var_dz_with_well.Patch.top.BCPressure.Cycle = "constant"
var_dz_with_well.Patch.top.BCPressure.RefGeom = "domain"
var_dz_with_well.Patch.top.BCPressure.RefPatch = "bottom"
var_dz_with_well.Patch.top.BCPressure.alltime.Value = -0.0001

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

var_dz_with_well.TopoSlopesX.Type = "Constant"
var_dz_with_well.TopoSlopesX.GeomNames = "domain"
var_dz_with_well.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

var_dz_with_well.TopoSlopesY.Type = "Constant"
var_dz_with_well.TopoSlopesY.GeomNames = "domain"
var_dz_with_well.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

var_dz_with_well.Mannings.Type = "Constant"
var_dz_with_well.Mannings.GeomNames = "domain"
var_dz_with_well.Mannings.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

var_dz_with_well.ICPressure.Type = "Constant"
var_dz_with_well.ICPressure.GeomNames = "domain"
var_dz_with_well.Geom.domain.ICPressure.Value = -10.0
var_dz_with_well.Geom.domain.ICPressure.RefGeom = "domain"
var_dz_with_well.Geom.domain.ICPressure.RefPatch = "top"

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

var_dz_with_well.PhaseSources.water.Type = "Constant"
var_dz_with_well.PhaseSources.water.GeomNames = "domain"
var_dz_with_well.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

var_dz_with_well.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

var_dz_with_well.Solver = "Richards"
var_dz_with_well.Solver.MaxIter = 2500

var_dz_with_well.Solver.Nonlinear.MaxIter = 200
var_dz_with_well.Solver.Nonlinear.ResidualTol = 1e-9
var_dz_with_well.Solver.Nonlinear.EtaChoice = "Walker1"
var_dz_with_well.Solver.Nonlinear.EtaValue = 1e-5
var_dz_with_well.Solver.Nonlinear.UseJacobian = True
var_dz_with_well.Solver.Nonlinear.DerivativeEpsilon = 1e-10

var_dz_with_well.Solver.Linear.KrylovDimension = 10

var_dz_with_well.Solver.Linear.Preconditioner = "MGSemi"
var_dz_with_well.Solver.Linear.Preconditioner = "PFMG"
var_dz_with_well.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
var_dz_with_well.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10

var_dz_with_well.Wells.Names = "pressure_well"
var_dz_with_well.Wells.CorrectForVarDz = 1
var_dz_with_well.Wells.pressure_well.InputType = "Vertical"
var_dz_with_well.Wells.pressure_well.Action = "Extraction"
var_dz_with_well.Wells.pressure_well.Type = "Pressure"
var_dz_with_well.Wells.pressure_well.X = 0.5
var_dz_with_well.Wells.pressure_well.Y = 0.5
var_dz_with_well.Wells.pressure_well.ZUpper = 10.5
var_dz_with_well.Wells.pressure_well.ZLower = 0.5
var_dz_with_well.Wells.pressure_well.Method = "Standard"
var_dz_with_well.Wells.pressure_well.Cycle = "constant"
var_dz_with_well.Wells.pressure_well.alltime.Pressure.Value = 0.5
var_dz_with_well.Wells.pressure_well.alltime.Saturation.water.Value = 1.0

# -----------------------------------------------------------------------------
# Run and do tests
# -----------------------------------------------------------------------------

# For our tests we will be comparing the pressure field at the 10th timestep
# We use np.allclose to compare instead of np.equals because changing the var
# dz causes tiny differences from floating point arithmatic. These changes make
# total sense and are unavoidable.

pressure_file = "var_dz_with_well.out.press.00010.pfb"

# base case single column
dir_name = get_absolute_path("test_output/single_column_1")
mkdir(dir_name)
var_dz_with_well.run(working_directory=dir_name)

base_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")

# single column test 1
dir_name = get_absolute_path("test_output/single_column_2")
mkdir(dir_name)

var_dz_with_well.ComputationalGrid.DZ = 10.0
var_dz_with_well.Geom.domain.Upper.Z = 140.0
var_dz_with_well.dzScale.nzListNumber = 14
var_dz_with_well.Cell._0.dzScale.Value = 0.1
var_dz_with_well.Cell._1.dzScale.Value = 0.2
var_dz_with_well.Cell._2.dzScale.Value = 0.1
var_dz_with_well.Cell._3.dzScale.Value = 0.1
var_dz_with_well.Cell._4.dzScale.Value = 0.1
var_dz_with_well.Cell._5.dzScale.Value = 0.1
var_dz_with_well.Cell._6.dzScale.Value = 0.1
var_dz_with_well.Cell._7.dzScale.Value = 0.1
var_dz_with_well.Cell._8.dzScale.Value = 0.1
var_dz_with_well.Cell._9.dzScale.Value = 0.1
var_dz_with_well.Cell._10.dzScale.Value = 0.1
var_dz_with_well.Cell._11.dzScale.Value = 0.1
var_dz_with_well.Cell._12.dzScale.Value = 0.1
var_dz_with_well.Cell._13.dzScale.Value = 0.1

var_dz_with_well.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
# assert(np.allclose(base_case_pressure, test_case_pressure))

# single column test 2
dir_name = get_absolute_path("test_output/single_column_3")
mkdir(dir_name)

var_dz_with_well.ComputationalGrid.DZ = 0.1
var_dz_with_well.Geom.domain.Upper.Z = 1.4
var_dz_with_well.Cell._0.dzScale.Value = 10
var_dz_with_well.Cell._1.dzScale.Value = 20
var_dz_with_well.Cell._2.dzScale.Value = 10
var_dz_with_well.Cell._3.dzScale.Value = 10
var_dz_with_well.Cell._4.dzScale.Value = 10
var_dz_with_well.Cell._5.dzScale.Value = 10
var_dz_with_well.Cell._6.dzScale.Value = 10
var_dz_with_well.Cell._7.dzScale.Value = 10
var_dz_with_well.Cell._8.dzScale.Value = 10
var_dz_with_well.Cell._9.dzScale.Value = 10
var_dz_with_well.Cell._10.dzScale.Value = 10
var_dz_with_well.Cell._11.dzScale.Value = 10
var_dz_with_well.Cell._12.dzScale.Value = 10
var_dz_with_well.Cell._13.dzScale.Value = 10

var_dz_with_well.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert np.allclose(base_case_pressure, test_case_pressure)

# Next we switch to a multicolumn setup and add a flux well in to make sure this works for both types
# of wells
var_dz_with_well.ComputationalGrid.NX = 2
var_dz_with_well.ComputationalGrid.NY = 2
var_dz_with_well.Process.Topology.P = 2
var_dz_with_well.Process.Topology.Q = 2
var_dz_with_well.Geom.domain.Upper.X = 2.0
var_dz_with_well.Geom.domain.Upper.Y = 2.0



var_dz_with_well.Wells.Names = "pressure_well flux_well"

var_dz_with_well.Wells.pressure_well.InputType = "Vertical"
var_dz_with_well.Wells.pressure_well.Action = "Extraction"
var_dz_with_well.Wells.pressure_well.Type = "Pressure"
var_dz_with_well.Wells.pressure_well.X = 0.5
var_dz_with_well.Wells.pressure_well.Y = 0.5
var_dz_with_well.Wells.pressure_well.ZUpper = 10.5
var_dz_with_well.Wells.pressure_well.ZLower = 0.5
var_dz_with_well.Wells.pressure_well.Method = "Standard"
var_dz_with_well.Wells.pressure_well.Cycle = "constant"
var_dz_with_well.Wells.pressure_well.alltime.Pressure.Value = 0.5
var_dz_with_well.Wells.pressure_well.alltime.Saturation.water.Value = 1.0

var_dz_with_well.Wells.flux_well.InputType = "Vertical"
var_dz_with_well.Wells.flux_well.Type = "Flux"
var_dz_with_well.Wells.flux_well.Action = "Extraction"
var_dz_with_well.Wells.flux_well.Cycle = "constant"
var_dz_with_well.Wells.flux_well.X = 1.5
var_dz_with_well.Wells.flux_well.Y = 0.5
var_dz_with_well.Wells.flux_well.ZLower = 0.5
var_dz_with_well.Wells.flux_well.ZUpper = 10.5
var_dz_with_well.Wells.flux_well.Method = "Standard"
var_dz_with_well.Wells.flux_well.alltime.Flux.water.Value = 7.5

# Multi column  base case
dir_name = get_absolute_path("test_output/multi_column_1")
mkdir(dir_name)

var_dz_with_well.ComputationalGrid.DZ = 1.0
var_dz_with_well.Geom.domain.Upper.Z = 14.0
var_dz_with_well.dzScale.nzListNumber = 14
var_dz_with_well.Cell._0.dzScale.Value = 1
var_dz_with_well.Cell._1.dzScale.Value = 2.0
var_dz_with_well.Cell._2.dzScale.Value = 1.0
var_dz_with_well.Cell._3.dzScale.Value = 1.0
var_dz_with_well.Cell._4.dzScale.Value = 1.0
var_dz_with_well.Cell._5.dzScale.Value = 1.0
var_dz_with_well.Cell._6.dzScale.Value = 1.0
var_dz_with_well.Cell._7.dzScale.Value = 1.0
var_dz_with_well.Cell._8.dzScale.Value = 1.0
var_dz_with_well.Cell._9.dzScale.Value = 1
var_dz_with_well.Cell._10.dzScale.Value = 1
var_dz_with_well.Cell._11.dzScale.Value = 1
var_dz_with_well.Cell._12.dzScale.Value = 1
var_dz_with_well.Cell._13.dzScale.Value = 1

var_dz_with_well.run(working_directory=dir_name)
base_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")

dir_name = get_absolute_path("test_output/multi_column_2")
mkdir(dir_name)

var_dz_with_well.ComputationalGrid.DZ = 10.0
var_dz_with_well.Geom.domain.Upper.Z = 140.0
var_dz_with_well.dzScale.nzListNumber = 14
var_dz_with_well.Cell._0.dzScale.Value = 0.1
var_dz_with_well.Cell._1.dzScale.Value = 0.2
var_dz_with_well.Cell._2.dzScale.Value = 0.1
var_dz_with_well.Cell._3.dzScale.Value = 0.1
var_dz_with_well.Cell._4.dzScale.Value = 0.1
var_dz_with_well.Cell._5.dzScale.Value = 0.1
var_dz_with_well.Cell._6.dzScale.Value = 0.1
var_dz_with_well.Cell._7.dzScale.Value = 0.1
var_dz_with_well.Cell._8.dzScale.Value = 0.1
var_dz_with_well.Cell._9.dzScale.Value = 0.1
var_dz_with_well.Cell._10.dzScale.Value = 0.1
var_dz_with_well.Cell._11.dzScale.Value = 0.1
var_dz_with_well.Cell._12.dzScale.Value = 0.1
var_dz_with_well.Cell._13.dzScale.Value = 0.1

var_dz_with_well.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
assert np.allclose(base_case_pressure, test_case_pressure)

dir_name = get_absolute_path("test_output/multi_column_3")
mkdir(dir_name)

var_dz_with_well.ComputationalGrid.DZ = 0.1
var_dz_with_well.Geom.domain.Upper.Z = 1.4
var_dz_with_well.Cell._0.dzScale.Value = 10
var_dz_with_well.Cell._1.dzScale.Value = 20
var_dz_with_well.Cell._2.dzScale.Value = 10
var_dz_with_well.Cell._3.dzScale.Value = 10
var_dz_with_well.Cell._4.dzScale.Value = 10
var_dz_with_well.Cell._5.dzScale.Value = 10
var_dz_with_well.Cell._6.dzScale.Value = 10
var_dz_with_well.Cell._7.dzScale.Value = 10
var_dz_with_well.Cell._8.dzScale.Value = 10
var_dz_with_well.Cell._9.dzScale.Value = 10
var_dz_with_well.Cell._10.dzScale.Value = 10
var_dz_with_well.Cell._11.dzScale.Value = 10
var_dz_with_well.Cell._12.dzScale.Value = 10
var_dz_with_well.Cell._13.dzScale.Value = 10

var_dz_with_well.run(working_directory=dir_name)

test_case_pressure = pf.read_pfb(f"{dir_name}/{pressure_file}")
passed = np.allclose(base_case_pressure, test_case_pressure)


if passed:
    print(f"vardz_with_well : PASSED")
else:
    print(f"vardz_with_well : FAILED")
