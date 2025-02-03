#
# Import the ParFlow TCL package
#
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs
import sys


run_name = "example_single"

example_single = Run(run_name, __file__)

correct_output_dir_name = get_absolute_path("correct_output")

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
example_single.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

example_single.Process.Topology.P = 1
example_single.Process.Topology.Q = 1
example_single.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
example_single.ComputationalGrid.Lower.X = -10.0
example_single.ComputationalGrid.Lower.Y = 10.0
example_single.ComputationalGrid.Lower.Z = 1.0

example_single.ComputationalGrid.DX = 8.8888888888888893
example_single.ComputationalGrid.DY = 10.666666666666666
example_single.ComputationalGrid.DZ = 1.0

example_single.ComputationalGrid.NX = 18
example_single.ComputationalGrid.NY = 15
example_single.ComputationalGrid.NZ = 8

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
example_single.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
example_single.GeomInput.domain_input.InputType = "Box"
example_single.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
example_single.Geom.domain.Lower.X = -10.0
example_single.Geom.domain.Lower.Y = 10.0
example_single.Geom.domain.Lower.Z = 1.0

example_single.Geom.domain.Upper.X = 150.0
example_single.Geom.domain.Upper.Y = 170.0
example_single.Geom.domain.Upper.Z = 9.0

example_single.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Background Geometry Input
# -----------------------------------------------------------------------------
example_single.GeomInput.background_input.InputType = "Box"
example_single.GeomInput.background_input.GeomName = "background"

# -----------------------------------------------------------------------------
# Background Geometry
# -----------------------------------------------------------------------------
example_single.Geom.background.Lower.X = -99999999.0
example_single.Geom.background.Lower.Y = -99999999.0
example_single.Geom.background.Lower.Z = -99999999.0

example_single.Geom.background.Upper.X = 99999999.0
example_single.Geom.background.Upper.Y = 99999999.0
example_single.Geom.background.Upper.Z = 99999999.0

# -----------------------------------------------------------------------------
# Source_Region Geometry Input
# -----------------------------------------------------------------------------
example_single.GeomInput.source_region_input.InputType = "Box"
example_single.GeomInput.source_region_input.GeomName = "source_region"

# -----------------------------------------------------------------------------
# Source_Region Geometry
# -----------------------------------------------------------------------------
example_single.Geom.source_region.Lower.X = 65.56
example_single.Geom.source_region.Lower.Y = 79.34
example_single.Geom.source_region.Lower.Z = 4.5

example_single.Geom.source_region.Upper.X = 74.44
example_single.Geom.source_region.Upper.Y = 89.99
example_single.Geom.source_region.Upper.Z = 5.5


# -----------------------------------------------------------------------------
# Concen_Region Geometry Input
# -----------------------------------------------------------------------------
example_single.GeomInput.concen_region_input.InputType = "Box"
example_single.GeomInput.concen_region_input.GeomName = "concen_region"

# -----------------------------------------------------------------------------
# Concen_Region Geometry
# -----------------------------------------------------------------------------
example_single.Geom.concen_region.Lower.X = 60.0
example_single.Geom.concen_region.Lower.Y = 80.0
example_single.Geom.concen_region.Lower.Z = 4.0

example_single.Geom.concen_region.Upper.X = 80.0
example_single.Geom.concen_region.Upper.Y = 100.0
example_single.Geom.concen_region.Upper.Z = 6.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
example_single.Geom.Perm.Names = "background"

example_single.Geom.background.Perm.Type = "Constant"
example_single.Geom.background.Perm.Value = 4.0

example_single.Perm.TensorType = "TensorByGeom"

example_single.Geom.Perm.TensorByGeom.Names = "background"

example_single.Geom.background.Perm.TensorValX = 1.0
example_single.Geom.background.Perm.TensorValY = 1.0
example_single.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

example_single.Phase.Names = "water"

example_single.Phase.water.Density.Type = "Constant"
example_single.Phase.water.Density.Value = 1.0

example_single.Phase.water.Viscosity.Type = "Constant"
example_single.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
example_single.Contaminants.Names = "tce"
example_single.Contaminants.tce.Degradation.Value = 0.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

example_single.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

example_single.TimingInfo.BaseUnit = 1.0
example_single.TimingInfo.StartCount = 0
example_single.TimingInfo.StartTime = 0.0
example_single.TimingInfo.StopTime = 1000.0
example_single.TimingInfo.DumpInterval = -1

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

example_single.Geom.Porosity.GeomNames = "background"

example_single.Geom.background.Porosity.Type = "Constant"
example_single.Geom.background.Porosity.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
example_single.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
example_single.Phase.water.Mobility.Type = "Constant"
example_single.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
example_single.Geom.Retardation.GeomNames = "background"
example_single.Geom.background.tce.Retardation.Type = "Linear"
example_single.Geom.background.tce.Retardation.Rate = 0.0

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
example_single.Cycle.Names = "constant"
example_single.Cycle.constant.Names = "alltime"
example_single.Cycle.constant.alltime.Length = 1
example_single.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
example_single.Wells.Names = "snoopy"

example_single.Wells.snoopy.InputType = "Recirc"

example_single.Wells.snoopy.Cycle = "constant"

example_single.Wells.snoopy.ExtractionType = "Flux"
example_single.Wells.snoopy.InjectionType = "Flux"

example_single.Wells.snoopy.X = 71.0
example_single.Wells.snoopy.Y = 90.0
example_single.Wells.snoopy.ExtractionZLower = 5.0
example_single.Wells.snoopy.ExtractionZUpper = 5.0
example_single.Wells.snoopy.InjectionZLower = 2.0
example_single.Wells.snoopy.InjectionZUpper = 2.0

example_single.Wells.snoopy.ExtractionMethod = "Standard"
example_single.Wells.snoopy.InjectionMethod = "Standard"

example_single.Wells.snoopy.alltime.Extraction.Flux.water.Value = 5.0
example_single.Wells.snoopy.alltime.Injection.Flux.water.Value = 7.5
example_single.Wells.snoopy.alltime.Injection.Concentration.water.tce.Fraction = 0.1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
example_single.BCPressure.PatchNames = "left right front back bottom top"

example_single.Patch.left.BCPressure.Type = "DirEquilRefPatch"
example_single.Patch.left.BCPressure.Cycle = "constant"
example_single.Patch.left.BCPressure.RefGeom = "domain"
example_single.Patch.left.BCPressure.RefPatch = "bottom"
example_single.Patch.left.BCPressure.alltime.Value = 14.0

example_single.Patch.right.BCPressure.Type = "DirEquilRefPatch"
example_single.Patch.right.BCPressure.Cycle = "constant"
example_single.Patch.right.BCPressure.RefGeom = "domain"
example_single.Patch.right.BCPressure.RefPatch = "bottom"
example_single.Patch.right.BCPressure.alltime.Value = 9.0

example_single.Patch.front.BCPressure.Type = "FluxConst"
example_single.Patch.front.BCPressure.Cycle = "constant"
example_single.Patch.front.BCPressure.alltime.Value = 0.0

example_single.Patch.back.BCPressure.Type = "FluxConst"
example_single.Patch.back.BCPressure.Cycle = "constant"
example_single.Patch.back.BCPressure.alltime.Value = 0.0

example_single.Patch.bottom.BCPressure.Type = "FluxConst"
example_single.Patch.bottom.BCPressure.Cycle = "constant"
example_single.Patch.bottom.BCPressure.alltime.Value = 0.0

example_single.Patch.top.BCPressure.Type = "FluxConst"
example_single.Patch.top.BCPressure.Cycle = "constant"
example_single.Patch.top.BCPressure.alltime.Value = 0.0

# #-----------------------------------------------------------------------------
# # Boundary Conditions: Temperature
# #-----------------------------------------------------------------------------
# example_single.BCTemperature.PatchNames = 'left right front back bottom top'
# #
# example_single.Patch.left.BCTemperature.Type = 'DirConst'
# example_single.Patch.left.BCTemperature.Cycle = 'constant'
# example_single.Patch.left.BCTemperature.alltime.Value = 288.15
# #
# example_single.Patch.right.BCTemperature.Type = 'DirConst'
# example_single.Patch.right.BCTemperature.Cycle = 'constant'
# example_single.Patch.right.BCTemperature.alltime.Value = 295.
# example_single.Patch.right.BCTemperature.alltime.Value = 293.15
# #
# example_single.Patch.front.BCTemperature.Type = 'DirConst'
# example_single.Patch.front.BCTemperature.Type = 'FluxConst'
# example_single.Patch.front.BCTemperature.Cycle = 'constant'
# example_single.Patch.front.BCTemperature.alltime.Value = 305.
# example_single.Patch.front.BCTemperature.alltime.Value = 0.0
# #
# example_single.Patch.back.BCTemperature.Type = 'DirConst'
# example_single.Patch.back.BCTemperature.Type = 'FluxConst'
# example_single.Patch.back.BCTemperature.Cycle = 'constant'
# example_single.Patch.back.BCTemperature.alltime.Value = 295.
# example_single.Patch.back.BCTemperature.alltime.Value = 0.0
# #
# example_single.Patch.bottom.BCTemperature.Type = 'FluxConst'
# example_single.Patch.bottom.BCTemperature.Cycle = 'constant'
# example_single.Patch.bottom.BCTemperature.alltime.Value = 0.0
# #
# example_single.Patch.top.BCTemperature.Type = 'FluxConst'
# example_single.Patch.top.BCTemperature.Cycle = 'constant'
# example_single.Patch.top.BCTemperature.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

example_single.PhaseSources.water.Type = "Constant"
example_single.PhaseSources.water.GeomNames = "background"
example_single.PhaseSources.water.Geom.background.Value = 0.0

example_single.PhaseSources.Type = "Constant"
example_single.PhaseSources.GeomNames = "background"
example_single.PhaseSources.Geom.background.Value = 0.0

example_single.PhaseConcen.water.tce.Type = "Constant"
example_single.PhaseConcen.water.tce.GeomNames = "concen_region"
example_single.PhaseConcen.water.tce.Geom.concen_region.Value = 0.8

# #-----------------------------------------------------------------------------
# # Temperature sources:
# #-----------------------------------------------------------------------------
# example_single.TempSources.Type = 'Constant'
# example_single.TempSources.GeomNames = 'background'
# example_single.TempSources.Geom.background.Value = 0.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

example_single.SpecificStorage.Type = "Constant"
example_single.SpecificStorage.GeomNames = "background"
example_single.Geom.background.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Heat Capacity
# -----------------------------------------------------------------------------

example_single.Phase.water.HeatCapacity.Type = "Constant"
example_single.Phase.water.HeatCapacity.GeomNames = "background"
example_single.Phase.water.Geom.background.HeatCapacity.Value = 4000.0

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

example_single.TopoSlopesX.Type = "Constant"
example_single.TopoSlopesX.GeomNames = "domain"
example_single.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

example_single.TopoSlopesY.Type = "Constant"
example_single.TopoSlopesY.GeomNames = "domain"
example_single.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

example_single.Mannings.Type = "Constant"
example_single.Mannings.GeomNames = "domain"
example_single.Mannings.Geom.domain.Value = 2.3e-7


# -----------------------------------------------------------------------------
# The Solver Impes MaxIter default value changed so to get previous
# results we need to set it back to what it was
# -----------------------------------------------------------------------------
example_single.Solver.MaxIter = 5

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/example_single")
mkdir(new_output_dir_name)
example_single.run(working_directory=new_output_dir_name)

# -----------------------------------------------------------------------------
# Check results for regression
# -----------------------------------------------------------------------------

passed = True

i = 0
timestep = str(i).rjust(5, "0")

sig_digits = 4
abs_value = 1e-12

test_files = ["press"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in {filename}",
        sig_digits,
        abs_value,
    ):
        passed = False

test_files = ["perm_x", "perm_y", "perm_z"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file_with_abs(
        new_output_dir_name + filename,
        correct_output_dir_name + filename,
        f"Max difference in {filename}",
        sig_digits,
        abs_value,
    ):
        passed = False

# This doesn't work yet, test does not work for pfsb files.
# test_files = ["concen.0.00"]
# for i in range(0,5):
#     timestep = str(i).rjust(5, '0')
#     for test_file in test_files:
#         filename = f"/{run_name}.out.{test_file}.{timestep}.pfsb"
#         if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in {test_file}", sig_digits):
#             passed = False


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------
# pfrun example_single
# pfundist example_single

# -----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
# -----------------------------------------------------------------------------
# if { [info exists ::env(PF_TEST) ] } {
#     set TEST example_single
#     source pftest.tcl
#     set sig_digits 4

#     set passed 1

#     #
#     # Tests
#     #
#     if ![pftestFile $TEST.out.press.00000.pfb "Max difference in Pressure" $sig_digits] {
# 	set passed 0
#     }

#     if ![pftestFile $TEST.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
# 	set passed 0
#     }
#     if ![pftestFile $TEST.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
# 	set passed 0
#     }
#     if ![pftestFile $TEST.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
# 	set passed 0
#     }
#
#     foreach i "00000 00001 00002 00003 00004 00005" {
# 	if ![pftestFile $TEST.out.concen.0.00.$i.pfsb "Max difference in concen timestep $i" $sig_digits] {
# 	    set passed 0
# 	}
#     }

#     if $passed {
# 	puts "$TEST : PASSED"
#     } {
# 	puts "$TEST : FAILED"
#     }
# }
# example_single.run()
