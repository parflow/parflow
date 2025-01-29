# -----------------------------------------------------------------------------
# running different configurations of tilted V
# -----------------------------------------------------------------------------
import os

import parflow
from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path
from parflow.tools.compare import pf_test_file_with_abs
import sys
import numpy as np

run_name = "tilted_v_with_reservoir_overland_flow"
overland = Run(run_name, __file__)

# -----------------------------------------------------------------------------

overland.FileVersion = 4

overland.Process.Topology.P = 1
overland.Process.Topology.Q = 1
overland.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------
gridnx = int(25)
gridny = int(25)
gridnz = int(1)
griddx = 1.0
griddy = 1.0
griddz = 0.5

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = gridnx
overland.ComputationalGrid.NY = gridny
overland.ComputationalGrid.NZ = gridnz

overland.ComputationalGrid.DX = griddx
overland.ComputationalGrid.DY = griddy
overland.ComputationalGrid.DZ = griddz

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
overland.GeomInput.Names = "domaininput"

overland.GeomInput.domaininput.GeomName = "domain"
overland.GeomInput.domaininput.GeomNames = "domain"
overland.GeomInput.domaininput.InputType = "SolidFile"
overland.Geom.domain.Patches = "bottom side slope1 channel slope2"

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
overland.Geom.Perm.Names = "domain"
overland.Geom.domain.Perm.Type = "Constant"
overland.Geom.domain.Perm.Value = 0.0000001

overland.Perm.TensorType = "TensorByGeom"

overland.Geom.Perm.TensorByGeom.Names = "domain"

overland.Geom.domain.Perm.TensorValX = 1.0
overland.Geom.domain.Perm.TensorValY = 1.0
overland.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
overland.SpecificStorage.Type = "Constant"
overland.SpecificStorage.GeomNames = "domain"
overland.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------
overland.Phase.Names = "water"

overland.Phase.water.Density.Type = "Constant"
overland.Phase.water.Density.Value = 1.0

overland.Phase.water.Viscosity.Type = "Constant"
overland.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
overland.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
overland.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------
overland.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------
overland.TimingInfo.BaseUnit = 1
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = 11.0
overland.TimingInfo.DumpInterval = 1.0
overland.TimeStep.Type = "Constant"
overland.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

overland.Geom.Porosity.GeomNames = "domain"
overland.Geom.domain.Porosity.Type = "Constant"
overland.Geom.domain.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
overland.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------
overland.Phase.RelPerm.Type = "VanGenuchten"
overland.Phase.RelPerm.GeomNames = "domain"

overland.Geom.domain.RelPerm.Alpha = 6.0
overland.Geom.domain.RelPerm.N = 2.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------
overland.Phase.Saturation.Type = "VanGenuchten"
overland.Phase.Saturation.GeomNames = "domain"

overland.Geom.domain.Saturation.Alpha = 6.0
overland.Geom.domain.Saturation.N = 2.0
overland.Geom.domain.Saturation.SRes = 0.2
overland.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
overland.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
overland.Cycle.Names = "constant rainrec"
overland.Cycle.constant.Names = "alltime"
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

overland.Cycle.rainrec.Names = "rain"
overland.Cycle.rainrec.rain.Length = 1
overland.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

overland.BCPressure.PatchNames = overland.Geom.domain.Patches

overland.Patch.side.BCPressure.Type = "FluxConst"
overland.Patch.side.BCPressure.Cycle = "constant"
overland.Patch.side.BCPressure.alltime.Value = 0.0

overland.Patch.bottom.BCPressure.Type = "FluxConst"
overland.Patch.bottom.BCPressure.Cycle = "constant"
overland.Patch.bottom.BCPressure.alltime.Value = 0.0

overland.Patch.slope1.BCPressure.Cycle = "constant"
overland.Patch.slope1.BCPressure.alltime.Value = -0.001

overland.Patch.slope2.BCPressure.Cycle = "constant"
overland.Patch.slope2.BCPressure.alltime.Value = -0.001

overland.Patch.channel.BCPressure.Cycle = "constant"
overland.Patch.channel.BCPressure.alltime.Value = -0.001

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
overland.Mannings.Type = "Constant"
overland.Mannings.GeomNames = "domain"
overland.Mannings.Geom.domain.Value = 3.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------
overland.PhaseSources.water.Type = "Constant"
overland.PhaseSources.water.GeomNames = "domain"
overland.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------
overland.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Slopes
# -----------------------------------------------------------------------------
overland.TopoSlopesX.Type = "PFBFile"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.FileName = "slopex.pfb"

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slopey.pfb"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------
overland.Solver = "Richards"
overland.Solver.MaxIter = 2500

overland.Solver.Nonlinear.MaxIter = 100
overland.Solver.Nonlinear.ResidualTol = 1e-9
overland.Solver.Nonlinear.EtaChoice = "EtaConstant"
overland.Solver.Nonlinear.EtaValue = 0.01
overland.Solver.Nonlinear.UseJacobian = False
overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland.Solver.Nonlinear.StepTol = 1e-20
overland.Solver.Nonlinear.Globalization = "LineSearch"
overland.Solver.Linear.KrylovDimension = 50
overland.Solver.Linear.MaxRestart = 2


overland.Solver.Linear.Preconditioner = "PFMG"
overland.Solver.PrintSubsurf = True
overland.Solver.PrintSlopes = True
overland.Solver.PrintMannings = True
overland.Solver.Drop = 1e-20
overland.Solver.AbsTol = 1e-10

overland.Solver.WriteSiloSubsurfData = False
overland.Solver.WriteSiloPressure = False
overland.Solver.WriteSiloSlopes = False

overland.Solver.WriteSiloSaturation = False
overland.Solver.WriteSiloConcentration = False


# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------
# set water table to be at the bottom of the domain, the top layer is initially dry
overland.ICPressure.Type = "PFBFile"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.FileName = (
    "tilted_v_with_reservoir_initial_pressure.pfb"
)

overland.Geom.domain.ICPressure.RefGeom = "domain"
overland.Geom.domain.ICPressure.RefPatch = "bottom"


# -----------------------------------------------------------------------------
# defining write function to save some space in the loop
# -----------------------------------------------------------------------------
def write_pfb_to_run_dir(myarray, fout, run_dir):
    fout = get_absolute_path(os.path.join(run_dir, fout))
    parflow.write_pfb(fout, myarray)


def dist_and_run(run_dir):
    overland.dist(os.path.join(run_dir, "slopex.pfb"))
    overland.dist(os.path.join(run_dir, "slopey.pfb"))

    overland.write(working_directory=run_dir)
    overland.write(file_format="yaml", working_directory=run_dir)
    overland.write(file_format="json", working_directory=run_dir)
    overland.run(working_directory=run_dir)


# Make an x direction solid file
inputs_dir = get_absolute_path("../../test/input")
solid_name = f"{inputs_dir}/tilted_v_with_reservoir"
solid_fname = solid_name + ".pfsol"

# -----------------------------------------------------------------------------
# setup the overland patch types
# -----------------------------------------------------------------------------
overland.Patch.slope1.BCPressure.Type = "OverlandFlow"
overland.Patch.slope2.BCPressure.Type = "OverlandFlow"
overland.Patch.channel.BCPressure.Type = "OverlandFlow"

# -----------------------------------------------------------------------------
# Draining right-- (negative X slope channel)
# -----------------------------------------------------------------------------
correct_output_dir = get_absolute_path("../correct_output")
print(correct_output_dir)


run_dir = get_absolute_path("test_output/tilted_v_with_reservoir_overland_flow")

mkdir(run_dir)

cp(solid_fname, run_dir)
cp(f"{inputs_dir}/tilted_v_with_reservoir_initial_pressure.pfb", run_dir)
cp(f"{inputs_dir}/tilted_v_with_reservoir_initial_pressure.pfb.dist", run_dir)
overland.GeomInput.domaininput.FileName = solid_fname

# Make slope files
MIDDLE_ROW = int(np.floor(gridny / 2))
slopey = np.full((gridnz, gridny, gridnx), 1.0)
slopey[:, 0:MIDDLE_ROW, :] = np.round(-0.01, 2)
slopey[:, MIDDLE_ROW + 1 :, :] = np.round(0.01, 2)
slopey[:, MIDDLE_ROW, :] = np.round(0.00, 2)
write_pfb_to_run_dir(slopey, "slopey.pfb", run_dir)

slopex = np.ones((gridnz, gridny, gridnx))
slopex = np.round(slopex * -0.01, 2)
write_pfb_to_run_dir(slopex, "slopex.pfb", run_dir)

overland.Reservoirs.Names = "reservoir"
overland.Reservoirs.Overland_Flow_Solver = "OverlandFlow"
overland.Reservoirs.reservoir.Intake_X = 12.5
overland.Reservoirs.reservoir.Intake_Y = 12.5
overland.Reservoirs.reservoir.Release_X = 13.5
overland.Reservoirs.reservoir.Release_Y = 12.5
overland.Reservoirs.reservoir.Has_Secondary_Intake_Cell = 0
overland.Reservoirs.reservoir.Secondary_Intake_X = -1
overland.Reservoirs.reservoir.Secondary_Intake_Y = -1

overland.Reservoirs.reservoir.Max_Storage = 100
overland.Reservoirs.reservoir.Storage = 50
overland.Reservoirs.reservoir.Min_Release_Storage = 0
overland.Reservoirs.reservoir.Release_Rate = 0

dist_and_run(run_dir)

passed = True

i = 10
timestep = str(i).rjust(5, "0")

sig_digits = 8
abs_value = 1e-12

test_files = ["press"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
    if not pf_test_file_with_abs(
        run_dir + filename,
        correct_output_dir + filename,
        f"Max difference in {filename}",
        abs_value,
        sig_digits,
    ):
        passed = False


if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
