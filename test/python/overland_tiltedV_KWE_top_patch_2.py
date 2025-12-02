# -----------------------------------------------------------------------------
# Running different configurations of tilted V with a seepage patch in the
# middle of the channel
# -----------------------------------------------------------------------------

import os
import sys
import numpy as np
from parflow import Run
from parflow.tools.fs import cp, mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file
from parflow.tools.io import write_pfb
from make_solid_files import make_solid_file

overland = Run("overland_tiltedV_KWE_top_patch_2", __file__)

# -----------------------------------------------------------------------------

overland.FileVersion = 4

overland.Process.Topology.P = 1
overland.Process.Topology.Q = 1
overland.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = 5
overland.ComputationalGrid.NY = 5
overland.ComputationalGrid.NZ = 2

overland.ComputationalGrid.DX = 10.0
overland.ComputationalGrid.DY = 10.0
overland.ComputationalGrid.DZ = 0.025

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

overland.GeomInput.Names = "domaininput"

overland.GeomInput.domaininput.GeomNames = "domain"
overland.GeomInput.domaininput.InputType = "SolidFile"
overland.GeomInput.domaininput.FileName = "new_solid_tiltedV.pfsol"
overland.Geom.domain.Patches = "bottom side slope channel seepage"


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
overland.TimingInfo.BaseUnit = 0.05
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = 2.0
overland.TimingInfo.DumpInterval = -2
overland.TimeStep.Type = "Constant"
overland.TimeStep.Value = 0.05

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

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland.Cycle.rainrec.Names = "rain rec"
overland.Cycle.rainrec.rain.Length = 2
overland.Cycle.rainrec.rec.Length = 300
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

overland.Patch.slope.BCPressure.Type = "OverlandKinematic"
overland.Patch.slope.BCPressure.Cycle = "rainrec"
overland.Patch.slope.BCPressure.rain.Value = -0.01
overland.Patch.slope.BCPressure.rec.Value = 0.0000
overland.Patch.slope.BCPressure.Seepage = False

overland.Patch.channel.BCPressure.Type = "OverlandKinematic"
overland.Patch.channel.BCPressure.Cycle = "rainrec"
overland.Patch.channel.BCPressure.rain.Value = -0.001
overland.Patch.channel.BCPressure.rec.Value = 0.0000
overland.Patch.channel.BCPressure.Seepage = False

overland.Patch.seepage.BCPressure.Type = "OverlandKinematic"
overland.Patch.seepage.BCPressure.Cycle = "rainrec"
overland.Patch.seepage.BCPressure.rain.Value = -0.001
overland.Patch.seepage.BCPressure.rec.Value = 0.0000
overland.Patch.seepage.BCPressure.Seepage = True

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
# Set solver parameters
# -----------------------------------------------------------------------------

overland.Solver = "Richards"
overland.Solver.MaxIter = 2500

overland.Solver.Nonlinear.MaxIter = 100
overland.Solver.Nonlinear.ResidualTol = 1e-10
overland.Solver.Nonlinear.EtaChoice = "EtaConstant"
overland.Solver.Nonlinear.EtaValue = 1e-5
overland.Solver.Nonlinear.UseJacobian = True
overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland.Solver.Nonlinear.StepTol = 1e-20
overland.Solver.Nonlinear.Globalization = "LineSearch"
overland.Solver.Linear.KrylovDimension = 50
overland.Solver.Linear.MaxRestart = 2
overland.Solver.OverlandKinematic.Epsilon = 1e-5

overland.Solver.Linear.Preconditioner = "PFMG"
overland.Solver.PrintSubsurf = False
overland.Solver.Drop = 1e-20
overland.Solver.AbsTol = 1e-10

overland.Solver.WriteSiloSubsurfData = False
overland.Solver.WriteSiloPressure = False
overland.Solver.WriteSiloSlopes = False

overland.Solver.WriteSiloSaturation = False
overland.Solver.WriteSiloConcentration = False

overland.Solver.PrintTop = True


# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland.ICPressure.Type = "HydroStaticPatch"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.Value = -3.0

overland.Geom.domain.ICPressure.RefGeom = "domain"

overland.Geom.domain.ICPressure.RefPatch = "bottom"


# -----------------------------------------------------------------------------
# New kinematic formulations with a seepage patch in the channel (non-zero channel)
# -----------------------------------------------------------------------------
runcheck = 1
correct_output_dir_name = get_absolute_path("../correct_output")
test_dir = get_absolute_path("test_output/overland_tiltedV_KWE_top_patch_2/")
overland.Solver.PrintSlopes = True

# Ypos

# Define slopes with PFBs
gridnz = overland.ComputationalGrid.NZ
gridny = overland.ComputationalGrid.NY
gridnx = overland.ComputationalGrid.NX

slopex = np.full((gridnz, gridny, gridnx), 1.0)
slopex[:, :, 0 : int(np.floor(gridnx / 2))] = np.round(-0.01, 2)
slopex[:, :, int(np.floor(gridnx / 2)) :] = np.round(0.01, 2)

slopey = np.ones((gridnz, gridny, gridnx))
slopey = np.round(slopey * 0.01, 2)

overland.TopoSlopesX.Type = "PFBFile"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.FileName = "slopex.pfb"

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slopey.pfb"

run_name = "TiltedV_OverlandKin_TopPatch2_Ypos"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
new_output_dir_name = get_absolute_path(os.path.join(test_dir, f"{run_name}"))
mkdir(new_output_dir_name)

write_pfb(os.path.join(new_output_dir_name, "slopex.pfb"), slopex)
overland.dist(os.path.join(new_output_dir_name, "slopex.pfb"))

write_pfb(os.path.join(new_output_dir_name, "slopey.pfb"), slopey)
overland.dist(os.path.join(new_output_dir_name, "slopey.pfb"))

# Solid file with river in y direction
solid_fname = "new_solid_tiltedV"
make_solid_file(
    nx=5,
    ny=5,
    bottom_val=1,
    side_val=2,
    top_val1=3,
    top_val2=4,
    top_val3=5,
    latsize=10.0,
    zdepth=0.049999999999,
    river_dir=2,
    root_name=solid_fname,
    out_dir=new_output_dir_name,
    pftools_path=os.path.join(os.environ["PARFLOW_DIR"], "bin"),
)

overland.run(working_directory=new_output_dir_name, skip_validation=True)
if runcheck == 1:
    passed = True
    for i in range(20):
        timestep = str(i).rjust(5, "0")
        filename = f"/{run_name}.out.press.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Pressure for timestep {timestep}",
        ):
            passed = False
        filename = f"/{run_name}.out.satur.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Saturation for timestep {timestep}",
        ):
            passed = False

    if passed:
        print(f"{run_name} : PASSED")
    else:
        print(f"{run_name} : FAILED")
        sys.exit(1)

# Yneg

# Define slopes with PFBs
gridnz = overland.ComputationalGrid.NZ
gridny = overland.ComputationalGrid.NY
gridnx = overland.ComputationalGrid.NX

slopex = np.full((gridnz, gridny, gridnx), 1.0)
slopex[:, :, 0 : int(np.floor(gridnx / 2))] = np.round(-0.01, 2)
slopex[:, :, int(np.floor(gridnx / 2)) :] = np.round(0.01, 2)

slopey = np.ones((gridnz, gridny, gridnx))
slopey = np.round(slopey * -0.01, 2)

overland.TopoSlopesX.Type = "PFBFile"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.FileName = "slopex.pfb"

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slopey.pfb"

run_name = "TiltedV_OverlandKin_TopPatch2_Yneg"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
new_output_dir_name = get_absolute_path(os.path.join(test_dir, f"{run_name}"))
mkdir(new_output_dir_name)

write_pfb(os.path.join(new_output_dir_name, "slopex.pfb"), slopex)
overland.dist(os.path.join(new_output_dir_name, "slopex.pfb"))

write_pfb(os.path.join(new_output_dir_name, "slopey.pfb"), slopey)
overland.dist(os.path.join(new_output_dir_name, "slopey.pfb"))

# Same solid file as Ypos
solid_fname = "new_solid_tiltedV"
make_solid_file(
    nx=5,
    ny=5,
    bottom_val=1,
    side_val=2,
    top_val1=3,
    top_val2=4,
    top_val3=5,
    latsize=10.0,
    zdepth=0.049999999999,
    river_dir=2,
    root_name=solid_fname,
    out_dir=new_output_dir_name,
    pftools_path=os.path.join(os.environ["PARFLOW_DIR"], "bin"),
)

overland.run(working_directory=new_output_dir_name, skip_validation=True)
if runcheck == 1:
    passed = True
    for i in range(20):
        timestep = str(i).rjust(5, "0")
        filename = f"/{run_name}.out.press.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Pressure for timestep {timestep}",
        ):
            passed = False
        filename = f"/{run_name}.out.satur.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Saturation for timestep {timestep}",
        ):
            passed = False

    if passed:
        print(f"{run_name} : PASSED")
    else:
        print(f"{run_name} : FAILED")
        sys.exit(1)


# Xpos

# Define slopes with PFBs
gridnz = overland.ComputationalGrid.NZ
gridny = overland.ComputationalGrid.NY
gridnx = overland.ComputationalGrid.NX

slopey = np.full((gridnz, gridny, gridnx), 1.0)
slopey[:, 0 : int(np.floor(gridny / 2)), :] = np.round(-0.01, 2)
slopey[:, int(np.floor(gridny / 2)) :, :] = np.round(0.01, 2)

slopex = np.ones((gridnz, gridny, gridnx))
slopex = np.round(slopex * 0.01, 2)

overland.TopoSlopesX.Type = "PFBFile"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.FileName = "slopex.pfb"

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slopey.pfb"

run_name = "TiltedV_OverlandKin_TopPatch2_Xpos"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
new_output_dir_name = get_absolute_path(os.path.join(test_dir, f"{run_name}"))
mkdir(new_output_dir_name)

write_pfb(os.path.join(new_output_dir_name, "slopex.pfb"), slopex)
overland.dist(os.path.join(new_output_dir_name, "slopex.pfb"))

write_pfb(os.path.join(new_output_dir_name, "slopey.pfb"), slopey)
overland.dist(os.path.join(new_output_dir_name, "slopey.pfb"))

# Solid file with river in x direction
solid_fname = "new_solid_tiltedV"
make_solid_file(
    nx=5,
    ny=5,
    bottom_val=1,
    side_val=2,
    top_val1=3,
    top_val2=4,
    top_val3=5,
    latsize=10.0,
    zdepth=0.049999999999,
    river_dir=1,
    root_name=solid_fname,
    out_dir=new_output_dir_name,
    pftools_path=os.path.join(os.environ["PARFLOW_DIR"], "bin"),
)

overland.run(working_directory=new_output_dir_name, skip_validation=True)
if runcheck == 1:
    passed = True
    for i in range(20):
        timestep = str(i).rjust(5, "0")
        filename = f"/{run_name}.out.press.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Pressure for timestep {timestep}",
        ):
            passed = False
        filename = f"/{run_name}.out.satur.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Saturation for timestep {timestep}",
        ):
            passed = False

    if passed:
        print(f"{run_name} : PASSED")
    else:
        print(f"{run_name} : FAILED")
        sys.exit(1)

# Xneg

# Define slopes with PFBs
gridnz = overland.ComputationalGrid.NZ
gridny = overland.ComputationalGrid.NY
gridnx = overland.ComputationalGrid.NX

slopey = np.full((gridnz, gridny, gridnx), 1.0)
slopey[:, 0 : int(np.floor(gridny / 2)), :] = np.round(-0.01, 2)
slopey[:, int(np.floor(gridny / 2)) :, :] = np.round(0.01, 2)

slopex = np.ones((gridnz, gridny, gridnx))
slopex = np.round(slopex * -0.01, 2)

overland.TopoSlopesX.Type = "PFBFile"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.FileName = "slopex.pfb"

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slopey.pfb"

run_name = "TiltedV_OverlandKin_TopPatch2_Xneg"
overland.set_name(run_name)
print("##########")
print(f"Running {run_name}")
new_output_dir_name = get_absolute_path(os.path.join(test_dir, f"{run_name}"))
mkdir(new_output_dir_name)

write_pfb(os.path.join(new_output_dir_name, "slopex.pfb"), slopex)
overland.dist(os.path.join(new_output_dir_name, "slopex.pfb"))

write_pfb(os.path.join(new_output_dir_name, "slopey.pfb"), slopey)
overland.dist(os.path.join(new_output_dir_name, "slopey.pfb"))

# Same solid file as Xpos
solid_fname = "new_solid_tiltedV"
make_solid_file(
    nx=5,
    ny=5,
    bottom_val=1,
    side_val=2,
    top_val1=3,
    top_val2=4,
    top_val3=5,
    latsize=10.0,
    zdepth=0.049999999999,
    river_dir=1,
    root_name=solid_fname,
    out_dir=new_output_dir_name,
    pftools_path=os.path.join(os.environ["PARFLOW_DIR"], "bin"),
)

overland.run(working_directory=new_output_dir_name, skip_validation=True)
if runcheck == 1:
    passed = True
    for i in range(20):
        timestep = str(i).rjust(5, "0")
        filename = f"/{run_name}.out.press.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Pressure for timestep {timestep}",
        ):
            passed = False
        filename = f"/{run_name}.out.satur.{timestep}.pfb"
        if not pf_test_file(
            new_output_dir_name + filename,
            correct_output_dir_name + filename,
            f"Max difference in Saturation for timestep {timestep}",
        ):
            passed = False

    if passed:
        print(f"{run_name} : PASSED")
    else:
        print(f"{run_name} : FAILED")
        sys.exit(1)
