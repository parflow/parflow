# ---------------------------------------------------------
#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR
# ---------------------------------------------------------


# ---------------------------------------------------------
# Import ParFlow
# ---------------------------------------------------------

import sys
import os
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb, ParflowBinaryReader
from parflow.tools.top import compute_top
from parflow.tools import hydrology
from parflow.tools.compare import pf_test_equal

# ---------------------------------------------------------
# Name of the run
# ---------------------------------------------------------
run_name = "water_balance"

wbx = Run(run_name, __file__)

# ---------------------------------------------------------
# Some controls for the test
# ---------------------------------------------------------

# ---------------------------------------------------------
# Control slopes
# -1 = slope to lower-x
#  0 = flat top (no overland flow)
#  1 = slope to upper-x
# ---------------------------------------------------------

use_slopes = 1

# ---------------------------------------------------------
# Flux on the top surface
# ---------------------------------------------------------

rain_flux = -0.05
rec_flux = 0.0

# ---------------------------------------------------------

wbx.FileVersion = 4

# ---------------------------------------------------------
# Processor topology
# ---------------------------------------------------------

wbx.Process.Topology.P = 1
wbx.Process.Topology.Q = 1
wbx.Process.Topology.R = 1

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

wbx.ComputationalGrid.Lower.X = 0.0
wbx.ComputationalGrid.Lower.Y = 0.0
wbx.ComputationalGrid.Lower.Z = 0.0

wbx.ComputationalGrid.NX = 30
wbx.ComputationalGrid.NY = 30
wbx.ComputationalGrid.NZ = 30

wbx.ComputationalGrid.DX = 10.0
wbx.ComputationalGrid.DY = 10.0
wbx.ComputationalGrid.DZ = 0.05

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------

wbx.GeomInput.Names = "domaininput leftinput rightinput channelinput"

wbx.GeomInput.domaininput.GeomName = "domain"
wbx.GeomInput.leftinput.GeomName = "left"
wbx.GeomInput.rightinput.GeomName = "right"
wbx.GeomInput.channelinput.GeomName = "channel"

wbx.GeomInput.domaininput.InputType = "Box"
wbx.GeomInput.leftinput.InputType = "Box"
wbx.GeomInput.rightinput.InputType = "Box"
wbx.GeomInput.channelinput.InputType = "Box"

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

wbx.Geom.domain.Lower.X = 0.0
wbx.Geom.domain.Lower.Y = 0.0
wbx.Geom.domain.Lower.Z = 0.0

wbx.Geom.domain.Upper.X = 300.0
wbx.Geom.domain.Upper.Y = 300.0
wbx.Geom.domain.Upper.Z = 1.5
wbx.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# ---------------------------------------------------------
# Left Slope Geometry
# ---------------------------------------------------------

wbx.Geom.left.Lower.X = 0.0
wbx.Geom.left.Lower.Y = 0.0
wbx.Geom.left.Lower.Z = 0.0

wbx.Geom.left.Upper.X = 300.0
wbx.Geom.left.Upper.Y = 140.0
wbx.Geom.left.Upper.Z = 1.5

# ---------------------------------------------------------
# Right Slope Geometry
# ---------------------------------------------------------

wbx.Geom.right.Lower.X = 0.0
wbx.Geom.right.Lower.Y = 160.0
wbx.Geom.right.Lower.Z = 0.0

wbx.Geom.right.Upper.X = 300.0
wbx.Geom.right.Upper.Y = 300.0
wbx.Geom.right.Upper.Z = 1.5

# ---------------------------------------------------------
# Channel Geometry
# ---------------------------------------------------------

wbx.Geom.channel.Lower.X = 0.0
wbx.Geom.channel.Lower.Y = 140.0
wbx.Geom.channel.Lower.Z = 0.0

wbx.Geom.channel.Upper.X = 300.0
wbx.Geom.channel.Upper.Y = 160.0
wbx.Geom.channel.Upper.Z = 1.5

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------

wbx.Geom.Perm.Names = "left right channel"

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

wbx.Geom.left.Perm.Type = "TurnBands"
wbx.Geom.left.Perm.LambdaX = 50.0
wbx.Geom.left.Perm.LambdaY = 50.0
wbx.Geom.left.Perm.LambdaZ = 0.5
wbx.Geom.left.Perm.GeomMean = 0.01

wbx.Geom.left.Perm.Sigma = 0.5
wbx.Geom.left.Perm.NumLines = 40
wbx.Geom.left.Perm.RZeta = 5.0
wbx.Geom.left.Perm.KMax = 100.0
wbx.Geom.left.Perm.DelK = 0.2
wbx.Geom.left.Perm.Seed = 33333
wbx.Geom.left.Perm.LogNormal = "Log"
wbx.Geom.left.Perm.StratType = "Bottom"

wbx.Geom.right.Perm.Type = "TurnBands"
wbx.Geom.right.Perm.LambdaX = 50.0
wbx.Geom.right.Perm.LambdaY = 50.0
wbx.Geom.right.Perm.LambdaZ = 0.5
wbx.Geom.right.Perm.GeomMean = 0.05

wbx.Geom.right.Perm.Sigma = 0.5
wbx.Geom.right.Perm.NumLines = 40
wbx.Geom.right.Perm.RZeta = 5.0
wbx.Geom.right.Perm.KMax = 100.0
wbx.Geom.right.Perm.DelK = 0.2
wbx.Geom.right.Perm.Seed = 13333
wbx.Geom.right.Perm.LogNormal = "Log"
wbx.Geom.right.Perm.StratType = "Bottom"

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface

wbx.Geom.left.Perm.Type = "Constant"
wbx.Geom.left.Perm.Value = 0.001

wbx.Geom.right.Perm.Type = "Constant"
wbx.Geom.right.Perm.Value = 0.01

wbx.Geom.channel.Perm.Type = "Constant"
wbx.Geom.channel.Perm.Value = 0.00001

wbx.Perm.TensorType = "TensorByGeom"

wbx.Geom.Perm.TensorByGeom.Names = "domain"

wbx.Geom.domain.Perm.TensorValX = 1.0
wbx.Geom.domain.Perm.TensorValY = 1.0
wbx.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

wbx.SpecificStorage.Type = "Constant"
wbx.SpecificStorage.GeomNames = "domain"
wbx.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

wbx.Phase.Names = "water"

wbx.Phase.water.Density.Type = "Constant"
wbx.Phase.water.Density.Value = 1.0

wbx.Phase.water.Viscosity.Type = "Constant"
wbx.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

wbx.Contaminants.Names = ""

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------

wbx.Geom.Retardation.GeomNames = ""

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

wbx.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

wbx.TimingInfo.BaseUnit = 0.1
wbx.TimingInfo.StartCount = 0
wbx.TimingInfo.StartTime = 0.0
wbx.TimingInfo.StopTime = 2.0
wbx.TimingInfo.DumpInterval = 0.1
wbx.TimeStep.Type = "Constant"
wbx.TimeStep.Value = 0.1

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

wbx.Geom.Porosity.GeomNames = "left right channel"

wbx.Geom.left.Porosity.Type = "Constant"
wbx.Geom.left.Porosity.Value = 0.25

wbx.Geom.right.Porosity.Type = "Constant"
wbx.Geom.right.Porosity.Value = 0.25

wbx.Geom.channel.Porosity.Type = "Constant"
wbx.Geom.channel.Porosity.Value = 0.01

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

wbx.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

wbx.Phase.RelPerm.Type = "VanGenuchten"
wbx.Phase.RelPerm.GeomNames = "domain"

wbx.Geom.domain.RelPerm.Alpha = 0.5
wbx.Geom.domain.RelPerm.N = 3.0

# ---------------------------------------------------------
# Saturation
# ---------------------------------------------------------

wbx.Phase.Saturation.Type = "VanGenuchten"
wbx.Phase.Saturation.GeomNames = "domain"

wbx.Geom.domain.Saturation.Alpha = 0.5
wbx.Geom.domain.Saturation.N = 3.0
wbx.Geom.domain.Saturation.SRes = 0.2
wbx.Geom.domain.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

wbx.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

wbx.Cycle.Names = "constant rainrec"
wbx.Cycle.constant.Names = "alltime"
wbx.Cycle.constant.alltime.Length = 1
wbx.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

wbx.Cycle.rainrec.Names = "r0 r1 r2 r3 r4 r5 r6"
wbx.Cycle.rainrec.r0.Length = 1
wbx.Cycle.rainrec.r1.Length = 1
wbx.Cycle.rainrec.r2.Length = 1
wbx.Cycle.rainrec.r3.Length = 1
wbx.Cycle.rainrec.r4.Length = 1
wbx.Cycle.rainrec.r5.Length = 1
wbx.Cycle.rainrec.r6.Length = 1

wbx.Cycle.rainrec.Repeat = 1
#
# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------

wbx.BCPressure.PatchNames = wbx.Geom.domain.Patches

wbx.Patch.x_lower.BCPressure.Type = "FluxConst"
wbx.Patch.x_lower.BCPressure.Cycle = "constant"
wbx.Patch.x_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.y_lower.BCPressure.Type = "FluxConst"
wbx.Patch.y_lower.BCPressure.Cycle = "constant"
wbx.Patch.y_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.z_lower.BCPressure.Type = "FluxConst"
wbx.Patch.z_lower.BCPressure.Cycle = "constant"
wbx.Patch.z_lower.BCPressure.alltime.Value = 0.0

wbx.Patch.x_upper.BCPressure.Type = "FluxConst"
wbx.Patch.x_upper.BCPressure.Cycle = "constant"
wbx.Patch.x_upper.BCPressure.alltime.Value = 0.0

wbx.Patch.y_upper.BCPressure.Type = "FluxConst"
wbx.Patch.y_upper.BCPressure.Cycle = "constant"
wbx.Patch.y_upper.BCPressure.alltime.Value = 0.0

wbx.Patch.z_upper.BCPressure.Type = "OverlandFlow"
wbx.Patch.z_upper.BCPressure.Cycle = "rainrec"
wbx.Patch.z_upper.BCPressure.r0.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r1.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r2.Value = rain_flux
wbx.Patch.z_upper.BCPressure.r3.Value = rain_flux
wbx.Patch.z_upper.BCPressure.r4.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r5.Value = rec_flux
wbx.Patch.z_upper.BCPressure.r6.Value = rec_flux

# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------

wbx.TopoSlopesX.Type = "Constant"
wbx.TopoSlopesX.GeomNames = "left right channel"
if use_slopes > 0:
    wbx.TopoSlopesX.Geom.left.Value = 0.000
    wbx.TopoSlopesX.Geom.right.Value = 0.000
    wbx.TopoSlopesX.Geom.channel.Value = 0.001 * use_slopes
else:
    wbx.TopoSlopesX.Geom.left.Value = 0.000
    wbx.TopoSlopesX.Geom.right.Value = 0.000
    wbx.TopoSlopesX.Geom.channel.Value = 0.000

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

wbx.TopoSlopesY.Type = "Constant"
wbx.TopoSlopesY.GeomNames = "left right channel"
if use_slopes > 0:
    wbx.TopoSlopesY.Geom.left.Value = -0.005
    wbx.TopoSlopesY.Geom.right.Value = 0.005
    wbx.TopoSlopesY.Geom.channel.Value = 0.000
else:
    wbx.TopoSlopesY.Geom.left.Value = 0.000
    wbx.TopoSlopesY.Geom.right.Value = 0.000
    wbx.TopoSlopesY.Geom.channel.Value = 0.000

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------

wbx.Mannings.Type = "Constant"
wbx.Mannings.GeomNames = "left right channel"
wbx.Mannings.Geom.left.Value = 5.0e-6
wbx.Mannings.Geom.right.Value = 5.0e-6
wbx.Mannings.Geom.channel.Value = 1.0e-6

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

wbx.PhaseSources.water.Type = "Constant"
wbx.PhaseSources.water.GeomNames = "domain"
wbx.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

wbx.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

wbx.Solver = "Richards"
wbx.Solver.MaxIter = 2500

wbx.Solver.AbsTol = 1e-10
wbx.Solver.Nonlinear.MaxIter = 20
wbx.Solver.Nonlinear.ResidualTol = 1e-9
wbx.Solver.Nonlinear.EtaChoice = "Walker1"
wbx.Solver.Nonlinear.EtaChoice = "EtaConstant"
wbx.Solver.Nonlinear.EtaValue = 0.01
wbx.Solver.Nonlinear.UseJacobian = False
wbx.Solver.Nonlinear.DerivativeEpsilon = 1e-8
wbx.Solver.Nonlinear.StepTol = 1e-30
wbx.Solver.Nonlinear.Globalization = "LineSearch"
wbx.Solver.Linear.KrylovDimension = 20
wbx.Solver.Linear.MaxRestart = 2

wbx.Solver.Linear.Preconditioner = "PFMG"
wbx.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
wbx.Solver.Linear.Preconditioner.PFMG.Smoother = "RBGaussSeidelNonSymmetric"
wbx.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
wbx.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1

wbx.Solver.PrintSubsurfData = True
wbx.Solver.PrintConcentration = True
wbx.Solver.PrintSlopes = True
wbx.Solver.PrintEvapTrans = True
wbx.Solver.PrintEvapTransSum = True
wbx.Solver.PrintOverlandSum = True
wbx.Solver.PrintMannings = True
wbx.Solver.PrintSpecificStorage = True

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
wbx.ICPressure.Type = "HydroStaticPatch"
wbx.ICPressure.GeomNames = "domain"

wbx.Geom.domain.ICPressure.Value = -3.0

wbx.Geom.domain.ICPressure.RefGeom = "domain"
wbx.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path("test_output/water_balance_x")
correct_output_dir_name = get_absolute_path("../correct_output")
mkdir(new_output_dir_name)
wbx.run(working_directory=new_output_dir_name)

passed = True
verbose = True

bc_dict = {
    0: wbx.Patch.z_upper.BCPressure.r0.Value,
    1: wbx.Patch.z_upper.BCPressure.r1.Value,
    2: wbx.Patch.z_upper.BCPressure.r2.Value,
    3: wbx.Patch.z_upper.BCPressure.r3.Value,
    4: wbx.Patch.z_upper.BCPressure.r4.Value,
    5: wbx.Patch.z_upper.BCPressure.r5.Value,
    6: wbx.Patch.z_upper.BCPressure.r6.Value,
}

slope_x = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.slope_x.pfb"))
slope_y = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.slope_y.pfb"))
mannings = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.mannings.pfb"))
specific_storage = read_pfb(
    os.path.join(new_output_dir_name, f"{run_name}.out.specific_storage.pfb")
)
porosity = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.porosity.pfb"))
mask = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.mask.pfb"))
top = compute_top(mask)

surface_area_of_domain = (
    wbx.ComputationalGrid.DX
    * wbx.ComputationalGrid.DY
    * wbx.ComputationalGrid.NX
    * wbx.ComputationalGrid.NY
)

prev_total_water_balance = 0.0

for i in range(20):
    if verbose:
        print("======================================================")
        print(f"Timestep {i}")
        print("======================================================")

    total_water_in_domain = 0.0

    filename = os.path.join(new_output_dir_name, f"{run_name}.out.press.{i:05d}.pfb")
    pressure = read_pfb(filename)
    header = ParflowBinaryReader(filename).read_header()
    dx, dy, dz = header["dx"], header["dy"], header["dz"]
    surface_storage = hydrology.calculate_surface_storage(pressure, dx, dy, mask)
    write_pfb(f"surface_storage.{i}.pfb", surface_storage)
    total_surface_storage = np.sum(surface_storage)
    if verbose:
        print(f"Surface storage\t\t\t\t\t : {total_surface_storage:.16e}")

    total_water_in_domain += total_surface_storage

    filename = f"{run_name}.out.satur.{i:05d}.pfb"
    saturation = read_pfb(os.path.join(new_output_dir_name, filename))
    water_table_depth = hydrology.calculate_water_table_depth(saturation, top, dz)
    write_pfb(f"water_table_depth.{i}.pfb", water_table_depth)
    nz = header["nz"]
    subsurface_storage = hydrology.calculate_subsurface_storage(
        porosity,
        pressure,
        saturation,
        specific_storage,
        dx,
        dy,
        np.array([dz] * nz),
        mask,
    )
    write_pfb(f"subsurface_storage.{i}.pfb", subsurface_storage)
    total_subsurface_storage = np.sum(subsurface_storage)
    if verbose:
        print(f"Subsurface storage\t\t\t\t : {total_subsurface_storage:.16e}")

    total_water_in_domain += total_subsurface_storage

    if verbose:
        print(f"Total water in domain\t\t\t\t : {total_water_in_domain:.16e}")
        print("")

    total_surface_runoff = 0.0
    if i > 0:
        surface_runoff = hydrology.calculate_overland_flow(
            pressure,
            slope_x[0],
            slope_y[0],
            mannings[0],
            dx,
            dy,
            flow_method="OverlandFlow",
            mask=mask,
        )
        total_surface_runoff = surface_runoff * wbx.TimingInfo.DumpInterval
        if verbose:
            print(f"Surface runoff from pftools\t\t\t : {total_surface_runoff:.16e}")

        filename = f"{run_name}.out.overlandsum.{i:05d}.pfb"
        surface_runoff2 = read_pfb(os.path.join(new_output_dir_name, filename))
        total_surface_runoff2 = np.sum(surface_runoff2)
        if verbose:
            print(
                f"Surface runoff from pfsimulator\t\t\t : {total_surface_runoff2:.16e}"
            )

        if not pf_test_equal(
            total_surface_runoff, total_surface_runoff2, "Surface runoff comparison"
        ):
            passed = False

    if i < 1:
        bc_index = 0
    elif 1 <= i < 7:
        bc_index = i - 1
    else:
        bc_index = 6

    bc_flux = bc_dict[bc_index]
    boundary_flux = bc_flux * surface_area_of_domain * wbx.TimingInfo.DumpInterval

    if verbose:
        print(f"BC flux\t\t\t\t\t\t : {boundary_flux:.16e}")

    expected_difference = boundary_flux + total_surface_runoff
    if verbose:
        print(f"Total Flux\t\t\t\t\t : {expected_difference:.16e}")

    if i > 0:
        if verbose:
            print("")
            print(
                f"Diff from prev total\t\t\t\t : {total_water_in_domain - prev_total_water_balance:.16e}"
            )

        if expected_difference != 0.0:
            percent_diff = (
                abs(
                    (prev_total_water_balance - total_water_in_domain)
                    - expected_difference
                )
                / abs(expected_difference)
                * 100
            )
            if verbose:
                print(
                    f"Percent diff from expected difference\t\t : {percent_diff:.12e}"
                )

        expected_water_balance = prev_total_water_balance - expected_difference
        percent_diff = abs(
            (total_water_in_domain - expected_water_balance)
            / expected_water_balance
            * 100
        )
        if verbose:
            print(f"Percent diff from expected total water sum\t : {percent_diff:.12e}")

        if percent_diff > 0.005:
            print("Error: Water balance is not correct")
            passed = False

    prev_total_water_balance = total_water_in_domain

if verbose:
    print("\n\n")

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
