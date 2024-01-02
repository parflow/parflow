#---------------------------------------------------------
#  This runs the tilted-v catchment problem
#  similar to that in Kollet and Maxwell (2006) AWR
#---------------------------------------------------------

#---------------------------------------------------------
# Import ParFlow
#---------------------------------------------------------

import sys
import os
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb, ParflowBinaryReader
from parflow.tools.top import compute_top
from parflow.tools import hydrology
from parflow.tools.compare import pf_test_equal

run_name = "water_balance"

wby = Run(run_name, __file__)

#---------------------------------------------------------
# Some controls for the test
#---------------------------------------------------------

#---------------------------------------------------------
# Control slopes 
#-1 = slope to lower-y
# 0 = flat top (no overland flow)
# 1 = slope to upper-y 
#---------------------------------------------------------

use_slopes = 1

#---------------------------------------------------------
# Flux on the top surface
#---------------------------------------------------------

rain_flux = -0.05
rec_flux = 0.0

#---------------------------------------------------------

wby.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------

wby.Process.Topology.P = 1
wby.Process.Topology.Q = 1
wby.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

wby.ComputationalGrid.Lower.X = 0.0
wby.ComputationalGrid.Lower.Y = 0.0
wby.ComputationalGrid.Lower.Z = 0.0

wby.ComputationalGrid.NX = 30
wby.ComputationalGrid.NY = 30
wby.ComputationalGrid.NZ = 30

wby.ComputationalGrid.DX = 10.0
wby.ComputationalGrid.DY = 10.0
wby.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

wby.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

wby.GeomInput.domaininput.GeomName = 'domain'
wby.GeomInput.leftinput.GeomName = 'left'
wby.GeomInput.rightinput.GeomName = 'right'
wby.GeomInput.channelinput.GeomName = 'channel'

wby.GeomInput.domaininput.InputType = 'Box'
wby.GeomInput.leftinput.InputType = 'Box'
wby.GeomInput.rightinput.InputType = 'Box'
wby.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------

wby.Geom.domain.Lower.X = 0.0
wby.Geom.domain.Lower.Y = 0.0
wby.Geom.domain.Lower.Z = 0.0

wby.Geom.domain.Upper.X = 300.0
wby.Geom.domain.Upper.Y = 300.0
wby.Geom.domain.Upper.Z = 1.5
wby.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------

wby.Geom.left.Lower.X = 0.0
wby.Geom.left.Lower.Y = 0.0
wby.Geom.left.Lower.Z = 0.0

wby.Geom.left.Upper.X = 140.0
wby.Geom.left.Upper.Y = 300.0
wby.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------

wby.Geom.right.Lower.X = 160.0
wby.Geom.right.Lower.Y = 0.0
wby.Geom.right.Lower.Z = 0.0

wby.Geom.right.Upper.X = 300.0
wby.Geom.right.Upper.Y = 300.0
wby.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------

wby.Geom.channel.Lower.X = 140.0
wby.Geom.channel.Lower.Y = 0.0
wby.Geom.channel.Lower.Z = 0.0

wby.Geom.channel.Upper.X = 160.0
wby.Geom.channel.Upper.Y = 300.0
wby.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

wby.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

wby.Geom.left.Perm.Type = 'TurnBands'
wby.Geom.left.Perm.LambdaX = 50.
wby.Geom.left.Perm.LambdaY = 50.
wby.Geom.left.Perm.LambdaZ = 0.5
wby.Geom.left.Perm.GeomMean = 0.01

wby.Geom.left.Perm.Sigma = 0.5
wby.Geom.left.Perm.NumLines = 40
wby.Geom.left.Perm.RZeta = 5.0
wby.Geom.left.Perm.KMax = 100.0
wby.Geom.left.Perm.DelK = 0.2
wby.Geom.left.Perm.Seed = 33333
wby.Geom.left.Perm.LogNormal = 'Log'
wby.Geom.left.Perm.StratType = 'Bottom'

wby.Geom.right.Perm.Type = 'TurnBands'
wby.Geom.right.Perm.LambdaX = 50.
wby.Geom.right.Perm.LambdaY = 50.
wby.Geom.right.Perm.LambdaZ = 0.5
wby.Geom.right.Perm.GeomMean = 0.05

wby.Geom.right.Perm.Sigma = 0.5
wby.Geom.right.Perm.NumLines = 40
wby.Geom.right.Perm.RZeta = 5.0
wby.Geom.right.Perm.KMax = 100.0
wby.Geom.right.Perm.DelK = 0.2
wby.Geom.right.Perm.Seed = 13333
wby.Geom.right.Perm.LogNormal = 'Log'
wby.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface

wby.Geom.left.Perm.Type = 'Constant'
wby.Geom.left.Perm.Value = 0.001

wby.Geom.right.Perm.Type = 'Constant'
wby.Geom.right.Perm.Value = 0.01

wby.Geom.channel.Perm.Type = 'Constant'
wby.Geom.channel.Perm.Value = 0.00001

wby.Perm.TensorType = 'TensorByGeom'

wby.Geom.Perm.TensorByGeom.Names = 'domain'

wby.Geom.domain.Perm.TensorValX = 1.0
wby.Geom.domain.Perm.TensorValY = 1.0
wby.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

wby.SpecificStorage.Type = 'Constant'
wby.SpecificStorage.GeomNames = 'domain'
wby.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

wby.Phase.Names = 'water'

wby.Phase.water.Density.Type = 'Constant'
wby.Phase.water.Density.Value = 1.0

wby.Phase.water.Viscosity.Type = 'Constant'
wby.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

wby.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

wby.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

wby.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

wby.TimingInfo.BaseUnit = 0.1
wby.TimingInfo.StartCount = 0
wby.TimingInfo.StartTime = 0.0
wby.TimingInfo.StopTime = 2.0
wby.TimingInfo.DumpInterval = 0.1
wby.TimeStep.Type = 'Constant'
wby.TimeStep.Value = 0.1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

wby.Geom.Porosity.GeomNames = 'left right channel'

wby.Geom.left.Porosity.Type = 'Constant'
wby.Geom.left.Porosity.Value = 0.25

wby.Geom.right.Porosity.Type = 'Constant'
wby.Geom.right.Porosity.Value = 0.25

wby.Geom.channel.Porosity.Type = 'Constant'
wby.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

wby.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

wby.Phase.RelPerm.Type = 'VanGenuchten'
wby.Phase.RelPerm.GeomNames = 'domain'

wby.Geom.domain.RelPerm.Alpha = 6.0
wby.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

wby.Phase.Saturation.Type = 'VanGenuchten'
wby.Phase.Saturation.GeomNames = 'domain'

wby.Geom.domain.Saturation.Alpha = 6.0
wby.Geom.domain.Saturation.N = 2.
wby.Geom.domain.Saturation.SRes = 0.2
wby.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

wby.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

wby.Cycle.Names = 'constant rainrec'
wby.Cycle.constant.Names = 'alltime'
wby.Cycle.constant.alltime.Length = 1
wby.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

wby.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
wby.Cycle.rainrec.r0.Length = 1
wby.Cycle.rainrec.r1.Length = 1
wby.Cycle.rainrec.r2.Length = 1
wby.Cycle.rainrec.r3.Length = 1
wby.Cycle.rainrec.r4.Length = 1
wby.Cycle.rainrec.r5.Length = 1
wby.Cycle.rainrec.r6.Length = 1

wby.Cycle.rainrec.Repeat = 1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

wby.BCPressure.PatchNames = wby.Geom.domain.Patches

wby.Patch.x_lower.BCPressure.Type = 'FluxConst'
wby.Patch.x_lower.BCPressure.Cycle = 'constant'
wby.Patch.x_lower.BCPressure.alltime.Value = 0.0

wby.Patch.y_lower.BCPressure.Type = 'FluxConst'
wby.Patch.y_lower.BCPressure.Cycle = 'constant'
wby.Patch.y_lower.BCPressure.alltime.Value = 0.0

wby.Patch.z_lower.BCPressure.Type = 'FluxConst'
wby.Patch.z_lower.BCPressure.Cycle = 'constant'
wby.Patch.z_lower.BCPressure.alltime.Value = 0.0

wby.Patch.x_upper.BCPressure.Type = 'FluxConst'
wby.Patch.x_upper.BCPressure.Cycle = 'constant'
wby.Patch.x_upper.BCPressure.alltime.Value = 0.0

wby.Patch.y_upper.BCPressure.Type = 'FluxConst'
wby.Patch.y_upper.BCPressure.Cycle = 'constant'
wby.Patch.y_upper.BCPressure.alltime.Value = 0.0

wby.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
wby.Patch.z_upper.BCPressure.Cycle = 'rainrec'
wby.Patch.z_upper.BCPressure.r0.Value = rec_flux
wby.Patch.z_upper.BCPressure.r1.Value = rec_flux
wby.Patch.z_upper.BCPressure.r2.Value = rain_flux
wby.Patch.z_upper.BCPressure.r3.Value = rain_flux
wby.Patch.z_upper.BCPressure.r4.Value = rec_flux
wby.Patch.z_upper.BCPressure.r5.Value = rec_flux
wby.Patch.z_upper.BCPressure.r6.Value = rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

wby.TopoSlopesX.Type = 'Constant'
wby.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  wby.TopoSlopesX.Geom.left.Value = -0.005
  wby.TopoSlopesX.Geom.right.Value = 0.005
  wby.TopoSlopesX.Geom.channel.Value = 0.00
else:
  wby.TopoSlopesX.Geom.left.Value = 0.00
  wby.TopoSlopesX.Geom.right.Value = 0.00
  wby.TopoSlopesX.Geom.channel.Value = 0.00

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

wby.TopoSlopesY.Type = 'Constant'
wby.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  wby.TopoSlopesY.Geom.left.Value = 0.000
  wby.TopoSlopesY.Geom.right.Value = 0.000
  wby.TopoSlopesY.Geom.channel.Value = 0.001*use_slopes
else:
  wby.TopoSlopesY.Geom.left.Value = 0.000
  wby.TopoSlopesY.Geom.right.Value = 0.000
  wby.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

wby.Mannings.Type = 'Constant'
wby.Mannings.GeomNames = 'left right channel'
wby.Mannings.Geom.left.Value = 5.e-6
wby.Mannings.Geom.right.Value = 5.e-6
wby.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

wby.PhaseSources.water.Type = 'Constant'
wby.PhaseSources.water.GeomNames = 'domain'
wby.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

wby.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

wby.Solver = 'Richards'
wby.Solver.MaxIter = 2500

wby.Solver.AbsTol = 1E-12
wby.Solver.Nonlinear.MaxIter = 300
wby.Solver.Nonlinear.ResidualTol = 1e-12
wby.Solver.Nonlinear.EtaChoice = 'Walker1'
wby.Solver.Nonlinear.EtaChoice = 'EtaConstant'
wby.Solver.Nonlinear.EtaValue = 0.001
wby.Solver.Nonlinear.UseJacobian = False
wby.Solver.Nonlinear.DerivativeEpsilon = 1e-16
wby.Solver.Nonlinear.StepTol = 1e-30
wby.Solver.Nonlinear.Globalization = 'LineSearch'
wby.Solver.Linear.KrylovDimension = 20
wby.Solver.Linear.MaxRestart = 2

wby.Solver.Linear.Preconditioner = 'PFMG'
wby.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
wby.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
wby.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
wby.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1

wby.Solver.PrintSubsurfData = True
wby.Solver.PrintConcentration = True
wby.Solver.PrintSlopes = True
wby.Solver.PrintEvapTrans = True
wby.Solver.PrintEvapTransSum = True
wby.Solver.PrintOverlandSum = True
wby.Solver.PrintMannings = True
wby.Solver.PrintSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
wby.ICPressure.Type = 'HydroStaticPatch'
wby.ICPressure.GeomNames = 'domain'

wby.Geom.domain.ICPressure.Value = -3.0

wby.Geom.domain.ICPressure.RefGeom = 'domain'
wby.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path('test_output/water_balance_x')
correct_output_dir_name = get_absolute_path('../correct_output')
mkdir(new_output_dir_name)
wby.run(working_directory=new_output_dir_name)

passed = True
verbose = True

bc_dict = { 0: wby.Patch.z_upper.BCPressure.r0.Value,
            1: wby.Patch.z_upper.BCPressure.r1.Value,
            2: wby.Patch.z_upper.BCPressure.r2.Value,
            3: wby.Patch.z_upper.BCPressure.r3.Value,
            4: wby.Patch.z_upper.BCPressure.r4.Value,
            5: wby.Patch.z_upper.BCPressure.r5.Value,
            6: wby.Patch.z_upper.BCPressure.r6.Value
          }

slope_x = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.slope_x.pfb"))
slope_y = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.slope_y.pfb"))
mannings = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.mannings.pfb"))
specific_storage = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.specific_storage.pfb"))
porosity = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.porosity.pfb"))
mask = read_pfb(os.path.join(new_output_dir_name, f"{run_name}.out.mask.pfb"))
top = compute_top(mask)



surface_area_of_domain = (wby.ComputationalGrid.DX *
                          wby.ComputationalGrid.DY *
                          wby.ComputationalGrid.NX *
                          wby.ComputationalGrid.NY
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
    dx, dy, dz = header['dx'], header['dy'], header['dz']
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
    nz = header['nz']
    subsurface_storage = hydrology.calculate_subsurface_storage(porosity, pressure, saturation, specific_storage, dx, dy, np.array([dz] * nz), mask)
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
        surface_runoff = hydrology.calculate_overland_flow(pressure, slope_x[0], slope_y[0], mannings[0],
                                                           dx, dy, flow_method="OverlandFlow", mask=mask)
        total_surface_runoff = surface_runoff * wby.TimingInfo.DumpInterval
        if verbose:
            print(f"Surface runoff from pftools\t\t\t : {total_surface_runoff:.16e}")

        filename = f"{run_name}.out.overlandsum.{i:05d}.pfb"
        surface_runoff2 = read_pfb(os.path.join(new_output_dir_name, filename))
        total_surface_runoff2 = np.sum(surface_runoff2)
        if verbose:
            print(f"Surface runoff from pfsimulator\t\t\t : {total_surface_runoff2:.16e}")

        if not pf_test_equal(total_surface_runoff, total_surface_runoff2, "Surface runoff comparison"):
            passed = False

    if i < 1:
        bc_index = 0
    elif 1 <= i < 7:
        bc_index = i - 1
    else:
        bc_index = 6

    bc_flux = bc_dict[bc_index]
    boundary_flux = bc_flux * surface_area_of_domain * wby.TimingInfo.DumpInterval

    if verbose:
        print(f"BC flux\t\t\t\t\t\t : {boundary_flux:.16e}")

    expected_difference = boundary_flux + total_surface_runoff
    if verbose:
        print(f"Total Flux\t\t\t\t\t : {expected_difference:.16e}")

    if i > 0:
        if verbose:
            print("")
            print(f"Diff from prev total\t\t\t\t : {total_water_in_domain - prev_total_water_balance:.16e}")

        if expected_difference != 0.0:
            percent_diff = (abs((prev_total_water_balance - total_water_in_domain) - expected_difference) /
                            abs(expected_difference) * 100)
            if verbose:
                print(f"Percent diff from expected difference\t\t : {percent_diff:.12e}")

        expected_water_balance = prev_total_water_balance - expected_difference
        percent_diff = abs((total_water_in_domain - expected_water_balance) / expected_water_balance * 100)
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
