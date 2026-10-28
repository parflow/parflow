# -----------------------------------------------------------------------------
# Spreading mound test for the OverlandKinematic water-surface (diffusion) term
#
# A 4 by 4 block of cells holds 1 cm of water at the center of a 20 by 20 grid
# of 10 m cells.  There is no rain and the sides are closed.  On a flat plane
# the mound spreads by the water-surface gradient alone, which the kinematic
# wave cannot do, and the result is symmetric.  On a tilted plane it also
# slides downhill.
#
# The mound sits on the corner shared by four subgrids of a 2 by 2 process
# layout.  The FrictionSlope and Pythagorean slope magnitudes read corner ghost
# cells, of the pressure and, when lagged, of the pressure at the previous time
# step, so the parallel variants of this test check those exchanges.
# -----------------------------------------------------------------------------

import sys, argparse
import os
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.compare import pf_test_file

overland = Run("overland_pond_diffusion", __file__)

# -----------------------------------------------------------------------------

overland.FileVersion = 4

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", default=1)
parser.add_argument("-q", "--q", default=1)
parser.add_argument("-r", "--r", default=1)
args = parser.parse_args()

overland.Process.Topology.P = args.p
overland.Process.Topology.Q = args.q
overland.Process.Topology.R = args.r

# ---------------------------------------------------------
# Case definition.  Times are in hours.
# ---------------------------------------------------------

n = 20
dx = 10.0
dz = 0.05
dt = 1.0 / 600.0
total_steps = 300
dump_steps = 50
depth_0 = 0.01
mannings = 5.52e-5

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = n
overland.ComputationalGrid.NY = n
overland.ComputationalGrid.NZ = 1

overland.ComputationalGrid.DX = dx
overland.ComputationalGrid.DY = dx
overland.ComputationalGrid.DZ = dz

# ---------------------------------------------------------
# Domain Geometry
# ---------------------------------------------------------

overland.GeomInput.Names = "domaininput"

overland.GeomInput.domaininput.GeomName = "domain"
overland.GeomInput.domaininput.InputType = "Box"

overland.Geom.domain.Lower.X = 0.0
overland.Geom.domain.Lower.Y = 0.0
overland.Geom.domain.Lower.Z = 0.0

overland.Geom.domain.Upper.X = n * dx
overland.Geom.domain.Upper.Y = n * dx
overland.Geom.domain.Upper.Z = dz
overland.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Perm: impermeable, with almost no storage, so the flow is on the surface
# -----------------------------------------------------------------------------

overland.Geom.Perm.Names = "domain"
overland.Geom.domain.Perm.Type = "Constant"
overland.Geom.domain.Perm.Value = 1.0e-12

overland.Perm.TensorType = "TensorByGeom"
overland.Geom.Perm.TensorByGeom.Names = "domain"
overland.Geom.domain.Perm.TensorValX = 1.0
overland.Geom.domain.Perm.TensorValY = 1.0
overland.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage and Porosity
# -----------------------------------------------------------------------------

overland.SpecificStorage.Type = "Constant"
overland.SpecificStorage.GeomNames = "domain"
overland.Geom.domain.SpecificStorage.Value = 1.0e-4

overland.Geom.Porosity.GeomNames = "domain"
overland.Geom.domain.Porosity.Type = "Constant"
overland.Geom.domain.Porosity.Value = 0.001

# -----------------------------------------------------------------------------
# Phases, contaminants, gravity
# -----------------------------------------------------------------------------

overland.Phase.Names = "water"
overland.Phase.water.Density.Type = "Constant"
overland.Phase.water.Density.Value = 1.0
overland.Phase.water.Viscosity.Type = "Constant"
overland.Phase.water.Viscosity.Value = 1.0

overland.Contaminants.Names = ""
overland.Geom.Retardation.GeomNames = ""
overland.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

overland.TimingInfo.BaseUnit = dt
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = total_steps * dt
overland.TimingInfo.DumpInterval = -dump_steps
overland.TimeStep.Type = "Constant"
overland.TimeStep.Value = dt

# -----------------------------------------------------------------------------
# Domain, relative permeability, saturation
# -----------------------------------------------------------------------------

overland.Domain.GeomName = "domain"

overland.Phase.RelPerm.Type = "VanGenuchten"
overland.Phase.RelPerm.GeomNames = "domain"
overland.Geom.domain.RelPerm.Alpha = 6.0
overland.Geom.domain.RelPerm.N = 2.0

overland.Phase.Saturation.Type = "VanGenuchten"
overland.Phase.Saturation.GeomNames = "domain"
overland.Geom.domain.Saturation.Alpha = 6.0
overland.Geom.domain.Saturation.N = 2.0
overland.Geom.domain.Saturation.SRes = 0.2
overland.Geom.domain.Saturation.SSat = 1.0

overland.Wells.Names = ""

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

overland.Cycle.Names = "constant"
overland.Cycle.constant.Names = "alltime"
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: closed on every side, no rain
# -----------------------------------------------------------------------------

overland.BCPressure.PatchNames = overland.Geom.domain.Patches

overland.Patch.x_lower.BCPressure.Type = "FluxConst"
overland.Patch.x_lower.BCPressure.Cycle = "constant"
overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland.Patch.x_upper.BCPressure.Type = "FluxConst"
overland.Patch.x_upper.BCPressure.Cycle = "constant"
overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland.Patch.y_lower.BCPressure.Type = "FluxConst"
overland.Patch.y_lower.BCPressure.Cycle = "constant"
overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland.Patch.y_upper.BCPressure.Type = "FluxConst"
overland.Patch.y_upper.BCPressure.Cycle = "constant"
overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

overland.Patch.z_lower.BCPressure.Type = "FluxConst"
overland.Patch.z_lower.BCPressure.Cycle = "constant"
overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland.Patch.z_upper.BCPressure.Cycle = "constant"
overland.Patch.z_upper.BCPressure.alltime.Value = 0.0

# ---------------------------------------------------------
# Topo slopes and Mannings coefficient
# ---------------------------------------------------------

overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.Geom.domain.Value = 0.0

overland.TopoSlopesY.Type = "Constant"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.Geom.domain.Value = 0.0

overland.Mannings.Type = "Constant"
overland.Mannings.GeomNames = "domain"
overland.Mannings.Geom.domain.Value = mannings

# -----------------------------------------------------------------------------
# Phase sources and exact solution
# -----------------------------------------------------------------------------

overland.PhaseSources.water.Type = "Constant"
overland.PhaseSources.water.GeomNames = "domain"
overland.PhaseSources.water.Geom.domain.Value = 0.0

overland.KnownSolution = "NoKnownSolution"

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

overland.Solver = "Richards"
overland.Solver.MaxIter = 25000

overland.Solver.Nonlinear.MaxIter = 200
overland.Solver.Nonlinear.ResidualTol = 1e-9
overland.Solver.Nonlinear.EtaChoice = "EtaConstant"
overland.Solver.Nonlinear.EtaValue = 0.01
overland.Solver.Nonlinear.UseJacobian = True
overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland.Solver.Nonlinear.StepTol = 1e-20
overland.Solver.Nonlinear.Globalization = "LineSearch"
overland.Solver.Linear.KrylovDimension = 50
overland.Solver.Linear.MaxRestart = 2
overland.Solver.OverlandKinematic.Epsilon = 1e-5

overland.Solver.Linear.Preconditioner = "PFMG"
overland.Solver.Linear.Preconditioner.PCMatrixType = "FullJacobian"
overland.Solver.PrintSubsurf = False
overland.Solver.Drop = 1e-20
overland.Solver.AbsTol = 1e-10

# ---------------------------------------------------------
# Initial conditions: the mound on a surface just below saturation
# ---------------------------------------------------------

overland.ICPressure.Type = "PFBFile"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.FileName = "ic_pressure.pfb"

ic_pressure = np.full((1, n, n), -0.001)
ic_pressure[0, 8:12, 8:12] = depth_0

# -----------------------------------------------------------------------------
# Run and check each case.  Each configuration is (run name, bed slope in x and
# in y, SlopeMagnitude, BedTermMagnitude, SurfaceTermMagnitude).  The first is
# the diffusive wave with a lagged friction-slope magnitude, which SlopeMagnitude
# FrictionSlope gives with the other two keys left at their defaults.
# -----------------------------------------------------------------------------

configurations = [
    ("Pond_Flat_FrictionSlope_Lagged", 0.0, "FrictionSlope", None, None),
    ("Pond_Flat_FrictionSlope_Implicit", 0.0, "FrictionSlope", "Implicit", "Implicit"),
    ("Pond_Tilted_FrictionSlope_Lagged", 7.0e-4, "FrictionSlope", "Lagged", "Lagged"),
    ("Pond_Tilted_Pythagorean_Lagged", 7.0e-4, "Pythagorean", "Lagged", "Lagged"),
]

# Water in the mound at the start, in m of depth summed over cells
mound_volume = depth_0 * 16
# Largest allowed relative change in the water on the surface
mass_tolerance = 1.0e-3
# Largest allowed asymmetry on the flat plane, relative to the peak depth
symmetry_tolerance = 1.0e-6

runcheck = 1
correct_output_dir_name = get_absolute_path("../correct_output")
n_dumps = total_steps // dump_steps + 1

for run_name, slope, slope_magnitude, bed_term, surface_term in configurations:
    overland.TopoSlopesX.Geom.domain.Value = slope
    overland.TopoSlopesY.Geom.domain.Value = slope
    overland.Solver.OverlandKinematic.Diffusion.SlopeMagnitude = slope_magnitude
    if bed_term is not None:
        overland.Solver.OverlandKinematic.Diffusion.BedTermMagnitude = bed_term
        overland.Solver.OverlandKinematic.Diffusion.SurfaceTermMagnitude = surface_term

    overland.set_name(run_name)
    print("##########")
    print(f"Running {run_name}")
    new_output_dir_name = get_absolute_path("test_output/" + f"{run_name}")
    mkdir(new_output_dir_name)
    write_pfb(
        os.path.join(new_output_dir_name, "ic_pressure.pfb"), ic_pressure, dist=False
    )
    overland.dist(os.path.join(new_output_dir_name, "ic_pressure.pfb"))
    overland.run(working_directory=new_output_dir_name)

    if runcheck == 1:
        passed = True
        for i in range(n_dumps):
            timestep = str(i).rjust(5, "0")
            if not pf_test_file(
                new_output_dir_name + f"/{run_name}.out.press.{timestep}.pfb",
                correct_output_dir_name + f"/{run_name}.out.press.{timestep}.pfb",
                f"Max difference in Pressure for timestep {timestep}",
            ):
                passed = False

        timestep = str(n_dumps - 1).rjust(5, "0")
        depth = np.maximum(
            read_pfb(new_output_dir_name + f"/{run_name}.out.press.{timestep}.pfb")[
                -1, :, :
            ],
            0.0,
        )
        print(
            f"Peak depth {depth.max():.4e} m, "
            f"{(depth > 1.0e-6).sum()} cells wet at the end"
        )
        if depth.max() > 0.9 * depth_0:
            print("FAILED : the mound did not spread")
            passed = False

        mass_error = abs(depth.sum() - mound_volume) / mound_volume
        print(f"Relative change in the water on the surface {mass_error:.3e}")
        if mass_error > mass_tolerance:
            print(f"FAILED : water is not conserved ({mass_error:.3e})")
            passed = False

        if slope == 0.0:
            asymmetry = (
                max(
                    np.abs(depth - depth[::-1, :]).max(),
                    np.abs(depth - depth[:, ::-1]).max(),
                    np.abs(depth - depth.T).max(),
                )
                / depth.max()
            )
            print(f"Asymmetry relative to the peak depth {asymmetry:.3e}")
            if asymmetry > symmetry_tolerance:
                print(f"FAILED : the mound is not symmetric ({asymmetry:.3e})")
                passed = False

        if passed:
            print(f"{run_name} : PASSED")
        else:
            print(f"{run_name} : FAILED")
            sys.exit(1)
