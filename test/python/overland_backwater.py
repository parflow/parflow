# -----------------------------------------------------------------------------
# Backwater test for the OverlandKinematic diffusive wave options
#
# A sloped plane drains to a reach of zero bed slope with a no-flow boundary.
# The domain is one cell wide and 40 cells of 20 m long.  Cells 0-9 have zero
# bed slope and cells 10-39 slope toward them at 5e-4.  Rain falls for 60 min
# and then stops.  The water ponds in the flat reach and the backwater extends
# up the slope.  The steady state is hydrostatic: the water surface is
# horizontal, so over the sloped cells the depth gradient cancels the bed slope.
#
# The kinematic wave cannot represent this backwater.  A scheme with different
# slope magnitudes under the two terms of the flux is not well balanced, and
# its steady state has a sloping water surface.  The test runs the schemes that
# share one magnitude and checks, besides the reference files, that the steady
# water surface is horizontal and that water is conserved.
# -----------------------------------------------------------------------------

import sys, argparse
import os
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.compare import pf_test_file

overland = Run("overland_backwater", __file__)

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

slope = 5.0e-4
n_flat = 10
ny = 40
dy = 20.0
dz = 0.005
dt = 0.5 / 60.0
rain_steps = 120
total_steps = 1200
dump_steps = 120
rain = 1.8e-4 * 60.0
mannings = 2.5e-4 / 60.0

# ---------------------------------------------------------
# Computational Grid
# ---------------------------------------------------------

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = 1
overland.ComputationalGrid.NY = ny
overland.ComputationalGrid.NZ = 1

overland.ComputationalGrid.DX = dy
overland.ComputationalGrid.DY = dy
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

overland.Geom.domain.Upper.X = dy
overland.Geom.domain.Upper.Y = ny * dy
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
# Time Cycles: rain, then recession
# -----------------------------------------------------------------------------

overland.Cycle.Names = "constant rainrec"
overland.Cycle.constant.Names = "alltime"
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

overland.Cycle.rainrec.Names = "rain rec"
overland.Cycle.rainrec.rain.Length = rain_steps
overland.Cycle.rainrec.rec.Length = total_steps - rain_steps
overland.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions: closed on every side, rain on top
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
overland.Patch.z_upper.BCPressure.Cycle = "rainrec"
overland.Patch.z_upper.BCPressure.rain.Value = -rain
overland.Patch.z_upper.BCPressure.rec.Value = 0.0

# ---------------------------------------------------------
# Topo slopes and Mannings coefficient
# ---------------------------------------------------------

overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "domain"
overland.TopoSlopesX.Geom.domain.Value = 0.0

overland.TopoSlopesY.Type = "PFBFile"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.FileName = "slope_y.pfb"

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
overland.Solver.MaxIter = 250000

overland.Solver.Nonlinear.MaxIter = 100
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
# Initial conditions: water pressure
# ---------------------------------------------------------

overland.ICPressure.Type = "HydroStaticPatch"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.Value = -1.0
overland.Geom.domain.ICPressure.RefGeom = "domain"
overland.Geom.domain.ICPressure.RefPatch = "z_upper"

# -----------------------------------------------------------------------------
# Bed slope and bed elevation.  The face between cells j and j+1 carries the
# slope of cell j.
# -----------------------------------------------------------------------------

slope_y = np.zeros(ny)
slope_y[n_flat:] = slope
bed = np.concatenate([[0.0], np.cumsum(slope_y[:-1] * dy)])

# -----------------------------------------------------------------------------
# Run and check each scheme.  Each configuration is (run name, SlopeMagnitude,
# BedTermMagnitude, SurfaceTermMagnitude).  The first is the diffusive wave with
# a lagged friction-slope magnitude, which SlopeMagnitude FrictionSlope gives
# with the other two keys left at their defaults.
# -----------------------------------------------------------------------------

configurations = [
    ("Backwater_FrictionSlope_Lagged", "FrictionSlope", None, None),
    ("Backwater_FrictionSlope_Implicit", "FrictionSlope", "Implicit", "Implicit"),
    ("Backwater_Pythagorean_Lagged", "Pythagorean", "Lagged", "Lagged"),
    ("Backwater_Pythagorean_Implicit", "Pythagorean", "Implicit", "Implicit"),
]

# Rain depth times the number of cells, in m of water per unit width and dy
rain_volume = rain * rain_steps * dt * ny
# Largest allowed range of water-surface elevation over the ponded reach at
# the end of the run (m).
# The bed rises 0.01 m per cell on the slope, and a scheme that is not well
# balanced leaves a water-surface slope of a third of that or more per cell.
# The Pythagorean runs are still approaching steady state at the end and are
# within 0.6 mm.
level_tolerance = 2.0e-3
# A cell is in the ponded reach if its bed is this far below the water surface
# at the no-flow boundary (m)
ponded_margin = 2.0e-3
# Largest allowed relative loss of water
mass_tolerance = 1.0e-3

runcheck = 1
correct_output_dir_name = get_absolute_path("../correct_output")
n_dumps = total_steps // dump_steps + 1

for run_name, slope_magnitude, bed_term, surface_term in configurations:
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
        os.path.join(new_output_dir_name, "slope_y.pfb"),
        slope_y.reshape(1, ny, 1),
        dist=False,
    )
    overland.dist(os.path.join(new_output_dir_name, "slope_y.pfb"))
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

        # The end of the run: ponded depth and water-surface elevation
        timestep = str(n_dumps - 1).rjust(5, "0")
        depth = np.maximum(
            read_pfb(new_output_dir_name + f"/{run_name}.out.press.{timestep}.pfb")[
                -1, :, 0
            ],
            0.0,
        )
        # The ponded reach is the set of cells whose bed lies below the water
        # surface at the no-flow boundary.  Cells above it hold only a thin
        # draining film.
        ponded = bed < depth[0] - ponded_margin
        surface = (bed + depth)[ponded]
        level_range = surface.max() - surface.min()
        print(
            f"Ponded reach covers {ponded.sum()} cells, "
            f"{(ponded[n_flat:]).sum()} of them on the slope; "
            f"water-surface range {level_range:.3e} m"
        )
        if ponded[n_flat:].sum() < 2:
            print("FAILED : the backwater does not extend up the slope")
            passed = False
        if level_range > level_tolerance:
            print(
                f"FAILED : the steady water surface is not horizontal ({level_range:.3e} m)"
            )
            passed = False

        mass_error = abs(depth.sum() - rain_volume) / rain_volume
        print(f"Relative difference between ponded water and rain {mass_error:.3e}")
        if mass_error > mass_tolerance:
            print(f"FAILED : water is not conserved ({mass_error:.3e})")
            passed = False

        if passed:
            print(f"{run_name} : PASSED")
        else:
            print(f"{run_name} : FAILED")
            sys.exit(1)
