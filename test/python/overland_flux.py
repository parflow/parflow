# -----------------------------------------------------------------------------
# test for qx_overland and qy_overland outputs
# Tests both OverlandFlow and OverlandKinematic methods
# -----------------------------------------------------------------------------

import sys
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb
from parflow.tools.compare import pf_test_file_with_abs

run_name = "overland_flux"
overland = Run(run_name, __file__)

overland.FileVersion = 4

overland.Process.Topology.P = 1
overland.Process.Topology.Q = 1
overland.Process.Topology.R = 1

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = 5
overland.ComputationalGrid.NY = 5
overland.ComputationalGrid.NZ = 1

overland.ComputationalGrid.DX = 10.0
overland.ComputationalGrid.DY = 10.0
overland.ComputationalGrid.DZ = 0.05

overland.GeomInput.Names = "domaininput leftinput rightinput channelinput"

overland.GeomInput.domaininput.GeomName = "domain"
overland.GeomInput.leftinput.GeomName = "left"
overland.GeomInput.rightinput.GeomName = "right"
overland.GeomInput.channelinput.GeomName = "channel"

overland.GeomInput.domaininput.InputType = "Box"
overland.GeomInput.leftinput.InputType = "Box"
overland.GeomInput.rightinput.InputType = "Box"
overland.GeomInput.channelinput.InputType = "Box"

overland.Geom.domain.Lower.X = 0.0
overland.Geom.domain.Lower.Y = 0.0
overland.Geom.domain.Lower.Z = 0.0

overland.Geom.domain.Upper.X = 50.0
overland.Geom.domain.Upper.Y = 50.0
overland.Geom.domain.Upper.Z = 0.05
overland.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

overland.Geom.left.Lower.X = 0.0
overland.Geom.left.Lower.Y = 0.0
overland.Geom.left.Lower.Z = 0.0

overland.Geom.left.Upper.X = 20.0
overland.Geom.left.Upper.Y = 50.0
overland.Geom.left.Upper.Z = 0.05

overland.Geom.right.Lower.X = 30.0
overland.Geom.right.Lower.Y = 0.0
overland.Geom.right.Lower.Z = 0.0

overland.Geom.right.Upper.X = 50.0
overland.Geom.right.Upper.Y = 50.0
overland.Geom.right.Upper.Z = 0.05

overland.Geom.channel.Lower.X = 20.0
overland.Geom.channel.Lower.Y = 0.0
overland.Geom.channel.Lower.Z = 0.0

overland.Geom.channel.Upper.X = 30.0
overland.Geom.channel.Upper.Y = 50.0
overland.Geom.channel.Upper.Z = 0.05

overland.Geom.Perm.Names = "domain"
overland.Geom.domain.Perm.Type = "Constant"
overland.Geom.domain.Perm.Value = 0.0000001

overland.Perm.TensorType = "TensorByGeom"

overland.Geom.Perm.TensorByGeom.Names = "domain"

overland.Geom.domain.Perm.TensorValX = 1.0
overland.Geom.domain.Perm.TensorValY = 1.0
overland.Geom.domain.Perm.TensorValZ = 1.0

overland.SpecificStorage.Type = "Constant"
overland.SpecificStorage.GeomNames = "domain"
overland.Geom.domain.SpecificStorage.Value = 1.0e-4

overland.Phase.Names = "water"

overland.Phase.water.Density.Type = "Constant"
overland.Phase.water.Density.Value = 1.0

overland.Phase.water.Viscosity.Type = "Constant"
overland.Phase.water.Viscosity.Value = 1.0

overland.Contaminants.Names = ""

overland.Geom.Retardation.GeomNames = ""

overland.Gravity = 1.0

overland.TimingInfo.BaseUnit = 0.05
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = 0.4
overland.TimingInfo.DumpInterval = -2
overland.TimeStep.Type = "Constant"
overland.TimeStep.Value = 0.05

overland.Geom.Porosity.GeomNames = "domain"
overland.Geom.domain.Porosity.Type = "Constant"
overland.Geom.domain.Porosity.Value = 0.01

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

overland.Cycle.Names = "constant rainrec"
overland.Cycle.constant.Names = "alltime"
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

overland.Cycle.rainrec.Names = "rain rec"
overland.Cycle.rainrec.rain.Length = 2
overland.Cycle.rainrec.rec.Length = 300
overland.Cycle.rainrec.Repeat = -1

overland.BCPressure.PatchNames = overland.Geom.domain.Patches

overland.Patch.x_lower.BCPressure.Type = "FluxConst"
overland.Patch.x_lower.BCPressure.Cycle = "constant"
overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland.Patch.y_lower.BCPressure.Type = "FluxConst"
overland.Patch.y_lower.BCPressure.Cycle = "constant"
overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland.Patch.z_lower.BCPressure.Type = "FluxConst"
overland.Patch.z_lower.BCPressure.Cycle = "constant"
overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland.Patch.x_upper.BCPressure.Type = "FluxConst"
overland.Patch.x_upper.BCPressure.Cycle = "constant"
overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland.Patch.y_upper.BCPressure.Type = "FluxConst"
overland.Patch.y_upper.BCPressure.Cycle = "constant"
overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

overland.Patch.z_upper.BCPressure.Type = "OverlandKinematic"
overland.Patch.z_upper.BCPressure.Cycle = "rainrec"
overland.Patch.z_upper.BCPressure.rain.Value = -0.01
overland.Patch.z_upper.BCPressure.rec.Value = 0.0000

overland.Mannings.Type = "Constant"
overland.Mannings.GeomNames = "domain"
overland.Mannings.Geom.domain.Value = 3.0e-6

overland.PhaseSources.water.Type = "Constant"
overland.PhaseSources.water.GeomNames = "domain"
overland.PhaseSources.water.Geom.domain.Value = 0.0

overland.KnownSolution = "NoKnownSolution"

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

overland.Solver.PrintQxOverland = True
overland.Solver.PrintQyOverland = True
overland.Solver.PrintSlopes = True
overland.Solver.PrintMannings = True
overland.Solver.PrintMask = True

overland.ICPressure.Type = "HydroStaticPatch"
overland.ICPressure.GeomNames = "domain"
overland.Geom.domain.ICPressure.Value = -3.0

overland.Geom.domain.ICPressure.RefGeom = "domain"
overland.Geom.domain.ICPressure.RefPatch = "z_upper"

overland.TopoSlopesX.Type = "Constant"
overland.TopoSlopesX.GeomNames = "left right channel"
overland.TopoSlopesX.Geom.left.Value = -0.01
overland.TopoSlopesX.Geom.right.Value = 0.01


overland.TopoSlopesX.Geom.channel.Value = 0.01

overland.TopoSlopesY.Type = "Constant"
overland.TopoSlopesY.GeomNames = "domain"
overland.TopoSlopesY.Geom.domain.Value = 0.01

overland.Solver.Nonlinear.UseJacobian = False
overland.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

output_dir = get_absolute_path(f"test_output/{run_name}")
mkdir(output_dir)
correct_output_dir = get_absolute_path("../correct_output")
overland.run(working_directory=output_dir)

passed = True

i = 5
timestep = str(i).rjust(5, "0")

sig_digits = 8
abs_value = 1e-12

test_files = ["qx_overland", "qy_overland"]
for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.{timestep}.pfb"
    if not pf_test_file_with_abs(
        output_dir + filename,
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
