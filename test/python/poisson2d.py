import sys
from parflow import Run
from parflow.tools.fs import cp, mkdir, chdir, get_absolute_path, rm
from parflow.tools.compare import pf_test_file, pf_test_file_with_abs

run_name = "Poisson2D"
poisson2d = Run(run_name, __file__)

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
poisson2d.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

poisson2d.Process.Topology.P = 1
poisson2d.Process.Topology.Q = 1
poisson2d.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
poisson2d.ComputationalGrid.Lower.X = 0.0
poisson2d.ComputationalGrid.Lower.Y = 0.0
poisson2d.ComputationalGrid.Lower.Z = 0.0

poisson2d.ComputationalGrid.DX = 0.1
poisson2d.ComputationalGrid.DY = 0.1
poisson2d.ComputationalGrid.DZ = 1.0

poisson2d.ComputationalGrid.NX = 10
poisson2d.ComputationalGrid.NY = 10
poisson2d.ComputationalGrid.NZ = 1

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
poisson2d.GeomInput.Names = "domain_input"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
poisson2d.GeomInput.domain_input.InputType = "Box"
poisson2d.GeomInput.domain_input.GeomName = "domain"

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
poisson2d.Geom.domain.Lower.X = 0.0
poisson2d.Geom.domain.Lower.Y = 0.0
poisson2d.Geom.domain.Lower.Z = 0.0

poisson2d.Geom.domain.Upper.X = 1.0
poisson2d.Geom.domain.Upper.Y = 1.0
poisson2d.Geom.domain.Upper.Z = 1.0

poisson2d.Geom.domain.Patches = "left right front back bottom top"

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
poisson2d.Geom.Perm.Names = "domain"

poisson2d.Geom.domain.Perm.Type = "Constant"
poisson2d.Geom.domain.Perm.Value = 1.0

poisson2d.Perm.TensorType = "TensorByGeom"

poisson2d.Geom.Perm.TensorByGeom.Names = "domain"

poisson2d.Geom.domain.Perm.TensorValX = 1.0
poisson2d.Geom.domain.Perm.TensorValY = 1.0
poisson2d.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

poisson2d.SpecificStorage.Type = "Constant"
poisson2d.SpecificStorage.GeomNames = ""
poisson2d.Geom.domain.SpecificStorage.Value = 0.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

poisson2d.Phase.Names = "water"

poisson2d.Phase.water.Density.Type = "Constant"
poisson2d.Phase.water.Density.Value = 1.0

poisson2d.Phase.water.Viscosity.Type = "Constant"
poisson2d.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
poisson2d.Contaminants.Names = ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
poisson2d.Geom.Retardation.GeomNames = ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

poisson2d.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

poisson2d.TimingInfo.BaseUnit = 1.0
poisson2d.TimingInfo.StartCount = 0
poisson2d.TimingInfo.StartTime = 0.0
poisson2d.TimingInfo.StopTime = 1.0
poisson2d.TimingInfo.DumpInterval = -1
poisson2d.TimeStep.Type = "Constant"
poisson2d.TimeStep.Value = 1.

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

poisson2d.Geom.Porosity.GeomNames = "domain"

poisson2d.Geom.domain.Porosity.Type = "Constant"
poisson2d.Geom.domain.Porosity.Value = 0.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
poisson2d.Domain.GeomName = "domain"

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

poisson2d.Phase.RelPerm.Type = "Constant"
poisson2d.Phase.RelPerm.GeomNames = "domain"
poisson2d.Geom.domain.RelPerm.Value = 1.0
#poisson2d.Geom.domain.RelPerm.Alpha = 1.
#poisson2d.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

poisson2d.Phase.Saturation.Type = "Constant"
poisson2d.Phase.Saturation.GeomNames = "domain"
poisson2d.Geom.domain.Saturation.Value = 1.
#poisson2d.Geom.domain.Saturation.Alpha = 1.
#poisson2d.Geom.domain.Saturation.N = 2.
#poisson2d.Geom.domain.Saturation.SRes = 0.2
#poisson2d.Geom.domain.Saturation.SSat = 1.

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
poisson2d.Wells.Names = ""


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
poisson2d.Cycle.Names = "constant"
poisson2d.Cycle.constant.Names = "alltime"
poisson2d.Cycle.constant.alltime.Length = 1
poisson2d.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
poisson2d.BCPressure.PatchNames = "left right front back bottom top"

poisson2d.Patch.left.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.left.BCPressure.Cycle = "constant"
poisson2d.Patch.left.BCPressure.RefGeom = "domain"
poisson2d.Patch.left.BCPressure.RefPatch = "bottom"
poisson2d.Patch.left.BCPressure.alltime.Value = 0.5

poisson2d.Patch.right.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.right.BCPressure.Cycle = "constant"
poisson2d.Patch.right.BCPressure.RefGeom = "domain"
poisson2d.Patch.right.BCPressure.RefPatch = "bottom"
poisson2d.Patch.right.BCPressure.alltime.Value = 0.5

poisson2d.Patch.front.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.front.BCPressure.Cycle = "constant"
poisson2d.Patch.front.BCPressure.RefGeom = "domain"
poisson2d.Patch.front.BCPressure.RefPatch = "bottom"
poisson2d.Patch.front.BCPressure.alltime.Value = 0.5

poisson2d.Patch.back.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.back.BCPressure.Cycle = "constant"
poisson2d.Patch.back.BCPressure.RefGeom = "domain"
poisson2d.Patch.back.BCPressure.RefPatch = "bottom"
poisson2d.Patch.back.BCPressure.alltime.Value = 0.5

poisson2d.Patch.bottom.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.bottom.BCPressure.Cycle = "constant"
poisson2d.Patch.bottom.BCPressure.RefGeom = "domain"
poisson2d.Patch.bottom.BCPressure.RefPatch = "bottom"
poisson2d.Patch.bottom.BCPressure.alltime.Value = 0.5

poisson2d.Patch.top.BCPressure.Type = "DirEquilRefPatch"
poisson2d.Patch.top.BCPressure.Cycle = "constant"
poisson2d.Patch.top.BCPressure.RefGeom = "domain"
poisson2d.Patch.top.BCPressure.RefPatch = "top"
poisson2d.Patch.top.BCPressure.alltime.Value = -0.5

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

poisson2d.TopoSlopesX.Type = "Constant"
poisson2d.TopoSlopesX.GeomNames = ""

poisson2d.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

poisson2d.TopoSlopesY.Type = "Constant"
poisson2d.TopoSlopesY.GeomNames = ""

poisson2d.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

poisson2d.Mannings.Type = "Constant"
poisson2d.Mannings.GeomNames = ""
poisson2d.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

poisson2d.ICPressure.Type = "HydroStaticPatch"
poisson2d.ICPressure.GeomNames = "domain"
poisson2d.Geom.domain.ICPressure.Value = 0.5
poisson2d.Geom.domain.ICPressure.RefGeom = "domain"
poisson2d.Geom.domain.ICPressure.RefPatch = "bottom"

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

poisson2d.PhaseSources.water.Type = "Constant"
poisson2d.PhaseSources.water.GeomNames = "domain"
poisson2d.PhaseSources.water.Geom.domain.Value = 1.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

poisson2d.KnownSolution = "NoKnownSolution"

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
poisson2d.Solver = "Richards"
poisson2d.Solver.MaxIter = 5

poisson2d.Solver.Nonlinear.MaxIter = 10
poisson2d.Solver.Nonlinear.ResidualTol = 1e-12
poisson2d.Solver.Nonlinear.EtaChoice = "EtaConstant"
poisson2d.Solver.Nonlinear.EtaValue = 1e-5
poisson2d.Solver.Nonlinear.UseJacobian = False
poisson2d.Solver.Nonlinear.DerivativeEpsilon = 1e-2

poisson2d.Solver.Linear.KrylovDimension = 10

poisson2d.Solver.Linear.Preconditioner = "MGSemi"
poisson2d.Solver.Linear.Preconditioner.MGSemi.MaxIter = 5
poisson2d.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

poisson2d.Solver.PrintVelocities = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
new_output_dir_name = get_absolute_path('test_output')
mkdir(new_output_dir_name)
poisson2d.run(working_directory=new_output_dir_name)
