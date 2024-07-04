#---------------------------------------------------------
#  Testing overland flow diffusive wave
# Running a parking lot sloping slab pointed in 8 directions
# With diffusive BC options
#---------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.compare import pf_test_file

overland = Run("overland_slopingslab_DWE", __file__)

#---------------------------------------------------------

overland.FileVersion = 4

overland.Process.Topology.P = 1
overland.Process.Topology.Q = 1
overland.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

overland.ComputationalGrid.Lower.X = 0.0
overland.ComputationalGrid.Lower.Y = 0.0
overland.ComputationalGrid.Lower.Z = 0.0

overland.ComputationalGrid.NX = 5
overland.ComputationalGrid.NY = 5
overland.ComputationalGrid.NZ = 1

overland.ComputationalGrid.DX = 10.0
overland.ComputationalGrid.DY = 10.0
overland.ComputationalGrid.DZ = .05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

overland.GeomInput.Names = 'domaininput'
overland.GeomInput.domaininput.GeomName = 'domain'
overland.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

overland.Geom.domain.Lower.X = 0.0
overland.Geom.domain.Lower.Y = 0.0
overland.Geom.domain.Lower.Z = 0.0

overland.Geom.domain.Upper.X = 50.0
overland.Geom.domain.Upper.Y = 50.0
overland.Geom.domain.Upper.Z = 0.05
overland.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

overland.Geom.Perm.Names = 'domain'
overland.Geom.domain.Perm.Type = 'Constant'
overland.Geom.domain.Perm.Value = 0.0000001

overland.Perm.TensorType = 'TensorByGeom'

overland.Geom.Perm.TensorByGeom.Names = 'domain'

overland.Geom.domain.Perm.TensorValX = 1.0
overland.Geom.domain.Perm.TensorValY = 1.0
overland.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

overland.SpecificStorage.Type = 'Constant'
overland.SpecificStorage.GeomNames = 'domain'
overland.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

overland.Phase.Names = 'water'

overland.Phase.water.Density.Type = 'Constant'
overland.Phase.water.Density.Value = 1.0

overland.Phase.water.Viscosity.Type = 'Constant'
overland.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

overland.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

overland.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

overland.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

overland.TimingInfo.BaseUnit = 0.05
overland.TimingInfo.StartCount = 0
overland.TimingInfo.StartTime = 0.0
overland.TimingInfo.StopTime = 1.0
overland.TimingInfo.DumpInterval = -2
overland.TimeStep.Type = 'Constant'
overland.TimeStep.Value = 0.05

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

overland.Geom.Porosity.GeomNames = 'domain'

overland.Geom.domain.Porosity.Type = 'Constant'
overland.Geom.domain.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

overland.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

overland.Phase.RelPerm.Type = 'VanGenuchten'
overland.Phase.RelPerm.GeomNames = 'domain'

overland.Geom.domain.RelPerm.Alpha = 6.0
overland.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

overland.Phase.Saturation.Type = 'VanGenuchten'
overland.Phase.Saturation.GeomNames = 'domain'

overland.Geom.domain.Saturation.Alpha = 6.0
overland.Geom.domain.Saturation.N = 2.
overland.Geom.domain.Saturation.SRes = 0.2
overland.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

overland.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

overland.Cycle.Names = 'constant rainrec'
overland.Cycle.constant.Names = 'alltime'
overland.Cycle.constant.alltime.Length = 1
overland.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

overland.Cycle.rainrec.Names = 'rain rec'
overland.Cycle.rainrec.rain.Length = 2
overland.Cycle.rainrec.rec.Length = 300
overland.Cycle.rainrec.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

overland.BCPressure.PatchNames = overland.Geom.domain.Patches

overland.Patch.x_lower.BCPressure.Type = 'FluxConst'
overland.Patch.x_lower.BCPressure.Cycle = 'constant'
overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

overland.Patch.y_lower.BCPressure.Type = 'FluxConst'
overland.Patch.y_lower.BCPressure.Cycle = 'constant'
overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

overland.Patch.z_lower.BCPressure.Type = 'FluxConst'
overland.Patch.z_lower.BCPressure.Cycle = 'constant'
overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

overland.Patch.x_upper.BCPressure.Type = 'FluxConst'
overland.Patch.x_upper.BCPressure.Cycle = 'constant'
overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

overland.Patch.y_upper.BCPressure.Type = 'FluxConst'
overland.Patch.y_upper.BCPressure.Cycle = 'constant'
overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
overland.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
overland.Patch.z_upper.BCPressure.Cycle = 'rainrec'
overland.Patch.z_upper.BCPressure.rain.Value = -0.01
overland.Patch.z_upper.BCPressure.rec.Value = 0.0000

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

overland.Mannings.Type = 'Constant'
overland.Mannings.GeomNames = 'domain'
overland.Mannings.Geom.domain.Value = 3.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

overland.PhaseSources.water.Type = 'Constant'
overland.PhaseSources.water.GeomNames = 'domain'
overland.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

overland.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

overland.Solver = 'Richards'
overland.Solver.MaxIter = 2500

overland.Solver.Nonlinear.MaxIter = 50
overland.Solver.Nonlinear.ResidualTol = 1e-9
overland.Solver.Nonlinear.EtaChoice = 'EtaConstant'
overland.Solver.Nonlinear.EtaValue = 0.01
overland.Solver.Nonlinear.UseJacobian = False

overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
overland.Solver.Nonlinear.StepTol = 1e-20
overland.Solver.Nonlinear.Globalization = 'LineSearch'
overland.Solver.Linear.KrylovDimension = 20
overland.Solver.Linear.MaxRestart = 2

overland.Solver.Linear.Preconditioner = 'PFMG'
overland.Solver.PrintSubsurf = False
overland.Solver.Drop = 1E-20
overland.Solver.AbsTol = 1E-10

overland.Solver.OverlandDiffusive.Epsilon = 1E-5

overland.Solver.WriteSiloSubsurfData = False
overland.Solver.WriteSiloPressure = False
overland.Solver.WriteSiloSlopes = False

overland.Solver.WriteSiloSaturation = False
overland.Solver.WriteSiloConcentration = False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
overland.ICPressure.Type = 'HydroStaticPatch'
overland.ICPressure.GeomNames = 'domain'
overland.Geom.domain.ICPressure.Value = -3.0

overland.Geom.domain.ICPressure.RefGeom = 'domain'
overland.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# Running all 8 direction combinations with the upwind formulation on and off (i.e. 16 total)
# Commented lines are from original TCL test - will need to convert to Python if running with
# Python pftools
#-----------------------------------------------------------------------------

#set runcheck to 1 if you want to run the pass fail tests
runcheck = 1
first = 1
correct_output_dir_name = get_absolute_path('../correct_output')

###############################
# Looping over slab configurations
###############################
x_slopes = [0.01, -0.01]
y_slopes = [0.01, -0.01]
names = ['posxposy', 'negxnegy']

for x_slope, y_slope, name in zip(x_slopes, y_slopes, names):
    print(f"{x_slope} {y_slope} {name}")
    overland.TopoSlopesX.Type  = "Constant"
    overland.TopoSlopesX.GeomNames = "domain"
    overland.TopoSlopesX.Geom.domain.Value = x_slope
    
    overland.TopoSlopesY.Type = "Constant"
    overland.TopoSlopesY.GeomNames = "domain"
    overland.TopoSlopesY.Geom.domain.Value = y_slope
    
    overland.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
    
    run_name = f"Slab.{name}.OverlandDif"
    overland.set_name(run_name)
    if first == 1:
        print(overland.get_name())
        first = 0
    else:
        print(f"Running {run_name} OverlandDiffusive")
    new_output_dir_name = get_absolute_path('test_output/' + run_name)
    mkdir(new_output_dir_name)
    overland.run(working_directory=new_output_dir_name)
    if runcheck == 1:
        passed = True
        for i in range(11):
            timestep = str(i).rjust(5, '0')
            filename = f"/{run_name}.out.press.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Pressure for timestep {timestep}"):
                passed = False
            filename = f"/{run_name}.out.satur.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Saturation for timestep {timestep}"):
                passed = False
        
        if passed:
            print(f"{run_name} : PASSED")
        else:
            print(f"{run_name} : FAILED")
            sys.exit(1)
                

#set to 1 if you want to run the pass fail tests
runcheck = 1

###############################
# Looping over slop configurations
###############################
x_slopes = [0.01, -0.01]
y_slopes = [0.01, -0.01]
names = ['posxposy', 'negxnegy']

for x_slope, y_slope, name in zip(x_slopes, y_slopes, names):
    print(f"{x_slope} {y_slope} {name}")
    overland.TopoSlopesX.Type  = "Constant"
    overland.TopoSlopesX.GeomNames = "domain"
    overland.TopoSlopesX.Geom.domain.Value = x_slope
    
    overland.TopoSlopesY.Type = "Constant"
    overland.TopoSlopesY.GeomNames = "domain"
    overland.TopoSlopesY.Geom.domain.Value = y_slope
    
    overland.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
    overland.Solver.Nonlinear.UseJacobian = True
    
    run_name = f"Slab.{name}.OverlandDif"
    overland.set_name(run_name)

    print(f"Running {run_name} OverlandDiffusive Jacobian True")
    new_output_dir_name = get_absolute_path('test_output/' + run_name)
    mkdir(new_output_dir_name)
    overland.run(working_directory=new_output_dir_name)
    if runcheck == 1:
        passed = True
        for i in range(11):
            timestep = str(i).rjust(5, '0')
            filename = f"/{run_name}.out.press.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Pressure for timestep {timestep}"):
                passed = False
            filename = f"/{run_name}.out.satur.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Saturation for timestep {timestep}"):
                passed = False
        
        if passed:
            print(f"{run_name} : PASSED")
        else:
            print(f"{run_name} : FAILED")
            sys.exit(1)
                


#set to 1 if you want to run the pass fail tests
runcheck = 1


###############################
# Looping over slop configurations
###############################
x_slopes = [0.01, -0.01]
y_slopes = [0.01, -0.01]
names = ['posxposy', 'negxnegy']

for x_slope, y_slope, name in zip(x_slopes, y_slopes, names):
    print(f"{x_slope} {y_slope} {name}")
    overland.TopoSlopesX.Type  = "Constant"
    overland.TopoSlopesX.GeomNames = "domain"
    overland.TopoSlopesX.Geom.domain.Value = x_slope
    
    overland.TopoSlopesY.Type = "Constant"
    overland.TopoSlopesY.GeomNames = "domain"
    overland.TopoSlopesY.Geom.domain.Value = y_slope
    
    overland.Patch.z_upper.BCPressure.Type = 'OverlandDiffusive'
    overland.Solver.Nonlinear.UseJacobian = True
    overland.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'
    
    run_name = f"Slab.{name}.OverlandDif"
    overland.set_name(run_name)

    print(f"Running {run_name} OverlandDiffusive Jacobian True Nonsymmetric Preconditioner")
    new_output_dir_name = get_absolute_path('test_output/' + run_name)
    mkdir(new_output_dir_name)
    overland.run(working_directory=new_output_dir_name)
    if runcheck == 1:
        passed = True
        for i in range(11):
            timestep = str(i).rjust(5, '0')
            filename = f"/{run_name}.out.press.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Pressure for timestep {timestep}"):
                passed = False
            filename = f"/{run_name}.out.satur.{timestep}.pfb"
            if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename,
                                f"Max difference in Saturation for timestep {timestep}"):
                passed = False
        
        if passed:
            print(f"{run_name} : PASSED")
        else:
            print(f"{run_name} : FAILED")
            sys.exit(1)
