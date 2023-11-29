#-----------------------------------------------------------------------------
# BH: this runs CLM with forced vegetation test case
# lai.dat, sai.dat, displa.dat and z0m.dat are required input text files.
# also veg_map.pfb must be provided.
# This tests 1D forcing of vegetation: a single time series is applied at each cell based on the cell's vegetation type (IGBP)
# These time series are stored as columns (as many columns as IGBP vegetation classes -18) in input files.
#-----------------------------------------------------------------------------

import sys, argparse
from parflow import Run
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.compare import pf_test_file
from parflow.tools.top import compute_top, extract_top

run_name = "clm_forc_veg"
clm_veg = Run(run_name, __file__)

#-----------------------------------------------------------------------------
# Making output directories and copying input files
#-----------------------------------------------------------------------------

new_output_dir_name = get_absolute_path('test_output/clm_forc_veg')
mkdir(new_output_dir_name)

directories = [
  'qflx_evap_grnd',
  'eflx_lh_tot',
  'qflx_evap_tot',
  'qflx_tran_veg',
  'correct_output',
  'qflx_infl',
  'swe_out',
  'eflx_lwrad_out',
  't_grnd',
  'diag_out',
  'qflx_evap_soi',
  'eflx_soil_grnd',
  'eflx_sh_tot',
  'qflx_evap_veg',
  'qflx_top_soil'
]

for directory in directories:
    mkdir(new_output_dir_name + '/' + directory)

cp('$PF_SRC/test/tcl/clm/drv_clmin.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/drv_vegm.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/drv_vegp.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/lai.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/sai.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/z0m.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/displa.dat', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/narr_1hr.sc3.txt.0', new_output_dir_name)
cp('$PF_SRC/test/tcl/clm/veg_map.cpfb', new_output_dir_name + '/veg_map.pfb')

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------

clm_veg.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--p', default=1)
parser.add_argument('-q', '--q', default=1)
parser.add_argument('-r', '--r', default=1)
args = parser.parse_args()

clm_veg.Process.Topology.P = args.p
clm_veg.Process.Topology.Q = args.q
clm_veg.Process.Topology.R = args.r

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
clm_veg.ComputationalGrid.Lower.X = 0.0
clm_veg.ComputationalGrid.Lower.Y = 0.0
clm_veg.ComputationalGrid.Lower.Z = 0.0

clm_veg.ComputationalGrid.DX = 1000.
clm_veg.ComputationalGrid.DY = 1000.
clm_veg.ComputationalGrid.DZ = 0.5

clm_veg.ComputationalGrid.NX = 5
clm_veg.ComputationalGrid.NY = 5
clm_veg.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------

clm_veg.GeomInput.Names = 'domain_input'

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------

clm_veg.GeomInput.domain_input.InputType = 'Box'
clm_veg.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------

clm_veg.Geom.domain.Lower.X = 0.0
clm_veg.Geom.domain.Lower.Y = 0.0
clm_veg.Geom.domain.Lower.Z = 0.0

clm_veg.Geom.domain.Upper.X = 5000.
clm_veg.Geom.domain.Upper.Y = 5000.
clm_veg.Geom.domain.Upper.Z = 5.

clm_veg.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

clm_veg.Geom.Perm.Names = 'domain'

clm_veg.Geom.domain.Perm.Type = 'Constant'
clm_veg.Geom.domain.Perm.Value = 0.2


clm_veg.Perm.TensorType = 'TensorByGeom'

clm_veg.Geom.Perm.TensorByGeom.Names = 'domain'

clm_veg.Geom.domain.Perm.TensorValX = 1.0
clm_veg.Geom.domain.Perm.TensorValY = 1.0
clm_veg.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

clm_veg.SpecificStorage.Type = 'Constant'
clm_veg.SpecificStorage.GeomNames = 'domain'
clm_veg.Geom.domain.SpecificStorage.Value = 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

clm_veg.Phase.Names = 'water'

clm_veg.Phase.water.Density.Type = 'Constant'
clm_veg.Phase.water.Density.Value = 1.0

clm_veg.Phase.water.Viscosity.Type = 'Constant'
clm_veg.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

clm_veg.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

clm_veg.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

clm_veg.TimingInfo.BaseUnit = 1.0
clm_veg.TimingInfo.StartCount = 0
clm_veg.TimingInfo.StartTime = 0.0
clm_veg.TimingInfo.StopTime = 5
clm_veg.TimingInfo.DumpInterval = -1
clm_veg.TimeStep.Type = 'Constant'
clm_veg.TimeStep.Value = 1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

clm_veg.Geom.Porosity.GeomNames = 'domain'
clm_veg.Geom.domain.Porosity.Type = 'Constant'
clm_veg.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

clm_veg.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------

clm_veg.Phase.water.Mobility.Type = 'Constant'
clm_veg.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

clm_veg.Phase.RelPerm.Type = 'VanGenuchten'
clm_veg.Phase.RelPerm.GeomNames = 'domain'

clm_veg.Geom.domain.RelPerm.Alpha = 3.5
clm_veg.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

clm_veg.Phase.Saturation.Type = 'VanGenuchten'
clm_veg.Phase.Saturation.GeomNames = 'domain'

clm_veg.Geom.domain.Saturation.Alpha = 3.5
clm_veg.Geom.domain.Saturation.N = 2.
clm_veg.Geom.domain.Saturation.SRes = 0.01
clm_veg.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

clm_veg.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

clm_veg.Cycle.Names = 'constant'
clm_veg.Cycle.constant.Names = 'alltime'
clm_veg.Cycle.constant.alltime.Length = 1
clm_veg.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

clm_veg.BCPressure.PatchNames = clm_veg.Geom.domain.Patches

clm_veg.Patch.x_lower.BCPressure.Type = 'FluxConst'
clm_veg.Patch.x_lower.BCPressure.Cycle = 'constant'
clm_veg.Patch.x_lower.BCPressure.alltime.Value = 0.0

clm_veg.Patch.y_lower.BCPressure.Type = 'FluxConst'
clm_veg.Patch.y_lower.BCPressure.Cycle = 'constant'
clm_veg.Patch.y_lower.BCPressure.alltime.Value = 0.0

clm_veg.Patch.z_lower.BCPressure.Type = 'FluxConst'
clm_veg.Patch.z_lower.BCPressure.Cycle = 'constant'
clm_veg.Patch.z_lower.BCPressure.alltime.Value = 0.0

clm_veg.Patch.x_upper.BCPressure.Type = 'FluxConst'
clm_veg.Patch.x_upper.BCPressure.Cycle = 'constant'
clm_veg.Patch.x_upper.BCPressure.alltime.Value = 0.0

clm_veg.Patch.y_upper.BCPressure.Type = 'FluxConst'
clm_veg.Patch.y_upper.BCPressure.Cycle = 'constant'
clm_veg.Patch.y_upper.BCPressure.alltime.Value = 0.0

clm_veg.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
clm_veg.Patch.z_upper.BCPressure.Cycle = 'constant'
clm_veg.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

clm_veg.TopoSlopesX.Type = 'Constant'
clm_veg.TopoSlopesX.GeomNames = 'domain'
clm_veg.TopoSlopesX.Geom.domain.Value = -0.001

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

clm_veg.TopoSlopesY.Type = 'Constant'
clm_veg.TopoSlopesY.GeomNames = 'domain'
clm_veg.TopoSlopesY.Geom.domain.Value = 0.001

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

clm_veg.Mannings.Type = 'Constant'
clm_veg.Mannings.GeomNames = 'domain'
clm_veg.Mannings.Geom.domain.Value = 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

clm_veg.PhaseSources.water.Type = 'Constant'
clm_veg.PhaseSources.water.GeomNames = 'domain'
clm_veg.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

clm_veg.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

clm_veg.Solver = 'Richards'
clm_veg.Solver.MaxIter = 500

clm_veg.Solver.Nonlinear.MaxIter = 15
clm_veg.Solver.Nonlinear.ResidualTol = 1e-9
clm_veg.Solver.Nonlinear.EtaChoice = 'EtaConstant'
clm_veg.Solver.Nonlinear.EtaValue = 0.01
clm_veg.Solver.Nonlinear.UseJacobian = True
clm_veg.Solver.Nonlinear.StepTol = 1e-20
clm_veg.Solver.Nonlinear.Globalization = 'LineSearch'
clm_veg.Solver.Linear.KrylovDimension = 15
clm_veg.Solver.Linear.MaxRestart = 2

clm_veg.Solver.Linear.Preconditioner = 'PFMG'
clm_veg.Solver.PrintSubsurf = False
clm_veg.Solver.Drop = 1E-20
clm_veg.Solver.AbsTol = 1E-9

clm_veg.Solver.LSM = 'CLM'
clm_veg.Solver.CLM.MetForcing = '1D'
clm_veg.Solver.CLM.MetFileName = 'narr_1hr.sc3.txt.0'
clm_veg.Solver.CLM.MetFilePath = '.'
clm_veg.Solver.CLM.ForceVegetation = True

clm_veg.Solver.PrintCLM = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

clm_veg.ICPressure.Type = 'HydroStaticPatch'
clm_veg.ICPressure.GeomNames = 'domain'
clm_veg.Geom.domain.ICPressure.Value = -2.0

clm_veg.Geom.domain.ICPressure.RefGeom = 'domain'
clm_veg.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

clm_veg.dist(new_output_dir_name + '/veg_map.pfb')

correct_output_dir_name = get_absolute_path("../../correct_output/clm_output")
clm_veg.run(working_directory=new_output_dir_name)

passed = True

test_files = ["perm_x", "perm_y", "perm_z"]

for test_file in test_files:
    filename = f"/{run_name}.out.{test_file}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in {test_file}"):
        passed = False

for i in range(6):
    timestep = str(i).rjust(5, '0')
    filename = f"/{run_name}.out.press.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Pressure for timestep {timestep}"):
        passed = False
    filename = f"/{run_name}.out.satur.{timestep}.pfb"
    if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in Saturation for timestep {timestep}"):
        passed = False

mask = read_pfb(new_output_dir_name + f"/{run_name}.out.mask.pfb")
top = compute_top(mask)
write_pfb(new_output_dir_name + f"/{run_name}.out.top_index.pfb", top)

data = read_pfb(new_output_dir_name + f"/{run_name}.out.press.00000.pfb")
top_data = extract_top(data, top)
write_pfb(new_output_dir_name + f"/{run_name}.out.top.press.00000.pfb", top_data)


filename = f"/{run_name}.out.top_index.pfb"
if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in top_index"):
    passed = False

filename = f"/{run_name}.out.top.press.00000.pfb"
if not pf_test_file(new_output_dir_name + filename, correct_output_dir_name + filename, f"Max difference in top_clm.out.press.00000.pfb"):
    passed = False

rm(new_output_dir_name)    
if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
