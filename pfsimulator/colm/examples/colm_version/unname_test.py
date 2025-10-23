# Last Change:  2024-01-13 08:20:53
# SCRIPT TO RUN THE DECADAL SIMULATION USING THE NEW PFTOOLS

import sys
import os
from datetime import datetime
from parflow.tools import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path, exists
from parflow.tools.settings import set_working_directory


# -----------------------------------------------------------------------------
# Get initial inputs
# -----------------------------------------------------------------------------

runname = 'unname_test'

CONCN = Run(runname, __file__)
CONCN.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------

CONCN.Process.Topology.P = 1
CONCN.Process.Topology.Q = 1
CONCN.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Make a directory for the simulation and copy inputs into it
# -----------------------------------------------------------------------------

set_working_directory(get_absolute_path('.'))

# ParFlow Inputs
# cp('../inputs/model_inputs/CONCN.slopex.pfb')
# cp('../inputs/model_inputs/CONCN.slopey.pfb')
# cp('../inputs/model_inputs/CONCN.0.Final1km_mannings_rv50_original_values.pfb')
# fill 1e-5 constant, may no use
# cp('../inputs/model_inputs/1km_CONCN_PME_fixed_GPU.pfb')
# fill 0
# cp('../inputs/model_inputs/GLHYMPS1.0_multi_efold.pfb')
# cp('../inputs/model_inputs/Shangguan_300m_FBZ_fix.pfb')
# cp('../inputs/model_inputs/CONCN.0_fix115.pfsol')
# remove

# -----------------------------------------------------------------------------
# Initial pressure
# -----------------------------------------------------------------------------

# curr_step = 1519
# cp('../inputs/model_inputs/initial_GPU.pfb')
# ip = 'initial_GPU.pfb'
# ip = runname+'.init.press.pfb'
#ip = 'unname.initial.pfb'
# ip = 'unname_test.out.press.02400.pfb'

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

CONCN.ComputationalGrid.Lower.X = 0.0
CONCN.ComputationalGrid.Lower.Y = 0.0
CONCN.ComputationalGrid.Lower.Z = 0.0

CONCN.ComputationalGrid.DX = 961.72
CONCN.ComputationalGrid.DY = 961.72
CONCN.ComputationalGrid.DZ = 200.0

CONCN.ComputationalGrid.NX = 6
CONCN.ComputationalGrid.NY = 5
CONCN.ComputationalGrid.NZ = 11

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

CONCN.GeomInput.Names = "domaininput indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

#CONCN.GeomInput.domaininput.InputType = 'SolidFile'
#CONCN.GeomInput.domaininput.GeomNames = 'domain'
#CONCN.GeomInput.domaininput.FileName = 'CONCN_str.pfsol'
#CONCN.Geom.domain.Patches = 'ocean land top sink bottom'

CONCN.GeomInput.domaininput.InputType = 'Box'
CONCN.GeomInput.domaininput.GeomName = 'domain'
CONCN.Geom.domain.Lower.X = 0.0
CONCN.Geom.domain.Lower.Y = 0.0
CONCN.Geom.domain.Lower.Z = 0.0
#
CONCN.Geom.domain.Upper.X = 6*961.72
CONCN.Geom.domain.Upper.Y = 5*961.72
CONCN.Geom.domain.Upper.Z = 2200.0
CONCN.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'
#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------

# CONCN.Geom.domain.Patches = "ocean land top lake sink bottom"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

CONCN.GeomInput.indi_input.InputType = 'IndicatorField'
CONCN.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'
CONCN.Geom.indi_input.FileName = 'unname.subsur.pfb'

CONCN.GeomInput.s1.Value = 1
CONCN.GeomInput.s2.Value = 2
CONCN.GeomInput.s3.Value = 3
CONCN.GeomInput.s4.Value = 4
CONCN.GeomInput.s5.Value = 5
CONCN.GeomInput.s6.Value = 6
CONCN.GeomInput.s7.Value = 7
CONCN.GeomInput.s8.Value = 8
CONCN.GeomInput.s9.Value = 9
CONCN.GeomInput.s10.Value = 10
CONCN.GeomInput.s11.Value = 11
CONCN.GeomInput.s12.Value = 12

CONCN.GeomInput.s13.Value = 13

CONCN.GeomInput.b1.Value = 19
CONCN.GeomInput.b2.Value = 20

CONCN.GeomInput.g1.Value = 21
CONCN.GeomInput.g2.Value = 22
CONCN.GeomInput.g3.Value = 23
CONCN.GeomInput.g4.Value = 24
CONCN.GeomInput.g5.Value = 25
CONCN.GeomInput.g6.Value = 26
CONCN.GeomInput.g7.Value = 27
CONCN.GeomInput.g8.Value = 28

#--------------------------------------------
# variable dz assignments
#------------------------------------------
CONCN.Solver.Nonlinear.VariableDz = True
CONCN.dzScale.GeomNames = 'domain'
CONCN.dzScale.Type = 'nzList'
CONCN.dzScale.nzListNumber = 11

# 10 layers, starts at 0 for the bottom to 9 at the top
# note this is opposite Noah/WRF
# layers are 0.1 m, 0.3 m, 0.6 m, 1.0 m, 5.0 m, 10.0 m, 25.0 m, 50.0 m, 100.0m, 200.0 m
# 200 m * 1.5 = 300 m
CONCN.Cell._0.dzScale.Value = 100/200
# 200 m * .5 = 100 m
CONCN.Cell._1.dzScale.Value = (3.4331-2.2961)/200
# 200 m * .25 = 50 m
CONCN.Cell._2.dzScale.Value = (2.2961-1.3828)/200
# 200 m * 0.125 = 25 m
CONCN.Cell._3.dzScale.Value = (1.3828-0.8289)/200
# 200 m * 0.05 = 10 m
CONCN.Cell._4.dzScale.Value = (0.8289-0.4929)/200
# 200 m * .025 = 5 m
CONCN.Cell._5.dzScale.Value = (0.4929-0.2891)/200
# 200 m * .005 = 1 m
CONCN.Cell._6.dzScale.Value = (0.2891-0.1655)/200
# 200 m * 0.003 = 0.6 m
CONCN.Cell._7.dzScale.Value = (0.1655-0.0906)/200
# 200 m * 0.0015 = 0.3 m
CONCN.Cell._8.dzScale.Value = (0.0906-0.0451)/200
# 200 m * 0.0005 = 0.1 m = 10 cm which is default top Noah layer
CONCN.Cell._9.dzScale.Value = (0.0451-0.0175)/200

CONCN.Cell._10.dzScale.Value = 0.0175/200

# ------------------------------------------------------------------------------
# Flow Barrier defined by Shangguan Depth to Bedrock
# --------------------------------------------------------------

#CONCN.Solver.Nonlinear.FlowBarrierZ = True
#CONCN.FBz.Type = 'PFBFile'
#CONCN.Geom.domain.FBz.FileName = 'Shangguan_300m_FBZ_fix.pfb'
#CONCN.dist('Shangguan_300m_FBZ_fix.pfb')

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

CONCN.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

CONCN.Geom.domain.Perm.Type = 'Constant'
CONCN.Geom.domain.Perm.Value = 0.02

CONCN.Geom.s1.Perm.Type = 'Constant'
CONCN.Geom.s1.Perm.Value = 0.269022595

CONCN.Geom.s2.Perm.Type = 'Constant'
CONCN.Geom.s2.Perm.Value = 0.043630356

CONCN.Geom.s3.Perm.Type = 'Constant'
CONCN.Geom.s3.Perm.Value = 0.015841225

CONCN.Geom.s4.Perm.Type = 'Constant'
CONCN.Geom.s4.Perm.Value = 0.007582087

CONCN.Geom.s5.Perm.Type = 'Constant'
CONCN.Geom.s5.Perm.Value = 0.01818816

CONCN.Geom.s6.Perm.Type = 'Constant'
CONCN.Geom.s6.Perm.Value = 0.005009435

CONCN.Geom.s7.Perm.Type = 'Constant'
CONCN.Geom.s7.Perm.Value = 0.005492736

CONCN.Geom.s8.Perm.Type = 'Constant'
CONCN.Geom.s8.Perm.Value = 0.004675077

CONCN.Geom.s9.Perm.Type = 'Constant'
CONCN.Geom.s9.Perm.Value = 0.003386794

CONCN.Geom.s10.Perm.Type = 'Constant'
CONCN.Geom.s10.Perm.Value = 0.004783973

CONCN.Geom.s11.Perm.Type = 'Constant'
CONCN.Geom.s11.Perm.Value = 0.003979136

CONCN.Geom.s12.Perm.Type = 'Constant'
CONCN.Geom.s12.Perm.Value = 0.006162952

CONCN.Geom.s13.Perm.Type = 'Constant'
CONCN.Geom.s13.Perm.Value = 0.005009435

CONCN.Geom.b1.Perm.Type = 'Constant'
CONCN.Geom.b1.Perm.Value = 0.005

CONCN.Geom.b2.Perm.Type = 'Constant'
CONCN.Geom.b2.Perm.Value = 0.01

CONCN.Geom.g1.Perm.Type = 'Constant'
CONCN.Geom.g1.Perm.Value = 0.02

CONCN.Geom.g2.Perm.Type = 'Constant'
CONCN.Geom.g2.Perm.Value = 0.03

CONCN.Geom.g3.Perm.Type = 'Constant'
CONCN.Geom.g3.Perm.Value = 0.04

CONCN.Geom.g4.Perm.Type = 'Constant'
CONCN.Geom.g4.Perm.Value = 0.05

CONCN.Geom.g5.Perm.Type = 'Constant'
CONCN.Geom.g5.Perm.Value = 0.06

CONCN.Geom.g6.Perm.Type = 'Constant'
CONCN.Geom.g6.Perm.Value = 0.08

CONCN.Geom.g7.Perm.Type = 'Constant'
CONCN.Geom.g7.Perm.Value = 0.1

CONCN.Geom.g8.Perm.Type = 'Constant'
CONCN.Geom.g8.Perm.Value = 0.2

CONCN.Perm.TensorType = 'TensorByGeom'
CONCN.Geom.Perm.TensorByGeom.Names = 'domain b1 b2 g1 g2 g4 g5 g6 g7'

CONCN.Geom.domain.Perm.TensorValX = 1.0
CONCN.Geom.domain.Perm.TensorValY = 1.0
CONCN.Geom.domain.Perm.TensorValZ = 1.0

CONCN.Geom.b1.Perm.TensorValX = 1.0
CONCN.Geom.b1.Perm.TensorValY = 1.0
CONCN.Geom.b1.Perm.TensorValZ = 0.1

CONCN.Geom.b2.Perm.TensorValX = 1.0
CONCN.Geom.b2.Perm.TensorValY = 1.0
CONCN.Geom.b2.Perm.TensorValZ = 0.1

CONCN.Geom.g1.Perm.TensorValX = 1.0
CONCN.Geom.g1.Perm.TensorValY = 1.0
CONCN.Geom.g1.Perm.TensorValZ = 0.1

CONCN.Geom.g2.Perm.TensorValX = 1.0
CONCN.Geom.g2.Perm.TensorValY = 1.0
CONCN.Geom.g2.Perm.TensorValZ = 0.1

CONCN.Geom.g4.Perm.TensorValX = 1.0
CONCN.Geom.g4.Perm.TensorValY = 1.0
CONCN.Geom.g4.Perm.TensorValZ = 0.1

CONCN.Geom.g5.Perm.TensorValX = 1.0
CONCN.Geom.g5.Perm.TensorValY = 1.0
CONCN.Geom.g5.Perm.TensorValZ = 0.1

CONCN.Geom.g6.Perm.TensorValX = 1.0
CONCN.Geom.g6.Perm.TensorValY = 1.0
CONCN.Geom.g6.Perm.TensorValZ = 0.1

CONCN.Geom.g7.Perm.TensorValX = 1.0
CONCN.Geom.g7.Perm.TensorValY = 1.0
CONCN.Geom.g7.Perm.TensorValZ = 0.1

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

CONCN.SpecificStorage.Type = 'Constant'
CONCN.SpecificStorage.GeomNames = 'domain'
CONCN.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

CONCN.Phase.Names = 'water'
CONCN.Phase.water.Density.Type = 'Constant'
CONCN.Phase.water.Density.Value = 1.0
CONCN.Phase.water.Viscosity.Type = 'Constant'
CONCN.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

CONCN.Contaminants.Names = ''

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

CONCN.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

CONCN.TimingInfo.BaseUnit = 1.0
CONCN.TimingInfo.StartCount = 0
CONCN.TimingInfo.StartTime = 0
CONCN.TimingInfo.StopTime = 8760
#CONCN.TimingInfo.StopTime = curr_step + 2
CONCN.TimingInfo.DumpInterval = 1.
CONCN.TimeStep.Type = 'Constant'
CONCN.TimeStep.Value = 1.

# CONCN.TimeStep.Type = 'Growth'
# CONCN.TimeStep.InitialStep = 0.1
# CONCN.TimeStep.GrowthFactor = 1.1
# CONCN.TimeStep.MaxStep = 100
# CONCN.TimeStep.MinStep = 0.1

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

CONCN.Geom.Porosity.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

CONCN.Geom.domain.Porosity.Type = 'Constant'
CONCN.Geom.domain.Porosity.Value = 0.33

CONCN.Geom.s1.Porosity.Type = 'Constant'
CONCN.Geom.s1.Porosity.Value = 0.375

CONCN.Geom.s2.Porosity.Type = 'Constant'
CONCN.Geom.s2.Porosity.Value = 0.39

CONCN.Geom.s3.Porosity.Type = 'Constant'
CONCN.Geom.s3.Porosity.Value = 0.387

CONCN.Geom.s4.Porosity.Type = 'Constant'
CONCN.Geom.s4.Porosity.Value = 0.439

CONCN.Geom.s5.Porosity.Type = 'Constant'
CONCN.Geom.s5.Porosity.Value = 0.489

CONCN.Geom.s6.Porosity.Type = 'Constant'
CONCN.Geom.s6.Porosity.Value = 0.399

CONCN.Geom.s7.Porosity.Type = 'Constant'
CONCN.Geom.s7.Porosity.Value = 0.384

CONCN.Geom.s8.Porosity.Type = 'Constant'
CONCN.Geom.s8.Porosity.Value = 0.482

CONCN.Geom.s9.Porosity.Type = 'Constant'
CONCN.Geom.s9.Porosity.Value = 0.442

CONCN.Geom.s10.Porosity.Type = 'Constant'
CONCN.Geom.s10.Porosity.Value = 0.385

CONCN.Geom.s11.Porosity.Type = 'Constant'
CONCN.Geom.s11.Porosity.Value = 0.481

CONCN.Geom.s12.Porosity.Type = 'Constant'
CONCN.Geom.s12.Porosity.Value = 0.459

CONCN.Geom.s13.Porosity.Type = 'Constant'
CONCN.Geom.s13.Porosity.Value = 0.399

CONCN.Geom.b1.Porosity.Type = 'Constant'
CONCN.Geom.b1.Porosity.Value = 0.1

CONCN.Geom.b2.Porosity.Type = 'Constant'
CONCN.Geom.b2.Porosity.Value = 0.05

CONCN.Geom.g1.Porosity.Type = 'Constant'
CONCN.Geom.g1.Porosity.Value = 0.12

CONCN.Geom.g2.Porosity.Type = 'Constant'
CONCN.Geom.g2.Porosity.Value = 0.3

CONCN.Geom.g3.Porosity.Type = 'Constant'
CONCN.Geom.g3.Porosity.Value = 0.01

CONCN.Geom.g4.Porosity.Type = 'Constant'
CONCN.Geom.g4.Porosity.Value = 0.15

CONCN.Geom.g5.Porosity.Type = 'Constant'
CONCN.Geom.g5.Porosity.Value = 0.22

CONCN.Geom.g6.Porosity.Type = 'Constant'
CONCN.Geom.g6.Porosity.Value = 0.27

CONCN.Geom.g7.Porosity.Type = 'Constant'
CONCN.Geom.g7.Porosity.Value = 0.06

CONCN.Geom.g8.Porosity.Type = 'Constant'
CONCN.Geom.g8.Porosity.Value = 0.3

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

CONCN.Domain.GeomName = 'domain'

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

CONCN.Phase.water.Mobility.Type = 'Constant'
CONCN.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

CONCN.Wells.Names = ''

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

CONCN.Phase.RelPerm.Type = 'VanGenuchten'
CONCN.Phase.RelPerm.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'

CONCN.Geom.domain.RelPerm.Alpha = 0.5
CONCN.Geom.domain.RelPerm.N = 2.5
CONCN.Geom.domain.RelPerm.NumSamplePoints = 20000
CONCN.Geom.domain.RelPerm.MinPressureHead = -500
CONCN.Geom.domain.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s1.RelPerm.Alpha = 3.548
CONCN.Geom.s1.RelPerm.N = 4.162
CONCN.Geom.s1.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s1.RelPerm.MinPressureHead = -300
CONCN.Geom.s1.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s2.RelPerm.Alpha = 3.467
CONCN.Geom.s2.RelPerm.N = 2.738
CONCN.Geom.s2.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s2.RelPerm.MinPressureHead = -300
CONCN.Geom.s2.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s3.RelPerm.Alpha = 2.692
CONCN.Geom.s3.RelPerm.N = 2.445
CONCN.Geom.s3.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s3.RelPerm.MinPressureHead = -300
CONCN.Geom.s3.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s4.RelPerm.Alpha = 0.501
CONCN.Geom.s4.RelPerm.N = 2.659
CONCN.Geom.s4.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s4.RelPerm.MinPressureHead = -300
CONCN.Geom.s4.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s5.RelPerm.Alpha = 0.661
CONCN.Geom.s5.RelPerm.N = 2.659
CONCN.Geom.s5.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s5.RelPerm.MinPressureHead = -300
CONCN.Geom.s5.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s6.RelPerm.Alpha = 1.122
CONCN.Geom.s6.RelPerm.N = 2.479
CONCN.Geom.s6.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s6.RelPerm.MinPressureHead = -300
CONCN.Geom.s6.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s7.RelPerm.Alpha = 2.089
CONCN.Geom.s7.RelPerm.N = 2.318
CONCN.Geom.s7.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s7.RelPerm.MinPressureHead = -300
CONCN.Geom.s7.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s8.RelPerm.Alpha = 0.832
CONCN.Geom.s8.RelPerm.N = 2.514
CONCN.Geom.s8.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s8.RelPerm.MinPressureHead = -300
CONCN.Geom.s8.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s9.RelPerm.Alpha = 1.585
CONCN.Geom.s9.RelPerm.N = 2.413
CONCN.Geom.s9.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s9.RelPerm.MinPressureHead = -300
CONCN.Geom.s9.RelPerm.InterpolationMethod = 'Linear'


CONCN.Geom.s10.RelPerm.Alpha = 3.311
CONCN.Geom.s10.RelPerm.N = 2.202
CONCN.Geom.s10.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s10.RelPerm.MinPressureHead = -300
CONCN.Geom.s10.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s11.RelPerm.Alpha = 1.622
CONCN.Geom.s11.RelPerm.N = 2.318
CONCN.Geom.s11.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s11.RelPerm.MinPressureHead = -300
CONCN.Geom.s11.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s12.RelPerm.Alpha = 1.514
CONCN.Geom.s12.RelPerm.N = 2.259
CONCN.Geom.s12.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s12.RelPerm.MinPressureHead = -300
CONCN.Geom.s12.RelPerm.InterpolationMethod = 'Linear'

CONCN.Geom.s13.RelPerm.Alpha = 1.122
CONCN.Geom.s13.RelPerm.N = 2.479
CONCN.Geom.s13.RelPerm.NumSamplePoints = 20000
CONCN.Geom.s13.RelPerm.MinPressureHead = -300
CONCN.Geom.s13.RelPerm.InterpolationMethod = 'Linear'

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

CONCN.Phase.Saturation.Type = 'VanGenuchten'
CONCN.Phase.Saturation.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'

CONCN.Geom.domain.Saturation.Alpha = 0.5
CONCN.Geom.domain.Saturation.N = 2.5
CONCN.Geom.domain.Saturation.SRes = 0.00001
CONCN.Geom.domain.Saturation.SSat = 1.0

CONCN.Geom.s1.Saturation.Alpha = 3.548
CONCN.Geom.s1.Saturation.N = 4.162
CONCN.Geom.s1.Saturation.SRes = 0.0001
CONCN.Geom.s1.Saturation.SSat = 1.0

CONCN.Geom.s2.Saturation.Alpha = 3.467
CONCN.Geom.s2.Saturation.N = 2.738
CONCN.Geom.s2.Saturation.SRes = 0.0001
CONCN.Geom.s2.Saturation.SSat = 1.0

CONCN.Geom.s3.Saturation.Alpha = 2.692
CONCN.Geom.s3.Saturation.N = 2.445
CONCN.Geom.s3.Saturation.SRes = 0.0001
CONCN.Geom.s3.Saturation.SSat = 1.0

CONCN.Geom.s4.Saturation.Alpha = 0.501
CONCN.Geom.s4.Saturation.N = 2.659
CONCN.Geom.s4.Saturation.SRes = 0.0001
CONCN.Geom.s4.Saturation.SSat = 1.0

CONCN.Geom.s5.Saturation.Alpha = 0.661
CONCN.Geom.s5.Saturation.N = 2.659
CONCN.Geom.s5.Saturation.SRes = 0.0001
CONCN.Geom.s5.Saturation.SSat = 1.0

CONCN.Geom.s6.Saturation.Alpha = 1.122
CONCN.Geom.s6.Saturation.N = 2.479
CONCN.Geom.s6.Saturation.SRes = 0.0001
CONCN.Geom.s6.Saturation.SSat = 1.0

CONCN.Geom.s7.Saturation.Alpha = 2.089
CONCN.Geom.s7.Saturation.N = 2.318
CONCN.Geom.s7.Saturation.SRes = 0.0001
CONCN.Geom.s7.Saturation.SSat = 1.0

CONCN.Geom.s8.Saturation.Alpha = 0.832
CONCN.Geom.s8.Saturation.N = 2.514
CONCN.Geom.s8.Saturation.SRes = 0.0001
CONCN.Geom.s8.Saturation.SSat = 1.0

CONCN.Geom.s9.Saturation.Alpha = 1.585
CONCN.Geom.s9.Saturation.N = 2.413
CONCN.Geom.s9.Saturation.SRes = 0.0001
CONCN.Geom.s9.Saturation.SSat = 1.0

CONCN.Geom.s10.Saturation.Alpha = 3.311
CONCN.Geom.s10.Saturation.N = 2.202
CONCN.Geom.s10.Saturation.SRes = 0.0001
CONCN.Geom.s10.Saturation.SSat = 1.0

CONCN.Geom.s11.Saturation.Alpha = 1.622
CONCN.Geom.s11.Saturation.N = 2.318
CONCN.Geom.s11.Saturation.SRes = 0.0001
CONCN.Geom.s11.Saturation.SSat = 1.0

CONCN.Geom.s12.Saturation.Alpha = 1.514
CONCN.Geom.s12.Saturation.N = 2.259
CONCN.Geom.s12.Saturation.SRes = 0.0001
CONCN.Geom.s12.Saturation.SSat = 1.0

CONCN.Geom.s13.Saturation.Alpha = 1.122
CONCN.Geom.s13.Saturation.N = 2.479
CONCN.Geom.s13.Saturation.SRes = 0.0001
CONCN.Geom.s13.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

CONCN.Cycle.Names = 'constant rainrec'
CONCN.Cycle.constant.Names = 'alltime'
CONCN.Cycle.constant.alltime.Length = 1
CONCN.Cycle.constant.Repeat = -1

CONCN.Cycle.rainrec.Names = 'rain rec'
CONCN.Cycle.rainrec.rain.Length = 10
CONCN.Cycle.rainrec.rec.Length = 150
CONCN.Cycle.rainrec.Repeat = -1

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

CONCN.BCPressure.PatchNames = CONCN.Geom.domain.Patches
#'x_lower x_upper y_lower y_upper z_lower z_upper'

CONCN.Patch.x_lower.BCPressure.Type = 'FluxConst'
CONCN.Patch.x_lower.BCPressure.Cycle = 'constant'
CONCN.Patch.x_lower.BCPressure.alltime.Value = 0.0

CONCN.Patch.x_upper.BCPressure.Type = 'FluxConst'
CONCN.Patch.x_upper.BCPressure.Cycle = 'constant'
CONCN.Patch.x_upper.BCPressure.alltime.Value = 0.0

CONCN.Patch.y_lower.BCPressure.Type = 'FluxConst'
CONCN.Patch.y_lower.BCPressure.Cycle = 'constant'
CONCN.Patch.y_lower.BCPressure.alltime.Value = 0.0

CONCN.Patch.y_upper.BCPressure.Type = 'FluxConst'
CONCN.Patch.y_upper.BCPressure.Cycle = 'constant'
CONCN.Patch.y_upper.BCPressure.alltime.Value = 0.0

CONCN.Patch.z_lower.BCPressure.Type = 'FluxConst'
CONCN.Patch.z_lower.BCPressure.Cycle = 'constant'
CONCN.Patch.z_lower.BCPressure.alltime.Value = 0.0

CONCN.Patch.z_upper.BCPressure.Type = 'OverlandKinematic'
CONCN.Patch.z_upper.BCPressure.Cycle = 'constant'
CONCN.Patch.z_upper.BCPressure.alltime.Value = 0

#CONCN.Solver.EvapTransFile = True
#CONCN.Solver.EvapTrans.FileName = 'CONCN_PME_GPU.pfb'
#CONCN.dist('CONCN_PME_GPU.pfb')

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

CONCN.TopoSlopesX.Type = 'PFBFile'
CONCN.TopoSlopesX.GeomNames = 'domain'
CONCN.TopoSlopesX.FileName = 'unname.slopex.pfb'

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

CONCN.TopoSlopesY.Type = 'PFBFile'
CONCN.TopoSlopesY.GeomNames = 'domain'
CONCN.TopoSlopesY.FileName = 'unname.slopey.pfb'

# -----------------------------------------------------------------------------
# Initial conditions: water pressure
# -----------------------------------------------------------------------------

# CONCN.ICPressure.Type = 'PFBFile'
# CONCN.ICPressure.GeomNames = 'domain'
# CONCN.Geom.domain.ICPressure.FileName = ip
# CONCN.dist(ip)

# CONCN.ICPressure.Type = 'HydroStaticPatch'
# CONCN.Geom.domain.ICPressure.RefPatch = 'z_upper'
# CONCN.Geom.domain.ICPressure.RefGeom = 'domain'
# CONCN.Geom.domain.ICPressure.Value = 372.

CONCN.ICPressure.Type                                   = 'HydroStaticPatch'
CONCN.ICPressure.GeomNames                              = 'domain'
CONCN.Geom.domain.ICPressure.Value                      = -0.1
CONCN.Geom.domain.ICPressure.RefGeom                    = 'domain'
CONCN.Geom.domain.ICPressure.RefPatch                   = 'z_upper'

#-----------------------------------------------------------------------------
# Distribute inputs
#-----------------------------------------------------------------------------
CONCN.dist('unname.slopex.pfb')
CONCN.dist('unname.slopey.pfb')
CONCN.dist('unname.subsur.pfb')
#CONCN.dist(ip)

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

CONCN.PhaseSources.water.Type = 'Constant'
CONCN.PhaseSources.water.GeomNames = 'domain'
CONCN.PhaseSources.water.Geom.domain.Value = 0.0

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

CONCN.Mannings.Type = 'PFBFile'
CONCN.Mannings.GeomNames = 'domain'
CONCN.Mannings.FileName = 'unname.manning.pfb'
CONCN.dist('unname.manning.pfb')

# CONCN.Mannings.Type = 'Constant'
# CONCN.Mannings.GeomNames = 'domain'
# CONCN.Mannings.Geom.domain.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

CONCN.KnownSolution = 'NoKnownSolution'

CONCN.Solver.LSM                   = 'CoLM'
CONCN.Solver.CLM.CLMFileDir        = 'clm_output_path'
CONCN.Solver.CLM.Print1dOut        = False
CONCN.Solver.CLM.CLMDumpInterval   = 1

CONCN.Solver.CLM.MetForcing        = '1D'
CONCN.Solver.CLM.MetFileName       = 'station0.txt'
CONCN.Solver.CLM.MetFilePath       = './' 
CONCN.Solver.CLM.MetFileNT         = 24
CONCN.Solver.CLM.IstepStart        = 1

CONCN.Solver.CLM.EvapBeta          = 'Linear'
CONCN.Solver.CLM.VegWaterStress    = 'Saturation'
CONCN.Solver.CLM.ResSat            = 0.2
CONCN.Solver.CLM.WiltingPoint      = 0.2
CONCN.Solver.CLM.FieldCapacity     = 1.00
CONCN.Solver.CLM.IrrigationType    = 'none'

CONCN.Solver.CLM.RootZoneNZ        = 10
CONCN.Solver.CLM.SoiLayer          = 10
CONCN.Solver.CLM.ReuseCount        = 1 #10 #4 #1
CONCN.Solver.CLM.WriteLogs         = False
CONCN.Solver.CLM.WriteLastRST      = True
CONCN.Solver.CLM.DailyRST          = True
CONCN.Solver.CLM.SingleFile        = True

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

CONCN.Solver = 'Richards'
CONCN.Solver.TerrainFollowingGrid = True
CONCN.Solver.TerrainFollowingGrid.SlopeUpwindFormulation = 'Upwind'

CONCN.Solver.MaxIter = 250000
#CONCN.Solver.Drop = 1E-30
CONCN.Solver.AbsTol = 1E-10
CONCN.Solver.MaxConvergenceFailures = 5
CONCN.Solver.Nonlinear.MaxIter = 250
CONCN.Solver.Nonlinear.ResidualTol = 1e-5
# CONCN.Solver.OverlandDiffusive.Epsilon = 0.1

# CONCN.Solver.PrintTop = True
## new solver settings for Terrain Following Grid
CONCN.Solver.Nonlinear.EtaChoice = 'EtaConstant'
CONCN.Solver.Nonlinear.EtaValue = 0.01
CONCN.Solver.Nonlinear.UseJacobian = True
# CONCN.Solver.Nonlinear.DerivativeEpsilon = 1e-16
CONCN.Solver.Nonlinear.StepTol = 1e-15
CONCN.Solver.Nonlinear.Globalization = 'LineSearch'
CONCN.Solver.Linear.KrylovDimension = 500
CONCN.Solver.Linear.MaxRestarts = 8

#CONCN.Solver.Linear.Preconditioner = 'MGSemi'
CONCN.Solver.Linear.Preconditioner = 'PFMG'
CONCN.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'
# CONCN.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 3
# CONCN.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 2

CONCN.Solver.PrintMask = True
CONCN.Solver.PrintVelocities = False
CONCN.Solver.PrintSaturation = True
CONCN.Solver.PrintPressure = True
#Writing output (no binary except Pressure, all silo):
CONCN.Solver.PrintSubsurfData = True
#pfset Solver.PrintLSMSink                        True
CONCN.Solver.WriteCLMBinary = False
CONCN.Solver.PrintCLM = True
CONCN.Solver.PrintEvapTrans = True

CONCN.Solver.WriteSiloSpecificStorage = False
CONCN.Solver.WriteSiloMannings = False
CONCN.Solver.WriteSiloMask = False
CONCN.Solver.WriteSiloSlopes = False
CONCN.Solver.WriteSiloSubsurfData = False
CONCN.Solver.WriteSiloPressure = False
CONCN.Solver.WriteSiloSaturation = False
CONCN.Solver.WriteSiloEvapTrans = False
CONCN.Solver.WriteSiloEvapTransSum = False
CONCN.Solver.WriteSiloOverlandSum = False
CONCN.Solver.WriteSiloCLM = False

# surface pressure test, new solver settings
CONCN.Solver.ResetSurfacePressure = True
CONCN.Solver.ResetSurfacePressure.ThresholdPressure = 50.
CONCN.Solver.ResetSurfacePressure.ResetPressure  =  -0.00001

CONCN.Solver.SurfacePredictor = True
CONCN.Solver.SurfacePredictor.PrintValues = False
CONCN.Solver.SurfacePredictor.PressureValue = 0.00001

# pfwritedb $runname

CONCN.run()
print("ParFlow run complete")
