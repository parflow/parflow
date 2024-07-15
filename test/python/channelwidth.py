# Import the ParFlow package
import os
from parflow import Run
import shutil
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm, exists

# Set our Run Name 
domain_example = Run("domain_example")

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
domain_example.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------
domain_example.Process.Topology.P = 1
domain_example.Process.Topology.Q = 1
domain_example.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
domain_example.ComputationalGrid.Lower.X = 0.0
domain_example.ComputationalGrid.Lower.Y = 0.0
domain_example.ComputationalGrid.Lower.Z = 0.0

domain_example.ComputationalGrid.DX      = 100.0
domain_example.ComputationalGrid.DY      = 2.0
domain_example.ComputationalGrid.DZ      = 1.0

domain_example.ComputationalGrid.NX      = 20
domain_example.ComputationalGrid.NY      = 1
domain_example.ComputationalGrid.NZ      = 10

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
domain_example.GeomInput.Names = 'domain_input'

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
domain_example.GeomInput.domain_input.InputType = 'Box'
domain_example.GeomInput.domain_input.GeomName  = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
domain_example.Geom.domain.Lower.X = 0.0
domain_example.Geom.domain.Lower.Y = 0.0
domain_example.Geom.domain.Lower.Z = 0.0

domain_example.Geom.domain.Upper.X = 2000.0
domain_example.Geom.domain.Upper.Y = 2.0
domain_example.Geom.domain.Upper.Z = 10.0

domain_example.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'


#--------------------------------------------
# Variable dz Assignments
#------------------------------------------
domain_example.Solver.Nonlinear.VariableDz = False
domain_example.dzScale.GeomNames           = 'domain'
domain_example.dzScale.Type                = 'nzList'
domain_example.dzScale.nzListNumber        = 10

# cells start at the bottom (0) and moves up to the top
# domain is 49 m thick, root zone is down to 4 cells 
# so the root zone is 2 m thick
domain_example.Cell._0.dzScale.Value  = 10.0   #10* 1.0 = 10  m  layer
domain_example.Cell._1.dzScale.Value  = 10.0   
domain_example.Cell._2.dzScale.Value  = 10.0   
domain_example.Cell._3.dzScale.Value  = 10.0
domain_example.Cell._4.dzScale.Value  = 5.0
domain_example.Cell._5.dzScale.Value  = 1.0
domain_example.Cell._6.dzScale.Value  = 1.0
domain_example.Cell._7.dzScale.Value  = 0.6   #0.6* 1.0 = 0.6  60 cm 3rd layer
domain_example.Cell._8.dzScale.Value  = 0.3   #0.3* 1.0 = 0.3  30 cm 2nd layer
domain_example.Cell._9.dzScale.Value  = 0.1   #0.1* 1.0 = 0.1  10 cm top layer

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
domain_example.Geom.Perm.Names              = 'domain'
domain_example.Geom.domain.Perm.Type        = 'Constant'
domain_example.Geom.domain.Perm.Value       = 0.01465  # m/h

domain_example.Perm.TensorType              = 'TensorByGeom'
domain_example.Geom.Perm.TensorByGeom.Names = 'domain'
domain_example.Geom.domain.Perm.TensorValX  = 1.0
domain_example.Geom.domain.Perm.TensorValY  = 1.0
domain_example.Geom.domain.Perm.TensorValZ  = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
domain_example.SpecificStorage.Type              = 'Constant'
domain_example.SpecificStorage.GeomNames         = 'domain'
domain_example.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
domain_example.Phase.Names = 'water'

domain_example.Phase.water.Density.Type     = 'Constant'
domain_example.Phase.water.Density.Value    = 1.0

domain_example.Phase.water.Viscosity.Type   = 'Constant'
domain_example.Phase.water.Viscosity.Value  = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
domain_example.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
domain_example.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup Timing
#-----------------------------------------------------------------------------
domain_example.TimingInfo.BaseUnit     = 1.0
domain_example.TimingInfo.StartCount   = 0
domain_example.TimingInfo.StartTime    = 0.0
domain_example.TimingInfo.StopTime     = 100.0
domain_example.TimingInfo.DumpInterval = 1.0
domain_example.TimeStep.Type           = 'Constant'
domain_example.TimeStep.Value          = 1.0


#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
domain_example.Geom.Porosity.GeomNames    = 'domain'

domain_example.Geom.domain.Porosity.Type  = 'Constant'
domain_example.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
domain_example.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
domain_example.Phase.water.Mobility.Type  = 'Constant'
domain_example.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
domain_example.Phase.RelPerm.Type        = 'VanGenuchten'
domain_example.Phase.RelPerm.GeomNames   = 'domain'

domain_example.Geom.domain.RelPerm.Alpha = 1.0
domain_example.Geom.domain.RelPerm.N     = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------
domain_example.Phase.Saturation.Type        = 'VanGenuchten'
domain_example.Phase.Saturation.GeomNames   = 'domain'

domain_example.Geom.domain.Saturation.Alpha = 1.0
domain_example.Geom.domain.Saturation.N     = 2.0
domain_example.Geom.domain.Saturation.SRes  = 0.2
domain_example.Geom.domain.Saturation.SSat  = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
domain_example.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
domain_example.Cycle.Names = 'constant'
domain_example.Cycle.constant.Names = 'alltime'
domain_example.Cycle.constant.alltime.Length = 1
domain_example.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
domain_example.BCPressure.PatchNames = 'x_lower x_upper y_lower y_upper z_lower z_upper'

domain_example.Patch.y_lower.BCPressure.Type          = 'FluxConst'
domain_example.Patch.y_lower.BCPressure.Cycle         = 'constant'
domain_example.Patch.y_lower.BCPressure.alltime.Value = 0.0

domain_example.Patch.z_lower.BCPressure.Type = 'FluxConst'
domain_example.Patch.z_lower.BCPressure.Cycle         = 'constant'
domain_example.Patch.z_lower.BCPressure.alltime.Value = 0.0

domain_example.Patch.x_lower.BCPressure.Type          = 'FluxConst'
domain_example.Patch.x_lower.BCPressure.Cycle         = 'constant'
domain_example.Patch.x_lower.BCPressure.alltime.Value = 0.0

domain_example.Patch.x_upper.BCPressure.Type          = 'DirEquilRefPatch'
domain_example.Patch.x_upper.BCPressure.RefGeom       = 'domain'
domain_example.Patch.x_upper.BCPressure.RefPatch      = 'z_upper'
domain_example.Patch.x_upper.BCPressure.Cycle         = 'constant'
domain_example.Patch.x_upper.BCPressure.alltime.Value = -1.0  # ocean boundary is 1m below land surface

domain_example.Patch.y_upper.BCPressure.Type          = 'FluxConst'
domain_example.Patch.y_upper.BCPressure.Cycle         = 'constant'
domain_example.Patch.y_upper.BCPressure.alltime.Value = 0.0

domain_example.Patch.z_upper.BCPressure.Type          = 'OverlandFlow'
domain_example.Patch.z_upper.BCPressure.Cycle         = 'constant'
domain_example.Patch.z_upper.BCPressure.alltime.Value = -0.01

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
domain_example.TopoSlopesX.Type              = 'Constant'
domain_example.TopoSlopesX.GeomNames         = 'domain'
domain_example.TopoSlopesX.Geom.domain.Value = -0.1  #slope in X-direction to allow ponded water to run off

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
domain_example.TopoSlopesY.Type              = 'Constant'
domain_example.TopoSlopesY.GeomNames         = 'domain'
domain_example.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Channel width in x-direction
#---------------------------------------------------------
domain_example.Solver.Nonlinear.ChannelWidthExistX = True
domain_example.ChannelWidthX.Type            = 'Constant'
domain_example.ChannelWidthX.GeomNames       = 'domain'
domain_example.ChannelWidthX.Geom.domain.Value = 1.0

#---------------------------------------------------------
# Channel width in y-direction
#---------------------------------------------------------
domain_example.Solver.Nonlinear.ChannelWidthExistY = True
domain_example.ChannelWidthY.Type            = 'Constant'
domain_example.ChannelWidthY.GeomNames       = 'domain'
domain_example.ChannelWidthY.Geom.domain.Value = 1.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
domain_example.Mannings.Type               = 'Constant'
domain_example.Mannings.GeomNames          = 'domain'
domain_example.Mannings.Geom.domain.Value  = 2.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
domain_example.PhaseSources.water.Type              = 'Constant'
domain_example.PhaseSources.water.GeomNames         = 'domain'
domain_example.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
domain_example.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
domain_example.Solver         = 'Richards'
domain_example.Solver.MaxIter = 9000

domain_example.Solver.Nonlinear.MaxIter           = 100
domain_example.Solver.Nonlinear.ResidualTol       = 1e-5
domain_example.Solver.Nonlinear.EtaChoice         = 'Walker1'
domain_example.Solver.Nonlinear.EtaValue          = 0.01
domain_example.Solver.Nonlinear.UseJacobian       = True
domain_example.Solver.Nonlinear.DerivativeEpsilon = 1e-12
domain_example.Solver.Nonlinear.StepTol           = 1e-30
domain_example.Solver.Nonlinear.Globalization     = 'LineSearch'
domain_example.Solver.Linear.KrylovDimension      = 100
domain_example.Solver.Linear.MaxRestarts          = 5
domain_example.Solver.Linear.Preconditioner       = 'PFMG'
domain_example.Solver.PrintSubsurf                = True
domain_example.Solver.Drop                        = 1E-20
domain_example.Solver.AbsTol                      = 1E-9

# Writing output options for ParFlow
write_pfb = True  #only PFB output for water balance example
#  PFB  no SILO
domain_example.Solver.PrintSubsurfData         = write_pfb
domain_example.Solver.PrintPressure            = write_pfb
domain_example.Solver.PrintSaturation          = write_pfb
domain_example.Solver.PrintCLM                 = write_pfb
domain_example.Solver.PrintMask                = write_pfb
domain_example.Solver.PrintSpecificStorage     = write_pfb
domain_example.Solver.PrintEvapTrans           = write_pfb
domain_example.Solver.PrintVelocities          = True 
domain_example.Solver.PrintChannelWidth        = True


#---------------------------------------------------
# LSM / CLM options
#---------------------------------------------------

# Writing output options for CLM
# no native CLM logs
domain_example.Solver.PrintLSMSink        = False
domain_example.Solver.CLM.CLMDumpInterval = 1
domain_example.Solver.CLM.CLMFileDir      = 'output/'
domain_example.Solver.CLM.BinaryOutDir    = False
domain_example.Solver.CLM.IstepStart      = 1
domain_example.Solver.WriteCLMBinary      = False
domain_example.Solver.CLM.WriteLogs       = False
domain_example.Solver.CLM.WriteLastRST    = True
domain_example.Solver.CLM.DailyRST        = False
domain_example.Solver.CLM.SingleFile      = True


#---------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------
domain_example.ICPressure.Type                 = 'HydroStaticPatch'
domain_example.ICPressure.GeomNames            = 'domain'
domain_example.Geom.domain.ICPressure.Value    = -10.00
domain_example.Geom.domain.ICPressure.RefGeom  = 'domain'
domain_example.Geom.domain.ICPressure.RefPatch = 'z_upper'

#-----------------------------------------------------------------------------
# Run ParFlow
#-----------------------------------------------------------------------------
base = os.path.join(os.getcwd(), "output")
mkdir(base)
print(f"base: {base}")
domain_example.run(working_directory=base)
